"""Merge fold + structure-compare + BLAST + design energies into the summary.

Produces the tidy per-sequence ``summary.tsv`` (designs) and
``natural_summary.tsv`` (controls) plus a human-readable ``report.md``.
Pure stdlib except an optional scipy import (guarded) for the
energy-vs-structure rank correlation.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

#: Structural assignment thresholds (TM-score normalized by the reference).
TM_FOLD_THRESHOLD = 0.5
TM_MARGIN = 0.05
#: pLDDT confidence bands (0–100).
PLDDT_HIGH = 70.0
PLDDT_MEDIUM = 50.0


# ── Classification ──────────────────────────────────────────────────────────


def fold_call(
    tm_a: float, tm_b: float, *, thresh: float = TM_FOLD_THRESHOLD, margin: float = TM_MARGIN
) -> str:
    """Assign a design to fold A, B, both (ambiguous), or neither, from TM.

    Uses the reference-normalized TM-scores. TM>=0.5 is the conventional
    "same fold" line (Xu & Zhang 2010). "ambiguous" = both above threshold
    and within ``margin`` of each other.
    """
    if math.isnan(tm_a) or math.isnan(tm_b):
        return "na"
    a, b = tm_a >= thresh, tm_b >= thresh
    if not a and not b:
        return "neither"
    if a and b:
        return "ambiguous" if abs(tm_a - tm_b) < margin else ("A" if tm_a > tm_b else "B")
    return "A" if a else "B"


def plddt_class(plddt: float) -> str:
    """Confidence band for a mean pLDDT: high / medium / low / na."""
    if math.isnan(plddt):
        return "na"
    if plddt >= PLDDT_HIGH:
        return "high"
    if plddt >= PLDDT_MEDIUM:
        return "medium"
    return "low"


# ── Loaders ─────────────────────────────────────────────────────────────────


def read_tsv(path: Path | str) -> list[dict[str, str]]:
    """Read a tab-separated file with a header row into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_design_energies(designed_tsv: Path | str) -> dict[str, dict[str, float | str]]:
    """Map ``design_chain{NNNN}`` -> polished energies + start type.

    Keys match the design FASTA record ids (``design_chain0000`` ...), built
    from the ``chain`` column of ``design/designed.tsv``.
    """
    out: dict[str, dict[str, float | str]] = {}
    for row in read_tsv(designed_tsv):
        chain = int(row["chain"])
        out[f"design_chain{chain:04d}"] = {
            "E_A": float(row["E_A_polish"]),
            "E_B": float(row["E_B_polish"]),
            "E_tot": float(row["E_tot_polish"]),
            "start_type": row.get("start_type", ""),
        }
    return out


# ── Row assembly ────────────────────────────────────────────────────────────

DESIGN_COLUMNS = [
    "sequence_id", "group", "length", "plddt_mean", "plddt_class", "ptm",
    "tm_A", "rmsd_A", "tm_B", "rmsd_B", "delta_tm", "fold_call",
    "E_A", "E_B", "E_tot", "delta_E", "start_type",
    "swissprot_top_hit", "swissprot_pident", "swissprot_evalue", "swissprot_annotation",
    "cmfam_top_hit", "cmfam_pident", "ppicfam_top_hit", "ppicfam_pident",
]

NATURAL_COLUMNS = [
    "sequence_id", "group", "length", "plddt_mean", "plddt_class", "ptm",
    "tm_A", "rmsd_A", "tm_B", "rmsd_B", "delta_tm", "fold_call",
]


def _f(x: str | float | None) -> float:
    """Parse to float, empty/None -> NaN."""
    if x is None or x == "":
        return float("nan")
    return float(x)


def _fmt(x: float, nd: int = 4) -> str:
    return "" if isinstance(x, float) and math.isnan(x) else f"{x:.{nd}f}"


def build_summary_rows(
    fold_rows: list[dict[str, str]],
    compare_by_id: dict[str, dict[str, str]],
    *,
    energies: dict[str, dict[str, float | str]] | None = None,
    swissprot: dict[str, object] | None = None,
    cmfam: dict[str, object] | None = None,
    ppicfam: dict[str, object] | None = None,
    annotations: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Join fold + compare (+ optional energies/BLAST) into summary rows.

    ``compare_by_id`` maps sequence id -> the row of ``structure_compare.tsv``.
    The BLAST args (``swissprot``/``cmfam``/``ppicfam``) map sequence id ->
    a ``BlastHit`` (or None); passing them selects the design layout.
    """
    is_design = energies is not None
    rows: list[dict[str, str]] = []
    for fr in fold_rows:
        sid = fr["id"]
        cmp_row = compare_by_id.get(sid, {})
        tm_a = _f(cmp_row.get("tm_ref_A"))
        tm_b = _f(cmp_row.get("tm_ref_B"))
        plddt = _f(fr.get("plddt_mean"))
        delta_tm = tm_a - tm_b
        row: dict[str, str] = {
            "sequence_id": sid,
            "group": fr.get("group", ""),
            "length": fr.get("length", ""),
            "plddt_mean": _fmt(plddt, 2),
            "plddt_class": plddt_class(plddt),
            "ptm": _fmt(_f(fr.get("ptm")), 4),
            "tm_A": _fmt(tm_a),
            "rmsd_A": _fmt(_f(cmp_row.get("rmsd_A")), 3),
            "tm_B": _fmt(tm_b),
            "rmsd_B": _fmt(_f(cmp_row.get("rmsd_B")), 3),
            "delta_tm": _fmt(delta_tm),
            "fold_call": fold_call(tm_a, tm_b),
        }
        if is_design:
            en = energies.get(sid, {})
            e_a, e_b = _f(en.get("E_A")), _f(en.get("E_B"))
            row.update({
                "E_A": _fmt(e_a, 4),
                "E_B": _fmt(e_b, 4),
                "E_tot": _fmt(_f(en.get("E_tot")), 4),
                "delta_E": _fmt(e_a - e_b, 4),
                "start_type": str(en.get("start_type", "")),
            })
            sp = (swissprot or {}).get(sid)
            cm = (cmfam or {}).get(sid)
            pp = (ppicfam or {}).get(sid)
            row.update({
                "swissprot_top_hit": sp.sseqid if sp else "",
                "swissprot_pident": _fmt(sp.pident, 1) if sp else "",
                "swissprot_evalue": f"{sp.evalue:.1e}" if sp else "",
                "swissprot_annotation": (annotations or {}).get(sp.sseqid, "") if sp else "",
                "cmfam_top_hit": cm.sseqid if cm else "",
                "cmfam_pident": _fmt(cm.pident, 1) if cm else "",
                "ppicfam_top_hit": pp.sseqid if pp else "",
                "ppicfam_pident": _fmt(pp.pident, 1) if pp else "",
            })
        rows.append(row)
    return rows


def write_tsv(rows: list[dict[str, str]], columns: list[str], path: Path | str) -> Path:
    """Write rows as a TSV with the given column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


# ── Stats + report ──────────────────────────────────────────────────────────


def _finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def _median(values: list[float]) -> float:
    v = sorted(_finite(values))
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _spearman(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Spearman rho + p-value over pairwise-finite (x, y); None if scipy absent
    or < 3 pairs."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return None
    try:
        from scipy import stats  # noqa: PLC0415 - optional
    except ImportError:
        return None
    xs, ys = zip(*pairs)
    res = stats.spearmanr(xs, ys)
    return float(res.statistic), float(res.pvalue)


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get(key, "")] = out.get(r.get(key, ""), 0) + 1
    return out


def write_report(
    design_rows: list[dict[str, str]],
    natural_rows: list[dict[str, str]],
    path: Path | str,
    *,
    meta: dict[str, str] | None = None,
) -> Path:
    """Write the human-readable ``report.md`` summarizing designs vs controls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Design characterization: fold + BLAST", ""]
    if meta:
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Reference caveat.
    lines += [
        "> Fold A = chorismate mutase (1ECM chain A); fold B = PPIC / parvulin",
        "> (1JNT chain A). TM-scores are normalized by the **reference** length,",
        "> so TM_A and TM_B are directly comparable. Note: 1ECM chain A is one arm",
        "> of a domain-swapped dimer, so an isolated-monomer TM_A is a lower bound.",
        "",
    ]

    def _block(title: str, rows: list[dict[str, str]]) -> None:
        lines.append(f"## {title} ({len(rows)})")
        lines.append("")
        if not rows:
            lines.append("_none_")
            lines.append("")
            return
        plddt = [_f(r["plddt_mean"]) for r in rows]
        tm_a = [_f(r["tm_A"]) for r in rows]
        tm_b = [_f(r["tm_B"]) for r in rows]
        lines.append(f"- median pLDDT: {_median(plddt):.1f}  "
                     f"(pLDDT class: {_count_by(rows, 'plddt_class')})")
        lines.append(f"- median TM_A: {_median(tm_a):.3f}   median TM_B: {_median(tm_b):.3f}")
        lines.append(f"- fold_call breakdown: {_count_by(rows, 'fold_call')}")
        lines.append("")

    _block("Designs", design_rows)

    # Natural controls, split by group (the correctness check).
    cm_nat = [r for r in natural_rows if "CM" in r.get("group", "")]
    ppic_nat = [r for r in natural_rows if "PPIC" in r.get("group", "")]
    _block("Natural controls — CM", cm_nat)
    _block("Natural controls — PPIC", ppic_nat)

    # Correctness assertions (naturals should match their own fold).
    lines.append("## Control sanity (naturals must match their own reference)")
    lines.append("")
    if cm_nat:
        ok = _median([_f(r["tm_A"]) for r in cm_nat]) > _median([_f(r["tm_B"]) for r in cm_nat])
        lines.append(f"- CM naturals: median TM_A > median TM_B  -> {'PASS' if ok else 'FAIL'}")
    if ppic_nat:
        ok = _median([_f(r["tm_B"]) for r in ppic_nat]) > _median([_f(r["tm_A"]) for r in ppic_nat])
        lines.append(f"- PPIC naturals: median TM_B > median TM_A -> {'PASS' if ok else 'FAIL'}")
    lines.append("")

    # Energy vs structure (the scientific payoff).
    if design_rows and "delta_E" in design_rows[0]:
        d_e = [_f(r["delta_E"]) for r in design_rows]  # E_A - E_B
        d_tm = [_f(r["delta_tm"]) for r in design_rows]  # TM_A - TM_B
        lines.append("## Energy vs. structure (designs)")
        lines.append("")
        lines.append("Does an energetically-more-CM design (lower E_A-E_B) fold more")
        lines.append("CM-like (higher TM_A-TM_B)? Expect a **negative** rank correlation.")
        lines.append("")
        sp = _spearman(d_e, d_tm)
        if sp is not None:
            lines.append(f"- Spearman(delta_E, delta_TM) = {sp[0]:+.3f}  (p = {sp[1]:.2g})")
        else:
            lines.append("- Spearman unavailable (scipy missing or < 3 finite pairs)")
        lines.append("")

    # Top BLAST readout for designs.
    if design_rows and "swissprot_top_hit" in design_rows[0]:
        lines.append("## Designs — top BLAST hits (per database, kept separate)")
        lines.append("")
        for label, hit_col, id_col, pid_col, ann_col in [
            ("SwissProt", "swissprot_top_hit", "sequence_id", "swissprot_pident", "swissprot_annotation"),
            ("CM family", "cmfam_top_hit", "sequence_id", "cmfam_pident", None),
            ("PPIC family", "ppicfam_top_hit", "sequence_id", "ppicfam_pident", None),
        ]:
            n_hit = sum(1 for r in design_rows if r.get(hit_col))
            pids = _finite([_f(r.get(pid_col, "")) for r in design_rows if r.get(hit_col)])
            med = f"{_median(pids):.1f}%" if pids else "n/a"
            lines.append(f"- **{label}**: {n_hit}/{len(design_rows)} designs with a hit; "
                         f"median best %id = {med}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
