"""Tests for the Mac characterization renderer (pure-python; no binaries).

Covers the stats computation (fold-call counts, control-sanity, Spearman) against
a tiny hand-built table with known answers, plus a smoke test that the three PDFs
+ the stats TSV are written non-empty. No TMalign/blastp/ESMFold — the renderer
consumes only the merged summary tables.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, before pyplot is imported by the module

from SBM.characterize import summary  # noqa: E402
from SBM.utils import utils_characterize_plot as ucp  # noqa: E402

# Designs: one fold-A (ΔE<0, ΔTM>0), one fold-B (ΔE>0, ΔTM<0), one neither (ΔE≈0).
# ΔE and ΔTM are perfectly anti-ranked -> Spearman rho = -1.
_DESIGN = [
    {"sequence_id": "design_chain0000", "group": "design", "plddt_mean": "80",
     "tm_A": "0.60", "tm_B": "0.30", "delta_tm": "0.30", "fold_call": "A",
     "delta_E": "-5.0", "swissprot_pident": "40.0", "cmfam_pident": "41.0",
     "ppicfam_pident": "50.0"},
    {"sequence_id": "design_chain0001", "group": "design", "plddt_mean": "60",
     "tm_A": "0.30", "tm_B": "0.70", "delta_tm": "-0.40", "fold_call": "B",
     "delta_E": "6.0", "swissprot_pident": "35.0", "cmfam_pident": "38.0",
     "ppicfam_pident": "55.0"},
    {"sequence_id": "design_chain0002", "group": "design", "plddt_mean": "40",
     "tm_A": "0.20", "tm_B": "0.20", "delta_tm": "0.00", "fold_call": "neither",
     "delta_E": "1.0", "swissprot_pident": "", "cmfam_pident": "", "ppicfam_pident": ""},
]

# Controls: CM naturals resemble fold A (tm_A > tm_B), PPIC naturals fold B.
_NATURAL = [
    {"sequence_id": "cm0", "group": "CM-natural", "plddt_mean": "90",
     "tm_A": "0.85", "tm_B": "0.25", "delta_tm": "0.60", "fold_call": "A"},
    {"sequence_id": "cm1", "group": "CM-natural", "plddt_mean": "88",
     "tm_A": "0.82", "tm_B": "0.26", "delta_tm": "0.56", "fold_call": "A"},
    {"sequence_id": "pp0", "group": "PPIC-natural", "plddt_mean": "91",
     "tm_A": "0.26", "tm_B": "0.78", "delta_tm": "-0.52", "fold_call": "B"},
    {"sequence_id": "pp1", "group": "PPIC-natural", "plddt_mean": "92",
     "tm_A": "0.24", "tm_B": "0.75", "delta_tm": "-0.51", "fold_call": "B"},
]


def _write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union, stable order
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _stats_dict(design_rows, natural_rows) -> dict[tuple[str, str], str]:
    return {(g, m): v for g, m, v in ucp.compute_stats_rows(design_rows, natural_rows)}


def test_stats_counts_and_medians():
    stats = _stats_dict(_DESIGN, _NATURAL)
    assert stats[("design", "n")] == "3"
    assert stats[("design", "median_plddt")] == "60.00"
    assert stats[("design", "fold_call_A")] == "1"
    assert stats[("design", "fold_call_B")] == "1"
    assert stats[("design", "fold_call_neither")] == "1"
    assert stats[("CM-natural", "n")] == "2"
    assert stats[("PPIC-natural", "n")] == "2"


def test_control_sanity_passes_for_matched_naturals():
    stats = _stats_dict(_DESIGN, _NATURAL)
    assert stats[("CM-natural", "control_sanity_tmA_gt_tmB")] == "PASS"
    assert stats[("PPIC-natural", "control_sanity_tmB_gt_tmA")] == "PASS"


def test_control_sanity_fails_when_naturals_mismatched():
    # Swap the CM naturals to resemble fold B -> the check must FAIL (loud, not hidden).
    bad = [dict(r) for r in _NATURAL if r["group"] == "CM-natural"]
    for r in bad:
        r["tm_A"], r["tm_B"] = r["tm_B"], r["tm_A"]
    stats = _stats_dict(_DESIGN, bad)
    assert stats[("CM-natural", "control_sanity_tmA_gt_tmB")] == "FAIL"


def test_control_sanity_na_when_no_tm_data():
    # A present natural group with no finite TM -> "na", NOT "FAIL" (missing data must
    # not read as a control failure).
    no_tm = [{"sequence_id": "cm0", "group": "CM-natural", "plddt_mean": "90",
              "tm_A": "", "tm_B": "", "delta_tm": "", "fold_call": "na"}]
    stats = _stats_dict(_DESIGN, no_tm)
    assert stats[("CM-natural", "control_sanity_tmA_gt_tmB")] == "na"


def test_spearman_is_perfect_anticorrelation():
    stats = _stats_dict(_DESIGN, _NATURAL)
    # ΔE and ΔTM are perfectly anti-ranked across the 3 designs -> rho = -1.
    assert float(stats[("design", "spearman_deltaE_deltaTM_rho")]) == -1.0
    assert ("design", "spearman_deltaE_deltaTM_p") in stats


def test_spearman_absent_with_too_few_pairs():
    # < 3 finite pairs -> no Spearman row (guarded, not a crash).
    stats = _stats_dict(_DESIGN[:2], _NATURAL)
    assert ("design", "spearman_deltaE_deltaTM_rho") not in stats


def test_render_writes_all_outputs(tmp_path):
    figs = tmp_path / "figs"
    ucp.render_overview(_DESIGN, _NATURAL, figs / "characterization_overview.pdf")
    ucp.render_tm_scatter(_DESIGN, _NATURAL, figs / "tm_A_vs_B.pdf")
    ucp.render_fold_call_breakdown(_DESIGN, _NATURAL, figs / "fold_call_breakdown.pdf")
    stats = ucp.write_stats(_DESIGN, _NATURAL, tmp_path / "characterization_stats.tsv")
    for name in ("characterization_overview.pdf", "tm_A_vs_B.pdf", "fold_call_breakdown.pdf"):
        assert (figs / name).stat().st_size > 0
    assert Path(stats).stat().st_size > 0


def test_render_designs_only_no_naturals(tmp_path):
    # No natural_summary -> designs-only figures still render (no crash).
    figs = tmp_path / "figs"
    ucp.render_overview(_DESIGN, [], figs / "characterization_overview.pdf")
    ucp.render_tm_scatter(_DESIGN, [], figs / "tm_A_vs_B.pdf")
    assert (figs / "characterization_overview.pdf").stat().st_size > 0


def test_round_trips_through_read_tsv(tmp_path):
    # The renderer's real input path: summary.read_tsv of on-disk TSVs.
    _write_tsv(_DESIGN, tmp_path / "summary.tsv")
    _write_tsv(_NATURAL, tmp_path / "natural_summary.tsv")
    design_rows = summary.read_tsv(tmp_path / "summary.tsv")
    natural_rows = summary.read_tsv(tmp_path / "natural_summary.tsv")
    stats = _stats_dict(design_rows, natural_rows)
    assert stats[("design", "n")] == "3"
    assert stats[("CM-natural", "control_sanity_tmA_gt_tmB")] == "PASS"
