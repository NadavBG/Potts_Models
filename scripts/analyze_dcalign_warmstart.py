"""Analyse the DCAlign warm-start fixed-point probe (iter-003 Phase-B, §10.x).

Run AFTER the Midway warm-start driver writes ``<run-dir>/<model>/warmstart_out.tsv``
(one per model) and the cache is pulled back. Compares, per curated home pair, the
frame BP settled on *starting from native* against the native frame and against the
production (random-init) frame from the source run, classifies each
(:mod:`SBM.energy.dcalign_warmstart`), and writes the case-A/B verdict.

    python scripts/analyze_dcalign_warmstart.py \
        --run-dir combine/combine-CM-PPIC-dcalign-warmstart

Outputs (under ``--out-dir``, default ``<run-dir>/analysis``):
    warmstart_rows.tsv        one tidy row per home pair (3 energies + agreements + label)
    warmstart_analysis.json   per-role/kind tallies + verdict
    warmstart.pdf             the two-panel figure
    warmstart_manifest.json   provenance
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import fields
from pathlib import Path

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.dcalign_baseline import DEFAULT_EQUAL_TOL
from SBM.energy.dcalign_warmstart import (
    WarmstartRow,
    analyze_warmstart_record,
    summarize_warmstart,
)
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [f.name for f in fields(WarmstartRow)]


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_tsv(rows: list[WarmstartRow], path: Path) -> None:
    lines = ["\t".join(_TSV_COLUMNS)]
    lines += ["\t".join(_fmt(getattr(r, c)) for c in _TSV_COLUMNS) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote warm-start rows: %s (%d rows)", path, len(rows))


def build_rows(run_dir: Path, src_run_dir: Path, equal_tol: float) -> list[WarmstartRow]:
    """One :class:`WarmstartRow` per curated home pair (native frames from the source)."""
    roles = json.loads((run_dir / "roles.json").read_text(encoding="utf-8"))
    model_entries = json.loads((src_run_dir / "models.json").read_text(encoding="utf-8"))["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}

    records = datasets.read_query_fasta(
        src_run_dir / "query" / "query.fasta", src_run_dir / "query" / "groups.json")
    by_id = {r.id: r for r in records}

    # Only models that actually own curated home pairs need a warm-start cache.
    needed = {by_id[sid].origin_model for sid in roles
              if sid in by_id and by_id[sid].origin_model in models}
    warm_caches, rand_caches = {}, {}
    for name in sorted(needed):
        warm_tsv = run_dir / name / "warmstart_out.tsv"
        if not warm_tsv.is_file():
            raise FileNotFoundError(
                f"warm-start cache for {name!r} not found at {warm_tsv} "
                "(run the Midway warm-start driver first)")
        warm_caches[name] = read_alignment_cache(warm_tsv)
        rand_tsv = src_run_dir / "dcalign" / "cache" / name / "alignments.tsv"
        rand_caches[name] = read_alignment_cache(rand_tsv) if rand_tsv.is_file() else {}

    rows, missing = [], []
    for sid, role in roles.items():
        record = by_id.get(sid)
        if record is None or record.origin_model not in models:
            continue
        model = models[record.origin_model]
        warm = warm_caches[model.name].get(sid)
        if warm is None:
            missing.append(sid)
            continue
        rand = rand_caches.get(model.name, {}).get(sid)
        rows.append(analyze_warmstart_record(record, model, role, warm, rand, equal_tol=equal_tol))
    if missing:
        raise ValueError(
            f"{len(missing)} curated id(s) have no warm-start cache entry: "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}")
    log.info("built %d warm-start row(s); %d failed", len(rows), sum(1 for r in rows if not r.ok))
    return rows


def _analyze_one(run_dir: Path, src_run_dir: Path, equal_tol: float, init_kind: str,
                 out_dir: Path, started_at) -> dict:
    """Analyse one run dir: write rows TSV + analysis JSON + figure + manifest; return summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(run_dir, src_run_dir, equal_tol)
    if not rows:
        raise ValueError(f"no warm-start rows produced for {run_dir}")
    summary = summarize_warmstart(rows, equal_tol=equal_tol, init_kind=init_kind)
    rows_tsv = out_dir / "warmstart_rows.tsv"
    _write_tsv(rows, rows_tsv)
    (out_dir / "warmstart_analysis.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    from SBM.utils import utils_dcalign_warmstart_plot
    utils_dcalign_warmstart_plot.render_warmstart(rows_tsv, out_dir / "warmstart.pdf",
                                                  equal_tol=equal_tol)
    manifest = provenance.build_run_manifest(
        run_id="analyze_dcalign_warmstart", command_line=provenance.current_command_line(),
        inputs={"run_dir": run_dir, "src_run_dir": src_run_dir},
        options={"equal_tol": equal_tol, "init_kind": init_kind}, seed=None,
        started_at=started_at, finished_at=dt.datetime.now(dt.timezone.utc), output_path=rows_tsv,
        extra={"analysis": summary})
    provenance.save_run_manifest(manifest, out_dir / "warmstart_manifest.json")
    return summary


def _sweep_verdict(table: list[dict], best: dict, baseline: dict | None) -> str:
    base_txt = (f"baseline beta0=1.0 recovered {baseline['n_stayed_native']}/{baseline['n']}"
                if baseline else "no beta0=1.0 baseline in the sweep")
    drift = sum(t["n_control_drift"] for t in table)
    drift_txt = "" if drift == 0 else f" WARNING: {drift} control drift(s) across the sweep."
    if best["frac_stayed_native"] >= 0.5:
        return (f"ANNEAL-FROM-HOT WORKS: beta0={best['beta0']:g} recovered "
                f"{best['n_stayed_native']}/{best['n']} worse pairs ({base_txt}). Set beta0="
                f"{best['beta0']:g} as the production anneal schedule and run the full combine."
                + drift_txt)
    return (f"ANNEAL-FROM-HOT did not close it: best beta0={best['beta0']:g} recovered only "
            f"{best['n_stayed_native']}/{best['n']} ({base_txt}). Annealing is exhausted — ship the "
            f"combine as-is and document the residual as a method limit." + drift_txt)


def run_sweep(sweep_root: Path, src_run_dir: Path, equal_tol: float, init_kind: str,
              started_at) -> dict:
    """Analyse every beta0 run dir in a sweep, tabulate recovery-vs-beta0, pick the best."""
    meta = json.loads((sweep_root / "sweep_meta.json").read_text(encoding="utf-8"))
    table = []
    for bstr, rd in sorted(meta["run_dirs"].items(), key=lambda kv: float(kv[0])):
        rd = Path(rd)
        summ = _analyze_one(rd, src_run_dir, equal_tol, init_kind, rd / "analysis", started_at)
        rec = summ["recover"]["overall"]
        table.append({
            "beta0": float(bstr), "n": rec["n_ok"], "n_stayed_native": rec["n_stayed_native"],
            "frac_stayed_native": rec["frac_stayed_native"],
            "median_delta_e_warm": rec["median_delta_e_warm"],
            "n_control_drift": summ["control"]["n_control_drift"],
        })
    best = max(table, key=lambda d: (d["frac_stayed_native"],
               -(d["median_delta_e_warm"] if d["median_delta_e_warm"] is not None else 1e18)))
    baseline = next((r for r in table if abs(r["beta0"] - 1.0) < 1e-9), None)
    verdict = _sweep_verdict(table, best, baseline)
    summary = {"kind": "annealsweep", "equal_tol": equal_tol, "init_kind": init_kind,
               "by_beta0": table, "best_beta0": best["beta0"], "verdict": verdict}
    (sweep_root / "annealsweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                         encoding="utf-8")
    from SBM.utils import utils_dcalign_warmstart_plot
    utils_dcalign_warmstart_plot.render_annealsweep(table, sweep_root / "annealsweep.pdf",
                                                    equal_tol=equal_tol)
    manifest = provenance.build_run_manifest(
        run_id="analyze_dcalign_annealsweep", command_line=provenance.current_command_line(),
        inputs={"sweep_root": sweep_root, "src_run_dir": src_run_dir},
        options={"equal_tol": equal_tol, "init_kind": init_kind}, seed=None,
        started_at=started_at, finished_at=dt.datetime.now(dt.timezone.utc),
        output_path=sweep_root / "annealsweep_summary.json", extra={"by_beta0": table})
    provenance.save_run_manifest(manifest, sweep_root / "annealsweep_manifest.json")
    print(verdict)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None, help="single warm-start/map/anneal run dir")
    p.add_argument("--sweep-root", type=Path, default=None,
                   help="anneal sweep root (beta<v>/ subdirs + sweep_meta.json) — recovery vs beta0")
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"),
                   help="source of native frames + models + random-init cache (default iter-002)")
    p.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/analysis")
    p.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL)
    p.add_argument("--init-kind", default="native", choices=("native", "map", "random"),
                   help="how BP was initialised (only sets the verdict wording); match the build")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    if (args.run_dir is None) == (args.sweep_root is None):
        p.error("pass exactly one of --run-dir or --sweep-root")

    if args.sweep_root is not None:
        run_sweep(args.sweep_root, args.src_run_dir, args.equal_tol, args.init_kind, started_at)
        return 0

    out_dir = args.out_dir or args.run_dir / "analysis"
    summary = _analyze_one(args.run_dir, args.src_run_dir, args.equal_tol, args.init_kind,
                           out_dir, started_at)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
