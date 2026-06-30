"""Analyze the iter-003 §10.20 Potts-init warm-start batch (after sync pull).

Reads each run dir's per-model ``warmstart_out.tsv`` and reports, vs the native
frame:

* ``sa-beta*`` (M1): did BP, warm-started at the couplings-aware Potts-align frame,
  reach native (ΔE_warm ≤ tol)? Per role/kind, via
  :func:`SBM.energy.dcalign_warmstart.summarize_warmstart`.
* ``perturb-k*`` (M3): recovery-vs-k — the fraction of worse pairs BP returns to
  native when started k columns off native. The k at which recovery collapses is
  the basin radius.
* ``diag-*`` (M4): cross-check DCAlign's 0-sweep ``compute_en`` at the init frame
  against numpy ``potts_energy`` (a gauge canary; NOT a Λ free energy).

Run on the Mac after ``scripts/sync_models.sh pull``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from SBM.energy import datasets
from SBM.energy.dcalign_warmstart import analyze_warmstart_record, summarize_warmstart
from SBM.energy.encoding import GAP, seq_to_ints
from SBM.energy.model import load_model
from SBM.energy.potts import potts_energy
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)
EQUAL_TOL = 1.0


def _load_common(src: Path):
    entries = json.loads((src / "models.json").read_text())["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in entries}
    records = datasets.read_query_fasta(src / "query" / "query.fasta", src / "query" / "groups.json")
    rand = {m["name"]: read_alignment_cache(
        src / "dcalign" / "cache" / m["name"] / "alignments.tsv") for m in entries}
    return models, {r.id: r for r in records}, rand


def _dir_caches(run_dir: Path) -> dict[str, dict]:
    """Per-model warmstart caches present in a run dir (model -> {id: DCAlignResult})."""
    names = [m["name"] for m in json.loads((run_dir / "models.json").read_text())["models"]]
    out = {}
    for name in names:
        tsv = run_dir / name / "warmstart_out.tsv"
        if tsv.is_file():
            out[name] = read_alignment_cache(tsv)
        else:
            log.warning("%s: missing %s (run not pulled yet?)", run_dir.name, tsv)
    return out


def _warmstart_rows(run_dir, models, by_id, rand, roles, tol):
    rows = []
    for name, cache in _dir_caches(run_dir).items():
        for sid, res in cache.items():
            if sid not in by_id:
                continue
            rows.append(analyze_warmstart_record(
                by_id[sid], models[name], roles.get(sid, "recover"),
                res, rand[name].get(sid), equal_tol=tol))
    return rows


def analyze_sa(run_dir, models, by_id, rand, roles, tol):
    rows = _warmstart_rows(run_dir, models, by_id, rand, roles, tol)
    summary = summarize_warmstart(rows, equal_tol=tol, init_kind="potts-align")
    rec = summary["recover"]["overall"]
    print(f"[{run_dir.name}] recover reached native {rec['n_stayed_native']}/{rec['n_ok']} "
          f"(median ΔE_warm {rec['median_delta_e_warm']}); "
          f"controls drift {summary['control']['n_control_drift']}/{summary['control']['n']}")
    return {"kind": "sastart", "summary": summary,
            "rows": [r.as_dict() for r in rows]}


def analyze_perturb(run_dirs, models, by_id, rand, roles, tol):
    """Recovery-vs-k over the perturb-k* dirs (the basin-width curve)."""
    curve = []
    detail = {}
    for k, run_dir in sorted(run_dirs.items()):
        rows = [r for r in _warmstart_rows(run_dir, models, by_id, rand, roles, tol)
                if r.role == "recover" and r.ok]
        n = len(rows)
        rec = sum(1 for r in rows if r.delta_e_warm <= tol)
        curve.append({"k": k, "n": n, "n_returned_native": rec,
                      "frac_returned": (rec / n) if n else None,
                      "median_delta_e_warm": float(np.median([r.delta_e_warm for r in rows]))
                      if rows else None})
        detail[k] = [r.as_dict() for r in rows]
        print(f"[perturb-k{k}] BP returned to native {rec}/{n} "
              f"(median ΔE_warm {curve[-1]['median_delta_e_warm']})")
    return {"kind": "basin_width", "recovery_vs_k": curve, "detail": detail}


def analyze_diag(run_dir, models, by_id, tol):
    """Cross-check 0-sweep compute_en against numpy potts_energy of the init frame."""
    rows, worst = [], 0.0
    for name, cache in _dir_caches(run_dir).items():
        for sid, res in cache.items():
            if sid not in by_id or not res.aligned_frame:
                continue
            frame = seq_to_ints(res.aligned_frame)
            e_numpy = potts_energy(frame, models[name])
            diff = abs(res.dcalign_energy - e_numpy)
            worst = max(worst, diff)
            rows.append({"sequence_id": sid, "model": name, "compute_en": res.dcalign_energy,
                         "potts_energy": e_numpy, "abs_diff": diff})
    ok = worst <= 5e-7
    print(f"[{run_dir.name}] compute_en vs potts_energy: max |Δ| = {worst:.2e} "
          f"({'PASS' if ok else 'FAIL'} at 5e-7)")
    return {"kind": "diag", "max_abs_diff": worst, "canary_passed": ok, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-root", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-pottsinit"))
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"))
    p.add_argument("--roles", type=Path, default=None)
    p.add_argument("--equal-tol", type=float, default=EQUAL_TOL)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models, by_id, rand = _load_common(args.src_run_dir)
    roles_path = args.roles or (args.batch_root / "sa-beta1" / "roles.json")
    roles = json.loads(roles_path.read_text()) if roles_path.is_file() else {}

    results = {}
    perturb_dirs = {}
    for run_dir in sorted(args.batch_root.glob("*/")):
        name = run_dir.name
        if not (run_dir / "models.json").is_file():
            continue
        if name.startswith("sa-"):
            results[name] = analyze_sa(run_dir, models, by_id, rand, roles, args.equal_tol)
        elif name.startswith("perturb-k"):
            perturb_dirs[int(name.split("perturb-k")[1])] = run_dir
        elif name.startswith("diag-"):
            results[name] = analyze_diag(run_dir, models, by_id, args.equal_tol)
    if perturb_dirs:
        results["basin_width"] = analyze_perturb(perturb_dirs, models, by_id, rand, roles,
                                                 args.equal_tol)

    out = args.batch_root / "pottsinit_batch_summary.json"
    out.write_text(json.dumps(results, indent=2, default=lambda o: None) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
