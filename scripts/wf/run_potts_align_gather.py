#!/usr/bin/env python
"""potts_align align step — gather shards into one cache per model (iter-003).

Standalone CLI (imports only ``SBM.*`` + stdlib; invoked by path from
``pipeline/external/sbatch_potts_align_gather.sh``). Merges
``<run_root>/potts_align/cache/shards/shard_*.tsv`` into one
``<model>/alignments.tsv`` per model — one row per query id, scored (real
energy) or a skip row (``engine`` ∈ {``skip_NgtL``, ``skip_subsample``,
``missing``}, nan energy, empty frame) — so the ``score`` cache-reader always
finds a row (a genuine gap is a loud error, not a silent skip). Writes a
per-model ``meta.json`` and a top-level ``gather_status.json`` carrying the
validation gates: the in-frame recompute canary, the home-term ΔE gate, and the
random-control-vs-naturals energy separation.

By default it errors if any in-scope pair is missing (an incomplete run —
re-submit the unfinished shards, then gather again). ``--allow-missing`` writes
those as ``missing`` skip rows (WARN) and a PARTIAL cache instead.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
from SBM import combine_config as cc
from SBM.energy import datasets
from SBM.energy.encoding import seq_to_ints
from SBM.energy.model import load_model
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align import SASchedule
from SBM.utils.potts_align_cache import (
    PottsAlignCacheResult,
    read_shard_cache,
    write_alignment_cache,
)

log = logging.getLogger(__name__)

_CANARY_TOL = 1e-6   # in-frame recompute vs cached energy
_DELTA_E_FLAG = 1.0  # home ΔE above native flagged for the PT tail (docs §6.8)


def _skip_row(p: dict, engine: str) -> PottsAlignCacheResult:
    return PottsAlignCacheResult(
        query_id=p["query_id"], model=p["model"], n_residues=p["n_residues"],
        gaps=p["gaps"], energy=float("nan"), engine=engine, is_global_exact=False,
        frame="", seed=p["seed"],
    )


def _gather_model(run_root: Path, model_entry: dict, manifest: dict, records: dict,
                  merged: dict, allow_missing: bool) -> tuple[dict, list[dict]]:
    """Write one model's alignments.tsv + meta.json; return (status, canary rows)."""
    name = model_entry["name"]
    model = load_model(model_entry["model_path"], name=name)
    pairs = [p for p in manifest["pairs"] if p["model"] == name]

    rows: list[PottsAlignCacheResult] = []
    missing: list[str] = []
    checks: list[dict] = []  # per-scored-row canary / ΔE diagnostics
    for p in sorted(pairs, key=lambda x: x["query_id"]):
        status = p["status"]
        if status in ("home", "cross"):
            res = merged.get((p["query_id"], name))
            if res is None:
                missing.append(p["query_id"])
                rows.append(_skip_row(p, "missing"))
                continue
            rows.append(res)
            # In-frame recompute canary + (for home) the ΔE gate.
            frame = seq_to_ints(res.frame)
            e_recompute = potts_energy(frame, model)
            entry = {"query_id": p["query_id"], "status": status, "engine": res.engine,
                     "is_global_exact": res.is_global_exact,
                     "abs_diff": abs(e_recompute - res.energy)}
            if status == "home":
                e_native = potts_energy(records[p["query_id"]].ints, model)
                entry["delta_e"] = res.energy - e_native
            checks.append(entry)
        else:  # skip_NgtL / skip_subsample
            rows.append(_skip_row(p, status))

    out_tsv = run_root / "potts_align" / "cache" / name / "alignments.tsv"
    write_alignment_cache(out_tsv, rows)

    n_requested = sum(1 for p in pairs if p["status"] in ("home", "cross"))
    meta = {
        "model_name": name, "L": model.L, "q": model.q, "model_sha256": model.sha256,
        "schedule": "g-adaptive PTSchedule.for_gap_count(g); enumerate when C(L,N) <= enum_max_frames",
        "enum_max_frames": SASchedule().enum_max_frames,
        "master_seed": manifest["master_seed"],
        "n_requested": n_requested, "n_scored": n_requested - len(missing),
        "n_skip_NgtL": sum(1 for p in pairs if p["status"] == "skip_NgtL"),
        "n_skip_subsample": sum(1 for p in pairs if p["status"] == "skip_subsample"),
        "n_missing": len(missing), "missing_ids": missing[:200],
        "partial": bool(missing),
        "cross_subsample": manifest["cross_subsample"],
        "git_commit": provenance.git_commit(), "numpy_version": np.__version__,
    }
    (out_tsv.parent / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("gathered model %r: %d rows (%d scored, %d missing) -> %s",
             name, len(rows), meta["n_scored"], len(missing), out_tsv)
    return {"model": name, "meta": {k: meta[k] for k in
            ("n_requested", "n_scored", "n_skip_NgtL", "n_skip_subsample", "n_missing", "partial")},
            "missing_ids": missing[:50]}, checks


def _gate_summary(name: str, checks: list[dict], random_sep: dict) -> dict:
    """The validation gates for one model, for gather_status.json."""
    diffs = np.array([c["abs_diff"] for c in checks], dtype=float) if checks else np.array([])
    canary = {"max_abs_diff": float(diffs.max()) if diffs.size else None,
              "median_abs_diff": float(np.median(diffs)) if diffs.size else None,
              "n": int(diffs.size)}
    canary_ok = bool(diffs.size == 0 or diffs.max() <= _CANARY_TOL)

    homes = [c for c in checks if c.get("status") == "home"]
    exact_viol = [c["query_id"] for c in homes if c["is_global_exact"] and c["delta_e"] > _CANARY_TOL]
    pt_flagged = [(c["query_id"], round(c["delta_e"], 4)) for c in homes
                  if not c["is_global_exact"] and c["delta_e"] > _DELTA_E_FLAG]
    dE = {"n_home": len(homes),
          "max_delta_e": max((c["delta_e"] for c in homes), default=None),
          "n_enumerated_violations": len(exact_viol), "enumerated_violations": exact_viol[:50],
          "n_pt_flagged": len(pt_flagged), "pt_flagged": pt_flagged[:50]}
    return {"in_frame_canary": canary, "canary_ok": canary_ok, "delta_e_gate": dE,
            "random_control": random_sep}


def _random_separation(manifest, records, merged) -> dict:
    """Median energy of random controls vs naturals, per model (figure sanity)."""
    rc_group = (manifest.get("random_control") or {}).get("group")
    out: dict = {}
    for name in manifest["models"]:
        e_rand, e_nat = [], []
        for p in manifest["pairs"]:
            if p["model"] != name or p["status"] not in ("home", "cross"):
                continue
            res = merged.get((p["query_id"], name))
            if res is None or not res.ok:
                continue
            grp = records[p["query_id"]].group
            (e_rand if grp == rc_group else e_nat).append(res.energy)
        out[name] = {
            "median_random": float(np.median(e_rand)) if e_rand else None,
            "median_natural": float(np.median(e_nat)) if e_nat else None,
            "n_random": len(e_rand), "n_natural": len(e_nat),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gather potts_align shards into one cache per model.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="write missing in-scope pairs as 'missing' skip rows (WARN) + a "
                             "PARTIAL cache, instead of erroring on an incomplete align run.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_root = Path(args.run_root)
    manifest = json.loads((_manifest := run_root / "potts_align" / "shards_manifest.json").read_text(encoding="utf-8"))
    cc.load_config(run_root / "config_snapshot.yaml")  # validate the snapshot is present/parseable
    records = {r.id: r for r in sorted(
        datasets.read_query_fasta(run_root / "query" / "query.fasta", run_root / "query" / "groups.json"),
        key=lambda r: r.id)}
    entries = json.loads((run_root / "models.json").read_text(encoding="utf-8"))["models"]

    # Merge shards; a (query_id, model) may appear in only one shard.
    merged: dict = {}
    shard_files = sorted((run_root / "potts_align" / "cache" / "shards").glob("shard_*.tsv"))
    for f in shard_files:
        for key, res in read_shard_cache(f).items():
            if key in merged:
                raise ValueError(f"pair {key!r} appears in more than one shard ({f}); shards must partition pairs")
            merged[key] = res

    # Coverage over in-scope pairs.
    in_scope = [(p["query_id"], p["model"]) for p in manifest["pairs"]
                if p["status"] in ("home", "cross")]
    missing = [k for k in in_scope if k not in merged]
    if missing and not args.allow_missing:
        raise RuntimeError(
            f"{len(missing)} in-scope pair(s) not produced by any shard (incomplete align run): "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}. Re-submit the unfinished shard "
            f"tasks then gather again, or pass --allow-missing to write a partial cache."
        )
    if missing:
        log.warning("PARTIAL cache: %d of %d in-scope pairs missing (--allow-missing)",
                    len(missing), len(in_scope))

    statuses, all_checks = [], {}
    for entry in entries:
        status, checks = _gather_model(run_root, entry, manifest, records, merged, args.allow_missing)
        statuses.append(status)
        all_checks[entry["name"]] = checks

    rand_sep = _random_separation(manifest, records, merged)
    gates = {name: _gate_summary(name, all_checks[name], rand_sep[name])
             for name in manifest["models"]}
    # Hard gate: the in-frame canary must hold for every model (loud failure).
    bad = [name for name, g in gates.items() if not g["canary_ok"]]
    status = {
        "models": statuses,
        "n_pairs_in_scope": len(in_scope), "n_scored": len(in_scope) - len(missing),
        "partial": bool(missing),
        "gates": gates,
        "git_commit": provenance.git_commit(), "numpy_version": np.__version__,
    }
    (run_root / "potts_align" / "gather_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8")
    log.info("gather complete -> %s", run_root / "potts_align" / "gather_status.json")
    if bad:
        raise RuntimeError(
            f"in-frame recompute CANARY failed for model(s) {bad} (max |Δ| > {_CANARY_TOL}); "
            f"the cluster energies disagree with a Mac in-frame recompute — gauge/handoff bug. "
            f"See potts_align/gather_status.json."
        )
    for name, g in gates.items():
        dE = g["delta_e_gate"]
        if dE["n_enumerated_violations"]:
            raise RuntimeError(
                f"model {name}: {dE['n_enumerated_violations']} enumerated home pair(s) have "
                f"ΔE > {_CANARY_TOL} above native — an enumerated frame CANNOT beat-or-tie fail "
                f"(bug). Examples: {dE['enumerated_violations'][:5]}"
            )
        if dE["n_pt_flagged"]:
            log.warning("model %s: %d PT home pair(s) sit ΔE > %g above native (hardest g>=13 tail, "
                        "expected; see gather_status.json): %s",
                        name, dE["n_pt_flagged"], _DELTA_E_FLAG, dE["pt_flagged"][:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
