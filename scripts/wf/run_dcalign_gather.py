#!/usr/bin/env python
"""DCAlign align step — gather shards into one cache per model (spec §10.9).

Standalone CLI (imports only ``SBM.*`` + stdlib; invoked by path from
``pipeline/external/sbatch_dcalign_gather.sh``). For each model it merges
``<run_root>/dcalign/cache/<model>/shards/shard_*.tsv`` into a single
``alignments.tsv`` (one row per query id, sorted), writes a provenance
``meta.json`` (cache layout spec §8), and a top-level ``gather_status.json``.

By default it **errors** if any requested id is missing from the merged shards
(an incomplete align run — re-submit the unfinished shards before gathering).
Pass ``--allow-missing`` to instead drop the absent ids with a WARN and write a
PARTIAL cache (the missing ids are recorded in ``meta.json`` / ``gather_status``
and the cache self-declares ``partial: true`` so the score step skips them).
Sequences DCAlign failed on are kept as empty-frame rows (recorded, counted in
``gather_status.json``) — they become a loud error only at score time if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from SBM import combine_config as cc
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import (
    dcalign_context,
    read_alignment_cache,
    write_alignment_cache,
)

log = logging.getLogger(__name__)


def _dcalign_provenance(cfg) -> dict:
    """Best-effort DCAlign clone commit + Julia version (None if env not set)."""
    sc = cfg.scoring
    try:
        ctx = dcalign_context(
            dcalign_path=sc.dcalign_path, julia=sc.julia,
            maxiter=sc.maxiter, seed=sc.dcalign_seed, pcount=sc.pcount,
        )
        return {"dcalign_commit": ctx.dcalign_git_commit, "julia_version": ctx.julia_version}
    except (RuntimeError, FileNotFoundError, PermissionError) as exc:
        log.warning("could not resolve DCAlign/julia for provenance: %s", exc)
        return {"dcalign_commit": None, "julia_version": None}


def _gather_model(run_root: Path, model_index: int, manifest: dict, cfg, prov: dict,
                  allow_missing: bool = False) -> dict:
    model_entry = json.loads((run_root / "models.json").read_text(encoding="utf-8"))["models"][model_index]
    model_name = model_entry["name"]
    cache_dir = run_root / "dcalign" / "cache" / model_name
    shards_dir = cache_dir / "shards"

    merged: dict = {}
    shard_files = sorted(shards_dir.glob("shard_*.tsv"))
    for f in shard_files:
        for sid, res in read_alignment_cache(f).items():
            if sid in merged:
                raise ValueError(f"id {sid!r} appears in more than one shard ({f}); shards must partition ids")
            merged[sid] = res

    requested = [sid for shard in manifest["shards"] for sid in shard]
    missing = [sid for sid in requested if sid not in merged]
    if missing and not allow_missing:
        raise RuntimeError(
            f"model {model_name!r}: {len(missing)} requested id(s) not produced by any shard "
            f"(incomplete align run): {missing[:5]}{' …' if len(missing) > 5 else ''}. "
            f"Re-submit the unfinished shard tasks, then gather again, or pass --allow-missing "
            f"to write a partial cache."
        )
    if missing:
        log.warning(
            "model %r: %d of %d requested id(s) missing from shards; writing a PARTIAL cache "
            "(--allow-missing): %s%s",
            model_name, len(missing), len(requested),
            missing[:5], " …" if len(missing) > 5 else "",
        )

    present = [sid for sid in sorted(requested) if sid in merged]
    ordered = [merged[sid] for sid in present]
    failed = [r.seq_id for r in ordered if not r.ok]
    out_tsv = cache_dir / "alignments.tsv"
    write_alignment_cache(out_tsv, ordered)

    model = load_model(model_entry["model_path"], name=model_name)
    meta = {
        "model_name": model_name,
        "L": model.L,
        "q": model.q,
        "model_sha256": model.sha256,
        "maxiter": cfg.scoring.maxiter,
        "seed": cfg.scoring.dcalign_seed,
        "pcount": cfg.scoring.pcount,
        "lambda_spec": cfg.scoring.lambda_spec,
        "partial": bool(missing),
        "n_requested": len(requested),
        "n_missing": len(missing),
        "missing_ids": missing,
        **prov,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        "gathered model %r: %d ids from %d shards (%d failed, %d missing) -> %s",
        model_name, len(ordered), len(shard_files), len(failed), len(missing), out_tsv,
    )
    return {
        "model": model_name,
        "n_ids": len(ordered),
        "n_shards": len(shard_files),
        "n_failed": len(failed),
        "failed_ids": failed[:50],
        "n_missing": len(missing),
        "missing_ids": missing[:50],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gather DCAlign shards into one cache per model.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="write a PARTIAL cache (WARN) instead of erroring when the shards did not "
             "produce every requested id (e.g. an OOM'd or timed-out shard).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_root = Path(args.run_root)
    manifest = json.loads((run_root / "dcalign" / "shards_manifest.json").read_text(encoding="utf-8"))
    cfg = cc.load_config(run_root / "config_snapshot.yaml")
    prov = _dcalign_provenance(cfg)

    statuses = [_gather_model(run_root, i, manifest, cfg, prov, allow_missing=args.allow_missing)
                for i in (0, 1)]
    status = {
        "models": statuses,
        "n_ids_requested": manifest["n_ids"],
        "partial": any(s["n_missing"] > 0 for s in statuses),
        **prov,
    }
    (run_root / "dcalign" / "gather_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    log.info("gather complete -> %s", run_root / "dcalign" / "gather_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
