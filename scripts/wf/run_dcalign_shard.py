#!/usr/bin/env python
"""DCAlign align step — one shard (cluster entrypoint, spec §10.9).

Standalone CLI (imports only ``SBM.*`` + stdlib; invoked by path from
``pipeline/external/sbatch_dcalign_shard.sh``, not as a Snakemake script). Two
modes:

``plan``  (run once, on the login node, by ``run_dcalign_align.sh``)
    Split the **sorted** query-id list (the same order
    ``score_two_models`` uses) into ``n_shards`` round-robin chunks and write
    ``<run_root>/dcalign/shards_manifest.json``. Round-robin spreads the
    heterogeneous per-sequence cost (max ~451 s when N<L) across shards.

``run``   (one Slurm array task per ``(model, shard)``)
    Load this task's model, take its shard's ids from the manifest, **skip ids
    already present in the shard TSV** (resume after a timeout/kill), and align
    the rest with DCAlign into
    ``<run_root>/dcalign/cache/<model>/shards/shard_<NNN>.tsv``. The Julia driver
    appends one flushed row per sequence, so a killed task leaves a valid
    partial cache and the next run continues where it stopped.

Every query is scored under BOTH models downstream, so each shard's id list is
shared across models; ``--model-index`` selects which model's couplings to use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from SBM import combine_config as cc
from SBM.energy import datasets
from SBM.energy.encoding import strip_gaps
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import align_sequences, dcalign_context, read_alignment_cache

log = logging.getLogger(__name__)


def _sorted_query_ids_and_records(run_root: Path):
    """Query records in the canonical (sorted-by-id) order, plus an id→record map."""
    fasta = run_root / "query" / "query.fasta"
    groups = run_root / "query" / "groups.json"
    records = sorted(datasets.read_query_fasta(fasta, groups), key=lambda r: r.id)
    return [r.id for r in records], {r.id: r for r in records}


def _model_entries(run_root: Path) -> list[dict]:
    return json.loads((run_root / "models.json").read_text(encoding="utf-8"))["models"]


def _shards_manifest_path(run_root: Path) -> Path:
    return run_root / "dcalign" / "shards_manifest.json"


def _round_robin(ids: list[str], n_shards: int) -> list[list[str]]:
    return [ids[k::n_shards] for k in range(n_shards)]


def cmd_plan(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    ids, _ = _sorted_query_ids_and_records(run_root)
    names = [m["name"] for m in _model_entries(run_root)]
    shards = _round_robin(ids, args.n_shards)
    manifest = {
        "n_shards": args.n_shards,
        "n_ids": len(ids),
        "models": names,
        "shards": shards,
    }
    out = _shards_manifest_path(run_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sizes = [len(s) for s in shards]
    log.info(
        "shards_manifest: %d ids over %d shards (per-shard %d..%d) x %d models -> %s",
        len(ids), args.n_shards, min(sizes), max(sizes), len(names), out,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    manifest = json.loads(_shards_manifest_path(run_root).read_text(encoding="utf-8"))
    n_shards = manifest["n_shards"]
    if not 0 <= args.shard < n_shards:
        raise ValueError(f"--shard {args.shard} out of range [0, {n_shards})")
    if args.model_index not in (0, 1):
        raise ValueError(f"--model-index must be 0 or 1, got {args.model_index}")

    model_entry = _model_entries(run_root)[args.model_index]
    model_name = model_entry["name"]
    if manifest["models"][args.model_index] != model_name:
        raise RuntimeError(
            f"models.json[{args.model_index}]={model_name!r} disagrees with "
            f"shards_manifest models[{args.model_index}]={manifest['models'][args.model_index]!r}"
        )
    shard_ids: list[str] = manifest["shards"][args.shard]

    cfg = cc.load_config(run_root / "config_snapshot.yaml")
    sc = cfg.scoring
    if sc.method != "dcalign":
        log.warning("config scoring.method=%r (not 'dcalign'); aligning anyway", sc.method)

    model = load_model(model_entry["model_path"], name=model_name)
    _, records = _sorted_query_ids_and_records(run_root)

    shard_tsv = run_root / "dcalign" / "cache" / model_name / "shards" / f"shard_{args.shard:03d}.tsv"
    shard_tsv.parent.mkdir(parents=True, exist_ok=True)

    done = set(read_alignment_cache(shard_tsv)) if shard_tsv.is_file() else set()
    remaining = [sid for sid in shard_ids if sid not in done]
    log.info(
        "model=%r shard=%d/%d: %d ids, %d already done, %d remaining",
        model_name, args.shard, n_shards, len(shard_ids), len(done), len(remaining),
    )
    if not remaining:
        log.info("nothing to do for this shard (fully cached)")
        return 0

    missing = [sid for sid in remaining if sid not in records]
    if missing:
        raise KeyError(f"shard ids absent from query.fasta: {missing[:5]}")
    seqs = [strip_gaps(records[sid].ints) for sid in remaining]

    ctx = dcalign_context(
        dcalign_path=sc.dcalign_path,
        julia=sc.julia,
        maxiter=sc.maxiter,
        seed=sc.dcalign_seed,
        pcount=sc.pcount,
        threads=args.threads,
    )
    log.info(
        "DCAlign clone=%s commit=%s julia=%s",
        ctx.dcalign_path, ctx.dcalign_git_commit, ctx.julia_version,
    )
    workdir = run_root / "dcalign" / "cache" / model_name / "work" / f"shard_{args.shard:03d}"
    align_sequences(
        ctx, model, seqs, remaining,
        out_dir=workdir, out_tsv=shard_tsv, lambda_spec=sc.lambda_spec,
    )
    log.info("shard %d for model %r complete -> %s", args.shard, model_name, shard_tsv)
    return 0


def _default_threads() -> int:
    env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    try:
        return max(1, int(env)) if env else 1
    except ValueError:
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DCAlign align step (one shard / plan).")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_plan = sub.add_parser("plan", help="write dcalign/shards_manifest.json")
    p_plan.add_argument("--run-root", required=True)
    p_plan.add_argument("--n-shards", type=int, required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="align one shard for one model")
    p_run.add_argument("--run-root", required=True)
    p_run.add_argument("--model-index", type=int, required=True, help="0 or 1")
    p_run.add_argument("--shard", type=int, required=True)
    p_run.add_argument("--threads", type=int, default=_default_threads())
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
