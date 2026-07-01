#!/usr/bin/env python
"""potts_align align step — one shard (cluster entrypoint, iter-003).

Standalone CLI (imports only ``SBM.*`` + stdlib; invoked by path from
``pipeline/external/sbatch_potts_align_shard.sh``, not as a Snakemake script).
Pure numpy — no Julia, no DCAlign. Two modes:

``plan``  (run once, on the login node, by ``run_potts_align_align.sh``)
    Enumerate every in-scope ``(query_id, model)`` pair — a *flat* list (each
    task loads both models), so the expensive PPIC→CM cross block is spread
    evenly by round-robin rather than piling onto one model half. Classify each
    pair (home / cross / skip_NgtL / skip_subsample), derive its per-pair seed
    (identical to ``score_two_models``: ``master_seed + 2*r_idx + j``), and write
    ``<run_root>/potts_align/shards_manifest.json``.

``run``   (one Slurm array task per shard)
    Load both models, take this shard's pairs from the manifest, **skip pairs
    already present in the shard TSV** (resume after a timeout/kill), and score
    the rest with :func:`SBM.energy.potts_align.potts_align` into
    ``<run_root>/potts_align/cache/shards/shard_<NNN>.tsv``, flushing one row per
    pair. potts_align is deterministic per seed and thread-independent, so a Mac
    re-score with the same seed reproduces the energy bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from SBM import combine_config as cc
from SBM.energy import datasets
from SBM.energy.encoding import ints_to_seq, strip_gaps
from SBM.energy.model import load_model
from SBM.energy.potts_align import potts_align
from SBM.utils.potts_align_cache import (
    TSV_HEADER,
    PottsAlignCacheResult,
    format_row,
    read_shard_cache,
)

log = logging.getLogger(__name__)

#: SeedSequence spawn-key salt for the cross-subsample RNG. Fixed + logged so the
#: 8000-id subset is reproducible and independent of any other seeded stream.
_CROSS_SUBSAMPLE_SALT = 0xC205


def _manifest_path(run_root: Path) -> Path:
    return run_root / "potts_align" / "shards_manifest.json"


def _shard_tsv_path(run_root: Path, shard: int) -> Path:
    return run_root / "potts_align" / "cache" / "shards" / f"shard_{shard:03d}.tsv"


def _sorted_records(run_root: Path):
    """Query records in the canonical (sorted-by-id) order — the same order
    ``score_two_models`` uses, so the derived per-pair seeds match."""
    fasta = run_root / "query" / "query.fasta"
    groups = run_root / "query" / "groups.json"
    records = sorted(datasets.read_query_fasta(fasta, groups), key=lambda r: r.id)
    return records


def _model_entries(run_root: Path) -> list[dict]:
    return json.loads((run_root / "models.json").read_text(encoding="utf-8"))["models"]


def _cross_subset(records, sc, master_seed: int) -> tuple[set[str], int]:
    """The seeded id subset for the capped cross block, plus the seed used.

    Eligible ids are records whose ``origin_model == pa_cross_subsample_origin``
    (they are scored under ``pa_cross_subsample_under`` as the cost-driver cross
    block). ``pa_cross_subsample_n <= 0`` (or no eligible ids) ⇒ no subsample
    (every eligible pair is in scope)."""
    if not sc.pa_cross_subsample_n or sc.pa_cross_subsample_n <= 0:
        return set(), 0
    eligible = sorted(r.id for r in records if r.origin_model == sc.pa_cross_subsample_origin)
    seed_seq = np.random.SeedSequence(master_seed, spawn_key=(_CROSS_SUBSAMPLE_SALT,))
    cross_seed = int(seed_seq.generate_state(1)[0])
    rng = np.random.default_rng(seed_seq)
    n = min(sc.pa_cross_subsample_n, len(eligible))
    if n >= len(eligible):
        return set(eligible), cross_seed
    subset = set(rng.choice(np.array(eligible, dtype=object), size=n, replace=False).tolist())
    return subset, cross_seed


def cmd_plan(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    cfg = cc.load_config(run_root / "config_snapshot.yaml")
    sc = cfg.scoring
    if sc.method != "potts_align":
        log.warning("config scoring.method=%r (not 'potts_align'); planning anyway", sc.method)
    master_seed = cfg.seed

    entries = _model_entries(run_root)
    model_L = {m["name"]: int(m["L"]) for m in entries}
    records = _sorted_records(run_root)
    cross_subset, cross_seed = _cross_subset(records, sc, master_seed)

    pairs: list[dict] = []
    for r_idx, record in enumerate(records):
        n_res = int(strip_gaps(record.ints).size)
        for j, entry in enumerate(entries):
            model = entry["name"]
            L = model_L[model]
            seed = master_seed + 2 * r_idx + j
            if record.origin_model == model:
                status = "home"  # in a model's own frame ⇒ N <= L by construction
            elif n_res > L:
                status = "skip_NgtL"  # needs insertions; out of scope
            elif (record.origin_model == sc.pa_cross_subsample_origin
                  and model == sc.pa_cross_subsample_under
                  and record.id not in cross_subset):
                status = "skip_subsample"  # capped cross block, not in the seeded subset
            else:
                status = "cross"
            pairs.append({"query_id": record.id, "model": model, "status": status,
                          "r_idx": r_idx, "j": j, "seed": seed,
                          "n_residues": n_res, "gaps": L - n_res})

    in_scope = [i for i, p in enumerate(pairs) if p["status"] in ("home", "cross")]
    shards = [in_scope[k::args.n_shards] for k in range(args.n_shards)]

    manifest = {
        "n_shards": args.n_shards,
        "n_pairs_total": len(pairs),
        "n_pairs_in_scope": len(in_scope),
        "n_skip_NgtL": sum(1 for p in pairs if p["status"] == "skip_NgtL"),
        "n_skip_subsample": sum(1 for p in pairs if p["status"] == "skip_subsample"),
        "models": [m["name"] for m in entries],
        "master_seed": master_seed,
        "seed_derivation": "seed = master_seed + 2*r_idx + j (r_idx = index in id-sorted query, j = model index)",
        "cross_subsample": {
            "origin": sc.pa_cross_subsample_origin, "under": sc.pa_cross_subsample_under,
            "n": sc.pa_cross_subsample_n, "salt": _CROSS_SUBSAMPLE_SALT, "seed": cross_seed,
            "ids": sorted(cross_subset),
        },
        "random_control": {"n": cfg.query.n_random, "length": cfg.query.random_length,
                           "group": f"random/N{cfg.query.random_length}" if cfg.query.n_random else None},
        "pairs": pairs,
        "shards": shards,  # lists of indices into `pairs`
    }
    out = _manifest_path(run_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest), encoding="utf-8")
    sizes = [len(s) for s in shards]
    log.info(
        "shards_manifest: %d in-scope pairs (of %d) over %d shards (per-shard %d..%d); "
        "skip N>L=%d, skip_subsample=%d -> %s",
        len(in_scope), len(pairs), args.n_shards, min(sizes), max(sizes),
        manifest["n_skip_NgtL"], manifest["n_skip_subsample"], out,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    manifest = json.loads(_manifest_path(run_root).read_text(encoding="utf-8"))
    n_shards = manifest["n_shards"]
    if not 0 <= args.shard < n_shards:
        raise ValueError(f"--shard {args.shard} out of range [0, {n_shards})")

    entries = _model_entries(run_root)
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in entries}
    records = {r.id: r for r in _sorted_records(run_root)}
    all_pairs = manifest["pairs"]
    shard_pairs = [all_pairs[i] for i in manifest["shards"][args.shard]]

    shard_tsv = _shard_tsv_path(run_root, args.shard)
    shard_tsv.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_shard_cache(shard_tsv).keys()) if shard_tsv.is_file() else set()
    remaining = [p for p in shard_pairs if (p["query_id"], p["model"]) not in done]
    log.info("shard %d/%d: %d pairs, %d already done, %d remaining",
             args.shard, n_shards, len(shard_pairs), len(done), len(remaining))
    if not remaining:
        log.info("nothing to do for this shard (fully cached)")
        return 0

    new_file = not shard_tsv.is_file()
    with open(shard_tsv, "a", encoding="utf-8") as fh:
        if new_file:
            fh.write(TSV_HEADER + "\n")
            fh.flush()
        for p in remaining:
            record = records[p["query_id"]]
            model = models[p["model"]]
            raw = strip_gaps(record.ints)
            res = potts_align(raw, model, seed=p["seed"], sequence_id=p["query_id"])
            row = PottsAlignCacheResult(
                query_id=p["query_id"], model=model.name, n_residues=int(raw.size),
                gaps=int(model.L - raw.size), energy=float(res.best_energy),
                engine=res.method, is_global_exact=bool(res.is_global_exact),
                frame=ints_to_seq(res.best_frame), seed=int(p["seed"]),
            )
            fh.write(format_row(row) + "\n")
            fh.flush()
    log.info("shard %d complete -> %s", args.shard, shard_tsv)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="potts_align align step (one shard / plan).")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_plan = sub.add_parser("plan", help="write potts_align/shards_manifest.json")
    p_plan.add_argument("--run-root", required=True)
    p_plan.add_argument("--n-shards", type=int, required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="score one shard's (query, model) pairs")
    p_run.add_argument("--run-root", required=True)
    p_run.add_argument("--shard", type=int, required=True)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
