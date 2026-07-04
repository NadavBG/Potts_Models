#!/usr/bin/env python
"""Two-model design align/run step — one shard (cluster entrypoint).

Mirrors ``run_potts_align_shard.py`` but the work unit is an *annealing chain*
(``SBM.design.anneal.anneal_chain``), not a (query, model) pair. Standalone CLI
(invoked by path from ``pipeline/external/sbatch_design_shard.sh``). Two modes:

``plan``  (once, login node)
    Read ``<run_dir>/design_config.json`` (written on the Mac by
    ``design_two_model.py --emit-config-only``), assign chain indices
    ``0..n_chains-1`` round-robin to ``--n-shards`` shards, write
    ``<run_dir>/shards_manifest.json``.

``run``   (one Slurm array task per shard)
    Run this shard's chains (per-chain seed = ``master_seed + chain_index``, so
    every chain is reproducible and independent), **skip chains already present
    in the shard JSONL** (resume after a timeout/kill), and flush one
    ``ChainResult.as_dict()`` JSON line per chain to
    ``<run_dir>/shards/shard_<NNN>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# scripts/ on sys.path so we can reuse the design CLI's polish-schedule table.
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from SBM.design.anneal import AnnealSchedule, anneal_chain  # noqa: E402
from SBM.energy.model import load_model  # noqa: E402

import design_two_model as d2m  # noqa: E402

log = logging.getLogger(__name__)


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "design_config.json").read_text(encoding="utf-8"))


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "shards_manifest.json"


def _shard_path(run_dir: Path, shard: int) -> Path:
    return run_dir / "shards" / f"shard_{shard:03d}.jsonl"


def _done_chains(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    return {json.loads(line)["chain_index"] for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def cmd_plan(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    cfg = _config(run_dir)
    n = int(cfg["n_chains"])
    shards = [list(range(k, n, args.n_shards)) for k in range(args.n_shards)]
    manifest = {
        "n_shards": args.n_shards, "n_chains": n, "master_seed": cfg["master_seed"],
        "do_polish": cfg["do_polish"], "polish_schedule": cfg["polish_schedule"],
        "seed_derivation": "seed = master_seed + chain_index",
        "shards": shards,
    }
    _manifest_path(run_dir).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sizes = [len(s) for s in shards]
    log.info("shards_manifest: %d chains over %d shards (per-shard %d..%d) -> %s",
             n, args.n_shards, min(sizes), max(sizes), _manifest_path(run_dir))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    cfg = _config(run_dir)
    manifest = json.loads(_manifest_path(run_dir).read_text(encoding="utf-8"))
    if not 0 <= args.shard < manifest["n_shards"]:
        raise ValueError(f"--shard {args.shard} out of range [0, {manifest['n_shards']})")
    chain_ids = manifest["shards"][args.shard]

    sched = AnnealSchedule(**cfg["schedule"])
    polish_pt = d2m._POLISH_SCHEDULES[cfg["polish_schedule"]]
    model_A = load_model(cfg["model_a_path"], name=cfg["name_a"])
    model_B = load_model(cfg["model_b_path"], name=cfg["name_b"])
    msa_a, msa_b = d2m._load_seed_msas(cfg)   # None unless natural starts are requested

    out = _shard_path(run_dir, args.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_chains(out)
    remaining = [i for i in chain_ids if i not in done]
    log.info("shard %d/%d: %d chains, %d done, %d remaining",
             args.shard, manifest["n_shards"], len(chain_ids), len(done), len(remaining))

    with open(out, "a", encoding="utf-8") as fh:
        for i in remaining:
            start_type, init = d2m.build_initial_state(cfg, i, model_A, model_B, msa_a, msa_b)
            res = anneal_chain(model_A, model_B, cfg["w_a"], cfg["w_b"], sched,
                               seed=cfg["master_seed"] + i, chain_index=i,
                               do_polish=cfg["do_polish"], polish_pt_schedule=polish_pt,
                               init_state=init, start_type=start_type)
            fh.write(json.dumps(res.as_dict()) + "\n")
            fh.flush()
    log.info("shard %d complete -> %s", args.shard, out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="two-model design (one shard / plan).")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_plan = sub.add_parser("plan", help="write shards_manifest.json")
    p_plan.add_argument("--run-dir", required=True)
    p_plan.add_argument("--n-shards", type=int, required=True)
    p_plan.set_defaults(func=cmd_plan)
    p_run = sub.add_parser("run", help="run one shard's chains")
    p_run.add_argument("--run-dir", required=True)
    p_run.add_argument("--shard", type=int, required=True)
    p_run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
