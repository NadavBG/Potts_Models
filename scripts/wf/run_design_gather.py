#!/usr/bin/env python
"""Two-model design gather step — merge shard JSONL into the run outputs.

Mirrors ``run_potts_align_gather.py``. Reads every ``<run_dir>/shards/shard_*.jsonl``
(one ``ChainResult`` per line), reconstructs the results, and writes the same
artifacts the local CLI (``design_two_model.py``) produces — ``trajectories.npz``,
``designed_sequences.fasta``, ``designed.tsv``, ``design_manifest.json`` — plus a
``gather_status.json``. Two gates: every planned chain must be present (unless
``--allow-missing``), and the warm-started polish must never be *worse* than the
joint-MC frame (``E_polish <= E_mc``; the whole point of warm-starting).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from SBM.design.anneal import ChainResult  # noqa: E402
from SBM.energy.model import load_model  # noqa: E402

import design_two_model as d2m  # noqa: E402

log = logging.getLogger(__name__)
_CANARY_TOL = 1e-6


def _read_shards(run_dir: Path) -> dict[int, ChainResult]:
    results: dict[int, ChainResult] = {}
    for shard_file in sorted((run_dir / "shards").glob("shard_*.jsonl")):
        for line in shard_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            results[int(d["chain_index"])] = ChainResult.from_dict(d)
    return results


def _polish_canary(results: list[ChainResult]) -> list[int]:
    """Chains where the (warm-started) polish came out worse than the MC frame."""
    bad = []
    for r in results:
        if r.E_A_polish is None or r.E_B_polish is None:
            continue
        if r.E_A_polish > r.E_A_mc + _CANARY_TOL or r.E_B_polish > r.E_B_mc + _CANARY_TOL:
            bad.append(r.chain_index)
    return bad


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--allow-missing", action="store_true",
                   help="write outputs even if some planned chains are absent")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "design_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "shards_manifest.json").read_text(encoding="utf-8"))
    n_chains = int(manifest["n_chains"])

    by_id = _read_shards(run_dir)
    missing = [i for i in range(n_chains) if i not in by_id]
    if missing and not args.allow_missing:
        raise SystemExit(
            f"gather: {len(missing)} of {n_chains} chains missing (e.g. {missing[:10]}); "
            "rerun the shards or pass --allow-missing")
    results = [by_id[i] for i in sorted(by_id)]
    if not results:
        raise SystemExit("gather: no chain results found under shards/")

    bad = _polish_canary(results)
    if bad:
        raise SystemExit(
            f"gather canary FAILED: warm-started polish worse than the MC frame for chains {bad}")

    model_A = load_model(cfg["model_a_path"], name=cfg["name_a"])
    model_B = load_model(cfg["model_b_path"], name=cfg["name_b"])
    np.savez_compressed(run_dir / "trajectories.npz", **d2m._stack_trajectories(results))
    d2m._write_fasta(results, run_dir / "designed_sequences.fasta")
    d2m._write_alignments(results, model_A, model_B, run_dir)
    df = d2m._write_table(results, model_A.name, model_B.name, run_dir / "designed.tsv")

    now = dt.datetime.now(dt.timezone.utc)
    d2m._write_manifest(run_dir / "design_manifest.json", config=cfg, results=results,
                        models=(model_A, model_B), started=now, finished=now, out_dir=run_dir)
    (run_dir / "gather_status.json").write_text(json.dumps(
        {"n_chains": n_chains, "n_gathered": len(results), "n_missing": len(missing),
         "canary_ok": True, "median_E_tot_mc": float(df["E_tot_mc"].median())}, indent=2) + "\n",
        encoding="utf-8")
    log.info("gathered %d/%d chains (missing %d) -> %s",
             len(results), n_chains, len(missing), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
