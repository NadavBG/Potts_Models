#!/usr/bin/env python3
"""Fold one shard of a FASTA with ESMFold (single-sequence). GPU stage.

Runs under ``bioM3_env`` (torch + transformers). Reads a FASTA (plain designs
or an aligned family MSA with ``--degap``), takes the round-robin shard
``--shard`` of ``--n-shards``, folds each record, writes ``<id>.pdb`` into
``--out-structures``, and appends an ``id,group,length,plddt_mean,ptm`` row to
``--out-scores`` (flushed per row). Resumable: records whose PDB already exists
*and* whose id is already in the scores TSV are skipped, so a TIME_LIMIT kill
leaves a valid partial cache and a re-submit continues.

    PYTHONPATH=<repo>/src python scripts/characterize/fold_sequences.py \
        --fasta combine/.../design/designed_sequences.fasta \
        --out-structures combine/.../characterize/structures \
        --out-scores     combine/.../characterize/data/fold_scores.shard000.tsv \
        --group design --n-shards 1 --shard 0
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from SBM.characterize import fold

logger = logging.getLogger("fold_sequences")

_SCORE_COLUMNS = ["id", "group", "length", "plddt_mean", "ptm"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--out-structures", required=True, type=Path)
    p.add_argument("--out-scores", required=True, type=Path)
    p.add_argument("--group", required=True, help="group label (design / CM-natural / PPIC-natural)")
    p.add_argument("--degap", action="store_true", help="strip alignment gaps (for MSA FASTAs)")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--chunk-size", type=int, default=None, help="ESMFold trunk chunk size")
    p.add_argument("--no-half", action="store_true", help="disable fp16 LM (more memory)")
    p.add_argument("--limit", type=int, default=None, help="fold only the first N (probe)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = fold.read_fasta(args.fasta)
    shard = fold.shard_records(records, args.n_shards, args.shard)
    if args.limit is not None:
        shard = shard[: args.limit]
    logger.info("shard %d/%d: %d of %d records from %s",
                args.shard, args.n_shards, len(shard), len(records), args.fasta)

    structures_dir = args.out_structures
    structures_dir.mkdir(parents=True, exist_ok=True)
    args.out_scores.parent.mkdir(parents=True, exist_ok=True)

    # Resume against ALL shard TSVs in the scores dir, not just this shard's
    # file, so a change in --n-shards between runs does not re-fold the cache.
    done = fold.done_ids(args.out_scores.parent, structures_dir)
    todo = [(rid, seq) for rid, seq in shard if rid not in done]
    logger.info("%d already done, %d to fold", len(done), len(todo))

    # Clean (degap if requested) and drop non-canonical rows loudly.
    clean: list[tuple[str, str]] = []
    for rid, seq in todo:
        s = fold.degap(seq) if args.degap else seq.upper()
        if not fold.is_canonical(s):
            logger.warning("skipping %s: non-canonical residues after cleaning", rid)
            continue
        clean.append((rid, s))

    if not clean:
        logger.info("nothing to fold in this shard")
        return 0

    model = fold.EsmFold.load(device=args.device, chunk_size=args.chunk_size,
                              half_precision=not args.no_half)

    write_header = not args.out_scores.exists() or args.out_scores.stat().st_size == 0
    t_start = time.time()
    n_folded = 0
    with open(args.out_scores, "a", newline="", encoding="utf-8") as sfh:
        writer = csv.DictWriter(sfh, fieldnames=_SCORE_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
            sfh.flush()
        for rid, seq in clean:
            t0 = time.time()
            res = model.fold(seq)
            fold.write_pdb(res.pdb, structures_dir / f"{rid}.pdb")
            writer.writerow({
                "id": rid, "group": args.group, "length": res.length,
                "plddt_mean": f"{res.plddt_mean:.2f}", "ptm": f"{res.ptm:.4f}",
            })
            sfh.flush()
            n_folded += 1
            logger.info("[%d/%d] %s  L=%d pLDDT=%.1f pTM=%.3f  (%.1fs)",
                        n_folded, len(clean), rid, res.length, res.plddt_mean,
                        res.ptm, time.time() - t0)

    elapsed = time.time() - t_start
    per = elapsed / n_folded if n_folded else float("nan")
    logger.info("folded %d sequences in %.1fs  (%.2fs/seq)", n_folded, elapsed, per)
    print(f"FOLD_TIMING shard={args.shard} n={n_folded} elapsed_s={elapsed:.1f} "
          f"per_seq_s={per:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
