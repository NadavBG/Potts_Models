#!/usr/bin/env python3
"""TM-align every predicted structure against the two reference folds. CPU stage.

Reads one or more gathered ``fold_scores.tsv`` (``id,group,...``) to know which
sequences were folded and their group, resolves each ``<id>.pdb`` across the
given ``--structures-dir`` search path, extracts single-chain reference PDBs
from ``--ref-a``/``--ref-b`` once, then runs ``TMalign <model> <ref>`` for both
references (multiprocessed over ids). Writes a tidy ``structure_compare.tsv``.

    .venv/bin/python scripts/characterize/compare_structures.py \
        --fold-scores combine/.../characterize/data/fold_scores.tsv \
        --structures-dir combine/.../characterize/structures \
        --ref-a data/structures/1ECM.pdb --chain-a A \
        --ref-b data/structures/1JNT.pdb --chain-b A \
        --tmalign pipeline/bin/TMalign \
        --out combine/.../characterize/data/structure_compare.tsv --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from SBM.characterize import tmscore

logger = logging.getLogger("compare_structures")

_COLUMNS = [
    "id", "group",
    "tm_query_A", "tm_ref_A", "rmsd_A", "aligned_len_A", "seq_id_A",
    "tm_query_B", "tm_ref_B", "rmsd_B", "aligned_len_B", "seq_id_B",
]


def _resolve_pdb(rid: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        cand = d / f"{rid}.pdb"
        if cand.exists():
            return cand
    return None


def _compare_one(
    rid: str, group: str, pdb: str, ref_a: str, ref_b: str, tmalign: str
) -> dict[str, str]:
    ra = tmscore.run_tmalign(tmalign, pdb, ref_a)
    rb = tmscore.run_tmalign(tmalign, pdb, ref_b)
    return {
        "id": rid, "group": group,
        "tm_query_A": f"{ra.tm_query:.5f}", "tm_ref_A": f"{ra.tm_ref:.5f}",
        "rmsd_A": f"{ra.rmsd:.3f}", "aligned_len_A": str(ra.aligned_len),
        "seq_id_A": f"{ra.seq_id:.4f}",
        "tm_query_B": f"{rb.tm_query:.5f}", "tm_ref_B": f"{rb.tm_ref:.5f}",
        "rmsd_B": f"{rb.rmsd:.3f}", "aligned_len_B": str(rb.aligned_len),
        "seq_id_B": f"{rb.seq_id:.4f}",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold-scores", required=True, type=Path, nargs="+")
    p.add_argument("--structures-dir", required=True, type=Path, nargs="+")
    p.add_argument("--ref-a", required=True, type=Path)
    p.add_argument("--chain-a", default="A")
    p.add_argument("--ref-b", required=True, type=Path)
    p.add_argument("--chain-b", default="A")
    p.add_argument("--tmalign", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--work-dir", type=Path, default=None,
                   help="where to write extracted ref chains (default: alongside --out)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not Path(args.tmalign).exists():
        raise SystemExit(f"TMalign binary not found: {args.tmalign} "
                         "(build with pipeline/external/build_tmalign.sh)")

    # Collect (id, group) from the fold-scores TSVs.
    id_group: dict[str, str] = {}
    for tsv in args.fold_scores:
        with open(tsv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                id_group[row["id"]] = row.get("group", "")
    logger.info("%d sequences to compare", len(id_group))

    # Extract single-chain reference PDBs once.
    work = args.work_dir or (args.out.parent / "ref_chains")
    work.mkdir(parents=True, exist_ok=True)
    ref_a = work / f"{args.ref_a.stem}_{args.chain_a}.pdb"
    ref_b = work / f"{args.ref_b.stem}_{args.chain_b}.pdb"
    na = tmscore.extract_chain(args.ref_a, args.chain_a, ref_a)
    nb = tmscore.extract_chain(args.ref_b, args.chain_b, ref_b)
    logger.info("reference A %s chain %s: %d residues", args.ref_a.name, args.chain_a, na)
    logger.info("reference B %s chain %s: %d residues", args.ref_b.name, args.chain_b, nb)

    # Build the work list, warning on missing PDBs.
    jobs: list[tuple[str, str, str]] = []
    n_missing = 0
    for rid, group in sorted(id_group.items()):
        pdb = _resolve_pdb(rid, args.structures_dir)
        if pdb is None:
            logger.warning("no PDB for %s in %s", rid, [str(d) for d in args.structures_dir])
            n_missing += 1
            continue
        jobs.append((rid, group, str(pdb)))
    if n_missing:
        logger.warning("%d/%d sequences have no structure and will be skipped",
                       n_missing, len(id_group))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_compare_one, rid, group, pdb, str(ref_a), str(ref_b),
                          str(args.tmalign)): rid
                for rid, group, pdb in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 500 == 0 or i == len(futs):
                logger.info("compared %d/%d", i, len(futs))

    results.sort(key=lambda r: r["id"])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    logger.info("wrote %d rows -> %s", len(results), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
