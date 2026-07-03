#!/usr/bin/env python3
"""BLAST designed sequences against SwissProt + the CM and PPIC families. CPU.

Three searches kept **cleanly separate**: SwissProt (annotated; reuses a
prebuilt DB), the CM family, and the PPIC family (built once by degapping the
family MSAs). Writes one raw ``-outfmt 6`` TSV per database plus a
human-readable ``blast_report.txt`` and a ``blast_manifest.json``. The
orchestrator (``characterize.py``) parses the raw TSVs into the summary.

    .venv/bin/python scripts/characterize/blast_sequences.py \
        --fasta combine/.../design/designed_sequences.fasta \
        --out-dir combine/.../characterize/data/blast \
        --swissprot-db <prefix> --swissprot-csv <csv> \
        --cm-fasta data/fasta/CM.fasta --ppic-fasta data/fasta/ppic_msa.fasta
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from SBM.characterize import blast, fold

logger = logging.getLogger("blast_sequences")

# Repo root, so default paths resolve regardless of cwd (the caslake job runs
# from /var/spool/slurm, where a relative "data/fasta/..." would not exist).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults confirmed by the environment survey (Midway; see docs/CHARACTERIZE.md).
DEFAULT_BLASTP = Path("/scratch/beagle3/nadavbg/conda_env/CM_env/bin/blastp")
DEFAULT_MAKEBLASTDB = Path("/scratch/beagle3/nadavbg/conda_env/CM_env/bin/makeblastdb")
DEFAULT_SWISSPROT_DB = Path(
    "/project/ranganathanr/nadavbg/BioM3/CM_workflow/runs/"
    "2026-03-25_fanout_test_2/blast_qc/db/swissprot"
)
DEFAULT_SWISSPROT_CSV = Path(
    "/project/ranganathanr/nadavbg/BioM3/reference_data/fully_annotated_swiss_prot.csv"
)


def _run_report(fh, label: str, query_ids: list[str],
                hits: dict[str, list[blast.BlastHit]],
                annotations: dict[str, str] | None, top_n: int) -> None:
    n_hit = sum(1 for q in query_ids if q in hits)
    fh.write(f"\n{'=' * 70}\n  {label}: {n_hit}/{len(query_ids)} designs with a hit\n{'=' * 70}\n")
    for qid in query_ids:
        bh = blast.best_hit(hits, qid)
        if bh is None:
            fh.write(f"  {qid:<22s}  no hit\n")
            continue
        ann = (annotations or {}).get(bh.sseqid, "")
        fh.write(f"  {qid:<22s}  {bh.sseqid:<24s} id={bh.pident:5.1f}% cov={bh.qcovs:3d}% "
                 f"e={bh.evalue:.1e} bits={bh.bitscore:6.0f}"
                 f"{('  ' + ann) if ann else ''}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", required=True, type=Path, help="design sequences (plain FASTA)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--swissprot-db", type=Path, default=DEFAULT_SWISSPROT_DB)
    p.add_argument("--swissprot-csv", type=Path, default=DEFAULT_SWISSPROT_CSV)
    p.add_argument("--cm-fasta", type=Path, default=_REPO_ROOT / "data/fasta/CM.fasta")
    p.add_argument("--ppic-fasta", type=Path, default=_REPO_ROOT / "data/fasta/ppic_msa.fasta")
    p.add_argument("--blastp", type=Path, default=DEFAULT_BLASTP)
    p.add_argument("--makeblastdb", type=Path, default=DEFAULT_MAKEBLASTDB)
    p.add_argument("--evalue", type=float, default=10.0)
    p.add_argument("--max-target-seqs", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--top-n", type=int, default=3)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for tool in (args.blastp, args.makeblastdb):
        if not Path(tool).exists():
            raise SystemExit(f"BLAST binary not found: {tool}")

    out = args.out_dir
    db_dir = out / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Query FASTA (designs, degapped defensively).
    records = [(rid, fold.degap(seq)) for rid, seq in fold.read_fasta(args.fasta)]
    query_ids = [rid for rid, _ in records]
    query_fasta = out / "query.fasta"
    blast.write_query_fasta(records, query_fasta)
    logger.info("%d design queries", len(records))

    # Databases: (label, db_prefix, out_tsv, annotations).
    jobs: list[tuple[str, Path, Path, dict[str, str] | None]] = []

    # SwissProt: reuse the prebuilt DB (loud error if absent).
    if not Path(f"{args.swissprot_db}.psq").exists() and not Path(f"{args.swissprot_db}.00.psq").exists():
        raise SystemExit(f"prebuilt SwissProt DB not found at prefix {args.swissprot_db}")
    logger.info("loading SwissProt annotations ...")
    sp_ann = blast.load_swissprot_annotations(args.swissprot_csv)
    jobs.append(("SwissProt", args.swissprot_db, out / "blast_swissprot.tsv", sp_ann))

    # Family DBs: degap + makeblastdb once.
    for label, fam_fasta, name in [("CM family", args.cm_fasta, "cmfam"),
                                   ("PPIC family", args.ppic_fasta, "ppicfam")]:
        degapped = db_dir / f"{name}.fasta"
        n = blast.degap_fasta(fam_fasta, degapped)
        db = blast.build_db(degapped, db_dir / name, makeblastdb=args.makeblastdb, title=name)
        logger.info("%s DB: %d sequences", label, n)
        jobs.append((label, db, out / f"blast_{name}.tsv", None))

    # Run each search.
    manifest: dict[str, object] = {"query_fasta": str(query_fasta),
                                   "n_queries": len(records), "databases": {}}
    parsed: dict[str, dict[str, list[blast.BlastHit]]] = {}
    for label, db, tsv, _ann in jobs:
        elapsed = blast.run_blastp(query_fasta, db, tsv, blastp=args.blastp,
                                   evalue=args.evalue, max_target_seqs=args.max_target_seqs,
                                   num_threads=args.threads)
        hits = blast.parse_blast_tsv(tsv)
        parsed[label] = hits
        n_hit = sum(1 for q in query_ids if q in hits)
        manifest["databases"][label] = {"db": str(db), "raw_tsv": str(tsv),
                                        "n_with_hit": n_hit, "elapsed_s": round(elapsed, 2)}
        logger.info("%s: %d/%d hits (%.1fs)", label, n_hit, len(records), elapsed)

    # Human-readable report.
    report = out / "blast_report.txt"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("BLAST QC — designed sequences\n")
        for label, _db, _tsv, ann in jobs:
            _run_report(fh, label, query_ids, parsed[label], ann, args.top_n)
    (out / "blast_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
    logger.info("wrote %s + blast_manifest.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
