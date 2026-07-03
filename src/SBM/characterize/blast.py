"""Self-contained protein BLAST for characterizing designed sequences.

Copies (does **not** import) the short-sequence blastp recipe proven in
``BioM3/CM_workflow/scripts/blast_qc.py`` — that script lives in another
repo and only ingests BioM3 ``.pt``/CSV, so we reimplement the ~80-residue
tuning (``-word_size 2 -matrix BLOSUM62 -seg no -comp_based_stats 0``,
``-outfmt 6``) here, driving BLAST+ binaries from the ``CM_env`` conda env.

Designs are searched against three databases kept **cleanly separate** —
SwissProt (annotated), the CM family, and the PPIC family — so the summary
reports "top SwissProt hit" and "top CM-family hit" and "top PPIC-family
hit" as distinct columns, never a merged "blast hit".
"""

from __future__ import annotations

import csv
import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

#: Tabular BLAST fields (order matters: parser zips against this header).
BLAST_OUTFMT = (
    "6 qseqid sseqid pident length mismatch gapopen "
    "qstart qend sstart send evalue bitscore qlen slen qcovs"
)
BLAST_OUTFMT_HEADER = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    "qlen", "slen", "qcovs",
]
_FLOAT_FIELDS = ("pident", "evalue", "bitscore")
_INT_FIELDS = (
    "length", "mismatch", "gapopen", "qstart", "qend",
    "sstart", "send", "qlen", "slen", "qcovs",
)


class BlastHit(NamedTuple):
    """One parsed BLAST hit row."""

    sseqid: str
    pident: float
    length: int
    qcovs: int
    evalue: float
    bitscore: float
    qlen: int
    slen: int


# ── Database preparation ────────────────────────────────────────────────────


def degap_fasta(in_fasta: Path | str, out_fasta: Path | str) -> int:
    """Write a gap-stripped copy of an (aligned) FASTA. Returns record count.

    ``makeblastdb`` rejects ``-`` in protein sequences, so family MSAs must be
    degapped first. The record id (first whitespace token of the header) is
    preserved so a hit reports which family member it matched. Empty rows are
    dropped (logged count).
    """
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_empty = 0
    with open(in_fasta, encoding="utf-8") as fin, open(
        out_fasta, "w", encoding="utf-8"
    ) as fout:
        header: str | None = None
        chunks: list[str] = []

        def flush() -> None:
            nonlocal n_written, n_empty
            if header is None:
                return
            seq = "".join(chunks).replace("-", "").replace(".", "").upper()
            if not seq:
                n_empty += 1
                return
            fout.write(f">{header}\n{seq}\n")
            n_written += 1

        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                toks = line[1:].split()
                header = toks[0] if toks else line[1:].strip()
                chunks = []
            elif header is not None:
                chunks.append(line.strip())
        flush()
    if n_empty:
        logger.warning("degap_fasta: dropped %d empty record(s)", n_empty)
    return n_written


def build_db(
    fasta: Path | str, db_path: Path | str, *, makeblastdb: Path | str, title: str
) -> Path:
    """Build (or reuse) a protein BLAST database from a FASTA."""
    db_path = Path(db_path)
    if db_path.with_suffix(".psq").exists() or Path(f"{db_path}.00.psq").exists():
        logger.info("reusing existing BLAST DB: %s", db_path)
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(makeblastdb), "-in", str(fasta), "-dbtype", "prot",
         "-out", str(db_path), "-title", title],
        check=True, capture_output=True, text=True,
    )
    logger.info("built BLAST DB: %s", db_path)
    return db_path


# ── Running + parsing ───────────────────────────────────────────────────────


def write_query_fasta(records: list[tuple[str, str]], path: Path | str) -> None:
    """Write ``[(id, seq), ...]`` as a FASTA query file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n{seq}\n")


def run_blastp(
    query_fasta: Path | str,
    db_path: Path | str,
    out_tsv: Path | str,
    *,
    blastp: Path | str,
    evalue: float = 10.0,
    max_target_seqs: int = 5,
    num_threads: int = 8,
) -> float:
    """Run blastp with the short-sequence tuning. Returns elapsed seconds."""
    cmd = [
        str(blastp),
        "-query", str(query_fasta),
        "-db", str(db_path),
        "-out", str(out_tsv),
        "-outfmt", BLAST_OUTFMT,
        "-evalue", str(evalue),
        "-max_target_seqs", str(max_target_seqs),
        "-num_threads", str(num_threads),
        # Tuning for short (~80 aa) protein queries (from blast_qc.py):
        "-word_size", "2",
        "-matrix", "BLOSUM62",
        "-seg", "no",
        "-comp_based_stats", "0",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"blastp failed (exit {proc.returncode}):\n{proc.stderr}")
    return elapsed


def parse_blast_tsv(tsv_path: Path | str) -> dict[str, list[BlastHit]]:
    """Parse tabular BLAST output into ``{qseqid: [BlastHit, ...]}`` (hit order
    preserved = descending bitscore as emitted by blastp)."""
    hits: dict[str, list[BlastHit]] = defaultdict(list)
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(BLAST_OUTFMT_HEADER):
                continue
            row = dict(zip(BLAST_OUTFMT_HEADER, fields))
            hits[row["qseqid"]].append(
                BlastHit(
                    sseqid=row["sseqid"],
                    pident=float(row["pident"]),
                    length=int(row["length"]),
                    qcovs=int(row["qcovs"]),
                    evalue=float(row["evalue"]),
                    bitscore=float(row["bitscore"]),
                    qlen=int(row["qlen"]),
                    slen=int(row["slen"]),
                )
            )
    return hits


def best_hit(hits: dict[str, list[BlastHit]], query_id: str) -> BlastHit | None:
    """Best (highest-bitscore, i.e. first) hit for a query, or None."""
    lst = hits.get(query_id)
    return lst[0] if lst else None


# ── SwissProt annotation lookup ─────────────────────────────────────────────


def load_swissprot_annotations(
    csv_path: Path | str, acc_col: str = "primary_Accession"
) -> dict[str, str]:
    """Map ``{accession: text_caption}`` from the SwissProt CSV.

    The prebuilt SwissProt DB was formatted with headers ``>{primary_Accession}``
    (no ``-parse_seqids``), so a hit's ``sseqid`` equals ``primary_Accession``
    and looks up directly here.
    """
    lookup: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        caption_col = None
        for col in ("text_caption", "[final]text_caption", "[clean]text_caption"):
            if reader.fieldnames and col in reader.fieldnames:
                caption_col = col
                break
        if caption_col is None:
            logger.warning("no caption column in %s; annotations disabled", csv_path)
            return lookup
        for row in reader:
            acc = row.get(acc_col, "")
            caption = (row.get(caption_col, "") or "").strip()
            if acc and caption:
                lookup[acc] = caption
    return lookup
