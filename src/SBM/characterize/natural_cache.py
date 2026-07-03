"""Model-scoped cache for natural-sequence TM-align results.

The natural controls (CM + PPIC family members) are folded once into the
top-level content-addressed store ``natural_folds/<sha8>/`` (``<sha8>`` = the
source FASTA's sha8) and TM-aligned against the two fixed reference folds. The
fold is a property of the FASTA content, not of any model run or combine run, so
two models sharing an MSA share it. Neither the fold nor the TM-align depends on
any design, so the natural TM-scores are a pure function of (natural PDB,
reference pair) — yet the combine ``characterize`` stage re-ran TMalign on **all**
naturals (~28k for CM+PPIC) every run, which dwarfs the 96 designs.

This module keys that TM-align result by the reference pair and stores it beside
the existing fold cache as::

    <sha8>/tm_vs_refs/<refkey>.tsv         # one structure_compare row per natural
    <sha8>/tm_vs_refs/<refkey>.meta.json   # provenance (ref paths/shas, chains, ...)

``<refkey>`` changes iff a reference PDB or its chain changes, so a stale cache is
impossible: a different reference mints a new file rather than silently reusing an
old alignment. ``tm_vs_refs`` is a distinct directory name from ``structures``, so
``sync_models.sh`` treats the (small) TSV like ``fold_scores/`` and mirrors it to
the Mac, while the (large) per-natural PDBs stay Midway-side.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Length (hex chars) of the reference-pair key embedded in the cache filename.
REFKEY_LEN = 12


def ref_pair_key(
    ref_a_sha256: str, chain_a: str, ref_b_sha256: str, chain_b: str,
    *, length: int = REFKEY_LEN,
) -> str:
    """Deterministic key for a (reference-A, reference-B) pair.

    Derived from each reference's content sha256 and selected chain. The
    extracted single-chain PDB that TMalign actually consumes is a deterministic
    function of (PDB content, chain id), so hashing those inputs is equivalent to
    hashing the extracted chains and cheaper. Not symmetric in A/B — the two
    references play distinct roles (fold A vs fold B).
    """
    sig = f"{ref_a_sha256}:{chain_a}|{ref_b_sha256}:{chain_b}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:length]


def cache_tsv(cache_dir: Path | str, refkey: str) -> Path:
    """Path of the cached natural TM-align TSV for ``refkey``."""
    return Path(cache_dir) / "tm_vs_refs" / f"{refkey}.tsv"


def cache_meta(cache_dir: Path | str, refkey: str) -> Path:
    """Path of the cache's provenance sidecar for ``refkey``."""
    return Path(cache_dir) / "tm_vs_refs" / f"{refkey}.meta.json"


def _ids_in_tsv(path: Path, id_col: str = "id") -> set[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row[id_col] for row in csv.DictReader(fh, delimiter="\t")}


def ids_in_fold_scores(cache_dir: Path | str) -> set[str]:
    """Union of sequence ids across a cache's ``fold_scores/*.tsv`` shards.

    This is the authoritative set of naturals for the model — the id list the TM
    cache must cover to be reusable.
    """
    ids: set[str] = set()
    for tsv in sorted((Path(cache_dir) / "fold_scores").glob("*.tsv")):
        ids |= _ids_in_tsv(tsv)
    return ids


def cache_covers(tsv_path: Path | str, required_ids: set[str]) -> bool:
    """True iff ``tsv_path`` exists and its id set is a superset of ``required_ids``.

    Superset (not equality) so a cache built over a larger id set still serves a
    query; a partial/interrupted prior write that is missing ids is a miss.
    """
    p = Path(tsv_path)
    if not p.exists() or not required_ids:
        return False
    return required_ids <= _ids_in_tsv(p)


def write_meta(
    meta_path: Path | str, *,
    refkey: str,
    ref_a: Path | str, ref_a_sha256: str, chain_a: str,
    ref_b: Path | str, ref_b_sha256: str, chain_b: str,
    tmalign: Path | str,
    n_rows: int,
    source_sha8: str,
) -> dict[str, object]:
    """Write the cache provenance sidecar and return the recorded dict."""
    meta: dict[str, object] = {
        "refkey": refkey,
        "ref_a": str(ref_a), "ref_a_sha256": ref_a_sha256, "chain_a": chain_a,
        "ref_b": str(ref_b), "ref_b_sha256": ref_b_sha256, "chain_b": chain_b,
        "tmalign": str(tmalign),
        "n_rows": n_rows,
        "source_sha8": source_sha8,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def merge_compare_tsvs(sources: list[Path], out: Path | str) -> int:
    """Union rows from several ``structure_compare`` TSVs into ``out`` (dedup by id).

    The first source to define an id wins; designs and naturals never share ids,
    so order only fixes determinism. The output header is taken from the first
    non-empty source, so all sources must share the ``compare_structures`` schema.
    Returns the number of unique rows written.
    """
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for src in sources:
        with open(src, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = reader.fieldnames or header
            for row in reader:
                rows.setdefault(row["id"], row)
    if header is None:
        raise ValueError(f"no rows in any of {[str(s) for s in sources]}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for _id, row in sorted(rows.items()):
            writer.writerow(row)
    return len(rows)
