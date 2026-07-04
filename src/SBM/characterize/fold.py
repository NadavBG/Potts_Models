"""ESMFold single-sequence structure prediction + FASTA/sharding helpers.

ESMFold (``facebook/esmfold_v1`` via HuggingFace ``transformers``) folds a
bare amino-acid sequence with no MSA — the correct predictor for de novo
designs, whose "MSA" would be ill-defined. Confidence is the per-residue
**pLDDT** (0–100, read from the output PDB B-factor column) and the global
**pTM**.

torch/transformers are imported lazily *inside* the runtime functions so
this module imports cleanly in envs without them (the CPU ``.venv`` and the
test suite). The GPU fold stage runs under ``bioM3_env`` (torch 2.0.1+cu117,
transformers 4.29), which vendors openfold — no separate openfold install.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger(__name__)

#: HuggingFace model id. Weights (~2.5 GB) download to the HF cache on first
#: use; pre-fetch on a login node (compute nodes may lack outbound network).
ESMFOLD_MODEL_ID = "facebook/esmfold_v1"

#: The 20 canonical amino acids ESMFold accepts. Anything else (gap, X, U, ...)
#: must be resolved/stripped by the caller before folding.
_CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ── FASTA I/O (self-contained; no BioPython so it imports under bioM3_env) ──


def read_fasta(path: Path | str) -> list[tuple[str, str]]:
    """Parse a FASTA file into ``[(record_id, sequence), ...]``.

    ``record_id`` is the first whitespace-delimited token after ``>``.
    Multi-line sequences are concatenated. Sequence is uppercased; no other
    normalization (the caller degaps / validates).
    """
    records: list[tuple[str, str]] = []
    rec_id: str | None = None
    chunks: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if rec_id is not None:
                    records.append((rec_id, "".join(chunks)))
                rec_id = line[1:].split()[0] if line[1:].split() else line[1:].strip()
                chunks = []
            elif rec_id is not None:
                chunks.append(line.strip())
    if rec_id is not None:
        records.append((rec_id, "".join(chunks)))
    return [(rid, seq.upper()) for rid, seq in records]


def degap(seq: str) -> str:
    """Remove alignment gap characters (``-`` and ``.``) and uppercase."""
    return seq.replace("-", "").replace(".", "").upper()


def is_canonical(seq: str) -> bool:
    """True iff every residue is one of the 20 canonical amino acids."""
    return bool(seq) and set(seq) <= _CANONICAL_AA


# ── Sharding (stable, index-based round-robin) ──────────────────────────────


def shard_indices(n_items: int, n_shards: int, shard: int) -> list[int]:
    """Indices assigned to ``shard`` under round-robin partitioning.

    Round-robin (``i % n_shards == shard``) balances per-shard sequence
    length better than contiguous blocks. Deterministic: the union over
    ``shard in range(n_shards)`` is exactly ``range(n_items)`` with no
    overlap. Mirrors the flat-shard convention of the potts_align scripts.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not (0 <= shard < n_shards):
        raise ValueError(f"shard {shard} out of range [0, {n_shards})")
    return list(range(shard, n_items, n_shards))


def shard_records(
    records: list[tuple[str, str]], n_shards: int, shard: int
) -> list[tuple[str, str]]:
    """Apply :func:`shard_indices` to a list of ``(id, seq)`` records."""
    idx = shard_indices(len(records), n_shards, shard)
    return [records[i] for i in idx]


# ── Resume: which ids are already folded ────────────────────────────────────


def done_ids(scores_dir: Path | str, structures_dir: Path | str) -> set[str]:
    """Ids already fully folded: a recorded score row AND a ``<id>.pdb`` on disk.

    Scans **every** ``*.tsv`` shard in ``scores_dir`` (not just the current
    shard's file), so resume is robust to a change in ``n_shards`` between runs.
    The round-robin partition reshuffles when the shard count changes, but the
    per-id ``<id>.pdb`` + score row is shared, content-addressed state that any
    shard may have produced. Reading only the current shard's TSV — the old
    behaviour — silently re-folds *and* re-appends every already-cached sequence
    whenever ``n_shards`` differs from the run that built the cache, which both
    wastes GPU-hours and corrupts the cache with cross-shard duplicate rows.

    The PDB check guards a torn write (a row flushed before its PDB, or a PDB
    deleted): such an id is treated as not-done and re-folded.
    """
    scores_dir = Path(scores_dir)
    if not scores_dir.is_dir():
        return set()
    structures_dir = Path(structures_dir)
    done: set[str] = set()
    for tsv in sorted(scores_dir.glob("*.tsv")):
        with open(tsv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                rid = row.get("id", "")
                if rid and rid not in done and (structures_dir / f"{rid}.pdb").exists():
                    done.add(rid)
    return done


# ── pLDDT extraction from an ESMFold PDB ────────────────────────────────────


def mean_plddt_from_pdb(pdb_str: str) -> float:
    """Mean per-residue pLDDT (0–100) = mean CA B-factor of an ESMFold PDB.

    ``output_to_pdb`` writes the per-residue pLDDT into the B-factor column
    (cols 61–66). Different transformers versions use different scales:
    older ones write 0–100, transformers 5.x writes **0–1**. We normalize to
    0–100 by detecting the scale from the maximum CA B-factor (a real 0–100
    pLDDT is always ≫ 1; a 0–1 pLDDT never exceeds 1). Returns NaN if no CA
    atoms are present.
    """
    vals: list[float] = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                continue
    if not vals:
        return float("nan")
    mean = sum(vals) / len(vals)
    if max(vals) <= 1.0:  # 0–1 scale (transformers 5.x) -> rescale to 0–100
        mean *= 100.0
    return mean


# ── ESMFold runtime (lazy torch/transformers import) ────────────────────────


class FoldResult(NamedTuple):
    """One folded sequence: PDB text plus confidence scalars."""

    pdb: str
    length: int
    plddt_mean: float
    ptm: float


class EsmFold:
    """Loaded ESMFold model + tokenizer, ready to fold sequences.

    Construct once per process (weights are large); reuse across a shard.
    """

    def __init__(self, model, tokenizer, device: str):
        self._model = model
        self._tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(
        cls,
        *,
        device: str | None = None,
        chunk_size: int | None = None,
        half_precision: bool = True,
    ) -> "EsmFold":
        """Load ``facebook/esmfold_v1``.

        ``device`` defaults to ``"cuda"`` when available else ``"cpu"``.
        ``half_precision`` casts the language-model trunk to fp16 (halves
        GPU memory; standard for ESMFold inference). ``chunk_size`` sets the
        trunk attention chunking (lower = less memory, slower); ``None``
        leaves the default (fine for short sequences).
        """
        import torch  # noqa: PLC0415 - lazy: only needed on the GPU path
        from transformers import AutoTokenizer, EsmForProteinFolding  # noqa: PLC0415

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading ESMFold (%s) on %s", ESMFOLD_MODEL_ID, device)
        tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_ID)
        model = EsmForProteinFolding.from_pretrained(ESMFOLD_MODEL_ID)
        model = model.eval().to(device)
        if half_precision and device.startswith("cuda"):
            # Cast only the ESM-2 language model; the folding trunk stays fp32.
            # Guarded: the submodule name is stable across transformers 4.29–5.x,
            # but fall back to fp32 (works, just more GPU memory) rather than
            # crash a whole shard if a future version renames it.
            try:
                model.esm = model.esm.half()
            except AttributeError:
                logger.warning("could not fp16-cast model.esm; running fp32")
        if chunk_size is not None:
            try:
                model.trunk.set_chunk_size(chunk_size)
            except AttributeError:
                logger.warning("could not set trunk chunk size; using default")
        return cls(model, tokenizer, device)

    def fold(self, sequence: str) -> FoldResult:
        """Fold one canonical amino-acid sequence into a :class:`FoldResult`."""
        if not is_canonical(sequence):
            bad = sorted(set(sequence) - _CANONICAL_AA)
            raise ValueError(
                f"sequence has non-canonical residues {bad}; degap/clean first"
            )
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        pdb = self._model.output_to_pdb(outputs)[0]
        ptm = float(outputs["ptm"].item()) if "ptm" in outputs else float("nan")
        return FoldResult(
            pdb=pdb,
            length=len(sequence),
            plddt_mean=mean_plddt_from_pdb(pdb),
            ptm=ptm,
        )


def write_pdb(pdb_str: str, path: Path | str) -> Path:
    """Write an ESMFold PDB string to ``path`` (creating parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_str, encoding="utf-8")
    return path
