"""Assemble and read the query set for two-model scoring.

The combine pipeline's first concrete run scores each family's *natural* and
*synthetic* sequences under both models (spec acceptance test 4). This module
turns model run directories into a tidy list of :class:`QueryRecord`s and
(de)serializes them as a mixed-length FASTA + a ``groups.json`` sidecar.

Each record keeps the sequence in *its origin model's frame* (length ``L`` of
the family it came from, gaps included). Scoring under the origin model is then
the exact in-frame energy; scoring under the *other* model strips the gaps and
re-aligns the raw residues to that model's frame (the latent-alignment path).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio import SeqIO

from .encoding import ints_to_seq, seq_to_ints

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryRecord:
    """One query sequence, in its origin model's frame (gaps allowed)."""

    id: str
    group: str  # e.g. "CM/natural", "PPIC/synthetic-T1"
    origin_model: str  # "CM" / "PPIC" / "" if external (no native frame)
    ints: np.ndarray  # integer sequence in the origin frame


def _temp_token(path: Path) -> str:
    """``align_T0.75.npy`` -> ``0.75`` (the sampling temperature in the name)."""
    stem = path.stem  # align_T0.75
    return stem.split("align_T", 1)[1] if "align_T" in stem else stem


def _subsample(rows: np.ndarray, cap: int, rng: np.random.Generator, label: str) -> np.ndarray:
    """Seeded subsample of ``rows`` to at most ``cap`` (0 = no cap). Logs drops."""
    n = rows.shape[0]
    if cap and n > cap:
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        log.info("group %s: capped %d -> %d sequences (seeded subsample)", label, n, cap)
        return rows[idx]
    return rows


def assemble_query_records(
    model_entries: list[dict],
    *,
    include: tuple[str, ...] = ("natural", "synthetic"),
    cap_per_group: int = 0,
    seed: int = 0,
) -> list[QueryRecord]:
    """Build the query set from each model's run directory.

    ``model_entries`` is a list of ``{"name", "run_dir"}`` dicts. ``natural``
    pulls ``<run_dir>/inputs/msa.npy``; ``synthetic`` pulls every
    ``<run_dir>/synthetic/align_T*.npy``. Capping is per group (seeded), and the
    drop is logged — never silent.
    """
    rng = np.random.default_rng(seed)
    records: list[QueryRecord] = []
    for entry in model_entries:
        name = entry["name"]
        run_dir = Path(entry["run_dir"])
        if "natural" in include:
            msa_path = run_dir / "inputs" / "msa.npy"
            if not msa_path.is_file():
                raise FileNotFoundError(f"no encoded MSA for model {name!r} at {msa_path}")
            rows = _subsample(np.load(msa_path), cap_per_group, rng, f"{name}/natural")
            for i, row in enumerate(rows):
                records.append(QueryRecord(f"{name}|natural|{i}", f"{name}/natural", name, row))
        if "synthetic" in include:
            for align_path in sorted((run_dir / "synthetic").glob("align_T*.npy")):
                temp = _temp_token(align_path)
                group = f"{name}/synthetic-T{temp}"
                rows = _subsample(np.load(align_path), cap_per_group, rng, group)
                for i, row in enumerate(rows):
                    records.append(QueryRecord(f"{name}|synthetic-T{temp}|{i}", group, name, row))
    return records


def write_query_fasta(records: list[QueryRecord], fasta_path: Path, groups_path: Path) -> None:
    """Write records as a (mixed-length) FASTA + a ``groups.json`` sidecar."""
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f">{r.id}\n{ints_to_seq(r.ints)}" for r in records]
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = {
        r.id: {"group": r.group, "origin_model": r.origin_model, "origin_L": int(r.ints.size)}
        for r in records
    }
    groups_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")


def read_query_fasta(fasta_path: Path, groups_path: Path | None = None) -> list[QueryRecord]:
    """Read a (mixed-length) query FASTA, optionally with a ``groups.json``.

    Without ``groups.json`` the records have no origin frame (``origin_model=""``,
    ``group="query"``) — every model scores them by latent alignment.
    """
    groups: dict = {}
    if groups_path is not None and Path(groups_path).is_file():
        groups = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    records: list[QueryRecord] = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        ints = seq_to_ints(str(rec.seq))  # raises loudly on non-canonical residues
        meta = groups.get(rec.id, {})
        records.append(
            QueryRecord(
                id=rec.id,
                group=meta.get("group", "query"),
                origin_model=meta.get("origin_model", ""),
                ints=ints,
            )
        )
    return records
