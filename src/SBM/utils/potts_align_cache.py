"""TSV I/O for the ``potts_align`` cluster cache (iter-003, docs/POTTS_ALIGN.md).

The couplings-aware gap-placement minimization (:mod:`SBM.energy.potts_align`) is
pure numpy but expensive for high-gap cases, so it runs sharded on a Slurm array
(``scripts/wf/run_potts_align_shard.py``). Each shard flushes one TSV row per
scored ``(query_id, model)`` pair; the gather
(``scripts/wf/run_potts_align_gather.py``) merges them into one
``<run_root>/potts_align/cache/<model>/alignments.tsv`` per model.

This module is the single definition of that TSV schema — the parser/writer the
shard, the gather, and the ``score`` cache-reader (``scripts/score_two_models.py``)
all share. It is a pure parser (no Potts kernel): the energy stored is the global
(or PT) in-frame Potts minimum the cluster computed, and the score branch
recomputes ``potts_energy(frame)`` as a ``<=1e-6`` gauge/handoff canary.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

#: TSV columns. Stable contract shared by the shard, the gather, and the reader.
#: A *shard* TSV holds rows for both models (the ``model`` column distinguishes
#: them); a *gathered* per-model ``alignments.tsv`` holds one model's rows (the
#: ``model`` column is then redundant but kept so the schema is identical).
TSV_COLUMNS: tuple[str, ...] = (
    "query_id",
    "model",
    "n_residues",
    "gaps",
    "energy",
    "engine",
    "is_global_exact",
    "frame",
    "seed",
)
TSV_HEADER: str = "\t".join(TSV_COLUMNS)


@dataclasses.dataclass(frozen=True)
class PottsAlignCacheResult:
    """One scored ``(query_id, model)`` pair from the potts_align cluster cache.

    ``frame`` is the length-``L`` global-minimum frame as an amino-acid string
    (gap ``-``); ``energy`` is its exact in-frame Potts energy (the value
    ``potts_align`` returned, already drift-canaried internally). ``engine`` is
    ``"enumerate"`` (provably global) / ``"pt"`` / ``"sa"`` and
    ``is_global_exact`` flags the enumerated cases. A ``nan`` energy with an
    empty ``frame`` marks an out-of-scope (``N>L``) skip row written by the
    gather — never produced by ``run`` (which only scores in-scope pairs).
    """

    query_id: str
    model: str
    n_residues: int
    gaps: int
    energy: float
    engine: str
    is_global_exact: bool
    frame: str
    seed: int

    @property
    def ok(self) -> bool:
        """True for a real scored row (non-empty frame); False for a skip row."""
        return bool(self.frame)


def _parse_bool(token: str) -> bool:
    return token.strip().lower() in ("1", "true", "t", "yes")


def _parse_int(token: str) -> int:
    token = token.strip()
    return int(token) if token else 0


def _parse_row(line: str) -> PottsAlignCacheResult:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(TSV_COLUMNS):
        raise ValueError(
            f"potts_align TSV row has {len(parts)} fields, expected {len(TSV_COLUMNS)} "
            f"({TSV_HEADER}); offending row: {line!r}"
        )
    query_id, model, n_residues, gaps, energy, engine, is_exact, frame, seed = parts
    e_str = energy.strip().lower()
    e_val = math.nan if e_str in ("", "nan") else float(energy)
    return PottsAlignCacheResult(
        query_id=query_id,
        model=model,
        n_residues=_parse_int(n_residues),
        gaps=_parse_int(gaps),
        energy=e_val,
        engine=engine.strip(),
        is_global_exact=_parse_bool(is_exact),
        frame=frame,
        seed=_parse_int(seed),
    )


def format_row(res: PottsAlignCacheResult) -> str:
    """Inverse of :func:`_parse_row` — one TSV line for ``res`` (round-trips)."""
    energy = "nan" if math.isnan(res.energy) else repr(res.energy)
    return "\t".join(
        (
            res.query_id,
            res.model,
            str(res.n_residues),
            str(res.gaps),
            energy,
            res.engine,
            "true" if res.is_global_exact else "false",
            res.frame,
            str(res.seed),
        )
    )


def write_alignment_cache(path: Path | str, results: list[PottsAlignCacheResult]) -> None:
    """Write a gathered ``alignments.tsv`` (header + one row per result)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [TSV_HEADER] + [format_row(r) for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_rows(tsv_path: Path):
    with open(tsv_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.startswith(TSV_COLUMNS[0] + "\t"):  # header
                continue
            yield _parse_row(line)


def read_potts_align_cache(tsv_path: Path | str) -> dict[str, PottsAlignCacheResult]:
    """Read a *gathered per-model* ``alignments.tsv`` keyed by ``query_id``.

    Tolerates an optional header. Raises on a duplicate ``query_id`` (one row per
    query in a per-model file). Use :func:`read_shard_cache` for a *shard* TSV,
    which mixes models and must be keyed by ``(query_id, model)``.
    """
    tsv_path = Path(tsv_path)
    out: dict[str, PottsAlignCacheResult] = {}
    for res in _iter_rows(tsv_path):
        if res.query_id in out:
            raise ValueError(f"duplicate query_id {res.query_id!r} in potts_align cache {tsv_path}")
        out[res.query_id] = res
    return out


def read_shard_cache(tsv_path: Path | str) -> dict[tuple[str, str], PottsAlignCacheResult]:
    """Read a *shard* TSV keyed by ``(query_id, model)`` (resume + gather merge).

    Raises on a duplicate ``(query_id, model)`` pair within the file.
    """
    tsv_path = Path(tsv_path)
    out: dict[tuple[str, str], PottsAlignCacheResult] = {}
    for res in _iter_rows(tsv_path):
        key = (res.query_id, res.model)
        if key in out:
            raise ValueError(f"duplicate (query_id, model)={key!r} in potts_align shard {tsv_path}")
        out[key] = res
    return out
