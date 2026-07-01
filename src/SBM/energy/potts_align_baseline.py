"""Baseline diagnostic: does ``potts_align`` recover the native ground state?

The combine ``potts_align`` run re-aligns each raw query to a model with the
couplings-aware gap-placement minimizer (:mod:`SBM.energy.potts_align`,
docs/POTTS_ALIGN.md) and caches the frame plus its exact in-frame Potts energy
(:mod:`SBM.utils.potts_align_cache`). For a **home pair** — a native sitting in
its own model's length-``L`` frame — the native gap placement is itself one of
the frames the minimizer searches, so the global minimum it returns can never be
*higher* than the native in-frame energy. This module turns "did we recover the
ground state?" into a reproducible, per-sequence quantity.

Sign convention: ``delta_e = E_potts_align − E_inframe``. Lower energy is better,
so each home pair falls into exactly one bucket:

* ``delta_e ≈ 0`` **and** column agreement 1 ⇒ the native frame *is* the global
  Potts minimum and potts_align recovered it — the ground state, recovered.
* ``delta_e ≈ 0`` **and** column agreement < 1 ⇒ a degenerate frame of equal
  energy (native still sits at the ground-state energy, via a different threading).
* ``delta_e < 0`` ⇒ potts_align found a strictly lower-energy frame; the native
  placement is **not** the ground state (the aligner beats the curated MSA frame).
* ``delta_e > 0`` ⇒ potts_align returned a frame **worse** than native. Impossible
  for the ``enumerate`` engine (provably global); for the ``pt``/``sa`` engines it
  flags a search that failed to reach even the native frame — surfaced, not hidden.

Only home pairs are comparable: a cross-family query has no native ``L``-frame
under the other model, so there is nothing to compare against (the caller skips
those). All energies reuse the validated :func:`SBM.energy.potts.potts_energy`
(zero-sum gauge via :func:`SBM.energy.model.load_model`); no new energy math.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from SBM.utils.potts_align_cache import PottsAlignCacheResult

from .datasets import QueryRecord
from .encoding import GAP, seq_to_ints
from .model import PottsModel
from .potts import potts_energies

#: Default |ΔE| (a.u.) below which the native is judged to sit *at* the ground
#: state rather than strictly above/below it. The in-frame recompute vs the
#: cached energy agrees to ≲1e-6 (the cache canary); this larger band absorbs
#: energetically-equivalent re-threadings so ``n_improved`` / ``n_worse`` count
#: only material differences.
DEFAULT_EQUAL_TOL = 1.0


def column_agreement(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fraction of the ``L`` positions where two length-``L`` frames agree.

    Counts both residue–residue and gap–gap matches (per-position state
    equality) — the "col agreement" metric between the native frame and the
    aligner's frame.
    """
    a = np.asarray(frame_a).ravel()
    b = np.asarray(frame_b).ravel()
    if a.size != b.size:
        raise ValueError(
            f"column_agreement needs equal-length frames; got {a.size} and {b.size}"
        )
    if a.size == 0:
        raise ValueError("column_agreement needs non-empty frames")
    return float(np.mean(a == b))


@dataclass(frozen=True)
class PottsAlignBaselineRow:
    """One home-pair comparison: native in-frame energy vs potts_align's minimum."""

    sequence_id: str
    group: str
    model: str
    n_residues: int  # ungapped residue count (what potts_align re-aligned)
    e_inframe: float  # native frame energy (the curated/synthetic MSA alignment)
    e_potts: float  # our in-frame recompute on potts_align's cached global-min frame
    delta_e: float  # e_potts − e_inframe (<0 ⇒ aligner beats native; >0 ⇒ search failure)
    col_agreement: float  # fraction of the L columns shared by the two frames
    cache_energy: float  # potts_align's own reported energy (NaN on a skip row)
    cache_abs_diff: float  # |e_potts − cache_energy| — the gauge/handoff canary
    engine: str  # "enumerate" (provably global) / "pt" / "sa"
    is_global_exact: bool  # True ⇒ the returned frame is the provable global minimum
    ok: bool  # False ⇒ potts_align produced no frame (skip / out-of-scope row)

    def as_dict(self) -> dict:
        return asdict(self)


def _build_row(
    record: QueryRecord,
    model: PottsModel,
    res: PottsAlignCacheResult,
    *,
    e_inframe: float,
    e_potts: float,
    frame: np.ndarray | None,
) -> PottsAlignBaselineRow:
    """Assemble one row from *precomputed* energies (shared by the per-record and
    batched paths so the row logic lives in exactly one place).

    ``frame`` is the cached frame as ints (length ``L``) or ``None`` for a skip
    row (``res.ok`` is False); ``e_potts`` is then NaN.
    """
    n_residues = int(np.count_nonzero(record.ints != GAP))
    if not res.ok:  # potts_align produced no frame for this pair (out of scope / skipped)
        return PottsAlignBaselineRow(
            sequence_id=record.id, group=record.group, model=model.name,
            n_residues=n_residues, e_inframe=e_inframe, e_potts=math.nan,
            delta_e=math.nan, col_agreement=math.nan, cache_energy=res.energy,
            cache_abs_diff=math.nan, engine=res.engine,
            is_global_exact=res.is_global_exact, ok=False,
        )
    cache_abs_diff = (
        math.nan if math.isnan(res.energy) else abs(e_potts - res.energy)
    )
    return PottsAlignBaselineRow(
        sequence_id=record.id, group=record.group, model=model.name,
        n_residues=n_residues, e_inframe=e_inframe, e_potts=e_potts,
        delta_e=e_potts - e_inframe,
        col_agreement=column_agreement(record.ints, frame),
        cache_energy=res.energy, cache_abs_diff=cache_abs_diff,
        engine=res.engine, is_global_exact=res.is_global_exact, ok=True,
    )


def rows_for_home_pairs(
    model: PottsModel, pairs: list[tuple[QueryRecord, PottsAlignCacheResult]]
) -> list[PottsAlignBaselineRow]:
    """Rows for a batch of home pairs under one ``model``, energies vectorized.

    ``pairs`` are ``(record, cache_result)`` tuples already known to be home pairs
    for ``model`` (record in ``model``'s frame, present in the cache). Native
    in-frame energies are one batched :func:`potts_energies` call and the cached
    global-min frames another, so this stays fast on the uncapped ~10^4-sequence
    query sets. A skip row (empty frame) keeps ``ok=False`` with a NaN potts
    energy — never dropped. Frame-length mismatches raise (a corrupt cache).
    """
    if not pairs:
        return []
    native = np.stack([rec.ints for rec, _ in pairs]).astype(np.int64)
    e_inframe = potts_energies(native, model)
    e_potts = np.full(len(pairs), math.nan)
    frames: list[np.ndarray | None] = [None] * len(pairs)
    ok_idx, ok_frames = [], []
    for i, (rec, res) in enumerate(pairs):
        if not res.ok:
            continue
        frame = seq_to_ints(res.frame)
        if frame.size != model.L:
            raise ValueError(
                f"cached potts_align frame length {frame.size} != model L={model.L} "
                f"for sequence {rec.id!r} under {model.name!r}"
            )
        frames[i] = frame
        ok_idx.append(i)
        ok_frames.append(frame)
    if ok_frames:
        e_potts_ok = potts_energies(np.stack(ok_frames).astype(np.int64), model)
        for k, i in enumerate(ok_idx):
            e_potts[i] = e_potts_ok[k]
    return [
        _build_row(rec, model, res, e_inframe=float(e_inframe[i]),
                   e_potts=float(e_potts[i]), frame=frames[i])
        for i, (rec, res) in enumerate(pairs)
    ]


def compare_record(
    record: QueryRecord, model: PottsModel, res: PottsAlignCacheResult
) -> PottsAlignBaselineRow:
    """Build one :class:`PottsAlignBaselineRow` for a home pair (record in ``model``'s frame).

    ``record.ints`` must already be in ``model``'s frame (length ``L``); the
    caller selects home pairs. A skip row (empty frame, e.g. an ``N>L`` pair the
    cluster did not score) is recorded with ``ok=False`` and NaN energies — never
    silently dropped — so the summary can surface the count. Thin wrapper over
    :func:`rows_for_home_pairs` (the CLI batches whole models at once).
    """
    return rows_for_home_pairs(model, [(record, res)])[0]


def _delta_stats(rows: list[PottsAlignBaselineRow], equal_tol: float) -> dict:
    """ΔE distribution + ground-state / improved / worse tallies over OK rows.

    The three ΔE buckets partition the OK rows:
    ``n_at_ground`` (|ΔE| ≤ tol), ``n_improved`` (ΔE < −tol), ``n_worse``
    (ΔE > tol). ``n_recovered_exact_frame`` is the subset of ``n_at_ground`` that
    also matched the native threading (column agreement == 1).
    """
    ok = [r for r in rows if r.ok]
    n_failed = sum(1 for r in rows if not r.ok)
    if not ok:
        return {"n": len(rows), "n_ok": 0, "n_failed": n_failed}
    delta = np.array([r.delta_e for r in ok], dtype=float)
    agree = np.array([r.col_agreement for r in ok], dtype=float)
    diff = np.array([r.cache_abs_diff for r in ok], dtype=float)
    at_ground = np.abs(delta) <= equal_tol
    n_at_ground = int(np.sum(at_ground))
    n_recovered_exact = int(np.sum(at_ground & np.isclose(agree, 1.0)))
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_failed": n_failed,
        "delta_e": {
            "min": float(delta.min()), "median": float(np.median(delta)),
            "mean": float(delta.mean()), "max": float(delta.max()),
        },
        "n_at_ground": n_at_ground,
        "frac_at_ground": n_at_ground / len(ok),
        "n_recovered_exact_frame": n_recovered_exact,
        "n_improved": int(np.sum(delta < -equal_tol)),
        "n_worse": int(np.sum(delta > equal_tol)),
        "col_agreement_median": float(np.median(agree)),
        "n_exact_engine": sum(1 for r in ok if r.is_global_exact),
        "cache_max_abs_diff": float(np.nanmax(diff)) if diff.size else None,
    }


def summarize(
    rows: list[PottsAlignBaselineRow], *, equal_tol: float = DEFAULT_EQUAL_TOL
) -> dict:
    """Overall + per-model + per-group ΔE summary (the ground-state-recovery readout).

    ``n_at_ground`` / ``frac_at_ground`` count rows with ``|delta_e| ≤ equal_tol``
    — natives already sitting at their model's global Potts minimum. ``n_worse``
    counts ``delta_e > equal_tol``: for a couplings-aware aligner that searches the
    native frame this should be **zero** (any positive count is a PT/SA search
    failure, not a modelling result). ``cache_max_abs_diff`` is the standing
    gauge/handoff canary (expected ≲ 1e-6).
    """
    by_model: dict[str, list[PottsAlignBaselineRow]] = {}
    by_group: dict[str, list[PottsAlignBaselineRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
        by_group.setdefault(f"{r.model} | {r.group}", []).append(r)
    return {
        "equal_tol": equal_tol,
        "delta_e_convention": (
            "delta_e = E_potts_align - E_inframe; <0 => aligner beats native frame, "
            ">0 => potts_align worse than native (search failure)"
        ),
        "overall": _delta_stats(rows, equal_tol),
        "by_model": {m: _delta_stats(rs, equal_tol) for m, rs in sorted(by_model.items())},
        "by_group": {g: _delta_stats(rs, equal_tol) for g, rs in sorted(by_group.items())},
    }
