"""Baseline diagnostic: DCAlign's best-attempt energy vs the native in-frame energy.

The combine ``dcalign`` run aligns each raw query to a model with the
couplings-aware DCAlign aligner and caches the result (see
:mod:`SBM.utils.dcalign_score`). A *couplings-aware* aligner should never score
an in-frame native **worse** than the trivial native frame it already sits in —
yet with the flat insertion prior it frequently does (spec §10.8 Blocker 1). This
module turns that comparison into a reproducible, per-sequence quantity, so the
prior-tuning experiment (§10.9 phase 2) has an explicit baseline to beat.

Only **home pairs** are comparable: a query is in *its own model's* frame (length
``L``), so its native in-frame energy is well-defined. A query under the *other*
model has a different length and no in-frame reference — there is nothing to
compare DCAlign against there, so those pairs are out of scope (the caller skips
them). All energies reuse the validated :func:`SBM.energy.potts.potts_energy`
(zero-sum gauge via :func:`SBM.energy.model.load_model`); no new energy math.

Sign convention: ``delta_e = E_dcalign − E_inframe``. Lower energy is better, so
``delta_e > 0`` ⇒ DCAlign did **worse** than the native frame (the Blocker-1
pathology), ``delta_e ≈ 0`` ⇒ it recovered the native frame, ``delta_e < 0`` ⇒ it
found a lower-energy frame (couplings-awareness helping, as intended).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from SBM.utils.dcalign_score import DCAlignResult

from .datasets import QueryRecord
from .encoding import GAP, seq_to_ints
from .model import PottsModel
from .potts import potts_energy

#: Default |ΔE| (a.u.) below which DCAlign is judged to have *recovered* the
#: native frame rather than done better/worse. The fp recompute noise is ~1e-12
#: (the cache canary); this larger band absorbs energetically-equivalent
#: re-threadings so ``n_worse``/``n_better`` count only material differences.
DEFAULT_EQUAL_TOL = 1.0


@dataclass(frozen=True)
class BaselineRow:
    """One home-pair comparison: native in-frame energy vs DCAlign's attempt."""

    sequence_id: str
    group: str
    model: str
    n_residues: int  # ungapped residue count (what DCAlign re-aligned)
    e_inframe: float  # native frame energy (the curated/synthetic MSA alignment)
    e_dcalign: float  # our in-frame recompute on DCAlign's cached frame
    delta_e: float  # e_dcalign − e_inframe (>0 ⇒ DCAlign worse)
    col_agreement: float  # fraction of the L columns shared by the two frames
    cache_energy: float  # DCAlign's own reported energy (NaN on failure)
    cache_abs_diff: float  # |e_dcalign − cache_energy| — the gauge/handoff canary
    converged: bool
    used_decimation: bool
    n_iter: int
    ok: bool  # False ⇒ DCAlign produced no frame (alignment failed)

    def as_dict(self) -> dict:
        return asdict(self)


def column_agreement(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Fraction of the ``L`` positions where two length-``L`` frames agree.

    Counts both residue–residue and gap–gap matches (per-position state
    equality). This is the "col agreement" metric reported in spec §10.8.
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


def compare_record(
    record: QueryRecord, model: PottsModel, dca: DCAlignResult
) -> BaselineRow:
    """Build one :class:`BaselineRow` for a home-pair (record in ``model``'s frame).

    ``record.ints`` must already be in ``model``'s frame (length ``L``); the
    caller selects home pairs. A DCAlign result with an **empty** frame marks an
    alignment failure: it is recorded with ``ok=False`` and NaN energies (never
    silently dropped), so the summary surfaces the count.
    """
    n_residues = int(np.count_nonzero(record.ints != GAP))
    if not dca.aligned_frame:  # DCAlign failed for this id
        return BaselineRow(
            sequence_id=record.id, group=record.group, model=model.name,
            n_residues=n_residues, e_inframe=potts_energy(record.ints, model),
            e_dcalign=math.nan, delta_e=math.nan, col_agreement=math.nan,
            cache_energy=dca.dcalign_energy, cache_abs_diff=math.nan,
            converged=dca.converged, used_decimation=dca.used_decimation,
            n_iter=dca.n_iter, ok=False,
        )
    e_inframe = potts_energy(record.ints, model)
    frame = seq_to_ints(dca.aligned_frame)
    if frame.size != model.L:
        raise ValueError(
            f"DCAlign frame length {frame.size} != model L={model.L} for "
            f"sequence {record.id!r} under {model.name!r}"
        )
    e_dcalign = potts_energy(frame, model)
    cache_abs_diff = (
        math.nan if math.isnan(dca.dcalign_energy) else abs(e_dcalign - dca.dcalign_energy)
    )
    return BaselineRow(
        sequence_id=record.id, group=record.group, model=model.name,
        n_residues=n_residues, e_inframe=e_inframe, e_dcalign=e_dcalign,
        delta_e=e_dcalign - e_inframe, col_agreement=column_agreement(record.ints, frame),
        cache_energy=dca.dcalign_energy, cache_abs_diff=cache_abs_diff,
        converged=dca.converged, used_decimation=dca.used_decimation,
        n_iter=dca.n_iter, ok=True,
    )


def _delta_stats(rows: list[BaselineRow], equal_tol: float) -> dict:
    """ΔE distribution + worse/better/equal tallies over the OK rows in ``rows``."""
    ok = [r for r in rows if r.ok]
    n_failed = sum(1 for r in rows if not r.ok)
    if not ok:
        return {"n": len(rows), "n_ok": 0, "n_failed": n_failed}
    delta = np.array([r.delta_e for r in ok], dtype=float)
    agree = np.array([r.col_agreement for r in ok], dtype=float)
    diff = np.array([r.cache_abs_diff for r in ok], dtype=float)
    n_worse = int(np.sum(delta > equal_tol))
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_failed": n_failed,
        "delta_e": {
            "min": float(delta.min()), "median": float(np.median(delta)),
            "mean": float(delta.mean()), "max": float(delta.max()),
        },
        "n_worse": n_worse,
        "frac_worse": n_worse / len(ok),
        "n_better": int(np.sum(delta < -equal_tol)),
        "n_near_equal": int(np.sum(np.abs(delta) <= equal_tol)),
        "col_agreement_median": float(np.median(agree)),
        "n_decimation": sum(1 for r in ok if r.used_decimation),
        "cache_max_abs_diff": float(np.nanmax(diff)) if diff.size else None,
    }


def summarize(rows: list[BaselineRow], *, equal_tol: float = DEFAULT_EQUAL_TOL) -> dict:
    """Overall + per-model + per-group ΔE summary (the baseline readout).

    ``n_worse`` / ``frac_worse`` count rows with ``delta_e > equal_tol`` — the
    Blocker-1 metric (a native scored worse than its own frame by more than
    ``equal_tol`` a.u.). ``cache_max_abs_diff`` is the standing gauge/handoff
    canary (spec §10.11; expected ≲ 1e-12).
    """
    by_model: dict[str, list[BaselineRow]] = {}
    by_group: dict[str, list[BaselineRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
        by_group.setdefault(f"{r.model} | {r.group}", []).append(r)
    return {
        "equal_tol": equal_tol,
        "delta_e_convention": "delta_e = E_dcalign - E_inframe; >0 => DCAlign worse than native frame",
        "overall": _delta_stats(rows, equal_tol),
        "by_model": {m: _delta_stats(rs, equal_tol) for m, rs in sorted(by_model.items())},
        "by_group": {g: _delta_stats(rs, equal_tol) for g, rs in sorted(by_group.items())},
    }


# ── DCAlign convergence report (orthogonal to the in-frame comparison) ─────
#
# DCAlign reports per-sequence convergence (`converged`) and whether it fell back
# to the decimation/nucleation path (`used_decimation`). This report counts those
# over *all* cached alignments — both a sequence's home model and the cross-family
# model — because cross-family frames are where convergence is most at risk and
# there is no in-frame reference there (so the baseline above does not cover them).


def convergence_by_group(
    caches: dict[str, dict[str, DCAlignResult]],
    groups: dict[str, dict],
) -> list[dict]:
    """Tidy per-(model, group) convergence counts over every cached alignment.

    ``caches`` maps model name → ``{seq_id: DCAlignResult}`` (both models);
    ``groups`` is the ``groups.json`` map (``seq_id`` → ``{"group", ...}``). A
    sequence absent from ``groups`` is bucketed as ``"(unknown)"``. One row per
    (model, group): ``n``, ``n_converged``, ``n_not_converged``, ``n_decimation``,
    ``n_failed`` (empty frame), and ``frac_not_converged``.
    """
    buckets: dict[tuple[str, str], list[DCAlignResult]] = {}
    for model_name, by_id in caches.items():
        for seq_id, res in by_id.items():
            group = groups.get(seq_id, {}).get("group", "(unknown)")
            buckets.setdefault((model_name, group), []).append(res)
    rows: list[dict] = []
    for (model_name, group), results in sorted(buckets.items()):
        n = len(results)
        n_not_conv = sum(1 for r in results if not r.converged)
        rows.append({
            "model": model_name,
            "group": group,
            "n": n,
            "n_converged": sum(1 for r in results if r.converged),
            "n_not_converged": n_not_conv,
            "n_decimation": sum(1 for r in results if r.used_decimation),
            "n_failed": sum(1 for r in results if not r.ok),
            "frac_not_converged": (n_not_conv / n) if n else 0.0,
        })
    return rows


def summarize_convergence(rows: list[dict]) -> dict:
    """Roll the tidy per-(model, group) convergence rows up to overall + per-model."""
    def _roll(subset: list[dict]) -> dict:
        n = sum(r["n"] for r in subset)
        n_not_conv = sum(r["n_not_converged"] for r in subset)
        return {
            "n": n,
            "n_converged": sum(r["n_converged"] for r in subset),
            "n_not_converged": n_not_conv,
            "n_decimation": sum(r["n_decimation"] for r in subset),
            "n_failed": sum(r["n_failed"] for r in subset),
            "frac_not_converged": (n_not_conv / n) if n else 0.0,
        }

    models = sorted({r["model"] for r in rows})
    return {
        "overall": _roll(rows),
        "by_model": {m: _roll([r for r in rows if r["model"] == m]) for m in models},
    }
