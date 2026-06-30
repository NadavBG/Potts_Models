"""Warm-start fixed-point probe analysis (iter-003 Phase-B, §10.x).

Given DCAlign belief propagation **initialised at the native frame** (the cache
produced by ``src/SBM/julia/run_dcalign_warmstart.jl``), this module decides, per
worse-than-native home pair, whether native is a stable fixed point of DCAlign's
objective or whether BP's own dynamics drive it away.

Three frames per home pair, all scored in-frame with the validated
:func:`SBM.energy.potts.potts_energy` (zero-sum gauge):

* **native** — ``record.ints`` (the home-model alignment; the warm-start origin).
* **warm-start** — where BP settled *starting from native*.
* **random-init** — what the production run found (the iter-002 cache); the
  worse-than-native frame whose residual motivated all of this.

Sign convention (matches :mod:`SBM.energy.dcalign_baseline`): ``delta_e =
E_frame − E_native``; ``>0`` ⇒ worse than native.

Verdict logic. The headline per-sequence signal is ``delta_e_warm`` (the energy BP
reached *from* the native start):

* ``delta_e_warm ≤ equal_tol`` ⇒ BP **stayed** at a native-quality frame: native is
  a stable fixed point the production random-init run simply missed → **case A**
  (a search / initialisation problem → annealing or native-biased init is the lever).
* ``delta_e_warm > equal_tol`` ⇒ BP **drifted** to a worse frame even starting at
  native: native is not a fixed point of DCAlign's objective → **case B** (the
  objective genuinely prefers the other frame → search-tuning is futile). The
  ``col_agree_rand`` / ``delta_e_rand`` comparison then says whether it drifted to
  *the* production frame (same basin) or a third one.

Controls (pairs the production run already aligned to native, ``delta_e_rand≈0``)
are a probe sanity check: warm-started at native they must **stay**; a control that
drifts (``control_drift``) means the probe itself is suspect, not the science.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from SBM.utils.dcalign_score import DCAlignResult

from .dcalign_baseline import DEFAULT_EQUAL_TOL, column_agreement
from .dcalign_residual import group_kind
from .datasets import QueryRecord
from .encoding import GAP, seq_to_ints
from .model import PottsModel
from .potts import potts_energy

#: Column-agreement at/above which two length-L frames are "the same frame".
#: Controls sit at exactly 1.0; this band absorbs a stray column without calling
#: two materially-different threadings identical.
DEFAULT_AGREE_THRESH = 0.99

#: Per-sequence verdict labels.
STAYED_NATIVE = "stayed_native"          # case A: BP kept a native-quality frame
FLOWED_TO_RAND = "flowed_to_rand"        # case B: drifted to the production frame
FLOWED_OTHER = "flowed_other"            # drifted worse, but to a third frame
CONTROL_OK = "control_ok"                # control stayed (probe sane)
CONTROL_DRIFT = "control_drift"          # control drifted (probe suspect!)
FAILED = "failed"                        # warm-start produced no frame


def classify_warmstart(
    *, role: str, delta_e_warm: float, delta_e_rand: float,
    col_agree_native: float, col_agree_rand: float,
    equal_tol: float = DEFAULT_EQUAL_TOL, agree_thresh: float = DEFAULT_AGREE_THRESH,
) -> str:
    """Label one warm-start outcome (see module docstring for the verdict logic)."""
    stayed = (delta_e_warm <= equal_tol) or (col_agree_native >= agree_thresh)
    if role == "control":
        return CONTROL_OK if stayed else CONTROL_DRIFT
    if stayed:
        return STAYED_NATIVE
    same_basin = (col_agree_rand >= agree_thresh) or (abs(delta_e_warm - delta_e_rand) <= equal_tol)
    return FLOWED_TO_RAND if same_basin else FLOWED_OTHER


@dataclass(frozen=True)
class WarmstartRow:
    """One home pair: native vs warm-start vs random-init energies + the verdict."""

    sequence_id: str
    model: str
    group: str
    kind: str          # natural | synthetic
    role: str          # recover | control
    e_native: float
    e_warmstart: float
    e_randominit: float
    delta_e_warm: float       # E_warmstart − E_native (≤ tol ⇒ stayed at native)
    delta_e_rand: float       # E_randominit − E_native (the original residual)
    col_agree_native: float   # warm-start frame vs native frame
    col_agree_rand: float     # warm-start frame vs random-init frame
    warm_converged: bool
    warm_used_decimation: bool
    warm_n_iter: int
    label: str
    ok: bool

    def as_dict(self) -> dict:
        return asdict(self)


def analyze_warmstart_record(
    record: QueryRecord, model: PottsModel, role: str,
    warm: DCAlignResult, rand: DCAlignResult | None,
    *, equal_tol: float = DEFAULT_EQUAL_TOL, agree_thresh: float = DEFAULT_AGREE_THRESH,
) -> WarmstartRow:
    """Build one :class:`WarmstartRow` (record in ``model``'s native frame).

    A failed warm-start (empty frame) is kept with ``ok=False`` / ``label=failed``
    (never dropped). ``rand`` may be ``None`` (no production cache for this id):
    then ``delta_e_rand``/``col_agree_rand`` are NaN and a worse warm-start can
    only be ``flowed_other``.
    """
    native = np.asarray(record.ints, dtype=np.int64)
    e_native = potts_energy(native, model)
    kind = group_kind(record.group)

    def _delta_rand() -> float:
        if rand is None or not rand.aligned_frame:
            return math.nan
        return potts_energy(seq_to_ints(rand.aligned_frame), model) - e_native

    delta_e_rand = _delta_rand()
    e_rand = (e_native + delta_e_rand) if not math.isnan(delta_e_rand) else math.nan

    if not warm.aligned_frame:
        return WarmstartRow(
            sequence_id=record.id, model=model.name, group=record.group, kind=kind, role=role,
            e_native=e_native, e_warmstart=math.nan, e_randominit=e_rand,
            delta_e_warm=math.nan, delta_e_rand=delta_e_rand,
            col_agree_native=math.nan, col_agree_rand=math.nan,
            warm_converged=warm.converged, warm_used_decimation=warm.used_decimation,
            warm_n_iter=warm.n_iter, label=FAILED, ok=False,
        )

    warm_frame = seq_to_ints(warm.aligned_frame)
    if warm_frame.size != model.L:
        raise ValueError(
            f"warm-start frame length {warm_frame.size} != L={model.L} for {record.id!r}")
    e_warm = potts_energy(warm_frame, model)
    delta_e_warm = e_warm - e_native
    col_native = column_agreement(native, warm_frame)
    col_rand = (
        math.nan if rand is None or not rand.aligned_frame
        else column_agreement(seq_to_ints(rand.aligned_frame), warm_frame)
    )
    label = classify_warmstart(
        role=role, delta_e_warm=delta_e_warm,
        delta_e_rand=0.0 if math.isnan(delta_e_rand) else delta_e_rand,
        col_agree_native=col_native, col_agree_rand=0.0 if math.isnan(col_rand) else col_rand,
        equal_tol=equal_tol, agree_thresh=agree_thresh,
    )
    return WarmstartRow(
        sequence_id=record.id, model=model.name, group=record.group, kind=kind, role=role,
        e_native=e_native, e_warmstart=e_warm, e_randominit=e_rand,
        delta_e_warm=delta_e_warm, delta_e_rand=delta_e_rand,
        col_agree_native=col_native, col_agree_rand=col_rand,
        warm_converged=warm.converged, warm_used_decimation=warm.used_decimation,
        warm_n_iter=warm.n_iter, label=label, ok=True,
    )


def _subset_stats(rows: list[WarmstartRow], equal_tol: float) -> dict:
    ok = [r for r in rows if r.ok]
    n_stayed = sum(1 for r in ok if r.delta_e_warm <= equal_tol)
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_failed": sum(1 for r in rows if not r.ok),
        "n_stayed_native": n_stayed,
        "frac_stayed_native": (n_stayed / len(ok)) if ok else 0.0,
        "n_flowed_to_rand": sum(1 for r in ok if r.label == FLOWED_TO_RAND),
        "n_flowed_other": sum(1 for r in ok if r.label == FLOWED_OTHER),
        "median_delta_e_warm": (float(np.median([r.delta_e_warm for r in ok])) if ok else None),
    }


def summarize_warmstart(
    rows: list[WarmstartRow], *, equal_tol: float = DEFAULT_EQUAL_TOL,
    init_kind: str = "native",
) -> dict:
    """Per-role / per-kind warm-start tallies + the case-A/B verdict.

    The recover (worse-than-native) rows carry the science; controls are a probe
    sanity check (``n_control_drift`` must be 0). The verdict reads
    ``frac_stayed_native`` over the recover rows: high ⇒ case A (anneal); ≈0 ⇒
    case B (native unstable; stop tuning). ``init_kind`` (``"native"`` for the
    fixed-point probe, ``"map"`` for the fields-MAP-init production test) only
    adjusts the verdict wording.
    """
    recover = [r for r in rows if r.role == "recover"]
    control = [r for r in rows if r.role == "control"]
    kinds = sorted({r.kind for r in recover})
    n_control_drift = sum(1 for r in control if r.label == CONTROL_DRIFT)
    return {
        "equal_tol": equal_tol,
        "init_kind": init_kind,
        "delta_e_convention": "delta_e_* = E_* - E_native; >0 => worse than native",
        "recover": {
            "overall": _subset_stats(recover, equal_tol),
            "by_kind": {k: _subset_stats([r for r in recover if r.kind == k], equal_tol)
                        for k in kinds},
        },
        "control": {**_subset_stats(control, equal_tol), "n_control_drift": n_control_drift},
        "verdict": build_warmstart_verdict(recover, control, equal_tol, init_kind=init_kind),
    }


def build_warmstart_verdict(
    recover: list[WarmstartRow], control: list[WarmstartRow], equal_tol: float,
    *, init_kind: str = "native",
) -> str:
    """One-paragraph case-A/B verdict from the recover rows + control sanity.

    ``init_kind="native"`` is the fixed-point probe (did native stay?);
    ``init_kind="map"`` is the production test (did BP reach native quality from
    the fields-MAP init?). Only the wording differs.
    """
    ok = [r for r in recover if r.ok]
    if not ok:
        return "No successful worse-than-native warm-starts; nothing to conclude."
    n = len(ok)
    n_stayed = sum(1 for r in ok if r.delta_e_warm <= equal_tol)
    n_rand = sum(1 for r in ok if r.label == FLOWED_TO_RAND)
    n_other = sum(1 for r in ok if r.label == FLOWED_OTHER)
    frac = n_stayed / n
    drift = sum(1 for r in control if r.label == CONTROL_DRIFT)
    sane = f"Controls: {len(control) - drift}/{len(control)} ok" + (
        "" if drift == 0 else f" — WARNING {drift} drifted (init suspect!)")
    is_map = init_kind == "map"
    started = "initialized at the fields-MAP frame" if is_map else "warm-started at native"
    verb = "REACHED" if is_map else "STAYED at"
    if frac >= 0.5:
        lever = (
            "fields-MAP init + couplings-aware BP recovers the min the random-init production runs "
            "miss — a production-usable lever (no ground truth needed)." if is_map else
            "native is a reachable fixed point the random-init production runs missed. "
            "Lever: annealing / native-biased init.")
        verdict = (
            f"CASE A (search/init problem). {n_stayed}/{n} worse pairs {verb} a native-quality "
            f"frame when BP was {started} — {lever} ")
    elif n_stayed == 0:
        why = ("the fields-MAP init is not close enough to native's basin; try a slower anneal."
               if is_map else
               "native is not a fixed point of DCAlign's objective — search-tuning cannot recover it.")
        verdict = (
            f"CASE B. 0/{n} worse pairs {verb} native: BP drifted to a worse frame even when "
            f"{started} ({n_rand} back to the production frame, {n_other} to a third frame). {why} ")
    else:
        verdict = (
            f"MIXED. {n_stayed}/{n} worse pairs {verb} a native-quality frame when {started} "
            f"(case A), {n - n_stayed} drifted ({n_rand} to the production frame, {n_other} "
            f"elsewhere). Lever is partial. ")
    return verdict + sane + "."
