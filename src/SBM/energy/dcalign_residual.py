"""iter-003 Phase-0 diagnostic: anatomy of the worse-than-native residual.

iter-002's informed ``deltan`` prior fixed most of the Blocker-1 pathology, but a
tail of home-pair sequences still scores worse in DCAlign's chosen frame than in
the trivially-available native frame (``delta_e > equal_tol``; see
:mod:`SBM.energy.dcalign_baseline`). Before paying for the §10.13 escalation
(literal-insertion seed, a cross-project data pull), this module answers two
questions from the *already-cached* iter-002 alignments — no new DCAlign run:

1. **Where does the residual live?** Worse-than-native counts split by
   natural vs synthetic. Synthetic queries are MCMC samples in the L-frame with
   no insertions by construction, so an insertion prior cannot move them; only
   the natural tail is even in scope for the escalation.
2. **Is the natural tail prior-shaped?** For each failing sequence, classify how
   DCAlign's frame disagrees with the native frame:

   - ``terminal`` — disagreement only at the N/C-termini, agreeing core intact.
   - ``register_shift`` — an internal block displaced by a constant column
     offset (the signature of a mis-set spacing prior — what a better Λ fixes).
   - ``gap_redistribution`` — scattered internal gap differences, no clean shift.

3. **Which lever could move it?** (the iter-003 decision, spec §10.14.) DCAlign's
   ``μint``/``μext`` penalize gap *count* per column, but the prior Λ decides gap
   *placement*. So for each worse pair we also compute the interior/terminal gap
   counts in both frames (:func:`gap_profile`) and bucket it (:func:`lever_bucket`,
   :func:`addressability`): :data:`PRIOR_ONLY` (equal gap counts — μ is provably
   neutral, only ``pcount``/prior-flattening can move it), :data:`MU_ADDRESSABLE`
   (a μ knob *could* help — candidate only), or :data:`MU_COUNTERPRODUCTIVE`. This
   is what tells us whether iter-003 should tune ``pcount`` or ``μint``/``μext``.

Both frames are length ``L`` integer arrays (gap = :data:`GAP`). The classifier
and gap-profile are purely geometric; the energies come from the validated
:func:`SBM.energy.dcalign_baseline.compare_record`, so this module introduces no
new energy math.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from SBM.utils.dcalign_score import DCAlignResult

from .dcalign_baseline import DEFAULT_EQUAL_TOL, compare_record
from .datasets import QueryRecord
from .encoding import GAP, seq_to_ints
from .model import PottsModel

#: Max |column offset| searched when testing for an internal register shift. A
#: family's insertions/deletions move blocks by a handful of columns; 15 spans
#: that comfortably for L≈90-96 without admitting spurious long-range "shifts".
MAX_SHIFT = 15

#: A candidate shift must reproduce at least this fraction of the native residues
#: (and beat the no-shift overlap) to be called a register shift rather than
#: coincidental partial overlap.
SHIFT_FRAC_THRESH = 0.5

#: A purely terminal disagreement must still leave at least this fraction of the
#: native residues sitting in an agreeing core; otherwise the "agreement" is only
#: trailing/leading gaps and the row is really a shift or redistribution.
TERMINAL_CORE_FRAC = 0.5


def group_kind(group: str) -> str:
    """``"CM/natural"`` -> ``"natural"``; ``"CM/synthetic-T1"`` -> ``"synthetic"``.

    Groups are ``<model>/<kind>[-T<temp>]`` (see
    :func:`SBM.energy.datasets.assemble_query_records`). Natural sets end in
    ``/natural``; everything else in the combine query is a synthetic temperature.
    """
    tail = group.rsplit("/", 1)[-1]
    return "natural" if tail == "natural" else "synthetic"


@dataclass(frozen=True)
class FrameGeometry:
    """Geometric classification of one native-vs-DCAlign frame pair."""

    n_disagree: int  # columns where the two L-frames differ in state
    lead_disagree: int  # disagreeing columns before the agreeing core
    trail_disagree: int  # disagreeing columns after the agreeing core
    interior_disagree: int  # disagreeing columns inside the agreeing core span
    n_agree_residue: int  # columns where both frames carry the SAME residue
    best_shift: int  # column offset that best re-aligns native residues (0 = none)
    best_shift_frac: float  # fraction of native residues that offset reproduces
    label: str  # recovered | terminal | register_shift | gap_redistribution


def _residue_overlap(native: np.ndarray, dca: np.ndarray, k: int) -> int:
    """# native-residue columns ``i`` whose residue reappears at ``i+k`` in ``dca``."""
    n = native.size
    count = 0
    for i in range(n):
        a = native[i]
        if a == GAP:
            continue
        j = i + k
        if 0 <= j < n and dca[j] == a:
            count += 1
    return count


def classify_frames(
    native: np.ndarray,
    dca: np.ndarray,
    *,
    max_shift: int = MAX_SHIFT,
    shift_frac_thresh: float = SHIFT_FRAC_THRESH,
    terminal_core_frac: float = TERMINAL_CORE_FRAC,
) -> FrameGeometry:
    """Classify how DCAlign's length-``L`` frame disagrees with the native frame.

    ``native`` and ``dca`` are equal-length integer arrays (gap = :data:`GAP`).
    Label order: ``recovered`` (identical) → ``terminal`` (disagreement only at
    the ends, agreeing core keeps ≥ ``terminal_core_frac`` of the residues) →
    ``register_shift`` (a nonzero offset reproduces ≥ ``shift_frac_thresh`` of the
    native residues and beats the no-shift overlap) → ``gap_redistribution``.
    """
    native = np.asarray(native, dtype=np.int64).ravel()
    dca = np.asarray(dca, dtype=np.int64).ravel()
    if native.size != dca.size:
        raise ValueError(f"frames must be equal length; got {native.size} and {dca.size}")
    if native.size == 0:
        raise ValueError("classify_frames needs non-empty frames")

    agree = native == dca
    disagree_cols = np.flatnonzero(~agree)
    n_disagree = int(disagree_cols.size)
    n_residues_native = int(np.count_nonzero(native != GAP))
    n_agree_residue = int(np.count_nonzero(agree & (native != GAP)))

    # No-shift residue overlap is the baseline a real shift must beat.
    overlap0 = _residue_overlap(native, dca, 0)
    best_shift, best_overlap = 0, overlap0
    for k in range(-max_shift, max_shift + 1):
        if k == 0:
            continue
        ov = _residue_overlap(native, dca, k)
        if ov > best_overlap:
            best_shift, best_overlap = k, ov
    best_shift_frac = (best_overlap / n_residues_native) if n_residues_native else 0.0

    if n_disagree == 0:
        return FrameGeometry(0, 0, 0, 0, n_agree_residue, 0, 0.0, "recovered")

    agree_cols = np.flatnonzero(agree)
    if agree_cols.size:
        core_lo, core_hi = int(agree_cols.min()), int(agree_cols.max())
        lead_disagree = int(np.count_nonzero(disagree_cols < core_lo))
        trail_disagree = int(np.count_nonzero(disagree_cols > core_hi))
        interior_disagree = n_disagree - lead_disagree - trail_disagree
    else:
        core_lo = core_hi = -1
        lead_disagree = trail_disagree = 0
        interior_disagree = n_disagree

    terminal = (
        interior_disagree == 0
        and (lead_disagree + trail_disagree) > 0
        and n_residues_native > 0
        and n_agree_residue >= terminal_core_frac * n_residues_native
    )
    if terminal:
        label = "terminal"
    elif best_shift != 0 and best_shift_frac >= shift_frac_thresh and best_overlap > overlap0:
        label = "register_shift"
    else:
        label = "gap_redistribution"

    return FrameGeometry(
        n_disagree=n_disagree,
        lead_disagree=lead_disagree,
        trail_disagree=trail_disagree,
        interior_disagree=interior_disagree,
        n_agree_residue=n_agree_residue,
        best_shift=best_shift if label == "register_shift" else 0,
        best_shift_frac=best_shift_frac,
        label=label,
    )


#: Lever buckets — which knob could push DCAlign toward the native frame.
PRIOR_ONLY = "prior_only"
MU_ADDRESSABLE = "mu_addressable"
MU_COUNTERPRODUCTIVE = "mu_counterproductive"


def gap_profile(frame: np.ndarray) -> tuple[int, int]:
    """``(n_interior_gaps, n_terminal_gaps)`` in a length-``L`` frame.

    Mirrors DCAlign's μint/μext split (``central!`` in ``iterate_bplc.jl``):
    *terminal* gaps are the leading + trailing gap runs (the μext region, before
    the first / after the last placed residue); *interior* gaps sit between
    placed residues (the μint region). An all-gap frame is all-terminal; a
    gapless frame is ``(0, 0)``. This is the geometry that decides whether the
    per-column μ penalties can discriminate two frames at all.
    """
    frame = np.asarray(frame, dtype=np.int64).ravel()
    is_gap = frame == GAP
    n_gap = int(is_gap.sum())
    residues = np.flatnonzero(~is_gap)
    if residues.size == 0:
        return 0, n_gap  # no placed residue → every gap is terminal
    lo, hi = int(residues[0]), int(residues[-1])
    n_int = int(is_gap[lo : hi + 1].sum())
    return n_int, n_gap - n_int


def lever_bucket(dn_int: int, dn_ext: int) -> str:
    """Which lever could move a worse-than-native pair toward the native frame.

    ``dn_* = n_*(native) − n_*(dcalign)``. A raised gap penalty μ ≥ 0 lowers
    native's *relative* objective only in a gap class where native has *fewer*
    gaps (``dn < 0``):

    - both ``dn == 0`` → :data:`PRIOR_ONLY`: equal gap counts, μ is provably
      neutral; only flattening the prior (``pcount``) can move it.
    - some ``dn < 0`` → :data:`MU_ADDRESSABLE`: a μ knob reduces native's
      relative penalty — a *candidate* (not a guarantee; see :func:`mu_floor`).
    - otherwise → :data:`MU_COUNTERPRODUCTIVE`: native has more gaps, so raising
      μ pushes *away* from native.
    """
    if dn_int == 0 and dn_ext == 0:
        return PRIOR_ONLY
    if min(dn_int, dn_ext) < 0:
        return MU_ADDRESSABLE
    return MU_COUNTERPRODUCTIVE


def mu_floor(delta_e: float, dn_int: int, dn_ext: int) -> float:
    """Smallest μ that *could* flip native vs DCAlign's frame — a lower bound.

    ``delta_e / |dn|`` for the most favourable helping knob (largest negative
    ``dn`` → smallest μ). This omits the unknown, native-disfavouring prior term
    (which is why DCAlign chose the worse frame in the first place) and assumes
    BP keeps the same two frames, so the true threshold is strictly higher — the
    number *sizes* the lever, it does not pick a value. NaN when no knob helps.
    """
    helpful = max(-dn_int if dn_int < 0 else 0, -dn_ext if dn_ext < 0 else 0)
    return float(delta_e) / helpful if helpful > 0 else float("nan")


@dataclass(frozen=True)
class ResidualRow:
    """One home-pair: its energy gap + frame-disagreement anatomy + lever."""

    sequence_id: str
    model: str
    group: str
    kind: str  # natural | synthetic
    delta_e: float  # E_dcalign − E_inframe (>equal_tol ⇒ worse than native)
    col_agreement: float
    n_residues_native: int
    n_residues_dcalign: int
    used_decimation: bool
    converged: bool
    n_disagree: int
    lead_disagree: int
    trail_disagree: int
    interior_disagree: int
    best_shift: int
    best_shift_frac: float
    label: str
    n_int_native: int  # interior gap columns in the native frame
    n_ext_native: int  # terminal gap columns in the native frame
    n_int_dcalign: int  # interior gap columns in DCAlign's frame (-1 if failed)
    n_ext_dcalign: int  # terminal gap columns in DCAlign's frame (-1 if failed)
    dn_int: int  # n_int_native − n_int_dcalign
    dn_ext: int  # n_ext_native − n_ext_dcalign
    lever: str  # prior_only | mu_addressable | mu_counterproductive | failed
    mu_floor: float  # lower-bound μ that could help (NaN unless mu_addressable)
    ok: bool  # False ⇒ DCAlign produced no frame

    def as_dict(self) -> dict:
        return asdict(self)


def analyze_record(
    record: QueryRecord, model: PottsModel, dca: DCAlignResult, **classify_kwargs
) -> ResidualRow:
    """Build one :class:`ResidualRow` for a home pair (record in ``model``'s frame).

    Energies/agreement reuse :func:`compare_record` (validated); the frame
    geometry is :func:`classify_frames` on the native frame (``record.ints``) vs
    DCAlign's cached frame. A failed alignment (empty frame) is kept with
    ``ok=False`` and a ``failed`` label — never silently dropped.
    """
    base = compare_record(record, model, dca)
    kind = group_kind(record.group)
    native = np.asarray(record.ints, dtype=np.int64)
    n_int_nat, n_ext_nat = gap_profile(native)  # native frame is always present
    if not base.ok:
        return ResidualRow(
            sequence_id=record.id, model=model.name, group=record.group, kind=kind,
            delta_e=base.delta_e, col_agreement=base.col_agreement,
            n_residues_native=base.n_residues, n_residues_dcalign=0,
            used_decimation=base.used_decimation, converged=base.converged,
            n_disagree=-1, lead_disagree=-1, trail_disagree=-1, interior_disagree=-1,
            best_shift=0, best_shift_frac=0.0, label="failed",
            n_int_native=n_int_nat, n_ext_native=n_ext_nat,
            n_int_dcalign=-1, n_ext_dcalign=-1, dn_int=0, dn_ext=0,
            lever="failed", mu_floor=float("nan"), ok=False,
        )
    dca_frame = seq_to_ints(dca.aligned_frame)
    geom = classify_frames(native, dca_frame, **classify_kwargs)
    n_int_dca, n_ext_dca = gap_profile(dca_frame)
    dn_int, dn_ext = n_int_nat - n_int_dca, n_ext_nat - n_ext_dca
    return ResidualRow(
        sequence_id=record.id, model=model.name, group=record.group, kind=kind,
        delta_e=base.delta_e, col_agreement=base.col_agreement,
        n_residues_native=base.n_residues,
        n_residues_dcalign=int(np.count_nonzero(dca_frame != GAP)),
        used_decimation=base.used_decimation, converged=base.converged,
        n_disagree=geom.n_disagree, lead_disagree=geom.lead_disagree,
        trail_disagree=geom.trail_disagree, interior_disagree=geom.interior_disagree,
        best_shift=geom.best_shift, best_shift_frac=geom.best_shift_frac,
        label=geom.label,
        n_int_native=n_int_nat, n_ext_native=n_ext_nat,
        n_int_dcalign=n_int_dca, n_ext_dcalign=n_ext_dca, dn_int=dn_int, dn_ext=dn_ext,
        lever=lever_bucket(dn_int, dn_ext), mu_floor=mu_floor(base.delta_e, dn_int, dn_ext),
        ok=True,
    )


def _decompose_subset(rows: list[ResidualRow], equal_tol: float) -> dict:
    ok = [r for r in rows if r.ok]
    n_worse = sum(1 for r in ok if r.delta_e > equal_tol)
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_failed": sum(1 for r in rows if not r.ok),
        "n_worse": n_worse,
        "frac_worse": (n_worse / len(ok)) if ok else 0.0,
    }


def decompose(rows: list[ResidualRow], *, equal_tol: float = DEFAULT_EQUAL_TOL) -> dict:
    """Worse-than-native tallies overall and split by model, kind, and group."""
    by_model: dict[str, list[ResidualRow]] = {}
    by_kind: dict[str, list[ResidualRow]] = {}
    by_group: dict[str, list[ResidualRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
        by_kind.setdefault(r.kind, []).append(r)
        by_group.setdefault(f"{r.model} | {r.group}", []).append(r)
    return {
        "equal_tol": equal_tol,
        "overall": _decompose_subset(rows, equal_tol),
        "by_model": {m: _decompose_subset(rs, equal_tol) for m, rs in sorted(by_model.items())},
        "by_kind": {k: _decompose_subset(rs, equal_tol) for k, rs in sorted(by_kind.items())},
        "by_group": {g: _decompose_subset(rs, equal_tol) for g, rs in sorted(by_group.items())},
    }


#: Disagreement labels a better insertion prior could plausibly fix (block
#: re-registration / terminal placement) vs. those it cannot (scattered gaps).
_PRIOR_SHAPED = ("terminal", "register_shift")


def anatomy(rows: list[ResidualRow], *, equal_tol: float = DEFAULT_EQUAL_TOL) -> dict:
    """Label counts among the worse-than-native rows, split by model and kind.

    Adds ``frac_prior_shaped`` (terminal+register_shift over the labelled worse
    rows) — the go/no-go signal: high on the natural tail ⇒ a better Λ is
    plausible; low (gap_redistribution-dominated) ⇒ pivot to gap-penalty tuning.
    """
    worse = [r for r in rows if r.ok and r.delta_e > equal_tol]

    def _counts(subset: list[ResidualRow]) -> dict:
        labels = ("terminal", "register_shift", "gap_redistribution")
        counts = {lab: sum(1 for r in subset if r.label == lab) for lab in labels}
        n = len(subset)
        prior_shaped = sum(counts[lab] for lab in _PRIOR_SHAPED)
        return {
            "n_worse": n,
            "labels": counts,
            "frac_prior_shaped": (prior_shaped / n) if n else 0.0,
        }

    models = sorted({r.model for r in worse})
    kinds = sorted({r.kind for r in worse})
    return {
        "equal_tol": equal_tol,
        "overall": _counts(worse),
        "by_kind": {k: _counts([r for r in worse if r.kind == k]) for k in kinds},
        "by_model_kind": {
            f"{m} | {k}": _counts([r for r in worse if r.model == m and r.kind == k])
            for m in models
            for k in kinds
        },
    }


def addressability(rows: list[ResidualRow], *, equal_tol: float = DEFAULT_EQUAL_TOL) -> dict:
    """Per-lever bucket counts among the worse-than-native rows.

    The iter-003 decision signal: of the worse pairs, how many are
    :data:`PRIOR_ONLY` (μ provably neutral — only ``pcount``/prior-flattening can
    move them), how many :data:`MU_ADDRESSABLE` (a μ knob *could* help —
    candidate only), how many :data:`MU_COUNTERPRODUCTIVE`. The μ_addressable
    count is split by knob (via μint when ``dn_int<0``, via μext when
    ``dn_ext<0``) with the median μ floor (a lower bound; see :func:`mu_floor`).
    Split overall / by kind / by (model, kind).
    """
    worse = [r for r in rows if r.ok and r.delta_e > equal_tol]
    buckets = (PRIOR_ONLY, MU_ADDRESSABLE, MU_COUNTERPRODUCTIVE)

    def _counts(subset: list[ResidualRow]) -> dict:
        counts = {b: sum(1 for r in subset if r.lever == b) for b in buckets}
        addr = [r for r in subset if r.lever == MU_ADDRESSABLE]
        floors = [r.mu_floor for r in addr if not math.isnan(r.mu_floor)]
        n = len(subset)
        return {
            "n_worse": n,
            "buckets": counts,
            "frac_prior_only": (counts[PRIOR_ONLY] / n) if n else 0.0,
            "mu_addressable_via_int": sum(1 for r in addr if r.dn_int < 0),
            "mu_addressable_via_ext": sum(1 for r in addr if r.dn_ext < 0),
            "mu_floor_median": float(np.median(floors)) if floors else None,
        }

    models = sorted({r.model for r in worse})
    kinds = sorted({r.kind for r in worse})
    return {
        "equal_tol": equal_tol,
        "overall": _counts(worse),
        "by_kind": {k: _counts([r for r in worse if r.kind == k]) for k in kinds},
        "by_model_kind": {
            f"{m} | {k}": _counts([r for r in worse if r.model == m and r.kind == k])
            for m in models
            for k in kinds
        },
    }


def insertion_free_check(rows: Iterable[ResidualRow], models: dict[str, int]) -> dict:
    """Per-model max ungapped residue count vs L — confirms queries carry no inserts.

    ``models`` maps model name → L. A home-pair query is ``strip_gaps`` of an
    L-frame row, so ``n_residues_native`` must be ≤ L; reporting the max states
    the "an insertion seed helps only via the prior, not the queries" premise
    from data instead of assumption.
    """
    out: dict[str, dict] = {}
    for name, L in models.items():
        n_res = [r.n_residues_native for r in rows if r.model == name and r.ok]
        out[name] = {
            "L": L,
            "max_n_residues": max(n_res) if n_res else 0,
            "insertion_free": (max(n_res) <= L) if n_res else True,
        }
    return out


def build_verdict(decomp: dict, anat: dict) -> str:
    """One-paragraph go/no-go drawn from the decomposition + anatomy numbers.

    The recommendation is advisory — the plan leaves the final Phase-1 call to the
    user. GO if the natural worse-than-native tail is mostly prior-shaped
    (terminal/register_shift); NO-GO if it is gap_redistribution-dominated or the
    residual is overwhelmingly synthetic (which the insertion prior cannot move).
    """
    nat = anat["by_kind"].get("natural", {"n_worse": 0, "frac_prior_shaped": 0.0})
    syn = anat["by_kind"].get("synthetic", {"n_worse": 0})
    n_nat, n_syn = nat["n_worse"], syn["n_worse"]
    total = n_nat + n_syn
    frac_syn = (n_syn / total) if total else 0.0
    frac_prior = nat["frac_prior_shaped"]
    rec = "GO" if (n_nat > 0 and frac_prior >= 0.5) else "NO-GO"
    return (
        f"Worse-than-native residual: {total} home pairs "
        f"({n_nat} natural, {n_syn} synthetic = {frac_syn:.0%} synthetic, which the "
        f"insertion prior cannot move). Of the natural tail, {frac_prior:.0%} is "
        f"prior-shaped (terminal/register_shift) vs gap_redistribution. "
        f"Recommendation: {rec} on the insertion-seed escalation "
        f"({'natural tail is plausibly fixable by a better Λ' if rec == 'GO' else 'residual is gap-redistribution / synthetic-dominated — pivot to μint/μext tuning'})."
    )


def lever_verdict(addr: dict) -> str:
    """One-paragraph lever recommendation from the addressability decomposition.

    The recommendation weights the **natural tail** — the known ground states are
    the recovery target; synthetics are diagnostic. ``pcount`` (prior-flattening)
    is indicated when that target is mostly :data:`PRIOR_ONLY`, since μ cannot
    separate equal-gap-count frames and ``pcount`` is the only knob that can move
    a prior_only pair at all. A μ sweep is indicated only when the target is
    mostly :data:`MU_ADDRESSABLE`; even then this screen *sizes* the lever (via
    the μ_floor lower bound), it does not pick a value. The advisory verdict
    deliberately ignores the synthetic-driven overall split, which can mislead.
    """
    o = addr["overall"]
    n = o["n_worse"]
    if n == 0:
        return "No worse-than-native pairs; nothing to tune."
    b = o["buckets"]
    nat = addr["by_kind"].get("natural")
    target = nat if (nat and nat["n_worse"]) else o  # fall back to overall if no naturals
    pcount_indicated = target["frac_prior_only"] >= 0.5
    via = "μext" if o["mu_addressable_via_ext"] >= o["mu_addressable_via_int"] else "μint"
    floor = o["mu_floor_median"]
    floor_str = f"median μ_floor≈{floor:.0f} a.u." if floor is not None else "no finite μ_floor"

    head = (
        f"Worse-than-native: {n} pairs — {b[PRIOR_ONLY]} prior_only, "
        f"{b[MU_ADDRESSABLE]} mu_addressable (almost all via {via}), "
        f"{b[MU_COUNTERPRODUCTIVE]} mu_counterproductive "
        f"({o['frac_prior_only']:.0%} prior_only overall). "
        f"Natural tail: {nat['n_worse'] if nat else 0} pairs, "
        f"{target['frac_prior_only']:.0%} prior_only."
    )
    if pcount_indicated:
        rec = (
            " Indicated lever: pcount (prior-flattening) — μ is provably neutral on the prior_only "
            "majority of the natural target, and pcount is the only knob that can move those (and "
            f"the prior_only synthetics). The mu_addressable rows are candidate-only (via {via}, "
            f"{floor_str} — a lower bound that ignores the native-disfavouring prior, so the true "
            "penalty is larger)."
        )
    else:
        rec = (
            f" Indicated lever: {via} gap penalty — most of the natural target has a candidate μ "
            f"direction ({floor_str}, a lower bound). A small μ sweep is warranted; this screen "
            "sizes the lever, it does not pick the value."
        )
    return head + rec
