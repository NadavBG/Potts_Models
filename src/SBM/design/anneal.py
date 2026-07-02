"""Two-model sequence design by *joint* simulated annealing (docs/two_model_progress.md #2).

Goal: find sequences that are simultaneously low-energy under two Potts models
(here CM, ``L=96``, and PPIC, ``L=91``), i.e. that minimize
``E_tot = w_A·E_A + w_B·E_B`` where each ``E_X`` is the query's energy *aligned*
into that model's frame.

The obstacle is cost. ``E_A``/``E_B`` are defined as the argmin over gap
placements (:func:`SBM.energy.potts_align.potts_align`); a length-91 query against
CM alone is a search over ``C(96, 5) ≈ 6.1e7`` frames — parallel tempering, ~8 s
per model per candidate. Re-aligning at every Metropolis step (~16 s/step) makes a
real anneal infeasible, and its cost scales with the (run-dependent) gap count.

**This engine folds the alignment into the Monte Carlo itself.** The state is the
core (ungapped) sequence *plus* its gap placement in each model's frame; every
proposal is an O(L) incremental energy update, so per-step cost is constant and
independent of gap count. Five move kinds anneal the whole state jointly:

* **substitute** a core residue (changes both frames at once),
* **slide** a residue into an adjacent/teleported gap column *in one frame* — the
  alignment degree of freedom, reusing the exact ``potts_align`` move
  (:func:`SBM.energy.potts_align._try_move`),
* **insert** / **delete** a core residue (birth/death; length ``N`` varies).

Insertion needs a free gap column in *both* frames, so at ``N = min(L_A, L_B) = 91``
(the shorter, PPIC frame is then gap-free) it is always rejected — this is what
enforces "never exceed N=91". Deletions are floored at ``schedule.min_length``.

Honest caveat: at finite temperature the alignment degrees of freedom sample a
thermal free-energy over frames, not the exact argmin; the two coincide as the
anneal cools toward ``T = 0.1`` (``beta_end``). This is an annealing *optimizer*
(the birth/death moves are not corrected for exact reversible-jump detailed
balance), so the authoritative per-model energies are obtained by a final real
:func:`~SBM.energy.potts_align.potts_align` "polish" on each finished sequence
(:func:`polish`), which also quantifies the thermal-alignment gap.

Energies follow the package convention ``E(S) = −(Σ_c h[c,f_c] + ½ Σ_{c,d} J)``
(lower is better; :mod:`SBM.energy.potts`); both models must be in the same
zero-sum gauge for ``E_A + E_B`` to be meaningful (:func:`SBM.energy.model.load_model`).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from SBM.energy.encoding import GAP, ints_to_seq
from SBM.energy.model import PottsModel
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align import (
    PTSchedule,
    PottsAlignResult,
    _frame_from_occupied,
    _try_move,
    potts_align,
)

log = logging.getLogger(__name__)

_MOVE_NAMES = ("sub", "slide_A", "slide_B", "insert", "delete")


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AnnealSchedule:
    """Joint-annealing schedule (all fields logged via :meth:`as_dict`).

    ``beta`` (inverse temperature) *rises* geometrically from ``beta_start`` to
    ``beta_end`` over ``n_steps`` (matching the ``potts_align`` convention), so
    ``T = 1/beta`` cools from ``1/beta_start`` to ``1/beta_end`` (default 1 → 0.1).
    ``p_*`` are the (renormalized) proposal probabilities of the five move kinds;
    ``record_every`` sub-samples the trajectory; ``min_length`` floors deletions;
    ``teleport_frac`` is passed through to the per-frame slide move.
    """

    n_steps: int = 500_000
    beta_start: float = 1.0
    beta_end: float = 10.0
    p_sub: float = 0.70
    p_slide_A: float = 0.10
    p_slide_B: float = 0.10
    p_insert: float = 0.05
    p_delete: float = 0.05
    teleport_frac: float = 0.3
    min_length: int = 70
    record_every: int = 1000

    def as_dict(self) -> dict:
        return asdict(self)

    def move_probs(self) -> np.ndarray:
        """Normalized proposal probabilities in :data:`_MOVE_NAMES` order."""
        p = np.array(
            [self.p_sub, self.p_slide_A, self.p_slide_B, self.p_insert, self.p_delete],
            dtype=np.float64,
        )
        total = p.sum()
        if total <= 0:
            raise ValueError("AnnealSchedule move probabilities sum to <= 0")
        return p / total


@dataclass
class DesignState:
    """Mutable joint state: core sequence + its gap placement in each frame.

    Invariant (test-enforced): ``frame_X[occ_X] == x`` in order, ``occ_X`` strictly
    monotone, ``len(x) == occ_A.size == occ_B.size``. ``E_A``/``E_B`` are the running
    incremental energies (re-verified from scratch by the drift canary at chain end).
    ``occ_*``/``frame_*`` for a *slide* are mutated in place by ``_try_move``; the
    indel moves reassign ``x``/``occ_*`` to new (shorter/longer) arrays.
    """

    x: np.ndarray          # (N,) core residues in 1..q-1, no gaps
    occ_A: np.ndarray      # (N,) monotone occupied columns in model A's frame
    occ_B: np.ndarray      # (N,) monotone occupied columns in model B's frame
    frame_A: np.ndarray    # (L_A,) in-frame sequence for model A (gap = 0)
    frame_B: np.ndarray    # (L_B,) in-frame sequence for model B
    E_A: float
    E_B: float

    @property
    def n_residues(self) -> int:
        return int(self.x.size)


@dataclass(frozen=True)
class _Ctx:
    """Per-run constants threaded through the move functions (avoids re-reads)."""

    JA: np.ndarray
    hA: np.ndarray
    idxA: np.ndarray
    LA: int
    JB: np.ndarray
    hB: np.ndarray
    idxB: np.ndarray
    LB: int
    q: int
    wA: float
    wB: float
    min_length: int
    teleport_frac: float


# --------------------------------------------------------------------------- #
# Incremental energy: one column changing state (covers sub / insert / delete)
# --------------------------------------------------------------------------- #

def _sub_delta(
    frame: np.ndarray, J: np.ndarray, h: np.ndarray, c: int, s0: int, s1: int, idx: np.ndarray
) -> float:
    """ΔE for changing ``frame[c]`` from state ``s0`` to ``s1`` (``frame[c] == s0`` now).

    Exact for ``E = −(Σ_c h[c,f_c] + ½ Σ_{c,d} J[c,d,f_c,f_d])`` — the same sum as
    :func:`SBM.utils.utils.compute_energies`, so the diagonal self term is carried.
    O(L): one length-L coupling gather against the unchanged columns (``d != c``),
    plus the field term and the ``½ J[c,c,·,·]`` self term. A *substitute* passes two
    residues; a *delete* passes ``s1 = GAP``; an *insert* passes ``s0 = GAP``.
    """
    f = frame
    dh = float(h[c, s1] - h[c, s0])
    row = J[c, idx, s1, f] - J[c, idx, s0, f]     # bond of column c to every column d
    off = float(row.sum() - row[c])               # drop d == c (handled by self term)
    self_c = 0.5 * float(J[c, c, s1, s1] - J[c, c, s0, s0])
    return -(dh + off + self_c)


def _accept(de: float, beta: float, rng: np.random.Generator) -> bool:
    """Metropolis rule at inverse temperature ``beta`` (matches ``potts_align``)."""
    return de <= 0.0 or rng.random() < math.exp(-beta * de)


# --------------------------------------------------------------------------- #
# Moves — each mutates ``st`` on acceptance and returns whether it accepted
# --------------------------------------------------------------------------- #

def _move_sub(st: DesignState, ctx: _Ctx, beta: float, rng: np.random.Generator) -> bool:
    """Substitute one core residue; changes column ``occ_A[k]`` in A and ``occ_B[k]`` in B."""
    n = st.x.size
    k = int(rng.integers(n))
    a = int(st.x[k])
    new = 1 + int(rng.integers(ctx.q - 2))        # uniform over the q-2 residues != a
    if new >= a:
        new += 1
    ca, cb = int(st.occ_A[k]), int(st.occ_B[k])
    dA = _sub_delta(st.frame_A, ctx.JA, ctx.hA, ca, a, new, ctx.idxA)
    dB = _sub_delta(st.frame_B, ctx.JB, ctx.hB, cb, a, new, ctx.idxB)
    if _accept(ctx.wA * dA + ctx.wB * dB, beta, rng):
        st.frame_A[ca] = new
        st.frame_B[cb] = new
        st.x[k] = new
        st.E_A += dA
        st.E_B += dB
        return True
    return False


def _move_slide(
    st: DesignState, ctx: _Ctx, beta: float, rng: np.random.Generator, which: str
) -> bool:
    """Alignment move: slide/teleport a residue into a gap column of *one* frame.

    Reuses the exact ``potts_align`` proposal + ΔE. Only ``E_A`` (or ``E_B``)
    changes, so the joint-Metropolis criterion ``exp(−beta·w_X·ΔE_X)`` is obtained
    by scaling the inverse temperature passed to ``_try_move`` by ``w_X``.
    """
    n = st.x.size
    if which == "A":
        de = _try_move(st.occ_A, st.frame_A, st.x, ctx.JA, ctx.hA, ctx.idxA,
                       ctx.LA, n, beta * ctx.wA, rng, teleport_frac=ctx.teleport_frac)
        if de != 0.0:
            st.E_A += de
            return True
        return False
    de = _try_move(st.occ_B, st.frame_B, st.x, ctx.JB, ctx.hB, ctx.idxB,
                   ctx.LB, n, beta * ctx.wB, rng, teleport_frac=ctx.teleport_frac)
    if de != 0.0:
        st.E_B += de
        return True
    return False


def _move_insert(st: DesignState, ctx: _Ctx, beta: float, rng: np.random.Generator) -> bool:
    """Birth move: insert a residue at core position ``k``, filling a gap in each frame.

    Between consecutive occupied columns every column is a gap, so the candidate
    slots in interval ``k`` are ``(occ[k-1], occ[k])`` exclusive. Needs a free slot
    in *both* frames — at ``N = min(L_A, L_B)`` the shorter frame is full, so this
    always rejects (the ``N ≤ 91`` cap).
    """
    n = st.x.size
    if n >= min(ctx.LA, ctx.LB):                  # shorter frame is gap-free: no room
        return False
    k = int(rng.integers(n + 1))
    loA = int(st.occ_A[k - 1]) if k > 0 else -1
    hiA = int(st.occ_A[k]) if k < n else ctx.LA
    loB = int(st.occ_B[k - 1]) if k > 0 else -1
    hiB = int(st.occ_B[k]) if k < n else ctx.LB
    if hiA - loA <= 1 or hiB - loB <= 1:          # no gap column in this interval
        return False
    new = 1 + int(rng.integers(ctx.q - 1))        # uniform over residues 1..q-1
    ca = int(rng.integers(loA + 1, hiA))
    cb = int(rng.integers(loB + 1, hiB))
    dA = _sub_delta(st.frame_A, ctx.JA, ctx.hA, ca, GAP, new, ctx.idxA)
    dB = _sub_delta(st.frame_B, ctx.JB, ctx.hB, cb, GAP, new, ctx.idxB)
    if _accept(ctx.wA * dA + ctx.wB * dB, beta, rng):
        st.frame_A[ca] = new
        st.frame_B[cb] = new
        st.occ_A = np.insert(st.occ_A, k, ca)
        st.occ_B = np.insert(st.occ_B, k, cb)
        st.x = np.insert(st.x, k, new)
        st.E_A += dA
        st.E_B += dB
        return True
    return False


def _move_delete(st: DesignState, ctx: _Ctx, beta: float, rng: np.random.Generator) -> bool:
    """Death move: remove core residue ``k``, vacating its column in both frames."""
    n = st.x.size
    if n <= ctx.min_length:
        return False
    k = int(rng.integers(n))
    a = int(st.x[k])
    ca, cb = int(st.occ_A[k]), int(st.occ_B[k])
    dA = _sub_delta(st.frame_A, ctx.JA, ctx.hA, ca, a, GAP, ctx.idxA)
    dB = _sub_delta(st.frame_B, ctx.JB, ctx.hB, cb, a, GAP, ctx.idxB)
    if _accept(ctx.wA * dA + ctx.wB * dB, beta, rng):
        st.frame_A[ca] = GAP
        st.frame_B[cb] = GAP
        st.occ_A = np.delete(st.occ_A, k)
        st.occ_B = np.delete(st.occ_B, k)
        st.x = np.delete(st.x, k)
        st.E_A += dA
        st.E_B += dB
        return True
    return False


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChainResult:
    """One annealing trajectory + its final designed sequence and polished energies."""

    chain_index: int
    seed: int
    w_A: float
    w_B: float
    start_type: str            # "random" | "natural_A" | "natural_B" (how the chain was seeded)
    schedule: AnnealSchedule
    # sub-sampled trajectory (length R = n_steps // record_every + 1)
    steps: np.ndarray
    temperatures: np.ndarray
    E_tot: np.ndarray
    E_A: np.ndarray
    E_B: np.ndarray
    n_residues: np.ndarray
    # final state (joint-MC, drift-checked)
    final_sequence: str
    final_n_residues: int
    E_A_mc: float
    E_B_mc: float
    final_frame_A: np.ndarray
    final_frame_B: np.ndarray
    # authoritative in-frame alignment for export: the polish argmin frame if polished,
    # else the joint-MC final frame. Length L_A / L_B, gap = 0. Serialized (small), so
    # the aligned-FASTA + ZAPPO figure can be produced from a cluster gather too.
    aln_frame_A: np.ndarray
    aln_frame_B: np.ndarray
    # authoritative argmin energies (real potts_align polish); None if not polished
    E_A_polish: float | None
    E_B_polish: float | None
    polish_exact_A: bool | None
    polish_exact_B: bool | None
    accept_counts: dict[str, int]
    propose_counts: dict[str, int]

    @property
    def E_tot_mc(self) -> float:
        return self.w_A * self.E_A_mc + self.w_B * self.E_B_mc

    @property
    def E_tot_polish(self) -> float | None:
        if self.E_A_polish is None or self.E_B_polish is None:
            return None
        return self.w_A * self.E_A_polish + self.w_B * self.E_B_polish

    @property
    def accept_rate(self) -> float:
        proposed = sum(self.propose_counts.values())
        accepted = sum(self.accept_counts.values())
        return accepted / proposed if proposed else 0.0

    def as_dict(self) -> dict:
        return {
            "chain_index": self.chain_index,
            "seed": self.seed,
            "w_A": self.w_A,
            "w_B": self.w_B,
            "start_type": self.start_type,
            "schedule": self.schedule.as_dict(),
            "trajectory": {
                "steps": self.steps.tolist(),
                "temperatures": self.temperatures.tolist(),
                "E_tot": self.E_tot.tolist(),
                "E_A": self.E_A.tolist(),
                "E_B": self.E_B.tolist(),
                "n_residues": self.n_residues.tolist(),
            },
            "final_sequence": self.final_sequence,
            "final_n_residues": self.final_n_residues,
            "aln_frame_A": self.aln_frame_A.tolist(),
            "aln_frame_B": self.aln_frame_B.tolist(),
            "E_A_mc": self.E_A_mc,
            "E_B_mc": self.E_B_mc,
            "E_tot_mc": self.E_tot_mc,
            "E_A_polish": self.E_A_polish,
            "E_B_polish": self.E_B_polish,
            "E_tot_polish": self.E_tot_polish,
            "polish_exact_A": self.polish_exact_A,
            "polish_exact_B": self.polish_exact_B,
            "accept_counts": self.accept_counts,
            "propose_counts": self.propose_counts,
            "accept_rate": self.accept_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChainResult":
        """Reconstruct from :meth:`as_dict` (e.g. a cluster shard JSONL line).

        The final in-frame arrays are not serialized (large, and not needed to
        write the run outputs), so they come back empty; re-run :func:`polish` if
        a frame is required.
        """
        t = d["trajectory"]
        return cls(
            chain_index=d["chain_index"], seed=d["seed"], w_A=d["w_A"], w_B=d["w_B"],
            start_type=d.get("start_type", "random"),
            schedule=AnnealSchedule(**d["schedule"]),
            steps=np.asarray(t["steps"], dtype=np.int64),
            temperatures=np.asarray(t["temperatures"], dtype=np.float64),
            E_tot=np.asarray(t["E_tot"], dtype=np.float64),
            E_A=np.asarray(t["E_A"], dtype=np.float64),
            E_B=np.asarray(t["E_B"], dtype=np.float64),
            n_residues=np.asarray(t["n_residues"], dtype=np.int64),
            final_sequence=d["final_sequence"], final_n_residues=d["final_n_residues"],
            E_A_mc=d["E_A_mc"], E_B_mc=d["E_B_mc"],
            final_frame_A=np.empty(0, dtype=np.int64), final_frame_B=np.empty(0, dtype=np.int64),
            aln_frame_A=np.asarray(d.get("aln_frame_A", []), dtype=np.int64),
            aln_frame_B=np.asarray(d.get("aln_frame_B", []), dtype=np.int64),
            E_A_polish=d["E_A_polish"], E_B_polish=d["E_B_polish"],
            polish_exact_A=d["polish_exact_A"], polish_exact_B=d["polish_exact_B"],
            accept_counts=d["accept_counts"], propose_counts=d["propose_counts"],
        )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def _initial_state(
    model_A: PottsModel, model_B: PottsModel, ctx: _Ctx, rng: np.random.Generator
) -> DesignState:
    """A random full-length (``N = min(L_A, L_B)``) start: identity frame in the
    shorter model, a random monotone placement in the longer one."""
    n0 = min(model_A.L, model_B.L)
    x = rng.integers(1, model_A.q, size=n0).astype(np.int64)
    # the shorter frame is fully occupied (its length == n0); the longer gets gaps.
    occ_A = (np.arange(model_A.L, dtype=np.int64) if model_A.L == n0
             else np.sort(rng.choice(model_A.L, size=n0, replace=False)).astype(np.int64))
    occ_B = (np.arange(model_B.L, dtype=np.int64) if model_B.L == n0
             else np.sort(rng.choice(model_B.L, size=n0, replace=False)).astype(np.int64))
    frame_A = _frame_from_occupied(occ_A, x, model_A.L)
    frame_B = _frame_from_occupied(occ_B, x, model_B.L)
    return DesignState(
        x=x, occ_A=occ_A, occ_B=occ_B, frame_A=frame_A, frame_B=frame_B,
        E_A=potts_energy(frame_A, model_A), E_B=potts_energy(frame_B, model_B),
    )


def initial_state_from_frame(
    frame_row: np.ndarray, model_A: PottsModel, model_B: PottsModel, *, home: str
) -> DesignState:
    """A start seeded from a *natural* sequence already aligned to one model's frame.

    ``frame_row`` is the natural in the ``home`` model's frame (length ``L_home``,
    ``gap = 0``); ``home`` is ``"A"`` or ``"B"``. Its non-gap columns become the core
    ``x`` at the native placement in the home frame (so the home energy starts at the
    natural's in-frame energy), and ``x`` is *left-packed* into the other frame — a
    poor but valid start the anneal's slide moves relax. The core length must be
    ``<= min(L_A, L_B)`` (the two-frame cap); the caller filters naturals to satisfy it.
    """
    if home not in ("A", "B"):
        raise ValueError(f"home must be 'A' or 'B', got {home!r}")
    home_model = model_A if home == "A" else model_B
    other_model = model_B if home == "A" else model_A
    frame_row = np.asarray(frame_row, dtype=np.int64)
    if frame_row.size != home_model.L:
        raise ValueError(f"frame_row length {frame_row.size} != L_{home}={home_model.L}")
    occ_home = np.nonzero(frame_row != GAP)[0].astype(np.int64)
    x = frame_row[occ_home].astype(np.int64)
    n = x.size
    cap = min(model_A.L, model_B.L)
    if not 0 < n <= cap:
        raise ValueError(f"natural core length {n} must be in (0, {cap}] to fit both frames")
    occ_other = np.arange(n, dtype=np.int64)          # left-pack; slides relax it
    frame_home = _frame_from_occupied(occ_home, x, home_model.L)
    frame_other = _frame_from_occupied(occ_other, x, other_model.L)
    if home == "A":
        occ_A, occ_B, frame_A, frame_B = occ_home, occ_other, frame_home, frame_other
    else:
        occ_A, occ_B, frame_A, frame_B = occ_other, occ_home, frame_other, frame_home
    return DesignState(
        x=x, occ_A=occ_A, occ_B=occ_B, frame_A=frame_A, frame_B=frame_B,
        E_A=potts_energy(frame_A, model_A), E_B=potts_energy(frame_B, model_B),
    )


def polish(x: np.ndarray, model: PottsModel, *, seed: int,
           init_frame: np.ndarray | None = None,
           pt_schedule: PTSchedule | None = None) -> PottsAlignResult:
    """Authoritative argmin energy of a finished design under one model.

    Thin wrapper over :func:`SBM.energy.potts_align.potts_align` (exact enumeration
    when affordable, else parallel tempering). This is the honest per-model energy
    the joint-MC ``E_X_mc`` approximates; run once per chain, not per step.

    ``init_frame`` (the joint-MC final frame) warm-starts the cold PT replicas at
    the good alignment the MC already found, so the polished energy can only
    match-or-improve it (a cold, high-gap PT search over ``C(L, g)`` frames often
    lands *worse* than the annealed frame). Ignored on the exact-enumeration branch,
    which returns the true global minimum regardless.
    """
    q = np.asarray(x, dtype=np.int64)
    init = [np.asarray(init_frame, dtype=np.int64)] if init_frame is not None else None
    return potts_align(q, model, seed=seed, init_frames=init, pt_schedule=pt_schedule)


def anneal_chain(
    model_A: PottsModel,
    model_B: PottsModel,
    w_A: float,
    w_B: float,
    schedule: AnnealSchedule,
    *,
    seed: int,
    chain_index: int = 0,
    do_polish: bool = True,
    polish_pt_schedule: PTSchedule | None = None,
    init_state: DesignState | None = None,
    start_type: str = "random",
) -> ChainResult:
    """Run one joint-annealing trajectory and return it with a polished final energy.

    ``seed`` is required and logged (per-chain determinism); ``model_A``/``model_B``
    must share the zero-sum gauge and the alphabet (``q``). Set ``do_polish=False``
    to skip the (per-chain) real ``potts_align`` and report only the joint-MC energy.

    ``init_state`` seeds the chain from a specific state (e.g. a natural via
    :func:`initial_state_from_frame`) instead of the default random start; it is
    deep-copied so the caller may reuse it. ``start_type`` is recorded on the result
    for downstream grouping/coloring (``"random"``/``"natural_A"``/``"natural_B"``).
    """
    if seed is None:
        raise ValueError("anneal_chain requires an explicit, logged seed")
    if model_A.q != model_B.q:
        raise ValueError(f"models disagree on q: {model_A.q} vs {model_B.q}")
    rng = np.random.default_rng(seed)
    ctx = _Ctx(
        JA=model_A.J, hA=model_A.h, idxA=np.arange(model_A.L), LA=model_A.L,
        JB=model_B.J, hB=model_B.h, idxB=np.arange(model_B.L), LB=model_B.L,
        q=model_A.q, wA=float(w_A), wB=float(w_B),
        min_length=schedule.min_length, teleport_frac=schedule.teleport_frac,
    )
    st = copy.deepcopy(init_state) if init_state is not None else _initial_state(model_A, model_B, ctx, rng)

    betas = np.geomspace(schedule.beta_start, schedule.beta_end, schedule.n_steps)
    cum = np.cumsum(schedule.move_probs())
    accept = {m: 0 for m in _MOVE_NAMES}
    propose = {m: 0 for m in _MOVE_NAMES}
    rec_steps, rec_T, rec_Et, rec_EA, rec_EB, rec_N = [], [], [], [], [], []

    def _record(step: int) -> None:
        rec_steps.append(step)
        rec_T.append(1.0 / betas[step])
        rec_EA.append(st.E_A)
        rec_EB.append(st.E_B)
        rec_Et.append(w_A * st.E_A + w_B * st.E_B)
        rec_N.append(st.n_residues)

    for step in range(schedule.n_steps):
        beta = betas[step]
        # clamp: normalized cumsum can round to <1.0, so a draw in the residual tail
        # would otherwise index past the last move (IndexError) for custom move probs.
        move = _MOVE_NAMES[min(int(np.searchsorted(cum, rng.random(), side="right")),
                               len(_MOVE_NAMES) - 1)]
        propose[move] += 1
        if move == "sub":
            ok = _move_sub(st, ctx, beta, rng)
        elif move == "slide_A":
            ok = _move_slide(st, ctx, beta, rng, "A")
        elif move == "slide_B":
            ok = _move_slide(st, ctx, beta, rng, "B")
        elif move == "insert":
            ok = _move_insert(st, ctx, beta, rng)
        else:
            ok = _move_delete(st, ctx, beta, rng)
        if ok:
            accept[move] += 1
        if step % schedule.record_every == 0:
            _record(step)
    if not rec_steps or rec_steps[-1] != schedule.n_steps - 1:
        _record(schedule.n_steps - 1)          # always capture the final (coldest) state

    # Drift canary: the running E accumulates thousands of ΔE; re-verify from scratch.
    E_A_mc = _canary(st.E_A, potts_energy(st.frame_A, model_A), "A", seed)
    E_B_mc = _canary(st.E_B, potts_energy(st.frame_B, model_B), "B", seed)

    EA_pol = EB_pol = None
    exact_A = exact_B = None
    aln_A, aln_B = st.frame_A.copy(), st.frame_B.copy()   # MC frame unless polish improves it
    if do_polish:
        res_A = polish(st.x, model_A, seed=seed, init_frame=st.frame_A,
                       pt_schedule=polish_pt_schedule)
        res_B = polish(st.x, model_B, seed=seed + 1, init_frame=st.frame_B,
                       pt_schedule=polish_pt_schedule)
        EA_pol, exact_A = res_A.best_energy, res_A.is_global_exact
        EB_pol, exact_B = res_B.best_energy, res_B.is_global_exact
        aln_A = np.asarray(res_A.best_frame, dtype=np.int64)   # authoritative argmin alignment
        aln_B = np.asarray(res_B.best_frame, dtype=np.int64)

    return ChainResult(
        chain_index=chain_index, seed=int(seed), w_A=float(w_A), w_B=float(w_B),
        start_type=start_type, schedule=schedule,
        steps=np.asarray(rec_steps, dtype=np.int64),
        temperatures=np.asarray(rec_T, dtype=np.float64),
        E_tot=np.asarray(rec_Et, dtype=np.float64),
        E_A=np.asarray(rec_EA, dtype=np.float64),
        E_B=np.asarray(rec_EB, dtype=np.float64),
        n_residues=np.asarray(rec_N, dtype=np.int64),
        final_sequence=ints_to_seq(st.x),
        final_n_residues=st.n_residues,
        E_A_mc=float(E_A_mc), E_B_mc=float(E_B_mc),
        final_frame_A=st.frame_A.copy(), final_frame_B=st.frame_B.copy(),
        aln_frame_A=aln_A, aln_frame_B=aln_B,
        E_A_polish=EA_pol, E_B_polish=EB_pol,
        polish_exact_A=exact_A, polish_exact_B=exact_B,
        accept_counts=accept, propose_counts=propose,
    )


def _canary(running: float, exact: float, tag: str, seed: int) -> float:
    """Warn + fall back to the from-scratch energy if incremental drift exceeds 1e-6."""
    if not math.isclose(running, exact, rel_tol=0, abs_tol=1e-6):
        log.warning(
            "anneal_chain[seed=%d]: incremental E_%s=%.8g vs exact %.8g (Δ=%.2e); using exact",
            seed, tag, running, exact, abs(exact - running),
        )
        return exact
    return running
