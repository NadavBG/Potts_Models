"""Couplings-aware Potts-energy aligner over gap placements (iter-003 §10.20).

The combine residual is a *search* failure: for the worse-than-native home pairs
DCAlign's belief propagation lands in a frame whose Potts energy is higher than
the native frame, even though native is reachable (a narrow BP basin, §10.18).
The production-legal lever is a **couplings-aware** init into that basin — an
alignment that minimizes the *full* Potts energy (the signal fields-MAP/Viterbi
lacks), produced **without** the ground-truth frame.

For an insert-free query (``N`` residues, model length ``L``) the alignment is a
monotone placement of the ``N`` residues into ``N`` of the ``L`` columns, the
other ``g = L − N`` columns being gaps — exactly ``C(L, g)`` frames. When that
count is small (e.g. the 1-gap PPIC home pairs: ``C(91, 1) = 91`` frames) the
*global* minimum is found by **exact enumeration** — no stochastic search, no
DCAlign. :func:`enumerate_align` is therefore the headline tool; the
multi-restart simulated annealing in :func:`sa_align` is only the fallback for
the few high-gap cases whose frame count is too large to enumerate, and it is
validated against enumeration on the small cases.

Frames follow the package convention (``SBM.energy.encoding``): length ``L``
integer arrays over ``"-ACDEFGHIKLMNPQRSTVWY"`` (gap = 0). Energies are the exact
in-frame Potts sum from :func:`SBM.energy.potts.potts_energy` /
:func:`~SBM.energy.potts.potts_energies` (the canonical batched implementation —
not re-derived here).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np

from .encoding import GAP
from .model import PottsModel
from .potts import potts_energies, potts_energy

log = logging.getLogger(__name__)

#: Default ceiling on enumerated frames. ``C(96, 3) ≈ 1.4e5`` (the 3-gap CM home
#: pairs) is enumerable in seconds; the 1/2-gap PPIC pairs are instant. Above
#: this the dispatcher falls back to SA (the 4+-gap CM cases).
DEFAULT_ENUM_MAX_FRAMES = 200_000
#: Rows per ``potts_energies`` call. ``compute_energies`` materializes an
#: ``(L, L, chunk)`` array, so this bounds peak memory (~0.3 GB at L=96).
DEFAULT_CHUNK = 4096
DEFAULT_TOPK = 4


class EnumerationInfeasible(RuntimeError):
    """Raised when ``C(L, g)`` exceeds the enumeration budget (use SA instead)."""


@dataclass(frozen=True)
class SASchedule:
    """Multi-restart simulated-annealing schedule (all fields logged).

    The move set is the single ±1 column shift (move one residue into an adjacent
    gap column, preserving monotone order). This is maximally ergodic in the
    sparse-residue regime that actually needs SA — the high-gap cases (7–14 gaps
    of ~96 columns) where residues have abundant adjacent gaps to bubble through
    — so no block moves are needed (and none are implemented, keeping the ΔE math
    a single two-column update). ``beta`` is the inverse temperature; it *rises*
    (cools) over each restart, matching the project/DCAlign convention.
    """

    n_restarts: int = 16
    n_steps: int = 5000
    beta_start: float = 0.5
    beta_end: float = 8.0
    topk: int = DEFAULT_TOPK
    enum_max_frames: int = DEFAULT_ENUM_MAX_FRAMES

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SARestart:
    """One annealing restart's outcome (diagnostics)."""

    seed: int
    final_energy: float
    best_energy: float
    n_accepted: int
    n_proposed: int


@dataclass(frozen=True)
class PottsAlignResult:
    """Couplings-aware alignment of one query — the production-legal init.

    Carries only the alignment (no ``E_native``/``ΔE``: the aligner never sees
    the ground-truth frame — that comparison is the analysis side's job).
    """

    sequence_id: str | None
    L: int
    n_residues: int
    method: str               # "enumerate" | "sa"
    is_global_exact: bool     # True iff the whole frame space was enumerated
    n_frames_evaluated: int
    best_frame: np.ndarray    # length-L, exactly n_residues non-gap, monotone
    best_energy: float
    topk_frames: list[np.ndarray]
    topk_energies: list[float]
    master_seed: int | None
    schedule: SASchedule | None
    restarts: list[SARestart] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id, "L": self.L, "n_residues": self.n_residues,
            "method": self.method, "is_global_exact": self.is_global_exact,
            "n_frames_evaluated": self.n_frames_evaluated,
            "best_frame": self.best_frame.tolist(), "best_energy": self.best_energy,
            "topk_frames": [f.tolist() for f in self.topk_frames],
            "topk_energies": list(self.topk_energies),
            "master_seed": self.master_seed,
            "schedule": self.schedule.as_dict() if self.schedule else None,
            "restarts": [asdict(r) for r in self.restarts],
        }


# --------------------------------------------------------------------------- #
# Validation + frame/placement conversions
# --------------------------------------------------------------------------- #

def _validate_query(query: np.ndarray, model: PottsModel) -> np.ndarray:
    q = np.asarray(query, dtype=np.int64)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D (raw ungapped residues), got shape {q.shape}")
    if q.size == 0:
        raise ValueError("query is empty")
    if q.size > model.L:
        raise ValueError(
            f"query has {q.size} residues > model L={model.L}; this aligner handles only "
            "insert-free queries (N <= L). Inserts (N > L) need DCAlign."
        )
    if q.min() <= GAP or q.max() >= model.q:
        raise ValueError(
            f"raw query must contain residues in 1..{model.q - 1} (no gaps); "
            f"got range [{q.min()}, {q.max()}]"
        )
    return q


def _frame_from_occupied(occupied: np.ndarray, residues: np.ndarray, L: int) -> np.ndarray:
    """Length-L frame: ``residues`` placed (in order) into the sorted ``occupied`` columns."""
    frame = np.zeros(L, dtype=np.int64)
    frame[occupied] = residues
    return frame


def _keep_topk(
    energies: np.ndarray, frames: np.ndarray, k: int
) -> tuple[list[float], list[np.ndarray]]:
    """The ``k`` lowest-energy frames, energy-ascending (frames are all distinct)."""
    order = np.argsort(energies, kind="stable")[:k]
    return [float(energies[i]) for i in order], [frames[i].copy() for i in order]


# --------------------------------------------------------------------------- #
# Exact enumeration (the headline path)
# --------------------------------------------------------------------------- #

def enumerate_align(
    query: np.ndarray,
    model: PottsModel,
    *,
    sequence_id: str | None = None,
    max_frames: int = DEFAULT_ENUM_MAX_FRAMES,
    topk: int = DEFAULT_TOPK,
    chunk: int = DEFAULT_CHUNK,
) -> PottsAlignResult:
    """Global Potts-energy minimum over all monotone insert-free frames.

    Enumerates every choice of ``N`` occupied columns out of ``L`` and returns the
    exact argmin (``is_global_exact=True``). Raises :class:`EnumerationInfeasible`
    when ``C(L, N) > max_frames`` (caller should use :func:`sa_align`).
    """
    q = _validate_query(query, model)
    N, L = q.size, model.L
    n_frames = math.comb(L, N)
    if n_frames > max_frames:
        raise EnumerationInfeasible(
            f"C({L}, {N}) = {n_frames} > max_frames={max_frames}; use sa_align")

    occ = np.fromiter(
        (c for combo in combinations(range(L), N) for c in combo),
        dtype=np.int64, count=n_frames * N,
    ).reshape(n_frames, N)
    frames = np.zeros((n_frames, L), dtype=np.int64)
    np.put_along_axis(frames, occ, q[np.newaxis, :], axis=1)

    energies = np.empty(n_frames, dtype=np.float64)
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        energies[start:stop] = potts_energies(frames[start:stop], model)

    best = int(np.argmin(energies))
    topk_e, topk_f = _keep_topk(energies, frames, topk)
    log.info("enumerate_align[%s]: %d frames, best E=%.6g", sequence_id, n_frames, energies[best])
    return PottsAlignResult(
        sequence_id=sequence_id, L=L, n_residues=N, method="enumerate",
        is_global_exact=True, n_frames_evaluated=n_frames,
        best_frame=frames[best].copy(), best_energy=float(energies[best]),
        topk_frames=topk_f, topk_energies=topk_e, master_seed=None, schedule=None,
    )


# --------------------------------------------------------------------------- #
# Simulated annealing (the high-gap fallback)
# --------------------------------------------------------------------------- #

def _move_delta(
    frame: np.ndarray, J: np.ndarray, h: np.ndarray, src: int, dst: int, a: int, idx: np.ndarray
) -> float:
    """ΔE for moving residue ``a`` from column ``src`` (→gap) to gap column ``dst`` (→``a``).

    Exact for the energy ``E = −(Σ_c h[c,f_c] + ½ Σ_{c,d} J[c,d,f_c,f_d])`` where
    ``frame`` is the configuration *before* the move (so ``f[src]=a``, ``f[dst]=0``).
    O(L): two length-L coupling gathers against the unchanged columns, plus the
    explicit src–dst pair and the two self terms.
    """
    f = frame
    df = (h[src, GAP] - h[src, a]) + (h[dst, a] - h[dst, GAP])
    s = J[src, idx, GAP, f] - J[src, idx, a, f]   # src: a -> gap, bonds to every column d
    t = J[dst, idx, a, f] - J[dst, idx, GAP, f]   # dst: gap -> a, bonds to every column d
    sum_src = float(s.sum() - s[src] - s[dst])    # drop d in {src,dst}: handled below
    sum_dst = float(t.sum() - t[src] - t[dst])
    pair = float(J[src, dst, GAP, a] - J[src, dst, a, GAP])              # src-dst bond (once)
    self_src = 0.5 * float(J[src, src, GAP, GAP] - J[src, src, a, a])    # src self term
    self_dst = 0.5 * float(J[dst, dst, a, a] - J[dst, dst, GAP, GAP])    # dst self term
    return -(df + sum_src + sum_dst + pair + self_src + self_dst)


def _anneal_once(
    q: np.ndarray, model: PottsModel, sched: SASchedule, seed: int,
    init_cols: np.ndarray | None = None,
) -> tuple[np.ndarray, float, SARestart]:
    """One annealing restart; returns its best frame.

    ``init_cols`` (a length-N monotone occupied-column vector) warm-starts the
    restart at a heuristic frame (e.g. fields-MAP or DCAlign's own frame — both
    production-legal). ``None`` ⇒ a random monotone placement. Warm starts are the
    lever for the high-gap cases whose ``C(L, g)`` is too large for SA to reach
    native from a random start in a bounded budget.
    """
    rng = np.random.default_rng(seed)
    L, N = model.L, q.size
    J, h, idx = model.J, model.h, np.arange(L)

    cols = (np.sort(np.asarray(init_cols, dtype=np.int64)) if init_cols is not None
            else np.sort(rng.choice(L, size=N, replace=False)))
    frame = _frame_from_occupied(cols, q, L)
    energy = potts_energy(frame, model)
    best_frame, best_energy = frame.copy(), energy

    betas = np.geomspace(sched.beta_start, sched.beta_end, sched.n_steps)
    n_acc = 0
    for step in range(sched.n_steps):
        r = int(rng.integers(N))
        direction = 1 if rng.random() < 0.5 else -1
        dst = int(cols[r]) + direction
        lo = int(cols[r - 1]) if r > 0 else -1          # exclusive lower bound
        hi = int(cols[r + 1]) if r < N - 1 else L       # exclusive upper bound
        if not (lo < dst < hi):                          # off-grid or collides with a neighbour
            continue
        a = int(q[r])
        src = int(cols[r])
        de = _move_delta(frame, J, h, src, dst, a, idx)
        if de <= 0.0 or rng.random() < math.exp(-betas[step] * de):
            frame[src] = GAP
            frame[dst] = a
            cols[r] = dst
            energy += de
            n_acc += 1
            if energy < best_energy:
                best_energy, best_frame = energy, frame.copy()

    # Re-verify the accumulated best against a from-scratch sum (drift canary).
    exact = potts_energy(best_frame, model)
    if not math.isclose(exact, best_energy, rel_tol=0, abs_tol=1e-6):
        log.warning("SA[seed=%d]: incremental best E=%.8g vs exact %.8g (Δ=%.2e); using exact",
                    seed, best_energy, exact, abs(exact - best_energy))
        best_energy = exact
    return best_frame, best_energy, SARestart(
        seed=seed, final_energy=float(energy), best_energy=float(best_energy),
        n_accepted=n_acc, n_proposed=sched.n_steps)


def _frame_to_cols(frame: np.ndarray, q: np.ndarray, model: PottsModel) -> np.ndarray:
    """Occupied columns of an insert-free warm-start frame (validated against ``q``)."""
    f = np.asarray(frame, dtype=np.int64)
    if f.size != model.L:
        raise ValueError(f"init frame length {f.size} != L={model.L}")
    occ = np.flatnonzero(f != GAP)
    if not np.array_equal(f[occ], q):
        raise ValueError("init frame residues (in order) do not match the query")
    return occ


def sa_align(
    query: np.ndarray,
    model: PottsModel,
    *,
    seed: int,
    schedule: SASchedule = SASchedule(),
    sequence_id: str | None = None,
    init_frames: list[np.ndarray] | None = None,
) -> PottsAlignResult:
    """Multi-restart SA minimization of the Potts energy over gap placements.

    Runs one annealing restart from each frame in ``init_frames`` (heuristic warm
    starts — fields-MAP / DCAlign frames, production-legal) *plus* ``n_restarts``
    random restarts, all by single ±1 column shifts. Returns the best frame over
    all restarts plus the ``topk`` distinct restart-best frames (the
    couplings-aware "multi-start" for the min-over-K warm start). ``seed`` is
    required and logged.
    """
    if seed is None:
        raise ValueError("sa_align requires an explicit, logged seed")
    q = _validate_query(query, model)
    warm_cols = [_frame_to_cols(f, q, model) for f in (init_frames or [])]
    n_total = schedule.n_restarts + len(warm_cols)
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(n_total)]

    frames, energies, restarts = [], [], []
    for i, s in enumerate(seeds):
        init = warm_cols[i] if i < len(warm_cols) else None
        bf, be, rec = _anneal_once(q, model, schedule, s, init_cols=init)
        frames.append(bf)
        energies.append(be)
        restarts.append(rec)

    frames_arr = np.array(frames)
    energies_arr = np.array(energies)
    best = int(np.argmin(energies_arr))
    # top-K distinct restart-best frames, energy-ascending.
    order = np.argsort(energies_arr, kind="stable")
    topk_f, topk_e, seen = [], [], set()
    for i in order:
        key = frames_arr[i].tobytes()
        if key in seen:
            continue
        seen.add(key)
        topk_f.append(frames_arr[i].copy())
        topk_e.append(float(energies_arr[i]))
        if len(topk_f) >= schedule.topk:
            break
    log.info("sa_align[%s]: %d restarts (%d warm), best E=%.6g (%d distinct topk)",
             sequence_id, n_total, len(warm_cols), energies_arr[best], len(topk_f))
    return PottsAlignResult(
        sequence_id=sequence_id, L=model.L, n_residues=q.size, method="sa",
        is_global_exact=False, n_frames_evaluated=n_total * schedule.n_steps,
        best_frame=frames_arr[best].copy(), best_energy=float(energies_arr[best]),
        topk_frames=topk_f, topk_energies=topk_e, master_seed=int(seed),
        schedule=schedule, restarts=restarts)


def potts_align(
    query: np.ndarray,
    model: PottsModel,
    *,
    seed: int,
    schedule: SASchedule = SASchedule(),
    sequence_id: str | None = None,
    init_frames: list[np.ndarray] | None = None,
) -> PottsAlignResult:
    """Dispatch: exact enumeration when ``C(L, N) <= schedule.enum_max_frames``, else SA.

    ``init_frames`` warm-starts the SA branch (ignored when enumeration is exact —
    enumeration already finds the global minimum).
    """
    q = _validate_query(query, model)
    n_frames = math.comb(model.L, q.size)
    if n_frames <= schedule.enum_max_frames:
        return enumerate_align(query, model, sequence_id=sequence_id,
                               max_frames=schedule.enum_max_frames, topk=schedule.topk)
    return sa_align(query, model, seed=seed, schedule=schedule, sequence_id=sequence_id,
                    init_frames=init_frames)


# --------------------------------------------------------------------------- #
# Basin-width perturbation (M3 diagnostic helper)
# --------------------------------------------------------------------------- #

def perturb_frame(native_frame: np.ndarray, k: int, *, rng: np.random.Generator) -> np.ndarray:
    """A length-L insert-free frame differing from ``native_frame`` by reassigning ``k`` columns.

    ``k`` native-occupied columns are vacated and ``k`` native-gap columns filled;
    the residues are re-threaded *in order* into the new sorted occupied set, so
    the residue count and monotone order are preserved. ``rng`` is seeded/logged
    by the caller (the basin-width probe).
    """
    native = np.asarray(native_frame, dtype=np.int64)
    if native.ndim != 1:
        raise ValueError(f"native_frame must be 1-D, got shape {native.shape}")
    occ = np.flatnonzero(native != GAP)
    gaps = np.flatnonzero(native == GAP)
    residues = native[occ]
    if k < 0 or k > min(occ.size, gaps.size):
        raise ValueError(
            f"k={k} out of range for {occ.size} occupied / {gaps.size} gap columns")
    if k == 0:
        return native.copy()
    drop = rng.choice(occ, size=k, replace=False)
    add = rng.choice(gaps, size=k, replace=False)
    kept = np.setdiff1d(occ, drop, assume_unique=True)
    new_occ = np.sort(np.concatenate([kept, add]))
    return _frame_from_occupied(new_occ, residues, native.size)
