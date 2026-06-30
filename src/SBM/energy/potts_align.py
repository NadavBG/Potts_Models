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
class PTSchedule:
    """Parallel-tempering (replica-exchange) schedule — the high-gap engine.

    ``n_replicas`` copies run the same ±1-shift Metropolis dynamics at a *fixed*
    geometric β ladder from ``beta_min`` (hot — crosses barriers) to ``beta_max``
    (cold — refines). After each block of ``sweep_moves`` proposed moves per
    replica, adjacent replicas attempt a configuration swap with the replica-exchange
    Metropolis rule ``min(1, exp((β_a−β_b)(E_a−E_b)))``, so a low-energy config found
    hot migrates cold. This crosses the deep-but-narrow basins that single-temperature
    SA (`SASchedule`) cannot, at the cost of `n_replicas`× the work per restart.
    ``sweep_moves=None`` ⇒ one sweep = ``N`` proposed moves (one per residue on average).
    """

    n_replicas: int = 12
    beta_min: float = 0.05
    beta_max: float = 10.0
    n_blocks: int = 3000
    sweep_moves: int | None = None
    n_restarts: int = 5
    teleport_frac: float = 0.3
    topk: int = DEFAULT_TOPK
    enum_max_frames: int = DEFAULT_ENUM_MAX_FRAMES

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def thorough(cls) -> "PTSchedule":
        """A heavier ladder for the hardest ``g≥13`` cases / one important sequence
        (~3× the default budget; ~90 s/seq at ``L≈96``)."""
        return cls(n_replicas=14, beta_min=0.04, beta_max=12.0, n_blocks=6000,
                   n_restarts=6, teleport_frac=0.3)

    @classmethod
    def for_gap_count(cls, g: int) -> "PTSchedule":
        """The intelligent default: the schedule cost is ``g``-independent (§6.8), but
        the *budget needed* to reach native grows with ``g`` — so escalate to
        :meth:`thorough` for the hardest tail (``g ≥ 13``), where the default budget
        leaves a residual, and use the light default otherwise. ``g`` is always known
        (``L − N``), so this needs no ground truth."""
        return cls.thorough() if g >= 13 else cls()


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


def _try_move(
    cols: np.ndarray, frame: np.ndarray, q: np.ndarray, J: np.ndarray, h: np.ndarray,
    idx: np.ndarray, L: int, N: int, beta: float, rng: np.random.Generator,
    teleport_frac: float = 0.0,
) -> float:
    """Propose one move and apply it with Metropolis at inverse-temp ``beta``.

    Two move kinds. **±1 shift** (local): nudge a residue into an adjacent gap.
    **Teleport** (non-local, fraction ``teleport_frac``): jump a residue to *any*
    empty column between its neighbours — the lever for high-gap cases, where the
    span between consecutive residues is wide and a sequence of ±1 shifts gets
    topologically trapped (a residue cannot cross a barrier of intermediate
    high-energy frames). At ``teleport_frac=0`` the RNG draw order is byte-identical
    to the pure-±1 version (the ``teleport_frac > 0`` short-circuit skips the extra
    draw), so existing seeds reproduce.

    Mutates ``cols``/``frame`` in place on acceptance and returns the applied ΔE
    (``0.0`` if illegal or rejected). Shared by SA (:func:`_anneal_once`) and PT
    (:func:`pt_align`).
    """
    r = int(rng.integers(N))
    lo = int(cols[r - 1]) if r > 0 else -1          # exclusive lower bound
    hi = int(cols[r + 1]) if r < N - 1 else L       # exclusive upper bound
    if teleport_frac > 0.0 and rng.random() < teleport_frac:
        if hi - lo <= 2:                             # only cols[r] sits between: no room
            return 0.0
        dst = int(rng.integers(lo + 1, hi))         # any column strictly between the neighbours
        if dst == int(cols[r]):                      # drew the current column: no-op
            return 0.0
    else:
        direction = 1 if rng.random() < 0.5 else -1
        dst = int(cols[r]) + direction
        if not (lo < dst < hi):                      # off-grid or collides with a neighbour
            return 0.0
    a = int(q[r])
    src = int(cols[r])
    de = _move_delta(frame, J, h, src, dst, a, idx)
    if de <= 0.0 or rng.random() < math.exp(-beta * de):
        frame[src] = GAP
        frame[dst] = a
        cols[r] = dst
        return de
    return 0.0


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
        de = _try_move(cols, frame, q, J, h, idx, L, N, betas[step], rng)
        if de != 0.0:
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


def _pt_one(
    q: np.ndarray, model: PottsModel, sched: PTSchedule, betas: np.ndarray, seed: int,
    warm_cols: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """One parallel-tempering run; returns the global-best (frame, energy) it found.

    ``n_replicas`` replicas evolve at the fixed β ladder ``betas`` (ascending: hot →
    cold). The ``warm_cols`` warm-start the *coldest* replicas (where refinement
    happens); the rest start random. Adjacent replicas swap configurations between
    sweep blocks (alternating even/odd pairing), so a low-energy config found hot
    flows cold.
    """
    rng = np.random.default_rng(seed)
    L, N = model.L, q.size
    J, h, idx = model.J, model.h, np.arange(L)
    K = sched.n_replicas
    sweep_moves = sched.sweep_moves or N

    # cols/frames/energies per replica; warm frames seed the coldest replicas (high index).
    cols_r, frame_r, e_r = [], [], []
    for k in range(K):
        wi = k - (K - len(warm_cols))   # >=0 for the coldest len(warm_cols) replicas
        cols = (warm_cols[wi].copy() if wi >= 0
                else np.sort(rng.choice(L, size=N, replace=False)))
        frame = _frame_from_occupied(cols, q, L)
        cols_r.append(cols)
        frame_r.append(frame)
        e_r.append(potts_energy(frame, model))
    best = int(np.argmin(e_r))
    best_frame, best_energy = frame_r[best].copy(), float(e_r[best])

    for block in range(sched.n_blocks):
        for k in range(K):
            cols, frame, e = cols_r[k], frame_r[k], e_r[k]
            for _ in range(sweep_moves):
                de = _try_move(cols, frame, q, J, h, idx, L, N, betas[k], rng,
                               teleport_frac=sched.teleport_frac)
                if de != 0.0:
                    e += de
                    if e < best_energy:
                        best_energy, best_frame = e, frame.copy()
            e_r[k] = e
        # replica-exchange swaps on adjacent pairs (alternate parity each block).
        for k in range(block % 2, K - 1, 2):
            delta = (betas[k] - betas[k + 1]) * (e_r[k] - e_r[k + 1])
            if delta >= 0.0 or rng.random() < math.exp(delta):
                cols_r[k], cols_r[k + 1] = cols_r[k + 1], cols_r[k]
                frame_r[k], frame_r[k + 1] = frame_r[k + 1], frame_r[k]
                e_r[k], e_r[k + 1] = e_r[k + 1], e_r[k]
    return best_frame, best_energy


def pt_align(
    query: np.ndarray,
    model: PottsModel,
    *,
    seed: int,
    schedule: PTSchedule = PTSchedule(),
    sequence_id: str | None = None,
    init_frames: list[np.ndarray] | None = None,
) -> PottsAlignResult:
    """Parallel-tempering minimization of the Potts energy over gap placements.

    The high-gap engine: ``n_restarts`` independent replica-exchange runs (each a β
    ladder of ``n_replicas`` replicas), warm-started from ``init_frames`` on the cold
    replicas. Crosses the deep-but-narrow basins single-temperature SA cannot.
    Returns the global-best frame plus the ``topk`` distinct per-restart bests.
    ``seed`` is required and logged.
    """
    if seed is None:
        raise ValueError("pt_align requires an explicit, logged seed")
    q = _validate_query(query, model)
    warm_cols = [_frame_to_cols(f, q, model) for f in (init_frames or [])]
    if len(warm_cols) > schedule.n_replicas:
        # _pt_one seeds one warm frame per replica (coldest first) and leaves the
        # rest random; with more warm frames than replicas it would silently drop
        # the lowest-index (most production-relevant) ones. Refuse loudly instead.
        raise ValueError(
            f"pt_align got {len(warm_cols)} init_frames but only "
            f"{schedule.n_replicas} replicas; pass at most n_replicas warm starts "
            f"(or raise PTSchedule.n_replicas) so none are dropped"
        )
    betas = np.geomspace(schedule.beta_min, schedule.beta_max, schedule.n_replicas)
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(schedule.n_restarts)]

    frames, energies = [], []
    for s in seeds:
        bf, be = _pt_one(q, model, schedule, betas, s, warm_cols)
        # drift canary: re-verify the incremental best from scratch.
        exact = potts_energy(bf, model)
        if not math.isclose(exact, be, rel_tol=0, abs_tol=1e-6):
            log.warning("PT[seed=%d]: incremental best E=%.8g vs exact %.8g (Δ=%.2e); using exact",
                        s, be, exact, abs(exact - be))
            be = exact
        frames.append(bf)
        energies.append(be)

    frames_arr = np.array(frames)
    energies_arr = np.array(energies)
    best = int(np.argmin(energies_arr))
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
    n_moves = schedule.n_restarts * schedule.n_blocks * schedule.n_replicas * (schedule.sweep_moves or q.size)
    log.info("pt_align[%s]: %d restarts × %d replicas × %d blocks (%d warm), best E=%.6g",
             sequence_id, schedule.n_restarts, schedule.n_replicas, schedule.n_blocks,
             len(warm_cols), energies_arr[best])
    return PottsAlignResult(
        sequence_id=sequence_id, L=model.L, n_residues=q.size, method="pt",
        is_global_exact=False, n_frames_evaluated=n_moves,
        best_frame=frames_arr[best].copy(), best_energy=float(energies_arr[best]),
        topk_frames=topk_f, topk_energies=topk_e, master_seed=int(seed),
        schedule=schedule, restarts=[])


def potts_align(
    query: np.ndarray,
    model: PottsModel,
    *,
    seed: int,
    schedule: SASchedule = SASchedule(),
    sequence_id: str | None = None,
    init_frames: list[np.ndarray] | None = None,
    fallback: str = "pt",
    pt_schedule: PTSchedule | None = None,
) -> PottsAlignResult:
    """Dispatch: exact enumeration when ``C(L, N) <= schedule.enum_max_frames``, else
    an approximate search.

    ``fallback`` selects the approximate engine when the space is too large to
    enumerate: ``"pt"`` (default — parallel tempering, the high-gap engine) or
    ``"sa"`` (single-temperature multi-restart SA). ``init_frames`` warm-starts the
    approximate branch (ignored when enumeration is exact). ``pt_schedule`` overrides
    the per-``g`` default; when ``None`` the PT branch auto-selects by gap count
    (:meth:`PTSchedule.for_gap_count` — heavier for the hardest ``g≥13`` tail).
    """
    q = _validate_query(query, model)
    g = model.L - q.size
    n_frames = math.comb(model.L, q.size)
    if n_frames <= schedule.enum_max_frames:
        return enumerate_align(query, model, sequence_id=sequence_id,
                               max_frames=schedule.enum_max_frames, topk=schedule.topk)
    log.warning(
        "potts_align[%s]: %d gaps ⇒ C(L=%d, N=%d) = %.3g frames > enum_max_frames=%d; "
        "falling back to APPROXIMATE %s (result is NOT a provable global minimum). Raise "
        "schedule.enum_max_frames for an exact run if the frame count is affordable "
        "(see docs/POTTS_ALIGN.md §cost-by-gap-count).",
        sequence_id, g, model.L, q.size, float(n_frames), schedule.enum_max_frames,
        "parallel tempering" if fallback == "pt" else "simulated annealing")
    if fallback == "pt":
        return pt_align(query, model, seed=seed,
                        schedule=pt_schedule or PTSchedule.for_gap_count(g),
                        sequence_id=sequence_id, init_frames=init_frames)
    if fallback == "sa":
        return sa_align(query, model, seed=seed, schedule=schedule, sequence_id=sequence_id,
                        init_frames=init_frames)
    raise ValueError(f"fallback must be 'pt' or 'sa', got {fallback!r}")


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
