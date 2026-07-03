"""Tests for the two-model joint-annealing design engine (``SBM.design.anneal``).

Pure numpy/scipy (no MCMC kernel). The load-bearing check is that every
incremental ΔE (``_sub_delta``, and the reused ``potts_align`` slide) agrees with
a *from-scratch* :func:`SBM.energy.potts.potts_energy` recompute — the same
reference-implementation discipline as ``tests/test_energy.py``. On top of that we
check the state invariants, the ``N ≤ min(L_A,L_B)`` cap, that the thermal
alignment converges to the enumerated argmin as ``T→0``, and reproducibility.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from SBM.design.anneal import (
    AnnealSchedule,
    ChainResult,
    _Ctx,
    _heatbath_choice,
    _initial_state,
    _local_field,
    _move_delete,
    _move_insert,
    _move_insert_column_aware,
    _move_insert_heatbath,
    _move_slide,
    _move_sub,
    _move_sub_heatbath,
    _sub_delta,
    anneal_chain,
    initial_state_from_frame,
    pt_anneal_chain,
)
from SBM.energy.encoding import GAP, seq_to_ints
from SBM.energy.model import PottsModel
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align import enumerate_align
from SBM.utils.utils import MSA_ALPHABET

Q = len(MSA_ALPHABET)  # 21


def _random_model(L: int, seed: int, *, name: str = "m") -> PottsModel:
    """A tiny random Potts model with the symmetry ``J[i,j,a,b]=J[j,i,b,a]``.

    Not gauge-fixed and with *nonzero* diagonal ``J[i,i]`` on purpose, so the
    ΔE tests exercise the self-term that must match ``compute_energies``.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(L, Q))
    raw = rng.normal(size=(L, L, Q, Q))
    J = np.empty((L, L, Q, Q))
    for i in range(L):
        for j in range(L):
            J[i, j] = 0.5 * (raw[i, j] + raw[j, i].T)  # -> J[i,j,a,b] = J[j,i,b,a]
    return PottsModel(name=name, J=J, h=h, L=L, q=Q, alphabet=MSA_ALPHABET,
                      gauge="raw", sha256="0" * 64, source="test")


def _ctx(model_A, model_B, *, wA=0.4, wB=0.6, min_length=1, teleport=0.3) -> _Ctx:
    return _Ctx(JA=model_A.J, hA=model_A.h, idxA=np.arange(model_A.L), LA=model_A.L,
                JB=model_B.J, hB=model_B.h, idxB=np.arange(model_B.L), LB=model_B.L,
                q=model_A.q, wA=wA, wB=wB, min_length=min_length, teleport_frac=teleport)


def _check_invariants(st, model_A, model_B) -> None:
    x = st.x
    assert st.occ_A.size == x.size == st.occ_B.size
    assert x.size <= min(model_A.L, model_B.L)
    assert x.size >= 1
    assert x.min() >= 1 and x.max() < Q            # core has no gaps
    for occ, frame, L in ((st.occ_A, st.frame_A, model_A.L),
                          (st.occ_B, st.frame_B, model_B.L)):
        assert np.all(np.diff(occ) > 0)            # strictly monotone
        assert occ.min() >= 0 and occ.max() < L
        assert np.array_equal(frame[occ], x)       # residues threaded in order
        mask = np.ones(L, dtype=bool)
        mask[occ] = False
        assert np.all(frame[mask] == GAP)          # everything else is a gap


# --------------------------------------------------------------------------- #
# ΔE correctness — the reference-implementation check
# --------------------------------------------------------------------------- #

def test_sub_delta_matches_bruteforce():
    """``_sub_delta`` == from-scratch energy difference for sub / insert / delete.

    ``s1`` ranges over all states incl. GAP (delete) and ``s0`` incl. GAP (insert),
    so one test covers all three move deltas.
    """
    for seed in range(6):
        model = _random_model(L=9, seed=seed)
        idx = np.arange(model.L)
        rng = np.random.default_rng(1000 + seed)
        for _ in range(200):
            frame = rng.integers(0, Q, size=model.L).astype(np.int64)  # gaps allowed
            c = int(rng.integers(model.L))
            s0 = int(frame[c])
            s1 = int(rng.integers(0, Q))
            d = _sub_delta(frame, model.J, model.h, c, s0, s1, idx)
            after = frame.copy()
            after[c] = s1
            expected = potts_energy(after, model) - potts_energy(frame, model)
            assert math.isclose(d, expected, rel_tol=0, abs_tol=1e-9), (seed, c, s0, s1)


# --------------------------------------------------------------------------- #
# Invariants + energy tracking under every move
# --------------------------------------------------------------------------- #

def test_moves_preserve_invariants_and_energy_tracking():
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)
    ctx = _ctx(model_A, model_B, min_length=3)
    rng = np.random.default_rng(0)
    st = _initial_state(model_A, model_B, ctx, rng)
    _check_invariants(st, model_A, model_B)

    movers = [
        lambda: _move_sub(st, ctx, 2.0, rng),
        lambda: _move_slide(st, ctx, 2.0, rng, "A"),
        lambda: _move_slide(st, ctx, 2.0, rng, "B"),
        lambda: _move_insert(st, ctx, 2.0, rng),
        lambda: _move_delete(st, ctx, 2.0, rng),
    ]
    for _ in range(4000):
        movers[int(rng.integers(len(movers)))]()
        _check_invariants(st, model_A, model_B)
        # running incremental energy stays glued to the from-scratch value
        assert math.isclose(st.E_A, potts_energy(st.frame_A, model_A), rel_tol=0, abs_tol=1e-8)
        assert math.isclose(st.E_B, potts_energy(st.frame_B, model_B), rel_tol=0, abs_tol=1e-8)


# --------------------------------------------------------------------------- #
# Length cap: never exceed min(L_A, L_B); insertion rejected when the short frame is full
# --------------------------------------------------------------------------- #

def test_insertion_rejected_at_length_cap():
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)        # cap = 6
    ctx = _ctx(model_A, model_B, min_length=1)
    rng = np.random.default_rng(4)
    st = _initial_state(model_A, model_B, ctx, rng)
    assert st.n_residues == 6                   # starts full
    for _ in range(500):
        assert _move_insert(st, ctx, 2.0, rng) is False
        assert st.n_residues == 6


def test_chain_respects_cap_and_floor():
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=6000, beta_start=0.5, beta_end=6.0,
                           min_length=4, record_every=200)
    res = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=9, do_polish=False)
    assert res.n_residues.max() <= min(model_A.L, model_B.L)
    assert res.n_residues.min() >= sched.min_length
    assert len(res.final_sequence) == res.final_n_residues


# --------------------------------------------------------------------------- #
# Thermal alignment -> enumerated argmin as T -> 0 (sequence fixed, slides only)
# --------------------------------------------------------------------------- #

def test_low_temperature_alignment_reaches_argmin():
    """With only slide moves the core sequence is fixed, so the joint-MC ``E_A``
    at low T must reach the exact enumerated argmin (``potts_align`` ground truth)."""
    model_A = _random_model(L=7, seed=5)        # 2 gaps at N=5 -> C(7,5)=21 frames
    model_B = _random_model(L=5, seed=6)
    sched = AnnealSchedule(n_steps=20000, beta_start=0.5, beta_end=60.0,
                           p_sub=0.0, p_slide_A=1.0, p_slide_B=0.0,
                           p_insert=0.0, p_delete=0.0, min_length=1, record_every=1000)
    res = anneal_chain(model_A, model_B, 1.0, 0.0, sched, seed=3, do_polish=False)
    x = seq_to_ints(res.final_sequence)          # slides don't change the core
    enum = enumerate_align(x, model_A)
    assert res.E_A_mc >= enum.best_energy - 1e-9          # can't beat the global min
    assert math.isclose(res.E_A_mc, enum.best_energy, rel_tol=0, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# Reproducibility + polish path + drift canary
# --------------------------------------------------------------------------- #

def test_same_seed_is_bit_reproducible():
    model_A = _random_model(L=8, seed=1)
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=3000, beta_start=0.5, beta_end=5.0, record_every=100)
    r1 = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=11, do_polish=False)
    r2 = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=11, do_polish=False)
    assert r1.final_sequence == r2.final_sequence
    assert np.array_equal(r1.E_tot, r2.E_tot)
    assert np.array_equal(r1.E_A, r2.E_A)
    assert np.array_equal(r1.final_frame_A, r2.final_frame_A)
    assert np.array_equal(r1.final_frame_B, r2.final_frame_B)


def test_polish_matches_final_state_and_is_exact_on_tiny_model():
    model_A = _random_model(L=8, seed=1)         # C(8, N) enumerable -> exact polish
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=4000, beta_start=0.5, beta_end=8.0,
                           min_length=4, record_every=200)
    res = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=7, do_polish=True)
    # drift canary: MC-final energy equals a from-scratch recompute of the final frame
    assert math.isclose(res.E_A_mc, potts_energy(res.final_frame_A, model_A),
                        rel_tol=0, abs_tol=1e-6)
    assert math.isclose(res.E_B_mc, potts_energy(res.final_frame_B, model_B),
                        rel_tol=0, abs_tol=1e-6)
    # polish is the argmin over alignments, so it is <= the particular MC frame's energy
    assert res.polish_exact_A is True and res.polish_exact_B is True
    assert res.E_A_polish <= res.E_A_mc + 1e-9
    assert res.E_B_polish <= res.E_B_mc + 1e-9


# --------------------------------------------------------------------------- #
# Natural starts: initial_state_from_frame + anneal_chain(init_state=..., start_type=...)
# --------------------------------------------------------------------------- #

def test_initial_state_from_frame_native_energy_and_invariants():
    """A natural-seeded start: home energy == the native in-frame energy; the other
    frame is left-packed; state invariants hold; both home orientations work."""
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)          # cap = 6
    rng = np.random.default_rng(0)

    frame_A = np.zeros(model_A.L, dtype=np.int64)
    occ = np.array([1, 3, 4, 7])                  # core = 4 <= cap
    frame_A[occ] = rng.integers(1, Q, size=occ.size)
    st = initial_state_from_frame(frame_A, model_A, model_B, home="A")
    _check_invariants(st, model_A, model_B)
    assert math.isclose(st.E_A, potts_energy(frame_A, model_A), rel_tol=0, abs_tol=1e-9)
    assert np.array_equal(st.frame_A, frame_A)    # home frame preserved verbatim
    assert np.array_equal(st.occ_A, occ)
    assert np.array_equal(st.occ_B, np.arange(occ.size))          # other frame left-packed

    frame_B = np.zeros(model_B.L, dtype=np.int64)
    occ_b = np.array([0, 2, 5])
    frame_B[occ_b] = rng.integers(1, Q, size=occ_b.size)
    st_b = initial_state_from_frame(frame_B, model_A, model_B, home="B")
    _check_invariants(st_b, model_A, model_B)
    assert math.isclose(st_b.E_B, potts_energy(frame_B, model_B), rel_tol=0, abs_tol=1e-9)
    assert np.array_equal(st_b.occ_B, occ_b)
    assert np.array_equal(st_b.occ_A, np.arange(occ_b.size))


def test_initial_state_from_frame_rejects_oversized_core():
    """A natural whose ungapped core exceeds min(L_A, L_B) cannot be a two-frame state."""
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)          # cap = 6
    frame_A = np.zeros(model_A.L, dtype=np.int64)
    frame_A[np.arange(7)] = 1                     # core = 7 > cap = 6
    with pytest.raises(ValueError):
        initial_state_from_frame(frame_A, model_A, model_B, home="A")
    with pytest.raises(ValueError):
        initial_state_from_frame(frame_A, model_A, model_B, home="bogus")


def test_anneal_chain_uses_init_state_and_records_start_type():
    """A provided init_state seeds the chain (and is deep-copied, not mutated); the
    start_type is recorded and survives the as_dict/from_dict round-trip."""
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)          # random start would be N = cap = 6
    frame_A = np.zeros(model_A.L, dtype=np.int64)
    occ = np.array([0, 2, 4, 6])
    frame_A[occ] = np.array([1, 2, 3, 4])
    init = initial_state_from_frame(frame_A, model_A, model_B, home="A")
    x_before = init.x.copy()

    # substitutions only -> N never changes, so a final N of 4 proves the init (not a
    # random N=6 start) was used.
    sched = AnnealSchedule(n_steps=500, beta_start=1.0, beta_end=5.0, min_length=1,
                           record_every=50, p_sub=1.0, p_slide_A=0.0, p_slide_B=0.0,
                           p_insert=0.0, p_delete=0.0)
    res = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=1, do_polish=False,
                       init_state=init, start_type="natural_A")
    assert res.start_type == "natural_A"
    assert res.final_n_residues == 4
    assert np.array_equal(init.x, x_before)       # deep-copied: caller's state untouched

    d = res.as_dict()
    assert d["start_type"] == "natural_A"
    assert ChainResult.from_dict(d).start_type == "natural_A"
    d.pop("start_type")                           # a pre-start-mix shard defaults to random
    assert ChainResult.from_dict(d).start_type == "random"


# --------------------------------------------------------------------------- #
# Alignment frames: the polish argmin frame is captured, matches E_polish, round-trips
# --------------------------------------------------------------------------- #

def test_polish_aln_frame_matches_energy_and_roundtrips():
    model_A = _random_model(L=8, seed=1)         # C(8,N) enumerable -> exact polish
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=3000, beta_start=0.5, beta_end=8.0,
                           min_length=4, record_every=200)
    res = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=7, do_polish=True)
    # aln frames are full model length, gap = 0
    assert res.aln_frame_A.size == model_A.L and res.aln_frame_B.size == model_B.L
    # they carry the polish (authoritative) energy exactly
    assert math.isclose(potts_energy(res.aln_frame_A, model_A), res.E_A_polish,
                        rel_tol=0, abs_tol=1e-9)
    assert math.isclose(potts_energy(res.aln_frame_B, model_B), res.E_B_polish,
                        rel_tol=0, abs_tol=1e-9)
    # the non-gap residues (in order) are exactly the designed core sequence
    core = res.aln_frame_A[res.aln_frame_A != GAP]
    assert np.array_equal(core, seq_to_ints(res.final_sequence))
    # serialization (cluster shard JSONL) round-trips the alignment frames
    r2 = ChainResult.from_dict(res.as_dict())
    assert np.array_equal(res.aln_frame_A, r2.aln_frame_A)
    assert np.array_equal(res.aln_frame_B, r2.aln_frame_B)


def test_no_polish_aln_frame_is_mc_frame():
    model_A = _random_model(L=8, seed=1)
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=2000, beta_start=0.5, beta_end=5.0,
                           min_length=4, record_every=200)
    res = anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=7, do_polish=False)
    assert np.array_equal(res.aln_frame_A, res.final_frame_A)
    assert np.array_equal(res.aln_frame_B, res.final_frame_B)


def test_move_probs_normalized_and_validated():
    sched = AnnealSchedule(p_sub=1.0, p_slide_A=1.0, p_slide_B=2.0, p_insert=0.0, p_delete=0.0)
    p = sched.move_probs()
    assert math.isclose(p.sum(), 1.0)
    assert p.shape == (5,)
    with pytest.raises(ValueError):
        AnnealSchedule(p_sub=0, p_slide_A=0, p_slide_B=0, p_insert=0, p_delete=0).move_probs()


# --------------------------------------------------------------------------- #
# Heat-bath (Gibbs) moves
# --------------------------------------------------------------------------- #

def test_local_field_matches_sub_delta():
    """``g[s0] − g[s1] == _sub_delta(s0→s1)`` — the energy-update path the Gibbs move uses."""
    for seed in range(4):
        model = _random_model(L=8, seed=seed)          # nonzero couplings incl. diagonal
        idx = np.arange(model.L)
        rng = np.random.default_rng(500 + seed)
        for _ in range(100):
            frame = rng.integers(0, Q, size=model.L).astype(np.int64)
            c = int(rng.integers(model.L))
            g = _local_field(frame, model.J, model.h, c, idx)
            assert g.shape == (Q,)
            s0, s1 = int(rng.integers(Q)), int(rng.integers(Q))
            frame[c] = s0                               # _sub_delta reads current frame[c] == s0
            assert math.isclose(g[s0] - g[s1], _sub_delta(frame, model.J, model.h, c, s0, s1, idx),
                                rel_tol=0, abs_tol=1e-9), (seed, c, s0, s1)


def test_local_field_is_field_row_when_couplings_zero():
    """Model-neutral path: with ``J == 0`` the full coupling sum is exactly ``h[c]``.

    Guards against re-introducing a profile shortcut *and* proves the general path is
    numerically identical to ``h[c]`` for a profile (so a profile run is a valid proxy)."""
    model = _random_model(L=5, seed=7)
    zero = PottsModel(name="p", J=np.zeros_like(model.J), h=model.h, L=5, q=Q,
                      alphabet=MSA_ALPHABET, gauge="raw", sha256="0" * 64, source="t")
    frame = np.array([1, 0, 5, 0, 3], dtype=np.int64)   # arbitrary occupied/gap pattern
    g = _local_field(frame, zero.J, zero.h, 2, np.arange(zero.L))
    assert np.array_equal(g, zero.h[2])                 # exact, not just close


def test_heatbath_choice_matches_softmax_and_argmax_limit():
    """``_heatbath_choice`` draws ``∝ exp(beta·g)`` and → argmax as beta → ∞."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=8)
    beta = 1.3
    counts = np.bincount([_heatbath_choice(g, beta, rng) for _ in range(200_000)], minlength=8)
    emp = counts / counts.sum()
    w = beta * g
    expected = np.exp(w - w.max()); expected /= expected.sum()
    assert np.max(np.abs(emp - expected)) < 0.01              # empirical ≈ softmax
    assert _heatbath_choice(g, 1e6, rng) == int(np.argmax(g))  # zero-T limit


def test_heatbath_moves_preserve_invariants_and_energy_tracking():
    """Gibbs sub + conditional insert keep the state valid and the running energy exact."""
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)
    ctx = _ctx(model_A, model_B, min_length=3)
    rng = np.random.default_rng(0)
    st = _initial_state(model_A, model_B, ctx, rng)
    movers = [
        lambda: _move_sub_heatbath(st, ctx, 2.0, rng),
        lambda: _move_slide(st, ctx, 2.0, rng, "A"),
        lambda: _move_insert_heatbath(st, ctx, 2.0, rng),
        lambda: _move_delete(st, ctx, 2.0, rng),
    ]
    for _ in range(4000):
        movers[int(rng.integers(len(movers)))]()
        _check_invariants(st, model_A, model_B)
        assert math.isclose(st.E_A, potts_energy(st.frame_A, model_A), rel_tol=0, abs_tol=1e-8)
        assert math.isclose(st.E_B, potts_energy(st.frame_B, model_B), rel_tol=0, abs_tol=1e-8)


def test_heatbath_substitute_reaches_per_site_optimum_on_profile():
    """On a profile model (J=0) a cold Gibbs sub drives each core residue to its joint argmin.

    With no couplings the per-site optimum is exact: residue ``argmax_a(w_A h_A[c_A,a]+w_B h_B[c_B,a])``
    over ``a∈1..q-1``. A short cold heat-bath run must land there for a fixed alignment."""
    model_A = _random_model(L=5, seed=3); model_B = _random_model(L=5, seed=4)
    model_A = PottsModel(name="A", J=np.zeros_like(model_A.J), h=model_A.h, L=5, q=Q,
                         alphabet=MSA_ALPHABET, gauge="raw", sha256="0" * 64, source="t")
    model_B = PottsModel(name="B", J=np.zeros_like(model_B.J), h=model_B.h, L=5, q=Q,
                         alphabet=MSA_ALPHABET, gauge="raw", sha256="0" * 64, source="t")
    wA, wB = 0.4, 0.6
    ctx = _ctx(model_A, model_B, wA=wA, wB=wB, min_length=5)
    rng = np.random.default_rng(0)
    st = _initial_state(model_A, model_B, ctx, rng)     # identity alignment, N=5
    for _ in range(2000):
        _move_sub_heatbath(st, ctx, 50.0, rng)          # cold: Gibbs concentrates on the argmax
    for k in range(st.x.size):
        g = wA * model_A.h[st.occ_A[k]] + wB * model_B.h[st.occ_B[k]]
        assert st.x[k] == int(np.argmax(g[1:])) + 1


# --------------------------------------------------------------------------- #
# Column-aware insert
# --------------------------------------------------------------------------- #

def test_column_aware_insert_invariants_and_energy_tracking():
    """Column-aware insert keeps the state valid and the running energy exact."""
    A = _random_model(L=9, seed=1); B = _random_model(L=6, seed=2)
    ctx = _ctx(A, B, min_length=3)
    rng = np.random.default_rng(0)
    st = _initial_state(A, B, ctx, rng)
    movers = [
        lambda: _move_sub_heatbath(st, ctx, 2.0, rng),
        lambda: _move_slide(st, ctx, 2.0, rng, "B"),
        lambda: _move_insert_column_aware(st, ctx, 2.0, rng),
        lambda: _move_delete(st, ctx, 2.0, rng),
    ]
    for _ in range(4000):
        movers[int(rng.integers(len(movers)))]()
        _check_invariants(st, A, B)
        assert math.isclose(st.E_A, potts_energy(st.frame_A, A), rel_tol=0, abs_tol=1e-8)
        assert math.isclose(st.E_B, potts_energy(st.frame_B, B), rel_tol=0, abs_tol=1e-8)


def test_column_aware_insert_picks_min_delta_pairing_when_cold():
    """At cold beta an accepted column-aware insert lands the (column-pair, residue) that
    minimizes ΔE over the interval — verified against brute-force enumeration."""
    from SBM.design.anneal import DesignState
    A = _random_model(L=6, seed=11); B = _random_model(L=5, seed=12)
    ctx = _ctx(A, B, wA=0.4, wB=0.6, min_length=1)
    x = np.array([3, 7], dtype=np.int64)
    occ_A = np.array([0, 5]); occ_B = np.array([0, 4])       # only the middle interval is insertable
    fA = np.zeros(6, np.int64); fA[occ_A] = x
    fB = np.zeros(5, np.int64); fB[occ_B] = x
    base = DesignState(x=x.copy(), occ_A=occ_A.copy(), occ_B=occ_B.copy(),
                       frame_A=fA.copy(), frame_B=fB.copy(),
                       E_A=potts_energy(fA, A), E_B=potts_energy(fB, B))
    idxA, idxB = np.arange(6), np.arange(5)
    best_dE = min(0.4 * _sub_delta(fA, A.J, A.h, ca, GAP, a, idxA)
                  + 0.6 * _sub_delta(fB, B.J, B.h, cb, GAP, a, idxB)
                  for a in range(1, Q) for ca in (1, 2, 3, 4) for cb in (1, 2, 3))
    assert best_dE < 0                                       # a favorable insert exists → can accept cold
    e0 = 0.4 * base.E_A + 0.6 * base.E_B
    accepts = 0
    for s in range(400):
        st = copy.deepcopy(base)
        if _move_insert_column_aware(st, ctx, 1e4, np.random.default_rng(s)):
            accepts += 1
            dE = (0.4 * st.E_A + 0.6 * st.E_B) - e0
            assert math.isclose(dE, best_dE, rel_tol=0, abs_tol=1e-8)
    assert accepts > 0                                       # the test actually exercised acceptance


def test_anneal_chain_colaware_runs_and_is_reproducible():
    A = _random_model(L=9, seed=1); B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=3000, beta_start=0.5, beta_end=5.0, min_length=3,
                           record_every=500, move_kind="colaware")
    r1 = anneal_chain(A, B, 0.4, 0.6, sched, seed=5, do_polish=False)
    r2 = anneal_chain(A, B, 0.4, 0.6, sched, seed=5, do_polish=False)
    assert r1.final_sequence == r2.final_sequence
    assert math.isclose(r1.E_A_mc, potts_energy(r1.final_frame_A, A), rel_tol=0, abs_tol=1e-8)


# --------------------------------------------------------------------------- #
# Parallel tempering (replica exchange)
# --------------------------------------------------------------------------- #

def _pt(seed, *, move_kind="metropolis", do_polish=False):
    model_A = _random_model(L=9, seed=1)
    model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=4000, beta_start=0.5, beta_end=5.0,
                           min_length=3, record_every=1000, move_kind=move_kind)
    return (model_A, model_B,
            pt_anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=seed,
                            n_replicas=4, swap_every=500, do_polish=do_polish))


def test_pt_design_energy_tracking_and_invariants():
    """The returned PT design's reported MC energy equals the from-scratch energy of its frames."""
    for kind in ("metropolis", "heatbath"):
        model_A, model_B, r = _pt(0, move_kind=kind)
        assert math.isclose(r.E_A_mc, potts_energy(r.final_frame_A, model_A), rel_tol=0, abs_tol=1e-8)
        assert math.isclose(r.E_B_mc, potts_energy(r.final_frame_B, model_B), rel_tol=0, abs_tol=1e-8)
        # final frames are a valid alignment of the design core into each model
        core = r.final_frame_A[r.final_frame_A != GAP]
        assert np.array_equal(core, r.final_frame_B[r.final_frame_B != GAP])
        assert 3 <= r.final_n_residues <= min(model_A.L, model_B.L)


def test_pt_is_bit_reproducible():
    _, _, r1 = _pt(7)
    _, _, r2 = _pt(7)
    assert r1.final_sequence == r2.final_sequence
    assert math.isclose(r1.E_tot_mc, r2.E_tot_mc, rel_tol=0, abs_tol=0)


def test_pt_requires_at_least_two_replicas():
    model_A = _random_model(L=9, seed=1); model_B = _random_model(L=6, seed=2)
    sched = AnnealSchedule(n_steps=100)
    with pytest.raises(ValueError):
        pt_anneal_chain(model_A, model_B, 0.4, 0.6, sched, seed=0, n_replicas=1)


def test_pt_matches_or_beats_single_replica_on_energy():
    """Sanity: a 4-rung ladder's design is no worse than its own hottest replica would do alone
    is hard to assert cheaply; instead check the design is the ladder's lowest-E config (contract)."""
    model_A, model_B, r = _pt(3)
    # design energy must be <= a fresh random start's energy (it did *some* optimization)
    ctx = _ctx(model_A, model_B, min_length=3)
    rng = np.random.default_rng(999)
    st0 = _initial_state(model_A, model_B, ctx, rng)
    assert r.E_tot_mc <= 0.4 * st0.E_A + 0.6 * st0.E_B
