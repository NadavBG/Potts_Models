"""Tests for the two-model joint-annealing design engine (``SBM.design.anneal``).

Pure numpy/scipy (no MCMC kernel). The load-bearing check is that every
incremental ΔE (``_sub_delta``, and the reused ``potts_align`` slide) agrees with
a *from-scratch* :func:`SBM.energy.potts.potts_energy` recompute — the same
reference-implementation discipline as ``tests/test_energy.py``. On top of that we
check the state invariants, the ``N ≤ min(L_A,L_B)`` cap, that the thermal
alignment converges to the enumerated argmin as ``T→0``, and reproducibility.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from SBM.design.anneal import (
    AnnealSchedule,
    ChainResult,
    _Ctx,
    _initial_state,
    _move_delete,
    _move_insert,
    _move_slide,
    _move_sub,
    _sub_delta,
    anneal_chain,
    initial_state_from_frame,
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
