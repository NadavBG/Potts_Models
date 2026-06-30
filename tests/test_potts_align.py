"""Tests for the couplings-aware Potts-energy aligner (iter-003 §10.20).

The gold-standard check mirrors ``test_energy.py``'s brute-force DP anchor: on a
tiny model whose whole alignment space is enumerable, the multi-restart SA must
find the *exact* global minimum that :func:`enumerate_align` computes. The
incremental two-column ΔE used in the SA inner loop is checked against a
from-scratch ``compute_energies`` recompute.

Run inside the project venv (imports ``SBM.utils.utils``).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from SBM.energy.encoding import GAP, Q
from SBM.energy.model import PottsModel
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align import (
    EnumerationInfeasible,
    PTSchedule,
    SASchedule,
    _move_delta,
    enumerate_align,
    perturb_frame,
    potts_align,
    pt_align,
    sa_align,
)
from SBM.utils.utils import MSA_ALPHABET, Zero_Sum_Gauge


def make_model(L, *, seed, scale_h=1.0, scale_J=0.5, q=Q, name="test"):
    """A random, zero-sum-gauged PottsModel of length L (mirrors test_energy)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(scale=scale_h, size=(L, q))
    J = rng.normal(scale=scale_J, size=(L, L, q, q))
    J = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))
    for i in range(L):
        J[i, i] = 0.0
    J_zg, h_zg = Zero_Sum_Gauge(J, h)
    return PottsModel(name=name, J=J_zg, h=h_zg, L=L, q=q, alphabet=MSA_ALPHABET,
                      gauge="zero_sum", sha256="0" * 64, source="<memory>")


def brute_force_min(query, model):
    """Exact min energy + frame over all monotone insert-free frames (independent)."""
    L, N = model.L, query.size
    best_e, best_f = math.inf, None
    for occ in itertools.combinations(range(L), N):
        frame = np.zeros(L, dtype=np.int64)
        frame[list(occ)] = query
        e = potts_energy(frame, model)
        if e < best_e:
            best_e, best_f = e, frame
    return best_e, best_f


# --------------------------------------------------------------------------- #

def test_enumerate_matches_independent_brute_force():
    model = make_model(7, seed=1)
    query = np.array([3, 7, 11, 19], dtype=np.int64)  # N=4 residues, C(7,4)=35 frames
    res = enumerate_align(query, model)
    bf_e, bf_frame = brute_force_min(query, model)
    assert res.is_global_exact and res.method == "enumerate"
    assert res.n_frames_evaluated == math.comb(7, 4) == 35
    assert math.isclose(res.best_energy, bf_e, rel_tol=0, abs_tol=1e-9)
    assert np.array_equal(res.best_frame, bf_frame)


def test_sa_finds_the_global_minimum():
    """The load-bearing check: SA reaches the exact enumerated global min."""
    model = make_model(8, seed=2)
    query = np.array([2, 5, 9, 14, 18], dtype=np.int64)  # N=5, C(8,5)=56 frames
    exact = enumerate_align(query, model)
    sa = sa_align(query, model, seed=0,
                  schedule=SASchedule(n_restarts=12, n_steps=800))
    assert sa.method == "sa" and not sa.is_global_exact
    assert math.isclose(sa.best_energy, exact.best_energy, rel_tol=0, abs_tol=1e-9)
    assert np.array_equal(sa.best_frame, exact.best_frame)


def test_pt_finds_the_global_minimum():
    """Parallel tempering reaches the exact enumerated global min on a tiny model."""
    model = make_model(8, seed=2)
    query = np.array([2, 5, 9, 14, 18], dtype=np.int64)  # N=5, C(8,5)=56 frames
    exact = enumerate_align(query, model)
    pt = pt_align(query, model, seed=0,
                  schedule=PTSchedule(n_replicas=6, n_blocks=200, n_restarts=2))
    assert pt.method == "pt" and not pt.is_global_exact
    assert math.isclose(pt.best_energy, exact.best_energy, rel_tol=0, abs_tol=1e-9)
    assert np.array_equal(pt.best_frame, exact.best_frame)


def test_pt_schedule_for_gap_count_escalates():
    from SBM.energy.potts_align import PTSchedule
    easy = PTSchedule.for_gap_count(8)
    hard = PTSchedule.for_gap_count(13)
    assert easy == PTSchedule()                  # moderate g → default
    assert hard == PTSchedule.thorough()         # g≥13 → heavier budget
    assert hard.n_blocks > easy.n_blocks and hard.n_restarts >= easy.n_restarts
    assert easy.teleport_frac == hard.teleport_frac == 0.3


def test_pt_is_deterministic_and_requires_seed():
    model = make_model(9, seed=4)
    q = np.array([3, 7, 10, 14, 19, 2], dtype=np.int64)
    sched = PTSchedule(n_replicas=4, n_blocks=50, n_restarts=2)
    a = pt_align(q, model, seed=11, schedule=sched)
    b = pt_align(q, model, seed=11, schedule=sched)
    assert np.array_equal(a.best_frame, b.best_frame) and a.best_energy == b.best_energy
    with pytest.raises(ValueError):
        pt_align(q, model, seed=None, schedule=sched)


def test_pt_rejects_more_warm_starts_than_replicas():
    """More init_frames than replicas would silently drop the lowest-index (most
    relevant) frames in _pt_one; pt_align must refuse loudly instead."""
    model = make_model(9, seed=4)
    q = np.array([3, 7, 10, 14, 19, 2], dtype=np.int64)
    sched = PTSchedule(n_replicas=2, n_blocks=20, n_restarts=1)
    frames = [enumerate_align(q, model).best_frame for _ in range(3)]  # 3 > 2 replicas
    with pytest.raises(ValueError, match="init_frames"):
        pt_align(q, model, seed=0, schedule=sched, init_frames=frames)


def test_incremental_delta_matches_full_recompute():
    """ΔE accumulator vs from-scratch recompute, for BOTH the ±1 local move and the
    non-local teleport move (|dst-src|>1). The teleport path is the newest code and
    the one most prone to a pair/self-term double-counting bug, so it must be checked
    at long jumps, not only adjacent shifts (a wider L leaves room for big gaps)."""
    model = make_model(14, seed=3)
    q = np.array([4, 8, 12, 16, 20, 1, 6], dtype=np.int64)  # N=7, L=14 → wide gaps
    rng = np.random.default_rng(7)
    idx = np.arange(model.L)
    checks = teleport_checks = 0
    for _ in range(400):
        cols = np.sort(rng.choice(model.L, size=q.size, replace=False))
        r = int(rng.integers(q.size))
        lo = int(cols[r - 1]) if r > 0 else -1          # exclusive lower bound
        hi = int(cols[r + 1]) if r < q.size - 1 else model.L  # exclusive upper bound
        # destination = any empty column strictly between the neighbours (the teleport
        # move set); ±1 is the special case where the chosen column is adjacent.
        candidates = [c for c in range(lo + 1, hi) if c != int(cols[r])]
        if not candidates:
            continue
        dst = int(rng.choice(candidates))
        frame = np.zeros(model.L, dtype=np.int64)
        frame[cols] = q
        before = potts_energy(frame, model)
        de = _move_delta(frame, model.J, model.h, int(cols[r]), dst, int(q[r]), idx)
        after_frame = frame.copy()
        after_frame[int(cols[r])] = GAP
        after_frame[dst] = int(q[r])
        after = potts_energy(after_frame, model)
        assert math.isclose(de, after - before, rel_tol=0, abs_tol=1e-9)
        checks += 1
        if abs(dst - int(cols[r])) > 1:
            teleport_checks += 1
    assert checks > 50           # exercised enough legal moves
    assert teleport_checks > 20  # and enough of them were teleport-distance jumps


def test_sa_is_deterministic():
    model = make_model(9, seed=4)
    q = np.array([3, 7, 10, 14, 19, 2], dtype=np.int64)
    sched = SASchedule(n_restarts=6, n_steps=300)
    a = sa_align(q, model, seed=11, schedule=sched)
    b = sa_align(q, model, seed=11, schedule=sched)
    assert np.array_equal(a.best_frame, b.best_frame)
    assert a.best_energy == b.best_energy


def test_dispatch_enumerate_vs_approx():
    model = make_model(20, seed=5)
    sched = SASchedule(enum_max_frames=100, n_restarts=2, n_steps=50)
    small = potts_align(np.array([5], dtype=np.int64), model, seed=0, schedule=sched)
    assert small.method == "enumerate" and small.is_global_exact  # C(20,1)=20
    big_pt = potts_align(np.array([5, 9], dtype=np.int64), model, seed=0, schedule=sched,
                         pt_schedule=PTSchedule(n_replicas=4, n_blocks=20))
    assert big_pt.method == "pt"  # C(20,2)=190 > 100; default fallback is PT
    big_sa = potts_align(np.array([5, 9], dtype=np.int64), model, seed=0, schedule=sched,
                         fallback="sa")
    assert big_sa.method == "sa"


def test_output_invariants():
    model = make_model(8, seed=6)
    q = np.array([2, 6, 11, 15], dtype=np.int64)
    for res in (enumerate_align(q, model),
                sa_align(q, model, seed=1, schedule=SASchedule(n_restarts=4, n_steps=200))):
        frame = res.best_frame
        assert frame.shape == (model.L,)
        assert int(np.count_nonzero(frame != GAP)) == q.size
        assert np.array_equal(frame[frame != GAP], q)  # residues in order
        assert frame.min() >= 0 and frame.max() < model.q


def test_topk_sorted_and_distinct():
    model = make_model(9, seed=7)
    q = np.array([3, 8, 12, 17, 1], dtype=np.int64)
    res = sa_align(q, model, seed=2, schedule=SASchedule(n_restarts=10, n_steps=300, topk=4))
    assert len(res.topk_frames) == len(res.topk_energies) <= 4
    assert res.topk_energies == sorted(res.topk_energies)
    keys = {f.tobytes() for f in res.topk_frames}
    assert len(keys) == len(res.topk_frames)  # distinct
    assert math.isclose(res.topk_energies[0], res.best_energy, rel_tol=0, abs_tol=1e-9)


def test_enumeration_infeasible_raises():
    model = make_model(40, seed=8)
    q = np.arange(1, 21, dtype=np.int64)  # N=20, C(40,20) ~ 1.4e11
    with pytest.raises(EnumerationInfeasible):
        enumerate_align(q, model, max_frames=10_000)


def test_guards():
    model = make_model(8, seed=9)
    with pytest.raises(ValueError):  # gap (0) in a raw query
        enumerate_align(np.array([3, 0, 5], dtype=np.int64), model)
    with pytest.raises(ValueError):  # N > L
        sa_align(np.arange(1, 11, dtype=np.int64), make_model(5, seed=9), seed=0)
    with pytest.raises(ValueError):  # missing seed
        sa_align(np.array([3, 5], dtype=np.int64), model, seed=None)


def test_perturb_frame():
    model = make_model(10, seed=10)
    q = np.array([4, 8, 12, 16, 1], dtype=np.int64)  # N=5, 5 gaps
    native = enumerate_align(q, model).best_frame
    rng = np.random.default_rng(0)
    for k in (1, 2, 4):
        pert = perturb_frame(native, k, rng=rng)
        assert pert.shape == native.shape
        assert int(np.count_nonzero(pert != GAP)) == q.size
        assert np.array_equal(pert[pert != GAP], q)  # residues preserved, in order
        assert not np.array_equal(pert, native)      # actually perturbed
    assert np.array_equal(perturb_frame(native, 0, rng=rng), native)  # k=0 is identity
    with pytest.raises(ValueError):
        perturb_frame(native, 99, rng=rng)  # k exceeds available columns
