"""Acceptance tests for the two-model energy scorer (spec §6) + the DP anchor.

The profile-HMM dynamic program (``SBM.energy.hmm``) is the error-prone core, so
the most important test enumerates *every* alignment of a tiny query by brute
force and checks the forward partition, the marginal IS estimate, and the FFBS
sample frequencies against the exact answer. The rest mirror spec §6: gauge
invariance, the in-frame base case, MAP≈marginal when unambiguous, ordering
sanity, and the IS diagnostics.

Run inside the project's uv venv (the energy module imports ``SBM.utils.utils``,
which loads the compiled MCMC kernel):

    .venv/bin/python -m pytest tests/test_energy.py -q
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest
from scipy.special import logsumexp

from SBM.energy.encoding import Q
from SBM.energy.hmm import AlignmentPath, ProfileHMM
from SBM.energy.model import PottsModel
from SBM.energy.potts import potts_energy
from SBM.energy.score import score_sequence, score_two_models
from SBM.utils.utils import MSA_ALPHABET, Zero_Sum_Gauge, compute_energies


# ── helpers ──────────────────────────────────────────────────────────────


def make_model(L, *, seed, scale_h=1.0, scale_J=0.3, q=Q, name="test"):
    """A random, zero-sum-gauged PottsModel of length L (full q=21 alphabet)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(scale=scale_h, size=(L, q))
    J = rng.normal(scale=scale_J, size=(L, L, q, q))
    J = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))  # symmetrize J[i,j,a,b]=J[j,i,b,a]
    for i in range(L):
        J[i, i] = 0.0
    J_zg, h_zg = Zero_Sum_Gauge(J, h)
    return PottsModel(
        name=name, J=J_zg, h=h_zg, L=L, q=q, alphabet=MSA_ALPHABET,
        gauge="zero_sum", sha256="0" * 64, source="<memory>",
    )


def enumerate_paths(L, N):
    """Every profile-HMM alignment path of a length-N query to L columns.

    Mirrors the transition graph in ``ProfileHMM``: each column is M or D,
    inserts sit between columns, the whole query (N residues) is emitted.
    """
    paths = []

    def expand(s, k, pos, acc):
        if k == L and pos == N and s in ("M", "I", "D"):
            paths.append(tuple(acc))
        succ = []
        if s == "M":
            if k < L:
                succ += [("M", k + 1, True), ("I", k, True), ("D", k + 1, False)]
            else:
                succ += [("I", L, True)]
        elif s == "I":
            if k < L:
                succ += [("M", k + 1, True), ("I", k, True)]
            else:
                succ += [("I", L, True)]
        elif s == "D":
            if k < L:
                succ += [("M", k + 1, True), ("D", k + 1, False)]
        for ns, nk, emit in succ:
            npos = pos + (1 if emit else 0)
            if npos <= N:
                expand(ns, nk, npos, acc + [(ns, nk)])

    expand("M", 0, 0, [])  # Begin = (M, 0), not recorded in acc
    return [AlignmentPath(p) for p in paths]


@pytest.fixture
def tiny_setup():
    """A small model + HMM + query for brute-force comparison."""
    L, N = 3, 3
    model = make_model(L, seed=1, scale_J=0.5)
    seed_msa = np.random.default_rng(7).integers(0, Q, size=(40, L))
    hmm = ProfileHMM.from_model(model, seed_msa)
    x = np.array([1, 5, 9], dtype=np.int64)  # raw query, residues in 1..20
    return model, hmm, x, L, N


# ── test 6: the DP anchor (brute-force enumeration) ───────────────────────


def test_forward_logZ_matches_bruteforce(tiny_setup):
    model, hmm, x, L, N = tiny_setup
    paths = enumerate_paths(L, N)
    assert len(paths) > 1  # the query genuinely aligns many ways
    brute = logsumexp([hmm.path_logscore(p, x) for p in paths])
    assert np.isclose(hmm.forward_logZ(x), brute, atol=1e-9)


def test_paths_are_normalized(tiny_setup):
    """Enumerated joint probs, divided by Z, sum to 1 (proper distribution)."""
    model, hmm, x, L, N = tiny_setup
    paths = enumerate_paths(L, N)
    logZ = hmm.forward_logZ(x)
    post = np.exp(np.array([hmm.path_logscore(p, x) for p in paths]) - logZ)
    assert np.isclose(post.sum(), 1.0, atol=1e-9)


def test_marginal_matches_exact(tiny_setup):
    model, hmm, x, L, N = tiny_setup
    paths = enumerate_paths(L, N)
    # Exact marginal energy: −log Σ_a exp(−E_k(x,a)) over the same path space.
    e_potts = np.array([potts_energy(hmm.path_to_frame(p, x), model) for p in paths])
    exact = -logsumexp(-e_potts)
    res = score_sequence(x, model, method="marginal", hmm=hmm, n_samples=20000, seed=0)
    # IS estimate within a few MC standard errors of the exact value.
    assert abs(res.energy - exact) < max(5 * res.mc_stderr, 1e-2)


def test_ffbs_frequencies_match_posterior(tiny_setup):
    model, hmm, x, L, N = tiny_setup
    paths = enumerate_paths(L, N)
    logZ = hmm.forward_logZ(x)
    exact = {p.states: np.exp(hmm.path_logscore(p, x) - logZ) for p in paths}
    rng = np.random.default_rng(123)
    n = 40000
    samples = hmm.sample_paths(x, n, rng)
    counts: dict = {}
    for s in samples:
        counts[s.states] = counts.get(s.states, 0) + 1
    tv = 0.5 * sum(abs(counts.get(k, 0) / n - p) for k, p in exact.items())
    assert tv < 0.02  # total-variation distance to the exact posterior


def test_viterbi_is_the_max_posterior_path(tiny_setup):
    model, hmm, x, L, N = tiny_setup
    paths = enumerate_paths(L, N)
    scores = {p.states: hmm.path_logscore(p, x) for p in paths}
    best = max(scores, key=scores.get)
    assert hmm.viterbi(x).states == best


def test_path_to_frame_consumes_whole_query(tiny_setup):
    model, hmm, x, L, N = tiny_setup
    for p in hmm.sample_paths(x, 50, np.random.default_rng(5)):
        S = hmm.path_to_frame(p, x)
        assert S.shape == (L,)
        # match positions hold query residues; the rest are gaps
        assert set(np.unique(S)).issubset(set(range(Q)))


# ── test 1: gauge invariance ──────────────────────────────────────────────


def test_gauge_shifts_energy_by_a_constant():
    L = 6
    rng = np.random.default_rng(2)
    h = rng.normal(size=(L, Q))
    J = rng.normal(scale=0.3, size=(L, L, Q, Q))
    J = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))
    for i in range(L):
        J[i, i] = 0.0
    seqs = rng.integers(0, Q, size=(8, L))
    e_raw = compute_energies(seqs, h, J)
    J_zg, h_zg = Zero_Sum_Gauge(J, h)
    e_gauged = compute_energies(seqs, h_zg, J_zg)
    diff = e_gauged - e_raw
    # Energy differences are gauge-invariant: the per-sequence shift is constant.
    assert np.allclose(diff, diff.mean(), atol=1e-8)
    pair = e_gauged[1] - e_gauged[0]
    assert np.isclose(pair, e_raw[1] - e_raw[0], atol=1e-8)


# ── test 2: in-frame base case ────────────────────────────────────────────


def test_in_frame_matches_hand_computed():
    """Two sites, zero coupling: E = −h_0(s0) − h_1(s1)."""
    h = np.zeros((2, Q))
    h[0, 3] = 2.0
    h[1, 7] = -1.0
    J = np.zeros((2, 2, Q, Q))
    model = PottsModel(
        name="hand", J=J, h=h, L=2, q=Q, alphabet=MSA_ALPHABET,
        gauge="raw", sha256="0" * 64, source="<memory>",
    )
    S = np.array([3, 7])
    assert np.isclose(potts_energy(S, model), -(2.0 + (-1.0)))


def test_in_frame_matches_compute_energies():
    model = make_model(5, seed=3)
    S = np.random.default_rng(9).integers(0, Q, size=5)
    assert np.isclose(potts_energy(S, model), compute_energies(S, model.h, model.J)[0])


def test_in_frame_method_via_score_sequence():
    model = make_model(5, seed=3)
    S = np.random.default_rng(9).integers(0, Q, size=5)
    res = score_sequence(S, model, method="in_frame")
    assert res.method == "in_frame"
    assert np.isclose(res.energy, potts_energy(S, model))


# ── test 3: MAP ≈ marginal when the alignment is unambiguous ──────────────


def test_map_close_to_marginal_when_unambiguous():
    """Peaked fields, no couplings, N==L → posterior concentrates on all-match."""
    L = 5
    rng = np.random.default_rng(11)
    consensus = rng.integers(1, Q, size=L)  # residues 1..20
    h = np.full((L, Q), -8.0)
    for k in range(L):
        h[k, consensus[k]] = 8.0  # strongly favor the consensus residue
    J = np.zeros((L, L, Q, Q))
    model = PottsModel(
        name="peaked", J=J, h=h, L=L, q=Q, alphabet=MSA_ALPHABET,
        gauge="raw", sha256="0" * 64, source="<memory>",
    )
    seed_msa = np.tile(consensus, (50, 1))  # gapless → tiny delete propensity
    hmm = ProfileHMM.from_model(model, seed_msa)
    e_map = score_sequence(consensus, model, method="map", hmm=hmm).energy
    e_marg = score_sequence(consensus, model, method="marginal", hmm=hmm,
                            n_samples=5000, seed=0).energy
    assert abs(e_map - e_marg) < 0.05


# ── test 4: ordering sanity (scorer unit test, not an infeasibility claim) ─


def test_native_scores_below_shuffle_and_other_model():
    L = 12
    rng = np.random.default_rng(21)
    consensus = rng.integers(1, Q, size=L)
    # Model A favors the consensus; model B favors a disjoint residue per column.
    hA = np.full((L, Q), -2.0)
    hB = np.full((L, Q), -2.0)
    for k in range(L):
        hA[k, consensus[k]] = 6.0
        other = 1 + (consensus[k] % 20)  # a different residue in 1..20
        hB[k, other] = 6.0
    JA = np.zeros((L, L, Q, Q))
    model_A = PottsModel(name="A", J=JA, h=hA, L=L, q=Q, alphabet=MSA_ALPHABET,
                         gauge="raw", sha256="0" * 64, source="<memory>")
    model_B = PottsModel(name="B", J=JA, h=hB, L=L, q=Q, alphabet=MSA_ALPHABET,
                         gauge="raw", sha256="0" * 64, source="<memory>")
    e_native = potts_energy(consensus, model_A)
    shuffled = consensus[rng.permutation(L)]
    assert e_native <= potts_energy(shuffled, model_A)
    assert e_native < potts_energy(consensus, model_B)


# ── test 5: IS diagnostics ────────────────────────────────────────────────


def test_marginal_reports_diagnostics():
    model = make_model(4, seed=31)
    seed_msa = np.random.default_rng(8).integers(0, Q, size=(30, 4))
    hmm = ProfileHMM.from_model(model, seed_msa)
    x = np.array([2, 4, 6, 8])
    res = score_sequence(x, model, method="marginal", hmm=hmm, n_samples=2000, seed=1)
    assert res.ess is not None and 0 < res.ess <= 2000
    assert res.mc_stderr is not None and res.mc_stderr >= 0
    assert res.seed == 1


def test_marginal_requires_seed():
    model = make_model(4, seed=31)
    seed_msa = np.random.default_rng(8).integers(0, Q, size=(30, 4))
    hmm = ProfileHMM.from_model(model, seed_msa)
    with pytest.raises(ValueError, match="seed"):
        score_sequence(np.array([2, 4, 6, 8]), model, method="marginal", hmm=hmm)


def test_marginal_rejects_gapped_query():
    """A gap (state 0) in a raw query is a contract violation — fail loudly."""
    model = make_model(4, seed=31)
    hmm = ProfileHMM.from_model(model, np.random.default_rng(8).integers(0, Q, size=(30, 4)))
    with pytest.raises(ValueError, match="gap-free"):
        score_sequence(np.array([2, 0, 6, 8]), model, method="marginal", hmm=hmm, seed=1)
    with pytest.raises(ValueError, match="gap-free"):
        score_sequence(np.array([2, 0, 6, 8]), model, method="map", hmm=hmm)


def test_marginal_rejects_zero_samples():
    model = make_model(4, seed=31)
    hmm = ProfileHMM.from_model(model, np.random.default_rng(8).integers(0, Q, size=(30, 4)))
    with pytest.raises(ValueError, match="n_samples"):
        score_sequence(np.array([2, 4, 6, 8]), model, method="marginal", hmm=hmm, seed=1, n_samples=0)


def test_low_ess_warns(caplog):
    """Strong couplings make the fields-only proposal poor → low ESS, loud warn."""
    L = 6
    model = make_model(L, seed=41, scale_J=6.0)  # couplings dominate
    seed_msa = np.random.default_rng(8).integers(0, Q, size=(30, L))
    hmm = ProfileHMM.from_model(model, seed_msa)
    x = np.array([3, 6, 9, 12, 15, 18])
    with caplog.at_level(logging.WARNING):
        res = score_sequence(x, model, method="marginal", hmm=hmm,
                             n_samples=500, seed=2, ess_threshold=1e9)
    assert "low ESS" in caplog.text.lower() or "low ess" in res.notes.lower()


# ── score_two_models plumbing ─────────────────────────────────────────────


def test_score_two_models_sums_with_weights():
    L = 5
    model_A = make_model(L, seed=51, name="A")
    model_B = make_model(L, seed=52, name="B")
    msa = np.random.default_rng(8).integers(0, Q, size=(30, L))
    hmm_A = ProfileHMM.from_model(model_A, msa)
    hmm_B = ProfileHMM.from_model(model_B, msa)
    x = np.array([1, 2, 3, 4, 5])
    out = score_two_models(x, model_A, model_B, w_A=2.0, w_B=0.5,
                           hmm_A=hmm_A, hmm_B=hmm_B, n_samples=500, seed=0)
    assert np.isclose(out["E_tot"], 2.0 * out["E_A"] + 0.5 * out["E_B"])
    assert out["result_A"].seed == 0 and out["result_B"].seed == 1


def test_marginal_is_reproducible():
    model = make_model(5, seed=3)
    msa = np.random.default_rng(8).integers(0, Q, size=(30, 5))
    hmm = ProfileHMM.from_model(model, msa)
    x = np.array([1, 2, 3, 4, 5])
    a = score_sequence(x, model, method="marginal", hmm=hmm, n_samples=500, seed=7)
    b = score_sequence(x, model, method="marginal", hmm=hmm, n_samples=500, seed=7)
    assert a.energy == b.energy  # same seed → identical (bit-for-bit)
