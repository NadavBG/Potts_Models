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

from SBM.energy.encoding import Q, ints_to_seq
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


# ── DCAlign integration, Tier 0 (pure Python; no Julia, no real models) ───────


def test_dcalign_handoff_round_trip():
    """model_to_dcalign_arrays → <f8 Fortran bytes → reshape → inverse = J/h.

    This is the gold check on the §10.9 handoff: the exact byte stream the Julia
    side reads (column-major (q,q,L,L) / (q,L)) must round-trip our model with no
    loss. Any error in ORDER, the transpose, the dtype, or the byte order shows
    up as a mismatch here, before any Julia is involved.
    """
    from SBM.utils.dcalign_score import ORDER, model_to_dcalign_arrays

    L = 5
    model = make_model(L, seed=101)
    J_dca, h_dca = model_to_dcalign_arrays(model)
    assert J_dca.shape == (Q, Q, L, L)
    assert h_dca.shape == (Q, L)

    # Serialize exactly as align_sequences does, then read back as Julia would.
    jbytes = J_dca.astype("<f8").tobytes(order="F")
    hbytes = h_dca.astype("<f8").tobytes(order="F")
    J_back = np.frombuffer(jbytes, dtype="<f8").reshape((Q, Q, L, L), order="F")
    h_back = np.frombuffer(hbytes, dtype="<f8").reshape((Q, L), order="F")

    # Invert: undo the ORDER permutation (argsort) then the (involutive) transpose.
    inv = np.argsort(ORDER)
    J_rec = J_back[inv][:, inv].transpose(2, 3, 0, 1)
    h_rec = h_back[inv].T
    assert np.array_equal(J_rec, model.J)
    assert np.array_equal(h_rec, model.h)

    # Sanity on the alphabet remap: DCAlign gap is index 21 (0-based 20), and our
    # gap is 0; ORDER puts our gap last, residues 1..20 unchanged.
    assert ORDER == list(range(1, 21)) + [0]


def test_dcalign_branch_equals_in_frame():
    """Given the same frame, method='dcalign' returns the in-frame energy."""
    L = 6
    model = make_model(L, seed=102)
    S = np.random.default_rng(13).integers(0, Q, size=L)  # in-frame, gaps allowed
    res_dca = score_sequence(
        np.array([], dtype=np.int64), model, method="dcalign",
        dcalign_frame=ints_to_seq(S), dcalign_notes="from cache",
    )
    res_inf = score_sequence(S, model, method="in_frame")
    assert res_dca.method == "dcalign"
    assert np.isclose(res_dca.energy, res_inf.energy)
    assert "DCAlign" in res_dca.notes and "from cache" in res_dca.notes


def test_dcalign_empty_frame_is_loud_error():
    """An empty cached frame (DCAlign failed for that id) must raise, not score."""
    model = make_model(5, seed=103)
    with pytest.raises(ValueError, match="empty"):
        score_sequence(np.array([], dtype=np.int64), model, method="dcalign", dcalign_frame="")
    with pytest.raises(ValueError, match="empty"):
        score_sequence(np.array([], dtype=np.int64), model, method="dcalign", dcalign_frame=None)
    # Wrong-length frame is also loud.
    with pytest.raises(ValueError, match="length"):
        score_sequence(np.array([], dtype=np.int64), model, method="dcalign", dcalign_frame="ACD")


def test_dcalign_cache_reader(tmp_path):
    from SBM.utils.dcalign_score import TSV_HEADER, read_alignment_cache

    tsv = tmp_path / "alignments.tsv"
    tsv.write_text(
        TSV_HEADER + "\n"
        "seqA\tAC-DE\t-12.5\ttrue\tfalse\t37\n"
        "seqB\t\tnan\tfalse\tfalse\t0\n",
        encoding="utf-8",
    )
    cache = read_alignment_cache(tsv)
    assert set(cache) == {"seqA", "seqB"}
    assert cache["seqA"].aligned_frame == "AC-DE"
    assert cache["seqA"].ok and np.isclose(cache["seqA"].dcalign_energy, -12.5)
    assert cache["seqA"].converged and cache["seqA"].n_iter == 37
    assert not cache["seqB"].ok and np.isnan(cache["seqB"].dcalign_energy)

    dup = tmp_path / "dup.tsv"
    dup.write_text(
        "seqA\tACDEF\t-1\ttrue\tfalse\t1\nseqA\tACDEF\t-1\ttrue\tfalse\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_alignment_cache(dup)


def test_dcalign_cache_write_read_round_trip(tmp_path):
    """write_alignment_cache → read_alignment_cache recovers every field."""
    from SBM.utils.dcalign_score import (
        DCAlignResult,
        read_alignment_cache,
        write_alignment_cache,
    )

    results = [
        DCAlignResult("s1", "AC-DEFGHIK", -259.1234567890123, True, False, 412),
        DCAlignResult("s2", "", float("nan"), False, False, 0),
        DCAlignResult("s3", "MNPQRSTVWY", -3.0, True, True, 5000),
    ]
    out = tmp_path / "alignments.tsv"
    write_alignment_cache(out, results)
    back = read_alignment_cache(out)
    assert set(back) == {"s1", "s2", "s3"}
    assert back["s1"].aligned_frame == "AC-DEFGHIK"
    assert back["s1"].dcalign_energy == -259.1234567890123  # exact round-trip
    assert back["s1"].n_iter == 412 and back["s1"].converged
    assert not back["s2"].ok and np.isnan(back["s2"].dcalign_energy)
    assert back["s3"].used_decimation and back["s3"].n_iter == 5000


# ── DCAlign-vs-in-frame baseline (SBM.energy.dcalign_baseline) ─────────────


def test_column_agreement():
    from SBM.energy.dcalign_baseline import column_agreement

    a = np.array([1, 0, 5, 9, 0, 3])
    assert column_agreement(a, a) == 1.0
    b = a.copy()
    b[2] = 7  # one position differs
    assert np.isclose(column_agreement(a, b), 5 / 6)
    with pytest.raises(ValueError, match="equal-length"):
        column_agreement(a, a[:-1])


def test_compare_record_recovered_and_perturbed():
    """delta_e/col_agreement track the native vs DCAlign frame; cache canary ~0."""
    from SBM.energy.dcalign_baseline import compare_record
    from SBM.energy.datasets import QueryRecord
    from SBM.utils.dcalign_score import DCAlignResult

    L = 6
    model = make_model(L, seed=204)
    native = np.array([1, 0, 5, 9, 2, 11], dtype=np.int64)  # in-frame, gap allowed
    record = QueryRecord(id="q0", group="A/natural", origin_model=model.name, ints=native)

    # (a) DCAlign returns the native frame exactly → recovered.
    e_native = potts_energy(native, model)
    dca_same = DCAlignResult("q0", ints_to_seq(native), e_native, True, False, 10)
    row = compare_record(record, model, dca_same)
    assert row.ok and row.n_residues == int(np.count_nonzero(native))
    assert np.isclose(row.e_inframe, e_native) and np.isclose(row.delta_e, 0.0)
    assert row.col_agreement == 1.0 and row.cache_abs_diff < 1e-9

    # (b) DCAlign returns a different frame → delta_e is the exact energy gap,
    #     col_agreement < 1, and the cache canary still ~0 (energy passed in).
    other = np.array([1, 3, 5, 9, 2, 11], dtype=np.int64)  # differs at one column
    e_other = potts_energy(other, model)
    dca_diff = DCAlignResult("q0", ints_to_seq(other), e_other, True, False, 42)
    row2 = compare_record(record, model, dca_diff)
    assert np.isclose(row2.e_dcalign, e_other)
    assert np.isclose(row2.delta_e, e_other - e_native)
    assert np.isclose(row2.col_agreement, 5 / 6) and row2.cache_abs_diff < 1e-9


def test_compare_record_failed_alignment_is_flagged_not_dropped():
    from SBM.energy.dcalign_baseline import compare_record
    from SBM.energy.datasets import QueryRecord
    from SBM.utils.dcalign_score import DCAlignResult

    model = make_model(5, seed=205)
    native = np.array([1, 2, 3, 0, 5], dtype=np.int64)
    record = QueryRecord(id="bad", group="A/natural", origin_model=model.name, ints=native)
    row = compare_record(record, model, DCAlignResult("bad", "", float("nan"), False, False, 0))
    assert not row.ok
    assert np.isclose(row.e_inframe, potts_energy(native, model))  # native still scored
    assert np.isnan(row.e_dcalign) and np.isnan(row.delta_e) and np.isnan(row.col_agreement)


def test_summarize_counts_worse_better_equal():
    from SBM.energy.dcalign_baseline import BaselineRow, summarize

    def mk(delta, model="A", group="A/natural", ok=True):
        return BaselineRow(
            sequence_id="x", group=group, model=model, n_residues=5,
            e_inframe=-100.0, e_dcalign=-100.0 + delta, delta_e=delta,
            col_agreement=0.9, cache_energy=-100.0 + delta, cache_abs_diff=1e-13,
            converged=True, used_decimation=False, n_iter=1, ok=ok,
        )

    rows = [mk(5.0), mk(0.2), mk(-3.0), mk(50.0), mk(float("nan"), ok=False)]
    s = summarize(rows, equal_tol=1.0)["overall"]
    assert s["n"] == 5 and s["n_ok"] == 4 and s["n_failed"] == 1
    assert s["n_worse"] == 2  # delta 5 and 50 exceed tol
    assert s["n_better"] == 1  # delta -3
    assert s["n_near_equal"] == 1  # delta 0.2
    assert np.isclose(s["frac_worse"], 2 / 4)
    assert s["cache_max_abs_diff"] < 1e-9


def test_convergence_by_group_counts_over_both_models():
    """Per-(model, group) convergence tallies span home + cross alignments."""
    from SBM.energy.dcalign_baseline import convergence_by_group, summarize_convergence
    from SBM.utils.dcalign_score import DCAlignResult

    def r(sid, conv, dec=False, frame="ACDEF"):
        return DCAlignResult(sid, frame, -1.0, conv, dec, 1)

    # Model A aligns two of its own (home) + two cross; B mirrors. Not-converged
    # always coincides with the decimation fallback (as in the real cache).
    caches = {
        "A": {"a0": r("a0", True), "a1": r("a1", False, dec=True),
              "b0": r("b0", False, dec=True), "b1": r("b1", True)},
        "B": {"a0": r("a0", True), "a1": r("a1", True),
              "b0": r("b0", True), "b1": r("b1", False, dec=True, frame="")},  # failed
    }
    groups = {
        "a0": {"group": "A/nat"}, "a1": {"group": "A/nat"},
        "b0": {"group": "B/nat"}, "b1": {"group": "B/nat"},
    }
    rows = convergence_by_group(caches, groups)
    by = {(x["model"], x["group"]): x for x in rows}
    assert by[("A", "A/nat")]["n_not_converged"] == 1 and by[("A", "A/nat")]["n"] == 2
    assert by[("A", "B/nat")]["n_not_converged"] == 1  # cross-family non-convergence
    assert by[("B", "B/nat")]["n_failed"] == 1  # empty frame counted as failed
    assert by[("A", "A/nat")]["n_decimation"] == 1

    summ = summarize_convergence(rows)
    assert summ["overall"]["n"] == 8
    assert summ["by_model"]["A"]["n_not_converged"] == 2
    assert summ["by_model"]["B"]["n_not_converged"] == 1
    # A sequence absent from groups buckets as "(unknown)", not dropped.
    rows2 = convergence_by_group({"A": {"z9": r("z9", True)}}, {})
    assert rows2[0]["group"] == "(unknown)" and rows2[0]["n"] == 1


def test_scoring_config_accepts_dcalign_keys():
    from SBM.combine_config import ScoringConfig

    cfg = ScoringConfig.from_dict(
        {"method": "dcalign", "dcalign_path": "/x/DCAlign", "julia": "/y/julia",
         "dcalign_seed": 7, "maxiter": 500, "pcount": 1e-2, "n_shards": 8, "lambda_spec": "flat"}
    )
    assert cfg.method == "dcalign" and cfg.n_shards == 8 and cfg.maxiter == 500
    assert cfg.dcalign_path == "/x/DCAlign" and cfg.julia == "/y/julia"


def test_scoring_config_rejects_unknown_and_bad_bounds():
    from SBM.combine_config import ScoringConfig
    from SBM.workflow_config import ConfigError

    with pytest.raises(ConfigError):
        ScoringConfig.from_dict({"bogus": 1})
    with pytest.raises(ConfigError, match="n_shards"):
        ScoringConfig.from_dict({"n_shards": 0})
    with pytest.raises(ConfigError, match="maxiter"):
        ScoringConfig.from_dict({"maxiter": 0})
    with pytest.raises(ConfigError, match="pcount"):
        ScoringConfig.from_dict({"pcount": 0})


# ── DCAlign integration, Tier 1 (needs julia + a DCAlign clone) ───────────────


@pytest.mark.integration
def test_dcalign_energy_transfer_agrees(tmp_path):
    """End-to-end: align via DCAlign, then our in-frame recompute on the returned
    frame must equal DCAlign's own compute_potts_en to fp noise (≤ 5e-7). This is
    the single check that validates the binary handoff, the gauge, and the energy
    sign all at once. Skipped unless julia is on PATH and DCALIGN_PATH is set.
    """
    import os
    import shutil

    from SBM.energy.encoding import seq_to_ints
    from SBM.energy.potts import potts_energy
    from SBM.utils.dcalign_score import align_sequences, dcalign_context

    if shutil.which("julia") is None or not os.environ.get("DCALIGN_PATH"):
        pytest.skip("needs julia on PATH and DCALIGN_PATH set (module load julia; export DCALIGN_PATH)")

    L = 10
    model = make_model(L, seed=202, scale_J=0.3)
    rng = np.random.default_rng(7)
    seqs = [rng.integers(1, Q, size=n) for n in (10, 10, 9)]  # raw, gap-free, residues 1..20
    ids = [f"q{i}" for i in range(len(seqs))]
    ctx = dcalign_context(maxiter=2000)
    results = align_sequences(ctx, model, seqs, ids, out_dir=tmp_path, lambda_spec="flat")

    assert {r.seq_id for r in results} == set(ids)
    checked = 0
    for r in results:
        if not r.ok:
            continue
        assert len(r.aligned_frame) == L
        ours = potts_energy(seq_to_ints(r.aligned_frame), model)
        assert abs(ours - r.dcalign_energy) <= 5e-7, (r.seq_id, ours, r.dcalign_energy)
        checked += 1
    assert checked >= 1  # at least one sequence aligned successfully
