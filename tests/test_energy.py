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
import json
import logging
import math

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


# ── potts_align scoring branch (iter-003) ─────────────────────────────────


def test_potts_align_branch_matches_enumerate():
    """method='potts_align' returns the exact global insert-free Potts minimum.

    On a small model the whole C(L,N) space is enumerated, so the score branch
    must equal ``enumerate_align`` (the oracle) exactly, and flag it exact.
    """
    from SBM.energy.potts_align import enumerate_align

    model = make_model(6, seed=204)
    raw = np.array([2, 4, 6, 8], dtype=np.int64)  # N=4, L=6 -> g=2, C(6,4)=15 frames
    res = score_sequence(raw, model, method="potts_align", seed=0)
    oracle = enumerate_align(raw, model)
    assert res.method == "potts_align"
    assert np.isclose(res.energy, oracle.best_energy, atol=1e-9)
    assert "engine=enumerate" in res.notes and "exact=True" in res.notes
    # representative_alignment is a length-L frame with exactly N non-gap residues
    assert len(res.representative_alignment) == model.L
    assert res.representative_alignment.count("-") == model.L - raw.size


def test_potts_align_g0_equals_in_frame():
    """A length-L query (g=0) enumerates a single frame == the in-frame energy."""
    model = make_model(6, seed=205)
    S = np.random.default_rng(9).integers(1, Q, size=model.L)  # residues 1..20, no gaps
    e_pa = score_sequence(S, model, method="potts_align", seed=3).energy
    e_inf = score_sequence(S, model, method="in_frame").energy
    assert np.isclose(e_pa, e_inf, atol=1e-9)


def test_potts_align_is_deterministic_per_seed():
    """Same seed -> identical energy (pure numpy, thread-independent)."""
    model = make_model(6, seed=206)
    raw = np.array([1, 3, 5, 7, 9], dtype=np.int64)  # N=5, g=1
    a = score_sequence(raw, model, method="potts_align", seed=11).energy
    b = score_sequence(raw, model, method="potts_align", seed=11).energy
    assert a == b


def test_potts_align_requires_seed():
    model = make_model(5, seed=207)
    with pytest.raises(ValueError, match="seed"):
        score_sequence(np.array([2, 4, 6]), model, method="potts_align")


def test_potts_align_rejects_gapped_and_N_gt_L():
    model = make_model(4, seed=208)
    with pytest.raises(ValueError, match="gap-free"):
        score_sequence(np.array([2, 0, 6]), model, method="potts_align", seed=1)
    with pytest.raises(ValueError, match="N<=L"):
        score_sequence(np.array([2, 4, 6, 8, 10]), model, method="potts_align", seed=1)


def test_potts_align_cache_round_trip(tmp_path):
    """write_alignment_cache round-trips; skip rows (nan/empty frame) parse; dups raise."""
    from SBM.utils.potts_align_cache import (
        PottsAlignCacheResult,
        read_potts_align_cache,
        read_shard_cache,
        write_alignment_cache,
    )

    rows = [
        PottsAlignCacheResult("qA", "CM", 93, 3, -251.8598621, "enumerate", True, "-TAEN", 42),
        PottsAlignCacheResult("qB", "CM", 91, 5, -21.9, "pt", False, "MVWFK", 45),
        PottsAlignCacheResult("qC", "PPIC", 95, -4, float("nan"), "skip_NgtL", False, "", 0),
    ]
    out = tmp_path / "alignments.tsv"
    write_alignment_cache(out, rows)
    by_id = read_potts_align_cache(out)
    assert set(by_id) == {"qA", "qB", "qC"}
    assert by_id["qA"].energy == -251.8598621 and by_id["qA"].is_global_exact and by_id["qA"].ok
    assert by_id["qB"].engine == "pt" and not by_id["qB"].is_global_exact
    assert not by_id["qC"].ok and np.isnan(by_id["qC"].energy) and by_id["qC"].engine == "skip_NgtL"

    shard = read_shard_cache(out)  # keyed by (query_id, model)
    assert ("qA", "CM") in shard and ("qC", "PPIC") in shard

    dup = tmp_path / "dup.tsv"
    dup.write_text(
        "qA\tCM\t5\t0\t-1.0\tenumerate\ttrue\tACDEF\t1\n"
        "qA\tCM\t5\t0\t-1.0\tenumerate\ttrue\tACDEF\t1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_potts_align_cache(dup)


# ── potts_align CLUSTER orchestration: plan -> run -> gather (iter-003) ────
#
# The in-process branch above is covered; this exercises the highest-risk part
# of the wiring — the standalone cluster CLIs (scripts/wf/run_potts_align_*.py):
# pair classification, round-robin sharding, the per-pair seed derivation (which
# MUST match score_two_models), coverage/missing detection, and the gather gates.


def _load_wf_module(stem):
    """Import a scripts/wf/ standalone CLI by path (not a package).

    run_potts_align_{shard,gather}.py import only ``SBM.*`` + stdlib at module
    scope (no ``snakemake`` global), so they load cleanly outside Snakemake.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "wf" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_wf_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_model_npy(path, L, seed):
    """Persist a make_model PottsModel as a model.npy load_model can read."""
    m = make_model(L, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array({"h": m.h, "J": m.J}, dtype=object), allow_pickle=True)


def _build_potts_align_run_root(root):
    """A tiny built combine run_root: two models (L=6 'A', L=5 'B'), a query set
    that hits every pair class (home / cross / skip_NgtL / skip_subsample), a
    valid config_snapshot.yaml and models.json. Returns (records-in-id-order)."""
    from pathlib import Path as _P

    yaml = pytest.importorskip("yaml")  # PyYAML ships with the workflow extra

    from SBM.energy.datasets import QueryRecord
    from SBM.energy.datasets import write_query_fasta

    L_A, L_B = 6, 5
    mdir_A = _P(root) / "models" / "A"
    mdir_B = _P(root) / "models" / "B"
    _write_model_npy(mdir_A / "model.npy", L_A, seed=301)
    _write_model_npy(mdir_B / "model.npy", L_B, seed=302)

    # Home records are stored in their origin model's L-frame (gaps allowed);
    # cross/random are raw N<=L. ids chosen so sorted order is deterministic.
    records = [
        QueryRecord("A|nat|0", "A/natural", "A", np.array([3, 5, 7, 9, 11, 13], dtype=np.int64)),   # N=6=L_A (g0); N>L_B -> skip_NgtL under B
        QueryRecord("A|nat|1", "A/natural", "A", np.array([3, 0, 5, 0, 7, 9], dtype=np.int64)),      # N=4 (g2) home A; cross under B
        QueryRecord("B|nat|0", "B/natural", "B", np.array([2, 4, 6, 8, 10], dtype=np.int64)),        # N=5=L_B home B; cross/subsample under A
        QueryRecord("B|nat|1", "B/natural", "B", np.array([12, 14, 16, 1, 3], dtype=np.int64)),      # N=5=L_B home B; cross/subsample under A
        QueryRecord("random|N5|0", "random/N5", "", np.array([5, 10, 15, 20, 2], dtype=np.int64)),   # cross under both
    ]
    write_query_fasta(records, _P(root) / "query" / "query.fasta", _P(root) / "query" / "groups.json")

    models_json = {"schema_version": 1, "models": [
        {"name": "A", "run_dir": str(mdir_A), "model_path": str(mdir_A / "model.npy"),
         "L": L_A, "q": Q, "weight": 1.0},
        {"name": "B", "run_dir": str(mdir_B), "model_path": str(mdir_B / "model.npy"),
         "L": L_B, "q": Q, "weight": 1.0},
    ]}
    (_P(root) / "models.json").write_text(json.dumps(models_json, indent=2), encoding="utf-8")

    cfg = {
        "run_name": "combine-potts-test", "seed": 42, "omp_num_threads": None,
        "models": [{"name": "A", "run_dir": str(mdir_A), "weight": 1.0},
                   {"name": "B", "run_dir": str(mdir_B), "weight": 1.0}],
        "query": {"source": "model_sets", "include": ["natural"], "cap_per_group": 0,
                  "n_random": 1, "random_length": 5},
        "scoring": {"method": "potts_align", "n_shards": 2,
                    "pa_cross_subsample_origin": "B", "pa_cross_subsample_under": "A",
                    "pa_cross_subsample_n": 1},
        "figures": {"enabled": True},
    }
    (_P(root) / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return sorted(records, key=lambda r: r.id)


def test_potts_align_cluster_plan_run_gather(tmp_path):
    """plan classifies pairs + derives matching seeds; run scores a shard; gather
    detects an incomplete run, then merges + passes the canary/ΔE gates."""
    import types

    shard = _load_wf_module("run_potts_align_shard")
    gather = _load_wf_module("run_potts_align_gather")
    sorted_records = _build_potts_align_run_root(tmp_path)

    # --- plan -------------------------------------------------------------
    shard.cmd_plan(types.SimpleNamespace(run_root=str(tmp_path), n_shards=2))
    manifest = json.loads((tmp_path / "potts_align" / "shards_manifest.json").read_text())
    status = {(p["query_id"], p["model"]): p["status"] for p in manifest["pairs"]}

    # every pair class is represented and classified correctly
    assert status[("A|nat|0", "A")] == "home"
    assert status[("A|nat|0", "B")] == "skip_NgtL"       # N=6 > L_B=5
    assert status[("A|nat|1", "A")] == "home"
    assert status[("A|nat|1", "B")] == "cross"
    assert status[("B|nat|0", "B")] == "home"
    assert status[("B|nat|1", "B")] == "home"
    assert status[("random|N5|0", "A")] == "cross"
    assert status[("random|N5|0", "B")] == "cross"
    # cross-subsample n=1 over the two B-origin ids under A: exactly one skips
    b_under_a = [status[("B|nat|0", "A")], status[("B|nat|1", "A")]]
    assert sorted(b_under_a) == ["cross", "skip_subsample"]

    # seed derivation is EXACTLY score_two_models' (master_seed + 2*r_idx + j over
    # id-sorted records, j = model index) — the reproducibility contract.
    idx = {r.id: i for i, r in enumerate(sorted_records)}
    model_j = {"A": 0, "B": 1}
    for p in manifest["pairs"]:
        assert p["seed"] == 42 + 2 * idx[p["query_id"]] + model_j[p["model"]]

    # shards partition the in-scope pairs exactly (no dup, no drop, in-scope only)
    flat = [i for s in manifest["shards"] for i in s]
    assert len(set(flat)) == len(flat)                          # no pair in two shards
    assert len(flat) == manifest["n_pairs_in_scope"]            # every in-scope pair placed
    assert all(manifest["pairs"][i]["status"] in ("home", "cross") for i in flat)

    # --- an incomplete run must fail loudly at gather ---------------------
    shard.cmd_run(types.SimpleNamespace(run_root=str(tmp_path), shard=0))
    with pytest.raises(RuntimeError, match="not produced by any shard"):
        gather.main(["--run-root", str(tmp_path)])

    # --- finish + gather --------------------------------------------------
    shard.cmd_run(types.SimpleNamespace(run_root=str(tmp_path), shard=1))
    assert gather.main(["--run-root", str(tmp_path)]) == 0

    gstatus = json.loads((tmp_path / "potts_align" / "gather_status.json").read_text())
    assert not gstatus["partial"]
    for name in ("A", "B"):
        g = gstatus["gates"][name]
        assert g["canary_ok"]                                    # in-frame recompute matches cluster energy
        assert g["delta_e_gate"]["n_enumerated_violations"] == 0  # an enumerated home can't beat native

    # each per-model cache has one row per query id (scored or skip)
    from SBM.utils.potts_align_cache import read_potts_align_cache
    cache_A = read_potts_align_cache(tmp_path / "potts_align" / "cache" / "A" / "alignments.tsv")
    cache_B = read_potts_align_cache(tmp_path / "potts_align" / "cache" / "B" / "alignments.tsv")
    assert set(cache_A) == set(cache_B) == {r.id for r in sorted_records}
    assert not cache_B["A|nat|0"].ok and math.isnan(cache_B["A|nat|0"].energy)   # skip_NgtL row
    assert cache_A["A|nat|0"].ok and cache_A["A|nat|0"].is_global_exact          # enumerated home


def test_scoring_config_accepts_methods_and_potts_keys():
    from SBM.combine_config import ScoringConfig

    cfg = ScoringConfig.from_dict(
        {"method": "potts_align", "n_shards": 8,
         "pa_cross_subsample_origin": "PPIC", "pa_cross_subsample_under": "CM",
         "pa_cross_subsample_n": 2000}
    )
    assert cfg.method == "potts_align" and cfg.n_shards == 8
    assert cfg.pa_cross_subsample_n == 2000
    assert ScoringConfig.from_dict({"method": "map"}).method == "map"
    assert ScoringConfig.from_dict({}).method == "map"  # default


def test_scoring_config_rejects_unknown_and_bad_bounds():
    from SBM.combine_config import ScoringConfig
    from SBM.workflow_config import ConfigError

    with pytest.raises(ConfigError):
        ScoringConfig.from_dict({"bogus": 1})
    with pytest.raises(ConfigError, match="method"):
        ScoringConfig.from_dict({"method": "dcalign"})  # retired -> now rejected
    with pytest.raises(ConfigError, match="n_shards"):
        ScoringConfig.from_dict({"n_shards": 0})
    with pytest.raises(ConfigError, match="pa_cross_subsample_n"):
        ScoringConfig.from_dict({"pa_cross_subsample_n": -1})
    with pytest.raises(ConfigError, match="pa_cross_subsample"):
        # n>0 requires both the origin and under model names
        ScoringConfig.from_dict({"pa_cross_subsample_n": 10})
