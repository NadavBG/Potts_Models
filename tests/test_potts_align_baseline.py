"""Unit tests for the potts_align ground-state-recovery baseline (pure logic).

Covers the row builder (:func:`compare_record` / :func:`rows_for_home_pairs`) and
the ΔE bucket accounting (:func:`summarize`). No cluster cache and no Julia — the
``PottsAlignCacheResult`` inputs are hand-built and energies come from the same
:func:`SBM.energy.potts.potts_energy` the production path uses.

Sign convention under test: ``delta_e = E_potts_align − E_inframe``; ``|ΔE|≤tol``
⇒ native at the ground state, ``ΔE<−tol`` ⇒ aligner beat native, ``ΔE>tol`` ⇒
potts_align worse than native (a search failure that must be surfaced, not hidden).

Run inside the project's uv venv::

    .venv/bin/python -m pytest tests/test_potts_align_baseline.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from SBM.energy.datasets import QueryRecord
from SBM.energy.encoding import ints_to_seq
from SBM.energy.model import PottsModel
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align_baseline import (
    DEFAULT_EQUAL_TOL,
    PottsAlignBaselineRow,
    compare_record,
    rows_for_home_pairs,
    summarize,
)
from SBM.utils.potts_align_cache import PottsAlignCacheResult
from SBM.utils.utils import MSA_ALPHABET, Zero_Sum_Gauge

Q = len(MSA_ALPHABET)


def make_model(L, *, seed, name="test"):
    """A random, zero-sum-gauged PottsModel of length L (mirrors test_energy)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(scale=1.0, size=(L, Q))
    J = rng.normal(scale=0.3, size=(L, L, Q, Q))
    J = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))
    for i in range(L):
        J[i, i] = 0.0
    J_zg, h_zg = Zero_Sum_Gauge(J, h)
    return PottsModel(name=name, J=J_zg, h=h_zg, L=L, q=Q, alphabet=MSA_ALPHABET,
                      gauge="zero_sum", sha256="0" * 64, source="<memory>")


def _cache_result(model, ints, *, engine="enumerate", is_global_exact=True, seed=1):
    """A PottsAlignCacheResult whose frame is ``ints`` and energy the true energy."""
    return PottsAlignCacheResult(
        query_id="q", model=model.name,
        n_residues=int(np.count_nonzero(ints != 0)),
        gaps=int(np.count_nonzero(ints == 0)),
        energy=potts_energy(ints, model),
        engine=engine, is_global_exact=is_global_exact,
        frame=ints_to_seq(ints), seed=seed,
    )


def _record(ints, model, rid="q"):
    return QueryRecord(rid, f"{model.name}/natural", model.name, np.asarray(ints, dtype=np.int64))


# ── recovery: native frame IS the returned frame ⇒ ΔE == 0, col agreement 1 ──
def test_recovers_native_frame_is_ground_state():
    model = make_model(6, seed=0)
    rng = np.random.default_rng(11)
    native = rng.integers(1, Q, size=model.L)  # gap-free length-L frame
    res = _cache_result(model, native)  # potts_align returned the native frame
    row = compare_record(_record(native, model), model, res)

    assert row.ok
    assert row.is_global_exact
    assert math.isclose(row.delta_e, 0.0, abs_tol=1e-9)
    assert math.isclose(row.col_agreement, 1.0)
    assert row.cache_abs_diff < 1e-9  # in-frame recompute matches the cached energy
    assert row.n_residues == model.L  # gap-free


# ── improvement: aligner returns a strictly lower-energy frame ⇒ ΔE < 0 ──
def test_lower_energy_frame_is_improvement():
    model = make_model(6, seed=1)
    rng = np.random.default_rng(3)
    cand = rng.integers(1, Q, size=(64, model.L))
    energies = np.array([potts_energy(c, model) for c in cand])
    native = cand[int(energies.argmax())]  # worst frame as the native
    best = cand[int(energies.argmin())]  # aligner finds the best frame
    res = _cache_result(model, best, engine="pt", is_global_exact=False)
    row = compare_record(_record(native, model), model, res)

    assert row.ok
    assert not row.is_global_exact
    assert row.delta_e < -DEFAULT_EQUAL_TOL  # strictly better than native
    assert math.isclose(row.e_potts, potts_energy(best, model))
    summ = summarize([row])["overall"]
    assert summ["n_improved"] == 1
    assert summ["n_at_ground"] == 0 and summ["n_worse"] == 0


# ── skip row (empty frame) is kept as ok=False with a finite in-frame energy ──
def test_skip_row_kept_not_dropped():
    model = make_model(5, seed=2)
    rng = np.random.default_rng(7)
    native = rng.integers(1, Q, size=model.L)
    skip = PottsAlignCacheResult(
        query_id="q", model=model.name, n_residues=0, gaps=0, energy=math.nan,
        engine="skip_NgtL", is_global_exact=False, frame="", seed=0,
    )
    assert not skip.ok
    row = compare_record(_record(native, model), model, skip)

    assert not row.ok
    assert math.isfinite(row.e_inframe)  # native energy still computed
    assert math.isnan(row.e_potts) and math.isnan(row.delta_e)
    summ = summarize([row])["overall"]
    assert summ["n_ok"] == 0 and summ["n_failed"] == 1


# ── frame-length mismatch is a loud error, not a silent skip ──
def test_wrong_length_frame_raises():
    model = make_model(6, seed=4)
    rng = np.random.default_rng(9)
    native = rng.integers(1, Q, size=model.L)
    bad = PottsAlignCacheResult(
        query_id="q", model=model.name, n_residues=5, gaps=0, energy=0.0,
        engine="enumerate", is_global_exact=True,
        frame=ints_to_seq(np.array([1, 2, 3, 4, 5])), seed=0,  # length 5 != L=6
    )
    with pytest.raises(ValueError, match="frame length"):
        compare_record(_record(native, model), model, bad)


# ── batched builder matches the one-at-a-time path exactly ──
def test_batched_matches_per_record():
    model = make_model(6, seed=5)
    rng = np.random.default_rng(13)
    pairs = []
    for i in range(8):
        native = rng.integers(1, Q, size=model.L)
        frame = rng.integers(1, Q, size=model.L)
        res = _cache_result(model, frame, seed=i)
        pairs.append((_record(native, model, rid=f"q{i}"), res))

    batched = rows_for_home_pairs(model, pairs)
    one_by_one = [compare_record(rec, model, res) for rec, res in pairs]
    # Energies agree to fp tolerance, not bit-for-bit: batched compute_energies
    # sums an (N, L) matrix in one call vs (1, L) per record, so the reduction
    # order differs by ~1 ULP. Non-float fields must match exactly.
    assert len(batched) == len(one_by_one)
    float_fields = {"e_inframe", "e_potts", "delta_e", "col_agreement", "cache_energy", "cache_abs_diff"}
    for b, o in zip(batched, one_by_one):
        bd, od = b.as_dict(), o.as_dict()
        for k in bd:
            if k in float_fields:
                assert math.isclose(bd[k], od[k], rel_tol=1e-12, abs_tol=1e-12), (k, bd[k], od[k])
            else:
                assert bd[k] == od[k], (k, bd[k], od[k])


# ── the ΔE partition (at-ground / improved / worse) and per-model/group rollups ──
def test_summarize_buckets_partition_ok_rows():
    def row(delta, col, ok=True, model="M", group="g"):
        return PottsAlignBaselineRow(
            sequence_id="s", group=group, model=model, n_residues=3,
            e_inframe=0.0, e_potts=delta, delta_e=delta, col_agreement=col,
            cache_energy=delta, cache_abs_diff=0.0, engine="enumerate",
            is_global_exact=True, ok=ok,
        )
    rows = [
        row(0.0, 1.0),               # at ground, exact frame
        row(0.5, 0.5),               # at ground (|ΔE|<=tol), degenerate frame
        row(-3.0, 0.2),              # improved
        row(3.0, 0.2),               # worse (search failure)
        row(math.nan, math.nan, ok=False),  # skip
    ]
    summ = summarize(rows, equal_tol=DEFAULT_EQUAL_TOL)
    ov = summ["overall"]
    assert ov["n"] == 5 and ov["n_ok"] == 4 and ov["n_failed"] == 1
    assert ov["n_at_ground"] == 2
    assert ov["n_recovered_exact_frame"] == 1  # only the col_agreement==1 one
    assert ov["n_improved"] == 1
    assert ov["n_worse"] == 1
    assert math.isclose(ov["frac_at_ground"], 0.5)
    # partition holds: at_ground + improved + worse == n_ok
    assert ov["n_at_ground"] + ov["n_improved"] + ov["n_worse"] == ov["n_ok"]
    assert "M" in summ["by_model"] and "M | g" in summ["by_group"]


def test_summarize_all_failed():
    rows = [PottsAlignBaselineRow(
        sequence_id=f"s{i}", group="g", model="M", n_residues=0,
        e_inframe=1.0, e_potts=math.nan, delta_e=math.nan, col_agreement=math.nan,
        cache_energy=math.nan, cache_abs_diff=math.nan, engine="skip", is_global_exact=False, ok=False,
    ) for i in range(3)]
    ov = summarize(rows)["overall"]
    assert ov["n_ok"] == 0 and ov["n_failed"] == 3 and ov["n"] == 3
