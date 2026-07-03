"""Tests for the post-hoc parameter-filtering ("derive") feature.

Covers the pure-numpy core (``SBM.derive``: fields-only, couplings-only, mask
subset, gauge idempotence, the closed-form energy of a J=0 model, and the
derived-dict provenance) and the config schema (``SBM.derive_config``: keep /
zero / MaskSpec parsing, round-trip, unknown-key rejection).

Pure numpy/scipy (no MCMC kernel):

    .venv/bin/python -m pytest tests/test_derive.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from SBM import derive
from SBM import derive_config as dc
from SBM.derive_config import KEEP, MaskSpec, ZERO
from SBM.energy.encoding import Q
from SBM.energy.model import PottsModel, load_model
from SBM.energy.potts import potts_energy
from SBM.utils.utils import MSA_ALPHABET, Zero_Sum_Gauge


# ── helpers ──────────────────────────────────────────────────────────────


def gauged_params(L, *, seed, q=Q):
    """A random, zero-sum-gauged (J, h) — like a trained model's arrays."""
    rng = np.random.default_rng(seed)
    h = rng.normal(scale=1.0, size=(L, q))
    J = rng.normal(scale=0.3, size=(L, L, q, q))
    J = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))  # J[i,j,a,b] = J[j,i,b,a]
    for i in range(L):
        J[i, i] = 0.0
    return Zero_Sum_Gauge(J, h)


def source_dict(L, *, seed):
    """A minimal model.npy-shaped dict with a gauged (J, h) and the usual keys."""
    J, h = gauged_params(L, seed=seed)
    return {
        "J": J,
        "h": h,
        "W_all": np.zeros((1, 3)),
        "Seeds": [1, 2, 3],
        "Execution times": [1.0],
        "J_norm": np.array([[0.0, 1.0, 0.9, 0.8]]),
        "J_norm_iters": [0, 1, 2],
        "align": np.zeros((2, L), dtype=int),
        "Train": np.zeros((2, L), dtype=int),
        "Test": None,
        "options0": {"Model": "BM", "m": 20, "n_states": 4, "Record_every": 1},
        "options1": {"q": Q, "L": L, "Seed": 42},
    }


# ── core: fields-only / couplings-only / mask ──────────────────────────────


def test_fields_only_zeros_J_and_preserves_h():
    J, h = gauged_params(6, seed=1)
    J_new, h_new = derive.apply_filter(J, h, zero_J=True)
    assert np.array_equal(J_new, np.zeros_like(J)), "fields-only must zero every coupling"
    # Source is already gauged; re-gauging with J=0 is idempotent on h.
    assert np.allclose(h_new, h)
    assert J_new.shape == J.shape  # full (L,L,q,q), not None/scalar


def test_couplings_only_zeros_h_for_gauged_source():
    J, h = gauged_params(6, seed=2)
    J_new, h_new = derive.apply_filter(J, h, zero_h=True)
    assert np.allclose(h_new, 0.0), "couplings-only on a gauged source leaves h at zero"
    assert np.allclose(J_new, J)  # already zero-sum, gauge is a no-op


def test_keep_all_is_identity_up_to_gauge():
    J, h = gauged_params(5, seed=3)
    J_new, h_new = derive.apply_filter(J, h)  # no zero, no mask
    assert np.allclose(J_new, J)
    assert np.allclose(h_new, h)


def test_mask_subset_zeros_pruned_blocks_keeps_others():
    L = 5
    J, h = gauged_params(L, seed=4)
    mask = np.zeros((L, L, Q, Q), dtype=int)
    mask[0, 1] = 1
    mask[1, 0] = 1  # symmetric keep of the (0,1) pair only
    J_new, _ = derive.apply_filter(J, h, mask_J=mask)
    # Centering a zeroed block leaves it zero, so pruned pairs are exactly 0.
    assert np.allclose(J_new[2, 3], 0.0)
    assert np.allclose(J_new[0, 2], 0.0)
    # The kept pair is not identically zero (it had real couplings).
    assert not np.allclose(J_new[0, 1], 0.0)


def test_apply_filter_rejects_conflicting_or_misshaped_args():
    J, h = gauged_params(4, seed=5)
    with pytest.raises(ValueError):
        derive.apply_filter(J, h, zero_J=True, mask_J=np.ones_like(J))
    with pytest.raises(ValueError):
        derive.apply_filter(J, h, zero_h=True, mask_h=np.ones_like(h))
    with pytest.raises(ValueError):
        derive.apply_filter(J, h, mask_J=np.ones((3, 3, Q, Q)))  # wrong shape


# ── core: closed-form energy of a fields-only model ───────────────────────


def test_fields_only_energy_is_pure_field_sum():
    L = 8
    J, h = gauged_params(L, seed=6)
    J_new, h_new = derive.apply_filter(J, h, zero_J=True)
    model = PottsModel(
        name="d", J=J_new, h=h_new, L=L, q=Q, alphabet=MSA_ALPHABET,
        gauge="zero_sum", sha256="0" * 64, source="<mem>",
    )
    rng = np.random.default_rng(0)
    S = rng.integers(0, Q, size=L)
    expected = -float(h_new[np.arange(L), S].sum())  # E = -Σ_i h(S_i), no couplings
    assert potts_energy(S, model) == pytest.approx(expected)


# ── core: derived-dict assembly + provenance ──────────────────────────────


def test_build_derived_dict_provenance_and_cleanup():
    src = source_dict(5, seed=7)
    J_new, h_new = derive.apply_filter(src["J"], src["h"], zero_J=True)
    note = {"Derived From": "results/foo/iter-001", "Source Model SHA256": "abc"}
    out = derive.build_derived_dict(src, J_new, h_new, provenance_note=note)
    # J_norm collapses to a single truthful point; 0 for a fields-only model.
    assert np.asarray(out["J_norm"]).shape == (1, 2)
    assert float(np.asarray(out["J_norm"])[0, 1]) == pytest.approx(0.0)
    assert out["J_norm_iters"] == [0]
    # Stale replicate/timing artifacts are dropped.
    for k in ("W_all", "Seeds", "Execution times"):
        assert k not in out
    # Lineage recorded; source options carried through.
    assert out["options1"]["Derived From"] == "results/foo/iter-001"
    assert out["options1"]["q"] == Q
    # The source dict is not mutated.
    assert "W_all" in src


def test_derived_dict_round_trips_through_load_model(tmp_path):
    src = source_dict(6, seed=8)
    J_new, h_new = derive.apply_filter(src["J"], src["h"], zero_J=True)
    out = derive.build_derived_dict(src, J_new, h_new, provenance_note={})
    path = tmp_path / "model.npy"
    np.save(path, out)
    model = load_model(path)  # shape checks + re-gauge must pass on the derived file
    assert model.L == 6 and model.q == Q
    assert np.allclose(model.J, 0.0)


# ── config schema ─────────────────────────────────────────────────────────


def test_filter_parses_keep_zero_and_maskspec():
    cfg = dc.from_dict(
        {
            "run_name": "x",
            "source_run_dir": "results/foo/iter-001",
            "filter": {"couplings": "zero", "fields": None},
        }
    )
    assert cfg.filter.couplings == ZERO
    assert cfg.filter.fields == KEEP
    assert cfg.filter.couplings_mask is None

    cfg2 = dc.from_dict(
        {
            "run_name": "x",
            "source_run_dir": "d",
            "filter": {"couplings": {"strategy": "fij", "percent": 90}, "fields": "zero"},
        }
    )
    assert isinstance(cfg2.filter.couplings, MaskSpec)
    assert cfg2.filter.couplings_mask.strategy == "fij"
    assert cfg2.filter.couplings_mask.percent == 90.0
    assert cfg2.filter.fields == ZERO


def test_config_round_trips_through_yaml():
    cfg = dc.from_dict(
        {
            "run_name": "x",
            "source_run_dir": "d",
            "filter": {"couplings": {"strategy": "fij", "percent": 90}},
        }
    )
    rt = dc.from_dict(yaml.safe_load(yaml.safe_dump(cfg.as_dict())))
    assert rt.filter.couplings_mask.strategy == "fij"
    assert rt.filter.fields == KEEP


def test_config_rejects_unknown_keys_and_bad_block():
    with pytest.raises(dc.ConfigError):
        dc.from_dict({"run_name": "x", "source_run_dir": "d", "bogus": 1})
    with pytest.raises(dc.ConfigError):
        dc.from_dict({"run_name": "x", "source_run_dir": "d", "filter": {"couplings": "drop"}})
    with pytest.raises(dc.ConfigError):
        dc.from_dict({"run_name": "x"})  # missing source_run_dir
