"""Tests for the combine scaffolding CLI (``scripts/new_combine.py``).

Covers the pure logic — model-name inference, the data-driven potts_align knob
picks (cross-subsample origin = larger-N family, random_length = min L), config
validity, and the clobber guard. Pure numpy, no MCMC/cluster:

    .venv/bin/python -m pytest tests/test_new_combine.py -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from SBM import combine_config as cc

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import new_combine  # noqa: E402


def _make_model_dir(base: Path, rel: str, length: int, n_seqs: int, seed: int = 0) -> Path:
    """Write a minimal model.npy + inputs/msa.npy under ``base/rel``."""
    d = base / rel
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = {"h": rng.normal(size=(length, 21)), "J": np.zeros((length, length, 21, 21))}
    np.save(d / "model.npy", np.array(model, dtype=object), allow_pickle=True)
    np.save(d / "inputs" / "msa.npy", np.zeros((n_seqs, length), dtype=int))
    return d


def _ns(**kw) -> argparse.Namespace:
    base = dict(
        name_a=None, name_b=None, run_name=None, method="potts_align",
        seed=42, no_design=False, design_local=False, no_characterize=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# ── name inference ────────────────────────────────────────────────────────


def test_model_name_from_iter_dir_uses_parent():
    assert new_combine._model_name(Path("results/CM-bm-dense/iter-002-base")) == "CM-bm-dense"


def test_model_name_from_flat_dir_uses_leaf():
    assert new_combine._model_name(Path("results/derive-CM-profile")) == "derive-CM-profile"


# ── validation ────────────────────────────────────────────────────────────


def test_validate_returns_L_and_N(tmp_path):
    d = _make_model_dir(tmp_path, "fam/iter-001-x", length=91, n_seqs=1234)
    assert new_combine._validate_model_dir(d) == (91, 1234)


def test_validate_missing_model_exits(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        new_combine._validate_model_dir(tmp_path / "empty")


def test_validate_missing_msa_exits(tmp_path):
    d = tmp_path / "nomsa"
    d.mkdir()
    np.save(d / "model.npy", np.array({"h": np.zeros((10, 21))}, dtype=object), allow_pickle=True)
    with pytest.raises(SystemExit):
        new_combine._validate_model_dir(d)


# ── data-driven knob picks ────────────────────────────────────────────────


def test_cross_subsample_origin_is_larger_family_and_random_length_is_min_L(tmp_path):
    # A: L=96 small (N=1258); B: L=91 large (N=26701) — the real CM/PPIC shapes.
    da = _make_model_dir(tmp_path, "CM-bm-dense/iter-002-base", 96, 1258)
    db = _make_model_dir(tmp_path, "PPIC-dense/iter-001-baseline", 91, 26701)
    data, name_a, name_b = new_combine._build_config_dict(_ns(dir_a=str(da), dir_b=str(db)))
    assert (name_a, name_b) == ("CM-bm-dense", "PPIC-dense")
    # Larger-N family (PPIC) is the subsampled cross origin (the PT cost driver).
    assert data["scoring"]["pa_cross_subsample_origin"] == "PPIC-dense"
    assert data["scoring"]["pa_cross_subsample_under"] == "CM-bm-dense"
    # random_length must be <= both Ls.
    assert data["query"]["random_length"] == 91


def test_generated_config_passes_full_validation(tmp_path):
    da = _make_model_dir(tmp_path, "A/iter-001-base", 96, 100)
    db = _make_model_dir(tmp_path, "B/iter-001-base", 91, 200)
    data, _, _ = new_combine._build_config_dict(_ns(dir_a=str(da), dir_b=str(db)))
    cfg = cc.from_dict(data)  # would raise on any bad combo / unknown key
    assert cfg.run_name == "combine-A-B"
    assert cfg.design.enabled and cfg.design.execution == "cluster"
    assert cfg.characterize.enabled


def test_non_potts_method_drops_potts_knobs(tmp_path):
    da = _make_model_dir(tmp_path, "A/iter-001-base", 96, 100)
    db = _make_model_dir(tmp_path, "B/iter-001-base", 91, 200)
    data, _, _ = new_combine._build_config_dict(
        _ns(dir_a=str(da), dir_b=str(db), method="map")
    )
    assert data["scoring"] == {"method": "map"}
    assert data["query"]["n_random"] == 0
    cc.from_dict(data)


def test_design_local_flag_sets_execution(tmp_path):
    da = _make_model_dir(tmp_path, "A/iter-001-base", 96, 100)
    db = _make_model_dir(tmp_path, "B/iter-001-base", 91, 200)
    data, _, _ = new_combine._build_config_dict(
        _ns(dir_a=str(da), dir_b=str(db), design_local=True)
    )
    assert data["design"]["execution"] == "local"


def test_identical_names_error(tmp_path):
    da = _make_model_dir(tmp_path, "same/iter-001-x", 96, 100)
    db = _make_model_dir(tmp_path, "same/iter-002-y", 91, 200)
    with pytest.raises(SystemExit):
        new_combine._build_config_dict(_ns(dir_a=str(da), dir_b=str(db)))


# ── the clobber guard (via main, in an isolated cwd) ──────────────────────


def test_config_reuse_and_clobber_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    da = _make_model_dir(tmp_path, "A/iter-001-base", 96, 100)
    db = _make_model_dir(tmp_path, "B/iter-001-base", 91, 200)
    argv = [str(da), str(db), "--tag", "t", "--run-name", "combine-x", "--config-only"]
    assert new_combine.main(argv) == 0
    cfg_path = tmp_path / "config" / "params_combine-x.yaml"
    assert cfg_path.is_file()
    # An IDENTICAL re-run reuses the config (no error) — a second iteration needs no --force.
    assert new_combine.main(argv) == 0
    # A DIFFERING config (different method) without --force is refused.
    with pytest.raises(SystemExit):
        new_combine.main(argv + ["--method", "map"])
    # --force overwrites it.
    assert new_combine.main(argv + ["--method", "map", "--force"]) == 0


def test_main_mints_run_dir_and_writes_runbook(tmp_path, monkeypatch):
    """The actual point of the tool: mint the iter dir + write RUNBOOK.txt."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    da = _make_model_dir(tmp_path, "A/iter-001-base", 96, 100)
    db = _make_model_dir(tmp_path, "B/iter-001-base", 91, 200)
    argv = [str(da), str(db), "--tag", "mint-test", "--run-name", "combine-y"]
    assert new_combine.main(argv) == 0
    rb = tmp_path / "combine" / "combine-y" / "iter-001-mint-test" / "RUNBOOK.txt"
    assert rb.is_file()
    text = rb.read_text()
    assert "STAGE 1" in text and "combine-y" in text
    # A second run mints iter-002 without --force (identical config reused).
    assert new_combine.main([str(da), str(db), "--tag", "again", "--run-name", "combine-y"]) == 0
    assert (tmp_path / "combine" / "combine-y" / "iter-002-again" / "RUNBOOK.txt").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
