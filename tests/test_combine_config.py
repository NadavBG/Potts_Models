"""Tests for the combine config schema (``SBM.combine_config``).

Focus: the ``design.w_a`` / ``design.w_b`` weight override added so a re-weighted
design (e.g. equal weights) can be run without re-deriving the naturals-based
E_tot weights. Two things are load-bearing:

  1. Schema validation: the overrides are both-or-neither and positive, and they
     round-trip through ``as_dict`` into ``config_snapshot.yaml``.
  2. ``resolve_design_config`` honors explicit weights WITHOUT reading
     ``energy_weights.json`` — this is what lets the design spec regenerate
     without the heavy scoring stage.

Pure numpy, no MCMC/cluster:

    .venv/bin/python -m pytest tests/test_combine_config.py -q
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from SBM import combine_config as cc
from SBM.workflow_config import ConfigError

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import design_two_model as d2m  # noqa: E402
from SBM.design.anneal import AnnealSchedule  # noqa: E402


def test_design_weights_default_to_none():
    d = cc.DesignConfig.from_dict({"enabled": True})
    assert d.w_a is None and d.w_b is None


def test_design_weights_override_accepted_and_roundtrips():
    d = cc.DesignConfig.from_dict({"enabled": True, "w_a": 0.5, "w_b": 0.5})
    assert (d.w_a, d.w_b) == (0.5, 0.5)
    # as_dict is what dump_config writes into config_snapshot.yaml, so the override
    # must survive the round-trip (otherwise the run is not reproducible from config).
    round_tripped = cc.DesignConfig.from_dict(asdict(d))
    assert (round_tripped.w_a, round_tripped.w_b) == (0.5, 0.5)


def test_design_weights_must_be_set_together():
    with pytest.raises(ConfigError, match="set together"):
        cc.DesignConfig.from_dict({"enabled": True, "w_a": 0.5})
    with pytest.raises(ConfigError, match="set together"):
        cc.DesignConfig.from_dict({"enabled": True, "w_b": 0.5})


def test_design_weights_must_be_positive():
    with pytest.raises(ConfigError, match="must be > 0"):
        cc.DesignConfig.from_dict({"enabled": True, "w_a": 0.0, "w_b": 0.5})
    with pytest.raises(ConfigError, match="must be > 0"):
        cc.DesignConfig.from_dict({"enabled": True, "w_a": 0.5, "w_b": -1.0})


def _write_models_json(combine_run: Path) -> None:
    """A minimal models.json (the only file resolve_design_config reads when the
    weights are overridden and there are no natural starts)."""
    combine_run.mkdir(parents=True, exist_ok=True)
    models = [
        {"name": "A", "model_path": "results/A/model.npy", "seed_msa": None, "L": 96},
        {"name": "B", "model_path": "results/B/model.npy", "seed_msa": None, "L": 91},
    ]
    (combine_run / "models.json").write_text(json.dumps({"models": models}), encoding="utf-8")


def test_resolve_design_config_honors_override_without_energy_weights(tmp_path):
    """With explicit w_a/w_b (and no natural starts, so no seed-MSA read), the spec
    resolves from models.json alone — energy_weights.json is never touched."""
    combine_run = tmp_path / "combine_run"
    _write_models_json(combine_run)
    assert not (combine_run / "data" / "energy_weights.json").exists()

    sched = AnnealSchedule(n_steps=1_000_000)
    config = d2m.resolve_design_config(
        combine_run=str(combine_run), schedule=sched, master_seed=0,
        start_random=4, start_natural_a=0, start_natural_b=0,
        do_polish=True, polish_schedule="fast", w_a=0.5, w_b=0.5,
    )
    assert config["w_a"] == 0.5 and config["w_b"] == 0.5
    assert config["weights_source"] is None          # did NOT read energy_weights.json
    assert config["schedule"]["n_steps"] == 1_000_000
    assert config["n_chains"] == 4


def test_resolve_design_config_reads_energy_weights_when_not_overridden(tmp_path):
    """Guard the other branch: with no override, the naturals-derived weights ARE
    read (so the default post-hoc path is unchanged)."""
    combine_run = tmp_path / "combine_run"
    _write_models_json(combine_run)
    weights = combine_run / "data" / "energy_weights.json"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_text(json.dumps({"w_A": 0.11, "w_B": 0.89}), encoding="utf-8")

    sched = AnnealSchedule(n_steps=5000)
    config = d2m.resolve_design_config(
        combine_run=str(combine_run), schedule=sched, master_seed=0,
        start_random=2, start_natural_a=0, start_natural_b=0,
        do_polish=False, polish_schedule="fast",
    )
    assert config["w_a"] == 0.11 and config["w_b"] == 0.89
    assert config["weights_source"] == str(weights)
