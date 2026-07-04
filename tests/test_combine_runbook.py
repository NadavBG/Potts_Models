"""Tests for the per-run combine RUNBOOK renderer (``SBM.combine_runbook``).

Pure string generation — no MCMC, no cluster:

    .venv/bin/python -m pytest tests/test_combine_runbook.py -q
"""

from __future__ import annotations

import pytest

from SBM import combine_config as cc
from SBM import combine_runbook

RR = "combine/combine-demo/iter-001-demo"
CFG = "config/params_combine-demo.yaml"


def make_cfg(
    *, method="potts_align", design=True, execution="cluster", characterize=True
):
    """A validated CombineRunConfig with the given toggles (fake run dirs)."""
    data = {
        "run_name": "combine-demo",
        "models": [
            {"name": "MODA", "run_dir": "results/A/iter-001"},
            {"name": "MODB", "run_dir": "results/B/iter-001"},
        ],
        "scoring": {"method": method},
        "figures": {"enabled": True},
        "design": {"enabled": design, "execution": execution},
        "characterize": {"enabled": characterize},
    }
    if method == "potts_align":
        data["scoring"].update(
            pa_cross_subsample_origin="MODB",
            pa_cross_subsample_under="MODA",
            pa_cross_subsample_n=2000,
        )
    return cc.from_dict(data)


def test_paths_are_interpolated_no_placeholders():
    text = combine_runbook.render_runbook(make_cfg(), RR, CFG)
    assert RR in text and CFG in text
    # $RR / $CFG / $SNAKE are threaded, and no <placeholder> survives.
    assert "$RR" in text and "$CFG" in text and "$SNAKE" in text
    assert "<combine_run>" not in text and "<config>" not in text


def test_potts_align_full_pipeline_shows_all_cluster_drivers():
    text = combine_runbook.render_runbook(make_cfg(method="potts_align"), RR, CFG)
    for driver in (
        "run_potts_align_align.sh",
        "finalize_potts_align.sh",
        "run_design.sh",
        "finalize_design.sh",
        "run_characterize.sh",
    ):
        assert driver in text, f"missing driver {driver}"
    assert "STAGE 1" in text and "STAGE 2" in text and "STAGE 3" in text


def test_map_method_collapses_to_single_local_command():
    text = combine_runbook.render_runbook(
        make_cfg(method="map", design=False, characterize=False), RR, CFG
    )
    # No cluster round-trip, no design/characterize stages.
    assert "run_potts_align_align.sh" not in text
    assert "STAGE 2" not in text and "STAGE 3" not in text
    assert "$SNAKE all" in text


def test_design_local_note_when_not_cluster():
    text = combine_runbook.render_runbook(
        make_cfg(method="map", design=True, execution="local", characterize=False), RR, CFG
    )
    assert "STAGE 2" in text
    # Local design has no push/anneal cluster block; it's produced by `all`.
    assert "run_design.sh" not in text
    assert "execution: local" in text


def test_auto_execution_surfaces_cluster_steps_conditionally():
    """auto may route to cluster — the runbook must NOT tell the user to do nothing."""
    text = combine_runbook.render_runbook(
        make_cfg(method="map", design=True, execution="auto", characterize=False), RR, CFG
    )
    assert "STAGE 2" in text
    # The cluster steps must be present (the bug was omitting them for auto)...
    assert "run_design.sh" in text and "finalize_design.sh" in text
    # ...and clearly conditional on the gate verdict.
    assert "auto" in text and "CLUSTER" in text and "LOCAL" in text


def test_design_handoff_forces_cluster_even_for_auto():
    """The hand-off is only emitted when actually cluster-routed; it must never
    say 'nothing to run' even if execution is auto."""
    text = combine_runbook.design_handoff_text(
        make_cfg(method="map", design=True, execution="auto"), RR, CFG
    )
    assert "run_design.sh" in text
    assert "Nothing extra to run here" not in text


def test_characterize_midway_block_does_not_call_sync():
    """The old docs bug: `sync_models.sh push` on Midway. Sync is Mac-only."""
    text = combine_runbook.render_runbook(
        make_cfg(method="map", design=False, characterize=True), RR, CFG
    )
    # Isolate the STAGE 3 section.
    stage3 = text.split("STAGE 3", 1)[1]
    midway = stage3.split("[MIDWAY]", 1)[1].split("[MAC]", 1)[0]
    assert "sync_models.sh" not in midway
    # The pull happens back on the Mac.
    assert "sync_models.sh pull" in stage3


def test_design_handoff_uses_current_drivers_not_stale_sbatch():
    text = combine_runbook.design_handoff_text(make_cfg(execution="cluster"), RR, CFG)
    assert "run_design.sh" in text and "finalize_design.sh" in text
    # The stale manual-array text must be gone.
    assert "sbatch --array" not in text
    assert "RUNBOOK.txt" in text


def test_runbook_ends_with_newline():
    text = combine_runbook.render_runbook(make_cfg(), RR, CFG)
    assert text.endswith("\n")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
