"""Snakemake wrapper: render the two design figures into ``<run_root>/figs/``.

Writes ``figs/design_trajectories.pdf`` and ``figs/design_phase_space.pdf`` (beside
``two_model_energy.pdf``) from the design run's ``trajectories.npz`` +
``design_manifest.json``, overlaying the natural clouds from the combine run's
``data/scores.tsv``."""

from pathlib import Path

from _common import setup_stage_logging

import render_design

setup_stage_logging(snakemake, "design_render")  # noqa: F821
run_root = Path(snakemake.params.run_root)  # noqa: F821
render_design.main([
    "--design-dir", str(run_root / "design"),
    "--figs-dir", str(run_root / "figs"),
    "--scores-tsv", str(snakemake.input.scores),  # noqa: F821
])
