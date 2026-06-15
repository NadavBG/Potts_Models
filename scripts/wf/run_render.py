"""Snakemake wrapper: render the model + synthetic-alignment figures."""

from _common import load_cfg_from_snakemake, setup_stage_logging

import render_figures

setup_stage_logging(snakemake, "render")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

run_root = snakemake.params.run_root  # noqa: F821
argv = [
    run_root,
    "--sector", cfg.figures.sector,
    "--max-seqs-per-group", str(cfg.figures.max_seqs_per_group),
]
if cfg.family:
    argv += ["--family", cfg.family]
if cfg.figures.which:
    argv += ["--figs", *cfg.figures.which]

rc = render_figures.main(argv)
if rc:
    raise SystemExit(rc)
