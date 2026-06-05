"""Snakemake wrapper: MSA-only statistics figure (no model dependency)."""

from _common import load_cfg_from_snakemake, setup_stage_logging

import render_msa_stats

setup_stage_logging(snakemake, "msa_stats")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

argv = [
    "--msa", snakemake.input.msa,  # noqa: F821 (run-local encoded MSA from encode_msa)
    "--theta", str(cfg.msa_stats.theta),
    "--lbda", str(cfg.msa_stats.lbda),
    "--Dia-prior", cfg.msa_stats.Dia_prior,
    "--sector", cfg.msa_stats.sector,
    "--out", str(snakemake.output[0]),  # noqa: F821
]
rc = render_msa_stats.main(argv)
if rc:
    raise SystemExit(rc)
