"""Snakemake wrapper: build the couplings (J) pruning mask."""

from _common import load_cfg_from_snakemake, setup_stage_logging

import build_mask

setup_stage_logging(snakemake, "build_mask_J")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

spec = cfg.pruning.couplings
build_mask.main(
    snakemake.input.msa,  # noqa: F821 (run-local encoded MSA from encode_msa)
    theta=cfg.pruning.theta,
    lbda=cfg.pruning.lbda,
    strategies=[spec.strategy],
    output_label=cfg.pruning.label,
    pct_J=[spec.percent],
    pct_h=[],
    Dia_prior=cfg.pruning.Dia_prior,
    out_file=str(snakemake.output[0]),  # noqa: F821
)
