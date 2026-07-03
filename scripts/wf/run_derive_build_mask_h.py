"""Snakemake wrapper: build the fields (h) mask for a derive run.

Same machinery as the single-model ``build_mask_h`` rule, but reads the mask
spec off ``cfg.filter.fields`` (a MaskSpec) and the source-copied MSA.
"""

from _common import load_derive_cfg_from_snakemake, setup_stage_logging

import build_mask

setup_stage_logging(snakemake, "build_mask_h")  # noqa: F821 (injected by Snakemake)
cfg = load_derive_cfg_from_snakemake(snakemake)  # noqa: F821

spec = cfg.filter.fields_mask
build_mask.main(
    snakemake.input.msa,  # noqa: F821 (run-local MSA copied by copy_inputs)
    theta=cfg.filter.theta,
    lbda=cfg.filter.lbda,
    strategies=[spec.strategy],
    output_label=cfg.filter.label,
    pct_J=[],
    pct_h=[spec.percent],
    Dia_prior=cfg.filter.Dia_prior,
    out_file=str(snakemake.output[0]),  # noqa: F821
)
