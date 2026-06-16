"""Snakemake wrapper: freeze the validated combine config into the run dir."""

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM import combine_config as cc

setup_stage_logging(snakemake, "snapshot_config")  # noqa: F821 (injected by Snakemake)
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
cc.dump_config(cfg, snakemake.output[0])  # noqa: F821
