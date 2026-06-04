"""Snakemake wrapper: freeze the validated config into the run dir."""

from _common import load_cfg_from_snakemake, setup_stage_logging

from SBM import workflow_config as wc

setup_stage_logging(snakemake, "snapshot_config")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821
wc.dump_config(cfg, snakemake.output[0])  # noqa: F821
