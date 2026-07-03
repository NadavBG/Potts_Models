"""Snakemake wrapper: freeze the validated derive config into the run dir."""

from _common import load_derive_cfg_from_snakemake, setup_stage_logging

from SBM import derive_config as dc

setup_stage_logging(snakemake, "snapshot_config")  # noqa: F821 (injected by Snakemake)
cfg = load_derive_cfg_from_snakemake(snakemake)  # noqa: F821
dc.dump_config(cfg, snakemake.output[0])  # noqa: F821
