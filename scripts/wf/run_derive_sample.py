"""Snakemake wrapper: sample one synthetic alignment from the derived model.

Mirrors ``run_sample.py`` but loads the derive schema (the two pipelines are
kept decoupled). The derived model samples through the standard MCMC kernel;
a fields-only model (``J ≡ 0``) samples independent-site sequences.
"""

from _common import load_derive_cfg_from_snakemake, setup_stage_logging

import sample_sbm

temp = snakemake.wildcards.temp  # noqa: F821 (injected by Snakemake)
setup_stage_logging(snakemake, f"sample_T{temp}")  # noqa: F821
cfg = load_derive_cfg_from_snakemake(snakemake)  # noqa: F821

run_root = snakemake.params.run_root  # noqa: F821
temps = [f"{t:.10g}" for t in cfg.sample.temperatures]
per_temp_seed = cfg.seed + temps.index(temp)

argv = [
    run_root,
    "--temperature", temp,
    "--N", str(cfg.sample.N),
    "--seed", str(per_temp_seed),
    "--output", str(snakemake.output.align),  # noqa: F821
    "--force",
]
rc = sample_sbm.main(argv)
if rc:
    raise SystemExit(rc)
