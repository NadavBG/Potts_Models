"""Snakemake wrapper: sample one synthetic alignment at one temperature.

One rule invocation per temperature (the ``{temp}`` wildcard) writing to a
fixed ``align_T{temp}.npy`` path, which sidesteps the seed-in-filename
nondeterminism of the default sampler output. The per-temperature seed
(``master_seed + temperature_index``) reproduces the offset the multi-T
sampler applies internally, so samples are bit-identical to the old path.
"""

from _common import load_cfg_from_snakemake, setup_stage_logging

import sample_sbm

temp = snakemake.wildcards.temp  # noqa: F821 (injected by Snakemake)
setup_stage_logging(snakemake, f"sample_T{temp}")  # noqa: F821
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

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
