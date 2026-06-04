"""Snakemake wrapper: ProteinMPNN foldability sweep (alternate sampling mode).

Needs ``PROTEINMPNN_PATH`` set (or ``mpnn.path`` could be added) unless
``mpnn.skip_scoring`` is true. The output dir name embeds the seed, which
is deterministic from the config, so the rule can declare it as output.
"""

from _common import load_cfg_from_snakemake, setup_stage_logging

import sample_sbm

setup_stage_logging(snakemake, "mpnn_sweep")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

run_root = snakemake.params.run_root  # noqa: F821
m = cfg.mpnn

argv = [
    run_root,
    "--mpnn-sweep",
    "--force",  # Snakemake owns the output dir's lifecycle; allow clean re-runs
    "--seed", str(cfg.mpnn_seed),
    "--mpnn-N-per-T", str(m.N_per_T),
    "--mpnn-temperatures", *[f"{t:.10g}" for t in m.temperatures],
    "--mpnn-controls", *m.controls,
    "--mpnn-pdb", m.pdb,
    "--mpnn-chain", m.chain,
    "--mpnn-model-name", m.model_name,
]
if m.skip_scoring:
    argv.append("--mpnn-skip-scoring")

rc = sample_sbm.main(argv)
if rc:
    raise SystemExit(rc)
