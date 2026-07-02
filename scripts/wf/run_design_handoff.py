"""Snakemake wrapper: emit the Midway hand-off note (cluster execution mode).

When the local/cluster gate chose CLUSTER (predicted local wall-time over the
budget, or ``design.execution: cluster``), the pipeline stops after writing
``design_config.json``. This writes a short ``MIDWAY_HANDOFF.txt`` telling the user
how to run the annealing array on Midway and pull the results back. Full runbook:
``docs/DESIGN_TWO_MODEL.md``."""

from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

log = setup_stage_logging(snakemake, "design_handoff")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
d = cfg.design
run_root = snakemake.params.run_root  # noqa: F821

text = f"""\
Two-model design — CLUSTER execution
====================================

The predicted local wall-time exceeded design.local_budget_minutes (or
design.execution was 'cluster'), so the anneal was NOT run on the Mac. The run
spec is written to:

    {run_root}/design/design_config.json

It requests {d.n_chains} chains ({d.start_random} random / {d.start_natural_a} CM /
{d.start_natural_b} PPIC), {d.steps} steps, polish={d.polish_schedule if d.polish else 'off'},
sharded over {d.n_shards} Slurm array tasks.

Run it on Midway, then pull the results back:

  1. (Mac) commit code + config, then push the models + config:
         git add -A && git commit
         bash scripts/sync_models.sh push

  2. (Midway) run the array (see the Midway Claude TODO in docs/DESIGN_TWO_MODEL.md
     for the pipeline/external/*design* sbatch scripts):
         scripts/wf/run_design_shard.py plan --run-dir {run_root}/design --n-shards {d.n_shards}
         sbatch --array=0-{d.n_shards - 1} ... pipeline/external/sbatch_design_shard.sh
         # then, afterok, the gather:
         scripts/wf/run_design_gather.py --run-dir {run_root}/design

  3. (Mac) pull the gathered artifacts, then render the figures. `all` will NOT render
     in cluster mode (it stops at this hand-off), so target the figures explicitly:
         bash scripts/sync_models.sh pull
         snakemake -s Snakefile.combine --configfile <this config> --config run_root={run_root} \\
             --cores 8 {run_root}/figs/design_alignment.pdf
         # (equivalently, and without snakemake:)
         python scripts/render_design.py --design-dir {run_root}/design --figs-dir {run_root}/figs

To instead run the anneal locally despite the estimate, set design.execution: local (or
raise design.local_budget_minutes) and re-run.
"""

out = Path(snakemake.output[0])  # noqa: F821
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
log.info("wrote Midway hand-off note -> %s", out)
