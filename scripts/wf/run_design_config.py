"""Snakemake wrapper: resolve the two-model design run spec (design_config.json).

Builds the self-contained design config (model paths, combining weights, and the
*seeded* natural-start row selection) from the combine run's ``models.json`` +
``data/energy_weights.json`` via ``design_two_model.resolve_design_config``, and
writes it to ``<run_root>/design/design_config.json``. Both the local runner and
the Midway cluster shards consume this one file, so local and cluster runs
reproduce identical starts."""

import json
from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

import design_two_model as d2m
from SBM.design.anneal import AnnealSchedule

log = setup_stage_logging(snakemake, "design_config")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
d = cfg.design
run_root = snakemake.params.run_root  # noqa: F821

sched = AnnealSchedule(
    n_steps=d.steps, beta_start=d.beta_start, beta_end=d.beta_end,
    teleport_frac=d.teleport_frac, min_length=d.min_length, record_every=d.record_every,
)
config = d2m.resolve_design_config(
    combine_run=run_root, schedule=sched, master_seed=d.seed,
    start_random=d.start_random, start_natural_a=d.start_natural_a,
    start_natural_b=d.start_natural_b, do_polish=d.polish, polish_schedule=d.polish_schedule,
    w_a=d.w_a, w_b=d.w_b,
)

out = Path(snakemake.output[0])  # noqa: F821
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
log.info("design_config: %d chains (%d random / %d %s / %d %s), polish=%s -> %s",
         config["n_chains"], config["start_random"], config["start_natural_a"],
         config["name_a"], config["start_natural_b"], config["name_b"],
         config["polish_schedule"] if config["do_polish"] else "off", out)
