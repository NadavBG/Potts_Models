"""Snakemake wrapper: emit the Midway hand-off note (cluster execution mode).

When the local/cluster gate chose CLUSTER (predicted local wall-time over the
budget, or ``design.execution: cluster``), the pipeline stops after writing
``design_config.json``. This writes ``design/MIDWAY_HANDOFF.txt`` telling the user
how to run the annealing array on Midway and pull the results back.

The text is the Stage-2 block of the master runbook (``SBM.combine_runbook``), so
the hand-off can never drift from ``RUNBOOK.txt`` or the real one-argument cluster
drivers. Full runbook: ``docs/RUNBOOK.md`` / ``docs/DESIGN_TWO_MODEL.md``."""

from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM import combine_runbook

log = setup_stage_logging(snakemake, "design_handoff")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
run_root = snakemake.params.run_root  # noqa: F821
config_path = snakemake.params.config_path  # noqa: F821

text = combine_runbook.design_handoff_text(cfg, run_root, config_path)
out = Path(snakemake.output[0])  # noqa: F821
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
log.info("wrote Midway hand-off note -> %s", out)
