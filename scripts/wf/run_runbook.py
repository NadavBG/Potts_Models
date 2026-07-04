"""Snakemake wrapper: (re)write the per-run ``RUNBOOK.txt``.

Regenerated on every ``snakemake … all`` from the validated config, so the
copy-pasteable step-by-step can never drift from the params actually in effect.
Shares :func:`SBM.combine_runbook.render_runbook` with ``scripts/new_combine.py``
(which writes the same file at scaffold time) and the design hand-off."""

from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM import combine_runbook

log = setup_stage_logging(snakemake, "runbook")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
run_root = snakemake.params.run_root  # noqa: F821
config_path = snakemake.params.config_path  # noqa: F821

text = combine_runbook.render_runbook(cfg, run_root, config_path)
out = Path(snakemake.output[0])  # noqa: F821
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
log.info("wrote runbook -> %s", out)
