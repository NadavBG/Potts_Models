"""Snakemake wrapper: run the two-model design anneal locally (on the Mac).

Runs every chain from ``design_config.json`` (the ``design_config`` rule), parallel
over the job's threads, writing ``trajectories.npz``, ``designed.tsv``, the designed
FASTA, and ``design_manifest.json`` into ``<run_root>/design/``. Used only when the
Snakefile's local/cluster gate chose LOCAL.

It shells out to ``scripts/design_two_model.py --from-config`` (rather than calling
``run_from_config`` in-process) on purpose: the anneal parallelizes with a
``ProcessPoolExecutor`` whose ``spawn`` workers re-import ``__main__``. In a
Snakemake ``script:`` the injected ``__main__`` references the ``snakemake`` object
and crashes the workers; as a subprocess, ``__main__`` is the guarded CLI, so the
workers re-import cleanly (the same path the CLI already uses)."""

import subprocess
import sys
from pathlib import Path

from _common import setup_stage_logging

import design_two_model as d2m

log = setup_stage_logging(snakemake, "design_local")  # noqa: F821
config_path = Path(snakemake.input.config)  # noqa: F821
out_dir = config_path.parent          # <run_root>/design
jobs = max(1, int(snakemake.threads))  # noqa: F821

cmd = [sys.executable, str(Path(d2m.__file__).resolve()), "--from-config", str(config_path),
       "--out-dir", str(out_dir), "--jobs", str(jobs)]
log.info("design_local: %s", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:                    # stream through the tee'd stdout -> per-stage log
    print(line, end="")
rc = proc.wait()
if rc != 0:
    raise SystemExit(f"design_two_model --from-config failed (exit {rc})")
