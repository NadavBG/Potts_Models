"""Snakemake wrapper: DCAlign convergence report (non-convergence per group).

Reads ``models.json`` + ``query/groups.json`` and the on-disk DCAlign cache for
both models, then drives ``scripts/report_dcalign_convergence.py`` to write a
tidy per-(model, group) TSV, a summary JSON, a provenance manifest, and (when
figures are enabled) the bar figure. Pure cache read — no Julia, no OpenMP.
Only defined for ``scoring.method == "dcalign"``.
"""

from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

setup_stage_logging(snakemake, "dcalign_convergence")  # noqa: F821
load_combine_cfg_from_snakemake(snakemake)  # noqa: F821 (validate; identity comes from models.json)

import report_dcalign_convergence  # noqa: E402

run_root = snakemake.params.run_root  # noqa: F821
out = snakemake.output  # noqa: F821

argv = [
    "--models-json", str(snakemake.input.models),  # noqa: F821
    "--groups", str(snakemake.input.groups),  # noqa: F821
    "--dcalign-cache", str(Path(run_root) / "dcalign" / "cache"),
    "--output", str(out.tsv),
    "--summary", str(out.summary),
    "--manifest", str(out.manifest),
]
if "fig" in out.keys():
    argv += ["--figure", str(out.fig)]

rc = report_dcalign_convergence.main(argv)
if rc:
    raise SystemExit(rc)
