"""Snakemake wrapper: potts_align ground-state-recovery baseline.

Reads ``models.json`` (model names + paths) and the query set, then drives
``scripts/compare_potts_align_baseline.py`` over the on-disk potts_align cluster
cache to write a tidy ``potts_align_vs_inframe.tsv``, a summary JSON, a provenance
manifest, and (when figures are enabled) the consolidated recovery figure. Pure
numpy energy recompute — no Julia, no OpenMP — so it runs anywhere the rest of the
Mac-side combine pipeline runs. Only defined for ``scoring.method == "potts_align"``.
"""

from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

setup_stage_logging(snakemake, "potts_align_baseline")  # noqa: F821
load_combine_cfg_from_snakemake(snakemake)  # noqa: F821 (validate; identity comes from models.json)

import compare_potts_align_baseline  # noqa: E402

run_root = snakemake.params.run_root  # noqa: F821
out = snakemake.output  # noqa: F821

argv = [
    "--models-json", str(snakemake.input.models),  # noqa: F821
    "--fasta", str(snakemake.input.fasta),  # noqa: F821
    "--groups", str(snakemake.input.groups),  # noqa: F821
    "--potts-align-cache", str(Path(run_root) / "potts_align" / "cache"),
    "--output", str(out.tsv),
    "--summary", str(out.summary),
    "--manifest", str(out.manifest),
]
if "fig" in out.keys():
    argv += ["--figure", str(out.fig)]

rc = compare_potts_align_baseline.main(argv)
if rc:
    raise SystemExit(rc)
