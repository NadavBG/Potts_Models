"""Snakemake wrapper: render the design-characterization figures + stats.

Renders ``figs/characterization_overview.pdf``, ``figs/tm_A_vs_B.pdf``,
``figs/fold_call_breakdown.pdf`` and ``characterize/data/characterization_stats.tsv``
from Midway's merged ``characterize/data/summary.tsv`` (+ optional
``natural_summary.tsv``). Pure numpy/matplotlib — the heavy characterization
compute runs on Midway; this only renders (docs/CHARACTERIZE.md)."""

import sys
from pathlib import Path

from _common import setup_stage_logging

# render_characterize.py lives in scripts/characterize/, not directly on the
# scripts/ dir that _common already put on sys.path.
_CHAR_DIR = Path(__file__).resolve().parent.parent / "characterize"
if str(_CHAR_DIR) not in sys.path:
    sys.path.insert(0, str(_CHAR_DIR))
import render_characterize  # noqa: E402

setup_stage_logging(snakemake, "characterize_render")  # noqa: F821
run_root = Path(snakemake.params.run_root)  # noqa: F821
summary_tsv = Path(snakemake.input.summary)  # noqa: F821
natural_tsv = summary_tsv.parent / "natural_summary.tsv"

argv = [
    "--summary", str(summary_tsv),
    "--figs-dir", str(run_root / "figs"),
    "--stats-out", str(summary_tsv.parent / "characterization_stats.tsv"),
]
if natural_tsv.exists():
    argv += ["--natural-summary", str(natural_tsv)]
render_characterize.main(argv)
