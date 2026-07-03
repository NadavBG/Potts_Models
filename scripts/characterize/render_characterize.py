#!/usr/bin/env python3
"""Render the design-characterization figures + stats (Mac CPU stage, .venv).

Reads Midway's merged ``summary.tsv`` (designs) + optional ``natural_summary.tsv``
(controls) and writes, via ``SBM.utils.utils_characterize_plot`` (lab_plotting +
inch-budget layout):

  * ``characterization_overview.pdf`` — consolidated 2x2 (fold / pLDDT / energy-vs-
    structure / BLAST).
  * ``tm_A_vs_B.pdf``            — standalone "which fold?" scatter.
  * ``fold_call_breakdown.pdf``  — fold-call composition per group.
  * ``characterization_stats.tsv`` — tidy (group, metric, value) summary.

Pure numpy/matplotlib — no TMalign/blastp/torch. The heavy characterization
compute (fold + TM + BLAST + merge) runs on Midway; this only renders.

The CLI flags (``--summary``/``--natural-summary``/``--figs-dir``) are kept
compatible with the Midway ``characterize.py`` driver; ``--stats-out`` is new
and optional (defaults beside the figures).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from SBM.characterize import summary
from SBM.utils import utils_characterize_plot as ucp

logger = logging.getLogger("render_characterize")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, type=Path, help="design summary.tsv")
    p.add_argument("--natural-summary", type=Path, default=None,
                   help="natural_summary.tsv (controls); overlaid where present")
    p.add_argument("--figs-dir", required=True, type=Path)
    p.add_argument("--stats-out", type=Path, default=None,
                   help="tidy stats TSV (default: <figs-dir>/characterization_stats.tsv)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    design_rows = summary.read_tsv(args.summary)
    natural_rows = (summary.read_tsv(args.natural_summary)
                    if args.natural_summary and args.natural_summary.exists() else [])
    if args.natural_summary and not args.natural_summary.exists():
        logger.warning("natural summary %s not found; rendering designs only",
                       args.natural_summary)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    ucp.render_overview(design_rows, natural_rows,
                        args.figs_dir / "characterization_overview.pdf")
    ucp.render_tm_scatter(design_rows, natural_rows,
                          args.figs_dir / "tm_A_vs_B.pdf")
    ucp.render_fold_call_breakdown(design_rows, natural_rows,
                                   args.figs_dir / "fold_call_breakdown.pdf")
    stats_out = args.stats_out or (args.figs_dir / "characterization_stats.tsv")
    ucp.write_stats(design_rows, natural_rows, stats_out)

    logger.info("wrote characterization figures -> %s (stats -> %s)",
                args.figs_dir, stats_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
