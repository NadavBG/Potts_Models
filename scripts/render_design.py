#!/usr/bin/env python
"""Render the two figures for a two-model design run (``design_two_model.py``).

Reads a design run directory's ``trajectories.npz`` + ``design_manifest.json`` and
writes ``figs/design_trajectories.pdf`` and ``figs/design_phase_space.pdf``. The
phase-space figure overlays the natural clouds from the combine run's
``data/scores.tsv`` (discovered via the manifest, or ``--scores-tsv``); pass
``--no-natives`` to skip the overlay.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from SBM.utils.utils_design_plot import (
    load_trajectories,
    render_alignment,
    render_lengths,
    render_phase_space,
    render_trajectories,
)

log = logging.getLogger("render_design")


def _discover_scores_tsv(manifest: dict) -> Path | None:
    """The combine run's scores.tsv sits next to the energy_weights.json input."""
    weights_input = manifest.get("inputs", {}).get("energy_weights") or {}
    path = weights_input.get("path")
    if not path:
        return None
    candidate = Path(path).parent / "scores.tsv"
    return candidate if candidate.is_file() else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design-dir", required=True, help="a design run dir (has trajectories.npz)")
    p.add_argument("--figs-dir", default=None,
                   help="where to write the PDFs (default <design-dir>/figs; the combine "
                        "pipeline passes <run_root>/figs)")
    p.add_argument("--scores-tsv", default=None, help="override the natives-overlay scores.tsv")
    p.add_argument("--no-natives", action="store_true", help="skip the natives overlay")
    p.add_argument("--cold-fraction", type=float, default=0.25,
                   help="trailing fraction of records pooled for the landing heatmap")
    p.add_argument("--no-letters", action="store_true",
                   help="ZAPPO alignment figure: color cells only, omit residue letters (faster)")
    args = p.parse_args(argv)

    design_dir = Path(args.design_dir)
    manifest = json.loads((design_dir / "design_manifest.json").read_text(encoding="utf-8"))
    models = manifest["extra"]["models"]
    weights = manifest["extra"]["weights"]
    model_names = (models["A"]["name"], models["B"]["name"])
    w = (float(weights["w_A"]), float(weights["w_B"]))

    if args.no_natives:
        scores_tsv = None
    elif args.scores_tsv:
        scores_tsv = Path(args.scores_tsv)
    else:
        scores_tsv = _discover_scores_tsv(manifest)

    traj = load_trajectories(design_dir / "trajectories.npz")
    figs = Path(args.figs_dir) if args.figs_dir else design_dir / "figs"
    render_trajectories(traj, model_names, figs / "design_trajectories.pdf")
    render_phase_space(traj, model_names, w, figs / "design_phase_space.pdf",
                       scores_tsv=scores_tsv, cold_fraction=args.cold_fraction)
    render_lengths(traj, model_names, figs / "design_lengths.pdf")

    aln_A, aln_B = design_dir / "design_aln_A.fasta", design_dir / "design_aln_B.fasta"
    if aln_A.is_file() and aln_B.is_file():
        render_alignment(aln_A, aln_B, model_names, figs / "design_alignment.pdf",
                         show_letters=not args.no_letters)
    else:
        log.warning("no aligned FASTAs (%s / %s); skipping ZAPPO alignment figure", aln_A, aln_B)
    log.info("wrote figures under %s", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
