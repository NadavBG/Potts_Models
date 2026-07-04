#!/usr/bin/env python3
"""Render the conceptual Mac<->Midway two-model combine workflow diagram.

A hand-drawn overview (not a Snakemake DAG): the three stages (combine, design,
characterize) as rows, each a MAC -> [push] -> MIDWAY -> [pull] -> MAC round-trip,
mirroring exactly the stages emitted in a run's ``RUNBOOK.txt``. Lands
``docs/workflow/combine_workflow.pdf``. Regenerated (with the two Snakemake DAGs)
by ``scripts/render_dag.sh``.

Colours follow ``lab_plotting.LAB_COLORS`` (Mac = blue negative_control, Midway =
orange highlight, sync arrows = grey chrome); saved via ``lab_plotting.save_figure``
so git provenance is baked into the PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import lab_plotting

_MAC = lab_plotting.LAB_COLORS["negative_control"]   # blue
_MIDWAY = lab_plotting.LAB_COLORS["highlight"]        # orange
_SYNC = lab_plotting.LAB_COLORS["chrome"]             # grey

# Column x-centres (axes fraction): stage label, then the MAC / MIDWAY / MAC boxes.
_LABEL_X = 0.015
_COL_X = {"mac1": 0.30, "midway": 0.56, "mac2": 0.83}
_BOX_W = 0.205
_BOX_H = 0.115

# Each row: (stage label, box specs). A box spec = (column, machine, text) or None
# to leave that column empty (stage 3 has no leading Mac step).
_ROWS = [
    (
        "SET UP",
        [
            ("mac1", "mac", "new_combine.py\n→ config + RUNBOOK.txt"),
            None,
            None,
        ],
    ),
    (
        "1  COMBINE",
        [
            ("mac1", "mac", "build inputs\n(query, models.json)"),
            ("midway", "midway", "potts_align cache\nrun_…align + finalize"),
            ("mac2", "mac", "score + E_tot weights\n+ figures"),
        ],
    ),
    (
        "2  DESIGN",
        [
            ("mac1", "mac", "design spec\n(from `all`)"),
            ("midway", "midway", "joint anneal\nrun_design + finalize"),
            ("mac2", "mac", "design figures"),
        ],
    ),
    (
        "3  CHARACTERIZE",
        [
            None,
            ("midway", "midway", "ESMFold + TM-align\n+ BLAST · run_characterize"),
            ("mac2", "mac", "characterization\nfigures"),
        ],
    ),
]


def _draw_box(ax, cx: float, cy: float, text: str, machine: str) -> None:
    face = _MAC if machine == "mac" else _MIDWAY
    ax.add_patch(
        FancyBboxPatch(
            (cx - _BOX_W / 2, cy - _BOX_H / 2), _BOX_W, _BOX_H,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0, edgecolor="white", facecolor=face, alpha=0.92,
            mutation_aspect=0.5, zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", color="white",
            fontsize=7.5, zorder=4, linespacing=1.15)


def _sync_arrow(ax, x0: float, x1: float, y: float, label: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.4, color=_SYNC, zorder=2,
        )
    )
    ax.text((x0 + x1) / 2, y + 0.028, label, ha="center", va="bottom",
            color=_SYNC, fontsize=6.5, style="italic")


def build_figure() -> plt.Figure:
    plt.style.use("lab-paper")
    n_rows = len(_ROWS)
    # Inch budget: a per-row band + top title + bottom legend, width for 3 columns.
    row_h = 0.95
    fig_h = row_h * n_rows + 1.1
    fig_w = 9.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")  # single full-figure axes; no gridspec to manage
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bottom = 0.86, 0.14
    ys = [top - (top - bottom) * i / (n_rows - 1) for i in range(n_rows)]

    ax.text(0.5, 0.965, "Two-model combine pipeline — Mac ↔ Midway",
            ha="center", va="center", fontsize=12, fontweight="bold")

    for (label, specs), cy in zip(_ROWS, ys):
        ax.text(_LABEL_X, cy, label, ha="left", va="center", fontsize=8.5,
                fontweight="bold", color="0.25")
        present = {}
        for spec in specs:
            if spec is None:
                continue
            col, machine, text = spec
            cx = _COL_X[col]
            _draw_box(ax, cx, cy, text, machine)
            present[col] = machine
        # Sync / flow arrows between adjacent occupied columns of this row.
        cols = ["mac1", "midway", "mac2"]
        occ = [c for c in cols if c in present]
        for a, b in zip(occ, occ[1:]):
            x0 = _COL_X[a] + _BOX_W / 2
            x1 = _COL_X[b] - _BOX_W / 2
            ma, mb = present[a], present[b]
            if ma == "mac" and mb == "midway":
                lbl = "push"
            elif ma == "midway" and mb == "mac":
                lbl = "pull"
            else:
                lbl = ""
            _sync_arrow(ax, x0, x1, cy, lbl)

    # Legend + note.
    ax.add_patch(FancyBboxPatch((0.30, 0.03), 0.02, 0.02, boxstyle="round,pad=0.002",
                                facecolor=_MAC, edgecolor="none"))
    ax.text(0.325, 0.04, "Mac (figures, scoring)", va="center", fontsize=7.5)
    ax.add_patch(FancyBboxPatch((0.55, 0.03), 0.02, 0.02, boxstyle="round,pad=0.002",
                                facecolor=_MIDWAY, edgecolor="none"))
    ax.text(0.575, 0.04, "Midway (heavy compute)", va="center", fontsize=7.5)
    ax.text(0.015, 0.04, "push / pull = scripts/sync_models.sh", va="center",
            fontsize=6.8, style="italic", color=_SYNC)
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the combine workflow overview diagram.")
    p.add_argument("--out-dir", default="docs/workflow", help="output directory")
    args = p.parse_args(argv)
    fig = build_figure()
    out_dir = Path(args.out_dir)
    # PDF is the canonical, provenance-stamped artifact.
    pdf = out_dir / "combine_workflow.pdf"
    lab_plotting.save_figure(fig, pdf, script_path=__file__)
    print(f"Wrote {pdf}")
    # PNG raster purely for inline embedding in docs/RUNBOOK.md (markdown renders
    # neither PDF nor matplotlib-metadata SVG). Bare savefig is intentional here:
    # this is a derived preview of the provenance-stamped PDF above, not a new
    # data figure. Kept in lock-step by scripts/render_dag.sh.
    png = out_dir / "combine_workflow.png"
    fig.savefig(png, dpi=150)
    print(f"Wrote {png}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
