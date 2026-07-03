#!/usr/bin/env python3
"""Figures for the design characterization. CPU stage (.venv).

Reads ``summary.tsv`` (designs) + ``natural_summary.tsv`` (controls) and writes
four PDFs into ``--figs-dir`` via ``lab_plotting.save_figure``:

  1. ``plddt_distribution.pdf`` — mean pLDDT, designs vs CM/PPIC naturals.
  2. ``tm_A_vs_B.pdf``          — the money plot: TM to fold A vs fold B.
  3. ``energy_vs_structure.pdf``— design ΔE (E_A−E_B) vs Δ TM-score (TM_A−TM_B).
  4. ``blast_identity.pdf``     — design best %id to SwissProt / CM / PPIC.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# scripts/lab_plotting.py is a sibling of the parent scripts/ dir.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

from SBM.characterize import summary  # noqa: E402

logger = logging.getLogger("render_characterize")

# Group colors: naturals green (lab convention), designs orange highlight.
_GROUP_COLOR = {
    "design": lab_plotting.LAB_COLORS["highlight"],       # orange
    "CM-natural": "#117733",                              # Tol dark green
    "PPIC-natural": lab_plotting.WONG_PALETTE[3],         # bluish green
}
_GROUP_LABEL = {"design": "Designs", "CM-natural": "CM naturals",
                "PPIC-natural": "PPIC naturals"}


def _col(rows: list[dict[str, str]], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key, "")
        out.append(float(v) if v not in ("", None) else math.nan)
    return np.asarray(out, dtype=float)


def _by_group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r.get("group", ""), []).append(r)
    return groups


def _ordered_groups(groups: dict[str, list]) -> list[str]:
    order = ["design", "CM-natural", "PPIC-natural"]
    return [g for g in order if g in groups] + [g for g in groups if g not in order]


def fig_plddt(all_rows: list[dict[str, str]], figs_dir: Path) -> None:
    groups = _by_group(all_rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for g in _ordered_groups(groups):
        vals = _col(groups[g], "plddt_mean")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=np.linspace(0, 100, 26), density=True, histtype="step",
                linewidth=1.8, color=_GROUP_COLOR.get(g, "0.4"),
                label=f"{_GROUP_LABEL.get(g, g)} (n={vals.size})")
    ax.axvline(70, color=lab_plotting.LAB_COLORS["chrome"], linestyle="--", linewidth=1)
    ax.set_xlabel("mean pLDDT (0–100)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    lab_plotting.save_figure(fig, figs_dir / "plddt_distribution.pdf",
                             script_path=Path(__file__))
    plt.close(fig)


def fig_tm_scatter(all_rows: list[dict[str, str]], figs_dir: Path) -> None:
    groups = _by_group(all_rows)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    # Naturals first (background), designs on top.
    for g in reversed(_ordered_groups(groups)):
        rows = groups[g]
        tm_a, tm_b = _col(rows, "tm_A"), _col(rows, "tm_B")
        is_design = g == "design"
        ax.scatter(tm_a, tm_b, s=28 if is_design else 10,
                   color=_GROUP_COLOR.get(g, "0.4"),
                   alpha=0.9 if is_design else 0.35,
                   edgecolor="black" if is_design else "none",
                   linewidth=0.4 if is_design else 0,
                   label=f"{_GROUP_LABEL.get(g, g)} (n={np.isfinite(tm_a).sum()})",
                   zorder=3 if is_design else 2)
    ax.plot([0, 1], [0, 1], color=lab_plotting.LAB_COLORS["chrome"],
            linestyle="--", linewidth=1, zorder=1)
    for t in (0.5,):
        ax.axvline(t, color="0.8", linewidth=0.8, zorder=0)
        ax.axhline(t, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("TM-score to fold A (chorismate mutase, 1ECM)")
    ax.set_ylabel("TM-score to fold B (PPIC, 1JNT)")
    ax.legend(frameon=False, fontsize="small", loc="upper right")
    fig.tight_layout()
    lab_plotting.save_figure(fig, figs_dir / "tm_A_vs_B.pdf", script_path=Path(__file__))
    plt.close(fig)


def fig_energy_vs_structure(design_rows: list[dict[str, str]], figs_dir: Path) -> None:
    d_e = _col(design_rows, "delta_E")   # E_A - E_B
    d_tm = _col(design_rows, "delta_tm")  # TM_A - TM_B
    mask = np.isfinite(d_e) & np.isfinite(d_tm)
    if mask.sum() < 3:
        logger.warning("energy_vs_structure: < 3 finite pairs; skipping figure")
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.scatter(d_e[mask], d_tm[mask], s=26, color=_GROUP_COLOR["design"],
               edgecolor="black", linewidth=0.4, zorder=3)
    try:
        lab_plotting.fit_line(ax, d_e[mask], d_tm[mask], confidence=0.95)
    except ValueError as exc:  # e.g. degenerate x
        logger.warning("fit_line skipped: %s", exc)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_xlabel("energy preference  E_A − E_B  (model energy units)")
    ax.set_ylabel("structure preference  TM_A − TM_B")
    fig.tight_layout()
    lab_plotting.save_figure(fig, figs_dir / "energy_vs_structure.pdf",
                             script_path=Path(__file__))
    plt.close(fig)


def fig_blast_identity(design_rows: list[dict[str, str]], figs_dir: Path) -> None:
    specs = [("SwissProt", "swissprot_pident", lab_plotting.WONG_PALETTE[5]),
             ("CM family", "cmfam_pident", _GROUP_COLOR["CM-natural"]),
             ("PPIC family", "ppicfam_pident", _GROUP_COLOR["PPIC-natural"])]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    any_data = False
    for label, key, color in specs:
        if key not in (design_rows[0] if design_rows else {}):
            continue
        vals = _col(design_rows, key)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        any_data = True
        ax.hist(vals, bins=np.linspace(0, 100, 21), histtype="step", linewidth=1.8,
                color=color, label=f"{label} (n={vals.size})")
    if not any_data:
        logger.warning("blast_identity: no BLAST hits to plot; skipping figure")
        plt.close(fig)
        return
    ax.set_xlabel("best-hit identity (%)")
    ax.set_ylabel("number of designs")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    lab_plotting.save_figure(fig, figs_dir / "blast_identity.pdf",
                             script_path=Path(__file__))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, type=Path, help="design summary.tsv")
    p.add_argument("--natural-summary", type=Path, default=None)
    p.add_argument("--figs-dir", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        plt.style.use("lab-paper")
    except OSError:
        logger.info("'lab-paper' stylesheet not available; using matplotlib defaults")

    design_rows = summary.read_tsv(args.summary)
    natural_rows = summary.read_tsv(args.natural_summary) if args.natural_summary else []
    all_rows = design_rows + natural_rows
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    fig_plddt(all_rows, args.figs_dir)
    fig_tm_scatter(all_rows, args.figs_dir)
    fig_energy_vs_structure(design_rows, args.figs_dir)
    fig_blast_identity(design_rows, args.figs_dir)
    logger.info("wrote figures -> %s", args.figs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
