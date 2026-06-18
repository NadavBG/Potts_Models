"""Consolidated figure for the DCAlign-vs-in-frame baseline diagnostic.

One figure answers "does couplings-aware DCAlign beat the native frame, or fall
into Blocker 1 (spec §10.8)?": for each model, a scatter of DCAlign's energy
against the native in-frame energy with the ``y = x`` diagonal (points **above**
the line are DCAlign-worse), and the distribution of ``ΔE = E_dcalign − E_inframe``
(mass to the right of 0 is the pathology). Rows are the two models; only one
figure file, per the lab "consolidate similar figures" convention. Routed through
``scripts/lab_plotting.py`` so the lab palette + PDF provenance stay the single
source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scripts/lab_plotting.py is a sibling script, not a package module. Add the
# scripts/ dir to sys.path so ``import lab_plotting`` works regardless of cwd
# (mirrors src/SBM/utils/utils_energy_plot.py and utils_mpnn_plot.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

#: Per-panel inch budgets (figsize is derived from these × the grid, not guessed).
_SCATTER_W, _HIST_W, _ROW_H = 3.0, 2.6, 3.0
_MARGIN_W, _MARGIN_H = 1.0, 0.9

#: Fallback worse-than-native threshold (a.u.); the CLI passes the run's actual
#: ``equal_tol`` (DEFAULT_EQUAL_TOL in SBM.energy.dcalign_baseline) so this is
#: only used when the figure is rendered standalone.
_DEFAULT_EQUAL_TOL = 1.0


def _group_colors(groups: list[str]) -> dict[str, str]:
    """Assign each source group a stable, colorblind-safe color (reserve black)."""
    palette = lab_plotting.WONG_PALETTE[1:]
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(groups))}


def _scatter(ax, sub: pd.DataFrame, colors: dict[str, str], model: str) -> None:
    """E_dcalign vs E_inframe with the y=x diagonal; above the line = DCAlign worse.

    Points are colored by source group; not-converged alignments (DCAlign fell
    back to decimation) are overlaid as open black rings so one can read whether
    the worse-than-native mass is converged-on-a-bad-frame (un-ringed, above the
    diagonal) or merely un-converged (ringed).
    """
    for g in sorted(sub["group"].unique()):
        gs = sub[sub["group"] == g]
        ax.scatter(gs["e_inframe"], gs["e_dcalign"], s=8, alpha=0.6,
                   color=colors[g], label=g, linewidths=0)
    not_conv = sub[~sub["converged"]] if "converged" in sub.columns else sub.iloc[:0]
    if len(not_conv):
        ax.scatter(not_conv["e_inframe"], not_conv["e_dcalign"], s=42,
                   facecolors="none", edgecolors=lab_plotting.LAB_COLORS["chrome"],
                   linewidths=0.9, zorder=4,
                   label=f"not converged (n={len(not_conv)})")
    lo = float(np.nanmin([sub["e_inframe"].min(), sub["e_dcalign"].min()]))
    hi = float(np.nanmax([sub["e_inframe"].max(), sub["e_dcalign"].max()]))
    ax.plot([lo, hi], [lo, hi], color=lab_plotting.LAB_COLORS["chrome"],
            lw=0.8, ls="--", zorder=0, label="y = x (recovered)")
    ax.set_xlabel(f"native in-frame E under {model} (a.u.)")
    ax.set_ylabel(f"DCAlign E under {model} (a.u.)")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper left")


def _hist(ax, sub: pd.DataFrame, model: str, equal_tol: float) -> None:
    """Distribution of ΔE = E_dcalign − E_inframe; 0 marks 'recovered the frame'.

    ``n_worse`` counts ΔE > ``equal_tol`` (the same Blocker-1 threshold the
    summary JSON uses), so the figure and the table never disagree.
    """
    delta = sub["delta_e"].to_numpy()
    n_worse = int(np.sum(delta > equal_tol))
    ax.hist(delta, bins=40, color=lab_plotting.LAB_COLORS.get("fit", "C3"), alpha=0.85)
    ax.axvline(0.0, color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--")
    ax.set_xlabel(r"$\Delta E = E_{\mathrm{DCAlign}} - E_{\mathrm{in\text{-}frame}}$ (a.u.)")
    ax.set_ylabel("sequences (count)")
    ax.set_title(f"{model}: {n_worse}/{len(delta)} worse than native "
                 rf"($\Delta E > {equal_tol:g}$)", fontsize="x-small")


def render_dcalign_baseline(
    tsv: Path, model_names: tuple[str, str], out_pdf: Path,
    *, equal_tol: float = _DEFAULT_EQUAL_TOL,
) -> Path:
    """Render the per-model baseline figure (scatter + ΔE histogram) to ``out_pdf``."""
    plt.style.use("lab-paper")
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["ok"]] if "ok" in df.columns else df
    df = df.dropna(subset=["e_inframe", "e_dcalign", "delta_e"])
    groups = sorted(df["group"].unique())
    colors = _group_colors(groups)

    models = [m for m in model_names if (df["model"] == m).any()]
    if not models:
        raise ValueError(f"no comparable rows for models {model_names} in {tsv}")

    n_rows = len(models)
    figsize = (_SCATTER_W + _HIST_W + _MARGIN_W, _ROW_H * n_rows + _MARGIN_H)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=figsize,
        gridspec_kw={"width_ratios": (_SCATTER_W, _HIST_W)},
        squeeze=False,
    )
    panel = iter("ABCDEFGH")
    for row, model in enumerate(models):
        sub = df[df["model"] == model]
        _scatter(axes[row][0], sub, colors, model)
        _hist(axes[row][1], sub, model, equal_tol)
        lab_plotting.panel_label(axes[row][0], next(panel))
        lab_plotting.panel_label(axes[row][1], next(panel))

    # The lab-paper style enables constrained_layout, which sizes the panel
    # spacing from the inch budgets above — no manual tight_layout/hspace.
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=dcalign_baseline"}
    )
    plt.close(fig)
    log.info("wrote DCAlign baseline figure: %s", saved)
    return Path(saved)
