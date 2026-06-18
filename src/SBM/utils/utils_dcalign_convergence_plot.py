"""Figure for the DCAlign convergence report: non-convergence counts per group.

How often did DCAlign fail to converge (fall back to its decimation/nucleation
path) per source group, under each model? One bar panel per model, x = query
group, bar height = not-converged count (annotated with the fraction). The
baseline figure (`utils_dcalign_baseline_plot`) already rings the *home-pair*
non-converged points; this one shows *all* alignments (home + cross), where the
cross-family frames carry most of the non-convergence. Routed through
``scripts/lab_plotting.py`` for the lab palette + PDF provenance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

#: Per-panel inch budgets (figsize derived from these × the grid, not guessed).
_PANEL_W, _PANEL_H, _MARGIN_W, _MARGIN_H = 3.4, 2.8, 0.8, 1.4


def _panel(ax, sub: pd.DataFrame, model: str) -> None:
    sub = sub.sort_values("group")
    groups = sub["group"].tolist()
    not_conv = sub["n_not_converged"].to_numpy()
    fracs = sub["frac_not_converged"].to_numpy()
    x = np.arange(len(groups))
    ax.bar(x, not_conv, color=lab_plotting.LAB_COLORS["fit"], alpha=0.85)
    for xi, cnt, frac in zip(x, not_conv, fracs):
        if cnt:
            ax.text(xi, cnt, f"{frac:.1%}", ha="center", va="bottom", fontsize="xx-small")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize="xx-small")
    ax.set_ylabel("not converged (count)")
    ax.set_title(f"under {model}", fontsize="x-small")
    ax.margins(y=0.15)  # headroom for the fraction labels


def render_dcalign_convergence(
    tsv: Path, model_names: tuple[str, str], out_pdf: Path
) -> Path:
    """Render the per-model non-convergence bar figure to ``out_pdf``."""
    plt.style.use("lab-paper")
    df = pd.read_csv(tsv, sep="\t")
    models = [m for m in model_names if (df["model"] == m).any()]
    if not models:
        raise ValueError(f"no convergence rows for models {model_names} in {tsv}")

    figsize = (_PANEL_W * len(models) + _MARGIN_W, _PANEL_H + _MARGIN_H)
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for col, model in enumerate(models):
        _panel(axes[0][col], df[df["model"] == model], model)
        lab_plotting.panel_label(axes[0][col], "AB"[col])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=dcalign_convergence"}
    )
    plt.close(fig)
    log.info("wrote DCAlign convergence figure: %s", saved)
    return Path(saved)
