"""Consolidated figure for the potts_align ground-state-recovery diagnostic.

One figure answers "does couplings-aware potts_align recover the native ground
state?": for each model, a scatter of potts_align's minimum energy against the
native in-frame energy with the ``y = x`` diagonal (a point **on** the line means
the native frame already is the global minimum; **below** means potts_align found
a strictly lower frame; **above** would be a search failure and should not occur),
and the distribution of ``ΔE = E_potts_align − E_inframe`` (mass at 0 = recovered,
mass left of 0 = the aligner beating the native frame). Rows are the two models;
only one figure file, per the lab "consolidate similar figures" convention. It is
routed through ``scripts/lab_plotting.py`` so the lab palette + PDF provenance
stay the single source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scripts/lab_plotting.py is a sibling script, not a package module. Add the
# scripts/ dir to sys.path so ``import lab_plotting`` works regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

#: Per-panel inch budgets (figsize is derived from these × the grid, not guessed).
_SCATTER_W, _HIST_W, _ROW_H = 3.0, 2.6, 3.0
_MARGIN_W, _MARGIN_H = 1.0, 0.9

#: Fallback ground-state tolerance (a.u.); the CLI passes the run's actual
#: ``equal_tol`` (DEFAULT_EQUAL_TOL in SBM.energy.potts_align_baseline) so this is
#: only used when the figure is rendered standalone.
_DEFAULT_EQUAL_TOL = 1.0


def _group_colors(groups: list[str]) -> dict[str, str]:
    """Assign each source group a stable, colorblind-safe color (reserve black)."""
    palette = lab_plotting.WONG_PALETTE[1:]
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(groups))}


def _scatter(ax, sub: pd.DataFrame, colors: dict[str, str], model: str, equal_tol: float) -> None:
    """E_potts_align vs E_inframe with the y=x diagonal.

    On the diagonal ⇒ native frame is the global minimum (recovered); below ⇒
    potts_align found a lower-energy frame; above ⇒ worse than native (a search
    failure, should not happen). Points are colored by source group. The
    **off-diagonal** frames the aligner could not prove global (PT/SA engine,
    ``is_global_exact`` false, |ΔE| > tol) are overlaid as open rings — the
    thousands of on-diagonal PT points are recovered and left un-ringed to avoid
    clutter. So any un-ringed off-diagonal point is a *provably*-global result
    (native genuinely wasn't the ground state), and every above-diagonal (worse)
    point is ringed by construction (only the heuristic engine can land there).
    """
    for g in sorted(sub["group"].unique()):
        gs = sub[sub["group"] == g]
        ax.scatter(gs["e_inframe"], gs["e_potts"], s=8, alpha=0.6,
                   color=colors[g], label=g, linewidths=0)
    if "is_global_exact" in sub.columns:
        heuristic = sub[(~sub["is_global_exact"]) & (sub["delta_e"].abs() > equal_tol)]
    else:
        heuristic = sub.iloc[:0]
    if len(heuristic):
        ax.scatter(heuristic["e_inframe"], heuristic["e_potts"], s=42,
                   facecolors="none", edgecolors=lab_plotting.LAB_COLORS["chrome"],
                   linewidths=0.9, zorder=4,
                   label=f"PT/SA off-diagonal, not provably global (n={len(heuristic)})")
    lo = float(np.nanmin([sub["e_inframe"].min(), sub["e_potts"].min()]))
    hi = float(np.nanmax([sub["e_inframe"].max(), sub["e_potts"].max()]))
    ax.plot([lo, hi], [lo, hi], color=lab_plotting.LAB_COLORS["chrome"],
            lw=0.8, ls="--", zorder=0, label="y = x (native = ground state)")
    ax.set_xlabel(f"native in-frame E under {model} (a.u.)")
    ax.set_ylabel(f"potts_align min E under {model} (a.u.)")
    ax.legend(fontsize="xx-small", frameon=False, loc="upper left")


def _hist(ax, sub: pd.DataFrame, model: str, equal_tol: float) -> None:
    """Distribution of ΔE = E_potts_align − E_inframe; 0 marks 'recovered the frame'.

    ``n_at_ground`` counts |ΔE| ≤ ``equal_tol`` (native already at the ground
    state), ``n_improved`` counts ΔE < −tol (aligner beat native). ``n_worse``
    (ΔE > tol) is a search failure and, when nonzero, is called out — the same
    thresholds the summary JSON uses, so figure and table never disagree.
    """
    delta = sub["delta_e"].to_numpy()
    n_at_ground = int(np.sum(np.abs(delta) <= equal_tol))
    n_improved = int(np.sum(delta < -equal_tol))
    n_worse = int(np.sum(delta > equal_tol))
    ax.hist(delta, bins=40, color=lab_plotting.LAB_COLORS.get("fit", "C3"), alpha=0.85)
    ax.axvline(0.0, color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--")
    ax.set_xlabel(r"$\Delta E = E_{\mathrm{potts\_align}} - E_{\mathrm{in\text{-}frame}}$ (a.u.)")
    ax.set_ylabel("sequences (count)")
    title = (f"{model}: {n_at_ground}/{len(delta)} at ground state "
             rf"($|\Delta E|\leq{equal_tol:g}$); {n_improved} improved")
    if n_worse:
        title += f"; {n_worse} WORSE"
    ax.set_title(title, fontsize="x-small")


def render_potts_align_baseline(
    tsv: Path, model_names: tuple[str, str], out_pdf: Path,
    *, equal_tol: float = _DEFAULT_EQUAL_TOL,
) -> Path:
    """Render the per-model recovery figure (scatter + ΔE histogram) to ``out_pdf``."""
    plt.style.use("lab-paper")
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["ok"]] if "ok" in df.columns else df
    df = df.dropna(subset=["e_inframe", "e_potts", "delta_e"])
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
        _scatter(axes[row][0], sub, colors, model, equal_tol)
        _hist(axes[row][1], sub, model, equal_tol)
        lab_plotting.panel_label(axes[row][0], next(panel))
        lab_plotting.panel_label(axes[row][1], next(panel))

    # The lab-paper style enables constrained_layout, which sizes the panel
    # spacing from the inch budgets above — no manual tight_layout/hspace.
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=potts_align_baseline"}
    )
    plt.close(fig)
    log.info("wrote potts_align baseline figure: %s", saved)
    return Path(saved)
