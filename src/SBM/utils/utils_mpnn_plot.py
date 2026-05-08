"""ProteinMPNN foldability sweep — figure recipe.

Reads ``mpnn_scores.json`` from a ``mpnn_sweep_*`` subdir and produces a
single-panel violin figure: one violin per sampled temperature,
controls (random, shuffled, natural) to the right of a separator,
and the WT score as a horizontal reference line spanning the panel.

Lives in ``SBM.utils`` so it imports without going through the
``utils_plot.plot_stats(plot=mode, ...)`` dispatcher in
``utils_plot.py`` — that file's contract (it draws into figures created
internally and is read by the renderer via ``plt.get_fignums()`` diff)
doesn't fit a self-contained recipe well.

Colors are routed through ``scripts/lab_plotting.py`` so the lab
palette stays the single source of truth: ``ART_COLOR_BY_T`` for
canonical sampling temperatures, ``ART_FALLBACK_PALETTE`` cycles for
the rest, ``RUN_GROUP_COLORS["Train"]`` for the natural-bootstrap
control, ``RUN_GROUP_COLORS["Random"]`` for the uniform-random
control, ``LAB_COLORS["highlight"]`` for shuffled-WT, and
``LAB_COLORS["reference"]`` for the WT line.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# scripts/lab_plotting.py is a sibling script, not a package module.
# Add the scripts/ dir to sys.path so ``import lab_plotting`` works
# regardless of the cwd the renderer is invoked from. Mirrors the
# pattern in scripts/render_figures.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

#: Order in which controls appear on the figure, after the temperature
#: violins. ``wt`` is plotted as a horizontal reference line, not a
#: violin, so it does not appear here.
_CONTROL_ORDER: tuple[str, ...] = ("random", "shuffled", "natural")

#: Pretty labels for the controls under each violin.
_CONTROL_LABELS: dict[str, str] = {
    "random": "Random",
    "shuffled": "Shuffled\nWT",
    "natural": "Natural\nMSA",
}


def _control_color(name: str) -> str:
    if name == "random":
        return lab_plotting.RUN_GROUP_COLORS["Random"]
    if name == "shuffled":
        return lab_plotting.LAB_COLORS["highlight"]
    if name == "natural":
        return lab_plotting.RUN_GROUP_COLORS["Train"]
    raise KeyError(f"unknown control name {name!r}")


def _temperature_color(T: float, fallback_index: int) -> str:
    return lab_plotting.color_for_artificial(T, fallback_index)


def _temp_label(T: float) -> str:
    """Tick label for a temperature column."""
    return f"T={T:g}"


def _parse_scores_json(scores_path: Path) -> dict[str, Any]:
    if not scores_path.is_file():
        raise FileNotFoundError(f"mpnn_scores.json not found at {scores_path}")
    with open(scores_path, encoding="utf-8") as f:
        return json.load(f)


def _collect_groups(
    payload: dict[str, Any],
) -> tuple[
    list[tuple[float, np.ndarray, int]],
    dict[str, np.ndarray | float],
]:
    """Pull arrays out of ``payload["groups"]``.

    Returns
    -------
    samples
        Sorted-by-T list of ``(temperature, scores_array, extra_residues_median)``.
    controls
        ``{name: scores_array_or_scalar}`` keyed by control name.
        ``"wt"`` is a single scalar; the rest are 1-D arrays.
    """
    samples: list[tuple[float, np.ndarray, int]] = []
    controls: dict[str, np.ndarray | float] = {}
    for key, group in payload["groups"].items():
        kind = group.get("kind")
        scores = group.get("scores")
        if scores is None or len(scores) == 0:
            log.warning("group %r has no scores recorded; skipping in figure", key)
            continue
        arr = np.asarray(scores, dtype=np.float64)
        if kind == "sample":
            T = float(group["temperature"])
            extras = group.get("extra_residue_counts") or []
            median_extra = int(np.median(extras)) if len(extras) else 0
            samples.append((T, arr, median_extra))
        elif kind == "control":
            name = group.get("name") or key
            if name == "wt":
                controls[name] = float(arr.mean())
            else:
                controls[name] = arr
        else:
            log.warning("group %r has unknown kind %r; skipping", key, kind)
    samples.sort(key=lambda triple: triple[0])
    return samples, controls


def _color_violin(parts: dict, color: str, *, alpha: float = 0.6) -> None:
    """Apply a single fill+edge color to a ``Axes.violinplot`` return dict."""
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
    # Median / extrema lines (only present if the violinplot was called
    # with showmedians / showextrema). They share the body color but are
    # opaque so they read on top of the (translucent) body.
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        seg = parts.get(key)
        if seg is None:
            continue
        seg.set_color(color)
        seg.set_alpha(1.0)


def _figure_size(n_cols: int) -> tuple[float, float]:
    """Compute a figsize from rcParams font/tick metrics + column count.

    Width budget:
        left margin + n_cols * per_col_width + right margin
    Per-col width is driven by the tick label font size so labels don't
    overlap.

    Height budget:
        top margin + plot_height + bottom margin
    Plot height is fixed at 3 inches; bottom margin scales with tick
    label height for the two-line control labels.
    """
    points_per_inch = 72.0
    tick_pt = float(mpl.rcParams.get("xtick.labelsize", 10.0))
    tick_in = tick_pt / points_per_inch
    # The widest tick label is one of "T=0.1" / "T=1" / "Shuffled\nWT" —
    # all under ~6 chars at the visible breakpoint. Allow ~ 0.55 ch per
    # point to convert; floor at 0.55" so violins are visually distinct.
    per_col = max(0.55, 6 * tick_in * 0.55)
    left = 1.0
    right = 0.4
    width = left + n_cols * per_col + right

    plot_height = 3.0
    top = 0.4
    bottom = 0.9  # room for two-line control labels
    height = top + plot_height + bottom
    return float(width), float(height)


def plot_mpnn_foldability(
    scores_path: Path,
    *,
    ax: Axes | None = None,
    show_extra_residue_warning: bool = True,
) -> Figure:
    """Render the foldability-sweep figure from a sweep's ``mpnn_scores.json``.

    Parameters
    ----------
    scores_path
        Absolute path to ``mpnn_scores.json`` written by
        ``scripts/mpnn_sweep.py``.
    ax
        Existing axes to draw on. If ``None`` (default), a new figure
        is created sized from the number of violin columns. When ``ax``
        is provided, the caller owns the figure layout.
    show_extra_residue_warning
        If True, append a "median extra residues per row" annotation
        below each T column whose median is non-zero — a high rate flags
        that the sample is putting residues at WT-gap MSA columns the
        structure can't score.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the panel. The caller saves via
        ``lab_plotting.save_figure`` for provenance metadata.
    """
    payload = _parse_scores_json(scores_path)
    samples, controls = _collect_groups(payload)
    if not samples and not controls:
        raise RuntimeError(
            f"no plottable groups in {scores_path}. "
            "Did you run with --mpnn-skip-scoring?"
        )

    control_keys = [c for c in _CONTROL_ORDER if c in controls]
    n_temps = len(samples)
    n_controls = len(control_keys)
    n_cols = n_temps + n_controls

    if ax is None:
        fig, ax = plt.subplots(figsize=_figure_size(n_cols))
    else:
        fig = ax.figure

    positions: list[float] = []
    tick_labels: list[str] = []
    cur_x = 1.0
    # ── Sample violins.
    for i, (T, scores, _median_extra) in enumerate(samples):
        parts = ax.violinplot(
            scores,
            positions=[cur_x],
            widths=0.7,
            showmedians=True,
            showextrema=True,
        )
        _color_violin(parts, _temperature_color(T, i), alpha=0.6)
        positions.append(cur_x)
        tick_labels.append(_temp_label(T))
        cur_x += 1.0

    # Separator between samples and controls.
    if n_temps > 0 and n_controls > 0:
        sep_x = cur_x - 0.5
        ax.axvline(
            sep_x,
            color=lab_plotting.LAB_COLORS["chrome"],
            linestyle=":",
            linewidth=0.8,
            zorder=0.5,
        )

    # ── Control violins.
    for name in control_keys:
        arr = np.asarray(controls[name], dtype=np.float64)
        parts = ax.violinplot(
            arr, positions=[cur_x], widths=0.7, showmedians=True, showextrema=True
        )
        _color_violin(parts, _control_color(name), alpha=0.6)
        positions.append(cur_x)
        tick_labels.append(_CONTROL_LABELS[name])
        cur_x += 1.0

    # ── WT reference line.
    if "wt" in controls:
        wt_score = float(controls["wt"])
        ax.axhline(
            wt_score,
            color=lab_plotting.LAB_COLORS["reference"],
            linewidth=1.4,
            zorder=1.5,
            label=f"WT = {wt_score:.3f}",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlim(0.3, cur_x - 0.3 if n_cols else 1.0)
    ax.set_xlabel("Group")
    ax.set_ylabel("MPNN score (NLL / residue)")
    mpnn_block = payload.get("mpnn") or {}
    ax.set_title(
        f"ProteinMPNN foldability sweep — L={payload.get('L', '?')}, "
        f"{mpnn_block.get('model_name', '?')}"
    )
    if "wt" in controls:
        ax.legend(loc="best", frameon=False, fontsize="small")

    # ── Extra-residue annotation.
    if show_extra_residue_warning:
        for (T, _scores, median_extra), pos in zip(samples, positions[:n_temps]):
            if median_extra > 0:
                ax.annotate(
                    f"+{median_extra}",
                    xy=(pos, ax.get_ylim()[1]),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize="x-small",
                    color=lab_plotting.LAB_COLORS["chrome"],
                )

    # Layout: do NOT call tight_layout — the lab-paper stylesheet
    # enables figure.constrained_layout.use, and the two managers
    # collide (matplotlib warns and the explicit tight call wins,
    # silently undoing constrained-layout sizing). Trust the style.
    return fig
