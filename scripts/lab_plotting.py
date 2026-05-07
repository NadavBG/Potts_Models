"""lab_plotting.py — small companion to lab-paper.mplstyle / lab-slides.mplstyle.

The stylesheets cover everything matplotlib's rcParams can encode. This
module covers what they can't: panel labels, figure save-with-metadata,
named semantic colors used across the lab, and a couple of figure-type
recipes Rama's deck specifically calls out (stacked marginal histograms,
fit-line overlays without spline interpolation).

Conventions encoded here trace to two sources, both worth keeping
separate when reading the code:

  - Rama Ranganathan, "Abstraction and Data Representation" (lab lecture):
    grids in general, simplicity, color used to represent information
    (red = reference, blue = negative control), no splines on fits,
    decent-sized symbols with error bars, stacked histograms for
    scatterplots.
  - General data-vis literature: perceptually uniform colormaps
    (Crameri et al. 2020, Nat. Commun. 11:5444), colorblind-safe
    categorical palette (Wong 2011, Nat. Methods 8:441).

Anything not directly defensible from one of those is a judgment call,
flagged inline.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
from pathlib import Path
from typing import Sequence

import matplotlib as _mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ── Colors ──────────────────────────────────────────────────────────────

#: Wong's 8-color colorblind-safe categorical palette. Order matches
#: Wong (2011) Nat. Methods 8:441 Fig. 2. Black is first to match the
#: lab convention that the primary trace is black. This is the default
#: in the stylesheets — colorblind-safe is the safer modern default.
WONG_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

#: The lab's traditional categorical palette, reverse-engineered from
#: the published figures and recent talk decks. Pure saturated colors
#: in the IGOR Pro / classic-MATLAB tradition. Use this when matching
#: the style of older lab figures or coordinated multi-figure layouts.
#: Note: red+green together are not safe for deuteranopic viewers; if
#: that matters, use WONG_PALETTE instead. Override the active style
#: with:
#:    plt.rcParams["axes.prop_cycle"] = cycler("color", LAB_TRADITIONAL_PALETTE)
LAB_TRADITIONAL_PALETTE: tuple[str, ...] = (
    "#000000",  # black            (primary / WT)
    "#D62728",  # red              (mutant / reference)
    "#2CA02C",  # green             (mutant / second condition)
    "#1F77B4",  # blue              (mutant / negative control)
    "#E377C2",  # magenta           (additional category)
    "#FF7F0E",  # orange            (additional category)
    "#BCBD22",  # yellow-green      (additional category)
    "#17BECF",  # cyan              (additional category)
)

#: Diverging colormap for centered-on-zero data (ΔE, log-fold change,
#: residuals). The deck uses this convention consistently for heatmaps
#: of mutational effects (e.g. slides 26, 27, 50 of the 2024 talk).
#: matplotlib's "RdBu_r" gives blue for negative, red for positive,
#: white at zero — the same convention.
DIVERGING_CMAP: str = "RdBu_r"

#: Semantic colors used consistently across the lab's figures.
#: The red/blue convention for reference vs. negative control is
#: shown explicitly on slide 11 of the figure-making deck.
LAB_COLORS: dict[str, str] = {
    "reference": "#D62728",  # red
    "negative_control": "#1F77B4",  # blue
    "data": "#000000",  # black, the default for raw points
    "fit": "#D62728",  # red, for fit overlays
    "highlight": "#E69F00",  # orange, for callouts
}


# ── Run-group colors for SBM/BM figures ─────────────────────────────────
#
# A trained run produces several alignments that show up across many
# figures (similarity, diversity, energy, PCA). Pinning a color per
# group here means a given group reads as the same color in every
# figure. Naturals are green per the user's preference; the warm side
# of the Wong palette covers artificials.

#: Naturals get distinct greens (Train darker so it reads as the
#: primary). #117733 is from Paul Tol's "Bright" qualitative palette
#: (also colorblind-safe); WONG_PALETTE doesn't include a second green
#: distinguishable from #009E73, so we mix sources rather than pick
#: two near-identical greens. Random is the worst-case anchor.
RUN_GROUP_COLORS: dict[str, str] = {
    "Train": "#117733",  # Tol Bright dark green — primary natural
    "Test": WONG_PALETTE[3],  # bluish green #009E73 — secondary natural
    "Random": "#888888",  # gray — worst-case anchor
}

#: Default-run sampling temperatures get fixed colors so figures from
#: the canonical workflow look the same every time. Other Ts cycle
#: through ``ART_FALLBACK_PALETTE``.
ART_COLOR_BY_T: dict[float, str] = {
    0.75: WONG_PALETTE[1],  # orange #E69F00
    1.0: WONG_PALETTE[6],  # vermillion #D55E00
}

#: Cycle for non-default temperatures. Order chosen so adjacent Ts
#: visually separate well on a typical figure.
ART_FALLBACK_PALETTE: tuple[str, ...] = (
    WONG_PALETTE[7],  # reddish purple
    WONG_PALETTE[5],  # blue
    WONG_PALETTE[4],  # yellow
    WONG_PALETTE[2],  # sky blue
)


def color_for_natural(name: str) -> str:
    """Color for a named natural-alignment group (``"Train"``, ``"Test"``,
    ``"Random"``). Raises ``KeyError`` for unknown names — those should
    be added to ``RUN_GROUP_COLORS`` deliberately, not silently fallbacked.
    """
    return RUN_GROUP_COLORS[name]


def color_for_artificial(temperature: float | None, fallback_index: int) -> str:
    """Color for an artificial alignment, keyed on its sampling
    temperature when possible.

    ``temperature`` may be ``None`` (sidecar JSON missing); in that
    case we fall back to ``ART_FALLBACK_PALETTE[fallback_index]``.
    ``fallback_index`` is the artificial's position in the run's
    artificial list — passed in (rather than hashed off T) because
    Python's hash randomization would otherwise give the same T a
    different color on every interpreter restart.

    Match against ``ART_COLOR_BY_T`` uses ``math.isclose`` so a
    sidecar JSON written as ``0.7500001`` still resolves to the
    canonical T=0.75 color rather than silently slipping into the
    fallback palette.
    """
    import math

    if temperature is not None:
        for fixed_T, color in ART_COLOR_BY_T.items():
            if math.isclose(temperature, fixed_T, rel_tol=1e-6, abs_tol=1e-9):
                return color
    return ART_FALLBACK_PALETTE[fallback_index % len(ART_FALLBACK_PALETTE)]


# ── Panel labels ────────────────────────────────────────────────────────


def panel_label(
    ax: Axes,
    label: str,
    *,
    loc: tuple[float, float] = (-0.18, 1.05),
    fontsize: float | None = None,
    weight: str = "bold",
    **kwargs,
) -> None:
    """Add a bold panel letter (A, B, ...) above the upper-left of ``ax``.

    Coordinates are in axes fraction so the label moves with the panel
    when the figure is resized. Default ``loc`` works for most panels;
    pass a different tuple if the y-axis label is long.

    Parameters
    ----------
    ax
        The axes to label.
    label
        The label text, typically a single letter.
    loc
        ``(x, y)`` in axes fraction. Defaults to just above-left of the
        plot box.
    fontsize
        Font size in points. Defaults to ``axes.titlesize`` from the
        active style.
    weight
        Font weight. Defaults to ``"bold"``.
    """
    if fontsize is None:
        fontsize = _mpl.rcParams["axes.titlesize"]
    ax.text(
        loc[0],
        loc[1],
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        va="bottom",
        ha="left",
        **kwargs,
    )


# ── Linear fit overlay (no splines) ─────────────────────────────────────


def fit_line(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    *,
    color: str | None = None,
    extend: float = 0.0,
    label: str | None = None,
    confidence: float | None = None,
    band_style: str = "dashed",
    **kwargs,
) -> tuple[float, float]:
    """Overlay a least-squares linear fit on ``ax``.

    The deck explicitly calls out fitting "with mathematical functions
    (here, a simple line)" and forbids splines. This is the simple-line
    case: solve ``y = m*x + b`` by ordinary least squares and draw the
    line as a straight segment, not a spline through the data.

    With ``confidence`` set (e.g. 0.95), also draws confidence bands
    above and below the fit line, matching the dashed-band convention
    on slide 79 of the 2024 talk deck.

    Parameters
    ----------
    ax
        Axes to draw on.
    x, y
        Data. Must be the same length; NaNs are dropped pairwise.
    color
        Line color. Defaults to ``LAB_COLORS["fit"]`` (red).
    extend
        Fraction of the x-range to extend the fit line beyond the data
        on each side. ``0.0`` draws only across the data range.
    label
        Optional legend label. The fitted slope and intercept are NOT
        stamped on the figure here; show them in the legend or caption.
    confidence
        If set (0 < confidence < 1, e.g. 0.95), draw confidence bands
        for the mean prediction (not prediction intervals for new
        observations). Defaults to None (no bands).
    band_style
        Line style for the confidence bands. ``"dashed"`` (default)
        matches the deck convention. Use ``"none"`` with a non-None
        ``confidence`` to draw a filled band instead, plus pass
        ``alpha=`` in kwargs.

    Returns
    -------
    slope, intercept
        Fitted parameters, for the caller to log or report.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        raise ValueError("fit_line needs at least 2 finite (x, y) pairs")
    xm, ym = x[mask], y[mask]
    slope, intercept = np.polyfit(xm, ym, 1)

    x_lo, x_hi = float(xm.min()), float(xm.max())
    span = x_hi - x_lo
    x_line = np.linspace(x_lo - extend * span, x_hi + extend * span, 100)
    y_line = slope * x_line + intercept

    line_color = color or LAB_COLORS["fit"]
    ax.plot(
        x_line,
        y_line,
        color=line_color,
        label=label,
        solid_capstyle="round",
        **kwargs,
    )

    if confidence is not None:
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1), e.g. 0.95")
        if n < 3:
            # Need at least 3 points for residual variance.
            raise ValueError(
                "fit_line confidence bands need at least 3 finite (x, y) pairs"
            )
        # Confidence band for the mean prediction at each x.
        # SE_mean(x*) = sqrt(MSE * (1/n + (x* - x̄)² / Σ(xᵢ - x̄)²))
        from scipy import stats  # local import: scipy is optional

        x_mean = xm.mean()
        ssx = float(((xm - x_mean) ** 2).sum())
        residuals = ym - (slope * xm + intercept)
        mse = float((residuals**2).sum()) / (n - 2)  # 2 fitted params
        se = np.sqrt(mse * (1.0 / n + (x_line - x_mean) ** 2 / ssx))
        t_crit = stats.t.ppf(0.5 + confidence / 2, n - 2)
        delta = t_crit * se

        if band_style == "dashed":
            for offset in (+delta, -delta):
                ax.plot(
                    x_line,
                    y_line + offset,
                    color=line_color,
                    linestyle="--",
                    linewidth=_mpl.rcParams["lines.linewidth"] * 0.7,
                )
        elif band_style == "none":
            ax.fill_between(
                x_line,
                y_line - delta,
                y_line + delta,
                color=line_color,
                alpha=kwargs.get("alpha", 0.15),
                linewidth=0,
            )
        else:
            raise ValueError(
                f'band_style must be "dashed" or "none", got {band_style!r}'
            )

    return float(slope), float(intercept)


# ── Stacked marginal histograms ─────────────────────────────────────────


def scatter_with_marginals(
    x: Sequence[float],
    y: Sequence[float],
    *,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
    scatter_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Build a scatterplot with stacked histograms on the top and right.

    Rama's deck calls these out explicitly ("Stacked histograms are
    essential for showing all the data in dense scatterplots", slide 16).
    Returns the figure plus a dict of axes keyed by ``"main"``,
    ``"top"``, ``"right"``.

    For category-stacked histograms (multiple groups), do that yourself
    by calling ``ax.hist(..., stacked=True)`` on the returned ``"top"``
    and ``"right"`` axes. This helper plots a single distribution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        hspace=0.05,
        wspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(x, y, **(scatter_kwargs or {}))
    ax_top.hist(x, bins=bins, **(hist_kwargs or {}))
    ax_right.hist(y, bins=bins, orientation="horizontal", **(hist_kwargs or {}))

    # Hide the redundant tick labels on the marginal axes.
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Hide inner spines so marginals abut the scatter cleanly. This
    # matches the deck convention (e.g. slide 16, slide 55 of the 2024
    # talk) where the histograms read as part of the scatter, not as
    # separate framed panels.
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    ax_right.tick_params(left=False)

    return fig, {"main": ax_main, "top": ax_top, "right": ax_right}


# ── Save with provenance metadata ──────────────────────────────────────


def _git_commit_hash() -> str | None:
    """Return the current git HEAD commit hash, or None if not in a repo."""
    if _shutil.which("git") is None:
        return None
    try:
        out = _subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    if _shutil.which("git") is None:
        return False
    try:
        out = _subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def _calling_script() -> Path | None:
    """Best-effort path to the script that imported this module.

    Notebooks return None (sys.argv[0] is a kernel launcher path that
    isn't useful). For notebooks, pass ``script_path=`` explicitly.
    """
    argv0 = _sys.argv[0] if _sys.argv else ""
    if not argv0 or "ipykernel_launcher" in argv0 or "jupyter" in argv0:
        return None
    p = Path(argv0).resolve()
    return p if p.is_file() else None


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    script_path: str | Path | None = None,
    write_sidecar: bool = True,
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    """Save a figure with provenance baked in.

    Reproducibility corollary to Sandve et al. 2013 rule 6 ("for every
    result, keep track of how it was produced"): every saved figure
    records the git commit, the script that made it, the timestamp,
    and the matplotlib version, both in PDF metadata and (optionally)
    as a sidecar copy of the script next to the figure file.

    Parameters
    ----------
    fig
        The figure to save.
    path
        Output path. Format is inferred from the suffix.
    script_path
        Path to the calling script. Auto-detected for plain Python; pass
        explicitly from notebooks.
    write_sidecar
        If True (default) and the script is available, copy it next to
        the figure file as ``<figure>.source.py`` so the recipe lives
        with the result.
    extra_metadata
        Additional ``key: value`` strings to embed in the PDF metadata
        dict. Keys should be ASCII; values get coerced to ``str``.

    Returns
    -------
    Path
        The figure path that was written, resolved.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if script_path is None:
        script_path_resolved = _calling_script()
    else:
        script_path_resolved = Path(script_path).resolve()

    commit = _git_commit_hash()
    dirty = _git_dirty()
    when = _dt.datetime.now(_dt.timezone.utc)

    # matplotlib's PDF backend expects CreationDate as a datetime object,
    # not an ISO string. Other keys take strings.
    metadata = {
        "Creator": f"matplotlib {_mpl.__version__}",
        "Producer": "lab_plotting.save_figure",
        "CreationDate": when,
        "Subject": f"git={commit or 'no-repo'}{'+dirty' if dirty else ''}; "
        f"saved={when.isoformat()}",
    }
    if script_path_resolved is not None:
        metadata["Author"] = str(script_path_resolved)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})

    # Only PDF and SVG support arbitrary metadata via savefig; for other
    # formats the metadata kwarg is silently ignored. Pass it anyway.
    fig.savefig(path, metadata=metadata)

    if write_sidecar and script_path_resolved is not None:
        sidecar = path.with_suffix(path.suffix + ".source.py")
        # Use copy2 to preserve mtime so it's clear when the script ran.
        _shutil.copy2(script_path_resolved, sidecar)

    return path


# ── Convenience: hash a figure deterministically (for tests) ────────────


def figure_hash(fig: Figure, dpi: int = 100) -> str:
    """Return a deterministic SHA-256 hash of the figure as PNG bytes.

    Useful for regression tests that pin a figure's pixel output across
    code changes. Be aware that minor matplotlib version bumps can
    change rendering in ways that break the hash without changing
    perceived correctness — treat hash mismatches as a prompt to look,
    not as a hard failure.
    """
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return _hashlib.sha256(buf.getvalue()).hexdigest()
