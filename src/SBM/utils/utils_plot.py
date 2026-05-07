"""Figure renderers for SBM/BM runs.

Original author: Marion CHAUVEAU (October 2022).

Refactored to take a list of artificial alignments (one per sampling
temperature) and emit consolidated multi-panel figures: a single
``Correlations`` figure with rows=temperatures × cols=orders replaces
the legacy ``Freq`` / ``Pair_freq`` / ``Corr3`` triplet, and PCA
becomes a ``1 × (1 + N_temps)`` grid (natural + each artificial).
``Similarity`` and ``Diversity`` are violin plots (one violin per
group: Train, optionally Test, and one per artificial T). ``Energy``
overlays histograms (Natural Train, optionally Test, each artificial
T, and a Random baseline). ``Length`` overlays length histograms
(still requires a Test split).

**Colors** are wired in by the caller (``scripts/render_figures.py``),
which pulls the canonical mapping from ``scripts/lab_plotting.py``.
``plot_stats`` accepts a ``natural_colors`` dict and reads
``item["color"]`` off each artificial alignment's dict; if either is
missing it raises rather than guessing. Keeping the canonical palette
in ``lab_plotting`` (alongside ``WONG_PALETTE`` and ``LAB_COLORS``)
avoids drift if the lab updates the palette and prevents this package
module from sys.path-hacking into ``scripts/``.
"""

####################### MODULES #######################

import logging

import numpy as np  # type: ignore
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import SBM.utils.utils as ut
from matplotlib.colors import Normalize  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from matplotlib import cm  # type: ignore

log = logging.getLogger(__name__)


def _pt_to_in(pt: float) -> float:
    """Convert points to inches (1 pt = 1/72 in)."""
    return float(pt) / 72.0


def _resolve_font_pt(spec) -> float:
    """Resolve a matplotlib font-size rcParam (number or alias like
    ``"medium"``/``"large"``) to absolute points using the active
    ``font.size`` and the standard scaling table.
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    base = float(mpl.rcParams["font.size"])
    return base * mpl.font_manager.font_scalings.get(spec, 1.0)


##########################################################

####################### PLOT STATISTICS #######################


def _art_label(base, temperature=None):
    """Append the sampling temperature to a label mentioning the
    artificial set: ``"Artificial set"`` → ``"Artificial set (T=0.75)"``.
    Returns ``base`` unchanged when ``temperature`` is None.
    """
    if temperature is None:
        return base
    return f"{base} (T={float(temperature):g})"


_REQUIRED_NATURAL_COLOR_KEYS: tuple[str, ...] = ("Train", "Test", "Random")


def _resolve_natural_colors(natural_colors):
    """Validate the natural-group color dict supplied by the caller.

    Required because the package shouldn't carry its own copy of the
    palette (single source of truth lives in
    ``scripts/lab_plotting.py``). Missing keys raise rather than
    silently defaulting, so a typo doesn't quietly produce a figure
    with the wrong color.
    """
    if natural_colors is None:
        raise ValueError(
            "plot_stats: natural_colors is required. The canonical mapping "
            "lives in scripts/lab_plotting.py (RUN_GROUP_COLORS); pass "
            "{name: lab_plotting.color_for_natural(name) for name in "
            "('Train', 'Test', 'Random')} from your renderer."
        )
    missing = [k for k in _REQUIRED_NATURAL_COLOR_KEYS if k not in natural_colors]
    if missing:
        raise ValueError(
            f"plot_stats: natural_colors missing keys {missing}; "
            f"need {list(_REQUIRED_NATURAL_COLOR_KEYS)}"
        )
    return natural_colors


def _color_for_artificial_item(item):
    """Pull the pre-stamped color off an artificial dict, raising if
    the renderer forgot to set it."""
    color = item.get("color")
    if not color:
        raise ValueError(
            "plot_stats: each artificial item needs a 'color' key. Stamp "
            "it in the renderer using lab_plotting.color_for_artificial("
            "item['temperature'], index)."
        )
    return color


def _drop_non_finite(arr, label):
    """Filter NaN/Inf from a 1-D distance array and log how many were
    dropped. Both ``compute_similarities`` and ``compute_diversity``
    can produce NaN when a sequence pair has zero non-gap-overlap
    positions (the gap-aware norm is zero, so ``1 − matches/norm``
    becomes ``1 − 0/0``). Filtering keeps the violin/histogram from
    silently inheriting NaN; the warning surfaces the dropped count
    so a sudden spike is visible rather than hidden.
    """
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    n_drop = int(arr.size - finite.sum())
    if n_drop:
        log.warning(
            "%s: dropped %d non-finite value(s) (likely all-gap rows). "
            "Inspect the alignment if the count is large.",
            label,
            n_drop,
        )
    return arr[finite]


def _violin_panel(groups, labels, colors, *, ylabel, ylim=None):
    """Draw one violin per group. Used by Similarity and Diversity.
    ``colors`` is one color string per group, in the same order as
    ``groups`` / ``labels``. Median + extrema markers stay on so the
    eye can compare central tendency without reading the violin shape
    alone. Returns ``(fig, ax)``.
    """
    if not (len(groups) == len(labels) == len(colors)):
        raise ValueError(
            f"_violin_panel: groups/labels/colors length mismatch "
            f"({len(groups)}/{len(labels)}/{len(colors)})"
        )
    empty = [name for g, name in zip(groups, labels) if len(g) == 0]
    if empty:
        # matplotlib's ``violinplot`` raises an opaque error on empty
        # input. Surface the named group(s) instead.
        raise ValueError(
            f"_violin_panel: empty group(s) {empty}. After non-finite "
            "filtering, every group must have at least one value."
        )
    fig, ax = plt.subplots(figsize=(0.7 * max(len(groups), 4) + 1.5, 2.8))
    positions = list(range(len(groups)))
    parts = ax.violinplot(
        groups,
        positions=positions,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def _scatter_panel(ax, x, y, *, label_pearson, xlabel, ylabel, title, diag_xy):
    """Render a single scatter panel into ``ax`` (data + diagonal +
    Pearson legend). ``diag_xy`` is a ``([x0, x1], [y0, y1])`` pair
    drawn as the equality line, or ``None`` to skip it.

    Sizes / fonts come from the active matplotlib style (lab-paper);
    do not override them inline here.
    """
    pears = float(np.corrcoef(x, y)[0, 1])
    # Invisible point as a label-only legend entry (carries the Pearson).
    ax.plot([], [], "o", color="white", label=f"{label_pearson}\nPearson: {pears:.2f}")
    ax.plot(x, y, "o", color="0.4")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if diag_xy is not None:
        ax.plot(diag_xy[0], diag_xy[1], color="black", linewidth=0.8)
    ax.legend()
    if title:
        ax.set_title(title)


# (statistic-key, column header, fixed range, diag_form)
# ``fixed_range`` is None for orders whose extent depends on the data
# (3rd-order tensor); the diagonal is then taken from the reference's
# min/max. Otherwise the diagonal is anchored at +/- range.
# ``diag_form``:
#   "0_to_R"   : [(0,0)-(R,R)] — frequencies in [0,1]
#   "minus_R"  : [(-R,-R)-(R,R)] — centered correlations
#   "data"     : [(min,min)-(max,max)] — dynamic
_ORDER_SPEC: tuple[tuple, ...] = (
    ("Freq", "1st order statistics", 1.0, "0_to_R"),
    ("Pair_freq", "Pairwise correlations", 0.4, "minus_R"),
    ("Three_corr", "3rd order correlations", None, "data"),
)


def _diag_for(stats_ref_vals, fixed_range, diag_form):
    """Build a `(x_pair, y_pair)` diagonal-line tuple for ``_scatter_panel``."""
    if diag_form == "0_to_R":
        return ([0, fixed_range], [0, fixed_range])
    if diag_form == "minus_R":
        return ([-fixed_range, fixed_range], [-fixed_range, fixed_range])
    if diag_form == "data":
        a = float(np.amin(stats_ref_vals))
        b = float(np.amax(stats_ref_vals))
        return ([a, b], [a, b])
    raise ValueError(f"unknown diag_form: {diag_form!r}")


def _flatten_for(key, arr, ind_pair):
    """Pick the right view of a stats array for a scatter:
    Pair_freq uses only the upper triangle; others flatten directly."""
    if key == "Pair_freq":
        return arr[ind_pair].flatten()
    return arr.flatten()


def plot_stats(
    output,
    plot="Correlations",
    *,
    artificial=None,
    natural_colors=None,
    sector_positions=None,
):
    """Render one figure for the requested plot mode.

    Parameters
    ----------
    output
        The model dict (carries ``align``, ``Train``, optionally ``Test``,
        ``h``, ``J``, ``J_norm``, ``options``).
    plot
        Mode name. One of ``Correlations``, ``PCA``, ``Energy``,
        ``Coupling_evol``, ``Params``, ``Similarity``, ``Diversity``,
        ``Length``.
    artificial
        List of dicts, one per sampling temperature, each with keys
        ``temperature`` (float | None), ``align_mod`` (ndarray, required
        for align-needing modes), ``stats`` (dict from compute_stats,
        required for Correlations), and ``color`` (str, required for
        align-needing modes — stamp it via
        ``lab_plotting.color_for_artificial`` in the renderer). Empty
        / None for ``Coupling_evol`` and ``Params``.
    natural_colors
        Required for align-needing modes: dict with keys ``"Train"``,
        ``"Test"``, ``"Random"`` mapping to color strings. The renderer
        sources these from ``lab_plotting.color_for_natural``.
    sector_positions
        Used only by ``Params``: iterable of MSA column indices to mark
        as a sector strip above the heatmaps. ``None`` / empty hides
        the strip — non-CM runs simply omit it.
    """
    # The signature changed in this refactor (removed positional ``Stats``,
    # ``ma``, ``temperature``; added kw-only ``artificial``). Catch the
    # legacy positional call eagerly so old notebooks don't silently
    # render nothing.
    if not isinstance(plot, str):
        raise TypeError(
            "plot_stats signature changed: pass artificial=[{...}] as a "
            "keyword. Got plot=%r (likely the old positional ``Stats``)"
            % type(plot).__name__
        )
    artificial = list(artificial) if artificial else []
    has_test = output.get("Test") is not None
    # Coupling_evol and Params are model-only; they don't read group colors.
    needs_colors = plot not in ("Coupling_evol", "Params")
    if needs_colors:
        natural_colors = _resolve_natural_colors(natural_colors)

    if plot == "Correlations":
        if not artificial:
            raise ValueError("Correlations plot requires at least one artificial set")
        ref_key = "Test" if has_test else "Train"
        ref_xlabel = "Test set" if has_test else "Train set"
        L = artificial[0]["align_mod"].shape[1]
        ind_pair = np.triu_indices(L, 1)

        # Compute per-column diagonals once. With ``sharey="col"`` the
        # axes auto-scale to the union of all rows' data, so a row-local
        # diagonal would visually mismatch the actual axis range. For
        # the dynamic ("data") forms we therefore concatenate every
        # row's reference values for that column.
        col_diags: list = []
        for j, (key, _header, fixed, form) in enumerate(_ORDER_SPEC):
            if form == "data":
                concat = np.concatenate(
                    [
                        _flatten_for(key, item["stats"][ref_key][key], ind_pair)
                        for item in artificial
                    ]
                )
                col_diags.append(_diag_for(concat, fixed, form))
            else:
                col_diags.append(_diag_for(None, fixed, form))

        n_rows = len(artificial)
        n_cols = len(_ORDER_SPEC)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.5 * n_cols, 3.5 * n_rows),
            sharex="col",
            sharey="col",
            squeeze=False,
        )
        for i, item in enumerate(artificial):
            stats_i = item["stats"]
            T = item["temperature"]
            row_label = _art_label("Artificial", T)
            for j, (key, header, _fixed, _form) in enumerate(_ORDER_SPEC):
                ax = axes[i, j]
                ref_vals = _flatten_for(key, stats_i[ref_key][key], ind_pair)
                art_vals = _flatten_for(key, stats_i["Artificial"][key], ind_pair)
                _scatter_panel(
                    ax,
                    ref_vals,
                    art_vals,
                    label_pearson=header,
                    xlabel=ref_xlabel if i == n_rows - 1 else "",
                    ylabel=row_label if j == 0 else "",
                    title=header if i == 0 else "",
                    diag_xy=col_diags[j],
                )

    if plot == "PCA":
        if not artificial:
            raise ValueError("PCA plot requires at least one artificial set")
        Max = 0.15
        align_nat = output["align"]
        # Match subsample size to the smallest population so density
        # estimates are comparable across panels.
        Ms = [item["align_mod"].shape[0] for item in artificial]
        M = min(min(Ms), align_nat.shape[0])

        nat_idx = np.random.choice(align_nat.shape[0], M, replace=False)
        bin_nat = ut.alg2bin(align_nat[nat_idx], N_aa=20)

        # Fit PC1/PC2 on the natural alignment once; project every
        # artificial alignment onto the same basis for fair comparison.
        cov = np.cov(bin_nat.T)
        W, V = np.linalg.eigh(cov)
        ind = np.argsort(W)[::-1]
        v1, v2 = V[:, ind[0]], V[:, ind[1]]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # Pin sign convention: ``np.linalg.eigh`` returns eigenvectors
        # whose sign is implementation-defined (and can flip between
        # numpy releases or BLAS backends). Force the largest-magnitude
        # entry to be positive so the plotted PCA is reproducible across
        # environments. Natural-vs-artificial within one figure agree
        # regardless, but the figure as a whole would otherwise mirror.
        if v1[np.argmax(np.abs(v1))] < 0:
            v1 = -v1
        if v2[np.argmax(np.abs(v2))] < 0:
            v2 = -v2
        conserved = float((W[ind[0]] + W[ind[1]]) / np.sum(W))
        log.info("PCA conserved variance: %.4f %%", conserved * 100)

        X_nat = np.column_stack((bin_nat @ v1, bin_nat @ v2))
        art_Xs = []
        for item in artificial:
            sub_idx = np.random.choice(item["align_mod"].shape[0], M, replace=False)
            bin_art = ut.alg2bin(item["align_mod"][sub_idx], N_aa=20)
            art_Xs.append(np.column_stack((bin_art @ v1, bin_art @ v2)))

        all_pts = np.concatenate([X_nat] + art_Xs, axis=0)
        shift = 0.4
        mi1 = float(np.amin(all_pts[:, 0])) - shift
        ma1 = float(np.amax(all_pts[:, 0])) + shift
        mi2 = float(np.amin(all_pts[:, 1])) - shift
        ma2 = float(np.amax(all_pts[:, 1])) + shift

        n_panels = 1 + len(artificial)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(3.5 * n_panels, 3.5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]
        # Title color matches each group's color in the violin/energy
        # figures so the same alignment reads as the same color
        # everywhere.
        panels = [(axes[0], X_nat, "Natural sequences", natural_colors["Train"])]
        for ax, item, X_a in zip(axes[1:], artificial, art_Xs):
            panels.append(
                (
                    ax,
                    X_a,
                    _art_label("Artificial sequences", item["temperature"]),
                    _color_for_artificial_item(item),
                )
            )
        # density_scatter draws the points with vmin=0, vmax=Max so the
        # color mapping is identical across panels. We add a single
        # shared colorbar after the loop instead of one per panel.
        for ax, pts, title, title_color in panels:
            density_scatter(pts[:, 0], pts[:, 1], Max=Max, ax=ax, add_colorbar=False)
            ax.set_xlim(mi1, ma1)
            ax.set_ylim(mi2, ma2)
            ax.set_xlabel("PC 1")
            ax.set_title(title, color=title_color)
        axes[0].set_ylabel("PC 2")
        norm = Normalize(vmin=0, vmax=Max)
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap="magma"),
            ax=axes.tolist(),
            shrink=0.8,
            label="Local density (KDE)",
        )

    if plot == "Energy":
        if not artificial:
            raise ValueError("Energy plot requires at least one artificial set")
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        Bins = 60
        # A random alignment as a worst-case baseline. Shape from the
        # first artificial set; all share L (validated upstream).
        rand = np.round(np.random.random(artificial[0]["align_mod"].shape)).astype(
            "int32"
        )

        # ut.compute_energies returns -(Σh + ½ΣJ); lower = more probable
        # under the Potts model (utils.py:561). Plot raw values — no
        # z-score — so the x-axis is in the model's native units and
        # the absolute energy scale is preserved across runs.
        Etrain = ut.compute_energies(output["Train"], output["h"], output["J"])
        Etest = (
            ut.compute_energies(output["Test"], output["h"], output["J"])
            if has_test
            else None
        )
        Erand = ut.compute_energies(rand, output["h"], output["J"])
        Emods = [
            (
                item["temperature"],
                ut.compute_energies(item["align_mod"], output["h"], output["J"]),
            )
            for item in artificial
        ]

        all_e = [Etrain, Erand] + [e for _, e in Emods]
        if Etest is not None:
            all_e.append(Etest)
        mi = float(np.amin(np.concatenate(all_e)))
        ma = float(np.amax(np.concatenate(all_e)))
        # Pad 2% of the data range, but force a non-zero span if every
        # group collapsed onto the same value (matplotlib raises on a
        # zero-width hist range).
        span = ma - mi
        pad = 0.02 * span if span > 0 else 0.5
        common = dict(bins=Bins, range=(mi - pad, ma + pad), alpha=0.55, density=True)
        ax.hist(
            Etrain,
            label="Natural (Train)",
            color=natural_colors["Train"],
            **common,
        )
        if Etest is not None:
            ax.hist(
                Etest,
                label="Natural (Test)",
                color=natural_colors["Test"],
                **common,
            )
        for item, (T, e) in zip(artificial, Emods):
            ax.hist(
                e,
                label=_art_label("Artificial", T),
                color=_color_for_artificial_item(item),
                **common,
            )
        ax.hist(
            Erand,
            label="Random",
            color=natural_colors["Random"],
            **common,
        )
        ax.legend()
        ax.set_xlabel("Statistical energy −(Σh + ½ΣJ)")
        ax.set_ylabel("Probability density")

    if plot == "Similarity":
        if not artificial:
            raise ValueError("Similarity plot requires at least one artificial set")
        groups, labels, colors = [], [], []
        # compute_similarities returns distance d ∈ [0,1] (utils.py:564);
        # plot identity = 1 − d so higher = more similar to a natural.
        # Train is compared against itself with self excluded (utils.py:577).
        groups.append(
            _drop_non_finite(
                1.0 - ut.compute_similarities(output["Train"]),
                label="Similarity[Train]",
            )
        )
        labels.append("Train")
        colors.append(natural_colors["Train"])
        if has_test:
            groups.append(
                _drop_non_finite(
                    1.0 - ut.compute_similarities(output["Test"], output["Train"]),
                    label="Similarity[Test]",
                )
            )
            labels.append("Test")
            colors.append(natural_colors["Test"])
        for item in artificial:
            T = item["temperature"]
            groups.append(
                _drop_non_finite(
                    1.0 - ut.compute_similarities(item["align_mod"], output["Train"]),
                    label=f"Similarity[Artificial T={T}]",
                )
            )
            labels.append(_art_label("Artificial", T))
            colors.append(_color_for_artificial_item(item))
        fig, ax = _violin_panel(
            groups,
            labels,
            colors,
            ylabel="Identity to closest natural sequence",
            ylim=(0, 1),
        )

    if plot == "Diversity":
        if not artificial:
            raise ValueError("Diversity plot requires at least one artificial set")
        groups, labels, colors = [], [], []
        # compute_diversity returns all N(N−1)/2 pairwise distances using
        # the same gap-aware Hamming metric (utils.py:591). Higher =
        # more internally diverse alignment. Plotted as distance (not
        # flipped), so the y-axis name matches "Diversity".
        groups.append(
            _drop_non_finite(
                ut.compute_diversity(output["Train"]),
                label="Diversity[Train]",
            )
        )
        labels.append("Train")
        colors.append(natural_colors["Train"])
        if has_test:
            groups.append(
                _drop_non_finite(
                    ut.compute_diversity(output["Test"]),
                    label="Diversity[Test]",
                )
            )
            labels.append("Test")
            colors.append(natural_colors["Test"])
        for item in artificial:
            T = item["temperature"]
            groups.append(
                _drop_non_finite(
                    ut.compute_diversity(item["align_mod"]),
                    label=f"Diversity[Artificial T={T}]",
                )
            )
            labels.append(_art_label("Artificial", T))
            colors.append(_color_for_artificial_item(item))
        fig, ax = _violin_panel(
            groups,
            labels,
            colors,
            ylabel="Pairwise distance within set",
            ylim=(0, 1),
        )

    if plot == "Length":
        if not artificial:
            raise ValueError("Length plot requires at least one artificial set")
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        Bins = 80
        Length_train = np.sum(output["Train"], axis=1)
        Length_test = np.sum(output["Test"], axis=1)
        Length_arts = [
            (item["temperature"], np.sum(item["align_mod"], axis=1))
            for item in artificial
        ]

        all_l = [Length_train, Length_test] + [L for _, L in Length_arts]
        mi = float(np.amin(np.concatenate(all_l)))
        ma = float(np.amax(np.concatenate(all_l)))
        common = dict(bins=Bins, range=(mi, ma), alpha=0.55, density=True)
        ax.hist(Length_train, label="Train", color=natural_colors["Train"], **common)
        ax.hist(Length_test, label="Test", color=natural_colors["Test"], **common)
        for item, (T, L) in zip(artificial, Length_arts):
            ax.hist(
                L,
                label=_art_label("Artificial", T),
                color=_color_for_artificial_item(item),
                **common,
            )
        ax.legend()
        ax.set_xlabel("Genome length")
        ax.set_ylabel("Probability density")

    if plot == "Coupling_evol":
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        # train_sbm.py stores J_norm as (N_av, 1 + N_records); the leading
        # column is the scalar-0 placeholder that ``output["J_norm"]``
        # holds before any recording is appended in Minimizer.
        j_norm = np.atleast_2d(output["J_norm"])[:, 1:]

        # Iteration label for each column. Prefer the explicit list saved
        # by Minimizer (new runs include a final recording at i = N_iter).
        # Fall back to reconstructing from Record_every for models written
        # before that field existed.
        iters = output.get("J_norm_iters")
        if iters is None or len(iters) != j_norm.shape[1]:
            record_every = output["options"].get("Record_every", 100)
            iters = np.arange(j_norm.shape[1]) * record_every
        else:
            iters = np.asarray(iters)

        for row in j_norm:
            ax.plot(iters, row, "o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Couplings norm")
        ax.set_title(
            f"{output['options']['Model']}, "
            f"N_chains={output['options']['n_states']} "
            f"m={output['options']['m']}"
        )

    if plot == "Params":
        h_in = output.get("h")
        J_in = output.get("J")
        if h_in is None or J_in is None:
            raise ValueError(
                "Params plot needs both 'h' and 'J' in the model dict; "
                f"got h={'present' if h_in is not None else 'None'}, "
                f"J={'present' if J_in is not None else 'None'}. Profile-"
                "only runs (Zero Couplings=True) write J as zeros after "
                "Zero_Sum_Gauge — re-train with couplings enabled or "
                "pass --figs without 'params'."
            )
        h_in = np.asarray(h_in)
        J_in = np.asarray(J_in)
        if (
            h_in.ndim != 2
            or J_in.ndim != 4
            or J_in.shape[:2] != (h_in.shape[0], h_in.shape[0])
        ):
            raise ValueError(
                "Params plot expects h shape (L, q) and J shape (L, L, q, q); "
                f"got h={h_in.shape}, J={J_in.shape}"
            )
        # Project onto zero-sum gauge defensively. ``Zero_Sum_Gauge`` is
        # idempotent, so this is a no-op for runs from train_sbm.py
        # (which already gauges before saving) and a correctness fix
        # for any hand-built model dict. Keeping it here makes the
        # "(zero-sum gauge)" colorbar label load-bearing rather than
        # a comment that hopes the inputs were gauged upstream.
        J, h = ut.Zero_Sum_Gauge(J_in, h_in)
        L, q = h.shape

        # Frobenius norm of each pairwise coupling block. The diagonal
        # is zero by construction (Jw never writes J[i,i]) and reads as
        # the bottom of the colormap.
        J_norm = np.linalg.norm(J, axis=(2, 3))

        sectors = sorted({int(i) for i in (sector_positions or [])})
        if sectors and (min(sectors) < 0 or max(sectors) >= L):
            raise ValueError(
                f"sector_positions out of range for L={L}: "
                f"min={min(sectors)}, max={max(sectors)}"
            )

        # Standard 1-letter alphabet used everywhere else in the package.
        # Catch q != 21 loudly so a non-protein alphabet doesn't render
        # silently mislabelled.
        alphabet = "-ACDEFGHIKLMNPQRSTVWY"
        if q != len(alphabet):
            raise ValueError(
                f"Params plot expects q={len(alphabet)} (gap + 20 AAs); got q={q}"
            )

        # Inch-budget layout. Cell dimensions follow the data
        # (panel_w * q/L for h, panel_w for J) so cells render as
        # visual squares under aspect="auto"; the inter-panel gap is
        # sized from rcParams so it tracks J's top decorations under
        # any matplotlib style. Each component names what it covers —
        # fig_w / fig_h are sums, not eyeballed totals. The character-
        # width factors (2.0/2.5 × ytick_pt) and the colorbar width
        # (13 pt) are still hardcoded — they're estimates that work for
        # lab-paper-sized fonts (≈9–10 pt) and would need a small bump
        # for much larger fonts.
        sector_y_axes = 1.06  # axes-fraction y for sector dots above h
        breathing = _pt_to_in(4.0)  # ~4pt visual padding allowance

        panel_w = 6.5
        h_panel_h = panel_w * q / L  # square cells: h is q × L
        J_panel_h = panel_w  # square cells: J is L × L

        # Inter-panel gap: J's top tick mark + tick-pad + tick-label +
        # axes.labelpad + xlabel font height + breathing. Pulled from
        # rcParams so the gap matches whatever style is active.
        ticklabel_pt = _resolve_font_pt(mpl.rcParams["xtick.labelsize"])
        xlabel_pt = _resolve_font_pt(mpl.rcParams["axes.labelsize"])
        tick_pad_pt = mpl.rcParams["xtick.major.pad"]
        tick_size_pt = mpl.rcParams["xtick.major.size"]
        labelpad_pt = mpl.rcParams.get("axes.labelpad", 4.0)
        hgap = (
            _pt_to_in(
                tick_size_pt + tick_pad_pt + ticklabel_pt + labelpad_pt + xlabel_pt
            )
            + breathing
        )

        # Top margin above h panel: dot-row offset (sector_y_axes − 1
        # in axes fraction × panel inches) + sector label height +
        # breathing. The label sits at the same y as the dots.
        sector_offset_in = (sector_y_axes - 1.0) * h_panel_h
        sector_label_pt = _resolve_font_pt(mpl.rcParams["axes.labelsize"])
        top_pad = sector_offset_in + _pt_to_in(sector_label_pt) + breathing

        # Bottom margin: just figure breathing — J's xlabel rides on top.
        bottom_pad = _pt_to_in(8.0)

        # Horizontal margins. Left: y-axis label + tick labels (single
        # AA chars or 1–2-digit position numbers) + paddings. Right:
        # colorbar tick labels (e.g. "2.00", ~4 chars) + label.
        ytick_pt = _resolve_font_pt(mpl.rcParams["ytick.labelsize"])
        ytick_pad_pt = mpl.rcParams["ytick.major.pad"]
        ytick_size_pt = mpl.rcParams["ytick.major.size"]
        left_pad = (
            _pt_to_in(
                xlabel_pt
                + labelpad_pt
                + ytick_size_pt
                + ytick_pad_pt
                + 2.0 * ytick_pt  # ≈ width of two chars (e.g. "60")
            )
            + breathing
        )
        cb_gap = _pt_to_in(6.0)
        cb_w = _pt_to_in(13.0)
        right_pad = (
            _pt_to_in(
                ytick_size_pt
                + ytick_pad_pt
                + 2.5 * ytick_pt  # ≈ width of "2.00"
                + labelpad_pt
                + xlabel_pt
            )
            + breathing
        )

        fig_w = left_pad + panel_w + cb_gap + cb_w + right_pad
        fig_h = top_pad + h_panel_h + hgap + J_panel_h + bottom_pad

        # Manual inch-precise layout: opt out of constrained_layout
        # (on by default in lab-paper) so it can't override our
        # positioning, and so savefig doesn't warn about missing
        # layoutgrids.
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.set_layout_engine("none")

        # Place axes by inch coords, normalized to fig dimensions.
        def _rect(left_in, bottom_in, w_in, h_in):
            return [
                left_in / fig_w,
                bottom_in / fig_h,
                w_in / fig_w,
                h_in / fig_h,
            ]

        y_J = bottom_pad
        y_h = bottom_pad + J_panel_h + hgap
        x_panel = left_pad
        x_cb = left_pad + panel_w + cb_gap

        ax_h = fig.add_axes(_rect(x_panel, y_h, panel_w, h_panel_h))
        ax_J = fig.add_axes(
            _rect(x_panel, y_J, panel_w, J_panel_h),
            sharex=ax_h,
        )
        cax_h = fig.add_axes(_rect(x_cb, y_h, cb_w, h_panel_h))
        cax_J = fig.add_axes(_rect(x_cb, y_J, cb_w, J_panel_h))

        # Fields panel: q × L heatmap, diverging cmap centered at 0
        # (h after Zero_Sum_Gauge has zero mean across a, so symmetric
        # vmin/vmax frames the data correctly).
        # Symmetric vmin/vmax. Floor h_max so an identically-zero h
        # (untrained / hand-built model) doesn't collapse the diverging
        # cmap to a single color band.
        h_max = max(float(np.max(np.abs(h))) if h.size else 1.0, 1e-12)
        im_h = ax_h.imshow(
            h.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-h_max,
            vmax=h_max,
            interpolation="nearest",
            extent=(-0.5, L - 0.5, q - 0.5, -0.5),
        )
        ax_h.set_yticks(range(q))
        ax_h.set_yticklabels(list(alphabet))
        ax_h.set_ylabel("Amino acid $a$")
        # No x-ticks on h: the position axis is shared with J, which
        # carries its labels at its top edge (matrix convention: origin
        # in the top-left).
        ax_h.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=False,
        )
        # lab-paper sets axes.grid: True, which would draw a faint grid
        # over the heatmap cells. Disable on heatmap axes only.
        ax_h.grid(False)
        fig.colorbar(im_h, cax=cax_h, label=r"Field $h_i(a)$ (zero-sum gauge)")

        # Couplings panel: L × L Frobenius-norm heatmap. Origin is in the
        # top-left (origin="upper" + flipped y-extent), so the x-axis
        # ticks and label belong on top to mirror that.
        im_J = ax_J.imshow(
            J_norm,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            interpolation="nearest",
            extent=(-0.5, L - 0.5, L - 0.5, -0.5),
        )
        ax_J.xaxis.tick_top()
        ax_J.xaxis.set_label_position("top")
        ax_J.set_xlabel("Sequence position $i$")
        ax_J.set_ylabel("Sequence position $j$")
        ax_J.grid(False)
        fig.colorbar(
            im_J,
            cax=cax_J,
            label=r"Coupling norm $\Vert J_{ij}\Vert_F$ (zero-sum gauge)",
        )

        # Sector marks: black dots tucked just above the top edge of the
        # h panel, sharing x positions with both heatmaps. The mixed
        # transform (data x, axes y) lets us pin y to a small offset
        # above the panel regardless of figure size or row aspect.
        if sectors:
            ax_h.scatter(
                sectors,
                [sector_y_axes] * len(sectors),
                transform=ax_h.get_xaxis_transform(),
                s=12,
                color="black",
                marker="o",
                clip_on=False,
                zorder=3,
            )
            # "Sector" label sits just to the left of the panel,
            # vertically aligned with the dot row. Fully axes-fraction
            # so its position doesn't depend on data extent or font
            # metrics that left_pad already accounts for.
            ax_h.text(
                -0.005,
                sector_y_axes,
                "Sector",
                transform=ax_h.transAxes,
                ha="right",
                va="center",
            )


def density_scatter(x, y, Max, *, ax=None, fig=None, markersize=10, add_colorbar=True):
    """Scatter plot colored by 2D KDE density.

    Pass ``ax`` to draw into an existing axes (used by the PCA panel
    grid). Pass ``add_colorbar=False`` to suppress the per-axes
    colorbar — the PCA caller draws a single shared colorbar instead.
    Color normalization is pinned to ``[0, Max]`` via explicit
    ``vmin``/``vmax`` so the palette is consistent across panels even
    when each panel's KDE has a different scale.
    """
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure
    ax.scatter(x, y, c=z, s=markersize, cmap="magma", vmin=0, vmax=Max)

    if add_colorbar:
        norm = Normalize(vmin=0, vmax=Max)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax)
    return ax
