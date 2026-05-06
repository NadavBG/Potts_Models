"""Figure renderers for SBM/BM runs.

Original author: Marion CHAUVEAU (October 2022).

Refactored to take a list of artificial alignments (one per sampling
temperature) and emit consolidated multi-panel figures: a single
``Correlations`` figure with rows=temperatures × cols=orders replaces
the legacy ``Freq`` / ``Pair_freq`` / ``Corr3`` triplet, and PCA
becomes a ``1 × (1 + N_temps)`` grid (natural + each artificial).
Energy / similarity / diversity / length overlay all temperatures in
one panel.
"""

####################### MODULES #######################

import logging

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import SBM.utils.utils as ut
from matplotlib.colors import Normalize  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from matplotlib import cm  # type: ignore

log = logging.getLogger(__name__)

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


def plot_stats(output, plot="Correlations", *, artificial=None):
    """Render one figure for the requested plot mode.

    Parameters
    ----------
    output
        The model dict (carries ``align``, ``Train``, optionally ``Test``,
        ``h``, ``J``, ``J_norm``, ``options``).
    plot
        Mode name. One of ``Correlations``, ``PCA``, ``Energy``,
        ``Coupling_evol``, ``Similarity``, ``Diversity``, ``Length``.
    artificial
        List of dicts, one per sampling temperature, each with keys
        ``temperature`` (float | None), ``align_mod`` (ndarray, required
        for align-needing modes), ``stats`` (dict from compute_stats,
        required for Correlations). Empty / None for ``Coupling_evol``.
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
        panels = [(axes[0], X_nat, "Natural sequences")]
        for ax, item, X_a in zip(axes[1:], artificial, art_Xs):
            panels.append(
                (ax, X_a, _art_label("Artificial sequences", item["temperature"]))
            )
        # density_scatter draws the points with vmin=0, vmax=Max so the
        # color mapping is identical across panels. We add a single
        # shared colorbar after the loop instead of one per panel.
        for ax, pts, title in panels:
            density_scatter(pts[:, 0], pts[:, 1], Max=Max, ax=ax, add_colorbar=False)
            ax.set_xlim(mi1, ma1)
            ax.set_ylim(mi2, ma2)
            ax.set_xlabel("PC 1")
            ax.set_title(title)
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
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        Bins = 60
        # A random alignment as a worst-case baseline. Shape from the
        # first artificial set; all share L (validated upstream).
        rand = np.round(np.random.random(artificial[0]["align_mod"].shape)).astype(
            "int32"
        )

        Etest = ut.compute_energies(output["Test"], output["h"], output["J"])
        Etrain = ut.compute_energies(output["Train"], output["h"], output["J"])
        Erand = ut.compute_energies(rand, output["h"], output["J"])
        Emods = [
            (
                item["temperature"],
                ut.compute_energies(item["align_mod"], output["h"], output["J"]),
            )
            for item in artificial
        ]
        mu, sd = float(np.mean(Etrain)), float(np.std(Etrain))
        Etest = (Etest - mu) / sd
        Etrain = (Etrain - mu) / sd
        Erand = (Erand - mu) / sd
        Emods = [(T, (e - mu) / sd) for T, e in Emods]

        all_e = [Etest, Etrain, Erand] + [e for _, e in Emods]
        mi = float(np.amin(np.concatenate(all_e)))
        ma = float(np.amax(np.concatenate(all_e)))
        common = dict(bins=Bins, range=(mi - 0.5, ma + 0.5), alpha=0.5, density=True)
        ax.hist(Etest, label="Test", **common)
        ax.hist(Etrain, label="Train", **common)
        for T, e in Emods:
            ax.hist(e, label=_art_label("Artificial", T), **common)
        ax.hist(Erand, label="Random", color="0.6", **common)
        ax.legend()
        ax.set_xlabel("Statistical energy (z-score)")
        ax.set_ylabel("Probability density")

    if plot == "Similarity":
        if not artificial:
            raise ValueError("Similarity plot requires at least one artificial set")
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        Bins = 80
        common = dict(bins=Bins, range=(0, 1), alpha=0.5, density=True)
        ax.hist(
            ut.compute_similarities(output["Test"], output["Train"]),
            label="Test",
            **common,
        )
        ax.hist(ut.compute_similarities(output["Train"]), label="Train", **common)
        for item in artificial:
            ax.hist(
                ut.compute_similarities(item["align_mod"], output["Train"]),
                label=_art_label("Artificial", item["temperature"]),
                **common,
            )
        ax.set_xlabel("Distance to closest natural seq")
        ax.set_ylabel("Probability density")
        ax.legend()

    if plot == "Diversity":
        if not artificial:
            raise ValueError("Diversity plot requires at least one artificial set")
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        Bins = 60
        common = dict(bins=Bins, range=(0, 1), alpha=0.5, density=True)
        ax.hist(ut.compute_diversity(output["Train"]), label="Train", **common)
        ax.hist(ut.compute_diversity(output["Test"]), label="Test", **common)
        for item in artificial:
            ax.hist(
                ut.compute_diversity(item["align_mod"]),
                label=_art_label("Artificial", item["temperature"]),
                **common,
            )
        ax.legend()
        ax.set_xlabel("Diversity")
        ax.set_ylabel("Probability density")

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
        common = dict(bins=Bins, range=(mi, ma), alpha=0.5, density=True)
        ax.hist(Length_train, label="Train", **common)
        ax.hist(Length_test, label="Test", **common)
        for T, L in Length_arts:
            ax.hist(L, label=_art_label("Artificial", T), **common)
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
