"""Render figures for an SBM run.

Reads ``<run_dir>/model.npy`` and saves figures into ``<run_dir>/figs/``.
Idempotent: re-running on the same run dir loads from cache and just
re-renders, so this is also the right script to run after editing
figure code.

By default only ``coupling_evol`` is rendered — it is the only figure
that does not require a synthetic alignment (which is a separate
post-training sampling step). To render diagnostic figures that
compare data statistics to a model-sampled MSA (``freq``, ``pair_freq``,
``corr3``, ``pca``, ``energy``, ``similarity``, ``diversity``,
``length``), pass them explicitly via ``--figs``. The synthetic
alignment is then drawn at inference temperature T=1 (matching how
the model was trained) and cached under ``fig_data/``.

Reuses ``SBM.utils.utils_plot.plot_stats`` for the nine plot modes.
PDFs go through ``lab_plotting.save_figure`` (sibling script in
``scripts/``) so the git commit, calling-script path, and run id end up
in the PDF metadata.

Usage::

    python scripts/render_figures.py <run_dir> [--figs name [name ...]]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import SBM.utils.utils as ut
import SBM.utils.utils_plot as up

# scripts/lab_plotting.py is a sibling, not a package module. Add this
# script's directory to sys.path so the plain `import lab_plotting` works
# regardless of whether the user `cd`'s into scripts/ first.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

#: All plot modes implemented in ``utils_plot.plot_stats``. Order is the
#: rendering order; figure files use these exact names as the stem.
ALL_FIGS: tuple[str, ...] = (
    "freq",
    "pair_freq",
    "corr3",
    "pca",
    "energy",
    "coupling_evol",
    "similarity",
    "diversity",
    "length",
)

#: Figures rendered by default. Restricted to those that need no
#: synthetic alignment, since sampling from the trained model is a
#: separate downstream step. Everything else in ALL_FIGS is opt-in.
DEFAULT_FIGS: tuple[str, ...] = ("coupling_evol",)

# ``utils_plot.plot_stats`` selects modes by string. Map our snake_case
# figure names to the ``plot=...`` strings that file expects.
_PLOT_MODES: dict[str, str] = {
    "freq": "Freq",
    "pair_freq": "Pair_freq",
    "corr3": "Corr3",
    "pca": "PCA",
    "energy": "Energy",
    "coupling_evol": "Coupling_evol",
    "similarity": "Similarity",
    "diversity": "Diversity",
    "length": "Length",
}

#: Figures whose data needs ``compute_stats(...)`` output.
_NEEDS_STATS: frozenset[str] = frozenset({"freq", "pair_freq", "corr3"})

#: Figures whose data needs an artificial alignment.
_NEEDS_ALIGN_MOD: frozenset[str] = frozenset(
    {"pca", "energy", "similarity", "diversity", "length"}
)

#: Figures that need a real test set (i.e. ``Test/Train=1`` at training).
#: ``compute_stats`` handles the missing case (it substitutes Train for
#: Test internally), so the freq/pair_freq/corr3 plots work without a
#: test set; the others read ``output["Test"]`` directly and crash.
_NEEDS_TEST: frozenset[str] = frozenset({"energy", "similarity", "diversity", "length"})


def _load_model(run_dir: Path) -> dict:
    model_path = run_dir / "model.npy"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no model at {model_path} — pass a run dir produced by "
            "scripts/train_sbm.py"
        )
    return np.load(model_path, allow_pickle=True).item()


def _read_seed(run_dir: Path) -> int | None:
    """Pull the master seed from the run's manifest, if present."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    seed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed")
    return int(seed) if seed is not None else None


def _seed_global_rng(seed: int | None) -> None:
    """Seed numpy's global RNG so the unseeded ``np.random.choice`` calls
    inside ``compute_stats`` and the PCA renderer produce reproducible
    subsamples. No-op if the run was started without ``--seed``.
    """
    if seed is None:
        log.info("no seed in manifest; figure subsamples will be nondeterministic")
        return
    np.random.seed(seed)


def _ensure_align_mod(
    model: dict, fig_data_dir: Path, *, seed: int | None
) -> np.ndarray:
    """Sample an artificial alignment from the model (or load from cache).

    Cached at ``fig_data/align_mod.npy``. Sample size matches the input MSA
    so similarity/diversity histograms have comparable supports across the
    Train / Test / Artificial splits.
    """
    cache = fig_data_dir / "align_mod.npy"
    if cache.is_file():
        log.info("loading cached align_mod from %s", cache)
        return np.load(cache)
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    n_seq = int(model["align"].shape[0])
    delta_t = int(model["options0"]["k_MCMC"])
    log.info("sampling artificial alignment: N=%d, delta_t=%d", n_seq, delta_t)
    align_mod = ut.Create_modAlign(model, n_seq, delta_t=delta_t, seed=seed)
    np.save(cache, align_mod)
    return align_mod


def _ensure_stats(model: dict, align_mod: np.ndarray, fig_data_dir: Path) -> dict:
    """Compute (or load cached) train/test/artificial statistics dict.

    Cached at ``fig_data/stats.npy``. ``compute_stats`` is the slowest
    bit because of the 3-point correlation tensor; the cache lets you
    re-render figures cheaply after tweaking plot code.
    """
    cache = fig_data_dir / "stats.npy"
    if cache.is_file():
        log.info("loading cached stats from %s", cache)
        return np.load(cache, allow_pickle=True).item()
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    log.info("computing stats (Train, Test, Artificial; this is the slow part)")
    stats = ut.compute_stats(model, align_mod)
    np.save(cache, stats)
    return stats


def _patch_legacy_options(model: dict) -> None:
    """``utils_plot.plot_stats`` reads ``output["options"]`` (not split
    into options0/options1) — and the Coupling_evol mode specifically
    reads ``options["n_states"]`` and ``options["m"]``. Older models
    written by hand had a single ``options`` key; train_sbm.py writes
    ``options0`` + ``options1``. Build a flat alias that covers both
    cases without modifying utils_plot.py.
    """
    if "options" in model:
        return
    flat: dict = {}
    flat.update(model.get("options0", {}))
    flat.update(model.get("options1", {}))
    # ``n_states`` was the legacy name for ``N_chains``; preserve back-compat
    # for the Coupling_evol title without inventing a new key elsewhere.
    flat.setdefault("n_states", flat.get("N_chains"))
    model["options"] = flat


def _rasterize_dense_artists(fig: plt.Figure) -> None:
    """Rasterize Line2D / scatter artists in-place.

    ``utils_plot.plot_stats`` builds correlation scatters with millions of
    points (``pair_freq``: ~L(L-1)/2 * q^2 = ~2M; ``corr3`` is worse).
    Rendered as vectors, the resulting PDFs are tens of MB. Rasterizing
    keeps PDF size reasonable while leaving axes / labels / titles as
    real text. ``savefig(..., dpi=...)`` controls the raster resolution
    (we leave matplotlib's default of 100, which is fine for the
    correlation-cloud visual).
    """
    for ax in fig.axes:
        # Flatten everything plottable; keep text untouched.
        for artist in ax.lines + ax.collections + ax.patches:
            artist.set_rasterized(True)


def _render_one(
    name: str,
    model: dict,
    stats: dict | None,
    figs_dir: Path,
    *,
    run_id: str,
) -> list[Path]:
    """Call ``plot_stats(plot=mode)`` and save every Figure it created.

    PCA produces two figures (natural / artificial); we save both with
    a numeric suffix. Other modes produce one.
    """
    mode = _PLOT_MODES[name]
    before = set(plt.get_fignums())
    up.plot_stats(model, stats or {}, plot=mode)
    after = set(plt.get_fignums())
    new_fignums = sorted(after - before)
    if not new_fignums:
        log.warning("plot_stats(plot=%r) produced no figure", mode)
        return []
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    multi = len(new_fignums) > 1
    for i, n in enumerate(new_fignums):
        fig = plt.figure(n)
        _rasterize_dense_artists(fig)
        stem = f"{name}_{i}" if multi else name
        path = figs_dir / f"{stem}.pdf"
        # Use 'Keywords' for the run id — matplotlib's PDF backend
        # filters out unknown infodict keys with a warning, so we stick
        # to a standard one.
        lab_plotting.save_figure(
            fig,
            path,
            script_path=Path(__file__),
            extra_metadata={"Keywords": f"sbm_run_id={run_id}"},
        )
        plt.close(fig)
        written.append(path)
    return written


def _filter_renderable(model: dict, requested: list[str]) -> list[str]:
    """Drop figures whose required data isn't present in the model.

    Currently: anything in ``_NEEDS_TEST`` is dropped if the model was
    trained with ``Test/Train=0`` (so ``output["Test"]`` is None).
    """
    keep: list[str] = []
    has_test = model.get("Test") is not None
    for name in requested:
        if name in _NEEDS_TEST and not has_test:
            log.warning("skipping %r: this run has no test set (Test/Train=0)", name)
            continue
        keep.append(name)
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render figures for an SBM run.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="path to a run directory produced by scripts/train_sbm.py",
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        choices=ALL_FIGS,
        default=list(DEFAULT_FIGS),
        metavar="NAME",
        help=(
            "which figures to render. Default: "
            + ", ".join(DEFAULT_FIGS)
            + " (the only figures that don't require sampling a synthetic "
            "alignment). Other choices, all of which need a synthetic "
            "alignment: "
            + ", ".join(f for f in ALL_FIGS if f not in DEFAULT_FIGS)
            + "."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")

    fig_data_dir = run_dir / "fig_data"
    figs_dir = run_dir / "figs"

    model = _load_model(run_dir)
    _patch_legacy_options(model)
    seed = _read_seed(run_dir)
    _seed_global_rng(seed)

    requested = _filter_renderable(model, list(args.figs))
    if not requested:
        log.warning("no figures to render")
        return 0

    # Activate lab style. Wrapped because the user may not have the
    # lab-paper stylesheet installed; falling back to mpl defaults is
    # acceptable for routine runs.
    try:
        plt.style.use("lab-paper")
    except (OSError, ValueError):
        log.info("'lab-paper' stylesheet not available; using matplotlib defaults")

    # Materialize fig_data once.
    needs_align_mod = bool(set(requested) & _NEEDS_ALIGN_MOD)
    needs_stats = bool(set(requested) & _NEEDS_STATS)
    align_mod = None
    stats = None
    if needs_align_mod or needs_stats:
        align_mod = _ensure_align_mod(model, fig_data_dir, seed=seed)
        model["align_mod"] = align_mod
    if needs_stats:
        stats = _ensure_stats(model, align_mod, fig_data_dir)

    run_id = run_dir.name
    written: list[Path] = []
    for name in requested:
        log.info("rendering %s", name)
        written.extend(_render_one(name, model, stats, figs_dir, run_id=run_id))

    log.info("wrote %d figure(s) under %s", len(written), figs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
