"""Render figures for an SBM run.

Reads ``<run_dir>/model.npy`` and saves figures into ``<run_dir>/figs/``.
Re-running rebuilds everything inside ``figs/``.

By default this renders every figure whose required data is present in
the run:

* ``coupling_evol`` and ``params`` are always rendered (both depend
  only on ``model.npy``). ``params`` is a two-panel field/coupling
  heatmap: the L×q ``h`` matrix on top and the L×L Frobenius norm of
  ``J[i,j]`` below; CM runs (L=96) get a sector strip above the panels
  using Emily's sector definition from ``CM_sector.py``.
* ``correlations`` and ``pca`` are rendered if at least one synthetic
  alignment is available (auto-discovered under
  ``<run_dir>/synthetic/`` or supplied with ``--synthetic-alignment``).
  Multiple alignments become multi-panel comparisons:
  ``correlations`` becomes a (rows = temperatures) × (cols = 1st/2nd/
  3rd order) grid and ``pca`` becomes a 1×(1+N) grid (natural + each
  artificial).
* ``energy``, ``similarity``, ``diversity`` are rendered when at
  least one synthetic alignment is available. ``similarity`` and
  ``diversity`` are violin plots (one violin per group: Train,
  optionally Test, and one per artificial T); ``energy`` is an
  overlaid histogram (Natural Train, optionally Test, each artificial
  T, plus a Random worst-case baseline). All three include Test as
  an extra group if the run has ``Test/Train > 0``, otherwise fall
  back to Train alone.
* ``length`` additionally requires that the run was trained with
  ``Test/Train > 0`` (i.e. ``model["Test"]`` is not None) since the
  histogram reads Test directly.

Figures whose data is missing are skipped with an info log message.
Pass ``--figs NAME [NAME ...]`` to override the default — explicitly
requesting a figure whose data is missing is an error rather than a
silent skip.

Reuses ``SBM.utils.utils_plot.plot_stats`` for the eight plot modes.
PDFs go through ``lab_plotting.save_figure`` (sibling script in
``scripts/``) so the git commit, calling-script path, and run id end up
in the PDF metadata. ``figs/inputs/sources.json`` records the absolute
paths and sha256s of the model and synthetic-alignment files that fed
the figures, without duplicating their bytes.

Usage::

    python scripts/render_figures.py <run_dir> \\
        [--synthetic-alignment PATH ...] [--figs name [name ...]]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import SBM.provenance as provenance
import SBM.utils.utils as ut
import SBM.utils.utils_mpnn_plot as ump
import SBM.utils.utils_plot as up

# scripts/lab_plotting.py is a sibling, not a package module. Add this
# script's directory to sys.path so the plain `import lab_plotting` works
# regardless of whether the user `cd`'s into scripts/ first.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import lab_plotting  # noqa: E402

# CM_sector.py lives at the repo root and defines the catalytic-sector
# residue positions used to annotate the params figure on CM runs.
# Loaded by absolute path through importlib so we don't have to push
# the repo root onto sys.path (which would risk shadowing same-named
# modules from a sibling project).
import importlib.util as _importlib_util  # noqa: E402

_CM_SECTOR_PATH = _SCRIPT_DIR.parent / "CM_sector.py"
_cm_spec = _importlib_util.spec_from_file_location("CM_sector", _CM_SECTOR_PATH)
if _cm_spec is None or _cm_spec.loader is None:
    raise ImportError(f"could not build module spec for {_CM_SECTOR_PATH}")
CM_sector = _importlib_util.module_from_spec(_cm_spec)
_cm_spec.loader.exec_module(CM_sector)

log = logging.getLogger(__name__)

#: All plot modes implemented in ``utils_plot.plot_stats``. Order is the
#: rendering order; figure files use these exact names as the stem.
#:
#: ``correlations`` is one figure with a (rows = temperatures) ×
#: (cols = 1st/2nd/3rd order statistics) grid — replaces the legacy
#: ``freq`` / ``pair_freq`` / ``corr3`` triplet so every order can be
#: read off side by side.
ALL_FIGS: tuple[str, ...] = (
    "correlations",
    "pca",
    "energy",
    "coupling_evol",
    "params",
    "similarity",
    "diversity",
    "length",
    "mpnn",
)

#: When ``--figs`` is omitted we try to render all of ``ALL_FIGS`` and
#: drop figures whose required data isn't present in the run (no
#: synthetic alignment, no test set). The always-renderable figures
#: depend only on ``model.npy``.
_ALWAYS_RENDERABLE: tuple[str, ...] = ("coupling_evol", "params")

# ``utils_plot.plot_stats`` selects modes by string. Map our snake_case
# figure names to the ``plot=...`` strings that file expects.
_PLOT_MODES: dict[str, str] = {
    "correlations": "Correlations",
    "pca": "PCA",
    "energy": "Energy",
    "coupling_evol": "Coupling_evol",
    "params": "Params",
    "similarity": "Similarity",
    "diversity": "Diversity",
    "length": "Length",
}

#: Figures whose data needs ``compute_stats(...)`` output.
_NEEDS_STATS: frozenset[str] = frozenset({"correlations"})

#: Figures whose data needs an artificial alignment.
_NEEDS_ALIGN_MOD: frozenset[str] = frozenset(
    {"correlations", "pca", "energy", "similarity", "diversity", "length"}
)

#: Figures that need a real test set (i.e. ``Test/Train>0`` at training).
#: ``correlations`` adapts to no-test by comparing against Train.
#: ``energy``, ``similarity``, and ``diversity`` now also adapt: they
#: include Test as an extra histogram / violin if present, otherwise
#: fall back to Train alone. ``length`` still reads ``output["Test"]``
#: directly, so it's the only mode listed here.
_NEEDS_TEST: frozenset[str] = frozenset({"length"})

#: Figures whose data needs a ProteinMPNN sweep subdir under
#: ``<run_dir>/synthetic/mpnn_sweep_*/`` containing ``mpnn_scores.json``.
_NEEDS_MPNN_SWEEP: frozenset[str] = frozenset({"mpnn"})


#: ``--sector`` choices wired through to ``_cm_sector_positions``.
#: ``"emily"`` and ``"rama"`` pull the corresponding ATS lists from
#: ``CM_sector``; ``"none"`` disables the sector strip on the params
#: figure entirely.
SECTOR_CHOICES: tuple[str, ...] = ("emily", "rama", "none")


def _looks_like_cm_run(run_dir: Path) -> bool:
    """``True`` iff the run directory looks like a CM run by layout.

    Both ``results/<family>/<run>/`` (training) and
    ``pruning/example_output/<family>/<run>/`` (worked example) put the
    family name as the parent directory, so the CM check is simply
    ``parent.name == "CM"`` (case-insensitive). L=96 alone is
    insufficient — another protein might happen to share that length —
    so the sector annotation is gated on the family marker too.
    """
    return run_dir.parent.name.lower() == "cm"


def _cm_sector_positions(run_dir: Path, L: int, choice: str = "emily") -> list[int]:
    """MSA-column indices for the requested CM catalytic-sector
    definition on the 96-AA alignment.

    ``choice`` is one of ``SECTOR_CHOICES``: ``"emily"`` (default,
    ``CM_SECTOR_EMILY``), ``"rama"`` (``CM_SECTOR_RAMA``,
    near-identical with a few differing residues), or ``"none"``.
    Returns ``[]`` (and logs the reason) when annotation is suppressed:
    ``choice == "none"``, ``L != 96``, or the run dir doesn't look like
    a CM run. Pass ``--sector none`` to silence the suppression log.
    """
    if choice not in SECTOR_CHOICES:
        raise ValueError(
            f"unknown sector choice {choice!r}; expected one of {SECTOR_CHOICES}"
        )
    if choice == "none":
        return []
    if L != 96:
        log.info(
            "sector annotation suppressed: L=%d is not 96. "
            "Pass --sector none to silence.",
            L,
        )
        return []
    if not _looks_like_cm_run(run_dir):
        log.info(
            "sector annotation suppressed: %s is not under a CM family "
            "directory. Pass --sector none to silence, or move the run "
            "under results/CM/ to enable.",
            run_dir,
        )
        return []
    ats_set = (
        CM_sector.CM_SECTOR_EMILY if choice == "emily" else CM_sector.CM_SECTOR_RAMA
    )
    # Defensive: every ATS in the chosen sector must be addressable in
    # the WT_ATS_96 → MSA-position map. Fails loudly if a future edit
    # to CM_sector.py points at a sentinel (-1) position.
    missing = [int(a) for a in ats_set if int(a) not in CM_sector.ATS_TO_POS_96]
    if missing:
        raise ValueError(
            f"CM sector definition {choice!r} references ATS values not "
            f"present in CM_sector.ATS_TO_POS_96: {missing}"
        )
    return sorted(CM_sector.ATS_TO_POS_96[int(ats)] for ats in ats_set)


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


def _autodiscover_synthetic(run_dir: Path) -> list[Path]:
    """Return every synthetic alignment under ``<run_dir>/synthetic/``,
    sorted by filename. The default sampling workflow writes one file
    per temperature, and the renderer fans them out into multi-panel
    figures (one panel per T).

    Restricted to direct children of ``synthetic/`` (``glob`` not
    ``rglob``) so artifacts from a ProteinMPNN sweep, which live in a
    ``mpnn_sweep_*/`` subdir, don't leak into the energy / similarity
    / diversity / pca / correlations figures (those are designed for
    the standard 2-T compare, not the 10-T foldability ladder).
    """
    syn_dir = run_dir / "synthetic"
    if not syn_dir.is_dir():
        return []
    return sorted(syn_dir.glob("*.npy"))


def _autodiscover_mpnn_sweep(run_dir: Path) -> Path | None:
    """Return the path to a sweep's ``mpnn_scores.json``, if any.

    Looks for ``<run_dir>/synthetic/mpnn_sweep_*/mpnn_scores.json``. If
    multiple sweeps exist, picks the lexicographically last (newest by
    convention since the dir name embeds the master seed) and warns.
    """
    syn_dir = run_dir / "synthetic"
    if not syn_dir.is_dir():
        return None
    candidates = sorted(syn_dir.glob("mpnn_sweep_*/mpnn_scores.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(
            "found %d mpnn_sweep dirs; using %s (latest by sort order)",
            len(candidates),
            candidates[-1].parent.name,
        )
    return candidates[-1]


def _read_sampling_temperature(align_path: Path) -> float | None:
    """Read the sampling temperature from the sidecar JSON next to a
    synthetic alignment (``align_T<T>_seed<seed>.json`` written by
    ``scripts/sample_sbm.py``). Returns ``None`` if the sidecar is
    absent or unreadable — figure labels then omit the temperature.
    """
    sidecar = align_path.with_suffix(".json")
    if not sidecar.is_file():
        log.warning(
            "no sidecar JSON found at %s — figure labels will not show "
            "the sampling temperature",
            sidecar,
        )
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not parse %s: %s", sidecar, exc)
        return None
    T = meta.get("temperature")
    return float(T) if T is not None else None


def _load_align_mod(path: Path, model: dict) -> np.ndarray:
    """Load a synthetic alignment from disk and validate against the model.

    Sampling is now a separate step (``scripts/sample_sbm.py``); this script
    only consumes the alignment file. We check ``L`` matches the training
    MSA and warn on mismatched ``N`` (similarity / diversity histograms
    visually compare distributions, so unequal supports are workable but
    worth flagging).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"synthetic alignment not found at {path}. Generate one with "
            "scripts/sample_sbm.sh <run_dir> first."
        )
    align_mod = np.load(path)
    train_align = model["align"]
    if not np.issubdtype(align_mod.dtype, np.integer):
        raise ValueError(
            f"synthetic alignment {path} has dtype {align_mod.dtype}; "
            "expected an integer alignment (output of sample_sbm.py)"
        )
    if align_mod.ndim != 2 or align_mod.shape[1] != train_align.shape[1]:
        raise ValueError(
            f"synthetic alignment shape {align_mod.shape} does not match "
            f"model L={train_align.shape[1]} (training shape {train_align.shape})"
        )
    log.info("loaded synthetic alignment %s (shape %s)", path, align_mod.shape)
    return align_mod


def _ensure_stats(
    model: dict,
    align_mod: np.ndarray,
    align_path: Path,
    figs_inputs_dir: Path,
    *,
    rng_seed: int | None,
) -> dict:
    """Compute (or load cached) train/test/artificial statistics for a
    single artificial alignment.

    Cache file is keyed on the alignment filename: a run with N
    artificial alignments produces N ``stats_<stem>.npy`` files. The
    Train/Test stats are duplicated across files (cheap); the slow
    artificial three-point correlation tensor is unique per file.

    ``compute_stats`` consumes the numpy global RNG to subsample the
    Train/Test rows for the 3-point tensor. We reseed before each
    call (with the run's master seed) so the Train/Test subsamples
    are *identical* across cache files — otherwise different rows of
    the consolidated ``correlations`` figure would be comparing the
    same artificial alignment against subtly different Test reference
    rows. Pass ``rng_seed=None`` only if reproducibility doesn't
    matter (e.g. the run had no `--seed`).
    ``render_sbm.sh`` blows away ``figs/`` before each call so the
    recompute path is exercised whenever inputs change.
    """
    cache = figs_inputs_dir / f"stats_{align_path.stem}.npy"
    if cache.is_file():
        log.info("loading cached stats from %s", cache)
        return np.load(cache, allow_pickle=True).item()
    figs_inputs_dir.mkdir(parents=True, exist_ok=True)
    log.info("computing stats for %s (slow: 3-point correlation)", align_path.name)
    if rng_seed is not None:
        np.random.seed(rng_seed)
    stats = ut.compute_stats(model, align_mod)
    np.save(cache, stats)
    return stats


def _write_sources_json(
    figs_inputs_dir: Path,
    *,
    run_dir: Path,
    model_path: Path,
    artificial: list[dict],
    figures: list[str],
) -> None:
    """Record what fed the figures: paths + sha256s of model.npy and
    each synthetic alignment used. Pointer file, not a copy.
    ``artificial`` is the per-alignment list of dicts assembled
    inline in ``main()``.
    """
    figs_inputs_dir.mkdir(parents=True, exist_ok=True)
    sources: dict = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "rendered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "figures": list(figures),
        "model": {
            "path": str(model_path),
            "sha256": provenance.file_sha256(model_path),
        },
        "synthetic_alignments": [
            {
                "path": str(item["path"]),
                "sha256": provenance.file_sha256(item["path"]),
                "sampling_temperature": item["temperature"],
            }
            for item in artificial
        ],
        "code": {
            "git_commit": provenance.git_commit(),
            "git_dirty": provenance.git_dirty(),
            "git_branch": provenance.git_branch(),
        },
    }
    out_path = figs_inputs_dir / "sources.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)
        f.write("\n")


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

    The consolidated ``correlations`` figure scatters millions of
    points per panel (pairwise: ~L(L-1)/2 * q^2 = ~2M; 3rd order is
    worse). Rendered as vectors the PDFs are tens of MB; rasterizing
    keeps file size reasonable while leaving axes / labels / titles as
    real text. ``savefig(..., dpi=...)`` controls the raster resolution
    (we leave matplotlib's default of 100, fine for cloud-of-points).
    """
    for ax in fig.axes:
        # Flatten everything plottable; keep text untouched.
        for artist in ax.lines + ax.collections + ax.patches:
            artist.set_rasterized(True)


def _render_one(
    name: str,
    model: dict,
    artificial: list[dict],
    natural_colors: dict[str, str],
    figs_dir: Path,
    *,
    run_id: str,
    sector_positions: list[int],
    mpnn_scores_path: Path | None = None,
) -> list[Path]:
    """Render one named figure and save the result(s) under ``figs_dir``.

    Most figures are dispatched to ``plot_stats(plot=mode, artificial=...)``,
    which draws into figures it creates internally; we diff
    ``plt.get_fignums()`` to capture them. The ``mpnn`` figure breaks
    that pattern — it reads ``mpnn_scores.json`` from the sweep subdir
    and is recipe-shaped (returns a Figure directly), so it's branched
    here rather than threaded through ``plot_stats``.

    ``artificial`` is the per-alignment list of dicts (with
    ``temperature``, ``align_mod``, ``stats``, ``color``); ignored by
    Coupling_evol, Params, and the mpnn branch. ``natural_colors``
    carries Train/Test/Random colors. ``sector_positions`` is read only
    by the Params figure. ``mpnn_scores_path`` is read only by ``mpnn``.

    ``write_sidecar=False`` on every save keeps the renderer from being
    copied once per PDF; one canonical copy lives in ``figs/inputs/``.
    """
    figs_dir.mkdir(parents=True, exist_ok=True)
    if name == "mpnn":
        if mpnn_scores_path is None:
            log.warning("mpnn figure requested without scores path; skipping")
            return []
        fig = ump.plot_mpnn_foldability(mpnn_scores_path)
        path = figs_dir / "mpnn.pdf"
        lab_plotting.save_figure(
            fig,
            path,
            script_path=Path(__file__),
            write_sidecar=False,
            extra_metadata={"Keywords": f"sbm_run_id={run_id}"},
        )
        plt.close(fig)
        return [path]

    mode = _PLOT_MODES[name]
    before = set(plt.get_fignums())
    up.plot_stats(
        model,
        plot=mode,
        artificial=artificial,
        natural_colors=natural_colors,
        sector_positions=sector_positions,
    )
    after = set(plt.get_fignums())
    new_fignums = sorted(after - before)
    if not new_fignums:
        log.warning("plot_stats(plot=%r) produced no figure", mode)
        return []
    if len(new_fignums) > 1:
        log.warning(
            "plot_stats(plot=%r) created %d figures; expected 1. Consolidation "
            "rule: similar plots belong in one multi-panel figure.",
            mode,
            len(new_fignums),
        )
    written: list[Path] = []
    multi = len(new_fignums) > 1
    for i, n in enumerate(new_fignums):
        fig = plt.figure(n)
        _rasterize_dense_artists(fig)
        stem = f"{name}_{i}" if multi else name
        path = figs_dir / f"{stem}.pdf"
        lab_plotting.save_figure(
            fig,
            path,
            script_path=Path(__file__),
            write_sidecar=False,
            extra_metadata={"Keywords": f"sbm_run_id={run_id}"},
        )
        plt.close(fig)
        written.append(path)
    return written


def _filter_renderable(
    model: dict,
    requested: list[str],
    *,
    have_align: bool,
    have_mpnn_sweep: bool,
    strict: bool,
) -> list[str]:
    """Drop figures whose required data isn't present in this run.

    Three reasons a figure may not be renderable:
      * no synthetic alignment available (anything in ``_NEEDS_ALIGN_MOD``);
      * no held-out test set on the run, i.e. ``Test/Train=0`` at
        training time so ``model["Test"]`` is None (anything in
        ``_NEEDS_TEST``).
      * no ProteinMPNN sweep dir under ``<run_dir>/synthetic/`` (anything
        in ``_NEEDS_MPNN_SWEEP``).

    ``strict`` controls how a missing requirement is reported. When the
    user explicitly listed figures via ``--figs`` we treat *any* missing
    requirement as a fatal error — they asked for something we can't
    make. In default mode we skip with an info log instead.
    """
    keep: list[str] = []
    has_test = model.get("Test") is not None
    for name in requested:
        if name in _NEEDS_TEST and not has_test:
            if strict:
                raise SystemExit(
                    f"error: {name!r} requires a held-out test set, but "
                    "this run was trained with Test/Train=0. Re-run "
                    "training with --TestTrain 1, or omit this figure."
                )
            log.warning("skipping %r: this run has no test set (Test/Train=0)", name)
            continue
        if name in _NEEDS_ALIGN_MOD and not have_align:
            if strict:
                raise SystemExit(
                    f"error: {name!r} requires a synthetic alignment, but "
                    "none was found under <run_dir>/synthetic/. Run "
                    "`bash scripts/sample_sbm.sh <run_dir>` first, or pass "
                    "--synthetic-alignment PATH."
                )
            log.info(
                "skipping %r: no synthetic alignment under <run_dir>/synthetic/",
                name,
            )
            continue
        if name in _NEEDS_MPNN_SWEEP and not have_mpnn_sweep:
            if strict:
                raise SystemExit(
                    f"error: {name!r} requires a ProteinMPNN sweep but no "
                    "mpnn_sweep_*/mpnn_scores.json was found under "
                    "<run_dir>/synthetic/. Run "
                    "`bash scripts/sample_sbm.sh <run_dir> --mpnn-sweep` first."
                )
            log.info(
                "skipping %r: no mpnn_sweep_*/mpnn_scores.json under "
                "<run_dir>/synthetic/",
                name,
            )
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
        "--synthetic-alignment",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "one or more synthetic alignment .npy files (produced by "
            "scripts/sample_sbm.py). If omitted, render_figures.py "
            "auto-discovers every alignment under <run_dir>/synthetic/. "
            "Multiple alignments become multi-panel comparisons (one "
            "panel per temperature). Figures that depend on a synthetic "
            "alignment (" + ", ".join(sorted(_NEEDS_ALIGN_MOD)) + ") are "
            "skipped if none is available."
        ),
    )
    parser.add_argument(
        "--sector",
        choices=SECTOR_CHOICES,
        default="emily",
        help=(
            "which CM catalytic-sector definition to mark on the "
            "params figure: 'emily' (default, CM_SECTOR_EMILY), 'rama' "
            "(CM_SECTOR_RAMA — near-identical, differs by a few "
            "residues), or 'none' to disable the sector strip. The "
            "annotation is auto-suppressed when L != 96 or when the "
            "run dir isn't under a CM family directory; pass 'none' "
            "to silence the info-log explaining why."
        ),
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        choices=ALL_FIGS,
        default=None,
        metavar="NAME",
        help=(
            "which figures to render. Default: every figure whose data "
            "is present in the run — always "
            + ", ".join(_ALWAYS_RENDERABLE)
            + "; the alignment-dependent ones ("
            + ", ".join(sorted(_NEEDS_ALIGN_MOD | _NEEDS_STATS))
            + ") if a synthetic alignment is available; the test-set "
            "ones ("
            + ", ".join(sorted(_NEEDS_TEST))
            + ") if the run has Test/Train>0. Naming a figure explicitly "
            "whose data is missing is an error rather than a silent skip."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Matplotlib's PDF backend subsets every embedded font through
    # fontTools, which logs each subsetting decision at INFO ("Glyph IDs
    # [...]", "Retaining N glyphs", "head subsetting not needed", ...).
    # That noise dwarfs our own progress messages; demote to WARNING.
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")

    figs_dir = run_dir / "figs"
    figs_inputs_dir = figs_dir / "inputs"

    model = _load_model(run_dir)
    _patch_legacy_options(model)
    seed = _read_seed(run_dir)
    _seed_global_rng(seed)

    # Resolve the synthetic-alignment list before filtering so default-
    # mode rendering can decide which figures are renderable. Explicit
    # --synthetic-alignment paths are honored verbatim; auto-discovery
    # picks up everything under <run_dir>/synthetic/.
    align_paths: list[Path] = []
    if args.synthetic_alignment is not None:
        for p in args.synthetic_alignment:
            resolved = p.resolve()
            if not resolved.is_file():
                parser.error(f"synthetic alignment not found at {resolved}")
            align_paths.append(resolved)
    else:
        align_paths = _autodiscover_synthetic(run_dir)
        if align_paths:
            log.info(
                "auto-detected %d synthetic alignment(s): %s",
                len(align_paths),
                ", ".join(p.name for p in align_paths),
            )

    mpnn_scores_path = _autodiscover_mpnn_sweep(run_dir)
    if mpnn_scores_path is not None:
        log.info("auto-detected ProteinMPNN sweep: %s", mpnn_scores_path.parent.name)

    explicit_figs = args.figs is not None
    requested_initial = list(args.figs) if explicit_figs else list(ALL_FIGS)
    requested = _filter_renderable(
        model,
        requested_initial,
        have_align=bool(align_paths),
        have_mpnn_sweep=mpnn_scores_path is not None,
        strict=explicit_figs,
    )
    if not requested:
        log.warning("no figures to render")
        return 0

    needs_stats = bool(set(requested) & _NEEDS_STATS)
    needs_align = bool(set(requested) & _NEEDS_ALIGN_MOD)

    # Activate lab style. Wrapped because the user may not have the
    # lab-paper stylesheet installed; falling back to mpl defaults is
    # acceptable for routine runs.
    try:
        plt.style.use("lab-paper")
    except (OSError, ValueError):
        log.info("'lab-paper' stylesheet not available; using matplotlib defaults")

    # Build per-alignment dicts: each carries its path, sampling
    # temperature, alignment array, (lazily) computed stats, and the
    # color used for this alignment in every figure (sourced from
    # lab_plotting so the lab palette stays the single source of truth).
    artificial: list[dict] = []
    if needs_align:
        if not align_paths:
            raise RuntimeError(
                "internal error: align-needing figure survived filtering "
                "without any synthetic alignment paths"
            )
        for k, p in enumerate(align_paths):
            align_mod = _load_align_mod(p, model)
            T = _read_sampling_temperature(p)
            artificial.append(
                {
                    "path": p,
                    "temperature": T,
                    "align_mod": align_mod,
                    "stats": None,
                    "color": lab_plotting.color_for_artificial(T, k),
                }
            )
        if needs_stats:
            for item in artificial:
                item["stats"] = _ensure_stats(
                    model,
                    item["align_mod"],
                    item["path"],
                    figs_inputs_dir,
                    rng_seed=seed,
                )

    # Natural-group colors (Train/Test/Random) come from lab_plotting
    # too. Pass-through to plot_stats; Coupling_evol and Params ignore
    # them.
    natural_colors = {
        name: lab_plotting.color_for_natural(name)
        for name in ("Train", "Test", "Random")
    }

    # Sector positions for the Params figure. ``--sector`` selects
    # which CM definition to use (default: Emily's); the helper returns
    # an empty list (with a logged reason) when the run isn't a CM run
    # or the user passed ``--sector none``.
    L_run = int(np.asarray(model["h"]).shape[0])
    sector_positions = _cm_sector_positions(run_dir, L_run, args.sector)
    if sector_positions:
        log.info(
            "marking %d CM sector position(s) on params figure (--sector=%s)",
            len(sector_positions),
            args.sector,
        )

    # One canonical copy of the rendering script next to the other
    # provenance under figs/inputs/, instead of one .source.py per PDF.
    figs_inputs_dir.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve()
    shutil.copy2(script_src, figs_inputs_dir / script_src.name)

    run_id = run_dir.name
    written: list[Path] = []
    rendered_names: list[str] = []
    try:
        for name in requested:
            log.info("rendering %s", name)
            paths = _render_one(
                name,
                model,
                artificial,
                natural_colors,
                figs_dir,
                run_id=run_id,
                sector_positions=sector_positions,
                mpnn_scores_path=mpnn_scores_path,
            )
            if paths:
                rendered_names.append(name)
                written.extend(paths)
    finally:
        # Record provenance even if rendering raised partway through, so a
        # half-populated figs/ still has a sources.json identifying which
        # model + alignments produced what landed on disk.
        _write_sources_json(
            figs_inputs_dir,
            run_dir=run_dir,
            model_path=run_dir / "model.npy",
            artificial=artificial,
            figures=rendered_names,
        )

    log.info("wrote %d figure(s) under %s", len(written), figs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
