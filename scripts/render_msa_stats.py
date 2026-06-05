"""Render an MSA's per-site and pairwise summary statistics as a 2×2 figure.

Mirrors the ranked statistics that ``pruning/build_mask.py`` uses to
generate pruning masks, but without the masking step. Four panels:

* A: per-site frequency :math:`f_i^a` (L × q)
* B: per-site KL divergence :math:`D_i^a` (L × q) — Dia strategy quantity
* C: Frobenius norm of pairwise frequency :math:`\\Vert f_{ij}\\Vert_F`
  (L × L) — fij strategy quantity
* D: SCA matrix (spectral norm of conservation-weighted correlation)
  (L × L) — sca strategy quantity

Layout: rows split per-site (top) vs. pairwise (bottom); columns split
raw stat (left) vs. conservation-weighted (right). Within each column
the top and bottom panels share the sequence-position x-axis.

All numerics match ``build_mask.py`` exactly: same sequence reweighting
(``CalcWeights(alg, 1 - theta)``), same gap-corrected background for
both ``Dia`` and ``SCA``, same alphabet (``-ACDEFGHIKLMNPQRSTVWY``).

Output goes to ``data/figs/msa_stats_<msa-stem>.pdf`` by default.

This CLI consumes a pre-encoded integer MSA (``.npy``). The pipeline
produces one per run at ``<run_dir>/inputs/msa.npy`` (via the
``encode_msa`` rule); pass that, or any array encoded with
``SBM.utils.utils.load_fasta``.

Usage::

    python scripts/render_msa_stats.py --msa <run_dir>/inputs/msa.npy \\
        [--theta 0.7] [--lbda 0.03] [--Dia-prior gap-corrected] \\
        [--sector emily|rama|none] [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pysca.scaTools as sca

from SBM.utils.utils import CalcWeights

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import lab_plotting  # noqa: E402

# pruning/build_mask.py is a top-level script (not a package module).
# Load it by absolute path so we reuse calcSCAMat and the gapless
# background frequencies without depending on a fragile relative import.
_BUILD_MASK_PATH = _REPO_ROOT / "pruning" / "build_mask.py"
_bm_spec = _importlib_util.spec_from_file_location("build_mask", _BUILD_MASK_PATH)
if _bm_spec is None or _bm_spec.loader is None:
    raise ImportError(f"could not load module spec for {_BUILD_MASK_PATH}")
build_mask = _importlib_util.module_from_spec(_bm_spec)
_bm_spec.loader.exec_module(build_mask)

_CM_SECTOR_PATH = _REPO_ROOT / "CM_sector.py"
_cm_spec = _importlib_util.spec_from_file_location("CM_sector", _CM_SECTOR_PATH)
if _cm_spec is None or _cm_spec.loader is None:
    raise ImportError(f"could not load module spec for {_CM_SECTOR_PATH}")
CM_sector = _importlib_util.module_from_spec(_cm_spec)
_cm_spec.loader.exec_module(CM_sector)

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
SECTOR_CHOICES: tuple[str, ...] = ("emily", "rama", "none")
log = logging.getLogger(__name__)

_REL_FONT = {
    "xx-small": 0.579,
    "x-small": 0.694,
    "small": 0.833,
    "medium": 1.0,
    "large": 1.2,
    "x-large": 1.44,
    "xx-large": 1.728,
}


def _pt_to_in(pt: float) -> float:
    return pt / 72.0


def _resolve_font_pt(spec: float | str) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    base = float(mpl.rcParams["font.size"])
    return base * _REL_FONT.get(spec, 1.0)


def _cm_sector_positions(msa_path: Path, L: int, choice: str) -> list[int]:
    """MSA-column indices for the requested CM catalytic-sector definition.

    Returns ``[]`` (with a logged reason) when the sector strip should be
    suppressed: ``choice == "none"``, ``L != 96``, or the MSA filename
    stem does not contain ``"CM"`` (case-insensitive).
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
    if "CM" not in msa_path.stem.upper():
        log.info(
            "sector annotation suppressed: %s does not look like a CM MSA "
            "(stem must contain 'CM'). Pass --sector none to silence.",
            msa_path,
        )
        return []
    ats_set = (
        CM_sector.CM_SECTOR_EMILY if choice == "emily" else CM_sector.CM_SECTOR_RAMA
    )
    missing = [int(a) for a in ats_set if int(a) not in CM_sector.ATS_TO_POS_96]
    if missing:
        raise ValueError(
            f"CM sector definition {choice!r} references ATS values not "
            f"present in CM_sector.ATS_TO_POS_96: {missing}"
        )
    return sorted(CM_sector.ATS_TO_POS_96[int(a)] for a in ats_set)


def compute_msa_stats(
    alg: np.ndarray,
    *,
    theta: float,
    lbda: float,
    Dia_prior: str,
) -> dict[str, np.ndarray]:
    """Compute (fia, Dia, ‖fij‖_F, SCA) from an MSA.

    Mirrors ``pruning/build_mask.py:main`` so the values are identical to
    what the corresponding mask strategies see at ranking time.

    Parameters
    ----------
    alg
        Integer MSA, shape ``(N_seq, L)`` with ``0 = gap`` and 1..20 the
        amino acids in ``ALPHABET[1:]``.
    theta, lbda
        Sequence-reweighting threshold and pseudocount, matching
        ``build_mask.py``'s defaults (0.7, 0.03).
    Dia_prior
        ``"gap-corrected"`` (default) uses the alignment's gap-corrected
        background; ``"uniform"`` uses ``np.ones(21) / 21``.

    Returns
    -------
    dict
        Keys ``"fia"`` (L, q), ``"Dia"`` (L, q), ``"fij_norm"`` (L, L),
        ``"SCA"`` (L, L).
    """
    if alg.ndim != 2:
        raise ValueError(f"MSA must be 2D (N_seq, L); got shape {alg.shape}")
    if not np.issubdtype(alg.dtype, np.integer):
        raise ValueError(f"MSA must be integer-valued; got dtype {alg.dtype}")
    # Validate the alphabet range loudly. The downstream pysca calls
    # hardcode ``Naa=21`` and would silently misbin out-of-range codes,
    # so reject them up front.
    a_min, a_max = int(alg.min()), int(alg.max())
    if a_min < 0 or a_max > 20:
        raise ValueError(
            f"MSA codes must lie in [0, 20] (0=gap, 1..20 = "
            f"{ALPHABET[1:]!r}); got min={a_min}, max={a_max}"
        )
    L = alg.shape[1]

    seqw, neff = CalcWeights(alg, 1 - theta, False)
    seqwn = seqw / neff
    bg_gaps = (1 - lbda) * np.sum(seqwn * (alg == 0).sum(axis=1)) / L + lbda * (1 / 21)
    freqs0 = np.hstack([[bg_gaps], (1 - bg_gaps) * build_mask.BACKGROUND_FREQS_GAPLESS])

    f1, fij_flat, _ = sca.freq(alg + 1, seqw=seqw, Naa=21, lbda=0)
    fia = f1.reshape(L, 21)
    fij = fij_flat.reshape(L, 21, L, 21).transpose(0, 2, 1, 3)
    fij_norm = np.linalg.norm(fij, axis=(2, 3))

    Dia_freq0 = np.ones(21) / 21 if Dia_prior == "uniform" else freqs0
    _, Dia_flat, _ = sca.posWeights(alg + 1, seqw, lbda, 21, Dia_freq0)
    Dia = Dia_flat.reshape(L, 21)

    SCA = build_mask.calcSCAMat(
        alg, seqw=seqw, lbda=lbda, freq0=freqs0, norm="spec", include_gaps=True
    )
    return {"fia": fia, "Dia": Dia, "fij_norm": fij_norm, "SCA": SCA}


def render_msa_stats_figure(
    stats: dict[str, np.ndarray],
    *,
    sector_positions: list[int],
) -> plt.Figure:
    """Build the 2×2 figure from precomputed stats.

    Layout is inch-precise: cell sizes follow the data shapes, all gaps
    and margins come from rcParams font / tick metrics, no eyeballed
    numbers. Each column shares its x-axis (sequence position) between
    the per-site and pairwise rows.
    """
    fia = stats["fia"]
    Dia = stats["Dia"]
    fij_norm = stats["fij_norm"]
    SCA = stats["SCA"]
    L, q = fia.shape
    if q != len(ALPHABET):
        raise ValueError(f"expected q={len(ALPHABET)}; got q={q}")
    if Dia.shape != (L, q):
        raise ValueError(f"Dia shape {Dia.shape} disagrees with fia {fia.shape}")
    if fij_norm.shape != (L, L) or SCA.shape != (L, L):
        raise ValueError(
            f"pairwise panels must be (L, L)=({L}, {L}); got "
            f"fij_norm={fij_norm.shape}, SCA={SCA.shape}"
        )

    # Inch-budget layout (mirrors utils_plot.plot_stats's Params block).
    # The two columns are independent stacks of (q×L) above (L×L); the
    # column gutter just needs room for the right column's left-side
    # tick gutter past the left column's colorbar + label.
    panel_w = 4.5
    h_top = panel_w * q / L
    h_bot = panel_w
    breathing = _pt_to_in(4.0)

    ticklabel_pt = _resolve_font_pt(mpl.rcParams["xtick.labelsize"])
    xlabel_pt = _resolve_font_pt(mpl.rcParams["axes.labelsize"])
    ytick_pt = _resolve_font_pt(mpl.rcParams["ytick.labelsize"])
    tick_pad_pt = mpl.rcParams["xtick.major.pad"]
    tick_size_pt = mpl.rcParams["xtick.major.size"]
    ytick_size_pt = mpl.rcParams["ytick.major.size"]
    ytick_pad_pt = mpl.rcParams["ytick.major.pad"]
    labelpad_pt = mpl.rcParams.get("axes.labelpad", 4.0)
    sector_y_axes = 1.06

    hgap = (
        _pt_to_in(tick_size_pt + tick_pad_pt + ticklabel_pt + labelpad_pt + xlabel_pt)
        + breathing
    )

    sector_offset_in = (sector_y_axes - 1.0) * h_top
    sector_label_pt = _resolve_font_pt(mpl.rcParams["axes.labelsize"])
    top_pad = sector_offset_in + _pt_to_in(sector_label_pt) + breathing
    bottom_pad = _pt_to_in(8.0)

    left_pad = (
        _pt_to_in(
            xlabel_pt + labelpad_pt + ytick_size_pt + ytick_pad_pt + 2.0 * ytick_pt
        )
        + breathing
    )
    cb_gap = _pt_to_in(6.0)
    cb_w = _pt_to_in(13.0)
    cb_label_pad = (
        _pt_to_in(
            ytick_size_pt + ytick_pad_pt + 2.5 * ytick_pt + labelpad_pt + xlabel_pt
        )
        + breathing
    )
    # Right column's left side needs the same budget the left column's
    # left_pad reserves: ylabel font height (rotated 90° → its visual
    # width is axes.labelsize) + labelpad + tick mark + tick pad + ~2
    # chars of tick label, plus breathing.
    col_inner_gutter = (
        _pt_to_in(
            xlabel_pt + labelpad_pt + ytick_size_pt + ytick_pad_pt + 2.0 * ytick_pt
        )
        + breathing
    )

    panel_block = panel_w + cb_gap + cb_w
    fig_w = (
        left_pad
        + panel_block
        + cb_label_pad
        + col_inner_gutter
        + panel_block
        + cb_label_pad
    )
    fig_h = top_pad + h_top + hgap + h_bot + bottom_pad

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")

    def _rect(
        left_in: float, bottom_in: float, w_in: float, h_in: float
    ) -> list[float]:
        return [left_in / fig_w, bottom_in / fig_h, w_in / fig_w, h_in / fig_h]

    y_bot = bottom_pad
    y_top = bottom_pad + h_bot + hgap
    x_L = left_pad
    x_L_cb = x_L + panel_w + cb_gap
    x_R = x_L_cb + cb_w + cb_label_pad + col_inner_gutter
    x_R_cb = x_R + panel_w + cb_gap

    ax_fia = fig.add_axes(_rect(x_L, y_top, panel_w, h_top))
    cax_fia = fig.add_axes(_rect(x_L_cb, y_top, cb_w, h_top))
    ax_Dia = fig.add_axes(_rect(x_R, y_top, panel_w, h_top))
    cax_Dia = fig.add_axes(_rect(x_R_cb, y_top, cb_w, h_top))
    ax_fij = fig.add_axes(_rect(x_L, y_bot, panel_w, h_bot), sharex=ax_fia)
    cax_fij = fig.add_axes(_rect(x_L_cb, y_bot, cb_w, h_bot))
    ax_SCA = fig.add_axes(_rect(x_R, y_bot, panel_w, h_bot), sharex=ax_Dia)
    cax_SCA = fig.add_axes(_rect(x_R_cb, y_bot, cb_w, h_bot))

    def _draw_per_site(
        ax: plt.Axes, cax: plt.Axes, mat: np.ndarray, label: str
    ) -> None:
        im = ax.imshow(
            mat.T,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            interpolation="nearest",
            extent=(-0.5, L - 0.5, q - 0.5, -0.5),
        )
        ax.set_yticks(range(q))
        ax.set_yticklabels(list(ALPHABET))
        ax.set_ylabel("Amino acid $a$")
        # Position axis is shared with the L×L panel below; the L×L
        # panel carries the position labels at its top edge (matrix
        # convention: origin in the top-left).
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=False,
        )
        ax.grid(False)
        fig.colorbar(im, cax=cax, label=label)

    _draw_per_site(ax_fia, cax_fia, fia, r"Frequency $f_i(a)$")
    _draw_per_site(ax_Dia, cax_Dia, Dia, r"KL divergence $D_i(a)$ (nats)")

    def _draw_pairwise(
        ax: plt.Axes, cax: plt.Axes, mat: np.ndarray, label: str
    ) -> None:
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            interpolation="nearest",
            extent=(-0.5, L - 0.5, L - 0.5, -0.5),
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Sequence position $i$")
        ax.set_ylabel("Sequence position $j$")
        ax.grid(False)
        fig.colorbar(im, cax=cax, label=label)

    _draw_pairwise(ax_fij, cax_fij, fij_norm, r"$\Vert f_{ij}\Vert_F$")
    _draw_pairwise(ax_SCA, cax_SCA, SCA, r"SCA $\tilde{C}_{ij}$ (spectral)")

    if sector_positions:
        for ax in (ax_fia, ax_Dia):
            ax.scatter(
                sector_positions,
                [sector_y_axes] * len(sector_positions),
                transform=ax.get_xaxis_transform(),
                s=12,
                color="black",
                marker="o",
                clip_on=False,
                zorder=3,
            )
        ax_fia.text(
            -0.005,
            sector_y_axes,
            "Sector",
            transform=ax_fia.transAxes,
            ha="right",
            va="center",
        )

    for ax, letter in (
        (ax_fia, "A"),
        (ax_Dia, "B"),
        (ax_fij, "C"),
        (ax_SCA, "D"),
    ):
        lab_plotting.panel_label(ax, letter)

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a 2×2 MSA-statistics figure (fia, Dia, ‖fij‖_F, SCA)."
    )
    parser.add_argument(
        "--msa",
        type=Path,
        required=True,
        help="path to an MSA .npy file (shape (N_seq, L), 0=gap)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.7,
        help="sequence-reweighting similarity threshold (default: 0.7)",
    )
    parser.add_argument(
        "--lbda",
        type=float,
        default=0.03,
        help="pseudocount for SCA / KL divergence (default: 0.03)",
    )
    parser.add_argument(
        "--Dia-prior",
        dest="Dia_prior",
        choices=["gap-corrected", "uniform"],
        default="gap-corrected",
        help=(
            "background distribution for the D_i^a panel: 'gap-corrected' "
            "(default) uses the alignment's gap rate plus standard 20-AA "
            "frequencies; 'uniform' uses np.ones(21)/21."
        ),
    )
    parser.add_argument(
        "--sector",
        choices=SECTOR_CHOICES,
        default="emily",
        help=(
            "CM catalytic-sector definition for the strip above per-site "
            "panels: 'emily' (default), 'rama', or 'none'. Auto-suppressed "
            "if L != 96 or the MSA stem doesn't contain 'CM'."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output PDF path. Default: data/figs/msa_stats_<msa-stem>.pdf "
            "(resolved relative to the repo root)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    msa_path = args.msa.resolve()
    if not msa_path.is_file():
        parser.error(f"MSA not found at {msa_path}")
    if msa_path.suffix != ".npy":
        parser.error(f"--msa must be a .npy file (got {msa_path.suffix})")

    out_path = args.out
    if out_path is None:
        out_path = _REPO_ROOT / "data" / "figs" / f"msa_stats_{msa_path.stem}.pdf"
    out_path = Path(out_path).resolve()

    log.info("loading MSA from %s", msa_path)
    alg = np.load(msa_path)
    log.info("MSA shape: %s, dtype: %s", alg.shape, alg.dtype)

    log.info(
        "computing statistics (theta=%.2f, lbda=%.3f, Dia_prior=%s)",
        args.theta,
        args.lbda,
        args.Dia_prior,
    )
    stats = compute_msa_stats(
        alg, theta=args.theta, lbda=args.lbda, Dia_prior=args.Dia_prior
    )

    sector_positions = _cm_sector_positions(msa_path, alg.shape[1], args.sector)
    if sector_positions:
        log.info(
            "marking %d CM sector position(s) (--sector=%s)",
            len(sector_positions),
            args.sector,
        )

    # The inch-budget layout reads xtick.labelsize / axes.labelsize / etc.
    # from rcParams, so a missing 'lab-paper' stylesheet would silently
    # drift the layout. Log at WARNING (not INFO) per global rules:
    # failures must be loud, and project policy mandates this stylesheet
    # for every saved figure.
    try:
        plt.style.use("lab-paper")
    except (OSError, ValueError):
        log.warning(
            "'lab-paper' stylesheet not available; falling back to matplotlib "
            "defaults — figure layout will not match other lab figures"
        )

    fig = render_msa_stats_figure(stats, sector_positions=sector_positions)

    # Rasterize heatmap images in-place so the PDF stays compact
    # (vector-rendered q×L and L×L heatmaps would blow up file size).
    for ax in fig.axes:
        for artist in ax.images + ax.collections:
            artist.set_rasterized(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lab_plotting.save_figure(fig, out_path, script_path=Path(__file__))
    plt.close(fig)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
