"""Figures for the two-model joint-annealing design run (``SBM.design.anneal``).

Two figures, both routed through ``scripts/lab_plotting.py`` (lab palette + PDF
provenance), both laid out from inch budgets computed off ``rcParams`` font
metrics (no hand-picked ``hspace``/``top``/``bottom``):

* ``design_trajectories.pdf`` — ``E_tot`` per step for every chain, with the
  shared annealing temperature ``T = 1/beta`` on a thin strip above.
* ``design_phase_space.pdf`` — the trajectories drawn in the ``E_A``–``E_B`` plane
  as arrowed paths (over the natural clouds, with the Pareto front of the final
  designs marked), beside a heatmap of where the cold-phase states land (basins).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

# scripts/lab_plotting.py is a sibling script, not a package module (same trick as
# utils_energy_plot.py). Reuse that module's wide-table + group-color helpers too.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

from SBM.utils.utils_energy_plot import _wide_table  # noqa: E402

log = logging.getLogger(__name__)

_BREATHING = 4.0 / 72.0  # 4 pt of slack, in inches


def _pt_to_in(pt: float) -> float:
    return float(pt) / 72.0


def _font_pt(rc_key: str) -> float:
    """Resolve an rcParams font size (may be a string like 'medium') to points."""
    return FontProperties(size=mpl.rcParams[rc_key]).get_size_in_points()


def _left_gutter(n_tick_chars: int = 6) -> float:
    """Inches reserved on the left for a y label + tick marks + ``n_tick_chars`` of tick text."""
    ylab, ytick = _font_pt("axes.labelsize"), _font_pt("ytick.labelsize")
    return _pt_to_in(
        ylab + mpl.rcParams["axes.labelpad"] + mpl.rcParams["ytick.major.size"]
        + mpl.rcParams["ytick.major.pad"] + 0.6 * n_tick_chars * ytick
    ) + _BREATHING


def _bottom_gutter() -> float:
    """Inches reserved below for an x label + tick marks + one line of tick text."""
    xlab, xtick = _font_pt("axes.labelsize"), _font_pt("xtick.labelsize")
    return _pt_to_in(
        xlab + mpl.rcParams["axes.labelpad"] + mpl.rcParams["xtick.major.size"]
        + mpl.rcParams["xtick.major.pad"] + xtick
    ) + _BREATHING


def _title_gutter() -> float:
    return _pt_to_in(_font_pt("axes.titlesize")) + 2 * _BREATHING


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def load_trajectories(npz_path: Path) -> dict[str, np.ndarray]:
    """Load the CLI's ``trajectories.npz`` into a plain dict of arrays."""
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def _natives_by_family(scores_tsv: Path, model_names: tuple[str, str]) -> pd.DataFrame | None:
    """Wide (E_A, E_B) table of the *natural* sequences, tagged by home family.

    Returns ``None`` if the scores file is unavailable so the overlay is simply
    skipped (the trajectories still render).
    """
    try:
        wide = _wide_table(Path(scores_tsv), model_names)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("no natives overlay (%s)", exc)
        return None
    nat = wide[wide["group"].astype(str).str.endswith("/natural")].copy()
    return nat if len(nat) else None


def _pareto_min_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of points not dominated in *both* coordinates (both minimized).

    A point is on the front if no other point has ``x' <= x`` and ``y' <= y`` with at
    least one strict — i.e. you cannot lower one model's energy without raising the
    other's. This is the jointly-satisfiable frontier.
    """
    n = x.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if dominated.any():
            keep[i] = False
    return keep


# Start-type → hue is *coherent with the family clouds*: a model-X natural start shares
# model X's hue (so it reads as "seeded from X"); random starts are black. The family
# cloud colors below use the same two hues, distinguished from the trajectories only by
# weight/alpha (faint scatter vs solid line). All colorblind-safe (Wong 2011).
_ORDER = ("random", "natural_A", "natural_B")


def _start_style(name_A: str, name_B: str) -> dict[str, tuple[str, str]]:
    """``start_type -> (color, legend label)``."""
    return {
        "random": (lab_plotting.WONG_PALETTE[0], "random start"),
        "natural_A": (lab_plotting.WONG_PALETTE[1], f"{name_A} natural start"),
        "natural_B": (lab_plotting.WONG_PALETTE[2], f"{name_B} natural start"),
    }


def _family_color(name_A: str, name_B: str) -> dict[str, str]:
    """Native-cloud color per family, matching the natural-start hues."""
    return {name_A: lab_plotting.WONG_PALETTE[1], name_B: lab_plotting.WONG_PALETTE[2]}


def _start_types(traj: dict[str, np.ndarray], n_chains: int) -> np.ndarray:
    """Per-chain start types, defaulting to 'random' for pre-start-mix trajectories.npz."""
    if "start_type" in traj:
        return traj["start_type"].astype(str)
    return np.array(["random"] * n_chains)


# --------------------------------------------------------------------------- #
# Figure 1: E_tot per step (+ temperature strip)
# --------------------------------------------------------------------------- #

def render_trajectories(
    traj: dict[str, np.ndarray], model_names: tuple[str, str], out_pdf: Path
) -> Path:
    """``E_tot`` vs step for all chains, with a shared temperature strip above."""
    plt.style.use("lab-paper")
    steps = traj["steps"]
    temps = traj["temperatures"]
    e_tot = traj["E_tot"]                     # (n_chains, R)
    n_chains = e_tot.shape[0]
    start_type = _start_types(traj, n_chains)
    styles = _start_style(*model_names)
    best = int(np.argmin(traj["final_E_tot_mc"]))

    panel_w, h_main, h_temp = 5.0, 3.2, 0.85
    vgap = 1.5 * _BREATHING
    left, bottom, top = _left_gutter(7), _bottom_gutter(), _title_gutter()
    right = 2 * _BREATHING
    fig_w = left + panel_w + right
    fig_h = bottom + h_main + vgap + h_temp + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    ax_main = fig.add_axes(_rect(left, bottom, panel_w, h_main))
    ax_temp = fig.add_axes(_rect(left, bottom + h_main + vgap, panel_w, h_temp), sharex=ax_main)

    for c in range(n_chains):
        color = styles.get(str(start_type[c]), (lab_plotting.WONG_PALETTE[0], ""))[0]
        ax_main.plot(steps, e_tot[c], lw=0.4, alpha=0.30, color=color, zorder=1)
    ax_main.plot(steps, e_tot[best], lw=1.3, color=lab_plotting.LAB_COLORS["reference"],
                 zorder=3)
    ax_main.set_xlabel("Monte-Carlo step")
    ax_main.set_ylabel("$E_{tot}$ (a.u.)")

    # Legend via full-opacity proxies (the plotted lines are faint); only start types present.
    present = [st for st in _ORDER if np.any(start_type == st)]
    handles = [Line2D([0], [0], color=styles[st][0], lw=1.5, label=styles[st][1]) for st in present]
    handles.append(Line2D([0], [0], color=lab_plotting.LAB_COLORS["reference"], lw=1.5,
                          label=f"best (chain {best}, {styles.get(str(start_type[best]), ('', '?'))[1]})"))
    ax_main.legend(handles=handles, fontsize="xx-small", frameon=False, loc="best")

    ax_temp.plot(steps, temps, lw=1.2, color=lab_plotting.LAB_COLORS["chrome"])
    ax_temp.set_ylabel("$T$")
    ax_temp.tick_params(labelbottom=False)
    ax_temp.set_ylim(0, float(temps.max()) * 1.05)
    lab_plotting.panel_label(ax_temp, "A")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=design_trajectories"})
    plt.close(fig)
    log.info("wrote design trajectories figure: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Figure 2: E_A-E_B phase space (arrowed paths + natives + Pareto) | landing heatmap
# --------------------------------------------------------------------------- #

def render_phase_space(
    traj: dict[str, np.ndarray],
    model_names: tuple[str, str],
    weights: tuple[float, float],
    out_pdf: Path,
    *,
    scores_tsv: Path | None = None,
    cold_fraction: float = 0.25,
) -> Path:
    """Two panels: arrowed ``E_A``–``E_B`` trajectories (A) and a landing heatmap (B)."""
    plt.style.use("lab-paper")
    name_A, name_B = model_names
    w_A, w_B = weights
    E_A, E_B = traj["E_A"], traj["E_B"]           # (n_chains, R)
    fa, fb = traj["final_E_A_mc"], traj["final_E_B_mc"]
    natives = _natives_by_family(scores_tsv, model_names) if scores_tsv else None
    start_type = _start_types(traj, E_A.shape[0])
    styles = _start_style(name_A, name_B)
    fam_colors = _family_color(name_A, name_B)

    # cold-phase pool (last fraction of records) = "where trajectories land".
    r = E_A.shape[1]
    c0 = max(0, int(r * (1.0 - cold_fraction)))
    cold_A, cold_B = E_A[:, c0:].ravel(), E_B[:, c0:].ravel()

    panel = 3.2
    left, bottom, top = _left_gutter(6), _bottom_gutter(), _title_gutter()
    mid_gap = _left_gutter(6)                      # right panel needs its own left gutter
    cb_gap, cb_w = _pt_to_in(6.0), _pt_to_in(13.0)
    cb_label = _pt_to_in(_font_pt("axes.labelsize") + mpl.rcParams["axes.labelpad"]
                         + mpl.rcParams["ytick.major.size"] + mpl.rcParams["ytick.major.pad"]
                         + 0.6 * 5 * _font_pt("ytick.labelsize")) + _BREATHING
    fig_w = left + panel + mid_gap + panel + cb_gap + cb_w + cb_label
    fig_h = bottom + panel + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    x_a = left
    x_b = left + panel + mid_gap
    x_cb = x_b + panel + cb_gap
    ax_a = fig.add_axes(_rect(x_a, bottom, panel, panel))
    ax_b = fig.add_axes(_rect(x_b, bottom, panel, panel))
    cax = fig.add_axes(_rect(x_cb, bottom, cb_w, panel))

    # ---- Panel A: paths + natives + Pareto front -------------------------- #
    # Native clouds (faint), colored by family with the same hues as the natural-start
    # trajectories so a chain reads as "seeded from this cloud".
    if natives is not None:
        for fam in natives["origin_model"].unique():
            sub = natives[natives["origin_model"] == fam]
            ax_a.scatter(sub[name_A], sub[name_B], s=5, alpha=0.12,
                         color=fam_colors.get(fam, lab_plotting.LAB_COLORS["chrome"]),
                         linewidths=0, zorder=0)
    # Trajectory paths + final-step arrow, colored by start type.
    has_arrow = E_A.shape[1] >= 2                 # need two records to draw a direction
    for c in range(E_A.shape[0]):
        color = styles.get(str(start_type[c]), (lab_plotting.WONG_PALETTE[0], ""))[0]
        ax_a.plot(E_A[c], E_B[c], lw=0.4, alpha=0.25, color=color, zorder=1)
        if has_arrow:
            ax_a.annotate("", xy=(E_A[c, -1], E_B[c, -1]), xytext=(E_A[c, -2], E_B[c, -2]),
                          arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.5, color=color), zorder=1)
    # Final designs, colored by start type.
    for st in _ORDER:
        m = start_type == st
        if m.any():
            ax_a.scatter(fa[m], fb[m], s=12, color=styles[st][0], edgecolors="none", zorder=3)
    front = _pareto_min_mask(fa, fb)
    order = np.argsort(fa[front])
    pareto_color = lab_plotting.LAB_COLORS["reference"]   # distinct from the Wong start hues
    ax_a.plot(fa[front][order], fb[front][order], drawstyle="steps-post", lw=1.0,
              color=pareto_color, zorder=4)
    ax_a.scatter(fa[front], fb[front], s=30, facecolors="none",
                 edgecolors=pareto_color, linewidths=1.2, zorder=5)

    lo = float(min(E_A.min(), E_B.min()))
    hi = float(max(E_A.max(), E_B.max()))
    ax_a.plot([lo, hi], [lo, hi], color=lab_plotting.LAB_COLORS["chrome"], lw=0.8,
              ls="--", zorder=0)
    # one E_tot iso-line through the best (lowest-E_tot) final design, showing the
    # descent direction: w_A*E_A + w_B*E_B = const  =>  E_B = (const - w_A*E_A)/w_B.
    best = int(np.argmin(w_A * fa + w_B * fb))
    const = w_A * fa[best] + w_B * fb[best]
    xs = np.array([lo, hi])
    ax_a.plot(xs, (const - w_A * xs) / w_B, color=lab_plotting.LAB_COLORS["chrome"],
              lw=0.7, ls=":", zorder=2)
    ax_a.set_xlabel(f"E under {name_A} model (a.u.)")
    ax_a.set_ylabel(f"E under {name_B} model (a.u.)")

    # Legend via full-opacity proxies (native clouds, start types present, Pareto, iso-line).
    handles = []
    if natives is not None:
        for fam in (name_A, name_B):
            if (natives["origin_model"] == fam).any():
                handles.append(Line2D([0], [0], marker="o", ls="none", ms=4, alpha=0.6,
                                      color=fam_colors[fam], label=f"{fam} natives"))
    handles += [Line2D([0], [0], color=styles[st][0], lw=1.5, label=styles[st][1])
                for st in _ORDER if np.any(start_type == st)]
    handles.append(Line2D([0], [0], color=pareto_color, lw=1.2, label="Pareto front"))
    handles.append(Line2D([0], [0], color=lab_plotting.LAB_COLORS["chrome"], lw=0.7, ls=":",
                          label="$E_{tot}$ iso-line (best)"))
    ax_a.legend(handles=handles, fontsize="xx-small", frameon=False, loc="best")
    ax_a.annotate(f"$w_A$={w_A:.3f}, $w_B$={w_B:.3f}", xy=(0.02, 0.98),
                  xycoords="axes fraction", va="top", ha="left", fontsize="xx-small")
    lab_plotting.panel_label(ax_a, "A")

    # ---- Panel B: landing heatmap ----------------------------------------- #
    rng = [[lo, hi], [lo, hi]]
    hh = ax_b.hist2d(cold_A, cold_B, bins=45, range=rng, cmap="viridis")
    ax_b.plot([lo, hi], [lo, hi], color="white", lw=0.6, ls="--", zorder=1)
    ax_b.set_xlabel(f"E under {name_A} model (a.u.)")
    ax_b.set_ylabel(f"E under {name_B} model (a.u.)")
    ax_b.set_title(f"cold-phase density (last {int(cold_fraction * 100)}%)", fontsize="x-small")
    fig.colorbar(hh[3], cax=cax, label="visited state count")
    lab_plotting.panel_label(ax_b, "B")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=design_phase_space"})
    plt.close(fig)
    log.info("wrote design phase-space figure: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Figure 3: final-length histogram (by start type)
# --------------------------------------------------------------------------- #

def render_lengths(
    traj: dict[str, np.ndarray], model_names: tuple[str, str], out_pdf: Path
) -> Path:
    """Histogram of the final design length ``N`` (residues), stacked by start type."""
    plt.style.use("lab-paper")
    N = (traj["final_n_residues"] if "final_n_residues" in traj
         else traj["n_residues"][:, -1]).astype(int)
    start_type = _start_types(traj, N.size)
    styles = _start_style(*model_names)
    lo, hi = int(N.min()), int(N.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    panel_w, panel_h = 4.4, 2.8
    left, bottom, top = _left_gutter(5), _bottom_gutter(), _title_gutter()
    right = 2 * _BREATHING
    fig_w, fig_h = left + panel_w + right, bottom + panel_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([left / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h])

    present = [st for st in _ORDER if np.any(start_type == st)]
    ax.hist([N[start_type == st] for st in present], bins=bins, stacked=True,
            color=[styles[st][0] for st in present],
            label=[styles[st][1] for st in present],
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("final design length $N$ (residues)")
    ax.set_ylabel("count")
    ax.legend(fontsize="xx-small", frameon=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=design_lengths"})
    plt.close(fig)
    log.info("wrote design length histogram: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Figure 4: ZAPPO-colored alignment of the designs in each model frame
# --------------------------------------------------------------------------- #

# ZAPPO colour scheme (physico-chemical property; Jalview standard). Domain convention
# the user asked for — not a lab-palette semantic color, so the exact ZAPPO hexes are used.
_ZAPPO = {
    **dict.fromkeys("ILVAM", "#FFAFAF"),   # aliphatic / hydrophobic — rose
    **dict.fromkeys("FWY", "#FFC800"),     # aromatic — orange
    **dict.fromkeys("KRH", "#6464FF"),     # positive — blue
    **dict.fromkeys("DE", "#FF0000"),      # negative — red
    **dict.fromkeys("STNQ", "#00FF00"),    # hydrophilic — green
    **dict.fromkeys("PG", "#FF00FF"),      # conformationally special — magenta
    "C": "#FFFF00",                        # cysteine — yellow
    "-": "#FFFFFF",                        # gap — white
}
_ZAPPO_LEGEND = [
    ("ILVAM", "aliphatic"), ("FWY", "aromatic"), ("KRH", "positive"), ("DE", "negative"),
    ("STNQ", "hydrophilic"), ("PG", "Pro/Gly"), ("C", "Cys"),
]
_ZAPPO_RGB = {ch: mpl.colors.to_rgb(hexc) for ch, hexc in _ZAPPO.items()}


def _read_aln_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Return (headers, sequences) from an aligned FASTA (equal-length gapped strings)."""
    headers, seqs, cur = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            headers.append(line[1:])
        elif line.strip():
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return headers, seqs


def _hdr_field(header: str, key: str, default: str = "") -> str:
    for tok in header.split():
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    return default


def _zappo_rgb_image(seqs: list[str]) -> np.ndarray:
    """(n_seq, L, 3) RGB image of an alignment under the ZAPPO scheme."""
    n, length = len(seqs), len(seqs[0])
    img = np.full((n, length, 3), 0.85)          # unknown residue -> light grey
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            img[i, j] = _ZAPPO_RGB.get(ch, (0.85, 0.85, 0.85))
    return img


def render_alignment(
    fasta_A: Path, fasta_B: Path, model_names: tuple[str, str], out_pdf: Path,
    *, cell_in: float = 0.085, show_letters: bool = True,
) -> Path:
    """ZAPPO-colored alignment of the designs in both model frames (side-by-side panels).

    Rows are chains (in run order, which is blocked by start type); a thin left strip is
    colored by start type. Letters are drawn on the colored cells (as in a Jalview/alnviz
    ZAPPO view); the aligned FASTAs (``design_aln_{A,B}.fasta``) are the interactive-viewer
    input. For detailed inspection open those in an alignment viewer with ZAPPO coloring."""
    plt.style.use("lab-paper")
    name_A, name_B = model_names
    hdr_A, seqs_A = _read_aln_fasta(fasta_A)
    _, seqs_B = _read_aln_fasta(fasta_B)
    start_type = np.array([_hdr_field(h, "start", "random") for h in hdr_A])
    styles = _start_style(name_A, name_B)
    n_seq = len(seqs_A)
    L_A, L_B = len(seqs_A[0]), len(seqs_B[0])

    strip_w = 0.16
    panel_gap = 0.5
    left = _left_gutter(5)                     # room for y (chain) ticks
    top = _title_gutter() + _pt_to_in(_font_pt("xtick.labelsize")) + 3 * _BREATHING  # ruler
    bottom = _bottom_gutter() + 5 * _BREATHING  # ZAPPO legend strip below
    right = 2 * _BREATHING
    wA_in, wB_in = L_A * cell_in, L_B * cell_in
    rows_in = n_seq * cell_in
    fig_w = left + strip_w + wA_in + panel_gap + wB_in + right
    fig_h = bottom + rows_in + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    ax_strip = fig.add_axes(_rect(left, bottom, strip_w, rows_in))
    ax_A = fig.add_axes(_rect(left + strip_w, bottom, wA_in, rows_in))
    ax_B = fig.add_axes(_rect(left + strip_w + wA_in + panel_gap, bottom, wB_in, rows_in))

    # start-type strip (rows top-to-bottom = chain 0..n-1)
    strip = np.array([mpl.colors.to_rgb(styles.get(str(st), ("#000000", ""))[0])
                      for st in start_type])[:, None, :]
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest",
                    extent=(0, 1, n_seq, 0))
    ax_strip.set_xticks([])
    ax_strip.set_ylabel("design (chain)")
    ax_strip.set_yticks(np.arange(0, n_seq, max(1, n_seq // 8)) + 0.5)
    ax_strip.set_yticklabels(np.arange(0, n_seq, max(1, n_seq // 8)))

    for ax, seqs, length, name in ((ax_A, seqs_A, L_A, name_A), (ax_B, seqs_B, L_B, name_B)):
        ax.imshow(_zappo_rgb_image(seqs), aspect="auto", interpolation="nearest",
                  extent=(0, length, n_seq, 0))
        ax.set_yticks([])
        step = 10
        ax.set_xticks(np.arange(0, length + 1, step))
        ax.tick_params(labelbottom=False, labeltop=True, top=True, labelsize="xx-small")
        ax.set_title(f"{name} frame (L={length})", fontsize="x-small", pad=14)
        if show_letters:
            fs = max(2.0, cell_in * 72 * 0.75)
            for i, s in enumerate(seqs):
                for j, ch in enumerate(s):
                    if ch != "-":
                        ax.text(j + 0.5, i + 0.5, ch, ha="center", va="center",
                                fontsize=fs, color="black")

    # ZAPPO legend + start-type legend below the panels
    handles = [Line2D([0], [0], marker="s", ls="none", ms=5, color=_ZAPPO[grp[0]],
                      markeredgecolor="0.5", markeredgewidth=0.3, label=f"{grp} ({lab})")
               for grp, lab in _ZAPPO_LEGEND]
    handles += [Line2D([0], [0], marker="s", ls="none", ms=5, color=styles[st][0],
                       label=styles[st][1]) for st in _ORDER if np.any(start_type == st)]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize="xx-small", frameon=False, bbox_to_anchor=(0.5, 0.0))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=design_alignment"})
    plt.close(fig)
    log.info("wrote design alignment (ZAPPO) figure: %s", saved)
    return Path(saved)
