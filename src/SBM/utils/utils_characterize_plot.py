"""Figures + stats for the design characterization (fold + BLAST).

The characterization *compute* (ESMFold, TM-align, BLAST, merge) runs on Midway
and lands the tidy tables ``characterize/data/summary.tsv`` (designs) and
``characterize/data/natural_summary.tsv`` (controls). These figures are made on
the Mac from those tables (pure numpy/matplotlib — no external binaries), routed
through ``scripts/lab_plotting.py`` (lab palette + PDF provenance) and laid out
from inch budgets computed off ``rcParams`` font metrics (no hand-picked
``hspace``/``top``/``bottom``).

Three figures + one stats table:

* ``characterization_overview.pdf`` — consolidated 2x2: (A) TM-to-fold-A vs
  TM-to-fold-B scatter, (B) mean-pLDDT distribution by group, (C) energy vs
  structure preference (designs), (D) BLAST best-hit identity (designs).
* ``tm_A_vs_B.pdf`` — the standalone headline "which fold?" scatter.
* ``fold_call_breakdown.pdf`` — fold-call composition (fraction) per group.
* ``characterization_stats.tsv`` — tidy (group, metric, value) summary; the
  numbers annotated on the figures come from here, so they are quotable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# scripts/lab_plotting.py is a sibling script, not a package module (same trick as
# utils_design_plot.py / utils_energy_plot.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

# Reuse the repo's inch-budget layout helpers rather than adding a third copy of
# the same rcParams math (utils_plot.py and utils_design_plot.py already carry it).
from SBM.utils.utils_design_plot import (  # noqa: E402
    _BREATHING, _bottom_gutter, _left_gutter, _title_gutter,
)
from SBM.characterize import summary  # noqa: E402

log = logging.getLogger(__name__)

# Group hues, from lab_plotting constants only (no hex literals). Coherent with the
# design figures' family hues: CM = orange, PPIC = sky blue. Designs are the primary
# subject, drawn on top in black.
_GROUP_ORDER = ("design", "CM-natural", "PPIC-natural")
_GROUP_COLOR = {
    "design": lab_plotting.LAB_COLORS["data"],        # black
    "CM-natural": lab_plotting.WONG_PALETTE[1],       # orange
    "PPIC-natural": lab_plotting.WONG_PALETTE[2],     # sky blue
}
_GROUP_LABEL = {"design": "designs", "CM-natural": "CM naturals",
                "PPIC-natural": "PPIC naturals"}

# fold_call segment colors: A / B tie to the family hues; the rest are neutral.
_FOLD_ORDER = ("A", "B", "ambiguous", "neither", "na")
_FOLD_COLOR = {
    "A": lab_plotting.WONG_PALETTE[1],                # CM orange
    "B": lab_plotting.WONG_PALETTE[2],                # PPIC sky blue
    "ambiguous": lab_plotting.LAB_COLORS["chrome"],   # grey 0.4
    "neither": "0.80",
    "na": "0.93",
}
_FOLD_LABEL = {"A": "fold A (CM)", "B": "fold B (PPIC)", "ambiguous": "ambiguous",
               "neither": "neither", "na": "n/a"}


# --------------------------------------------------------------------------- #
# Small data helpers (rows are lists of str->str dicts from summary.read_tsv)
# --------------------------------------------------------------------------- #

def _col(rows: list[dict[str, str]], key: str) -> np.ndarray:
    """Column ``key`` as float; missing/empty -> NaN."""
    out = [float(v) if (v := r.get(key, "")) not in ("", None) else np.nan for r in rows]
    return np.asarray(out, dtype=float)


def _by_group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r.get("group", ""), []).append(r)
    return groups


def _present_groups(groups: dict[str, list]) -> list[str]:
    """Known groups in canonical order, then any extras in encounter order."""
    return ([g for g in _GROUP_ORDER if g in groups]
            + [g for g in groups if g not in _GROUP_ORDER])


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get(key, "")] = out.get(r.get(key, ""), 0) + 1
    return out


def _median(values) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _spearman(x, y) -> tuple[float, float] | None:
    """Spearman rho + p over pairwise-finite (x, y); None if < 3 pairs or scipy absent."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    try:
        from scipy import stats  # scipy is optional; fit_line uses it too
    except ImportError:
        log.warning("scipy missing; Spearman skipped")
        return None
    res = stats.spearmanr(x[m], y[m])
    return float(res.statistic), float(res.pvalue)


def _group_proxies(groups: dict[str, list], *, with_counts_key: str | None = None) -> list[Line2D]:
    """Legend handles for the groups present, in canonical order."""
    handles = []
    for g in _present_groups(groups):
        label = _GROUP_LABEL.get(g, g)
        if with_counts_key is not None:
            n = int(np.isfinite(_col(groups[g], with_counts_key)).sum())
            label += f" (n={n})"
        handles.append(Line2D([0], [0], marker="o", ls="none", ms=5,
                              color=_GROUP_COLOR.get(g, lab_plotting.LAB_COLORS["chrome"]),
                              label=label))
    return handles


# --------------------------------------------------------------------------- #
# Panels (each draws into a supplied Axes so the overview and the standalones
# share exactly one implementation per panel).
# --------------------------------------------------------------------------- #

def _panel_tm(ax, groups: dict[str, list], *, legend: bool = True) -> None:
    # Naturals first (background, rasterized — can be 10^4+ points), designs on top.
    for g in reversed(_present_groups(groups)):
        rows = groups[g]
        tm_a, tm_b = _col(rows, "tm_A"), _col(rows, "tm_B")
        is_design = g == "design"
        ax.scatter(tm_a, tm_b, s=26 if is_design else 6,
                   color=_GROUP_COLOR.get(g, lab_plotting.LAB_COLORS["chrome"]),
                   alpha=0.9 if is_design else 0.25,
                   edgecolor="black" if is_design else "none",
                   linewidth=0.3 if is_design else 0,
                   rasterized=not is_design,
                   zorder=3 if is_design else 1)
    ax.plot([0, 1], [0, 1], color=lab_plotting.LAB_COLORS["chrome"], ls="--", lw=0.8, zorder=0)
    ax.axvline(summary.TM_FOLD_THRESHOLD, color="0.85", lw=0.7, zorder=0)
    ax.axhline(summary.TM_FOLD_THRESHOLD, color="0.85", lw=0.7, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("TM-score to fold A (CM, 1ECM)")
    ax.set_ylabel("TM-score to fold B (PPIC, 1JNT)")
    if legend:
        ax.legend(handles=_group_proxies(groups, with_counts_key="tm_A"),
                  fontsize="xx-small", frameon=False, loc="upper right")


def _panel_plddt(ax, groups: dict[str, list], *, legend: bool = True) -> None:
    bins = np.linspace(0, 100, 26)
    for g in _present_groups(groups):
        vals = _col(groups[g], "plddt_mean")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.6,
                color=_GROUP_COLOR.get(g, "0.4"))
    ax.axvline(summary.PLDDT_HIGH, color=lab_plotting.LAB_COLORS["chrome"], ls="--", lw=1.0)
    ax.set_xlabel("mean pLDDT (0–100)")
    ax.set_ylabel("density (per group)")
    if legend:
        ax.legend(handles=_group_proxies(groups, with_counts_key="plddt_mean"),
                  fontsize="xx-small", frameon=False)


def _panel_energy_structure(ax, design_rows: list[dict[str, str]]) -> None:
    d_e, d_tm = _col(design_rows, "delta_E"), _col(design_rows, "delta_tm")
    m = np.isfinite(d_e) & np.isfinite(d_tm)
    ax.scatter(d_e[m], d_tm[m], s=22, color=_GROUP_COLOR["design"],
               edgecolor="black", linewidth=0.3, zorder=3)
    if int(m.sum()) >= 3:
        try:
            lab_plotting.fit_line(ax, d_e[m], d_tm[m], confidence=0.95)
        except ValueError as exc:  # degenerate x, etc.
            log.warning("fit_line skipped: %s", exc)
    ax.axhline(0, color="0.85", lw=0.7)
    ax.axvline(0, color="0.85", lw=0.7)
    ax.set_xlabel(r"energy preference  $E_A - E_B$  (a.u.)")
    ax.set_ylabel(r"structure preference  $TM_A - TM_B$")
    rho_p = _spearman(d_e, d_tm)
    if rho_p is not None:
        ax.annotate(f"Spearman $\\rho$={rho_p[0]:+.3f}\n(p={rho_p[1]:.2g}, n={int(m.sum())})",
                    xy=(0.03, 0.03), xycoords="axes fraction", va="bottom", ha="left",
                    fontsize="xx-small")


def _panel_blast(ax, design_rows: list[dict[str, str]]) -> None:
    specs = [("SwissProt", "swissprot_pident", lab_plotting.WONG_PALETTE[7]),
             ("CM family", "cmfam_pident", _GROUP_COLOR["CM-natural"]),
             ("PPIC family", "ppicfam_pident", _GROUP_COLOR["PPIC-natural"])]
    have = design_rows[0] if design_rows else {}
    bins = np.linspace(0, 100, 21)
    any_data = False
    for label, key, color in specs:
        if key not in have:
            continue
        vals = _col(design_rows, key)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        any_data = True
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.6, color=color,
                label=f"{label} (n={vals.size})")
    ax.set_xlabel("best-hit identity (%)")
    ax.set_ylabel("number of designs")
    if any_data:
        ax.legend(fontsize="xx-small", frameon=False)
    else:
        ax.annotate("no BLAST columns", xy=(0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize="x-small", color="0.5")


# --------------------------------------------------------------------------- #
# Figure 1: consolidated 2x2 overview
# --------------------------------------------------------------------------- #

def render_overview(design_rows: list[dict[str, str]],
                    natural_rows: list[dict[str, str]], out_pdf: Path) -> Path:
    """Consolidated 2x2 characterization overview."""
    plt.style.use("lab-paper")
    all_rows = design_rows + natural_rows
    groups = _by_group(all_rows)

    panel = 2.8
    left = _left_gutter(4)
    col_gap = _left_gutter(4)          # right column needs its own y-label gutter
    right = 2 * _BREATHING
    bottom = _bottom_gutter()
    row_gap = _bottom_gutter() + _title_gutter()   # top row's x-label + bottom row's title
    top = _title_gutter()
    fig_w = left + panel + col_gap + panel + right
    fig_h = bottom + panel + row_gap + panel + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")

    def _rect(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    x_left, x_right = left, left + panel + col_gap
    y_top, y_bot = bottom + panel + row_gap, bottom
    ax_a = fig.add_axes(_rect(x_left, y_top, panel, panel))
    ax_b = fig.add_axes(_rect(x_right, y_top, panel, panel))
    ax_c = fig.add_axes(_rect(x_left, y_bot, panel, panel))
    ax_d = fig.add_axes(_rect(x_right, y_bot, panel, panel))

    _panel_tm(ax_a, groups)
    ax_a.set_title("which fold?", fontsize="x-small")
    _panel_plddt(ax_b, groups)
    ax_b.set_title("fold confidence", fontsize="x-small")
    _panel_energy_structure(ax_c, design_rows)
    ax_c.set_title("energy vs structure (designs)", fontsize="x-small")
    _panel_blast(ax_d, design_rows)
    ax_d.set_title("sequence identity (designs)", fontsize="x-small")
    for ax, lbl in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")):
        lab_plotting.panel_label(ax, lbl)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=characterization_overview"})
    plt.close(fig)
    log.info("wrote characterization overview: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Figure 2: standalone TM_A vs TM_B (the headline)
# --------------------------------------------------------------------------- #

def render_tm_scatter(design_rows: list[dict[str, str]],
                      natural_rows: list[dict[str, str]], out_pdf: Path) -> Path:
    """Standalone TM-to-A vs TM-to-B scatter (designs over the natural clouds)."""
    plt.style.use("lab-paper")
    groups = _by_group(design_rows + natural_rows)

    panel = 4.0
    left, bottom, top = _left_gutter(4), _bottom_gutter(), _title_gutter()
    right = 2 * _BREATHING
    fig_w, fig_h = left + panel + right, bottom + panel + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([left / fig_w, bottom / fig_h, panel / fig_w, panel / fig_h])
    _panel_tm(ax, groups)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=tm_A_vs_B"})
    plt.close(fig)
    log.info("wrote TM scatter: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Figure 3: fold-call composition per group (fraction, so groups are comparable)
# --------------------------------------------------------------------------- #

def render_fold_call_breakdown(design_rows: list[dict[str, str]],
                               natural_rows: list[dict[str, str]], out_pdf: Path) -> Path:
    """Stacked fractional bars of ``fold_call`` per group.

    Fractions (not counts) so the 96 designs are visually comparable to the tens
    of thousands of naturals; the natural bars are the control (CM -> fold A,
    PPIC -> fold B) and the design bar reads out the A/B/neither split.
    """
    plt.style.use("lab-paper")
    groups = _by_group(design_rows + natural_rows)
    order = _present_groups(groups)
    folds_present = [f for f in _FOLD_ORDER
                     if any(_count_by(groups[g], "fold_call").get(f) for g in order)]

    panel_w = max(2.2, 0.9 * len(order) + 0.6)
    panel_h = 2.8
    left, bottom, top = _left_gutter(4), _bottom_gutter(), _title_gutter()
    # room on the right for the fold legend outside the axes
    right = _left_gutter(12)
    fig_w, fig_h = left + panel_w + right, bottom + panel_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([left / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h])

    x = np.arange(len(order))
    bottoms = np.zeros(len(order))
    for f in folds_present:
        fracs = []
        for g in order:
            counts = _count_by(groups[g], "fold_call")
            n = sum(counts.values())
            fracs.append(counts.get(f, 0) / n if n else 0.0)
        fracs = np.asarray(fracs)
        ax.bar(x, fracs, bottom=bottoms, width=0.7, color=_FOLD_COLOR.get(f, "0.6"),
               edgecolor="white", linewidth=0.4, label=_FOLD_LABEL.get(f, f))
        bottoms += fracs
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_GROUP_LABEL.get(g, g)}\n(n={len(groups[g])})" for g in order],
                       fontsize="xx-small")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of group")
    ax.legend(fontsize="xx-small", frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=fold_call_breakdown"})
    plt.close(fig)
    log.info("wrote fold-call breakdown: %s", saved)
    return Path(saved)


# --------------------------------------------------------------------------- #
# Stats table (tidy long: group, metric, value) — the numbers the figures cite
# --------------------------------------------------------------------------- #

def compute_stats_rows(design_rows: list[dict[str, str]],
                       natural_rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Per-group summary as tidy (group, metric, value) triples."""
    groups = _by_group(design_rows + natural_rows)
    rows: list[tuple[str, str, str]] = []
    for g in _present_groups(groups):
        grp = groups[g]
        rows.append((g, "n", str(len(grp))))
        med_plddt = _median(_col(grp, "plddt_mean"))
        rows.append((g, "median_plddt", f"{med_plddt:.2f}" if np.isfinite(med_plddt) else ""))
        med_a, med_b = _median(_col(grp, "tm_A")), _median(_col(grp, "tm_B"))
        rows.append((g, "median_tm_A", f"{med_a:.4f}" if np.isfinite(med_a) else ""))
        rows.append((g, "median_tm_B", f"{med_b:.4f}" if np.isfinite(med_b) else ""))
        counts = _count_by(grp, "fold_call")
        for f in _FOLD_ORDER:
            if counts.get(f):
                rows.append((g, f"fold_call_{f}", str(counts[f])))

    # Control-sanity: a family's naturals must resemble their own reference. Emit "na"
    # (not "FAIL") when a group has no finite TM data, so "FAIL" always means a genuine
    # control failure rather than missing structures.
    def _sanity(rows_sub, key_hi, key_lo):
        m_hi, m_lo = _median(_col(rows_sub, key_hi)), _median(_col(rows_sub, key_lo))
        if not (np.isfinite(m_hi) and np.isfinite(m_lo)):
            return "na"
        return "PASS" if m_hi > m_lo else "FAIL"

    cm_nat = [r for r in natural_rows if "CM" in r.get("group", "")]
    ppic_nat = [r for r in natural_rows if "PPIC" in r.get("group", "")]
    if cm_nat:
        rows.append(("CM-natural", "control_sanity_tmA_gt_tmB", _sanity(cm_nat, "tm_A", "tm_B")))
    if ppic_nat:
        rows.append(("PPIC-natural", "control_sanity_tmB_gt_tmA", _sanity(ppic_nat, "tm_B", "tm_A")))

    # Energy-vs-structure rank correlation (designs).
    sp = _spearman(_col(design_rows, "delta_E"), _col(design_rows, "delta_tm"))
    if sp is not None:
        rows.append(("design", "spearman_deltaE_deltaTM_rho", f"{sp[0]:+.4f}"))
        rows.append(("design", "spearman_deltaE_deltaTM_p", f"{sp[1]:.2g}"))
    return rows


def write_stats(design_rows: list[dict[str, str]],
                natural_rows: list[dict[str, str]], out_tsv: Path) -> Path:
    """Write the tidy (group, metric, value) stats table."""
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["group\tmetric\tvalue"]
    lines += [f"{g}\t{m}\t{v}" for g, m, v in compute_stats_rows(design_rows, natural_rows)]
    out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote characterization stats: %s", out_tsv)
    return out_tsv
