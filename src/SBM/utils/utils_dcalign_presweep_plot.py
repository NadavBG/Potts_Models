"""Figure for the DCAlign inference-knob pre-screen (see scripts/dcalign_presweep.py).

Two modes, matching the scorer's ``--aggregate``:

* **none** (the ``pcount`` sweep) — one series per (role, kind) over the swept value.
  Panel A: fraction still worse-than-native vs value; Panel B: median ΔE vs value.
  The lever works if the **recover** curves fall toward 0 while the **control**
  curves stay at 0 (a rising control curve is the regression cost).
* **min** (the ``dcalign_seed`` multi-seed sweep) — the production multi-seed-min.
  Panel A: cumulative **recovery fraction vs K** (number of seeds; per-sequence min
  ΔE over the first K seeds). Panel B: the per-sequence ΔE **seed-spread**
  (max−min across seeds) — the basin-sensitivity diagnostic: spread ≫ equal_tol
  means seeds reach different basins (multi-seed/annealing can help); spread ≈ 0
  means a seed-robust wrong attractor.

Markers are measured points connected by straight segments (a sweep, not a fit).
Routes through ``scripts/lab_plotting.py`` for the palette + PDF provenance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)

_DEFAULT_EQUAL_TOL = 1.0

#: Per-panel inch budgets (figsize derived from these, not guessed).
_PANEL_W, _PANEL_H = 3.2, 3.2
_MARGIN_W, _MARGIN_H = 1.3, 1.0

#: Color by kind (natural green, synthetic vermillion); linestyle by role.
_KIND_COLOR = {"natural": "#117733", "synthetic": lab_plotting.WONG_PALETTE[6]}
_ROLE_STYLE = {"recover": ("-", "o"), "control": ("--", "s")}
_SERIES = [("recover", "natural"), ("recover", "synthetic"),
           ("control", "natural"), ("control", "synthetic")]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def render_presweep(
    rows_tsv: Path, out_pdf: Path, *, scoring_key: str, aggregate: str,
    equal_tol: float = _DEFAULT_EQUAL_TOL,
) -> Path:
    """Render the two-panel pre-screen figure to ``out_pdf`` (mode by ``aggregate``)."""
    plt.style.use("lab-paper")
    df = pd.read_csv(rows_tsv, sep="\t")
    df = df[_as_bool(df["ok"])]
    if "in_common" in df.columns:
        df = df[_as_bool(df["in_common"])]
    if df.empty:
        raise ValueError(f"no successful in-common rows in {rows_tsv}")
    if aggregate == "min":
        return _render_multiseed(df, out_pdf, scoring_key=scoring_key, equal_tol=equal_tol)
    return _render_per_value(df, out_pdf, scoring_key=scoring_key, equal_tol=equal_tol)


def _new_fig():
    figsize = (2 * _PANEL_W + _MARGIN_W, _PANEL_H + _MARGIN_H)
    return plt.subplots(1, 2, figsize=figsize)


def _save(fig, out_pdf: Path) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=dcalign_presweep"})
    plt.close(fig)
    log.info("wrote DCAlign pre-screen figure: %s", saved)
    return Path(saved)


def _render_per_value(df: pd.DataFrame, out_pdf: Path, *, scoring_key: str, equal_tol: float) -> Path:
    fig, (ax_a, ax_b) = _new_fig()
    for role, kind in _SERIES:
        sub = df[(df["role"] == role) & (df["kind"] == kind)]
        if sub.empty:
            continue
        ls, marker = _ROLE_STYLE[role]
        color = _KIND_COLOR[kind]
        label = f"{role} {kind}"
        grp = sub.groupby(scoring_key)
        vals = np.array(sorted(sub[scoring_key].unique()), dtype=float)
        frac_worse = [float((grp.get_group(v)["delta_e"] > equal_tol).mean()) for v in vals]
        med_de = [float(grp.get_group(v)["delta_e"].median()) for v in vals]
        ax_a.plot(vals, frac_worse, marker=marker, ls=ls, color=color, label=label, markersize=4)
        ax_b.plot(vals, med_de, marker=marker, ls=ls, color=color, label=label, markersize=4)

    vals_all = np.array(sorted(df[scoring_key].unique()), dtype=float)
    use_log = vals_all.min() > 0 and (vals_all.max() / vals_all.min()) >= 10
    for ax in (ax_a, ax_b):
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel(scoring_key)
    ax_a.set_ylabel(rf"worse than native ($\Delta E > {equal_tol:g}$ a.u.) (fraction)")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.set_title("recovery (recover down) vs regression (control flat)", fontsize="x-small")
    ax_a.legend(fontsize="xx-small", frameon=False)
    ax_b.axhline(equal_tol, color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--")
    ax_b.axhline(0.0, color=lab_plotting.LAB_COLORS["chrome"], lw=0.6, ls=":")
    ax_b.set_ylabel(r"median $\Delta E$ (DCAlign $-$ native) (a.u.)")
    ax_b.set_title(r"energy gap ($\Delta E<0$ = beat native)", fontsize="x-small")
    ax_b.legend(fontsize="xx-small", frameon=False)
    lab_plotting.panel_label(ax_a, "A")
    lab_plotting.panel_label(ax_b, "B")
    return _save(fig, out_pdf)


def _seq_matrix(df: pd.DataFrame, scoring_key: str) -> tuple[list[float], dict]:
    """Per-sequence ΔE vector over the seeds (ascending value order)."""
    values = sorted(df[scoring_key].unique())
    out: dict = {}
    for sid, g in df.groupby("sequence_id"):
        by_value = g.set_index(scoring_key)["delta_e"]
        if not all(v in by_value.index for v in values):
            continue
        out[sid] = {"role": g.iloc[0]["role"], "kind": g.iloc[0]["kind"],
                    "de": np.array([float(by_value.loc[v]) for v in values])}
    return values, out


def _render_multiseed(df: pd.DataFrame, out_pdf: Path, *, scoring_key: str, equal_tol: float) -> Path:
    values, recs = _seq_matrix(df, scoring_key)
    ks = np.arange(1, len(values) + 1)
    fig, (ax_a, ax_b) = _new_fig()

    # Panel A: cumulative recovery fraction vs K (per-seq min ΔE over first K seeds).
    for role, kind in _SERIES:
        sids = [s for s, r in recs.items() if r["role"] == role and r["kind"] == kind]
        if not sids:
            continue
        ls, marker = _ROLE_STYLE[role]
        de = np.stack([recs[s]["de"] for s in sids])  # (n_seq, n_val)
        cummin = np.minimum.accumulate(de, axis=1)
        frac_rec = (cummin <= equal_tol).mean(axis=0)
        ax_a.plot(ks, frac_rec, marker=marker, ls=ls, color=_KIND_COLOR[kind],
                  label=f"{role} {kind}", markersize=4)
    ax_a.set_xlabel("seeds combined (K)")
    ax_a.set_xticks(ks)
    ax_a.set_ylabel(r"recovered ($\min_K \Delta E \leq$ tol) (fraction)")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.set_title("multi-seed-min recovery vs K", fontsize="x-small")
    ax_a.legend(fontsize="xx-small", frameon=False)

    # Panel B: per-sequence ΔE seed-spread (max−min) — the basin-sensitivity signal.
    box_data, box_labels, box_colors = [], [], []
    for role, kind in _SERIES:
        sids = [s for s, r in recs.items() if r["role"] == role and r["kind"] == kind]
        if not sids:
            continue
        spreads = [float(recs[s]["de"].max() - recs[s]["de"].min()) for s in sids]
        box_data.append(spreads)
        box_labels.append(f"{role[:3]}\n{kind[:4]}")
        box_colors.append(_KIND_COLOR[kind])
    positions = np.arange(len(box_data))
    bp = ax_b.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                      showfliers=True, flierprops={"marker": ".", "markersize": 3})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")
    ax_b.axhline(equal_tol, color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--")
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(box_labels, fontsize="xx-small")
    ax_b.set_ylabel(r"$\Delta E$ seed-spread (max$-$min over seeds) (a.u.)")
    ax_b.set_title("basin sensitivity (spread >> tol = seeds differ)", fontsize="x-small")

    lab_plotting.panel_label(ax_a, "A")
    lab_plotting.panel_label(ax_b, "B")
    return _save(fig, out_pdf)
