"""Consolidated figure for the iter-003 Phase-0 residual diagnostic.

One figure answers "is the worse-than-native residual something a better
insertion prior can fix?" (see :mod:`SBM.energy.dcalign_residual`):

* **A — where the residual lives**: worse-than-native counts per model, stacked
  by natural vs synthetic. Synthetic (MCMC samples, no insertions) is out of
  reach for the insertion prior, so its share bounds what the escalation cannot do.
* **B — what shape the residual is**: among the worse-than-native rows, the
  frame-disagreement labels (terminal / register_shift / gap_redistribution)
  stacked per kind. A natural tail dominated by terminal+register_shift is
  prior-shaped (a better Λ is plausible); gap_redistribution is the diffuse
  failure mode the insertion prior cannot touch.
* **C — which lever could move it**: among the worse-than-native rows, the
  lever buckets (prior_only / mu_addressable / mu_counterproductive) stacked per
  kind. A tail dominated by prior_only means μint/μext are provably neutral
  (equal gap counts) and the lever is ``pcount`` (prior-flattening); only a large
  mu_addressable share would justify a μ sweep. This is the iter-003 decision.

Reads one tidy long-form table (``residual_rows.tsv``, one home pair per row) and
routes through ``scripts/lab_plotting.py`` so the palette + PDF provenance stay
the single source of truth (mirrors ``utils_dcalign_baseline_plot.py``).
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
_PANEL_W, _PANEL_H = 3.0, 3.2
_MARGIN_W, _MARGIN_H = 1.2, 1.0

#: Naturals green (lab convention), synthetics warm (out of the prior's reach).
_KIND_COLORS = {"natural": "#117733", "synthetic": lab_plotting.WONG_PALETTE[6]}
_KIND_ORDER = ("natural", "synthetic")

#: Prior-shaped labels get cool colors; the gap-penalty failure mode is warm.
_LABEL_COLORS = {
    "terminal": lab_plotting.WONG_PALETTE[2],  # sky blue
    "register_shift": lab_plotting.WONG_PALETTE[5],  # blue
    "gap_redistribution": lab_plotting.WONG_PALETTE[6],  # vermillion
}
_LABEL_ORDER = ("terminal", "register_shift", "gap_redistribution")

#: Lever buckets: μ-out-of-reach (warm), μ-candidate (cool), μ-harmful (black).
_LEVER_COLORS = {
    "prior_only": lab_plotting.WONG_PALETTE[6],  # vermillion — pcount, not μ
    "mu_addressable": lab_plotting.WONG_PALETTE[2],  # sky blue — μ could help
    "mu_counterproductive": lab_plotting.WONG_PALETTE[0],  # black — μ hurts
}
_LEVER_ORDER = ("prior_only", "mu_addressable", "mu_counterproductive")


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse the ``true``/``false`` tokens the TSV writer emits into bool."""
    return series.astype(str).str.strip().str.lower().eq("true")


def _stacked(ax, x, segments: dict[str, list[int]], colors: dict[str, str]) -> None:
    bottom = np.zeros(len(x))
    for key, counts in segments.items():
        counts = np.asarray(counts, dtype=float)
        ax.bar(x, counts, bottom=bottom, color=colors[key], label=key.replace("_", " "))
        bottom += counts


def render_residual_anatomy(
    rows_tsv: Path,
    model_names: tuple[str, str],
    out_pdf: Path,
    *,
    equal_tol: float = _DEFAULT_EQUAL_TOL,
) -> Path:
    """Render the three-panel residual-anatomy figure to ``out_pdf``."""
    plt.style.use("lab-paper")
    df = pd.read_csv(rows_tsv, sep="\t")
    df = df[_as_bool(df["ok"])]
    worse = df[df["delta_e"] > equal_tol]

    models = [m for m in model_names if (df["model"] == m).any()]
    if not models:
        raise ValueError(f"no rows for models {model_names} in {rows_tsv}")

    figsize = (3 * _PANEL_W + _MARGIN_W, _PANEL_H + _MARGIN_H)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=figsize)

    # Panel A — worse-than-native per model, stacked by kind.
    xa = np.arange(len(models))
    seg_a = {
        kind: [int(((worse["model"] == m) & (worse["kind"] == kind)).sum()) for m in models]
        for kind in _KIND_ORDER
    }
    _stacked(ax_a, xa, seg_a, _KIND_COLORS)
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels(models, fontsize="xx-small", rotation=20, ha="right")
    ax_a.set_ylabel(rf"worse than native ($\Delta E > {equal_tol:g}$ a.u.) (count)")
    ax_a.set_title("residual by model", fontsize="x-small")
    ax_a.legend(fontsize="xx-small", frameon=False, title="kind", title_fontsize="xx-small")

    # Panel B — disagreement-label composition among the worse rows, per kind.
    kinds = [k for k in _KIND_ORDER if (worse["kind"] == k).any()]
    xb = np.arange(len(kinds))
    seg_b = {
        lab: [int(((worse["kind"] == k) & (worse["label"] == lab)).sum()) for k in kinds]
        for lab in _LABEL_ORDER
    }
    _stacked(ax_b, xb, seg_b, _LABEL_COLORS)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(kinds, fontsize="xx-small")
    ax_b.set_ylabel("worse-than-native (count)")
    nat_worse = worse[worse["kind"] == "natural"]
    frac_prior = (
        float(nat_worse["label"].isin(("terminal", "register_shift")).mean())
        if len(nat_worse)
        else 0.0
    )
    ax_b.set_title(f"disagreement shape (natural prior-shaped: {frac_prior:.0%})",
                   fontsize="x-small")
    ax_b.legend(fontsize="xx-small", frameon=False, title="frame disagreement",
                title_fontsize="xx-small")

    # Panel C — lever addressability among the worse rows, per kind: which knob
    # could move each pair. prior_only ⇒ pcount (μ provably neutral on equal gap
    # counts); mu_addressable ⇒ a μ direction could help (candidate only).
    xc = np.arange(len(kinds))
    seg_c = {
        lev: [int(((worse["kind"] == k) & (worse["lever"] == lev)).sum()) for k in kinds]
        for lev in _LEVER_ORDER
    }
    _stacked(ax_c, xc, seg_c, _LEVER_COLORS)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(kinds, fontsize="xx-small")
    ax_c.set_ylabel("worse-than-native (count)")
    frac_prior_only = (
        float((worse["lever"] == "prior_only").mean()) if len(worse) else 0.0
    )
    ax_c.set_title(f"lever (prior_only: {frac_prior_only:.0%})", fontsize="x-small")
    ax_c.legend(fontsize="xx-small", frameon=False, title="addressable by",
                title_fontsize="xx-small")

    lab_plotting.panel_label(ax_a, "A")
    lab_plotting.panel_label(ax_b, "B")
    lab_plotting.panel_label(ax_c, "C")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=dcalign_residual_anatomy"}
    )
    plt.close(fig)
    log.info("wrote residual-anatomy figure: %s", saved)
    return Path(saved)
