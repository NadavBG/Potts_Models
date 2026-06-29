"""Figure for the DCAlign warm-start fixed-point probe (scripts/analyze_dcalign_warmstart.py).

Two panels over the worse-than-native (recover) home pairs:

* **Panel A** — scatter of where BP landed *from* the native start (``delta_e_warm``,
  y) vs how bad the production frame was (``delta_e_rand``, x), coloured by kind.
  The ``y = 0`` line is "stayed at native" (**case A**); the ``y = x`` diagonal is
  "drifted to the production frame's energy" (**case B**). Points hugging y=0 ⇒ the
  native frame is a reachable fixed point the random-init runs missed; points on the
  diagonal ⇒ native is unstable under DCAlign's own dynamics.
* **Panel B** — label counts (stayed_native / flowed_to_rand / flowed_other) by kind.

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

_PANEL_W, _PANEL_H = 3.4, 3.2
_MARGIN_W, _MARGIN_H = 1.4, 1.0

_KIND_COLOR = {"natural": "#117733", "synthetic": lab_plotting.WONG_PALETTE[6]}
_LABELS = ("stayed_native", "flowed_to_rand", "flowed_other")
_LABEL_SHORT = {"stayed_native": "stayed\n(case A)", "flowed_to_rand": "->prod\n(case B)",
                "flowed_other": "->other"}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def render_warmstart(rows_tsv: Path, out_pdf: Path, *, equal_tol: float = 1.0) -> Path:
    """Render the two-panel warm-start figure to ``out_pdf``."""
    plt.style.use("lab-paper")
    df = pd.read_csv(rows_tsv, sep="\t")
    df = df[_as_bool(df["ok"]) & (df["role"] == "recover")]
    if df.empty:
        raise ValueError(f"no successful recover rows in {rows_tsv}")

    figsize = (2 * _PANEL_W + _MARGIN_W, _PANEL_H + _MARGIN_H)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    # Panel A: ΔE_warm vs ΔE_rand.
    lo, hi = 0.0, float(max(df["delta_e_rand"].max(), df["delta_e_warm"].max(), equal_tol) * 1.1)
    ax_a.plot([lo, hi], [lo, hi], color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--",
              zorder=0, label="drift to prod (case B)")
    ax_a.axhline(equal_tol, color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls=":", zorder=0)
    for kind, sub in df.groupby("kind"):
        ax_a.scatter(sub["delta_e_rand"], sub["delta_e_warm"], s=22,
                     color=_KIND_COLOR.get(kind, "0.4"), label=kind, edgecolor="white", lw=0.4)
    ax_a.set_xlabel(r"$\Delta E$ production frame (a.u.)")
    ax_a.set_ylabel(r"$\Delta E$ warm-start from native (a.u.)")
    ax_a.set_title("stayed at native (y=0) vs drifted (y=x)", fontsize="x-small")
    ax_a.legend(fontsize="xx-small", frameon=False)

    # Panel B: label counts by kind.
    kinds = sorted(df["kind"].unique())
    x = np.arange(len(_LABELS))
    width = 0.8 / max(len(kinds), 1)
    for i, kind in enumerate(kinds):
        counts = [int((df[df["kind"] == kind]["label"] == lab).sum()) for lab in _LABELS]
        ax_b.bar(x + i * width, counts, width, color=_KIND_COLOR.get(kind, "0.4"), label=kind)
    ax_b.set_xticks(x + width * (len(kinds) - 1) / 2)
    ax_b.set_xticklabels([_LABEL_SHORT[l] for l in _LABELS], fontsize="xx-small")
    ax_b.set_ylabel("worse pairs (count)")
    ax_b.set_title("outcome by kind", fontsize="x-small")
    ax_b.legend(fontsize="xx-small", frameon=False)

    lab_plotting.panel_label(ax_a, "A")
    lab_plotting.panel_label(ax_b, "B")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=dcalign_warmstart"})
    plt.close(fig)
    log.info("wrote DCAlign warm-start figure: %s", saved)
    return Path(saved)
