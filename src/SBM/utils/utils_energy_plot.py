"""Consolidated figure for the two-model energy combine run.

One figure answers "how do the two models score the same sequences": a scatter
of ``E_A`` vs ``E_B`` with stacked marginal histograms, points colored by source
group (each family's natural + synthetic sets). A native of family ``k`` sits low
on its own model's axis and high on the other's — the visual form of acceptance
test 4. Routed through ``scripts/lab_plotting.py`` so the lab palette and PDF
provenance metadata stay the single source of truth.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# scripts/lab_plotting.py is a sibling script, not a package module. Add the
# scripts/ dir to sys.path so ``import lab_plotting`` works regardless of cwd.
# Mirrors the pattern in src/SBM/utils/utils_mpnn_plot.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import lab_plotting  # noqa: E402

log = logging.getLogger(__name__)


def _wide_table(scores_tsv: Path, model_names: tuple[str, str]) -> pd.DataFrame:
    """Pivot the tidy scores TSV to one row per sequence with E per model."""
    long = pd.read_csv(scores_tsv, sep="\t")
    name_A, name_B = model_names
    wide = long.pivot_table(index="sequence_id", columns="model", values="energy")
    meta = long.drop_duplicates("sequence_id").set_index("sequence_id")[["group", "origin_model"]]
    wide = wide.join(meta)
    missing = [n for n in (name_A, name_B) if n not in wide.columns]
    if missing:
        raise ValueError(f"scores TSV missing energies for model(s) {missing}")
    return wide.dropna(subset=[name_A, name_B])


def _group_colors(groups: list[str]) -> dict[str, str]:
    """Assign each source group a stable, colorblind-safe color."""
    palette = lab_plotting.WONG_PALETTE[1:]  # reserve black for chrome
    return {g: palette[i % len(palette)] for i, g in enumerate(sorted(groups))}


def render_two_model_energy(
    scores_tsv: Path,
    model_names: tuple[str, str],
    out_pdf: Path,
) -> Path:
    """Render the E_A vs E_B scatter + marginals to ``out_pdf``.

    The axis labels carry the exact model variant (e.g. "E under CM-bm-dense
    model"); the precise run dir + sha256 of each model live in the run's
    ``models.json`` / ``manifest.json``, so the figure stays uncluttered.
    """
    plt.style.use("lab-paper")
    name_A, name_B = model_names
    wide = _wide_table(scores_tsv, model_names)
    groups = sorted(wide["group"].unique())
    colors = _group_colors(groups)

    fig, axes = lab_plotting.scatter_with_marginals(
        wide[name_A].to_numpy(), wide[name_B].to_numpy(),
        bins=40, scatter_kwargs={"s": 8, "alpha": 0.0},  # real points drawn per-group below
    )
    ax = axes["main"]
    for g in groups:
        sub = wide[wide["group"] == g]
        ax.scatter(sub[name_A], sub[name_B], s=8, alpha=0.6, color=colors[g], label=g, linewidths=0)

    lo = float(np.nanmin([wide[name_A].min(), wide[name_B].min()]))
    hi = float(np.nanmax([wide[name_A].max(), wide[name_B].max()]))
    ax.plot([lo, hi], [lo, hi], color=lab_plotting.LAB_COLORS["chrome"], lw=0.8, ls="--", zorder=0)
    ax.set_xlabel(f"E under {name_A} model (a.u.)")
    ax.set_ylabel(f"E under {name_B} model (a.u.)")
    ax.legend(fontsize="x-small", frameon=False, loc="best")
    lab_plotting.panel_label(axes["top"], "A")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    saved = lab_plotting.save_figure(
        fig, out_pdf, extra_metadata={"Keywords": "sbm_figure=two_model_energy"}
    )
    plt.close(fig)
    log.info("wrote energy figure: %s", saved)
    return Path(saved)
