"""Derive the two-model ``E_tot`` combining weights post-hoc from the naturals.

The combine pipeline scores every query under both models, giving per-model
energies ``E_A``, ``E_B``. The consolidated total is ``E_tot = w_A·E_A + w_B·E_B``.
The two models have different *native* energy scales (e.g. CM naturals sit at
E≈−257, PPIC at E≈−180), so equal weights bias ``E_tot`` — and any annealing on
it — toward whichever family sits lower on its own model.

This module removes that bias by choosing weights that equalize the two families'
**median native energy**. With ``w_A + w_B = 1``:

    w_A·m_A = w_B·m_B   ⇒   w_A = m_B/(m_A+m_B),  w_B = m_A/(m_A+m_B)

where ``m_A`` (``m_B``) is the median energy of family A's (B's) naturals scored
under their *home* model. The weights land in ``data/energy_weights.json`` and a
``w_A`` sweep in ``data/energy_weight_sweep.tsv`` (which backs
``figs/energy_weights.pdf``). Pure numpy/pandas — no MCMC, no model load.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("sequence_id", "group", "origin_model", "model", "energy")


def _home_block_median(long: pd.DataFrame, name: str) -> tuple[float, int]:
    """Median energy of ``name``'s naturals scored under ``name``'s own model.

    Home block = rows with ``origin_model == model == name`` whose group is a
    natural set (``group`` ends with ``/natural``); NaN energies (e.g. potts_align
    skip rows) are dropped. This excludes cross pairs, the random control, and any
    synthetic groups, so the median reflects the family's native-energy scale.
    """
    home = long[
        (long["origin_model"] == name)
        & (long["model"] == name)
        & (long["group"].astype(str).str.endswith("/natural"))
    ]
    energies = pd.to_numeric(home["energy"], errors="coerce").to_numpy(dtype=float)
    energies = energies[np.isfinite(energies)]
    if energies.size == 0:
        raise ValueError(
            f"no finite native-energy rows for model {name!r} "
            f"(need origin_model==model=={name!r} and group '*/natural'); "
            "cannot derive weights from an empty home block"
        )
    return float(np.median(energies)), int(energies.size)


def native_median_energies(
    scores_tsv: Path | str, name_A: str, name_B: str
) -> dict[str, dict[str, float]]:
    """Median native energy + row count for each model's home naturals.

    Reads the tidy ``scores.tsv`` written by ``score_two_models.py`` and returns
    ``{name: {"median": m, "n": k}}`` for both models.
    """
    long = pd.read_csv(scores_tsv, sep="\t")
    missing = [c for c in _REQUIRED_COLUMNS if c not in long.columns]
    if missing:
        raise ValueError(f"scores TSV {scores_tsv} missing required column(s) {missing}")
    m_A, n_A = _home_block_median(long, name_A)
    m_B, n_B = _home_block_median(long, name_B)
    return {name_A: {"median": m_A, "n": n_A}, name_B: {"median": m_B, "n": n_B}}


def equalize_weights(m_A: float, m_B: float) -> tuple[float, float]:
    """Normalized weights (``w_A + w_B = 1``) equalizing ``w_A·m_A = w_B·m_B``.

    Closed form: ``w_A = m_B/(m_A+m_B)``, ``w_B = m_A/(m_A+m_B)``. Raises loudly on
    the degenerate cases where no solution has both weights in ``(0, 1)`` — opposite
    signs, or ``m_A + m_B ≈ 0`` — rather than returning a silently unusable weight.
    """
    total = m_A + m_B
    if not math.isfinite(total) or math.isclose(total, 0.0, abs_tol=1e-12):
        raise ValueError(
            f"cannot equalize weights: m_A + m_B = {total!r} is ~0 or non-finite "
            f"(m_A={m_A!r}, m_B={m_B!r})"
        )
    if (m_A > 0) != (m_B > 0):
        raise ValueError(
            f"cannot equalize weights: native medians have opposite signs "
            f"(m_A={m_A:.6g}, m_B={m_B:.6g}); w_A·m_A = w_B·m_B has no solution with "
            "both weights in (0, 1)"
        )
    w_A = m_B / total
    w_B = m_A / total
    if not (0.0 < w_A < 1.0 and 0.0 < w_B < 1.0):
        log.warning(
            "derived weights fall outside (0, 1): w_A=%.6g, w_B=%.6g (m_A=%.6g, m_B=%.6g)",
            w_A, w_B, m_A, m_B,
        )
    return w_A, w_B


def weight_sweep(m_A: float, m_B: float, n: int = 201) -> pd.DataFrame:
    """Weighted median energies as ``w_A`` sweeps ``[0, 1]`` (``w_B = 1 − w_A``).

    Columns: ``w_A``, ``weighted_median_A`` (``w_A·m_A``), ``weighted_median_B``
    (``(1 − w_A)·m_B``). The two columns cross at the equalizing ``w_A``.
    """
    if n < 2:
        raise ValueError(f"weight_sweep needs n >= 2 grid points, got {n}")
    w_A = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "w_A": w_A,
            "weighted_median_A": w_A * m_A,
            "weighted_median_B": (1.0 - w_A) * m_B,
        }
    )


def compute_and_write(
    scores_tsv: Path | str,
    name_A: str,
    name_B: str,
    out_json: Path | str,
    out_sweep_tsv: Path | str,
    *,
    sweep_points: int = 201,
) -> dict:
    """Derive the equalizing weights from ``scores_tsv`` and write both artifacts.

    Writes ``out_json`` (medians, counts, formula, weights, equalized energy) and
    ``out_sweep_tsv`` (the ``w_A`` sweep backing the figure). Returns the result dict.
    """
    medians = native_median_energies(scores_tsv, name_A, name_B)
    m_A = medians[name_A]["median"]
    m_B = medians[name_B]["median"]
    w_A, w_B = equalize_weights(m_A, m_B)
    equalized = w_A * m_A  # == w_B * m_B by construction

    result = {
        "schema_version": 1,
        "models": {"A": name_A, "B": name_B},
        "native_median_energy": {name_A: m_A, name_B: m_B},
        "native_count": {name_A: medians[name_A]["n"], name_B: medians[name_B]["n"]},
        "formula": (
            "w_A = m_B/(m_A+m_B); w_B = m_A/(m_A+m_B); w_A+w_B=1; "
            "equalizes w_A*m_A = w_B*m_B where m_X = median native energy of family X"
        ),
        "weights": {name_A: w_A, name_B: w_B},
        "w_A": w_A,
        "w_B": w_B,
        "equalized_weighted_median": equalized,
        "source_scores_tsv": str(scores_tsv),
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    out_sweep_tsv = Path(out_sweep_tsv)
    out_sweep_tsv.parent.mkdir(parents=True, exist_ok=True)
    weight_sweep(m_A, m_B, n=sweep_points).to_csv(out_sweep_tsv, sep="\t", index=False)

    log.info(
        "derived weights: w_%s=%.6f w_%s=%.6f (native medians %.4f / %.4f -> equalized %.4f)",
        name_A, w_A, name_B, w_B, m_A, m_B, equalized,
    )
    return result
