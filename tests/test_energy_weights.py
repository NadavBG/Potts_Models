"""Tests for the post-hoc E_tot weight derivation (SBM.utils.energy_weights).

Pure numpy/pandas (no MCMC, no model load), like tests/test_energy.py. The weight
math has a closed form, so the checks compare against it directly rather than
pinning current output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SBM.utils import energy_weights

# The tidy header score_two_models.py writes; the module only needs a subset.
_HEADER = "sequence_id\tgroup\torigin_model\tmodel\tweight\tmethod\tenergy\tess\tmc_stderr\tseed"


def _row(seq_id: str, group: str, origin: str, model: str, energy) -> str:
    e = "nan" if energy is None else f"{energy:.10g}"
    return f"{seq_id}\t{group}\t{origin}\t{model}\t1\tmarginal\t{e}\t\t\t"


def _write_scores(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([_HEADER, *rows]) + "\n", encoding="utf-8")


# ── closed-form weight math ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "m_A, m_B",
    [(-257.086, -179.997), (-20.0, -5.0), (-100.0, -100.0), (10.0, 40.0)],
)
def test_equalize_weights_closed_form(m_A: float, m_B: float) -> None:
    w_A, w_B = energy_weights.equalize_weights(m_A, m_B)
    # Normalized and matches the closed form.
    assert np.isclose(w_A + w_B, 1.0)
    assert np.isclose(w_A, m_B / (m_A + m_B))
    assert np.isclose(w_B, m_A / (m_A + m_B))
    # The defining property: the weighted median native energies are equal.
    assert np.isclose(w_A * m_A, w_B * m_B)


def test_equalize_weights_equal_medians_is_half() -> None:
    w_A, w_B = energy_weights.equalize_weights(-42.0, -42.0)
    assert np.isclose(w_A, 0.5) and np.isclose(w_B, 0.5)


def test_equalize_weights_opposite_signs_raises() -> None:
    with pytest.raises(ValueError, match="opposite signs"):
        energy_weights.equalize_weights(-100.0, 50.0)


def test_equalize_weights_zero_sum_raises() -> None:
    with pytest.raises(ValueError, match="~0 or non-finite"):
        energy_weights.equalize_weights(-30.0, 30.0)


def test_weight_sweep_endpoints_and_crossing() -> None:
    m_A, m_B = -20.0, -5.0
    sweep = energy_weights.weight_sweep(m_A, m_B, n=101)
    first, last = sweep.iloc[0], sweep.iloc[-1]
    # w_A=0: only B contributes m_B; w_A=1: only A contributes m_A.
    assert np.isclose(first["weighted_median_A"], 0.0)
    assert np.isclose(first["weighted_median_B"], m_B)
    assert np.isclose(last["weighted_median_A"], m_A)
    assert np.isclose(last["weighted_median_B"], 0.0)
    # At the derived w_A the two curves cross.
    w_A, _ = energy_weights.equalize_weights(m_A, m_B)
    assert np.isclose(w_A * m_A, (1.0 - w_A) * m_B)


# ── native median selection + end-to-end write ──────────────────────────


def test_native_median_energies_selects_home_naturals_only(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    _write_scores(scores, [
        # Home block A: CM naturals under CM -> median of {-10,-20,-30} = -20 (NaN dropped).
        _row("CM|natural|0", "CM/natural", "CM", "CM", -10.0),
        _row("CM|natural|1", "CM/natural", "CM", "CM", -20.0),
        _row("CM|natural|2", "CM/natural", "CM", "CM", -30.0),
        _row("CM|natural|3", "CM/natural", "CM", "CM", None),  # skip row -> dropped
        # Cross: CM naturals under PPIC (must not count for either home median).
        _row("CM|natural|0", "CM/natural", "CM", "PPIC", -1.0),
        # Home block B: PPIC naturals under PPIC -> median of {-4,-6} = -5.
        _row("PPIC|natural|0", "PPIC/natural", "PPIC", "PPIC", -4.0),
        _row("PPIC|natural|1", "PPIC/natural", "PPIC", "PPIC", -6.0),
        # Synthetic (same origin/model as home A) must be excluded by the /natural filter.
        _row("CM|synthetic|0", "CM/synthetic-T1.0", "CM", "CM", -1000.0),
        # Random control (origin "") must be excluded.
        _row("random|0", "random/N91", "", "CM", -2.0),
    ])
    med = energy_weights.native_median_energies(scores, "CM", "PPIC")
    assert med["CM"] == {"median": -20.0, "n": 3}
    assert med["PPIC"] == {"median": -5.0, "n": 2}


def test_compute_and_write_produces_expected_artifacts(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    _write_scores(scores, [
        _row("CM|natural|0", "CM/natural", "CM", "CM", -10.0),
        _row("CM|natural|1", "CM/natural", "CM", "CM", -30.0),
        _row("PPIC|natural|0", "PPIC/natural", "PPIC", "PPIC", -5.0),
    ])
    out_json = tmp_path / "energy_weights.json"
    out_sweep = tmp_path / "energy_weight_sweep.tsv"
    result = energy_weights.compute_and_write(scores, "CM", "PPIC", out_json, out_sweep)

    # m_A=-20, m_B=-5 -> w_A=0.2, w_B=0.8, equalized=-4.
    assert np.isclose(result["w_A"], 0.2)
    assert np.isclose(result["w_B"], 0.8)
    assert np.isclose(result["equalized_weighted_median"], -4.0)
    assert result["native_median_energy"] == {"CM": -20.0, "PPIC": -5.0}

    on_disk = json.loads(out_json.read_text(encoding="utf-8"))
    assert on_disk["weights"] == {"CM": pytest.approx(0.2), "PPIC": pytest.approx(0.8)}
    assert out_sweep.is_file()
    header = out_sweep.read_text(encoding="utf-8").splitlines()[0]
    assert header == "w_A\tweighted_median_A\tweighted_median_B"


def test_empty_home_block_raises(tmp_path: Path) -> None:
    scores = tmp_path / "scores.tsv"
    # PPIC has no home naturals -> loud failure rather than a bogus weight.
    _write_scores(scores, [
        _row("CM|natural|0", "CM/natural", "CM", "CM", -10.0),
        _row("PPIC|natural|0", "PPIC/natural", "PPIC", "CM", -1.0),  # cross only
    ])
    with pytest.raises(ValueError, match="empty home block"):
        energy_weights.native_median_energies(scores, "CM", "PPIC")
