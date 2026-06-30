"""Unit tests for the warm-start fixed-point probe logic
(:mod:`SBM.energy.dcalign_warmstart`).

Covers the per-sequence classifier, the energy/frame plumbing of
``analyze_warmstart_record`` (on a tiny hand-built Potts model with known
energies), and the case-A/B verdict + control sanity in ``summarize_warmstart``.
Expected values are reasoned from the construction, not pinned from a run.

    .venv/bin/python -m pytest tests/test_dcalign_warmstart.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from SBM.energy.dcalign_warmstart import (
    CONTROL_DRIFT,
    CONTROL_OK,
    FAILED,
    FLOWED_OTHER,
    FLOWED_TO_RAND,
    STAYED_NATIVE,
    WarmstartRow,
    analyze_warmstart_record,
    build_warmstart_verdict,
    classify_warmstart,
    summarize_warmstart,
)
from SBM.energy.datasets import QueryRecord
from SBM.energy.model import PottsModel
from SBM.utils.dcalign_score import DCAlignResult

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"


# ── classifier (pure) ─────────────────────────────────────────────────────────

def test_classify_recover_stayed_is_case_a():
    assert classify_warmstart(role="recover", delta_e_warm=0.2, delta_e_rand=30.0,
                              col_agree_native=0.95, col_agree_rand=0.4) == STAYED_NATIVE


def test_classify_recover_stayed_via_col_agreement_override():
    # Energetically ~equivalent re-threading just above tol but native by columns.
    assert classify_warmstart(role="recover", delta_e_warm=1.5, delta_e_rand=30.0,
                              col_agree_native=0.995, col_agree_rand=0.3) == STAYED_NATIVE


def test_classify_recover_flowed_to_rand_is_case_b():
    assert classify_warmstart(role="recover", delta_e_warm=30.0, delta_e_rand=30.0,
                              col_agree_native=0.5, col_agree_rand=1.0) == FLOWED_TO_RAND


def test_classify_recover_flowed_other():
    assert classify_warmstart(role="recover", delta_e_warm=15.0, delta_e_rand=30.0,
                              col_agree_native=0.5, col_agree_rand=0.5) == FLOWED_OTHER


def test_classify_control_ok_and_drift():
    assert classify_warmstart(role="control", delta_e_warm=0.0, delta_e_rand=0.0,
                              col_agree_native=1.0, col_agree_rand=1.0) == CONTROL_OK
    assert classify_warmstart(role="control", delta_e_warm=20.0, delta_e_rand=0.0,
                              col_agree_native=0.4, col_agree_rand=0.4) == CONTROL_DRIFT


# ── analyze_warmstart_record plumbing (tiny known-energy model) ───────────────

def _tiny_model() -> PottsModel:
    """L=3, q=21, zero J; h boosts native residues A,C,D at cols 0,1,2 (E_native=-30)."""
    L, q = 3, 21
    h = np.zeros((L, q))
    native = np.array([1, 2, 3])  # A, C, D
    for i in range(L):
        h[i, native[i]] = 10.0
    J = np.zeros((L, L, q, q))
    return PottsModel(name="M", J=J, h=h, L=L, q=q, alphabet=ALPHABET,
                      gauge="zero_sum", sha256="x", source="x")


def _record() -> QueryRecord:
    return QueryRecord(id="s1", group="M/natural", origin_model="M",
                       ints=np.array([1, 2, 3]))


def _dca(frame: str, energy: float = float("nan")) -> DCAlignResult:
    return DCAlignResult(seq_id="s1", aligned_frame=frame, dcalign_energy=energy,
                         converged=True, used_decimation=False, n_iter=1)


def test_analyze_stayed_at_native():
    row = analyze_warmstart_record(_record(), _tiny_model(), "recover", _dca("ACD"), None)
    assert row.ok
    assert row.delta_e_warm == pytest.approx(0.0, abs=1e-9)
    assert row.e_native == pytest.approx(-30.0)
    assert row.col_agree_native == pytest.approx(1.0)
    assert row.label == STAYED_NATIVE


def test_analyze_flowed_to_rand():
    # warm-start drifts to "EFG" (E=0 ⇒ ΔE=+30), exactly the production frame.
    row = analyze_warmstart_record(_record(), _tiny_model(), "recover",
                                   _dca("EFG"), _dca("EFG"))
    assert row.delta_e_warm == pytest.approx(30.0)
    assert row.delta_e_rand == pytest.approx(30.0)
    assert row.col_agree_rand == pytest.approx(1.0)
    assert row.label == FLOWED_TO_RAND


def test_analyze_flowed_other():
    # warm drifts to "EFD" (E=-h[2,3]=-10 ⇒ ΔE=+20); production was "EFG" (ΔE=+30).
    row = analyze_warmstart_record(_record(), _tiny_model(), "recover",
                                   _dca("EFD"), _dca("EFG"))
    assert row.delta_e_warm == pytest.approx(20.0)
    assert row.label == FLOWED_OTHER


def test_analyze_failed_frame_is_kept_not_dropped():
    row = analyze_warmstart_record(_record(), _tiny_model(), "recover", _dca(""), None)
    assert not row.ok and row.label == FAILED
    assert np.isnan(row.delta_e_warm)
    assert row.e_native == pytest.approx(-30.0)  # native still scored


# ── summarize + verdict ───────────────────────────────────────────────────────

def _row(role: str, label: str, de_warm: float, de_rand: float = 30.0) -> WarmstartRow:
    return WarmstartRow(
        sequence_id="x", model="M", group="M/natural", kind="natural", role=role,
        e_native=-30.0, e_warmstart=-30.0 + de_warm, e_randominit=-30.0 + de_rand,
        delta_e_warm=de_warm, delta_e_rand=de_rand,
        col_agree_native=1.0 if de_warm <= 1.0 else 0.5, col_agree_rand=0.5,
        warm_converged=True, warm_used_decimation=False, warm_n_iter=1, label=label, ok=True)


def test_summarize_case_a_when_most_stay():
    rows = [_row("recover", STAYED_NATIVE, 0.0) for _ in range(3)] + \
           [_row("recover", FLOWED_TO_RAND, 30.0)] + \
           [_row("control", CONTROL_OK, 0.0)]
    out = summarize_warmstart(rows)
    assert out["recover"]["overall"]["n_stayed_native"] == 3
    assert out["recover"]["overall"]["frac_stayed_native"] == pytest.approx(0.75)
    assert out["control"]["n_control_drift"] == 0
    assert out["verdict"].startswith("CASE A")


def test_summarize_case_b_when_none_stay():
    rows = [_row("recover", FLOWED_TO_RAND, 30.0) for _ in range(4)]
    out = summarize_warmstart(rows)
    assert out["recover"]["overall"]["n_stayed_native"] == 0
    assert out["verdict"].startswith("CASE B")


def test_summarize_mixed_and_control_drift_warned():
    rows = [_row("recover", STAYED_NATIVE, 0.0),
            _row("recover", FLOWED_TO_RAND, 30.0),
            _row("recover", FLOWED_OTHER, 15.0),
            _row("control", CONTROL_DRIFT, 20.0)]
    out = summarize_warmstart(rows)
    assert out["control"]["n_control_drift"] == 1
    assert "MIXED" in out["verdict"]
    assert "WARNING" in out["verdict"]


def test_verdict_handles_no_successful_rows():
    assert "nothing to conclude" in build_warmstart_verdict([], [], 1.0)


def test_verdict_map_init_wording_case_a():
    rows = [_row("recover", STAYED_NATIVE, 0.0) for _ in range(3)] + \
           [_row("recover", FLOWED_TO_RAND, 30.0)]
    out = summarize_warmstart(rows, init_kind="map")
    assert out["init_kind"] == "map"
    assert out["verdict"].startswith("CASE A")
    assert "fields-MAP" in out["verdict"] and "REACHED" in out["verdict"]


def test_verdict_map_init_case_b_points_to_anneal():
    out = summarize_warmstart([_row("recover", FLOWED_TO_RAND, 30.0) for _ in range(4)],
                              init_kind="map")
    assert out["verdict"].startswith("CASE B")
    assert "anneal" in out["verdict"]
