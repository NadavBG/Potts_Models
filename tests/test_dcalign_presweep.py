"""Unit tests for the DCAlign inference-knob pre-screen logic
(``scripts/dcalign_presweep.py``).

Covers the two pieces with non-trivial logic that the back-compat integration check
(re-scoring the archived pcsweep) does *not* exercise:

* ``curate_ids`` hardest-by-ΔE selection (vs seeded random),
* ``summarize_multiseed`` — the cumulative recovery-vs-K curve and per-sequence
  seed-spread (the multi-seed-min aggregation), plus the seed-0 canary.

The expected numbers are computed by hand from small inputs (cumulative-min over
seeds), not pinned from a prior run.

    .venv/bin/python -m pytest tests/test_dcalign_presweep.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import dcalign_presweep as dp  # noqa: E402

KEY = "dcalign_seed"


def _residual_tsv(tmp_path: Path) -> Path:
    """A tiny residual_rows.tsv: one model, naturals with a clear ΔE ranking plus
    good controls."""
    rows = [
        # sequence_id, model, kind, delta_e, ok
        ("n1", "M", "natural", 2.0, True),
        ("n2", "M", "natural", 6.0, True),
        ("n3", "M", "natural", 4.0, True),
        ("n4", "M", "natural", 5.0, True),
        ("n5", "M", "natural", 3.0, True),
        ("g1", "M", "natural", 0.5, True),
        ("g2", "M", "natural", 0.2, True),
        ("g3", "M", "natural", 0.0, True),
        ("bad", "M", "natural", 9.9, False),  # dropped (not ok)
    ]
    path = tmp_path / "residual_rows.tsv"
    header = "sequence_id\tmodel\tkind\tdelta_e\tok"
    lines = [header] + [f"{a}\t{b}\t{c}\t{d}\t{'true' if e else 'false'}" for a, b, c, d, e in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_curate_ids_hardest_takes_largest_delta_e(tmp_path):
    roles = dp.curate_ids(_residual_tsv(tmp_path), equal_tol=1.0, n_controls=2, seed=0,
                          cap_recover_per_group=2, hardest=True)
    recover = {sid for sid, role in roles.items() if role == "recover"}
    # ΔE ranking: n2(6) > n4(5) > n3(4) > n5(3) > n1(2); hardest 2 = n2, n4.
    assert recover == {"n2", "n4"}
    controls = {sid for sid, role in roles.items() if role == "control"}
    assert len(controls) == 2 and controls <= {"g1", "g2", "g3"}
    assert "bad" not in roles  # not-ok rows are dropped


def test_curate_ids_random_is_seeded_and_capped(tmp_path):
    tsv = _residual_tsv(tmp_path)
    a = dp.curate_ids(tsv, equal_tol=1.0, n_controls=0, seed=7, cap_recover_per_group=3, hardest=False)
    b = dp.curate_ids(tsv, equal_tol=1.0, n_controls=0, seed=7, cap_recover_per_group=3, hardest=False)
    assert a == b  # same seed → reproducible
    recover = {sid for sid, role in a.items() if role == "recover"}
    assert len(recover) == 3 and recover <= {"n1", "n2", "n3", "n4", "n5"}


def _seed_rows(spec: dict[str, tuple[str, str, list[float]]], seeds=(0, 1, 2)) -> list[dict]:
    """Build scorer rows from ``{sid: (role, kind, [ΔE per seed])}``."""
    rows = []
    for sid, (role, kind, des) in spec.items():
        for seed, de in zip(seeds, des):
            rows.append({KEY: float(seed), "role": role, "in_common": True,
                         "sequence_id": sid, "kind": kind, "delta_e": de, "ok": True})
    return rows


def test_summarize_multiseed_recovery_vs_k_and_spread():
    spec = {
        "rn": ("recover", "natural", [10.0, 0.5, 5.0]),   # recovers at K=2 (cummin 0.5)
        "rs": ("recover", "synthetic", [8.0, 9.0, 7.0]),  # never ≤ tol
        "cn": ("control", "natural", [0.0, 0.0, 0.0]),    # always recovered
    }
    out = dp.summarize_multiseed(_seed_rows(spec), equal_tol=1.0, scoring_key=KEY)
    assert out["aggregate"] == "min" and out["values"] == [0.0, 1.0, 2.0]

    rec_all = out["by_role"]["recover"]["all"]["recovery_vs_k"]
    # cummins per K: rn=[10,0.5,0.5], rs=[8,8,7]; recovered (≤1): K1=0, K2=1, K3=1 of 2.
    assert [c["n_recovered"] for c in rec_all] == [0, 1, 1]
    assert [c["n"] for c in rec_all] == [2, 2, 2]
    assert rec_all[1]["frac_recovered"] == pytest.approx(0.5)
    # monotone non-decreasing recovery (cumulative min can only help)
    assert all(rec_all[i + 1]["n_recovered"] >= rec_all[i]["n_recovered"] for i in range(2))

    rec_nat = out["by_role"]["recover"]["natural"]["recovery_vs_k"]
    assert [c["n_recovered"] for c in rec_nat] == [0, 1, 1]
    rec_syn = out["by_role"]["recover"]["synthetic"]["recovery_vs_k"]
    assert [c["n_recovered"] for c in rec_syn] == [0, 0, 0]

    # controls are good at every K
    ctrl_all = out["by_role"]["control"]["all"]["recovery_vs_k"]
    assert [c["n_recovered"] for c in ctrl_all] == [1, 1, 1]

    # seed-spread = max−min ΔE across seeds: rn=9.5, rs=2.0 → median 5.75
    assert out["by_role"]["recover"]["all"]["seed_spread"]["median_spread"] == pytest.approx(5.75)


def test_summarize_multiseed_beat_native():
    spec = {"rn": ("recover", "natural", [10.0, -2.0, 3.0])}  # K=2 beats native (< -tol)
    out = dp.summarize_multiseed(_seed_rows(spec), equal_tol=1.0, scoring_key=KEY)
    curve = out["by_role"]["recover"]["all"]["recovery_vs_k"]
    assert [c["n_beat_native"] for c in curve] == [0, 1, 1]


def test_seed_canary_pass_and_fail(tmp_path):
    spec = {
        "rn": ("recover", "natural", [10.0, 0.5, 5.0]),
        "cn": ("control", "natural", [0.0, 0.0, 0.0]),
    }
    rows = _seed_rows(spec)
    # source residual: baseline (seed 0) ΔE = rn 10.0, cn 0.0
    src = tmp_path / "residual_rows.tsv"
    src.write_text("sequence_id\tdelta_e\nrn\t10.0\ncn\t0.0\n", encoding="utf-8")
    ok = dp._seed_canary(rows, KEY, src)
    assert ok["checked"] and ok["passed"] and ok["n_compared"] == 2
    assert ok["max_abs_delta_e_diff"] == pytest.approx(0.0, abs=1e-12)

    bad = tmp_path / "residual_bad.tsv"
    bad.write_text("sequence_id\tdelta_e\nrn\t10.5\ncn\t0.0\n", encoding="utf-8")  # 0.5 off
    fail = dp._seed_canary(rows, KEY, bad)
    assert fail["checked"] and not fail["passed"]
    assert fail["max_abs_delta_e_diff"] == pytest.approx(0.5)


def test_seed_canary_missing_source_is_reported_not_silent(tmp_path):
    rows = _seed_rows({"rn": ("recover", "natural", [10.0, 0.5, 5.0])})
    out = dp._seed_canary(rows, KEY, tmp_path / "does_not_exist.tsv")
    assert out["checked"] is False and "missing" in out["reason"]
