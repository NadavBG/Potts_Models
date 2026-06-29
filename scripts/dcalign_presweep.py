"""DCAlign inference-knob pre-screen for the worse-than-native residual.

The worse-than-native DCAlign residual is a **BP basin-selection failure** (spec
§10.16): DCAlign converges, but to a frame whose in-frame Potts energy is *higher*
than the trivially-available native frame (``ΔE > equal_tol``). The prior/gap-bias
levers were refuted — flattening the prior (``pcount``, §10.15) and the gap
penalties (``μint``/``μext``, §10.14) don't help. The remaining levers are the ones
that change *which basin* BP's annealed marginals land in: the BP **seed** (random
message init + sweep order + Λ noise), and — at more cost — annealing slowness
(``Δβ``/``Δt``).

This tool re-aligns a small curated subset of an iter's home pairs under a sweep of
**any one** ``scoring.*`` knob, every other DCAlign setting pinned to the source
iter, and scores each result in-frame. The DCAlign *alignment* runs on Midway
(DCAlign's FastaIO/GZip stack is broken on macOS ARM — see spec §10.15), so this
tool splits cleanly:

* ``build`` — Mac-side. Curates the subset and writes one cluster-ready run dir per
  swept value (``config_snapshot.yaml`` with ``scoring.<key>`` overridden +
  ``models.json`` + ``query/``), plus a ``sweep_meta.json`` recording the swept key
  so ``score`` is self-describing. Prints the Midway command sequence. No DCAlign.
* ``score`` — Mac-side, after the cluster caches are synced back. Reads each value
  dir's cache, scores every curated home pair in-frame (reusing the validated
  :func:`SBM.energy.dcalign_residual.analyze_record`), and writes the combined
  ``presweep_rows.tsv`` + ``presweep_summary.json`` + ``presweep.pdf``. No DCAlign.

Two scoring aggregations (``--aggregate``):

* ``none`` — per-value comparison (the ``pcount`` use case): recovery of the worse
  pairs and regression of the good controls **at each value independently**.
* ``min``  — **multi-seed-min** (the ``dcalign_seed`` use case): the per-sequence
  *minimum* ΔE over the swept seeds (= the production multi-seed behavior), a
  cumulative **recovery-vs-K** curve (how many seeds you need), and the per-sequence
  ΔE **seed-spread** (the basin-sensitivity diagnostic: high spread ⇒ seeds reach
  different basins, so multi-seed / annealing can help; near-zero ⇒ a seed-robust
  wrong attractor). ``auto`` (the default) picks ``min`` for ``dcalign_seed``,
  ``none`` otherwise.

Curated subset (seeded, logged — never silent):
  * **recover**  — worse-than-native home pairs (``delta_e > equal_tol``), natural
                   *and* synthetic. Each is DCAlign failing to find the energy
                   minimum it should, so all are in scope; ``--hardest`` + a per-group
                   cap selects the largest-ΔE failures (a sharp test).
  * **control**  — a seeded sample of currently-good home pairs
                   (``delta_e ≤ equal_tol``) per (model, kind); these must NOT
                   regress. (Under ``min`` aggregation controls cannot regress, so
                   the real control there is the seed-0 canary below.)

Canary: the baseline value (the first/smallest swept value) reproduces the source
iter — for the ``min`` (seed) sweep, seed 0's recover ΔE must match the source
``residual_rows.tsv`` to ≤5e-7 (checked and reported).

Usage::

    # multi-seed pre-screen (the basin test): hardest worse pairs, seeds 0..5
    python scripts/dcalign_presweep.py build \
        --src-run-dir combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior \
        --sweep-root combine/combine-CM-PPIC-dcalign-seedsweep \
        --scoring-key dcalign_seed --tag-prefix seed --values 0 1 2 3 4 5 \
        --hardest --cap-recover-per-group 4 --n-controls 2
    # ... push, run the printed Midway loop, pull, then:
    python scripts/dcalign_presweep.py score \
        --sweep-root combine/combine-CM-PPIC-dcalign-seedsweep

    # pcount pre-screen (the original use case, unchanged defaults):
    python scripts/dcalign_presweep.py build \
        --src-run-dir combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.dcalign_baseline import DEFAULT_EQUAL_TOL
from SBM.energy.dcalign_residual import ResidualRow, analyze_record
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_RESIDUAL_FIELDS = [f.name for f in fields(ResidualRow)]
_DEFAULT_PCOUNTS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
_DEFAULT_SWEEP_ROOT = Path("combine/combine-CM-PPIC-dcalign-pcsweep")
_DEFAULT_SHARDS = 8  # curated set is small; 2*8 array tasks is plenty
_CANARY_TOL = 5e-7  # in-frame recompute vs source iter (the standing manifest canary)


def _value_tag(prefix: str, value: float) -> str:
    return f"{prefix}{value:g}"


def _row_columns(scoring_key: str) -> list[str]:
    """TSV columns; the swept value is the first column, named for the swept knob
    (so a ``pcount`` sweep writes a ``pcount`` column, a ``dcalign_seed`` sweep a
    ``dcalign_seed`` column — self-documenting and back-compatible)."""
    return [scoring_key, "role", "in_common"] + _RESIDUAL_FIELDS


def _load_models(models_json: Path) -> list[dict]:
    data = json.loads(Path(models_json).read_text(encoding="utf-8"))
    models = data["models"]
    if len(models) != 2:
        raise ValueError(f"expected exactly two models in {models_json}, got {len(models)}")
    return models


def curate_ids(
    residual_tsv: Path,
    *,
    equal_tol: float,
    n_controls: int,
    seed: int,
    cap_recover_per_group: int = 0,
    hardest: bool = False,
) -> dict[str, str]:
    """Return ``{sequence_id: role}`` for the curated subset (seeded, logged).

    Roles (kind — natural/synthetic — is read back from each row at score time):

    * ``recover`` — worse-than-native home pairs (``delta_e > equal_tol``), natural
      *and* synthetic. ``cap_recover_per_group>0`` caps per (model, kind): with
      ``hardest`` the cap takes the **largest-ΔE** failures (a sharp, reproducible
      test); otherwise it takes a seeded random sample. 0 = include all.
    * ``control`` — good home pairs (``delta_e ≤ equal_tol``), seeded sample per
      (model, kind); the lever must not push these worse.
    """
    df = pd.read_csv(residual_tsv, sep="\t")
    df = df[df["ok"].astype(str).str.strip().str.lower().eq("true")]
    rng = np.random.default_rng(seed)
    roles: dict[str, str] = {}

    worse = df[df["delta_e"] > equal_tol]
    for (model, kind), grp in worse.groupby(["model", "kind"]):
        if cap_recover_per_group and grp.shape[0] > cap_recover_per_group:
            if hardest:
                ids = grp.sort_values("delta_e", ascending=False).head(
                    cap_recover_per_group)["sequence_id"].to_numpy()
                how = "hardest"
            else:
                ids = rng.choice(np.sort(grp["sequence_id"].to_numpy()),
                                 size=cap_recover_per_group, replace=False)
                how = "random"
            log.info("model %s %s: %d worse, capped to %d recover (%s)", model, kind,
                     grp.shape[0], len(ids), how)
        else:
            ids = grp["sequence_id"].to_numpy()
            log.info("model %s %s: %d worse -> recover", model, kind, len(ids))
        for sid in ids:
            roles[sid] = "recover"

    good = df[df["delta_e"] <= equal_tol]
    for (model, kind), grp in good.groupby(["model", "kind"]):
        ids = np.sort(grp["sequence_id"].to_numpy())
        take = min(n_controls, ids.size)
        chosen = rng.choice(ids, size=take, replace=False) if take else np.array([], dtype=ids.dtype)
        for sid in chosen:
            roles[sid] = "control"
        log.info("model %s %s: %d good available, sampled %d control(s)",
                 model, kind, ids.size, take)

    return roles


def _subset_query(src_query_dir: Path, dst_query_dir: Path, ids: set[str]) -> None:
    """Write the curated subset of ``query.fasta`` + ``groups.json`` (ids preserved).

    Subsetting the source iter's own query files keeps each record's ``origin_model``
    (needed for home-pair scoring) without rebuilding it.
    """
    dst_query_dir.mkdir(parents=True, exist_ok=True)
    records = datasets.read_query_fasta(src_query_dir / "query.fasta", src_query_dir / "groups.json")
    kept = [r for r in records if r.id in ids]
    found = {r.id for r in kept}
    missing = ids - found
    if missing:
        raise ValueError(f"{len(missing)} curated id(s) absent from {src_query_dir}: "
                         f"{sorted(missing)[:5]}")
    datasets.write_query_fasta(kept, dst_query_dir / "query.fasta", dst_query_dir / "groups.json")
    log.info("wrote curated query: %d records -> %s", len(kept), dst_query_dir)


# ── build: make one cluster-ready run dir per swept value ─────────────────────


def cmd_build(args: argparse.Namespace) -> int:
    src = args.src_run_dir
    scoring_key = args.scoring_key
    src_cfg = yaml.safe_load((src / "config_snapshot.yaml").read_text(encoding="utf-8"))
    if scoring_key not in src_cfg.get("scoring", {}):
        raise KeyError(f"scoring.{scoring_key} not in {src}/config_snapshot.yaml — "
                       f"cannot sweep an unknown knob")
    src_type = type(src_cfg["scoring"][scoring_key])
    residual_tsv = src / "analysis" / "residual_rows.tsv"
    for p in (src / "config_snapshot.yaml", src / "models.json",
              src / "query" / "query.fasta", residual_tsv):
        if not p.is_file():
            raise FileNotFoundError(f"required source input missing: {p}")

    roles = curate_ids(residual_tsv, equal_tol=args.equal_tol, n_controls=args.n_controls,
                       seed=args.seed, cap_recover_per_group=args.cap_recover_per_group,
                       hardest=args.hardest)
    if not roles:
        raise ValueError("no curated ids — nothing to sweep")

    sweep_root = args.sweep_root
    sweep_root.mkdir(parents=True, exist_ok=True)
    curated_query = sweep_root / "query"
    _subset_query(src / "query", curated_query, set(roles))
    (sweep_root / "roles.json").write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")

    aggregate = "min" if scoring_key == "dcalign_seed" else "none"
    (sweep_root / "sweep_meta.json").write_text(json.dumps({
        "scoring_key": scoring_key, "tag_prefix": args.tag_prefix,
        "values": list(args.values), "src_run_dir": str(src),
        "equal_tol": args.equal_tol, "aggregate": aggregate,
    }, indent=2) + "\n", encoding="utf-8")

    started_at = dt.datetime.now(dt.timezone.utc)
    for value in args.values:
        v_dir = sweep_root / _value_tag(args.tag_prefix, value)
        (v_dir / "query").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "models.json", v_dir / "models.json")
        for fn in ("query.fasta", "groups.json"):
            shutil.copy2(curated_query / fn, v_dir / "query" / fn)
        cfg = json.loads(json.dumps(src_cfg))  # deep copy
        cfg["run_name"] = f"{src_cfg['run_name']}-{args.tag_prefix}sweep"
        cfg["scoring"][scoring_key] = src_type(value)  # keep the field's native type
        cfg["scoring"]["n_shards"] = args.n_shards
        cfg["query"] = {"source": "fasta",
                        "fasta": str((curated_query / "query.fasta")),
                        "include": src_cfg["query"]["include"], "cap_per_group": 0}
        (v_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (v_dir / "iteration_note.md").write_text(
            f"# DCAlign inference pre-screen — {scoring_key}={src_type(value)!r}\n\n"
            f"Curated subset of {src} ({len(roles)} home pairs: worse pairs + controls), "
            f"all DCAlign settings pinned to that iter except `scoring.{scoring_key}`. "
            f"Basin-selection lever test (spec §10.16).\n\n"
            f"**Midway resources are NOT set here.** Use `cpus=1` per array task and fan out "
            f"over the array — the iter-002-pcsweep OOM was `cpus=4`/`mem=8G`.\n",
            encoding="utf-8")
        log.info("built run dir: %s (%s=%s)", v_dir, scoring_key, src_type(value))

    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="dcalign_presweep_build", command_line=provenance.current_command_line(),
        inputs={"src_run_dir": src, "residual_tsv": residual_tsv},
        options={"scoring_key": scoring_key, "values": list(args.values),
                 "n_controls": args.n_controls, "hardest": args.hardest,
                 "cap_recover_per_group": args.cap_recover_per_group, "n_shards": args.n_shards,
                 "equal_tol": args.equal_tol, "n_curated": len(roles)},
        seed=args.seed, started_at=started_at, finished_at=finished_at,
        output_path=sweep_root / "roles.json", extra={"roles": roles},
    )
    provenance.save_run_manifest(manifest, sweep_root / "sweep_manifest.json")

    n_rec = sum(1 for r in roles.values() if r == "recover")
    n_ctrl = sum(1 for r in roles.values() if r == "control")
    print(f"\nBuilt {len(args.values)} '{scoring_key}' run dir(s) under {sweep_root}")
    print(f"  curated {len(roles)} home pairs: {n_rec} recover (worse), {n_ctrl} control (good)")
    print(_midway_commands(sweep_root, args.tag_prefix, args.values, args.n_shards, aggregate))
    return 0


def _midway_commands(
    sweep_root: Path, tag_prefix: str, values: list[float], n_shards: int, aggregate: str
) -> str:
    dirs = " ".join(_value_tag(tag_prefix, v) for v in values)
    agg = f" --aggregate {aggregate}" if aggregate != "auto" else ""
    return (
        "\nNext steps (DCAlign alignment runs on Midway — see docs/PIPELINE.md):\n"
        f"  # 1. Mac: push the built run dirs\n"
        f"  bash scripts/sync_models.sh push\n"
        f"  # 2. Midway login node: submit one array per value\n"
        f"  #    (resources are Midway-side: cpus=1 per task, fan out over the array —\n"
        f"  #     the iter-002-pcsweep OOM was cpus=4/mem=8G)\n"
        f"  for d in {dirs}; do\n"
        f"      bash pipeline/external/run_dcalign_align.sh {sweep_root}/$d {n_shards}\n"
        f"  done\n"
        f"  # 3. Midway, AFTER the arrays finish: validate + reclaim scratch\n"
        f"  for d in {dirs}; do bash pipeline/external/finalize_dcalign_push.sh {sweep_root}/$d; done\n"
        f"  # 4. Mac: pull caches and score\n"
        f"  bash scripts/sync_models.sh pull\n"
        f"  .venv/bin/python scripts/dcalign_presweep.py score --sweep-root {sweep_root}{agg}\n"
    )


# ── score: read synced caches and decide ──────────────────────────────────────


def _score_value_dir(
    scoring_key: str, value: float, v_dir: Path, models: dict,
    records_by_id: dict, roles: dict[str, str],
) -> tuple[list[dict], list[str]]:
    """Score curated home pairs from ``v_dir``'s cache → ``(rows, missing_ids)``.

    The cluster run can be incomplete (e.g. an OOM-killed shard), so curated ids
    absent from the cache are **skipped and counted** (loud WARN), not fatal — a
    fair cross-value comparison is recovered later on the common set. A wholly
    missing ``alignments.tsv`` is still a hard error (the caller pre-checks and
    skips the value instead).
    """
    cache_dir = v_dir / "dcalign" / "cache"
    caches = {}
    for name in models:
        tsv = cache_dir / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(f"missing cache for {name!r} at {tsv}")
        caches[name] = read_alignment_cache(tsv)
    rows: list[dict] = []
    missing: list[str] = []
    for sid, role in roles.items():
        rec = records_by_id[sid]
        model = models.get(rec.origin_model)
        if model is None:
            continue
        dca = caches[model.name].get(sid)
        if dca is None:
            missing.append(sid)
            continue
        row = analyze_record(rec, model, dca)
        rows.append({scoring_key: value, "role": role, "in_common": False, **row.as_dict()})
    n_failed = sum(1 for r in rows if not r["ok"])
    if missing:
        log.warning("%s=%g: %d/%d curated id(s) absent from cache (incomplete run): %s%s",
                    scoring_key, value, len(missing), len(roles), missing[:5],
                    " …" if len(missing) > 5 else "")
    log.info("%s=%g: scored %d present (%d failed-frame), %d missing",
             scoring_key, value, len(rows), n_failed, len(missing))
    return rows, missing


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_rows(rows: list[dict], columns: list[str], path: Path) -> None:
    lines = ["\t".join(columns)]
    lines += ["\t".join(_fmt(r[c]) for c in columns) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote sweep rows: %s (%d rows)", path, len(rows))


def _stats(sub: pd.DataFrame, equal_tol: float) -> dict:
    ok = sub[sub["ok"]]
    n = len(ok)
    n_worse = int((ok["delta_e"] > equal_tol).sum())
    return {
        "n": n,
        "n_worse_than_native": n_worse,
        "n_recovered": n - n_worse,
        "n_beat_native": int((ok["delta_e"] < -equal_tol).sum()),
        "frac_recovered": ((n - n_worse) / n) if n else 0.0,
        "median_delta_e": float(ok["delta_e"].median()) if n else None,
    }


def _by_kind(sub: pd.DataFrame, equal_tol: float) -> dict:
    d = {"all": _stats(sub, equal_tol)}
    for kind in ("natural", "synthetic"):
        ks = sub[sub["kind"] == kind]
        if len(ks):
            d[kind] = _stats(ks, equal_tol)
    return d


def summarize(rows: list[dict], *, equal_tol: float, scoring_key: str, baseline_value: float) -> dict:
    """Per-value recovery (worse pairs) + regression (good pairs), split by kind
    (the ``none`` aggregation, e.g. the ``pcount`` sweep).

    Both roles count the same direction (``delta_e ≤ equal_tol`` = at/below native):
    for ``recover`` that is the win (``n_recovered``), for ``control`` its complement
    is the cost (``n_worse_than_native`` = regressions). ``n_beat_native``
    (``delta_e < -equal_tol``) flags a strictly *lower* energy than native.
    """
    df = pd.DataFrame(rows)
    out: dict = {"aggregate": "none", "equal_tol": equal_tol, "scoring_key": scoring_key,
                 f"by_{scoring_key}": {}}
    by = out[f"by_{scoring_key}"]
    for value, g in df.groupby(scoring_key):
        by[f"{value:g}"] = {
            role: _by_kind(g[g["role"] == role], equal_tol)
            for role in ("recover", "control")
            if (g["role"] == role).any()
        }

    # Recommendation: most worse pairs recovered (all kinds) with control regressions
    # no worse than baseline.
    base = by.get(f"{baseline_value:g}", {})
    base_ctrl_worse = base.get("control", {}).get("all", {}).get("n_worse_than_native", 0)
    best_value, best_recovered = None, -1
    for value_str, stats in by.items():
        ctrl_worse = stats.get("control", {}).get("all", {}).get("n_worse_than_native", 0)
        recovered = stats.get("recover", {}).get("all", {}).get("n_recovered", 0)
        if ctrl_worse <= base_ctrl_worse and recovered > best_recovered:
            best_value, best_recovered = value_str, recovered
    out[f"baseline_{scoring_key}"] = f"{baseline_value:g}"
    out[f"recommended_{scoring_key}"] = best_value
    return out


def _seq_delta_matrix(df: pd.DataFrame, scoring_key: str) -> tuple[list[float], dict]:
    """Pivot ``ok`` rows to ``{sequence_id: {role, kind, de: [ΔE per value]}}`` in
    ascending value order. Only sequences present at every value are kept (the
    common set already guarantees this; the guard is defensive)."""
    ok = df[df["ok"]]
    values = sorted(ok[scoring_key].unique())
    out: dict = {}
    for sid, g in ok.groupby("sequence_id"):
        by_value = g.set_index(scoring_key)["delta_e"]
        if not all(v in by_value.index for v in values):
            continue
        out[sid] = {"role": g.iloc[0]["role"], "kind": g.iloc[0]["kind"],
                    "de": [float(by_value.loc[v]) for v in values]}
    return values, out


def _recovery_vs_k(sids: list[str], recs: dict, values: list[float], equal_tol: float) -> list[dict]:
    """Cumulative recovery as seeds are added in value order: at K, each sequence's
    energy is the min ΔE over the first K seeds (= production multi-seed-min)."""
    curve = []
    for k in range(1, len(values) + 1):
        cummins = [min(recs[s]["de"][:k]) for s in sids]
        n = len(cummins)
        n_rec = sum(1 for x in cummins if x <= equal_tol)
        curve.append({
            "k": k, "value_added": values[k - 1], "n": n,
            "n_recovered": n_rec, "frac_recovered": (n_rec / n) if n else 0.0,
            "n_beat_native": sum(1 for x in cummins if x < -equal_tol),
            "median_min_delta_e": float(np.median(cummins)) if n else None,
        })
    return curve


def _spread(sids: list[str], recs: dict) -> dict:
    """Per-sequence ΔE spread (max−min across seeds): the basin-sensitivity signal."""
    spreads = [max(recs[s]["de"]) - min(recs[s]["de"]) for s in sids]
    if not spreads:
        return {"n": 0}
    return {"n": len(spreads), "median_spread": float(np.median(spreads)),
            "mean_spread": float(np.mean(spreads)), "max_spread": float(np.max(spreads))}


def summarize_multiseed(rows: list[dict], *, equal_tol: float, scoring_key: str) -> dict:
    """Per-sequence min ΔE over the swept seeds (the ``min`` aggregation): a
    recovery-vs-K curve (how many seeds are needed) and the per-sequence ΔE spread
    (basin-sensitivity), per role and split by kind."""
    df = pd.DataFrame(rows)
    values, recs = _seq_delta_matrix(df, scoring_key)
    out: dict = {"aggregate": "min", "equal_tol": equal_tol, "scoring_key": scoring_key,
                 "values": values, "by_role": {}}
    def _entry(sids: list[str]) -> dict:
        return {"n": len(sids),
                "recovery_vs_k": _recovery_vs_k(sids, recs, values, equal_tol),
                "seed_spread": _spread(sids, recs)}

    for role in ("recover", "control"):
        sids = [s for s, r in recs.items() if r["role"] == role]
        if not sids:
            continue
        entry = {"n": len(sids), "all": _entry(sids)}
        for kind in ("natural", "synthetic"):
            ks = [s for s in sids if recs[s]["kind"] == kind]
            if ks:
                entry[kind] = _entry(ks)
        out["by_role"][role] = entry
    return out


def _seed_canary(rows: list[dict], scoring_key: str, src_residual_tsv: Path) -> dict:
    """Compare the baseline value's (seed 0) per-id ΔE to the source iter's
    ``residual_rows.tsv``. They are the same alignment, so ΔE must match to
    ``_CANARY_TOL`` — the standing in-frame-recompute canary. Missing source file or
    ids is reported (WARN), never silently passed."""
    df = pd.DataFrame(rows)
    baseline = min(df[scoring_key].unique())
    base = df[(df[scoring_key] == baseline) & df["ok"]]
    if not src_residual_tsv.is_file():
        log.warning("canary: source residual_rows.tsv not found at %s — skipping", src_residual_tsv)
        return {"checked": False, "reason": f"missing {src_residual_tsv}"}
    src = pd.read_csv(src_residual_tsv, sep="\t").set_index("sequence_id")
    diffs, n_missing = [], 0
    for _, r in base.iterrows():
        sid = r["sequence_id"]
        if sid not in src.index:
            n_missing += 1
            continue
        diffs.append(abs(float(r["delta_e"]) - float(src.loc[sid, "delta_e"])))
    max_diff = max(diffs) if diffs else None
    ok = bool(diffs) and max_diff <= _CANARY_TOL
    if not ok:
        log.warning("canary FAILED: baseline %s=%g max |ΔE diff| vs source = %s (tol %g), %d missing",
                    scoring_key, baseline, max_diff, _CANARY_TOL, n_missing)
    else:
        log.info("canary OK: baseline %s=%g reproduces source ΔE (max diff %.2e, %d ids)",
                 scoring_key, baseline, max_diff, len(diffs))
    return {"checked": True, "baseline_value": float(baseline), "n_compared": len(diffs),
            "n_missing_in_source": n_missing, "max_abs_delta_e_diff": max_diff,
            "tol": _CANARY_TOL, "passed": ok}


def _verdict_none(summary: dict) -> str:
    key = summary["scoring_key"]
    by = summary[f"by_{key}"]
    base_str, rec_str = summary[f"baseline_{key}"], summary[f"recommended_{key}"]
    base_r = by.get(base_str, {}).get("recover", {})
    r = by.get(rec_str, {}) if rec_str else {}
    ra, rn, rs = r.get("recover", {}).get("all", {}), r.get("recover", {}).get("natural", {}), \
        r.get("recover", {}).get("synthetic", {})
    ca = r.get("control", {}).get("all", {})
    return (
        f"baseline {key}={base_str}: "
        f"{base_r.get('all', {}).get('n_recovered', 0)}/{base_r.get('all', {}).get('n', 0)} "
        f"worse pairs recovered. Recommended {key}={rec_str}: "
        f"{ra.get('n_recovered', 0)}/{ra.get('n', 0)} recovered "
        f"(natural {rn.get('n_recovered', 0)}/{rn.get('n', 0)}, "
        f"synthetic {rs.get('n_recovered', 0)}/{rs.get('n', 0)}; "
        f"{ra.get('n_beat_native', 0)} beat native), "
        f"{ca.get('n_worse_than_native', 0)}/{ca.get('n', 0)} controls regressed."
    )


def _verdict_min(summary: dict) -> str:
    rec = summary["by_role"].get("recover", {})
    curve = rec.get("all", {}).get("recovery_vs_k", [])
    if not curve:
        return "no recover sequences in the common set."
    k1, klast = curve[0], curve[-1]
    spread = rec.get("all", {}).get("seed_spread", {})
    rn = rec.get("natural", {}).get("recovery_vs_k", [{}])[-1]
    rs = rec.get("synthetic", {}).get("recovery_vs_k", [{}])[-1]
    return (
        f"multi-seed over K={klast['k']} seeds: "
        f"{klast['n_recovered']}/{klast['n']} worse pairs recovered "
        f"(natural {rn.get('n_recovered', 0)}/{rn.get('n', 0)}, "
        f"synthetic {rs.get('n_recovered', 0)}/{rs.get('n', 0)}; "
        f"{klast['n_beat_native']} beat native); single-seed baseline "
        f"{k1['n_recovered']}/{k1['n']}. Median ΔE seed-spread = "
        f"{spread.get('median_spread', 0.0):.2f} a.u. "
        f"(high ⇒ seed-sensitive basins; near-0 ⇒ seed-robust)."
    )


def cmd_score(args: argparse.Namespace) -> int:
    sweep_root = args.sweep_root
    meta_path = sweep_root / "sweep_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Legacy sweeps (built before the knob was a parameter) had no sweep_meta:
        # default to the pcount layout so the archived pcsweep still scores.
        log.warning("no sweep_meta.json under %s — assuming legacy pcount layout", sweep_root)
        meta = {"scoring_key": "pcount", "tag_prefix": "pc", "aggregate": "none",
                "equal_tol": DEFAULT_EQUAL_TOL, "src_run_dir": ""}
    scoring_key = meta["scoring_key"]
    tag_prefix = meta["tag_prefix"]
    equal_tol = args.equal_tol if args.equal_tol is not None else meta["equal_tol"]
    aggregate = args.aggregate or meta.get("aggregate", "none")
    if aggregate == "auto":
        aggregate = "min" if scoring_key == "dcalign_seed" else "none"
    columns = _row_columns(scoring_key)

    roles = json.loads((sweep_root / "roles.json").read_text(encoding="utf-8"))
    value_dirs = sorted(d for d in sweep_root.glob(f"{tag_prefix}*")
                        if (d / "models.json").is_file())
    if not value_dirs:
        raise FileNotFoundError(f"no {tag_prefix}<val> run dirs with models.json under {sweep_root}")
    # models.json is identical across value dirs; read it from the first one.
    model_entries = _load_models(value_dirs[0] / "models.json")
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    records_by_id = {r.id: r for r in datasets.read_query_fasta(
        sweep_root / "query" / "query.fasta", sweep_root / "query" / "groups.json")}

    started_at = dt.datetime.now(dt.timezone.utc)
    rows: list[dict] = []
    coverage: dict[str, dict] = {}
    scored_values: list[float] = []
    for v_dir in value_dirs:
        value = float(v_dir.name[len(tag_prefix):])
        if not all((v_dir / "dcalign" / "cache" / name / "alignments.tsv").is_file()
                   for name in models):
            log.warning("%s=%g: no alignments.tsv cache — skipping (gather/sync incomplete?)",
                        scoring_key, value)
            coverage[f"{value:g}"] = {"n_curated": len(roles), "n_present": 0,
                                      "n_missing": len(roles), "scored": False}
            continue
        v_rows, missing = _score_value_dir(scoring_key, value, v_dir, models, records_by_id, roles)
        rows.extend(v_rows)
        scored_values.append(value)
        n_ok = sum(1 for r in v_rows if r["ok"])
        coverage[f"{value:g}"] = {"n_curated": len(roles), "n_present": len(v_rows), "n_ok": n_ok,
                                  "n_failed": len(v_rows) - n_ok, "n_missing": len(missing),
                                  "scored": True}
    if not rows:
        raise ValueError(
            "no value had a usable alignments.tsv cache. The cluster gather step likely did not "
            "merge shards/ → dcalign/cache/<model>/alignments.tsv (sync excludes shards/). "
            "Re-run the gather/finalize on Midway, then sync_models.sh pull.")

    # Fair cross-value comparison: restrict to ids successfully scored at EVERY scored
    # value (an incomplete run drops different ids at different values).
    ok_ids = [set(r["sequence_id"] for r in rows if r[scoring_key] == v and r["ok"])
              for v in scored_values]
    common = set.intersection(*ok_ids) if ok_ids else set()
    for r in rows:
        r["in_common"] = r["sequence_id"] in common
    log.info("scored %d/%d value(s); common fully-scored set = %d/%d curated home pairs",
             len(scored_values), len(value_dirs), len(common), len(roles))

    values_sorted = sorted(scored_values)
    rows_tsv = sweep_root / "presweep_rows.tsv"
    _write_rows(rows, columns, rows_tsv)
    common_rows = [r for r in rows if r["in_common"]]
    if not common_rows:
        raise ValueError("the common fully-scored set is empty — too few sequences overlap across "
                         "values to compare; inspect coverage in the logs / presweep_rows.tsv")

    if aggregate == "min":
        summary = summarize_multiseed(common_rows, equal_tol=equal_tol, scoring_key=scoring_key)
        summary["canary"] = _seed_canary(
            common_rows, scoring_key, Path(meta["src_run_dir"]) / "analysis" / "residual_rows.tsv")
        summary["verdict"] = _verdict_min(summary)
    else:
        summary = summarize(common_rows, equal_tol=equal_tol, scoring_key=scoring_key,
                            baseline_value=values_sorted[0])
        summary["verdict"] = _verdict_none(summary)
    summary["coverage"] = coverage
    summary["n_common"] = len(common)
    summary["scored_on"] = "common fully-scored set across all scored values"
    (sweep_root / "presweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    log.info("wrote summary: %s", sweep_root / "presweep_summary.json")

    from SBM.utils import utils_dcalign_presweep_plot
    utils_dcalign_presweep_plot.render_presweep(
        rows_tsv, sweep_root / "presweep.pdf", scoring_key=scoring_key,
        aggregate=aggregate, equal_tol=equal_tol)

    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="dcalign_presweep_score", command_line=provenance.current_command_line(),
        inputs={"sweep_root": sweep_root, "models_json": value_dirs[0] / "models.json"},
        options={"scoring_key": scoring_key, "aggregate": aggregate, "values": values_sorted,
                 "equal_tol": equal_tol,
                 "models": {m["name"]: m.get("model_sha256") for m in model_entries}},
        seed=None, started_at=started_at, finished_at=finished_at, output_path=rows_tsv,
        extra={"summary": summary},
    )
    provenance.save_run_manifest(manifest, sweep_root / "presweep_manifest.json")
    print(f"Coverage: scored {len(scored_values)}/{len(value_dirs)} values; "
          f"common fully-scored set = {len(common)}/{len(roles)} curated home pairs.")
    for value_str, cov in coverage.items():
        if cov["scored"]:
            print(f"  {scoring_key}={value_str}: {cov['n_ok']} ok, {cov['n_failed']} failed-frame, "
                  f"{cov['n_missing']} missing of {cov['n_curated']}")
        else:
            print(f"  {scoring_key}={value_str}: NO CACHE (skipped)")
    if aggregate == "min" and not summary["canary"].get("passed", True):
        print("WARNING: seed-0 canary did NOT reproduce the source iter — inspect before trusting.")
    print(summary["verdict"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="curate + write one cluster run dir per swept value (Mac-side)")
    b.add_argument("--src-run-dir", type=Path, required=True,
                   help="the iter dcalign run to pre-screen (source of query + residual_rows)")
    b.add_argument("--scoring-key", default="pcount",
                   help="the scoring.* knob to sweep (e.g. pcount, dcalign_seed)")
    b.add_argument("--tag-prefix", default="pc", help="value-dir name prefix (e.g. pc, seed)")
    b.add_argument("--values", type=float, nargs="+", default=_DEFAULT_PCOUNTS,
                   help="values to sweep (first/smallest = baseline / canary)")
    b.add_argument("--n-controls", type=int, default=20,
                   help="good (non-regression) controls per (model, kind), seeded")
    b.add_argument("--cap-recover-per-group", type=int, default=0,
                   help="cap worse pairs per (model, kind) (0 = include all; >0 for a quick run)")
    b.add_argument("--hardest", action="store_true",
                   help="when capping, take the largest-ΔE worse pairs (else seeded random)")
    b.add_argument("--n-shards", type=int, default=_DEFAULT_SHARDS, help="Slurm array shards per model")
    b.add_argument("--seed", type=int, default=0, help="master seed for curated sampling")
    b.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL)
    b.add_argument("--sweep-root", type=Path, default=_DEFAULT_SWEEP_ROOT)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score synced cluster caches + decide (Mac-side)")
    s.add_argument("--sweep-root", type=Path, default=_DEFAULT_SWEEP_ROOT)
    s.add_argument("--aggregate", choices=["auto", "none", "min"], default=None,
                   help="none = per-value; min = multi-seed-min recovery-vs-K "
                        "(default: from sweep_meta.json — min for dcalign_seed, else none)")
    s.add_argument("--equal-tol", type=float, default=None,
                   help="override the build-time equal_tol (default: from sweep_meta.json)")
    s.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
