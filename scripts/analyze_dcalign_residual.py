"""iter-003 Phase-0 diagnostic gate: anatomy of the worse-than-native residual.

Post-hoc analysis of an existing combine ``dcalign`` run (e.g. iter-002) — reads
the cached DCAlign frames and the query set, classifies every home pair's
frame disagreement (:mod:`SBM.energy.dcalign_residual`), and writes a decision
bundle under ``<run-dir>/analysis/``. NO new DCAlign run: the cache is fixed, so
the output is deterministic.

It answers the Phase-1 gate question: of the home pairs DCAlign still scores worse
than their native frame, how many are natural (the only ones an insertion prior
could help) vs synthetic (out of reach), and is the natural tail prior-shaped
(terminal/register_shift) or a gap-penalty problem (gap_redistribution)?

It also answers the iter-003 lever question (§10.14): per worse pair, compute the
interior/terminal gap counts in both frames and bucket which knob could move it —
``prior_only`` (μ provably neutral; only pcount can), ``mu_addressable`` (a μ knob
could help — candidate only), or ``mu_counterproductive``. The ``lever_verdict``
names the indicated lever (pcount vs μint/μext).

Usage::

    python scripts/analyze_dcalign_residual.py \
        --run-dir combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior

Outputs (under ``--out-dir``, default ``<run-dir>/analysis``):
    residual_rows.tsv        one tidy row per home pair (energy gap + anatomy + gap counts/lever)
    residual_analysis.json   decomposition + label anatomy + lever addressability
                             + insertion-free check + verdict + lever_verdict
    residual_anatomy.pdf      the two-panel figure
    residual_manifest.json   provenance
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import fields
from pathlib import Path

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.dcalign_baseline import DEFAULT_EQUAL_TOL
from SBM.energy.dcalign_residual import (
    ResidualRow,
    addressability,
    analyze_record,
    anatomy,
    build_verdict,
    decompose,
    insertion_free_check,
    lever_verdict,
)
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [f.name for f in fields(ResidualRow)]


def _load_models(models_json: Path) -> list[dict]:
    data = json.loads(Path(models_json).read_text(encoding="utf-8"))
    models = data["models"]
    if len(models) != 2:
        raise ValueError(f"expected exactly two models in {models_json}, got {len(models)}")
    return models


def build_residual_rows(
    records: list[datasets.QueryRecord], models: dict, caches: dict[str, dict]
) -> list[ResidualRow]:
    """One :class:`ResidualRow` per home pair (record in its own model's frame).

    Mirrors ``compare_dcalign_baseline.build_rows``: cross-family / external
    records (no home model, or wrong length) are skipped with a logged count; a
    home record absent from its model's cache is a loud error.
    """
    rows: list[ResidualRow] = []
    skipped_no_home = skipped_wrong_len = 0
    missing: list[str] = []
    for record in records:
        model = models.get(record.origin_model)
        if model is None:
            skipped_no_home += 1
            continue
        if record.ints.size != model.L:
            log.warning("record %r origin %r length %d != L=%d; skipping",
                        record.id, model.name, record.ints.size, model.L)
            skipped_wrong_len += 1
            continue
        dca = caches[model.name].get(record.id)
        if dca is None:
            missing.append(record.id)
            continue
        rows.append(analyze_record(record, model, dca))
    if missing:
        raise ValueError(
            f"{len(missing)} home-pair sequence(s) have no DCAlign cache entry under their "
            f"own model: {missing[:5]}{' …' if len(missing) > 5 else ''}"
        )
    n_failed = sum(1 for r in rows if not r.ok)
    log.info("built %d home-pair row(s); skipped %d (no home) + %d (wrong length); %d failed",
             len(rows), skipped_no_home, skipped_wrong_len, n_failed)
    return rows


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_tsv(rows: list[ResidualRow], path: Path) -> None:
    lines = ["\t".join(_TSV_COLUMNS)]
    lines += ["\t".join(_fmt(getattr(r, c)) for c in _TSV_COLUMNS) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote residual rows: %s (%d rows)", path, len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="combine dcalign run dir (e.g. .../iter-002-nonuniform-prior)")
    parser.add_argument("--models-json", type=Path, default=None, help="default <run-dir>/models.json")
    parser.add_argument("--fasta", type=Path, default=None, help="default <run-dir>/query/query.fasta")
    parser.add_argument("--groups", type=Path, default=None, help="default <run-dir>/query/groups.json")
    parser.add_argument("--dcalign-cache", type=Path, default=None, help="default <run-dir>/dcalign/cache")
    parser.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/analysis")
    parser.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL,
                        help="|ΔE| (a.u.) below which DCAlign is judged to have recovered the frame")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    run_dir = args.run_dir
    models_json = args.models_json or run_dir / "models.json"
    fasta = args.fasta or run_dir / "query" / "query.fasta"
    groups = args.groups or run_dir / "query" / "groups.json"
    cache_dir = args.dcalign_cache or run_dir / "dcalign" / "cache"
    out_dir = args.out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_entries = _load_models(models_json)
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    caches: dict[str, dict] = {}
    for name in models:
        tsv = cache_dir / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(f"DCAlign cache for {name!r} not found at {tsv}")
        caches[name] = read_alignment_cache(tsv)
        log.info("loaded %d DCAlign alignments for model %r", len(caches[name]), name)

    records = datasets.read_query_fasta(fasta, groups)
    rows = build_residual_rows(records, models, caches)
    if not rows:
        raise ValueError("no home-pair rows produced (query has no sequences in either model's frame)")

    decomp = decompose(rows, equal_tol=args.equal_tol)
    anat = anatomy(rows, equal_tol=args.equal_tol)
    addr = addressability(rows, equal_tol=args.equal_tol)
    ins_free = insertion_free_check(rows, {m.name: m.L for m in models.values()})
    verdict = build_verdict(decomp, anat)
    lever_rec = lever_verdict(addr)

    rows_tsv = out_dir / "residual_rows.tsv"
    _write_tsv(rows, rows_tsv)
    analysis = {
        "equal_tol": args.equal_tol,
        "decomposition": decomp,
        "anatomy": anat,
        "addressability": addr,
        "insertion_free_check": ins_free,
        "verdict": verdict,
        "lever_verdict": lever_rec,
    }
    (out_dir / "residual_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n",
                                                     encoding="utf-8")
    log.info("wrote analysis JSON: %s", out_dir / "residual_analysis.json")

    from SBM.utils import utils_dcalign_residual_plot
    names = [m["name"] for m in model_entries]
    utils_dcalign_residual_plot.render_residual_anatomy(
        rows_tsv, (names[0], names[1]), out_dir / "residual_anatomy.pdf", equal_tol=args.equal_tol
    )

    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="analyze_dcalign_residual",
        command_line=provenance.current_command_line(),
        inputs={"models_json": models_json, "query_fasta": fasta, "groups": groups},
        options={"equal_tol": args.equal_tol, "dcalign_cache": str(cache_dir),
                 "models": {m["name"]: m.get("model_sha256") for m in model_entries}},
        seed=None, started_at=started_at, finished_at=finished_at, output_path=rows_tsv,
        extra={"analysis": analysis},
    )
    provenance.save_run_manifest(manifest, out_dir / "residual_manifest.json")

    print(verdict)
    print(lever_rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
