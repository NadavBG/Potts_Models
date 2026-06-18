"""Baseline: DCAlign's best-attempt energy vs the native in-frame energy.

For each query sequence in **its own model's frame** (a "home pair"), compare the
native in-frame Potts energy against the energy DCAlign found by re-aligning the
raw residues (read from the on-disk DCAlign cache, spec §10.9). The point is a
quantitative baseline for the flat-prior aligner *before* the informed-prior
tuning (spec §10.8 Blocker 1, §10.9 phase 2): a couplings-aware aligner should
never score an in-frame native worse than the native frame, and this report says
how often it does.

``delta_e = E_dcalign − E_inframe`` (lower energy = better), so ``delta_e > 0``
means DCAlign did worse than the trivial native frame. Cross-family pairs have no
in-frame reference (different length) and are skipped — see
:mod:`SBM.energy.dcalign_baseline`.

Usage::

    python scripts/compare_dcalign_baseline.py \
        --models-json combine/<run>/iter-.../models.json \
        --fasta combine/<run>/iter-.../query/query.fasta \
        --groups combine/<run>/iter-.../query/groups.json \
        --dcalign-cache combine/<run>/iter-.../dcalign/cache \
        --output dcalign_vs_inframe.tsv --summary dcalign_vs_inframe.json \
        --figure figs/dcalign_vs_inframe.pdf --manifest manifest_baseline.json

Deterministic (no sampling): in-frame energies and the cached DCAlign frames are
fixed, so the same inputs give the same numbers.
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
from SBM.energy.dcalign_baseline import DEFAULT_EQUAL_TOL, BaselineRow, compare_record, summarize
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [f.name for f in fields(BaselineRow)]


def _load_models(models_json: Path) -> list[dict]:
    data = json.loads(Path(models_json).read_text(encoding="utf-8"))
    models = data["models"]
    if len(models) != 2:
        raise ValueError(f"expected exactly two models in {models_json}, got {len(models)}")
    return models


def _load_meta(cache_dir: Path, model_name: str) -> dict | None:
    """Best-effort read of a model's DCAlign cache ``meta.json`` (provenance)."""
    meta_path = cache_dir / model_name / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read DCAlign meta %s: %s", meta_path, exc)
        return None


def build_rows(
    records: list[datasets.QueryRecord],
    models: dict,
    caches: dict[str, dict],
) -> list[BaselineRow]:
    """One :class:`BaselineRow` per home pair (record in its own model's frame).

    ``models`` maps name → loaded :class:`PottsModel`; ``caches`` maps name →
    ``{seq_id: DCAlignResult}``. Cross-family / external records (no home model,
    or wrong length) are skipped with a logged count; a home record absent from
    its model's cache is a loud error (the align step did not cover it).
    """
    rows: list[BaselineRow] = []
    skipped_no_home = skipped_wrong_len = 0
    missing: list[str] = []
    for record in records:
        model = models.get(record.origin_model)
        if model is None:  # external FASTA / no native frame
            skipped_no_home += 1
            continue
        if record.ints.size != model.L:
            log.warning("record %r claims origin %r but length %d != L=%d; skipping",
                        record.id, model.name, record.ints.size, model.L)
            skipped_wrong_len += 1
            continue
        dca = caches[model.name].get(record.id)
        if dca is None:
            missing.append(record.id)
            continue
        rows.append(compare_record(record, model, dca))
    if missing:
        raise ValueError(
            f"{len(missing)} home-pair sequence(s) have no DCAlign cache entry under their "
            f"own model (the align step did not cover them): {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}"
        )
    n_failed = sum(1 for r in rows if not r.ok)
    log.info(
        "built %d home-pair comparison(s); skipped %d (no home model) + %d (wrong length); "
        "%d DCAlign failures (empty frame, kept as ok=false)",
        len(rows), skipped_no_home, skipped_wrong_len, n_failed,
    )
    if n_failed:
        log.warning("%d sequence(s) failed DCAlign alignment (NaN energies in the table)", n_failed)
    return rows


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_tsv(rows: list[BaselineRow], path: Path) -> None:
    lines = ["\t".join(_TSV_COLUMNS)]
    lines += ["\t".join(_fmt(getattr(r, c)) for c in _TSV_COLUMNS) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote baseline table: %s (%d rows)", path, len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-json", type=Path, required=True, help="combine run's models.json")
    parser.add_argument("--fasta", type=Path, required=True, help="query.fasta (frames in origin-model coords)")
    parser.add_argument("--groups", type=Path, required=True, help="groups.json (origin model per id)")
    parser.add_argument("--dcalign-cache", type=Path, required=True,
                        help="dir holding <model>/alignments.tsv (combine/<run>/dcalign/cache)")
    parser.add_argument("--output", type=Path, required=True, help="tidy baseline TSV")
    parser.add_argument("--summary", type=Path, default=None, help="summary JSON (ΔE stats per model/group)")
    parser.add_argument("--manifest", type=Path, default=None, help="provenance manifest JSON")
    parser.add_argument("--figure", type=Path, default=None, help="consolidated baseline PDF (optional)")
    parser.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL,
                        help="|ΔE| (a.u.) below which DCAlign is judged to have recovered the native frame")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    model_entries = _load_models(args.models_json)
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    caches: dict[str, dict] = {}
    for name in models:
        tsv = args.dcalign_cache / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(
                f"DCAlign cache for model {name!r} not found at {tsv}; run the align step "
                "(pipeline/external/run_dcalign_align.sh) before the baseline comparison."
            )
        caches[name] = read_alignment_cache(tsv)
        log.info("loaded %d DCAlign alignments for model %r", len(caches[name]), name)

    records = datasets.read_query_fasta(args.fasta, args.groups)
    rows = build_rows(records, models, caches)
    if not rows:
        raise ValueError(
            "no home-pair comparisons produced — the query has no sequences in either "
            "model's frame (an external FASTA with no origin frame has no in-frame reference)."
        )

    summary = summarize(rows, equal_tol=args.equal_tol)
    _write_tsv(rows, args.output)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log.info("wrote baseline summary: %s", args.summary)
    if args.figure is not None:
        from SBM.utils import utils_dcalign_baseline_plot
        names = [m["name"] for m in model_entries]
        utils_dcalign_baseline_plot.render_dcalign_baseline(
            args.output, (names[0], names[1]), args.figure, equal_tol=args.equal_tol
        )
    if args.manifest is not None:
        _write_manifest(args, model_entries, summary, started_at)

    ov = summary["overall"]
    if ov.get("n_ok"):
        print(f"home pairs: {ov['n_ok']}  worse-than-native (ΔE>{args.equal_tol:g}): "
              f"{ov['n_worse']} ({ov['frac_worse']:.1%})  median ΔE={ov['delta_e']['median']:+.2f}  "
              f"cache canary max|Δ|={ov['cache_max_abs_diff']:.2g}")
    return 0


def _write_manifest(args, model_entries, summary, started_at) -> None:
    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="compare_dcalign_baseline",
        command_line=provenance.current_command_line(),
        inputs={
            "models_json": args.models_json,
            "query_fasta": args.fasta,
            "groups": args.groups,
        },
        options={
            "equal_tol": args.equal_tol,
            "dcalign_cache": str(args.dcalign_cache),
            "models": {m["name"]: m.get("model_sha256") for m in model_entries},
            "dcalign_meta": {m["name"]: _load_meta(args.dcalign_cache, m["name"]) for m in model_entries},
        },
        seed=None,
        started_at=started_at,
        finished_at=finished_at,
        output_path=args.output,
        extra={"summary": summary},
    )
    provenance.save_run_manifest(manifest, args.manifest)
    log.info("wrote manifest: %s", args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
