"""Ground-state recovery: potts_align's minimum energy vs the native in-frame energy.

For each query sequence in **its own model's frame** (a "home pair"), compare the
native in-frame Potts energy against the global (or PT) minimum ``potts_align``
found by re-placing the raw residues over the ``L`` columns (read from the on-disk
potts_align cluster cache, docs/POTTS_ALIGN.md). Because the native gap placement
is itself one of the frames the minimizer searches, its minimum can never be
higher than the native energy — so this report answers "does potts_align recover
the ground state, and how often is the native frame already it?".

``delta_e = E_potts_align − E_inframe`` (lower energy = better):
  ``delta_e ≈ 0`` ⇒ native is at the ground state (recovered),
  ``delta_e < 0`` ⇒ the aligner found a strictly lower frame (native isn't the
  ground state), ``delta_e > 0`` ⇒ potts_align did *worse* than native — a PT/SA
  search failure (impossible for the enumerate engine). Cross-family pairs have no
  in-frame reference (different length) and are skipped — see
  :mod:`SBM.energy.potts_align_baseline`.

Usage::

    python scripts/compare_potts_align_baseline.py \
        --models-json combine/<run>/iter-.../models.json \
        --fasta combine/<run>/iter-.../query/query.fasta \
        --groups combine/<run>/iter-.../query/groups.json \
        --potts-align-cache combine/<run>/iter-.../potts_align/cache \
        --output potts_align_vs_inframe.tsv --summary potts_align_vs_inframe.json \
        --figure figs/potts_align_vs_inframe.pdf --manifest manifest_baseline.json

Deterministic (no sampling): in-frame energies and the cached frames are fixed,
so the same inputs give the same numbers.
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
from SBM.energy.model import load_model
from SBM.energy.potts_align_baseline import (
    DEFAULT_EQUAL_TOL,
    PottsAlignBaselineRow,
    rows_for_home_pairs,
    summarize,
)
from SBM.utils.potts_align_cache import read_potts_align_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [f.name for f in fields(PottsAlignBaselineRow)]


def _load_models(models_json: Path) -> list[dict]:
    data = json.loads(Path(models_json).read_text(encoding="utf-8"))
    models = data["models"]
    if len(models) != 2:
        raise ValueError(f"expected exactly two models in {models_json}, got {len(models)}")
    return models


def _load_meta(cache_dir: Path, model_name: str) -> dict | None:
    """Best-effort read of a model's potts_align cache ``meta.json`` (provenance)."""
    meta_path = cache_dir / model_name / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read potts_align meta %s: %s", meta_path, exc)
        return None


def build_rows(
    records: list[datasets.QueryRecord],
    models: dict,
    caches: dict[str, dict],
) -> list[PottsAlignBaselineRow]:
    """One :class:`PottsAlignBaselineRow` per home pair (record in its own model's frame).

    ``models`` maps name → loaded :class:`PottsModel`; ``caches`` maps name →
    ``{seq_id: PottsAlignCacheResult}``. Cross-family / external records (no home
    model, or wrong length) are skipped with a logged count; a home record absent
    from its model's cache is a loud error (the align step did not cover it).
    Energies are computed per model in two batched calls for speed.
    """
    per_model: dict[str, list[tuple[datasets.QueryRecord, object]]] = {}
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
        res = caches[model.name].get(record.id)
        if res is None:
            missing.append(record.id)
            continue
        per_model.setdefault(model.name, []).append((record, res))
    if missing:
        raise ValueError(
            f"{len(missing)} home-pair sequence(s) have no potts_align cache entry under "
            f"their own model (the align step did not cover them): {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}"
        )
    rows: list[PottsAlignBaselineRow] = []
    for name, pairs in per_model.items():
        rows.extend(rows_for_home_pairs(models[name], pairs))
    n_failed = sum(1 for r in rows if not r.ok)
    log.info(
        "built %d home-pair comparison(s); skipped %d (no home model) + %d (wrong length); "
        "%d potts_align skip rows (empty frame, kept as ok=false)",
        len(rows), skipped_no_home, skipped_wrong_len, n_failed,
    )
    if n_failed:
        log.warning("%d home pair(s) had no potts_align frame (NaN energies in the table)", n_failed)
    return rows


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_tsv(rows: list[PottsAlignBaselineRow], path: Path) -> None:
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
    parser.add_argument("--potts-align-cache", type=Path, required=True,
                        help="dir holding <model>/alignments.tsv (combine/<run>/potts_align/cache)")
    parser.add_argument("--output", type=Path, required=True, help="tidy baseline TSV")
    parser.add_argument("--summary", type=Path, default=None, help="summary JSON (ΔE stats per model/group)")
    parser.add_argument("--manifest", type=Path, default=None, help="provenance manifest JSON")
    parser.add_argument("--figure", type=Path, default=None, help="consolidated baseline PDF (optional)")
    parser.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL,
                        help="|ΔE| (a.u.) below which the native is judged to sit at the ground state")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    model_entries = _load_models(args.models_json)
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    caches: dict[str, dict] = {}
    for name in models:
        tsv = args.potts_align_cache / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(
                f"potts_align cache for model {name!r} not found at {tsv}; run the align step "
                "(pipeline/external/run_potts_align_align.sh) before the baseline comparison."
            )
        caches[name] = read_potts_align_cache(tsv)
        log.info("loaded %d potts_align alignments for model %r", len(caches[name]), name)

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
        from SBM.utils import utils_potts_align_baseline_plot
        names = [m["name"] for m in model_entries]
        utils_potts_align_baseline_plot.render_potts_align_baseline(
            args.output, (names[0], names[1]), args.figure, equal_tol=args.equal_tol
        )
    if args.manifest is not None:
        _write_manifest(args, model_entries, summary, started_at)

    ov = summary["overall"]
    if ov.get("n_ok"):
        print(f"home pairs: {ov['n_ok']}  at ground state (|ΔE|<={args.equal_tol:g}): "
              f"{ov['n_at_ground']} ({ov['frac_at_ground']:.1%})  improved (ΔE<-tol): "
              f"{ov['n_improved']}  worse (ΔE>tol): {ov['n_worse']}  median ΔE="
              f"{ov['delta_e']['median']:+.3g}  cache canary max|Δ|={ov['cache_max_abs_diff']:.2g}")
        if ov["n_worse"]:
            log.warning("%d home pair(s) scored WORSE than native (ΔE>%g) — a PT/SA search "
                        "failure, not a modelling result", ov["n_worse"], args.equal_tol)
    return 0


def _write_manifest(args, model_entries, summary, started_at) -> None:
    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="compare_potts_align_baseline",
        command_line=provenance.current_command_line(),
        inputs={
            "models_json": args.models_json,
            "query_fasta": args.fasta,
            "groups": args.groups,
        },
        options={
            "equal_tol": args.equal_tol,
            "potts_align_cache": str(args.potts_align_cache),
            "models": {m["name"]: m.get("model_sha256") for m in model_entries},
            "potts_align_meta": {m["name"]: _load_meta(args.potts_align_cache, m["name"]) for m in model_entries},
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
