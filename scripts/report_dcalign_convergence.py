"""DCAlign convergence report: how many sequences failed to converge, per group.

DCAlign either converges or falls back to its decimation/nucleation path; the
cache records both per sequence. This report counts non-convergence per
(model, group) over **every** cached alignment — a sequence's home model *and*
the cross-family model — since cross-family frames are where convergence is most
at risk and the in-frame baseline (``compare_dcalign_baseline.py``) does not cover
them. It pairs with that baseline to answer "are the worse-than-native alignments
un-converged, or converged-on-a-bad-frame?": non-convergence is rare on home pairs,
so most worse-than-native energies are converged.

Usage::

    python scripts/report_dcalign_convergence.py \
        --models-json combine/<run>/iter-.../models.json \
        --groups combine/<run>/iter-.../query/groups.json \
        --dcalign-cache combine/<run>/iter-.../dcalign/cache \
        --output dcalign_convergence.tsv --summary dcalign_convergence.json \
        --figure figs/dcalign_convergence.pdf --manifest manifest.json

Deterministic: reads the on-disk cache only (no Julia, no sampling).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import SBM.provenance as provenance
from SBM.energy.dcalign_baseline import convergence_by_group, summarize_convergence
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [
    "model", "group", "n", "n_converged", "n_not_converged",
    "n_decimation", "n_failed", "frac_not_converged",
]


def _load_models(models_json: Path) -> list[dict]:
    data = json.loads(Path(models_json).read_text(encoding="utf-8"))
    models = data["models"]
    if len(models) != 2:
        raise ValueError(f"expected exactly two models in {models_json}, got {len(models)}")
    return models


def _load_meta(cache_dir: Path, model_name: str) -> dict | None:
    meta_path = cache_dir / model_name / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read DCAlign meta %s: %s", meta_path, exc)
        return None


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_tsv(rows: list[dict], path: Path) -> None:
    lines = ["\t".join(_TSV_COLUMNS)]
    lines += ["\t".join(_fmt(r[c]) for c in _TSV_COLUMNS) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote convergence table: %s (%d rows)", path, len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-json", type=Path, required=True, help="combine run's models.json")
    parser.add_argument("--groups", type=Path, required=True, help="groups.json (group per seq id)")
    parser.add_argument("--dcalign-cache", type=Path, required=True,
                        help="dir holding <model>/alignments.tsv (combine/<run>/dcalign/cache)")
    parser.add_argument("--output", type=Path, required=True, help="tidy convergence TSV")
    parser.add_argument("--summary", type=Path, default=None, help="summary JSON (overall + per model)")
    parser.add_argument("--manifest", type=Path, default=None, help="provenance manifest JSON")
    parser.add_argument("--figure", type=Path, default=None, help="non-convergence bar figure PDF (optional)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    model_entries = _load_models(args.models_json)
    names = [m["name"] for m in model_entries]
    caches: dict[str, dict] = {}
    for name in names:
        tsv = args.dcalign_cache / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(
                f"DCAlign cache for model {name!r} not found at {tsv}; run the align step "
                "(pipeline/external/run_dcalign_align.sh) before the convergence report."
            )
        caches[name] = read_alignment_cache(tsv)
        log.info("loaded %d DCAlign alignments for model %r", len(caches[name]), name)

    groups = json.loads(Path(args.groups).read_text(encoding="utf-8"))
    rows = convergence_by_group(caches, groups)
    summary = summarize_convergence(rows)

    _write_tsv(rows, args.output)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log.info("wrote convergence summary: %s", args.summary)
    if args.figure is not None:
        from SBM.utils import utils_dcalign_convergence_plot
        utils_dcalign_convergence_plot.render_dcalign_convergence(
            args.output, (names[0], names[1]), args.figure
        )
    if args.manifest is not None:
        _write_manifest(args, model_entries, summary, started_at)

    for name, st in summary["by_model"].items():
        print(f"{name}: {st['n_not_converged']}/{st['n']} not converged "
              f"({st['frac_not_converged']:.1%}); decimation={st['n_decimation']} failed={st['n_failed']}")
    return 0


def _write_manifest(args, model_entries, summary, started_at) -> None:
    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="report_dcalign_convergence",
        command_line=provenance.current_command_line(),
        inputs={"models_json": args.models_json, "groups": args.groups},
        options={
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
