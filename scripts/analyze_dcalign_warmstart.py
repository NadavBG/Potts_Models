"""Analyse the DCAlign warm-start fixed-point probe (iter-003 Phase-B, §10.x).

Run AFTER the Midway warm-start driver writes ``<run-dir>/<model>/warmstart_out.tsv``
(one per model) and the cache is pulled back. Compares, per curated home pair, the
frame BP settled on *starting from native* against the native frame and against the
production (random-init) frame from the source run, classifies each
(:mod:`SBM.energy.dcalign_warmstart`), and writes the case-A/B verdict.

    python scripts/analyze_dcalign_warmstart.py \
        --run-dir combine/combine-CM-PPIC-dcalign-warmstart

Outputs (under ``--out-dir``, default ``<run-dir>/analysis``):
    warmstart_rows.tsv        one tidy row per home pair (3 energies + agreements + label)
    warmstart_analysis.json   per-role/kind tallies + verdict
    warmstart.pdf             the two-panel figure
    warmstart_manifest.json   provenance
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
from SBM.energy.dcalign_warmstart import (
    WarmstartRow,
    analyze_warmstart_record,
    summarize_warmstart,
)
from SBM.energy.model import load_model
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

_TSV_COLUMNS = [f.name for f in fields(WarmstartRow)]


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_tsv(rows: list[WarmstartRow], path: Path) -> None:
    lines = ["\t".join(_TSV_COLUMNS)]
    lines += ["\t".join(_fmt(getattr(r, c)) for c in _TSV_COLUMNS) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote warm-start rows: %s (%d rows)", path, len(rows))


def build_rows(run_dir: Path, src_run_dir: Path, equal_tol: float) -> list[WarmstartRow]:
    """One :class:`WarmstartRow` per curated home pair (native frames from the source)."""
    roles = json.loads((run_dir / "roles.json").read_text(encoding="utf-8"))
    model_entries = json.loads((src_run_dir / "models.json").read_text(encoding="utf-8"))["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}

    records = datasets.read_query_fasta(
        src_run_dir / "query" / "query.fasta", src_run_dir / "query" / "groups.json")
    by_id = {r.id: r for r in records}

    # Only models that actually own curated home pairs need a warm-start cache.
    needed = {by_id[sid].origin_model for sid in roles
              if sid in by_id and by_id[sid].origin_model in models}
    warm_caches, rand_caches = {}, {}
    for name in sorted(needed):
        warm_tsv = run_dir / name / "warmstart_out.tsv"
        if not warm_tsv.is_file():
            raise FileNotFoundError(
                f"warm-start cache for {name!r} not found at {warm_tsv} "
                "(run the Midway warm-start driver first)")
        warm_caches[name] = read_alignment_cache(warm_tsv)
        rand_tsv = src_run_dir / "dcalign" / "cache" / name / "alignments.tsv"
        rand_caches[name] = read_alignment_cache(rand_tsv) if rand_tsv.is_file() else {}

    rows, missing = [], []
    for sid, role in roles.items():
        record = by_id.get(sid)
        if record is None or record.origin_model not in models:
            continue
        model = models[record.origin_model]
        warm = warm_caches[model.name].get(sid)
        if warm is None:
            missing.append(sid)
            continue
        rand = rand_caches.get(model.name, {}).get(sid)
        rows.append(analyze_warmstart_record(record, model, role, warm, rand, equal_tol=equal_tol))
    if missing:
        raise ValueError(
            f"{len(missing)} curated id(s) have no warm-start cache entry: "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}")
    log.info("built %d warm-start row(s); %d failed", len(rows), sum(1 for r in rows if not r.ok))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="warm-start probe run dir")
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"),
                   help="source of native frames + models + random-init cache (default iter-002)")
    p.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/analysis")
    p.add_argument("--equal-tol", type=float, default=DEFAULT_EQUAL_TOL)
    p.add_argument("--init-kind", default="native", choices=("native", "map"),
                   help="how BP was initialised (only sets the verdict wording); match the build")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(run_dir, args.src_run_dir, args.equal_tol)
    if not rows:
        raise ValueError("no warm-start rows produced (no curated home pairs matched)")
    summary = summarize_warmstart(rows, equal_tol=args.equal_tol, init_kind=args.init_kind)

    rows_tsv = out_dir / "warmstart_rows.tsv"
    _write_tsv(rows, rows_tsv)
    (out_dir / "warmstart_analysis.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    log.info("wrote analysis JSON: %s", out_dir / "warmstart_analysis.json")

    from SBM.utils import utils_dcalign_warmstart_plot
    utils_dcalign_warmstart_plot.render_warmstart(
        rows_tsv, out_dir / "warmstart.pdf", equal_tol=args.equal_tol)

    finished_at = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id="analyze_dcalign_warmstart", command_line=provenance.current_command_line(),
        inputs={"run_dir": run_dir, "src_run_dir": args.src_run_dir},
        options={"equal_tol": args.equal_tol}, seed=None,
        started_at=started_at, finished_at=finished_at, output_path=rows_tsv,
        extra={"analysis": summary},
    )
    provenance.save_run_manifest(manifest, out_dir / "warmstart_manifest.json")

    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
