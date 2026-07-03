#!/usr/bin/env python3
"""Orchestrate the CPU side of design characterization. CPU stage (.venv).

Consumes the GPU fold outputs (design ``characterize/structures/`` + the per-MSA
``natural_folds/`` caches) and produces the deliverables:

  * gather per-shard fold scores,
  * TM-align every model vs 1ECM (A) and 1JNT (B)  [compare_structures.py],
  * BLAST the designs vs SwissProt + CM/PPIC families  [blast_sequences.py],
  * merge fold + structure + BLAST + design energies -> ``data/summary.tsv`` and
    ``data/natural_summary.tsv`` + ``report.md`` + ``provenance/manifest.json``,
  * figures  [render_characterize.py].

Designed to hang off sequence generation: point ``--run-dir`` at a built combine
iteration dir and it finds ``design/designed_sequences.fasta``, ``design/designed.tsv``
and ``models.json`` by convention.

    .venv/bin/python scripts/characterize/characterize.py \
        --run-dir combine/combine-profiles/iter-001-profile-eval \
        --natural-cache-a results/CM-bm-profile/iter-001-profile/natural_folds/<sha8> \
        --natural-cache-b results/PPIC-profile/iter-001-no-couplings/natural_folds/<sha8> \
        --jobs 16
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

import SBM.provenance as provenance
from SBM.characterize import blast, summary

logger = logging.getLogger("characterize")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _gather_fold_scores(shard_dir: Path, out_tsv: Path) -> int:
    """Concatenate per-shard ``fold_scores/*.tsv`` into one (dedup by id)."""
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for tsv in sorted(shard_dir.glob("*.tsv")):
        with open(tsv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = reader.fieldnames or header
            for row in reader:
                rows[row["id"]] = row
    if header is None:
        raise SystemExit(f"no fold-score shards in {shard_dir}")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for _id, row in sorted(rows.items()):
            writer.writerow(row)
    return len(rows)


def _run(cmd: list[str]) -> None:
    logger.info("+ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="built combine iteration dir")
    p.add_argument("--design-fasta", type=Path, default=None)
    p.add_argument("--designed-tsv", type=Path, default=None)
    p.add_argument("--natural-cache-a", type=Path, default=None,
                   help="natural_folds/<sha8> dir for model A (CM); omit for designs-only")
    p.add_argument("--natural-cache-b", type=Path, default=None,
                   help="natural_folds/<sha8> dir for model B (PPIC); omit for designs-only")
    p.add_argument("--ref-a", type=Path, default=_REPO_ROOT / "data/structures/1ECM.pdb")
    p.add_argument("--chain-a", default="A")
    p.add_argument("--ref-b", type=Path, default=_REPO_ROOT / "data/structures/1JNT.pdb")
    p.add_argument("--chain-b", default="A")
    p.add_argument("--tmalign", type=Path, default=_REPO_ROOT / "pipeline/bin/TMalign")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--skip-blast", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = args.run_dir.resolve()
    out = run_dir / "characterize"
    data = out / "data"
    figs = out / "figs"
    design_fasta = args.design_fasta or (run_dir / "design/designed_sequences.fasta")
    designed_tsv = args.designed_tsv or (run_dir / "design/designed.tsv")
    here = Path(__file__).resolve().parent

    has_naturals = args.natural_cache_a is not None and args.natural_cache_b is not None

    # 1. Gather fold scores (design + optionally naturals).
    design_scores = data / "fold_scores.tsv"
    n_design = _gather_fold_scores(out / "structures" / "fold_scores", design_scores)
    natural_scores = data / "natural_fold_scores.tsv"
    n_natural = 0
    if has_naturals:
        tmp_rows: dict[str, dict[str, str]] = {}
        header = None
        for cache in (args.natural_cache_a, args.natural_cache_b):
            for tsv in sorted((cache / "fold_scores").glob("*.tsv")):
                with open(tsv, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    header = reader.fieldnames or header
                    for row in reader:
                        tmp_rows[row["id"]] = row
        natural_scores.parent.mkdir(parents=True, exist_ok=True)
        with open(natural_scores, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
            writer.writeheader()
            for _id, row in sorted(tmp_rows.items()):
                writer.writerow(row)
        n_natural = len(tmp_rows)
    logger.info("gathered %d design + %d natural fold scores%s", n_design, n_natural,
                "" if has_naturals else " (designs-only: no natural controls)")

    # 2. TM-align (design + optionally naturals) vs both references.
    compare_tsv = data / "structure_compare.tsv"
    fold_scores_args = [str(design_scores)] + ([str(natural_scores)] if has_naturals else [])
    struct_dirs = [str(out / "structures")]
    if has_naturals:
        struct_dirs += [str(args.natural_cache_a / "structures"),
                        str(args.natural_cache_b / "structures")]
    _run([sys.executable, str(here / "compare_structures.py"),
          "--fold-scores", *fold_scores_args,
          "--structures-dir", *struct_dirs,
          "--ref-a", str(args.ref_a), "--chain-a", args.chain_a,
          "--ref-b", str(args.ref_b), "--chain-b", args.chain_b,
          "--tmalign", str(args.tmalign), "--out", str(compare_tsv),
          "--jobs", str(args.jobs)])

    # 3. BLAST designs (separate SwissProt / CM / PPIC).
    blast_dir = data / "blast"
    if not args.skip_blast:
        _run([sys.executable, str(here / "blast_sequences.py"),
              "--fasta", str(design_fasta), "--out-dir", str(blast_dir),
              "--threads", str(args.jobs)])

    # 4. Merge -> summary.tsv + natural_summary.tsv + report.md.
    compare_by_id = {r["id"]: r for r in summary.read_tsv(compare_tsv)}
    fold_design = summary.read_tsv(design_scores)
    fold_natural = summary.read_tsv(natural_scores) if has_naturals else []
    energies = summary.load_design_energies(designed_tsv)

    sp = cm = pp = {}
    annotations: dict[str, str] = {}
    if not args.skip_blast:
        sp_hits = blast.parse_blast_tsv(blast_dir / "blast_swissprot.tsv")
        cm_hits = blast.parse_blast_tsv(blast_dir / "blast_cmfam.tsv")
        pp_hits = blast.parse_blast_tsv(blast_dir / "blast_ppicfam.tsv")
        sp = {q: blast.best_hit(sp_hits, q) for q in sp_hits}
        cm = {q: blast.best_hit(cm_hits, q) for q in cm_hits}
        pp = {q: blast.best_hit(pp_hits, q) for q in pp_hits}
        annotations = blast.load_swissprot_annotations(
            "/project/ranganathanr/nadavbg/BioM3/reference_data/fully_annotated_swiss_prot.csv"
        )

    design_rows = summary.build_summary_rows(
        fold_design, compare_by_id, energies=energies,
        swissprot=sp, cmfam=cm, ppicfam=pp, annotations=annotations)
    natural_rows = summary.build_summary_rows(fold_natural, compare_by_id)

    summary.write_tsv(design_rows, summary.DESIGN_COLUMNS, data / "summary.tsv")
    if has_naturals:
        summary.write_tsv(natural_rows, summary.NATURAL_COLUMNS, data / "natural_summary.tsv")
    summary.write_report(design_rows, natural_rows, out / "report.md",
                         meta={"run_dir": str(run_dir),
                               "n_designs": str(len(design_rows)),
                               "n_naturals": str(len(natural_rows))})
    logger.info("wrote summary.tsv (%d), natural_summary.tsv (%d), report.md",
                len(design_rows), len(natural_rows))

    # 5. Figures.
    if not args.skip_render:
        render_cmd = [sys.executable, str(here / "render_characterize.py"),
                      "--summary", str(data / "summary.tsv"), "--figs-dir", str(figs)]
        if has_naturals:
            render_cmd += ["--natural-summary", str(data / "natural_summary.tsv")]
        _run(render_cmd)

    # 6. Provenance manifest.
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id=provenance.make_run_id(label="characterize"),
        command_line=provenance.current_command_line(),
        inputs={"design_fasta": design_fasta, "designed_tsv": designed_tsv,
                "ref_a": args.ref_a, "ref_b": args.ref_b,
                "design_fold_scores": design_scores, "natural_fold_scores": natural_scores},
        options={"chain_a": args.chain_a, "chain_b": args.chain_b,
                 "tmalign": str(args.tmalign), "jobs": args.jobs,
                 "skip_blast": args.skip_blast,
                 "natural_cache_a": str(args.natural_cache_a),
                 "natural_cache_b": str(args.natural_cache_b)},
        seed=None, started_at=started, finished_at=finished,
        extra={"n_designs": len(design_rows), "n_naturals": len(natural_rows)})
    provenance.save_run_manifest(manifest, out / "provenance/manifest.json")
    logger.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
