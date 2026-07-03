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
        --natural-cache-a natural_folds/<cm_sha8> \
        --natural-cache-b natural_folds/<ppic_sha8> \
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
from SBM.characterize import blast, natural_cache, summary

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


def _run_compare(
    here: Path, args: argparse.Namespace,
    fold_scores: list[Path], structures_dirs: list[Path],
    out: Path, work_dir: Path,
) -> None:
    """Invoke compare_structures.py for one (fold-score set, structures) group."""
    _run([sys.executable, str(here / "compare_structures.py"),
          "--fold-scores", *[str(f) for f in fold_scores],
          "--structures-dir", *[str(d) for d in structures_dirs],
          "--ref-a", str(args.ref_a), "--chain-a", args.chain_a,
          "--ref-b", str(args.ref_b), "--chain-b", args.chain_b,
          "--tmalign", str(args.tmalign), "--out", str(out),
          "--work-dir", str(work_dir), "--jobs", str(args.jobs)])


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
    p.add_argument("--force-natural-tm", action="store_true",
                   help="recompute the natural TM-align even if the per-model "
                        "tm_vs_refs cache already covers it")
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

    # 2. TM-align vs both references. Designs are new every run (compute fresh);
    #    the ~28k naturals are a pure function of (natural PDB, reference pair) so
    #    they are cached content-addressed at natural_folds/<sha8>/tm_vs_refs/<refkey>.tsv
    #    (top-level tree, keyed by source-FASTA sha8) and reused across combine runs
    #    (natural_cache). The final structure_compare.tsv is the union the downstream
    #    merge already expects.
    compare_tsv = data / "structure_compare.tsv"
    ref_chains = data / "ref_chains"  # shared single-chain reference extraction

    design_compare = data / "design_compare.tsv"
    _run_compare(here, args, [design_scores], [out / "structures"],
                 design_compare, ref_chains)

    compare_sources = [design_compare]
    natural_tm: dict[str, str] = {}  # family -> "hit" | "miss" (provenance)
    refkey = ""
    if has_naturals:
        sha_a = provenance.file_sha256(args.ref_a)
        sha_b = provenance.file_sha256(args.ref_b)
        refkey = natural_cache.ref_pair_key(sha_a, args.chain_a, sha_b, args.chain_b)
        for family, cache in (("A", args.natural_cache_a), ("B", args.natural_cache_b)):
            tsv = natural_cache.cache_tsv(cache, refkey)
            required = natural_cache.ids_in_fold_scores(cache)
            if not args.force_natural_tm and natural_cache.cache_covers(tsv, required):
                logger.info("natural TM cache HIT  (%s, %d naturals) -> %s",
                            family, len(required), tsv)
                natural_tm[family] = "hit"
            else:
                why = "forced" if args.force_natural_tm else "miss"
                logger.info("natural TM cache %s (%s, %d naturals): TM-aligning -> %s",
                            why, family, len(required), tsv)
                shards = sorted((cache / "fold_scores").glob("*.tsv"))
                _run_compare(here, args, shards, [cache / "structures"],
                             tsv, ref_chains)
                natural_cache.write_meta(
                    natural_cache.cache_meta(cache, refkey), refkey=refkey,
                    ref_a=args.ref_a, ref_a_sha256=sha_a, chain_a=args.chain_a,
                    ref_b=args.ref_b, ref_b_sha256=sha_b, chain_b=args.chain_b,
                    tmalign=args.tmalign, n_rows=len(required),
                    source_sha8=Path(cache).name)
                natural_tm[family] = "miss"
            compare_sources.append(tsv)

    n_compare = natural_cache.merge_compare_tsvs(compare_sources, compare_tsv)
    logger.info("structure_compare.tsv: %d rows (%d source tables)%s",
                n_compare, len(compare_sources),
                f", natural TM {natural_tm}" if has_naturals else "")

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
                 "force_natural_tm": args.force_natural_tm,
                 "natural_cache_a": str(args.natural_cache_a),
                 "natural_cache_b": str(args.natural_cache_b)},
        seed=None, started_at=started, finished_at=finished,
        extra={"n_designs": len(design_rows), "n_naturals": len(natural_rows),
               "natural_tm_refkey": refkey, "natural_tm_cache": natural_tm})
    provenance.save_run_manifest(manifest, out / "provenance/manifest.json")
    logger.info("done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
