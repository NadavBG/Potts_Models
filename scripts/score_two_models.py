"""Score sequences under two fitted Potts models (spec §4 deliverable).

Computes ``E_A``, ``E_B`` and ``E_tot = w_A·E_A + w_B·E_B`` for one sequence or
a FASTA of many. Each model keeps its native length; a raw sequence is aligned
to each model independently (latent alignment collapsed by ``--method``). Both
models are loaded in the zero-sum gauge so the combined energy is well-defined.

Usage::

    # single sequence under two models
    python scripts/score_two_models.py \
        --model-a results/CM-bm-dense/iter-002-base-model/model.npy \
        --model-b results/PPIC-dense/iter-001-baseline/model.npy \
        --name-a CM --name-b PPIC \
        --seq ACDEF... --method marginal --seed 42

    # batch FASTA, writing a tidy scores table + manifest
    python scripts/score_two_models.py --model-a ... --model-b ... \
        --fasta query.fasta --groups groups.json \
        --method auto --seed 42 --output scores.tsv --manifest manifest.json

Methods: ``auto`` (in-frame for a sequence in its own model's frame, else
marginal), ``in_frame``, ``map`` (fields-MAP), ``marginal`` (IS free energy,
default), ``dcalign`` (couplings-aware; reads a precomputed on-disk alignment
cache via ``--dcalign-cache`` and recomputes the energy in-frame).
``map``/``in_frame``/``dcalign`` are deterministic; ``marginal`` requires ``--seed``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.datasets import QueryRecord
from SBM.energy.encoding import ints_to_seq, seq_to_ints, strip_gaps
from SBM.energy.hmm import ProfileHMM
from SBM.energy.model import PottsModel, load_model, load_seed_msa, seed_msa_path
from SBM.energy.score import DEFAULT_ESS_THRESHOLD, DEFAULT_N_SAMPLES, ScoreResult, score_sequence
from SBM.utils.dcalign_score import DCAlignResult, read_alignment_cache

log = logging.getLogger(__name__)

_TSV_HEADER = (
    "sequence_id\tgroup\torigin_model\tmodel\tweight\tmethod\tenergy\tess\tmc_stderr\tseed"
)


def _build_model(model_path: Path, name: str | None, seed_msa: Path | None) -> tuple[PottsModel, ProfileHMM]:
    model = load_model(model_path, name=name)
    msa = np.load(seed_msa) if seed_msa is not None else load_seed_msa(model_path)
    hmm = ProfileHMM.from_model(model, msa)
    return model, hmm


def _method_for(record: QueryRecord, model: PottsModel, method: str) -> str:
    """Resolve ``auto`` to in_frame (own frame) or marginal (cross / external)."""
    if method != "auto":
        return method
    if record.origin_model == model.name:
        if record.ints.size == model.L:
            return "in_frame"
        # Provenance mismatch: tagged as this model's native but wrong length.
        log.warning(
            "record %r claims origin %r but length %d != model L=%d; "
            "falling back to marginal re-alignment instead of in-frame",
            record.id, model.name, record.ints.size, model.L,
        )
    return "marginal"


def _score_one(
    record: QueryRecord,
    model: PottsModel,
    hmm: ProfileHMM,
    *,
    method: str,
    n_samples: int,
    seed: int | None,
    ess_threshold: float,
    dcalign_result: DCAlignResult | None = None,
) -> ScoreResult:
    resolved = _method_for(record, model, method)
    if resolved == "in_frame":
        return score_sequence(record.ints, model, method="in_frame")
    if resolved == "dcalign":
        # The expensive alignment is precomputed; we read its cached frame and
        # recompute the energy in-frame. A missing/failed alignment is a loud
        # error (caller passes None / an empty-frame result), never a silent skip.
        if dcalign_result is None:
            raise ValueError(
                f"method='dcalign' but no cached alignment for sequence {record.id!r} "
                f"under model {model.name!r}; the DCAlign align step did not cover it."
            )
        notes = (
            f"dcalign_energy={dcalign_result.dcalign_energy:.6g}; "
            f"converged={dcalign_result.converged}; "
            f"used_decimation={dcalign_result.used_decimation}; "
            f"n_iter={dcalign_result.n_iter}"
        )
        return score_sequence(
            strip_gaps(record.ints), model, method="dcalign",
            dcalign_frame=dcalign_result.aligned_frame, dcalign_notes=notes,
        )
    raw = strip_gaps(record.ints)
    return score_sequence(
        raw, model, method=resolved, hmm=hmm,
        n_samples=n_samples, seed=seed, ess_threshold=ess_threshold,
    )


def _load_dcalign_caches(
    cache_dir: Path, model_names: list[str]
) -> dict[str, dict[str, DCAlignResult]]:
    """Load each model's ``<cache_dir>/<model>/alignments.tsv`` into a dict."""
    caches: dict[str, dict[str, DCAlignResult]] = {}
    for name in model_names:
        tsv = cache_dir / name / "alignments.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(
                f"DCAlign cache for model {name!r} not found at {tsv}. Run the align "
                "step (pipeline/external/run_dcalign_align.sh) before scoring."
            )
        caches[name] = read_alignment_cache(tsv)
        log.info("loaded %d DCAlign alignments for model %r from %s", len(caches[name]), name, tsv)
    return caches


def _load_dcalign_meta(cache_dir: Path, model_name: str) -> dict | None:
    """Best-effort read of a model's cache ``meta.json`` (provenance block)."""
    meta_path = cache_dir / model_name / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read DCAlign meta %s: %s", meta_path, exc)
        return None


def _dcalign_manifest_block(rows: list[dict], args, model_A: PottsModel, model_B: PottsModel) -> dict:
    """Per-model DCAlign provenance (cache meta.json) + the agreement canary.

    ``agreement`` is max/median ``|our in-frame energy − DCAlign's own energy|``
    over the scored rows — the spike's ``≤5e-7`` invariant, now a standing canary
    that the model handoff and shared zero-sum gauge stayed correct.
    """
    block: dict = {"cache_dir": str(args.dcalign_cache), "models": {}}
    for model in (model_A, model_B):
        diffs = [
            r["_dcalign_abs_diff"] for r in rows
            if r["model"] == model.name and "_dcalign_abs_diff" in r
        ]
        agreement = None
        if diffs:
            arr = np.asarray(diffs, dtype=float)
            agreement = {
                "max_abs_diff": float(arr.max()),
                "median_abs_diff": float(np.median(arr)),
                "n": int(arr.size),
            }
        block["models"][model.name] = {
            "meta": _load_dcalign_meta(args.dcalign_cache, model.name),
            "agreement": agreement,
        }
    return block


def _records_from_args(args: argparse.Namespace) -> list[QueryRecord]:
    if args.seq is not None:
        return [QueryRecord(id="query", group="query", origin_model="", ints=seq_to_ints(args.seq))]
    records = datasets.read_query_fasta(args.fasta, args.groups)
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        # Duplicate ids would make the index→seed mapping ambiguous and collapse
        # rows in the tidy TSV; reject loudly rather than score silently.
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate sequence id(s) in {args.fasta}: {dupes[:5]}")
    return sorted(records, key=lambda r: r.id)  # stable order → reproducible seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score sequences under two Potts models.")
    parser.add_argument("--model-a", type=Path, required=True, help="path to model A's model.npy")
    parser.add_argument("--model-b", type=Path, required=True, help="path to model B's model.npy")
    parser.add_argument("--name-a", default=None, help="label for model A (default: run-dir name)")
    parser.add_argument("--name-b", default=None, help="label for model B (default: run-dir name)")
    parser.add_argument("--seed-msa-a", type=Path, default=None, help="seed MSA for A's HMM (default: A's inputs/msa.npy)")
    parser.add_argument("--seed-msa-b", type=Path, default=None, help="seed MSA for B's HMM (default: B's inputs/msa.npy)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", default=None, help="a single amino-acid sequence (ungapped or gapped)")
    src.add_argument("--fasta", type=Path, default=None, help="FASTA of sequences to score (mixed lengths allowed)")
    parser.add_argument("--groups", type=Path, default=None, help="groups.json sidecar (origin model per id)")
    parser.add_argument("--method", choices=["auto", "in_frame", "map", "marginal", "dcalign"], default="marginal")
    parser.add_argument("--dcalign-cache", type=Path, default=None,
                        help="for --method dcalign: dir holding <model>/alignments.tsv (combine/<run>/dcalign/cache)")
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0), metavar=("W_A", "W_B"))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="IS samples for marginal")
    parser.add_argument("--seed", type=int, default=None, help="master seed (required for marginal)")
    parser.add_argument("--ess-threshold", type=float, default=DEFAULT_ESS_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None, help="write tidy scores TSV here")
    parser.add_argument("--detail", type=Path, default=None, help="write per-sequence JSON here")
    parser.add_argument("--alignments", type=Path, default=None,
                        help="write a human-readable per-sequence alignments report (best frame per model + energies)")
    parser.add_argument("--manifest", type=Path, default=None, help="write provenance manifest here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    needs_seed = args.method in ("marginal", "auto")
    if needs_seed and args.seed is None:
        parser.error(f"--method {args.method} can run the marginal estimator; pass --seed for reproducibility")
    if args.method == "auto":
        log.warning(
            "method='auto' scores a sequence under its HOME model with its original MSA "
            "alignment (in_frame) but RE-ALIGNS it to the other model — so E_A and E_B "
            "come from different alignment procedures and are NOT strictly comparable. "
            "Use method='map' (or 'marginal') to align both models the same way."
        )

    started_at = dt.datetime.now(dt.timezone.utc)
    model_A, hmm_A = _build_model(args.model_a, args.name_a, args.seed_msa_a)
    model_B, hmm_B = _build_model(args.model_b, args.name_b, args.seed_msa_b)
    w_A, w_B = args.weights
    records = _records_from_args(args)

    dcalign_caches: dict[str, dict[str, DCAlignResult]] = {}
    dcalign_partial: dict[str, bool] = {}
    if args.method == "dcalign":
        if args.dcalign_cache is None:
            parser.error("--method dcalign requires --dcalign-cache DIR (combine/<run>/dcalign/cache)")
        dcalign_caches = _load_dcalign_caches(args.dcalign_cache, [model_A.name, model_B.name])
        for name in (model_A.name, model_B.name):
            meta = _load_dcalign_meta(args.dcalign_cache, name)
            dcalign_partial[name] = bool(meta and meta.get("partial"))
            if dcalign_partial[name]:
                log.warning(
                    "model %r has a PARTIAL DCAlign cache (%s of %s requested ids missing); "
                    "query sequences with no cached alignment will be SKIPPED, not scored.",
                    name, meta.get("n_missing", "?"), meta.get("n_requested", "?"),
                )

    log.info("scoring %d sequence(s) under %r (L=%d) and %r (L=%d), method=%s",
             len(records), model_A.name, model_A.L, model_B.name, model_B.L, args.method)

    rows: list[dict] = []
    detail: list[dict] = []
    skipped: list[dict] = []
    for r_idx, record in enumerate(records):
        if args.method == "dcalign":
            absent = [m.name for m in (model_A, model_B)
                      if record.id not in dcalign_caches.get(m.name, {})]
            # Tolerate an absent alignment only when every cache that lacks it
            # self-declares partial; otherwise fall through to the loud ValueError
            # in _score_one (a complete cache must cover every query id).
            if absent and all(dcalign_partial.get(name) for name in absent):
                log.warning("skipping %r: no DCAlign alignment under %s (partial cache)",
                            record.id, ", ".join(absent))
                skipped.append({"sequence_id": record.id, "group": record.group,
                                "missing_under": absent})
                continue
        per_model: dict[str, ScoreResult] = {}
        for j, (model, hmm) in enumerate(((model_A, hmm_A), (model_B, hmm_B))):
            seed_rj = None if args.seed is None else args.seed + 2 * r_idx + j
            dca_res = dcalign_caches.get(model.name, {}).get(record.id) if dcalign_caches else None
            res = _score_one(
                record, model, hmm, method=args.method,
                n_samples=args.n_samples, seed=seed_rj, ess_threshold=args.ess_threshold,
                dcalign_result=dca_res,
            )
            per_model[model.name] = res
            row = {
                "sequence_id": record.id, "group": record.group,
                "origin_model": record.origin_model, "model": model.name,
                "weight": w_A if j == 0 else w_B, "method": res.method,
                "energy": res.energy, "ess": res.ess, "mc_stderr": res.mc_stderr,
                "seed": res.seed, "_ess_threshold": args.ess_threshold,
            }
            if dca_res is not None:
                # Standing canary: our in-frame recompute vs DCAlign's own energy
                # on the same frame must agree to fp noise (the spike's ≤5e-7).
                row["_dcalign_energy"] = dca_res.dcalign_energy
                row["_dcalign_abs_diff"] = abs(res.energy - dca_res.dcalign_energy)
            rows.append(row)
        e_tot = w_A * per_model[model_A.name].energy + w_B * per_model[model_B.name].energy
        rep_A, rep_B = per_model[model_A.name], per_model[model_B.name]
        detail.append({
            "sequence_id": record.id, "group": record.group, "origin_model": record.origin_model,
            "query": ints_to_seq(strip_gaps(record.ints)),
            "E_A": rep_A.energy, "E_B": rep_B.energy, "E_tot": e_tot,
            "diagnostics": {
                model_A.name: {"method": rep_A.method, "ess": rep_A.ess,
                               "mc_stderr": rep_A.mc_stderr, "alignment": rep_A.representative_alignment},
                model_B.name: {"method": rep_B.method, "ess": rep_B.ess,
                               "mc_stderr": rep_B.mc_stderr, "alignment": rep_B.representative_alignment},
            },
        })

    finished_at = dt.datetime.now(dt.timezone.utc)
    if skipped:
        log.warning("skipped %d of %d query sequence(s) with no DCAlign alignment (partial cache)",
                    len(skipped), len(records))
    _emit(rows, detail, args, model_A, model_B, w_A, w_B, started_at, finished_at, skipped=skipped)
    return 0


def _tsv_row(r: dict) -> str:
    ess = "" if r["ess"] is None else f"{r['ess']:.6g}"
    stderr = "" if r["mc_stderr"] is None else f"{r['mc_stderr']:.6g}"
    seed = "" if r["seed"] is None else str(r["seed"])
    return (
        f"{r['sequence_id']}\t{r['group']}\t{r['origin_model']}\t{r['model']}\t"
        f"{r['weight']:g}\t{r['method']}\t{r['energy']:.10g}\t{ess}\t{stderr}\t{seed}"
    )


def _emit(rows, detail, args, model_A, model_B, w_A, w_B, started_at, finished_at, skipped=None) -> None:
    if args.output is not None:
        lines = [_TSV_HEADER] + [_tsv_row(r) for r in rows]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("wrote scores: %s", args.output)
    if args.detail is not None:
        Path(args.detail).parent.mkdir(parents=True, exist_ok=True)
        Path(args.detail).write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    if args.alignments is not None:
        _write_alignments(detail, model_A, model_B, args.alignments)
        log.info("wrote alignments report: %s", args.alignments)
    if args.manifest is not None:
        _write_manifest(rows, args, model_A, model_B, w_A, w_B, started_at, finished_at, skipped=skipped)
    # stdout summary (primary CLI output)
    for d in detail[:10]:
        print(f"{d['sequence_id']}\tE_{model_A.name}={d['E_A']:.4f}\t"
              f"E_{model_B.name}={d['E_B']:.4f}\tE_tot={d['E_tot']:.4f}")
    if len(detail) > 10:
        print(f"... ({len(detail)} sequences total; see --output / --detail)")


def _align_block(diag: dict, model_name: str) -> list[str]:
    """One model's lines in the per-sequence alignment report."""
    d = diag[model_name]
    head = f"  [{model_name}]  E={d['energy']:+.3f}  method={d['method']}"
    if d.get("ess") is not None:
        head += f"  ESS={d['ess']:.1f}"
    aln = d.get("alignment") or "(unavailable)"
    return [head, f"    {aln}  (L={len(aln) if d.get('alignment') else '?'})"]


def _write_alignments(detail: list[dict], model_A: PottsModel, model_B: PottsModel, path: Path) -> None:
    """Human-readable report: each sequence's best frame under each model, stacked.

    The two model frames are non-homologous and differ in length, so the
    alignments are stacked (not column-aligned): one line per model showing how
    the same raw query threads into that model's frame, with both energies. The
    frame shown is the alignment the reported energy uses (in-frame sequence,
    Viterbi/fields-MAP path, or the dominant importance-sampling frame).
    """
    lines = [
        "# Per-sequence best alignment under each model (frames are independent;",
        "# different lengths, NOT column-aligned). Gap = '-'. Energies in a.u.,",
        f"# zero-sum gauge. E_tot = w_A*E_{model_A.name} + w_B*E_{model_B.name}.",
        "",
    ]
    for d in detail:
        diag = {
            model_A.name: {**d["diagnostics"][model_A.name], "energy": d["E_A"]},
            model_B.name: {**d["diagnostics"][model_B.name], "energy": d["E_B"]},
        }
        lines.append(f"### {d['sequence_id']}   group={d['group']}   E_tot={d['E_tot']:+.3f}")
        lines.append(f"  query (N={len(d['query'])}): {d['query']}")
        lines += _align_block(diag, model_A.name)
        lines += _align_block(diag, model_B.name)
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(rows, args, model_A, model_B, w_A, w_B, started_at, finished_at, skipped=None) -> None:
    ess_A = [r["ess"] for r in rows if r["model"] == model_A.name and r["ess"] is not None]
    ess_B = [r["ess"] for r in rows if r["model"] == model_B.name and r["ess"] is not None]

    def _ess_stats(vals):
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"min": float(arr.min()), "median": float(np.median(arr)),
                "low_count": int((arr < args.ess_threshold).sum()), "n": int(arr.size)}

    extra: dict = {
        "n_rows": len(rows),
        "ess_summary": {model_A.name: _ess_stats(ess_A), model_B.name: _ess_stats(ess_B)},
    }
    if skipped:
        extra["n_skipped_no_alignment"] = len(skipped)
        extra["skipped_no_alignment"] = skipped[:50]
    if args.method == "dcalign":
        extra["dcalign"] = _dcalign_manifest_block(rows, args, model_A, model_B)

    manifest = provenance.build_run_manifest(
        run_id="score_two_models",
        command_line=provenance.current_command_line(),
        inputs={
            "model_a": args.model_a, "model_b": args.model_b,
            "seed_msa_a": args.seed_msa_a or seed_msa_path(args.model_a),
            "seed_msa_b": args.seed_msa_b or seed_msa_path(args.model_b),
            "query_fasta": args.fasta,
        },
        options={
            "method": args.method, "n_samples": args.n_samples, "weights": [w_A, w_B],
            "ess_threshold": args.ess_threshold,
            "model_a_name": model_A.name, "model_b_name": model_B.name,
            "gauge": model_A.gauge,
        },
        seed=args.seed,
        started_at=started_at,
        finished_at=finished_at,
        output_path=args.output,
        extra=extra,
    )
    provenance.save_run_manifest(manifest, args.manifest)
    log.info("wrote manifest: %s", args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
