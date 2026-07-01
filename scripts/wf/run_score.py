"""Snakemake wrapper: score the query set under both models.

Reads ``models.json`` (from resolve_models) for the two model paths / names /
weights / seed-MSA paths, then drives ``scripts/score_two_models.py`` to write a
tidy ``scores.tsv``, a per-sequence ``scores_detail.json``, and a provenance
``manifest.json``. The master seed flows through so marginal scoring is
reproducible (per-sequence seeds are derived deterministically downstream).
"""

import json
import os
from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

setup_stage_logging(snakemake, "score")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821

if cfg.omp_num_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(cfg.omp_num_threads)

import score_two_models  # noqa: E402 (after OMP_NUM_THREADS, like run_train.py)

models = json.loads(Path(snakemake.input.models).read_text(encoding="utf-8"))["models"]  # noqa: F821
model_A, model_B = models[0], models[1]

argv = [
    "--model-a", model_A["model_path"], "--model-b", model_B["model_path"],
    "--name-a", model_A["name"], "--name-b", model_B["name"],
    "--fasta", str(snakemake.input.fasta),  # noqa: F821
    "--groups", str(snakemake.input.groups),  # noqa: F821
    "--method", cfg.scoring.method,
    "--weights", str(model_A["weight"]), str(model_B["weight"]),
    "--n-samples", str(cfg.scoring.n_samples),
    "--seed", str(cfg.seed),
    "--ess-threshold", str(cfg.scoring.ess_threshold),
    "--output", str(snakemake.output.scores),  # noqa: F821
    "--detail", str(snakemake.output.detail),  # noqa: F821
    "--alignments", str(snakemake.output.alignments),  # noqa: F821
    "--manifest", str(snakemake.output.manifest),  # noqa: F821
]
if model_A.get("seed_msa"):
    argv += ["--seed-msa-a", model_A["seed_msa"]]
if model_B.get("seed_msa"):
    argv += ["--seed-msa-b", model_B["seed_msa"]]

# DCAlign reads a precomputed alignment cache built by the sbatch align step
# (pipeline/external/run_dcalign_align.sh) under <run_root>/dcalign/cache.
if cfg.scoring.method == "dcalign":
    run_root = snakemake.params.run_root  # noqa: F821
    argv += ["--dcalign-cache", str(Path(run_root) / "dcalign" / "cache")]

# potts_align likewise reads a cluster-built cache under <run_root>/potts_align/cache
# (pipeline/external/run_potts_align_align.sh); the score step recomputes each
# energy in-frame as a canary and writes the tidy scores.tsv.
if cfg.scoring.method == "potts_align":
    run_root = snakemake.params.run_root  # noqa: F821
    argv += ["--potts-align-cache", str(Path(run_root) / "potts_align" / "cache")]

rc = score_two_models.main(argv)
if rc:
    raise SystemExit(rc)
