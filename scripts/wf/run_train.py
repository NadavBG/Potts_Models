"""Snakemake wrapper: train the Potts model into the run dir."""

import json
import os
from pathlib import Path

from _common import load_cfg_from_snakemake, setup_stage_logging

setup_stage_logging(snakemake, "train")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

# Pin OpenMP threads BEFORE importing the C++ MCMC kernel (it reads
# OMP_NUM_THREADS at load time); recorded in the training manifest. With
# a fixed seed + thread count, the trained arrays are bit-identical.
if cfg.omp_num_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(cfg.omp_num_threads)

import train_sbm  # noqa: E402 (imported after OMP_NUM_THREADS is set)

run_root = snakemake.params.run_root  # noqa: F821
# Mask paths are None when pruning is disabled (the rule still lists them
# as inputs for DAG ordering only when they exist).
prune_J = snakemake.params.prune_J  # noqa: F821
prune_h = snakemake.params.prune_h  # noqa: F821

train_sbm.run_SBM(
    Input_MSA=cfg.msa,
    fam=cfg.family or cfg.run_name,
    Model=cfg.train.mode,
    train_file=None,
    N_iter=cfg.train.N_iter,
    m=cfg.train.m,
    N_chains_list=[cfg.train.N_chains],
    Nb_rep=1,
    Nb_av=1,
    k_MCMC=cfg.train.k_MCMC,
    TestTrain=cfg.train.TestTrain,
    ParamInit="Zero",
    lambdJ=cfg.train.lambda_J,
    lambdh=cfg.train.lambda_h,
    theta=cfg.train.theta,
    ignore_gaps=cfg.train.ignore_gaps,
    prune_J_file=prune_J,
    prune_h_file=prune_h,
    results_path=None,
    seed=cfg.seed,
    label=cfg.run_name,
    optimizer=cfg.train.optimizer,
    record_every=cfg.train.record_every,
    explicit_run_dir=run_root,
)

# Small per-stage meta consumed by the aggregate run_manifest rule.
Path(snakemake.output.meta).write_text(  # noqa: F821
    json.dumps(
        {
            "stage": "train",
            "run_root": str(run_root),
            "mode": cfg.train.mode,
            "N_chains": cfg.train.N_chains,
            "N_iter": cfg.train.N_iter,
            "seed": cfg.seed,
            "pruned_J": prune_J,
            "pruned_h": prune_h,
        },
        indent=2,
    ),
    encoding="utf-8",
)
