#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: train an SBM model on the chorismate mutase (CM) family.
# Run from the repo root after installing the package (`pip install -e .`).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${REPO_ROOT}"

python scripts/train_sbm.py CM "${REPO_ROOT}/data/MSA_array/MSA_CM.npy" \
    --TestTrain 0 \
    --m 1 \
    --rep 1 \
    --N_av 1 \
    --N_iter 400 \
    --theta 0.3 \
    --ParamInit zero \
    --lambdJ 0 \
    --lambdh 0 \
    --N_chains 70 \
    --seed 42
