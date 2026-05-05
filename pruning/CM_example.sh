#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: build a pruning mask for the CM family, then train an SBM
# model that respects it. Run from this directory after installing the
# package and the optional `sca` extra (`pip install -e ".[sca]"`).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
FULL_CM_ALG="${REPO_ROOT}/data/MSA_array/MSA_CM.npy"

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Build a pruning mask from the full alignment.
python build_mask.py \
    --alg "${FULL_CM_ALG}" \
    --theta 0.7 \
    --lbda 0.03 \
    --strategies "fij" "cij" "sca" \
    --ext ".npy" \
    --label "CM" \
    --path "./prune_output" \
    --percent 98

# 2. Train an SBM model that uses the mask.
python "${REPO_ROOT}/scripts/train_sbm.py" SCAPruned_CM "${FULL_CM_ALG}" \
    --TestTrain 0 \
    --m 20 \
    --rep 1 \
    --N_av 1 \
    --N_iter 400 \
    --theta 0.3 \
    --ParamInit zero \
    --lambdJ 0.01 \
    --lambdh 0.01 \
    --N_chains 100 \
    --prune "./prune_output/98.00_SCA_CM_SeqW_0.7.npy" \
    --results_path "./example_output/" \
    --seed 42
