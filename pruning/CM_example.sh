#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: build a pruning mask for the CM family, then train an
# SBM model that respects it. Output lands at
#   results/CM/<YYYY-MM-DD>_CM-pruned_<idx>/
# with model.npy, manifest.json, command.sh, fig_data/, and figs/.
#
# Requires the [sca] optional extra (`pip install -e ".[sca]"`).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
FULL_CM_ALG="${REPO_ROOT}/data/MSA_array/MSA_CM.npy"

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Build a pruning mask from the full alignment (98% of couplings zeroed).
python build_mask.py \
    --alg "${FULL_CM_ALG}" \
    --theta 0.7 \
    --lbda 0.03 \
    --strategies "fij" "cij" "sca" \
    --ext ".npy" \
    --label "CM" \
    --path "./prune_output" \
    --percent 98

# 2. Train an SBM model that uses the SCA-derived mask. Higher-rank Hessian
#    (m=20) and small L2 regularization (1e-2 each on J and h) are
#    appropriate when most couplings are constrained to zero.
bash "${REPO_ROOT}/scripts/run_sbm.sh" \
    SBM "${FULL_CM_ALG}" \
    --prune "$(pwd)/prune_output/98.00_SCA_CM_SeqW_0.7.npy" \
    --label CM-pruned \
    --results-path "$(pwd)/example_output" \
    -- --m 20 --lambdJ 0.01 --lambdh 0.01 --N_chains 100
