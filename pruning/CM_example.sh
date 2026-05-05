#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: build a pruning mask for the CM family, then train a
# BM positive control that respects it. Output lands at
#   results/CM/<YYYY-MM-DD>_CM-bm-pruned_<idx>/
# with model.npy, manifest.json, command.sh, fig_data/, and figs/.
#
# This is the BM ("positive control") regime from Summary Note 3:
# m=20, lambda_J=lambda_h=0.01, N_chains=100. run_sbm.sh applies these
# defaults automatically when MODE=BM, so no manual overrides are needed.
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

# 2. Train a BM model that uses the SCA-derived mask. The BM defaults
#    in run_sbm.sh (m=20, lambda=0.01, N_chains=100) are appropriate
#    when most couplings are constrained to zero.
bash "${REPO_ROOT}/scripts/run_sbm.sh" \
    BM "${FULL_CM_ALG}" \
    --prune "$(pwd)/prune_output/98.00_SCA_CM_SeqW_0.7.npy" \
    --label CM-bm-pruned \
    --results-path "$(pwd)/example_output"
