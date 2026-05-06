#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: train an SBM model on the chorismate mutase (CM) family
# and render its figures. Output lands at
#   results/CM/<YYYY-MM-DD>_CM-example_<idx>/
# with model.npy, manifest.json, command.sh, fig_data/, and figs/.
#
# Run from anywhere; this thin wrapper dispatches to scripts/run_sbm.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"

bash "${REPO_ROOT}/scripts/run_sbm.sh" \
    SBM \
    "${REPO_ROOT}/data/MSA_array/MSA_CM.npy" \
    --label CM-example \
    "$@"
