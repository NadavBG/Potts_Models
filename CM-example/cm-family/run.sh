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

# The MSA enters as an aligned FASTA; encode it into the integer .npy that
# run_sbm.sh consumes (the FASTA is the source of truth, the array derived).
MSA_TMP_DIR="$(mktemp -d -t cm_example_msa.XXXXXX)"
trap 'rm -rf "${MSA_TMP_DIR}"' EXIT
python "${REPO_ROOT}/scripts/encode_msa.py" \
    --fasta "${REPO_ROOT}/data/fasta/CM.fasta" \
    --out "${MSA_TMP_DIR}/MSA_CM.npy"

bash "${REPO_ROOT}/scripts/run_sbm.sh" \
    SBM \
    "${MSA_TMP_DIR}/MSA_CM.npy" \
    --label CM-example \
    "$@"
