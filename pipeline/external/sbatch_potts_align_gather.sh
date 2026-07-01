#!/usr/bin/env bash
# Gather potts_align shards into one alignments.tsv per model (iter-003).
# Submitted by run_potts_align_align.sh with --dependency=afterok on the shard
# array, so it runs once every shard COMPLETED. Errors loudly if any in-scope
# pair is missing (an incomplete align run) or if the in-frame canary fails.
#
# Usage (set by the driver):
#   sbatch --dependency=afterok:<arrayid> sbatch_potts_align_gather.sh <run_root>

#SBATCH --job-name=potts_align_gather
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/potts_align_gather_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$1"
REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
# Run from the repo root: gather load_model()s the repo-root-relative model paths
# (for the sha256 + the ΔE-gate / canary recompute). RUN_ROOT is passed absolute.
cd "${REPO_DIR}"

export OMP_NUM_THREADS=1
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

python "${REPO_DIR}/scripts/wf/run_potts_align_gather.py" --run-root "${RUN_ROOT}"
echo "[potts_align_gather] done"
