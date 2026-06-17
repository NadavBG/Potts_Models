#!/bin/bash
# Gather DCAlign shards into one alignments.tsv per model (spec §10.9).
# Submitted by run_dcalign_align.sh with --dependency=afterok on the shard array,
# so it runs once every shard COMPLETED. Errors loudly if any requested id is
# missing from the shards (an incomplete align run).
#
# Usage (set by the driver):
#   sbatch --dependency=afterok:<arrayid> sbatch_dcalign_gather.sh <run_root>

#SBATCH --job-name=dcalign_gather
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dcalign_gather_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$1"
REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
# Run from the repo root: gather load_model()s the repo-root-relative model paths in
# models.json to stamp model_sha256, but is submitted from RUN_ROOT/dcalign. RUN_ROOT
# is passed absolute, so the git -C above resolves regardless of CWD.
cd "${REPO_DIR}"

# julia + DCAlign clone are loaded so gather can stamp the cache meta.json with
# the DCAlign commit + julia version (provenance); the merge itself is pure Python.
module load julia/1.10.2
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/midway3/nadavbg/julia_depot}"
export DCALIGN_PATH="${DCALIGN_PATH:-$(realpath "${REPO_DIR}/../DCAlign")}"
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

python "${REPO_DIR}/scripts/wf/run_dcalign_gather.py" --run-root "${RUN_ROOT}"
echo "[dcalign_gather] done"
