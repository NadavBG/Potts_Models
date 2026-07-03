#!/usr/bin/env bash
# Gather two-model design shards into the run outputs (docs/DESIGN_TWO_MODEL.md).
# Submitted by run_design.sh with --dependency=afterok on the shard array, so it
# runs once every shard COMPLETED. Merges every <run_root>/design/shards/shard_*.jsonl
# into trajectories.npz + designed_sequences.fasta + designed.tsv +
# design_aln_{A,B}.fasta + design_manifest.json + gather_status.json. Two gates
# (in the wrapper): every planned chain present, and the warm-started polish never
# worse than the joint-MC frame (E_polish <= E_mc).
#
# Usage (set by the driver):
#   sbatch --dependency=afterok:<arrayid> sbatch_design_gather.sh <run_root>

#SBATCH --job-name=design_gather
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/design_gather_%j.log
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
# from design_config.json (for the sha256 + the polish-canary recompute).
cd "${REPO_DIR}"

export OMP_NUM_THREADS=1
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

python "${REPO_DIR}/scripts/wf/run_design_gather.py" --run-dir "${RUN_ROOT}/design"
echo "[design_gather] done"
