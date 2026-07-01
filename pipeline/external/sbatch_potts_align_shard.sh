#!/usr/bin/env bash
# One potts_align shard (iter-003). Submitted as an array job by
# run_potts_align_align.sh — not invoked by hand. FLAT layout: array task t IS
# the shard (each task loads both models; the shard's rows carry the model), so
# there is no model_index//n_shards arithmetic. Pure numpy — no Julia.
#
# Usage (set by the driver):
#   sbatch --array=0-<N-1>%<conc> sbatch_potts_align_shard.sh <run_root>
#
# Compute nodes have NO outbound network, so this does no git. It resumes: pairs
# already present in the shard TSV are skipped (the wrapper flushes per row, so a
# TIME_LIMIT kill leaves a valid partial cache and a re-submit continues).

#SBATCH --job-name=potts_align_shard
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
# PT is a single-core Python loop; fan out over the array, do NOT request many cpus.
#SBATCH --cpus-per-task=1
# J is (L,L,q,q) f8 ~33MB/model x2; enumerate g<=3 materializes <=110MB; the
# compute_energies chunk is ~0.3GB. 2G is ~6x headroom.
#SBATCH --mem=2G
# Measured ~74s per g=5 PT pair on a caslake core; ~192 shards -> ~1h/shard.
# 3h gives ample margin (and resume makes a TIME_LIMIT non-fatal anyway).
#SBATCH --time=03:00:00
#SBATCH --output=logs/potts_align_shard_%A_%a.log
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$1"
: "${SLURM_ARRAY_TASK_ID:?must run as a Slurm array job}"

REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
# Run from the repo root: models.json stores repo-root-relative model paths.
cd "${REPO_DIR}"

# No module load julia, no DCALIGN_PATH — pure numpy. Keep numpy single-core
# (PT is a serial Python loop; billed per task, so extra threads only compete).
export OMP_NUM_THREADS=1
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

echo "[potts_align_shard] task=${SLURM_ARRAY_TASK_ID} run_root=${RUN_ROOT} cpus=${SLURM_CPUS_PER_TASK:-1}"
python "${REPO_DIR}/scripts/wf/run_potts_align_shard.py" run \
    --run-root "${RUN_ROOT}" \
    --shard "${SLURM_ARRAY_TASK_ID}"
echo "[potts_align_shard] done"
