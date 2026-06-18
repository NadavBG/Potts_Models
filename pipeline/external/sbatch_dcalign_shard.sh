#!/bin/bash
# One DCAlign align shard for one model (spec §10.9). Submitted as an array job
# by run_dcalign_align.sh — not invoked by hand. Array task t maps to
# model_index = t // n_shards, shard = t % n_shards.
#
# Usage (set by the driver):
#   sbatch --array=0-<2N-1>%<conc> sbatch_dcalign_shard.sh <run_root> <n_shards>
#
# Compute nodes have NO outbound network, so this does no git. It resumes: ids
# already present in the shard TSV are skipped (the Julia driver flushes per row,
# so a TIME_LIMIT kill leaves a valid partial cache and a re-submit continues).

#SBATCH --job-name=dcalign_shard
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# 8G (not 4): lambda_spec="deltan" builds the seed dist array (~1.8 GB for PPIC) and
# N~=L inflates palign — measured per-task peak ~4.5 GB at cpus=2 (iter-004 smoke,
# 2026-06-18), so the old 4G cap OOM'd. 8G = ~1.75x headroom; DCALIGN_MEM overrides.
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dcalign_shard_%A_%a.log
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <run_root> <n_shards>" >&2
    exit 2
fi
RUN_ROOT="$1"
N_SHARDS="$2"
: "${SLURM_ARRAY_TASK_ID:?must run as a Slurm array job}"

REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
# Run from the repo root: models.json stores repo-root-relative model paths
# (e.g. results/<fam>/<iter>/model.npy) but the driver submits from RUN_ROOT/dcalign,
# so load_model would otherwise fail to find the model. RUN_ROOT is passed absolute.
cd "${REPO_DIR}"

module load julia/1.10.2
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/midway3/nadavbg/julia_depot}"
export DCALIGN_PATH="${DCALIGN_PATH:-$(realpath "${REPO_DIR}/../DCAlign")}"
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % N_SHARDS ))
echo "[dcalign_shard] task=${SLURM_ARRAY_TASK_ID} -> model_index=${MODEL_IDX} shard=${SHARD} (n_shards=${N_SHARDS})"
echo "[dcalign_shard] DCALIGN_PATH=${DCALIGN_PATH} cpus=${SLURM_CPUS_PER_TASK:-1}"

python "${REPO_DIR}/scripts/wf/run_dcalign_shard.py" run \
    --run-root "${RUN_ROOT}" \
    --model-index "${MODEL_IDX}" \
    --shard "${SHARD}" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"

echo "[dcalign_shard] done"
