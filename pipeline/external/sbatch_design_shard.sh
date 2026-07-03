#!/usr/bin/env bash
# One two-model design shard: anneal this shard's chains over E_tot
# (docs/DESIGN_TWO_MODEL.md). Submitted as an array job by run_design.sh — not
# invoked by hand. Array task t IS the shard; the chain indices it owns are read
# from <run_root>/design/shards_manifest.json (round-robin, written by `plan`).
# Pure numpy — no Julia.
#
# Usage (set by the driver):
#   sbatch --array=0-<N-1>%<conc> --time=<sized> sbatch_design_shard.sh <run_root>
#
# Compute nodes have NO outbound network, so this does no git. It resumes: chains
# already present in shards/shard_<NNN>.jsonl are skipped (the wrapper flushes one
# line per chain, so a TIME_LIMIT kill leaves a valid partial shard and a re-submit
# continues). The driver sizes --time from chains_per_shard x (steps + polish); the
# 04:00:00 default below is a floor for the by-hand case and resume makes a
# TIME_LIMIT non-fatal regardless.

#SBATCH --job-name=design_shard
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The anneal + warm-started polish is a single-core numpy loop; fan out over the
# array, do NOT request many cpus (extra threads only compete for the one core).
#SBATCH --cpus-per-task=1
# Two models (J ~ (L,L,q,q) f8 ~33MB each) + the polish PT arrays are well under
# 1G; 2G is ample headroom.
#SBATCH --mem=2G
# Overridden by the driver (--time=<sized> from chains_per_shard x per-chain cost);
# this is the by-hand default. Resume makes a TIME_LIMIT kill non-fatal.
#SBATCH --time=04:00:00
#SBATCH --output=logs/design_shard_%A_%a.log
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
# Run from the repo root: design_config.json stores repo-root-relative model +
# seed_msa paths, resolved by load_model() from cwd.
cd "${REPO_DIR}"

# Keep numpy single-core (the anneal loop is serial Python; billed per task, so
# extra threads only compete). No Julia, no DCALIGN_PATH — pure numpy.
export OMP_NUM_THREADS=1
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

echo "[design_shard] task=${SLURM_ARRAY_TASK_ID} run_root=${RUN_ROOT} cpus=${SLURM_CPUS_PER_TASK:-1}"
python "${REPO_DIR}/scripts/wf/run_design_shard.py" run \
    --run-dir "${RUN_ROOT}/design" \
    --shard "${SLURM_ARRAY_TASK_ID}"
echo "[design_shard] done"
