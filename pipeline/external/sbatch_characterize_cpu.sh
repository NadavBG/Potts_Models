#!/usr/bin/env bash
# CPU side of design characterization on caslake: gather fold shards, TM-align
# every model vs 1ECM/1JNT (multiprocessed), BLAST the designs, merge -> summary
# + report + figures + manifest. Submitted by run_characterize.sh with an
# afterok dependency on the GPU fold arrays. Runs in the repo .venv (numpy /
# matplotlib / lab_plotting); BLAST binaries come from CM_env (paths baked into
# blast_sequences.py).
#
# Usage (set by the driver):
#   sbatch sbatch_characterize_cpu.sh <run_dir> <natural_cache_a> <natural_cache_b>

#SBATCH --job-name=characterize_cpu
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/characterize_cpu_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <run_dir> <natural_cache_a> <natural_cache_b>" >&2
    exit 2
fi
RUN_DIR="$1"; CACHE_A="$2"; CACHE_B="$3"

# SLURM runs the batch script from /var/spool/slurm; derive the repo from the
# (absolute, in-repo) run_dir argument's git toplevel, not ${BASH_SOURCE[0]}.
REPO_ROOT="$(git -C "${RUN_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${REPO_ROOT}" && -d "${REPO_ROOT}/scripts/characterize" ]] || {
    echo "ERROR: cannot locate repo root from run_dir ${RUN_DIR}" >&2; exit 3; }
VENV="${REPO_ROOT}/.venv"
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: no venv at ${VENV}" >&2; exit 3; }
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

exec python "${REPO_ROOT}/scripts/characterize/characterize.py" \
    --run-dir "${RUN_DIR}" \
    --natural-cache-a "${CACHE_A}" \
    --natural-cache-b "${CACHE_B}" \
    --jobs "${SLURM_CPUS_PER_TASK:-16}"
