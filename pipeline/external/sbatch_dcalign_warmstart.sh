#!/bin/bash
# DCAlign warm-start fixed-point probe (spec §10.17). ONE array task per model;
# each task runs that model's whole in-dir (12 home pairs) through our warm-start
# driver run_dcalign_warmstart.jl, which warm-starts BP at the native frame.
#
# Fan-out is over the 2-model ARRAY: task 0 = models[0] (CM), task 1 = models[1]
# (PPIC) run concurrently on separate nodes. Within a task the queries are threaded
# across cpus-per-task; the driver deepcopies J/h/Λ per sequence and pins BLAS to 1,
# so this mirrors the production deltan-align memory profile exactly.
#
# Usage (from the Midway LOGIN node, repo root):
#   mkdir -p logs
#   sbatch pipeline/external/sbatch_dcalign_warmstart.sh \
#       combine/combine-CM-PPIC-dcalign-warmstart
# (the --array range is baked in below: exactly 2 models.)

#SBATCH --job-name=dcalign_warmstart
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=caslake
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Memory is the OOM trap: lambda_spec="deltan" builds the seed dist array (~1.8 GB
# for PPIC) and palign keeps per-thread J/h/Λ copies, so peak scales with cpus.
# Production align measured ~4.5 GB at cpus=2 and runs safely at cpus=4/8G; 12G here
# adds ~1.7x headroom over that for the warm-start (same structures). Edit if needed.
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dcalign_warmstart_%A_%a.log
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail

# The pinned, UNMODIFIED DCAlign clone (spec §10.17). The warm-start lives in our
# driver; the clone is a read-only dependency that MUST match the Mac byte-for-byte.
PINNED_COMMIT=cab443ffad133e6e68eff8e50b11e8fc59178dbd

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>   (e.g. combine/combine-CM-PPIC-dcalign-warmstart)" >&2
    exit 2
fi
RUN_ROOT="$(realpath "$1")"
: "${SLURM_ARRAY_TASK_ID:?must run as a Slurm array job (sbatch ...)}"

REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
# Run from the repo root: models.json stores repo-root-relative model paths.
cd "${REPO_DIR}"

module load julia/1.10.2
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/midway3/nadavbg/julia_depot}"
export DCALIGN_PATH="${DCALIGN_PATH:-$(realpath "${REPO_DIR}/../DCAlign")}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"

# Enforce the clone pin loudly (no silent run against a drifted/dirty clone).
HEAD_COMMIT="$(git -C "${DCALIGN_PATH}" rev-parse HEAD)"
if [[ "${HEAD_COMMIT}" != "${PINNED_COMMIT}" ]]; then
    echo "[warmstart] FATAL: DCAlign clone ${DCALIGN_PATH} is at ${HEAD_COMMIT}," >&2
    echo "           expected pinned ${PINNED_COMMIT}. Check out the pin and resubmit." >&2
    exit 3
fi
if [[ -n "$(git -C "${DCALIGN_PATH}" status --porcelain)" ]]; then
    echo "[warmstart] FATAL: DCAlign clone has local modifications; it must be pristine." >&2
    exit 3
fi

# Map array task -> model name (models.json order; matches the build staging).
MODEL="$(python -c "import json; print(json.load(open('${RUN_ROOT}/models.json'))['models'][${SLURM_ARRAY_TASK_ID}]['name'])")"
IN_DIR="${RUN_ROOT}/${MODEL}"
OUT_TSV="${IN_DIR}/warmstart_out.tsv"
[[ -d "${IN_DIR}" ]] || { echo "[warmstart] FATAL: in-dir ${IN_DIR} missing (run build_dcalign_warmstart.py + push)" >&2; exit 4; }

# The driver appends and does not de-dup ids; start each submission clean so a
# re-run can't produce duplicate rows (read_alignment_cache raises on dups). The
# job is short (~10 min) and --time is generous, so resume-on-TIME_LIMIT is moot.
rm -f "${OUT_TSV}"

echo "[warmstart] task=${SLURM_ARRAY_TASK_ID} model=${MODEL} threads=${JULIA_NUM_THREADS} clone=${HEAD_COMMIT}"
echo "[warmstart] in_dir=${IN_DIR}"

julia --project="${DCALIGN_PATH}" "${REPO_DIR}/src/SBM/julia/run_dcalign_warmstart.jl" \
    "${IN_DIR}" "${OUT_TSV}"

echo "[warmstart] done -> ${OUT_TSV} ($(grep -c . "${OUT_TSV}" 2>/dev/null || echo 0) lines)"
