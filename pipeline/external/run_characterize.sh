#!/usr/bin/env bash
# Login-node driver for design characterization (fold + structure + BLAST).
# Submits three ESMFold GPU array jobs — designs, CM naturals, PPIC naturals —
# then a caslake CPU job (afterok on all three) that TM-aligns, BLASTs, merges,
# and renders. Run on a Midway LOGIN node AFTER the one-time prep:
#   bash pipeline/external/build_tmalign.sh
#   bash pipeline/external/prefetch_esmfold.sh
#
# Usage (login node):
#   bash pipeline/external/run_characterize.sh <run_dir>
#
#   <run_dir>  a built combine iteration dir with design/designed_sequences.fasta,
#              design/designed.tsv, and models.json (e.g.
#              combine/combine-profiles/iter-001-profile-eval).
#
# Env knobs (defaults in brackets):
#   ESMFOLD_MAX_CONCURRENT [8]  GPUs used at once per array (beagle3 cap is 32)
#   N_SHARDS_DESIGN   [1]        shards for the 96 designs
#   N_SHARDS_CM       [4]        shards for the ~1.3k CM naturals
#   N_SHARDS_PPIC     [64]       shards for the ~27k PPIC naturals
#   CM_FASTA   [data/fasta/CM.fasta]      PPIC_FASTA [data/fasta/ppic_msa.fasta]
set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_dir>" >&2
    exit 2
fi
RUN_DIR="$(realpath "$1")"
[[ -d "${RUN_DIR}" ]] || { echo "ERROR: run_dir not found: ${RUN_DIR}" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHARD_JOB="${SCRIPT_DIR}/sbatch_esmfold_shard.sh"
CPU_JOB="${SCRIPT_DIR}/sbatch_characterize_cpu.sh"
VENV="${REPO_ROOT}/.venv"

CONC="${ESMFOLD_MAX_CONCURRENT:-8}"
N_DESIGN="${N_SHARDS_DESIGN:-1}"
N_CM="${N_SHARDS_CM:-4}"
N_PPIC="${N_SHARDS_PPIC:-64}"
CM_FASTA="${CM_FASTA:-${REPO_ROOT}/data/fasta/CM.fasta}"
PPIC_FASTA="${PPIC_FASTA:-${REPO_ROOT}/data/fasta/ppic_msa.fasta}"

DESIGN_FASTA="${RUN_DIR}/design/designed_sequences.fasta"
for f in "${DESIGN_FASTA}" "${RUN_DIR}/design/designed.tsv" "${RUN_DIR}/models.json" \
         "${CM_FASTA}" "${PPIC_FASTA}" "${TMALIGN:=${REPO_ROOT}/pipeline/bin/TMalign}"; do
    [[ -e "${f}" ]] || { echo "FATAL: missing ${f}" >&2; exit 5; }
done
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: no venv at ${VENV}" >&2; exit 3; }

# Resolve model run dirs + the per-MSA cache dirs (keyed by source-FASTA sha8).
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
# NB: the script-level IFS=$'\n\t' strips space from IFS, so `read` would not
# split the space-separated python output — force space-splitting for this read.
IFS=' ' read -r RUN_A RUN_B SHA_CM SHA_PPIC < <(python - "${RUN_DIR}/models.json" "${CM_FASTA}" "${PPIC_FASTA}" <<'PY'
import json, sys
from SBM.provenance import file_sha256
models = json.load(open(sys.argv[1]))["models"]
run_a = next(m["run_dir"] for m in models if "CM" in m["name"])
run_b = next(m["run_dir"] for m in models if "PPIC" in m["name"])
print(run_a, run_b, file_sha256(sys.argv[2])[:8], file_sha256(sys.argv[3])[:8])
PY
)
CACHE_A="${REPO_ROOT}/${RUN_A}/natural_folds/${SHA_CM}"
CACHE_B="${REPO_ROOT}/${RUN_B}/natural_folds/${SHA_PPIC}"
echo "model A (CM)   run_dir=${RUN_A}  cache=${CACHE_A}"
echo "model B (PPIC) run_dir=${RUN_B}  cache=${CACHE_B}"

# Submit from characterize/ so #SBATCH --output=logs/... lands there.
CHAR_DIR="${RUN_DIR}/characterize"
mkdir -p "${CHAR_DIR}/logs" "${CHAR_DIR}/structures/fold_scores" \
         "${CACHE_A}/structures" "${CACHE_A}/fold_scores" \
         "${CACHE_B}/structures" "${CACHE_B}/fold_scores"
cd "${CHAR_DIR}"

submit_fold () {   # name fasta out_struct scores_dir group n_shards degap
    local name="$1" fasta="$2" out_struct="$3" scores_dir="$4" group="$5" n="$6" degap="$7"
    local amax=$(( n - 1 ))
    sbatch --parsable --array=0-${amax}%${CONC} --job-name="esmfold_${name}" \
        "${SHARD_JOB}" "${fasta}" "${out_struct}" "${scores_dir}" "${group}" "${n}" "${degap}"
}

echo
echo "submitting ESMFold fold arrays (concurrency ${CONC}/array):"
JID_DESIGN=$(submit_fold design "${DESIGN_FASTA}" "${CHAR_DIR}/structures" \
    "${CHAR_DIR}/structures/fold_scores" design "${N_DESIGN}" 0)
echo "  designs      : ${JID_DESIGN}  (${N_DESIGN} shards)"
JID_CM=$(submit_fold cm "${CM_FASTA}" "${CACHE_A}/structures" \
    "${CACHE_A}/fold_scores" CM-natural "${N_CM}" 1)
echo "  CM naturals  : ${JID_CM}  (${N_CM} shards)"
JID_PPIC=$(submit_fold ppic "${PPIC_FASTA}" "${CACHE_B}/structures" \
    "${CACHE_B}/fold_scores" PPIC-natural "${N_PPIC}" 1)
echo "  PPIC naturals: ${JID_PPIC}  (${N_PPIC} shards)"

CPU_JID=$(sbatch --parsable \
    --dependency=afterok:"${JID_DESIGN}":"${JID_CM}":"${JID_PPIC}" \
    "${CPU_JOB}" "${RUN_DIR}" "${CACHE_A}" "${CACHE_B}")
echo "  CPU merge    : ${CPU_JID}  (after all fold arrays)"

printf '%s\n%s\n%s\n%s\n' "${JID_DESIGN}" "${JID_CM}" "${JID_PPIC}" "${CPU_JID}" \
    > "${CHAR_DIR}/.char_jids"

echo
echo "monitor:  squeue -u \$USER    |    pipeline/job_tally.sh -w 10"
echo "outputs land in ${CHAR_DIR}/{data,figs} + report.md when CPU merge mails END."
