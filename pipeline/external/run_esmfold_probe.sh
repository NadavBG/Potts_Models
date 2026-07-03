#!/usr/bin/env bash
# 20-sequence ESMFold timing + SU probe on one beagle3 GPU. Run on a login node
# AFTER prefetch_esmfold.sh. Brackets a single fold shard (--limit 20) with
# `accounts balance` so the measured s/seq and the actual SU charge can be read
# off before committing the full ~28k fold. Reuses sbatch_esmfold_shard.sh with
# a limit (7th positional arg); writes to a throwaway probe dir.
#
# Usage (login node):
#   bash pipeline/external/run_esmfold_probe.sh <run_dir> [<n_probe>]
set -euo pipefail
IFS=$'\n\t'

RUN_DIR="$(realpath "${1:?usage: $0 <run_dir> [<n_probe>]}")"
N_PROBE="${2:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARD_JOB="${SCRIPT_DIR}/sbatch_esmfold_shard.sh"
DESIGN_FASTA="${RUN_DIR}/design/designed_sequences.fasta"
[[ -f "${DESIGN_FASTA}" ]] || { echo "FATAL: missing ${DESIGN_FASTA}" >&2; exit 5; }

PROBE_DIR="${RUN_DIR}/characterize/_probe"
mkdir -p "${PROBE_DIR}/structures" "${PROBE_DIR}/fold_scores" "${RUN_DIR}/characterize/logs"
cd "${RUN_DIR}/characterize"

echo "=== accounts balance BEFORE ==="
accounts balance 2>/dev/null || echo "(accounts balance unavailable)"

JID=$(sbatch --parsable --array=0-0 --job-name=esmfold_probe \
    --time=00:30:00 \
    "${SHARD_JOB}" "${DESIGN_FASTA}" "${PROBE_DIR}/structures" \
    "${PROBE_DIR}/fold_scores" design 1 0 "${N_PROBE}")
echo "probe job: ${JID}  (folds ${N_PROBE} designs)"
echo
echo "when it finishes (squeue -u \$USER), read timing + cost:"
echo "  grep FOLD_TIMING ${RUN_DIR}/characterize/logs/esmfold_shard_${JID}_0.log"
echo "  accounts balance     # compare to the BEFORE snapshot above"
