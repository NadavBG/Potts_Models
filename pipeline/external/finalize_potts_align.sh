#!/usr/bin/env bash
# Login-node finalizer for a potts_align align run (iter-003). Run on a Midway
# login node after the gather job mails END.
#
# Usage:
#   bash pipeline/external/finalize_potts_align.sh <run_root>
#
# What it does:
#   1. Read <run_root>/potts_align/.shard_jids and verify every job's sacct State
#      is COMPLETED. Abort if any is still PENDING/RUNNING/FAILED/CANCELLED.
#   2. Confirm gather produced cache/<model>/alignments.tsv for both models.
#   3. Reclaim space: tar+zstd the raw per-shard TSVs and logs. (There is NO
#      per-shard work/ dir to delete — potts_align writes no scratch binaries.)
#
# It does NOT move the cache off Midway and combine/ stays out of git. The score
# + render step then runs ON THE MAC (snakemake off the login node): pull the
# durable cache with scripts/sync_models.sh (its generic combine excludes prune
# shards/ + logs/ + *.tar.zst automatically), then `snakemake ... all`. See
# docs/POTTS_ALIGN.md §11.

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$(realpath "$1")"
PA_DIR="${RUN_ROOT}/potts_align"
JIDS_FILE="${PA_DIR}/.shard_jids"
[[ -f "${JIDS_FILE}" ]] || { echo "ERROR: ${JIDS_FILE} not found (run run_potts_align_align.sh first)." >&2; exit 2; }

# --- Step 1: validate every job COMPLETED -----------------------------------
JID_LIST="$(paste -sd, "${JIDS_FILE}")"
echo "checking sacct for jobs ${JID_LIST}..."
SACCT_OUT="$(sacct -j "${JID_LIST}" -X -P -o JobID,State 2>/dev/null || true)"
[[ -n "${SACCT_OUT}" ]] || { echo "ERROR: sacct returned nothing; chain still running or records lost." >&2; exit 6; }

failed=()
incomplete=()
while IFS='|' read -r jid state; do
    [[ "${jid}" == "JobID" || -z "${jid}" ]] && continue
    case "${state}" in
        COMPLETED) : ;;
        FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) failed+=("${jid}:${state}") ;;
        *) incomplete+=("${jid}:${state}") ;;
    esac
done <<< "${SACCT_OUT}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "ERROR: failed jobs: ${failed[*]}" >&2; exit 7
fi
if [[ ${#incomplete[@]} -gt 0 ]]; then
    echo "ERROR: jobs not COMPLETED yet: ${incomplete[*]} — wait for the END email." >&2; exit 8
fi
echo "sacct OK: all jobs COMPLETED."

# --- Step 2: confirm gather outputs ------------------------------------------
mapfile -t ALIGN_FILES < <(find "${PA_DIR}/cache" -mindepth 2 -maxdepth 2 -name alignments.tsv 2>/dev/null)
if [[ ${#ALIGN_FILES[@]} -lt 2 ]]; then
    echo "ERROR: expected 2 cache/<model>/alignments.tsv, found ${#ALIGN_FILES[@]} — did gather run?" >&2
    exit 9
fi
echo "gather outputs: ${#ALIGN_FILES[@]} alignments.tsv"
[[ -f "${PA_DIR}/gather_status.json" ]] && cat "${PA_DIR}/gather_status.json"

# --- Step 3: reclaim space ---------------------------------------------------
command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not on PATH." >&2; exit 10; }
# Archive then DELETE the raw sources (set -o pipefail aborts before rm if the
# tar|zstd fails). The per-model alignments.tsv the score step needs lives in
# cache/<model>/, not cache/shards/, so removing the raw shards is safe.
if [[ -d "${PA_DIR}/cache/shards" ]]; then
    tar -C "${PA_DIR}/cache" -cf - shards | zstd -19 --force --quiet -o "${PA_DIR}/cache/shards.tar.zst"
    rm -rf "${PA_DIR}/cache/shards"
    echo "  ${PA_DIR}/cache/shards.tar.zst (raw shards/ removed)"
fi
if [[ -d "${PA_DIR}/logs" ]]; then
    tar -C "${PA_DIR}" -cf - logs | zstd -19 --force --quiet -o "${PA_DIR}/potts_align_logs.tar.zst"
    rm -rf "${PA_DIR}/logs"
    echo "  ${PA_DIR}/potts_align_logs.tar.zst (raw logs/ removed)"
fi

echo
echo "finalize complete. Durable cache (what scoring needs) is at:"
for align in "${ALIGN_FILES[@]}"; do
    echo "  ${align}"
done
echo
echo "Now score + render ON THE MAC (do NOT run snakemake on the login node):"
echo "  # on the Mac, from the repo root:"
echo "  scripts/sync_models.sh pull   # brings cache/<model>/alignments.tsv (+ meta.json)"
echo "  snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \\"
echo "      --config run_root=${RUN_ROOT} --cores 8 all"
echo "(swap the --configfile for the tiny config on the tiny smoke run.) See docs/POTTS_ALIGN.md §11."
