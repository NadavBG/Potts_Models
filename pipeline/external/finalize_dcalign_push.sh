#!/usr/bin/env bash
# Login-node finalizer for a DCAlign align run (spec §10.9). Run on a Midway
# login node after the gather job mails END. (The name keeps a historical
# "_push"; it no longer touches git — see the transport note below.)
#
# Usage:
#   bash pipeline/external/finalize_dcalign_push.sh <run_root>
#
# What it does:
#   1. Read <run_root>/dcalign/.shard_jids and verify every job's sacct State is
#      COMPLETED. Abort if any is still PENDING/RUNNING/FAILED/CANCELLED.
#   2. Confirm gather produced cache/<model>/alignments.tsv for both models.
#   3. Reclaim space: delete the transient per-shard model binaries
#      (cache/<model>/work/), then tar+zstd the raw shard TSVs and logs.
#
# It does NOT move the cache off Midway, and combine/ stays out of git. Scoring
# runs on the Mac, so pull the small durable cache there with the rsync wrapper:
#   scripts/sync_models.sh pull        # brings cache/<model>/alignments.tsv (+ meta.json)
# then score + render locally. Full runbook: docs/PIPELINE.md.

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$(realpath "$1")"
DCALIGN_DIR="${RUN_ROOT}/dcalign"
JIDS_FILE="${DCALIGN_DIR}/.shard_jids"
[[ -f "${JIDS_FILE}" ]] || { echo "ERROR: ${JIDS_FILE} not found (run run_dcalign_align.sh first)." >&2; exit 2; }

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
mapfile -t ALIGN_FILES < <(find "${DCALIGN_DIR}/cache" -mindepth 2 -maxdepth 2 -name alignments.tsv 2>/dev/null)
if [[ ${#ALIGN_FILES[@]} -lt 2 ]]; then
    echo "ERROR: expected 2 cache/<model>/alignments.tsv, found ${#ALIGN_FILES[@]} — did gather run?" >&2
    exit 9
fi
echo "gather outputs: ${#ALIGN_FILES[@]} alignments.tsv"
[[ -f "${DCALIGN_DIR}/gather_status.json" ]] && cat "${DCALIGN_DIR}/gather_status.json"

# --- Step 3: reclaim space ---------------------------------------------------
command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not on PATH." >&2; exit 10; }
echo "reclaiming space: removing transient per-shard model binaries..."
find "${DCALIGN_DIR}/cache" -mindepth 2 -maxdepth 2 -type d -name work -exec rm -rf {} + 2>/dev/null || true
for model_dir in "${DCALIGN_DIR}"/cache/*/; do
    [[ -d "${model_dir}/shards" ]] || continue
    name="$(basename "${model_dir%/}")"
    tar -C "${model_dir}" -cf - shards | zstd -19 --force --quiet -o "${model_dir}/shards_${name}.tar.zst"
    echo "  ${model_dir}shards_${name}.tar.zst"
done
if [[ -d "${DCALIGN_DIR}/logs" ]]; then
    tar -C "${DCALIGN_DIR}" -cf - logs | zstd -19 --force --quiet -o "${DCALIGN_DIR}/dcalign_logs.tar.zst"
    echo "  ${DCALIGN_DIR}/dcalign_logs.tar.zst"
fi

# --- Done: point at the rsync transport (no git) -----------------------------
echo
echo "finalize complete. Durable cache (what scoring needs) is at:"
for align in "${ALIGN_FILES[@]}"; do
    echo "  ${align}"
done
echo
echo "Scoring runs on the Mac. Pull the durable cache there with rsync:"
echo "  scripts/sync_models.sh pull"
echo "then run the combine pipeline locally — see docs/PIPELINE.md."
