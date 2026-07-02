#!/usr/bin/env bash
# Login-node finalizer for a two-model design anneal (docs/DESIGN_TWO_MODEL.md).
# Run on a Midway login node after the gather job mails END.
#
# Usage:
#   bash pipeline/external/finalize_design.sh <run_root>
#
# What it does:
#   1. Read <run_root>/design/.shard_jids and verify every job's sacct State is
#      COMPLETED. Abort if any is still PENDING/RUNNING/FAILED/CANCELLED.
#   2. Confirm gather produced the run outputs (trajectories.npz, designed.tsv,
#      design_manifest.json, gather_status.json).
#   3. Reclaim space: tar+zstd the raw per-shard JSONL and the job logs.
#
# It does NOT move the outputs off Midway and combine/ stays out of git. The
# render step then runs ON THE MAC (snakemake/render off the login node): pull the
# gathered artifacts with scripts/sync_models.sh (its generic combine excludes
# prune shards/ + logs/ + *.tar.zst automatically), then render the figures. See
# docs/DESIGN_TWO_MODEL.md.

set -euo pipefail
IFS=$'\n\t'

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run_root>" >&2
    exit 2
fi
RUN_ROOT="$(realpath "$1")"
DESIGN_DIR="${RUN_ROOT}/design"
JIDS_FILE="${DESIGN_DIR}/.shard_jids"
[[ -f "${JIDS_FILE}" ]] || { echo "ERROR: ${JIDS_FILE} not found (run run_design.sh first)." >&2; exit 2; }

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
missing=()
for rel in trajectories.npz designed.tsv design_manifest.json gather_status.json; do
    [[ -f "${DESIGN_DIR}/${rel}" ]] || missing+=("${rel}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: gather outputs missing under ${DESIGN_DIR}: ${missing[*]} — did gather run?" >&2
    exit 9
fi
echo "gather outputs present under ${DESIGN_DIR}"
cat "${DESIGN_DIR}/gather_status.json"

# --- Step 3: reclaim space ---------------------------------------------------
command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not on PATH." >&2; exit 10; }
# Archive then DELETE the raw sources (set -o pipefail aborts before rm if the
# tar|zstd fails). The gathered artifacts the render step needs live at the
# design/ top level, not in shards/, so removing the raw shards is safe.
if [[ -d "${DESIGN_DIR}/shards" ]]; then
    tar -C "${DESIGN_DIR}" -cf - shards | zstd -19 --force --quiet -o "${DESIGN_DIR}/design_shards.tar.zst"
    rm -rf "${DESIGN_DIR}/shards"
    echo "  ${DESIGN_DIR}/design_shards.tar.zst (raw shards/ removed)"
fi
if [[ -d "${DESIGN_DIR}/logs" ]]; then
    tar -C "${DESIGN_DIR}" -cf - logs | zstd -19 --force --quiet -o "${DESIGN_DIR}/design_logs.tar.zst"
    rm -rf "${DESIGN_DIR}/logs"
    echo "  ${DESIGN_DIR}/design_logs.tar.zst (raw logs/ removed)"
fi

echo
echo "finalize complete. Gathered artifacts (what render needs) are at:"
echo "  ${DESIGN_DIR}/{trajectories.npz, designed.tsv, designed_sequences.fasta,"
echo "                 design_aln_A.fasta, design_aln_B.fasta, design_manifest.json}"
echo
echo "Now render ON THE MAC (do NOT run snakemake/render on the login node):"
echo "  # on the Mac, from the repo root:"
echo "  scripts/sync_models.sh pull   # brings the gathered design/ artifacts"
echo "  python scripts/render_design.py --design-dir ${RUN_ROOT}/design --figs-dir ${RUN_ROOT}/figs"
echo "See docs/DESIGN_TWO_MODEL.md."
