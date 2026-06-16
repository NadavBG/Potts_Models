#!/usr/bin/env bash
# Login-node finalizer for a DCAlign align run (spec §10.9). Run after the
# gather job mails END.
#
# Usage (on a Midway login node):
#   bash pipeline/external/finalize_dcalign_push.sh <run_root> [--push]
#
# What it does, always:
#   1. Read <run_root>/dcalign/.shard_jids and verify every job's sacct State is
#      COMPLETED. Abort if any is still PENDING/RUNNING/FAILED/CANCELLED.
#   2. Confirm gather produced cache/<model>/alignments.tsv for both models.
#   3. Reclaim space: delete the transient per-shard model binaries
#      (cache/<model>/work/), then tar+zstd the raw shard TSVs and logs.
#
# With --push (opt-in; combine/ is gitignored, so this force-adds):
#   4. git add -f the small DURABLE cache (alignments.tsv, meta.json,
#      gather_status.json, shards_manifest.json + the compressed shards/logs),
#      then git pull --rebase and git push. Without --push it just prints what
#      it would commit (pushing is irreversible, so it is never the default).
#
# Compute happens on Midway and the cheap `score` step also runs here, so the
# alignment cache normally does NOT need to leave Midway — --push is only for
# preserving an expensive run across machines.

set -euo pipefail
IFS=$'\n\t'

export GIT_SSL_CAINFO="${GIT_SSL_CAINFO:-/etc/pki/tls/certs/ca-bundle.crt}"

PUSH=0
ARGS=()
for a in "$@"; do
    case "${a}" in
        --push) PUSH=1 ;;
        *) ARGS+=("${a}") ;;
    esac
done
if [[ ${#ARGS[@]} -ne 1 ]]; then
    echo "Usage: $0 <run_root> [--push]" >&2
    exit 2
fi
RUN_ROOT="$(realpath "${ARGS[0]}")"
DCALIGN_DIR="${RUN_ROOT}/dcalign"
JIDS_FILE="${DCALIGN_DIR}/.shard_jids"
[[ -f "${JIDS_FILE}" ]] || { echo "ERROR: ${JIDS_FILE} not found (run run_dcalign_align.sh first)." >&2; exit 2; }

REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"

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

# --- Step 4: optional git push (force, since combine/ is gitignored) ---------
DURABLE=()
while IFS= read -r f; do DURABLE+=("${f}"); done < <(
    find "${DCALIGN_DIR}" \( -name alignments.tsv -o -name meta.json \
        -o -name gather_status.json -o -name shards_manifest.json \
        -o -name '*.tar.zst' \) -type f
)
if [[ "${PUSH}" -ne 1 ]]; then
    echo
    echo "(--push not given) durable artifacts that WOULD be committed (git add -f):"
    printf '  %s\n' "${DURABLE[@]}"
    echo "Re-run with --push to commit + push them."
    exit 0
fi

echo
echo "git add -f durable artifacts + pull --rebase + push..."
cd "${REPO_DIR}"
git add -f "${DURABLE[@]}"
if git diff --cached --quiet; then
    echo "  nothing to commit."
else
    git commit -m "DCAlign align cache: $(basename "$(dirname "${RUN_ROOT}")")/$(basename "${RUN_ROOT}")

Gathered alignments + provenance from chain $(head -1 "${JIDS_FILE}")..$(tail -1 "${JIDS_FILE}").

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git pull --rebase --quiet origin "${BRANCH}"
git push origin "${BRANCH}"
echo "done."
