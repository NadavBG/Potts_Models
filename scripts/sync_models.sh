#!/usr/bin/env bash
# Sync trained models between the Mac and Midway with end-to-end checksum
# verification. Replaces the old (never-populated) Git-LFS handoff: models are
# big binary blobs (model.npy ~47 MB; 4.4 GB across all families) that do not
# belong in git. They live on BOTH machines so larger cross-model comparisons
# can run on the cluster.
#
# Usage:
#   scripts/sync_models.sh hash                  # (re)build results/SHA256SUMS locally
#   scripts/sync_models.sh push   [opts]         # Mac -> Midway, then verify on Midway
#   scripts/sync_models.sh pull   [opts]         # Midway -> Mac, then verify locally
#   scripts/sync_models.sh verify [--remote]     # check results/SHA256SUMS (no transfer)
#   scripts/sync_models.sh status [opts]         # diff local vs remote manifests (no transfer)
#
# Options:
#   --dry-run      show what rsync would transfer; skip the verify step
#   --with-figs    also sync figs/ and mpnn_tmp/ (default: durable artifacts only)
#   --mirror       add rsync --delete: delete destination files absent from the
#                  synced set (excluded dirs like figs/ are NOT deleted); prompts
#   --yes          skip the --mirror confirmation prompt
#   --host HOST    override SBM_MIDWAY_HOST
#   --repo PATH    override SBM_MIDWAY_REPO (remote repo root; /results is appended)
#
# Config resolution (first wins): CLI flag -> environment variable ->
# scripts/sync_models.local.sh (sourced if present) -> built-in default.
# See docs/MODEL_SYNC.md.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_HOST="midway3.rcc.uchicago.edu"
DEFAULT_REPO="/project/ranganathanr/nadavbg/Potts_Models"

log()  { printf '%s\n' "$*" >&2; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- machine-specific config (gitignored, optional) -------------------------
# The file uses `: "${VAR:=...}"` so an inline env var still wins over it.
LOCAL_CONF="${REPO_ROOT}/scripts/sync_models.local.sh"
if [[ -f "${LOCAL_CONF}" ]]; then
    # shellcheck source=/dev/null
    source "${LOCAL_CONF}"
fi

# --- argument parsing -------------------------------------------------------
CMD="${1:-}"
[[ -n "${CMD}" ]] || die "no subcommand. One of: hash, push, pull, verify, status. See --help in the header."
shift || true

DRY_RUN=0
WITH_FIGS=0
MIRROR=0
ASSUME_YES=0
VERIFY_REMOTE=0
CLI_HOST=""
CLI_REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1 ;;
        --with-figs) WITH_FIGS=1 ;;
        --mirror)    MIRROR=1 ;;
        --yes)       ASSUME_YES=1 ;;
        --remote)    VERIFY_REMOTE=1 ;;
        --host)      CLI_HOST="${2:-}"; shift ;;
        --repo)      CLI_REPO="${2:-}"; shift ;;
        *)           die "unknown option: $1" ;;
    esac
    shift
done

HOST="${CLI_HOST:-${SBM_MIDWAY_HOST:-${DEFAULT_HOST}}}"
REPO="${CLI_REPO:-${SBM_MIDWAY_REPO:-${DEFAULT_REPO}}}"
# Strip a trailing slash so "${REPO}/results" is well-formed.
REPO="${REPO%/}"

# --- tool detection ---------------------------------------------------------
# rsync detection is lazy (only push/pull transfer), so hash/verify/status do
# not emit a spurious openrsync warning. Prefer GNU rsync; macOS ships openrsync
# (protocol 29) which lacks some flags — fall back to Homebrew's if present.
RSYNC=""
RSYNC_VER=""
IS_GNU_RSYNC=1
# First line of `rsync --version`. Capture-then-slice instead of piping to
# `head -1`: GNU rsync prints a long version block and an early-closing `head`
# pipe makes rsync exit with SIGPIPE (141), which `set -o pipefail` + `set -e`
# would turn into a silent death of this whole script.
rsync_version_line() {
    local v
    v="$("$1" --version 2>&1)" || true
    printf '%s\n' "${v%%$'\n'*}"
}
detect_rsync() {
    [[ -n "${RSYNC}" ]] && return 0
    RSYNC="${SBM_RSYNC:-rsync}"
    command -v "${RSYNC}" >/dev/null 2>&1 || die "rsync not found (looked for '${RSYNC}'). Install it or set SBM_RSYNC."
    RSYNC_VER="$(rsync_version_line "${RSYNC}")"
    if [[ -z "${SBM_RSYNC:-}" ]] && printf '%s' "${RSYNC_VER}" | grep -qi 'openrsync'; then
        if [[ -x /opt/homebrew/bin/rsync ]] && ! /opt/homebrew/bin/rsync --version 2>&1 | grep -qi 'openrsync'; then
            RSYNC=/opt/homebrew/bin/rsync
            RSYNC_VER="$(rsync_version_line "${RSYNC}")"
        fi
    fi
    if printf '%s' "${RSYNC_VER}" | grep -qi 'openrsync'; then
        IS_GNU_RSYNC=0
        warn "using openrsync ('${RSYNC_VER}'); some flags are unsupported. For a smoother transfer: brew install rsync, or set SBM_RSYNC."
    fi
}

# sha256: GNU coreutils `sha256sum`, else BSD `shasum -a 256`. Both support -c.
if command -v sha256sum >/dev/null 2>&1; then
    SHA_BIN=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    SHA_BIN=(shasum -a 256)
else
    die "neither sha256sum nor shasum found."
fi

# --- the durable file set ---------------------------------------------------
# Mirrored by both the manifest (find below) and rsync (--exclude). A divergence
# is caught loudly by the post-transfer verify (a manifest entry that did not
# land prints FAILED), never silent.
rsync_excludes() {
    # Always-excluded caches/junk.
    printf '%s\n' '__pycache__/' '.snakemake/' '.DS_Store' '*.pyc'
    if [[ "${WITH_FIGS}" -eq 0 ]]; then
        printf '%s\n' 'figs/' 'mpnn_tmp/'
    fi
}

# Emit a NUL-separated list of durable files under results/ (cwd = repo root).
find_durable() {
    if [[ "${WITH_FIGS}" -eq 1 ]]; then
        find results -type d \( -name __pycache__ -o -name .snakemake \) -prune -o \
             -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' -print
    else
        find results -type d \( -name figs -o -name mpnn_tmp -o -name __pycache__ -o -name .snakemake \) -prune -o \
             -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' -print
    fi
}

# Build results/SHA256SUMS locally (cwd = repo root). Deterministic order.
build_local_manifest() {
    ( cd "${REPO_ROOT}"
      [[ -d results ]] || die "no results/ directory at ${REPO_ROOT}."
      local files; files="$(mktemp)"
      trap 'rm -f "${files}"' EXIT   # EXIT, not RETURN: this runs in a ( ) subshell
      find_durable | LC_ALL=C sort > "${files}"
      [[ -s "${files}" ]] || die "no durable files found under results/ (nothing to sync)."
      tr '\n' '\0' < "${files}" | xargs -0 "${SHA_BIN[@]}" > results/SHA256SUMS
      log "wrote results/SHA256SUMS ($(wc -l < results/SHA256SUMS | tr -d ' ') files)"
    )
}

# Build the manifest on Midway. Self-contained so it does not depend on this
# script existing remotely. Echoes the remote file count on success.
build_remote_manifest() {
    ssh "${HOST}" 'bash -s' -- "${REPO}" "${WITH_FIGS}" <<'REOF'
set -euo pipefail
repo="$1"; with_figs="$2"
cd "$repo" || { echo "ERROR: remote repo not found: $repo" >&2; exit 1; }
[ -d results ] || { echo "ERROR: no results/ under $repo" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
files="$(mktemp)"; trap 'rm -f "$files"' EXIT
if [ "$with_figs" -eq 1 ]; then
    find results -type d \( -name __pycache__ -o -name .snakemake \) -prune -o \
         -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' -print
else
    find results -type d \( -name figs -o -name mpnn_tmp -o -name __pycache__ -o -name .snakemake \) -prune -o \
         -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' -print
fi | LC_ALL=C sort > "$files"
[ -s "$files" ] || { echo "ERROR: no durable files under $repo/results" >&2; exit 1; }
tr '\n' '\0' < "$files" | xargs -0 $SHA > results/SHA256SUMS
wc -l < results/SHA256SUMS
REOF
}

# --- verification (independent of rsync) ------------------------------------
# Returns nonzero and prints the failing lines if any file mismatches/missing.
check_failures() { grep -c 'FAILED' || true; }

verify_local() {
    ( cd "${REPO_ROOT}"
      [[ -f results/SHA256SUMS ]] || die "no results/SHA256SUMS locally (run 'hash', 'push', or 'pull' first)."
      local out rc=0
      out="$("${SHA_BIN[@]}" -c results/SHA256SUMS 2>&1)" || rc=$?
      local fails; fails="$(printf '%s\n' "${out}" | check_failures)"
      if [[ "${rc}" -ne 0 || "${fails}" -ne 0 ]]; then
          printf '%s\n' "${out}" | grep 'FAILED' >&2 || true
          die "local verify FAILED: ${fails} file(s) reported FAILED (checksum exit ${rc}). See lines above."
      fi
      log "local verify OK ($(wc -l < results/SHA256SUMS | tr -d ' ') files)"
    )
}

# Run the manifest check on Midway. The remote command is sent as a heredoc with
# the repo path as a positional arg (quoted remotely), and selects the checksum
# tool by existence — NOT a `sha256sum -c || shasum -c` fallback, which would
# silently retry-on-failure and could mask a real mismatch.
verify_remote() {
    log "remote verify on ${HOST}:${REPO}/results ..."
    local out rc=0
    out="$(ssh "${HOST}" 'bash -s' -- "${REPO}" <<'REOF'
set -uo pipefail
cd "$1" || { echo "verify: cannot cd to remote repo $1" >&2; exit 12; }
[ -f results/SHA256SUMS ] || { echo "verify: results/SHA256SUMS missing on remote" >&2; exit 8; }
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c results/SHA256SUMS
else
    shasum -a 256 -c results/SHA256SUMS
fi
REOF
    )" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        printf '%s\n' "${out}" | grep -iE 'FAILED|No such|missing|cannot' >&2 || printf '%s\n' "${out}" >&2
        die "remote verify FAILED (exit ${rc}) on ${HOST}. See lines above."
    fi
    log "remote verify OK"
}

# --- transfer ---------------------------------------------------------------
build_rsync_args() {
    detect_rsync
    # -a is portable to both GNU rsync and macOS openrsync. -h/--partial/
    # --itemize-changes/--stats are GNU-only; openrsync gets the minimal set.
    RSYNC_ARGS=(-a)
    if [[ "${IS_GNU_RSYNC}" -eq 1 ]]; then
        RSYNC_ARGS+=(-h --partial --itemize-changes --stats)
    else
        RSYNC_ARGS+=(-v)
    fi
    [[ "${DRY_RUN}" -eq 1 ]] && RSYNC_ARGS+=(-n)
    [[ "${MIRROR}"  -eq 1 ]] && RSYNC_ARGS+=(--delete)
    local pat
    while IFS= read -r pat; do RSYNC_ARGS+=("--exclude=${pat}"); done < <(rsync_excludes)
}

confirm_mirror() {
    [[ "${MIRROR}" -eq 1 ]] || return 0
    [[ "${ASSUME_YES}" -eq 1 ]] && return 0
    if [[ ! -t 0 ]]; then
        die "--mirror deletes files at the destination and stdin is not a TTY; pass --yes to proceed non-interactively."
    fi
    local ans
    read -r -p "--mirror will DELETE destination files absent from the source. Continue? [y/N] " ans
    [[ "${ans}" =~ ^[Yy]$ ]] || die "aborted by user."
}

print_summary() {
    local action="$1"
    log "----------------------------------------------------------------"
    log "sync_models ${action}$([[ ${DRY_RUN} -eq 1 ]] && echo ' (dry-run)')"
    log "  host        : ${HOST}"
    log "  remote path : ${REPO}/results"
    log "  local path  : ${REPO_ROOT}/results"
    [[ -n "${RSYNC_VER}" ]] && log "  rsync       : ${RSYNC} — ${RSYNC_VER}"
    log "  scope       : $([[ ${WITH_FIGS} -eq 1 ]] && echo 'everything (--with-figs)' || echo 'durable only')"
    log "  git HEAD    : $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?')$( [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]] && echo ' (dirty)')"
    log "----------------------------------------------------------------"
}

do_push() {
    confirm_mirror
    build_rsync_args
    print_summary push
    build_local_manifest
    log "rsync up: ${REPO_ROOT}/results/ -> ${HOST}:${REPO}/results/"
    "${RSYNC}" "${RSYNC_ARGS[@]}" "${REPO_ROOT}/results/" "${HOST}:${REPO}/results/"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "dry-run: skipped remote verify."
        return 0
    fi
    verify_remote
    log "push complete and verified."
}

do_pull() {
    confirm_mirror
    build_rsync_args
    print_summary pull
    log "rebuilding manifest on ${HOST} ..."
    build_remote_manifest >/dev/null
    log "rsync down: ${HOST}:${REPO}/results/ -> ${REPO_ROOT}/results/"
    "${RSYNC}" "${RSYNC_ARGS[@]}" "${HOST}:${REPO}/results/" "${REPO_ROOT}/results/"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "dry-run: skipped local verify."
        return 0
    fi
    verify_local
    log "pull complete and verified."
}

# Diff local vs remote manifests; report only-local / only-remote / mismatch.
do_status() {
    print_summary status
    build_local_manifest
    log "rebuilding manifest on ${HOST} ..."
    build_remote_manifest >/dev/null
    local lt rt remote_raw; lt="$(mktemp)"; rt="$(mktemp)"; remote_raw="$(mktemp)"
    trap 'rm -f "${lt}" "${rt}" "${remote_raw}"' RETURN
    # Normalize each manifest to "path<TAB>hash", sorted by path.
    normalize() { awk '{h=$1; $1=""; sub(/^[ \t]+/,""); print $0"\t"h}' | LC_ALL=C sort; }
    normalize < "${REPO_ROOT}/results/SHA256SUMS" > "${lt}"
    ssh "${HOST}" "cat ${REPO}/results/SHA256SUMS" > "${remote_raw}"
    normalize < "${remote_raw}" > "${rt}"

    local report; report="$(LC_ALL=C join -t"$(printf '\t')" -a1 -a2 -e '__ABSENT__' -o '0,1.2,2.2' "${lt}" "${rt}")"
    local only_local only_remote mismatch ok
    only_local=0; only_remote=0; mismatch=0; ok=0
    while IFS=$'\t' read -r path lh rh; do
        [[ -z "${path}" ]] && continue
        if   [[ "${lh}" == "__ABSENT__" ]]; then only_remote=$((only_remote+1)); printf '  only on Midway : %s\n' "${path}" >&2
        elif [[ "${rh}" == "__ABSENT__" ]]; then only_local=$((only_local+1));  printf '  only on Mac    : %s\n' "${path}" >&2
        elif [[ "${lh}" != "${rh}" ]];        then mismatch=$((mismatch+1));    printf '  HASH MISMATCH  : %s\n' "${path}" >&2
        else ok=$((ok+1)); fi
    done <<< "${report}"
    log "----------------------------------------------------------------"
    log "status: ${ok} in sync, ${only_local} only on Mac, ${only_remote} only on Midway, ${mismatch} mismatched"
    if [[ "${mismatch}" -gt 0 ]]; then
        die "manifests disagree on ${mismatch} file(s) — same path, different content."
    fi
}

case "${CMD}" in
    hash)   build_local_manifest ;;
    push)   do_push ;;
    pull)   do_pull ;;
    verify) if [[ "${VERIFY_REMOTE}" -eq 1 ]]; then verify_remote; else verify_local; fi ;;
    status) do_status ;;
    -h|--help|help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}" ;;
    *)      die "unknown subcommand: ${CMD} (expected hash, push, pull, verify, or status)." ;;
esac
