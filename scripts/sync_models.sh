#!/usr/bin/env bash
# Sync the durable artifacts that don't belong in git between the Mac and
# Midway, with end-to-end checksum verification. Covers two trees:
#   results/  trained models (big .npy blobs; ~0.5 GB/run)
#   combine/  two-model runs, incl. the DCAlign cache the Mac reads to score
# Replaces the old (never-populated) Git-LFS handoff. Both live on BOTH
# machines: models so the cluster alignment can read them, and the DCAlign
# cache so the combine SCORING can run on the Mac after that alignment. The
# Mac-primary / Midway-for-DCAlign split is documented in docs/PIPELINE.md;
# sync specifics in docs/MODEL_SYNC.md.
#
# Each tree gets its own <tree>/SHA256SUMS manifest. A tree absent on one side
# is skipped (a fresh Mac has no combine/ until a run); a command fails only if
# NO tree is present. Override the tree list with SBM_SYNC_ROOTS="results ...".
#
# Usage:
#   scripts/sync_models.sh hash                  # (re)build each tree's SHA256SUMS locally
#   scripts/sync_models.sh push   [opts]         # Mac -> Midway, then verify on Midway
#   scripts/sync_models.sh pull   [opts]         # Midway -> Mac, then verify locally
#   scripts/sync_models.sh verify [--remote]     # check each SHA256SUMS (no transfer)
#   scripts/sync_models.sh status [opts]         # diff local vs remote manifests (no transfer)
#
# Options:
#   --dry-run      show what rsync would transfer; skip the verify step
#   --with-figs    also sync figs/ and mpnn_tmp/ (default: durable artifacts only)
#   --mirror       add rsync --delete: delete destination files absent from the
#                  synced set (excluded dirs like figs/ are NOT deleted); prompts
#   --yes          skip the --mirror confirmation prompt
#   --no-verify    transfer only; skip the checksum verify (verify later with
#                  `verify`/`verify --remote`)
#   --host HOST    override SBM_MIDWAY_HOST
#   --repo PATH    override SBM_MIDWAY_REPO (remote repo root; each synced tree —
#                  results/, combine/ — is appended)
#
# ssh connections are multiplexed (ControlMaster), so a whole command — transfer
# AND verify — authenticates to Midway only ONCE (one Duo/password prompt).
#
# Config resolution (first wins): CLI flag -> environment variable ->
# scripts/sync_models.local.sh (sourced if present) -> built-in default.
# See docs/MODEL_SYNC.md.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Trees synced (repo-root-relative). results/ = trained models; combine/ =
# two-model runs incl. the DCAlign cache. Override with SBM_SYNC_ROOTS.
# IFS is $'\n\t' here (no space), so split explicitly on whitespace.
IFS=$' \t\n' read -r -a SYNC_ROOTS <<< "${SBM_SYNC_ROOTS:-results combine}"

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
NO_VERIFY=0
VERIFY_REMOTE=0
CLI_HOST=""
CLI_REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1 ;;
        --with-figs) WITH_FIGS=1 ;;
        --mirror)    MIRROR=1 ;;
        --yes)       ASSUME_YES=1 ;;
        --no-verify) NO_VERIFY=1 ;;
        --remote)    VERIFY_REMOTE=1 ;;
        --host)      CLI_HOST="${2:-}"; shift ;;
        --repo)      CLI_REPO="${2:-}"; shift ;;
        *)           die "unknown option: $1" ;;
    esac
    shift
done

HOST="${CLI_HOST:-${SBM_MIDWAY_HOST:-${DEFAULT_HOST}}}"
REPO="${CLI_REPO:-${SBM_MIDWAY_REPO:-${DEFAULT_REPO}}}"
# Strip a trailing slash so "${REPO}/<tree>" is well-formed for each synced tree.
REPO="${REPO%/}"

# --- ssh connection sharing (multiplexing) ----------------------------------
# Without this, one `push` opens TWO ssh connections (rsync + verify) and a
# Duo/password prompt fires for EACH. ControlMaster shares a single
# authenticated connection across every ssh+rsync in the command: the first to
# connect authenticates, the rest reuse the socket. setup_mux is idempotent;
# close_mux (EXIT trap) tears the master down so no socket lingers.
SSH_CTL=""
SSH_OPTS=()
setup_mux() {
    [[ -n "${SSH_CTL}" ]] && return 0
    SSH_CTL="$(mktemp -u /tmp/sbm_sync_ctl.XXXXXXXX)"
    SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${SSH_CTL}" -o ControlPersist=60)
    trap 'close_mux' EXIT
    log "ssh connection sharing on — you authenticate to ${HOST} once for this command."
}
close_mux() {
    [[ -n "${SSH_CTL}" ]] || return 0
    ssh -o "ControlPath=${SSH_CTL}" -O exit "${HOST}" >/dev/null 2>&1 || true
}

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
# rsync --exclude patterns for ONE tree. This set MUST mirror find_durable()'s
# prunes for the same tree, or the post-transfer verify FAILs on a manifested
# file rsync skipped. combine/ additionally drops the heavy/regenerable DCAlign
# scratch so only the small per-model alignments.tsv (+ meta.json) travels.
rsync_excludes() {
    local root="$1"
    # Always-excluded caches/junk.
    printf '%s\n' '__pycache__/' '.snakemake/' '.DS_Store' '*.pyc'
    if [[ "${WITH_FIGS}" -eq 0 ]]; then
        printf '%s\n' 'figs/' 'mpnn_tmp/'
    fi
    if [[ "${root}" == "combine" ]]; then
        # work/   ~7-8 GB/model of BP-solver scratch (deleted by the finalizer)
        # shards/ raw per-shard TSVs, already merged into alignments.tsv
        # logs/   machine-local job logs (scoring regenerates its own on the Mac)
        # *.tar.zst  the finalizer's archives (not needed to score)
        printf '%s\n' 'work/' 'shards/' 'logs/' '*.tar.zst'
    fi
}

# Emit a newline-separated list of durable files under ONE tree (cwd = repo
# root). The prunes mirror rsync_excludes() for the same tree.
find_durable() {
    local root="$1"
    local prune=(-name __pycache__ -o -name .snakemake)
    if [[ "${WITH_FIGS}" -eq 0 ]]; then
        prune+=(-o -name figs -o -name mpnn_tmp)
    fi
    local extra=()
    if [[ "${root}" == "combine" ]]; then
        prune+=(-o -name work -o -name shards -o -name logs)
        extra=(! -name '*.tar.zst')
    fi
    find "${root}" -type d \( "${prune[@]}" \) -prune -o \
         -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' "${extra[@]}" -print
}

# Build <tree>/SHA256SUMS locally for each present tree (cwd = repo root,
# entries repo-root-relative). Deterministic order. Skips an absent tree; dies
# only if none of SYNC_ROOTS is present.
build_local_manifest() {
    ( cd "${REPO_ROOT}"
      local files; files="$(mktemp)"
      trap 'rm -f "${files}"' EXIT   # EXIT, not RETURN: this runs in a ( ) subshell
      local root present=0
      for root in "${SYNC_ROOTS[@]}"; do
          [[ -d "${root}" ]] || continue
          find_durable "${root}" | LC_ALL=C sort > "${files}"
          if [[ ! -s "${files}" ]]; then
              warn "${root}/ has no durable files — skipping its manifest."
              continue
          fi
          tr '\n' '\0' < "${files}" | xargs -0 "${SHA_BIN[@]}" > "${root}/SHA256SUMS"
          log "wrote ${root}/SHA256SUMS ($(wc -l < "${root}/SHA256SUMS" | tr -d ' ') files)"
          present=$((present+1))
      done
      [[ "${present}" -gt 0 ]] || die "none of [${SYNC_ROOTS[*]}] present at ${REPO_ROOT} (nothing to sync)."
    )
}

# Build the manifest on Midway. Self-contained so it does not depend on this
# script existing remotely. Echoes the remote file count on success.
build_remote_manifest() {
    setup_mux
    ssh "${SSH_OPTS[@]}" "${HOST}" 'bash -s' -- "${REPO}" "${WITH_FIGS}" "${SYNC_ROOTS[@]}" <<'REOF'
set -euo pipefail
repo="$1"; with_figs="$2"; shift 2; roots=("$@")
cd "$repo" || { echo "ERROR: remote repo not found: $repo" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
present=0
for root in "${roots[@]}"; do
    [ -d "$root" ] || continue
    prune=(-name __pycache__ -o -name .snakemake)
    [ "$with_figs" -eq 0 ] && prune+=(-o -name figs -o -name mpnn_tmp)
    extra=()
    if [ "$root" = "combine" ]; then
        prune+=(-o -name work -o -name shards -o -name logs)
        extra=(! -name '*.tar.zst')
    fi
    files="$(mktemp)"
    find "$root" -type d \( "${prune[@]}" \) -prune -o \
         -type f ! -name '.DS_Store' ! -name '*.pyc' ! -name 'SHA256SUMS' "${extra[@]}" -print \
         | LC_ALL=C sort > "$files"
    if [ -s "$files" ]; then
        tr '\n' '\0' < "$files" | xargs -0 "${SHA[@]}" > "$root/SHA256SUMS"
        present=$((present+1))
    fi
    rm -f "$files"
done
[ "$present" -gt 0 ] || { echo "ERROR: none of [${roots[*]}] under $repo" >&2; exit 1; }
echo "$present"
REOF
}

# --- verification (independent of rsync) ------------------------------------
# Returns nonzero and prints the failing lines if any file mismatches/missing.
check_failures() { grep -c 'FAILED' || true; }

verify_local() {
    ( cd "${REPO_ROOT}"
      local root checked=0
      for root in "${SYNC_ROOTS[@]}"; do
          [[ -f "${root}/SHA256SUMS" ]] || continue
          local out rc=0
          out="$("${SHA_BIN[@]}" -c "${root}/SHA256SUMS" 2>&1)" || rc=$?
          local fails; fails="$(printf '%s\n' "${out}" | check_failures)"
          if [[ "${rc}" -ne 0 || "${fails}" -ne 0 ]]; then
              printf '%s\n' "${out}" | grep 'FAILED' >&2 || true
              die "local verify FAILED for ${root}/: ${fails} file(s) reported FAILED (checksum exit ${rc}). See lines above."
          fi
          log "local verify OK: ${root}/ ($(wc -l < "${root}/SHA256SUMS" | tr -d ' ') files)"
          checked=$((checked+1))
      done
      [[ "${checked}" -gt 0 ]] || die "no <tree>/SHA256SUMS locally (run 'hash', 'push', or 'pull' first)."
    )
}

# Run the manifest check on Midway. The remote command is sent as a heredoc with
# the repo path as a positional arg (quoted remotely), and selects the checksum
# tool by existence — NOT a `sha256sum -c || shasum -c` fallback, which would
# silently retry-on-failure and could mask a real mismatch.
verify_remote() {
    setup_mux
    log "remote verify on ${HOST}:${REPO} (${SYNC_ROOTS[*]}) ..."
    local out rc=0
    out="$(ssh "${SSH_OPTS[@]}" "${HOST}" 'bash -s' -- "${REPO}" "${SYNC_ROOTS[@]}" <<'REOF'
set -uo pipefail
repo="$1"; shift; roots=("$@")
cd "$repo" || { echo "verify: cannot cd to remote repo $repo" >&2; exit 12; }
if command -v sha256sum >/dev/null 2>&1; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
checked=0
for root in "${roots[@]}"; do
    [ -f "$root/SHA256SUMS" ] || continue
    "${SHA[@]}" -c "$root/SHA256SUMS" || exit 1
    checked=$((checked+1))
done
[ "$checked" -gt 0 ] || { echo "verify: no <tree>/SHA256SUMS on remote" >&2; exit 8; }
REOF
    )" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        printf '%s\n' "${out}" | grep -iE 'FAILED|No such|missing|cannot|verify:' >&2 || printf '%s\n' "${out}" >&2
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
    # Per-tree --exclude patterns are appended in do_push/do_pull (they differ
    # by tree), so RSYNC_ARGS holds only the tree-independent flags here.
    # Route rsync's ssh through the shared master so the transfer reuses the
    # one authenticated connection (no second Duo/password prompt).
    [[ -n "${SSH_CTL}" ]] && RSYNC_ARGS+=(-e "ssh -o ControlMaster=auto -o ControlPath=${SSH_CTL} -o ControlPersist=60")
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
    log "  trees       : ${SYNC_ROOTS[*]} (only those present are transferred)"
    log "  remote repo : ${HOST}:${REPO}"
    log "  local repo  : ${REPO_ROOT}"
    [[ -n "${RSYNC_VER}" ]] && log "  rsync       : ${RSYNC} — ${RSYNC_VER}"
    log "  scope       : $([[ ${WITH_FIGS} -eq 1 ]] && echo 'everything (--with-figs)' || echo 'durable only')"
    log "  git HEAD    : $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '?')$( [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]] && echo ' (dirty)')"
    log "----------------------------------------------------------------"
}

do_push() {
    confirm_mirror
    setup_mux              # one auth here; rsync + verify reuse the connection
    build_rsync_args
    print_summary push
    build_local_manifest
    local root pat synced=0
    for root in "${SYNC_ROOTS[@]}"; do
        [[ -d "${REPO_ROOT}/${root}" ]] || { log "skip ${root}/ (absent on Mac)"; continue; }
        local args=("${RSYNC_ARGS[@]}")
        while IFS= read -r pat; do args+=("--exclude=${pat}"); done < <(rsync_excludes "${root}")
        log "rsync up: ${REPO_ROOT}/${root}/ -> ${HOST}:${REPO}/${root}/"
        "${RSYNC}" "${args[@]}" "${REPO_ROOT}/${root}/" "${HOST}:${REPO}/${root}/"
        synced=$((synced+1))
    done
    [[ "${synced}" -gt 0 ]] || die "no trees present on Mac to push."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "dry-run: skipped remote verify."
        return 0
    fi
    if [[ "${NO_VERIFY}" -eq 1 ]]; then
        log "push complete. Verify skipped (--no-verify); run later: scripts/sync_models.sh verify --remote"
        return 0
    fi
    verify_remote
    log "push complete and verified."
}

do_pull() {
    confirm_mirror
    setup_mux              # one auth here; remote manifest + rsync reuse it
    build_rsync_args
    print_summary pull
    log "rebuilding manifest(s) on ${HOST} ..."
    build_remote_manifest >/dev/null
    # Which trees exist on the remote? One round-trip over the shared connection
    # (repo path + roots passed as positional args, not interpolated into the
    # remote command); rsync of an absent source dir would otherwise error out.
    local remote_roots
    remote_roots="$(ssh "${SSH_OPTS[@]}" "${HOST}" 'bash -s' -- "${REPO}" "${SYNC_ROOTS[@]}" <<'REOF'
repo="$1"; shift
cd "$repo" 2>/dev/null || exit 0
for r in "$@"; do [ -d "$r" ] && printf '%s\n' "$r"; done
REOF
    )"
    [[ -n "${remote_roots}" ]] || die "none of [${SYNC_ROOTS[*]}] present on ${HOST}:${REPO}."
    local root pat
    while IFS= read -r root; do
        [[ -n "${root}" ]] || continue
        local args=("${RSYNC_ARGS[@]}")
        while IFS= read -r pat; do args+=("--exclude=${pat}"); done < <(rsync_excludes "${root}")
        log "rsync down: ${HOST}:${REPO}/${root}/ -> ${REPO_ROOT}/${root}/"
        "${RSYNC}" "${args[@]}" "${HOST}:${REPO}/${root}/" "${REPO_ROOT}/${root}/"
    done <<< "${remote_roots}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "dry-run: skipped local verify."
        return 0
    fi
    if [[ "${NO_VERIFY}" -eq 1 ]]; then
        log "pull complete. Verify skipped (--no-verify); run later: scripts/sync_models.sh verify"
        return 0
    fi
    verify_local
    log "pull complete and verified."
}

# Diff local vs remote manifests per tree; report only-local / only-remote /
# mismatch, totalled across trees.
do_status() {
    print_summary status
    build_local_manifest
    log "rebuilding manifest(s) on ${HOST} ..."
    build_remote_manifest >/dev/null
    local lt rt remote_raw; lt="$(mktemp)"; rt="$(mktemp)"; remote_raw="$(mktemp)"
    trap 'rm -f "${lt}" "${rt}" "${remote_raw}"' RETURN
    # Normalize each manifest to "path<TAB>hash", sorted by path.
    normalize() { awk '{h=$1; $1=""; sub(/^[ \t]+/,""); print $0"\t"h}' | LC_ALL=C sort; }
    local ok=0 only_local=0 only_remote=0 mismatch=0 root
    for root in "${SYNC_ROOTS[@]}"; do
        if [[ -f "${REPO_ROOT}/${root}/SHA256SUMS" ]]; then
            normalize < "${REPO_ROOT}/${root}/SHA256SUMS" > "${lt}"
        else : > "${lt}"; fi
        ssh "${SSH_OPTS[@]}" "${HOST}" 'bash -s' -- "${REPO}" "${root}" > "${remote_raw}" <<'REOF'
cat "$1/$2/SHA256SUMS" 2>/dev/null || true
REOF
        if [[ -s "${remote_raw}" ]]; then normalize < "${remote_raw}" > "${rt}"; else : > "${rt}"; fi
        [[ -s "${lt}" || -s "${rt}" ]] || continue
        local report; report="$(LC_ALL=C join -t"$(printf '\t')" -a1 -a2 -e '__ABSENT__' -o '0,1.2,2.2' "${lt}" "${rt}")"
        local path lh rh
        while IFS=$'\t' read -r path lh rh; do
            [[ -z "${path}" ]] && continue
            if   [[ "${lh}" == "__ABSENT__" ]]; then only_remote=$((only_remote+1)); printf '  only on Midway : %s\n' "${path}" >&2
            elif [[ "${rh}" == "__ABSENT__" ]]; then only_local=$((only_local+1));   printf '  only on Mac    : %s\n' "${path}" >&2
            elif [[ "${lh}" != "${rh}" ]];        then mismatch=$((mismatch+1));     printf '  HASH MISMATCH  : %s\n' "${path}" >&2
            else ok=$((ok+1)); fi
        done <<< "${report}"
    done
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
