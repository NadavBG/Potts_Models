#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────
# render_sbm.sh — render figures for a trained SBM/BM run.
#
# Layout:
#   <RUN_DIR>/figs/                  ← always regenerated
#       inputs/
#           stats.npy                (cache of compute_stats output)
#           sources.json             (paths + sha256 of model.npy and
#                                     the synthetic alignment used)
#       coupling_evol.pdf
#       <other figures>.pdf
#
# Default fig set is `coupling_evol` (the only figure that does not
# need a synthetic alignment). For anything else, sample a synthetic
# alignment first with `scripts/sample_sbm.sh`; render_figures.py will
# auto-discover the result under <RUN_DIR>/synthetic/, preferring the
# file whose temperature matches the run's mode (BM=0.75, SBM=1.0).
# ─────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
    bash scripts/render_sbm.sh <RUN_DIR> [options]

Required:
    <RUN_DIR>           a run directory produced by scripts/run_sbm.sh

Optional:
    --synthetic PATH    synthetic alignment .npy
                        (default: auto-discover from <RUN_DIR>/synthetic/)
    --figs NAME ...     figure types to render (default: coupling_evol)
                        all choices: coupling_evol, freq, pair_freq, corr3,
                        pca, energy, similarity, diversity, length
    -h, --help          this message

Behaviour:
    * <RUN_DIR>/figs/ is deleted and regenerated on every invocation.
    * Auto-discovery (when --synthetic is not given) is performed by
      render_figures.py, which prefers the file whose temperature
      matches the run's mode default and falls back to the newest .npy
      with a warning otherwise.

Examples:
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM-bm_0
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM-bm_0 \
        --figs coupling_evol freq pca
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM-bm_0 \
        --synthetic results/CM/.../synthetic/align_T0.75_seed42.npy \
        --figs energy similarity
EOF
}

# ── Resolve repo root and chdir there ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

RUN_DIR="$1"
shift

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "error: run dir not found at '${RUN_DIR}'" >&2
    exit 1
fi
if [[ ! -f "${RUN_DIR}/model.npy" ]]; then
    echo "error: '${RUN_DIR}' has no model.npy" >&2
    exit 1
fi

SYNTHETIC=""
FIGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)
            SYNTHETIC="$2"; shift 2 ;;
        --figs)
            shift
            # Collect fig names until the next flag (anything starting with -).
            # Stops on -h / --foo / -X equally; render_figures.py validates
            # the names against its fixed `choices=ALL_FIGS` list.
            while [[ $# -gt 0 && "$1" != -* ]]; do
                FIGS+=("$1"); shift
            done
            ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${SYNTHETIC}" && ! -f "${SYNTHETIC}" ]]; then
    echo "error: synthetic alignment not found at '${SYNTHETIC}'" >&2
    exit 1
fi

# ── Regenerate figs/ from scratch ──────────────────────────────────────
echo "── Rendering figures for ${RUN_DIR} ─────────────────────────────"
rm -rf -- "${RUN_DIR}/figs"

PY_ARGS=("${RUN_DIR}")
if [[ -n "${SYNTHETIC}" ]]; then
    PY_ARGS+=(--synthetic-alignment "${SYNTHETIC}")
fi
if [[ ${#FIGS[@]} -gt 0 ]]; then
    PY_ARGS+=(--figs "${FIGS[@]}")
fi

python scripts/render_figures.py "${PY_ARGS[@]}"
echo
echo "── Done ─────────────────────────────────────────────────────────────"
echo "Figures: ${RUN_DIR}/figs/"
