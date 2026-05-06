#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────
# render_sbm.sh — render figures for a trained SBM/BM run.
#
# Layout:
#   <RUN_DIR>/figs/                  ← always regenerated
#       inputs/
#           render_figures.py             (canonical copy of the renderer)
#           stats_<align_stem>.npy        (one cache per alignment)
#           sources.json                  (paths + sha256 of model.npy
#                                          and every synthetic alignment)
#       coupling_evol.pdf
#       correlations.pdf                  (rows = temperatures × cols = orders)
#       pca.pdf                           (1×(1+N_temps) panels)
#       <test-set-only figures>.pdf
#
# By default, render every figure whose data is present in the run:
#   * coupling_evol always (depends only on model.npy)
#   * correlations, pca if at least one synthetic alignment exists
#     (auto-discovered: every .npy under <RUN_DIR>/synthetic/)
#   * energy, similarity, diversity, length additionally if the run was
#     trained with Test/Train > 0
# Pass --figs NAME [NAME ...] to render an explicit subset; in that
# mode, requesting a figure whose data is missing is an error.
# ─────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
    bash scripts/render_sbm.sh <RUN_DIR> [options]

Required:
    <RUN_DIR>           a run directory produced by scripts/run_sbm.sh

Optional:
    --synthetic PATH... one or more synthetic alignment .npy files
                        (default: auto-discover every .npy under
                         <RUN_DIR>/synthetic/)
    --figs NAME ...     figure types to render
                        (default: every figure whose data is present)
                        all choices: coupling_evol, correlations, pca,
                        energy, similarity, diversity, length
    -h, --help          this message

Behaviour:
    * <RUN_DIR>/figs/ is deleted and regenerated on every invocation.
    * Multiple alignments under synthetic/ become multi-panel
      comparisons (correlations becomes rows × 3 grid; pca becomes
      1 × (1 + N_temps) panels).

Examples:
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM_0
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM_0 \
        --figs coupling_evol correlations pca
    bash scripts/render_sbm.sh results/CM/2026-05-06_CM_0 \
        --synthetic results/CM/.../synthetic/align_T0.75_seed42.npy \
                    results/CM/.../synthetic/align_T1_seed42.npy \
        --figs correlations
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

SYNTHETICS=()
FIGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)
            shift
            # Collect alignment paths until the next flag (anything
            # starting with -). render_figures.py validates that each
            # path exists.
            while [[ $# -gt 0 && "$1" != -* ]]; do
                SYNTHETICS+=("$1"); shift
            done
            ;;
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

if [[ ${#SYNTHETICS[@]} -gt 0 ]]; then
    for s in "${SYNTHETICS[@]}"; do
        if [[ ! -f "${s}" ]]; then
            echo "error: synthetic alignment not found at '${s}'" >&2
            exit 1
        fi
    done
fi

# ── Regenerate figs/ from scratch ──────────────────────────────────────
echo "── Rendering figures for ${RUN_DIR} ─────────────────────────────"
rm -rf -- "${RUN_DIR}/figs"

PY_ARGS=("${RUN_DIR}")
if [[ ${#SYNTHETICS[@]} -gt 0 ]]; then
    PY_ARGS+=(--synthetic-alignment "${SYNTHETICS[@]}")
fi
if [[ ${#FIGS[@]} -gt 0 ]]; then
    PY_ARGS+=(--figs "${FIGS[@]}")
fi

python scripts/render_figures.py "${PY_ARGS[@]}"
echo
echo "── Done ─────────────────────────────────────────────────────────────"
echo "Figures: ${RUN_DIR}/figs/"
