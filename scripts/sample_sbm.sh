#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────
# sample_sbm.sh — generate a synthetic alignment from a trained model.
#
# Output lands in:
#   <RUN_DIR>/synthetic/align_T<T>_seed<seed>[_<label>].npy
#   <RUN_DIR>/synthetic/align_T<T>_seed<seed>[_<label>].json   (params)
#
# Mode-aware temperature default: BM samples at T=0.75, SBM at T=1.0
# (Summary Note 3). Override with --temperature.
#
# This is a thin wrapper around scripts/sample_sbm.py.
# ─────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
    bash scripts/sample_sbm.sh <RUN_DIR> [options]

Required:
    <RUN_DIR>           a run directory produced by scripts/run_sbm.sh
                        (must contain model.npy and manifest.json)

Optional:
    --N N               number of synthetic sequences
                        (default: size of training MSA)
    --temperature T     sampling temperature
                        (default: 0.75 for BM, 1.0 for SBM)
    --delta_t N         Metropolis sweeps per chain
                        (default: options0.k_MCMC from model.npy)
    --seed N            master RNG seed
                        (default: master seed from manifest)
    --label NAME        suffix added to default filename
    --output PATH       full output .npy path
                        (overrides default location under synthetic/)
    --force             overwrite an existing alignment / sidecar at the
                        target path (default: refuse, to protect prior
                        samples)
    -h, --help          this message

Examples:
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM-bm_0
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM-bm_0 --temperature 1.0
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM-sbm_0 \
        --N 5000 --label highT --temperature 1.5
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

# ── Forward all remaining args to sample_sbm.py ────────────────────────
PY_ARGS=("${RUN_DIR}")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --N|--temperature|--delta_t|--seed|--label|--output)
            PY_ARGS+=("$1" "$2"); shift 2 ;;
        --force)
            PY_ARGS+=("$1"); shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

python scripts/sample_sbm.py "${PY_ARGS[@]}"
