#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────
# run_sbm.sh — train an SBM/BM Potts model and render its figures in one
# command. Output is a single self-describing run directory:
#
#   results/<fam>/<YYYY-MM-DD>_<label>_<idx>/
#       model.npy        — trained parameters (J, h, ...)
#       manifest.json    — git, seed, hashes, options, package versions
#       command.sh       — re-runnable copy of this invocation
#       fig_data/        — cached intermediates (artificial alignment, stats)
#       figs/            — rendered PDFs (one per plot type)
#
# Two important inputs:
#   <MSA_NPY>      a numerical alignment (NumPy .npy, shape N×L, int)
#   --prune PATH   optional pruning mask
#
# Everything else has a default. See `bash run_sbm.sh --help`.
# ─────────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage:
    bash scripts/run_sbm.sh <MODE> <MSA_NPY> [options] [-- extra train_sbm.py args]

Required positional:
    <MODE>                BM | SBM (case-insensitive). BM uses vanilla
                          gradient descent; SBM uses an L-BFGS-style update.
    <MSA_NPY>             path to the numerical MSA (.npy)

Optional:
    --prune PATH          path to a pruning mask .npy (the second
                          important input; off by default)
    --label NAME          label embedded in the run dir name
                          (default: family name derived from MSA filename)
    --seed N              master RNG seed (default: 42)
    --results-path DIR    output root (default: <repo>/results)
    --fam NAME            family name (default: stripped from
                          basename(MSA), e.g. MSA_CM.npy → CM)
    --N_iter N            number of GD iterations
                          (default: 400 for both modes)
    --N_chains N          number of MCMC chains
                          (default: 70 for both modes)
    --k_MCMC N            Metropolis sweeps per chain per step
                          (default: 10000)
    --no-figures          skip rendering (just train)
    -h, --help            this message

Anything after `--` is forwarded verbatim to scripts/train_sbm.py, so
you can reach the long tail of options (e.g. --m 20 --lambdJ 0.01).

Examples:
    bash scripts/run_sbm.sh SBM data/MSA_array/MSA_CM.npy --label CM-example
    bash scripts/run_sbm.sh BM  data/MSA_array/MSA_CM.npy --label CM-bm
    bash scripts/run_sbm.sh SBM data/MSA_array/MSA_CM.npy \
        --prune ./mask.npy --label CM-pruned -- --m 20 --lambdJ 0.01
EOF
}

# ── Resolve repo root and chdir there ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ────────────────────────────────────────────────────────────
# These apply to both modes. Mode-specific overrides are set after we
# parse <MODE> below.
N_ITER=400
N_CHAINS=70
K_MCMC=10000
SEED=42
N_AV=1
REP=1
TEST_TRAIN=0
THETA=0.3
PARAM_INIT=zero
LAMBD_J=0
LAMBD_H=0

PRUNE=""
LABEL=""
RESULTS_PATH=""
FAM=""
RENDER_FIGURES=1
EXTRA_ARGS=()

# ── Positional + named arg parsing ─────────────────────────────────────
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

if [[ $# -lt 2 ]]; then
    echo "error: <MODE> and <MSA_NPY> are required" >&2
    usage >&2
    exit 2
fi

MODE_RAW="$1"
MSA="$2"
shift 2

# Normalize and validate MODE.
MODE="$(echo "${MODE_RAW}" | tr '[:lower:]' '[:upper:]')"
case "${MODE}" in
    BM|SBM) ;;
    *)
        echo "error: MODE must be 'BM' or 'SBM' (got '${MODE_RAW}')" >&2
        exit 2
        ;;
esac

# Mode-specific default overrides. Today both modes share most defaults;
# this block is the place to diverge them later (e.g. different N_iter
# or theta) without surprising callers.
case "${MODE}" in
    BM)
        # BM uses --alpha (decaying learning rate) instead of --m. There's
        # no --alpha CLI flag on train_sbm.py at the moment; the BM default
        # alpha=0.2 is set in src/SBM/SBM_GD/SBM_proteins.py:ParseOptions.
        : # keep the shared defaults above
        ;;
    SBM)
        : # keep the shared defaults above
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prune)         PRUNE="$2"; shift 2 ;;
        --label)         LABEL="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --results-path)  RESULTS_PATH="$2"; shift 2 ;;
        --fam)           FAM="$2"; shift 2 ;;
        --N_iter)        N_ITER="$2"; shift 2 ;;
        --N_chains)      N_CHAINS="$2"; shift 2 ;;
        --k_MCMC)        K_MCMC="$2"; shift 2 ;;
        --no-figures)    RENDER_FIGURES=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; EXTRA_ARGS=("$@"); break ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# ── Validate inputs ─────────────────────────────────────────────────────
if [[ ! -f "${MSA}" ]]; then
    echo "error: MSA not found at '${MSA}'" >&2
    exit 1
fi
if [[ -n "${PRUNE}" && ! -f "${PRUNE}" ]]; then
    echo "error: --prune mask not found at '${PRUNE}'" >&2
    exit 1
fi

# ── Derive defaults from the MSA filename ───────────────────────────────
# data/MSA_array/MSA_CM.npy → fam=CM, label=CM (unless --fam / --label given)
if [[ -z "${FAM}" ]]; then
    msa_base="$(basename "${MSA}" .npy)"
    FAM="${msa_base#MSA_}"
fi
if [[ -z "${LABEL}" ]]; then
    LABEL="${FAM}"
fi

# ── Build train_sbm.py invocation ──────────────────────────────────────
TRAIN_ARGS=(
    "${FAM}"
    "${MSA}"
    --mod "${MODE}"
    --N_iter "${N_ITER}"
    --N_chains "${N_CHAINS}"
    --k_MCMC "${K_MCMC}"
    --N_av "${N_AV}"
    --rep "${REP}"
    --TestTrain "${TEST_TRAIN}"
    --theta "${THETA}"
    --ParamInit "${PARAM_INIT}"
    --lambdJ "${LAMBD_J}"
    --lambdh "${LAMBD_H}"
    --seed "${SEED}"
    --label "${LABEL}"
)

if [[ -n "${PRUNE}" ]]; then
    TRAIN_ARGS+=(--prune "${PRUNE}")
fi
if [[ -n "${RESULTS_PATH}" ]]; then
    TRAIN_ARGS+=(--results_path "${RESULTS_PATH}")
fi
# Append --m only for SBM (BM ignores it but ParseOptions tolerates extra
# keys; still, less noise this way).
if [[ "${MODE}" == "SBM" ]]; then
    TRAIN_ARGS+=(--m 1)
fi
# Forward anything the user supplied after `--`. argparse lets the last
# occurrence of a flag win, so user overrides above defaults.
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "── Training (${MODE} mode) ──────────────────────────────────────────"
echo "MSA:    ${MSA}"
[[ -n "${PRUNE}" ]] && echo "Prune:  ${PRUNE}"
echo "Label:  ${LABEL}"
echo "Seed:   ${SEED}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset; OpenMP picks default>}"
echo

# ── Invoke training, capturing the run-dir path ────────────────────────
# train_sbm.py prints "Run written: <path>" on stdout when each run dir
# is finalized. Tee the output so the user sees progress while we keep a
# copy for path extraction.
TRAIN_LOG="$(mktemp -t sbm_train.XXXXXX)"
trap 'rm -f "${TRAIN_LOG}"' EXIT

python scripts/train_sbm.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

# Extract every run dir written (handles --rep > 1 or N_chains-list, even
# though our defaults don't trigger those). `mapfile` is bash 4+, and
# macOS still ships bash 3.2 by default, so use a portable read-loop.
RUN_DIRS=()
while IFS= read -r line; do
    RUN_DIRS+=("${line}")
done < <(grep '^Run written: ' "${TRAIN_LOG}" | sed 's/^Run written: //')
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "error: training produced no run directory (training failed?)" >&2
    exit 1
fi

# ── Render figures ─────────────────────────────────────────────────────
if [[ "${RENDER_FIGURES}" -ne 1 ]]; then
    echo "── Skipping figures (--no-figures) ─────────────────────────────────"
    echo "Run dir(s):"
    for d in "${RUN_DIRS[@]}"; do echo "  ${d}"; done
    exit 0
fi

for run_dir in "${RUN_DIRS[@]}"; do
    echo
    echo "── Rendering figures for ${run_dir} ─────────────────────────────"
    python scripts/render_figures.py "${run_dir}"
done

echo
echo "── Done ─────────────────────────────────────────────────────────────"
for d in "${RUN_DIRS[@]}"; do echo "  ${d}"; done
