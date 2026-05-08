#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────
# sample_sbm.sh — generate synthetic alignments from a trained model.
#
# Two modes, selected by flag:
#
# 1. Standard sampling (default).
#    Output lands in:
#      <RUN_DIR>/synthetic/align_T<T>_seed<seed>[_<label>].npy
#      <RUN_DIR>/synthetic/align_T<T>_seed<seed>[_<label>].json (params)
#    By default samples at BOTH T=0.75 and T=1.0 (one alignment per T) —
#    every downstream analysis in this project compares the two. Pass
#    `--temperature T1 [T2 ...]` to override. Default N is 2000 sequences
#    regardless of training MSA size.
#
# 2. ProteinMPNN foldability sweep (--mpnn-sweep).
#    Samples N sequences at each of a temperature ladder (default
#    0.1..1.0 step 0.1; 100 seqs/T), builds interpretability controls
#    (WT, uniform random, shuffled WT, natural MSA bootstrap), and
#    scores them with the upstream ProteinMPNN repo. Output lands in:
#      <RUN_DIR>/synthetic/mpnn_sweep_seed<seed>/
#    See docs/MPNN_FOLDABILITY.md for setup (clone of dauparas/ProteinMPNN
#    + PROTEINMPNN_PATH env var) and the figure pipeline.
#    Mutually exclusive with the standard --temperature/--N flags.
#
# This is a thin wrapper around scripts/sample_sbm.py — all flags are
# forwarded verbatim.
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
                        (default: 2000)
    --temperature T...  one or more sampling temperatures; one .npy +
                        .json is written per value
                        (default: 0.75 1.0)
    --delta_t N         Metropolis sweeps per chain
                        (default: options0.k_MCMC from model.npy)
    --seed N            master RNG seed
                        (default: master seed from manifest)
    --label NAME        suffix added to default filename
    --output PATH       full output .npy path; only valid with a single
                        temperature
    --force             overwrite an existing alignment / sidecar at the
                        target path (default: refuse, to protect prior
                        samples)
    -h, --help          this message

ProteinMPNN sweep mode (alternate, mutually exclusive with the above
sampling flags):
    --mpnn-sweep                  enable sweep mode
    --mpnn-temperatures T...      ladder (default: 0.1 0.2 ... 1.0)
    --mpnn-N-per-T N              sequences per T and per control (100)
    --mpnn-controls C...          subset of {wt,random,shuffled,natural,none}
    --mpnn-pdb PATH               PDB to score against
                                  (default: data/structures/1ECM.pdb)
    --mpnn-chain CHAIN            PDB chain id (default: A)
    --mpnn-wt-fasta PATH          WT FASTA (default: data/fasta/CM.fasta;
                                  first record taken as the WT)
    --mpnn-path PATH              clone of github.com/dauparas/ProteinMPNN
                                  (default: $PROTEINMPNN_PATH)
    --mpnn-model-name NAME        weights basename (default: v_48_020)
    --mpnn-device {cpu,cuda,mps}  default: auto-detect (upstream picks)
    --mpnn-backbone-noise EPS     augmentation epsilon (default: 0.0)
    --mpnn-skip-scoring           sample + write controls + manifest only

Examples:
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0 --temperature 1.0
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0 \
        --temperature 0.5 0.75 1.0 1.5
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0 \
        --N 5000 --label highT --temperature 1.5

    # ProteinMPNN sweep with defaults (10 T × 100 seqs + 4 controls):
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0 --mpnn-sweep

    # Smoke test — sample only, no scoring:
    bash scripts/sample_sbm.sh results/CM/2026-05-06_CM_0 --mpnn-sweep \
        --mpnn-temperatures 0.5 1.0 --mpnn-N-per-T 5 \
        --mpnn-controls wt random --mpnn-skip-scoring
EOF
}

# ── Resolve repo root and chdir there ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Machine-specific defaults (gitignored) ─────────────────────────────
# Place exports for PROTEINMPNN_PATH / PROTEINMPNN_PYTHON in
# scripts/sample_sbm.local.sh. The file is gitignored so machine paths
# stay out of tracked code.
if [[ -f scripts/sample_sbm.local.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/sample_sbm.local.sh
fi

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

# ── Forward all remaining args to sample_sbm.py verbatim ────────────────
# We don't case-match individual flags here because --temperature takes a
# variable number of values (nargs="+") and case-by-case parsing would
# fight with that. argparse on the Python side validates and explains
# unknown flags.
python scripts/sample_sbm.py "${RUN_DIR}" "$@"
