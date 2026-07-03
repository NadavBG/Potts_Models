#!/usr/bin/env bash
# One ESMFold fold shard on a beagle3 GPU. Submitted as an array job by
# run_characterize.sh (or run_esmfold_probe.sh) — not by hand. Array task t IS
# the shard: fold_sequences.py folds the round-robin shard t of the FASTA.
# Resumable: records whose <id>.pdb exists and are already in the shard scores
# TSV are skipped, so a TIME_LIMIT kill leaves a valid partial cache and a
# re-submit continues.
#
# Compute nodes have NO outbound network: HF weights must be pre-cached
# (pipeline/external/prefetch_esmfold.sh) and we force HF_HUB_OFFLINE=1.
#
# Interpreter: the RCC-provided AI env (torch+cu128, transformers 5.6.2,
# EsmForProteinFolding). We do NOT use bioM3_env — its interpreter segfaults on
# startup on Midway. Override with ESMFOLD_PYTHON if needed.
#
# Usage (set by the driver):
#   sbatch --array=0-<N-1>%<conc> sbatch_esmfold_shard.sh \
#       <fasta> <out_structures> <out_scores_dir> <group> <n_shards> <degap 0|1> [<limit>]

#SBATCH --job-name=esmfold_shard
#SBATCH --account=pi-ranganathanr
#SBATCH --partition=beagle3
#SBATCH --qos=beagle3
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# ESMFold(3B) fp16 folds ~90 aa in ~2 s on an A100; a 64-shard split of the 27k
# PPIC naturals is ~420 seqs/shard (~15 min) + ~2 min model load. 2 h is ample
# and resume makes a TIME_LIMIT non-fatal.
#SBATCH --time=02:00:00
#SBATCH --output=logs/esmfold_shard_%A_%a.log
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=nadavbg@rcc.uchicago.edu

set -euo pipefail
IFS=$'\n\t'

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <fasta> <out_structures> <out_scores_dir> <group> <n_shards> <degap 0|1> [<limit>]" >&2
    exit 2
fi
FASTA="$1"; OUT_STRUCT="$2"; OUT_SCORES_DIR="$3"; GROUP="$4"
N_SHARDS="$5"; DEGAP="$6"; LIMIT="${7:-}"

# SLURM runs the batch script from /var/spool/slurm, so ${BASH_SOURCE[0]} is
# useless here — derive the repo from the (absolute, in-repo) FASTA argument's
# git toplevel, exactly like the potts_align shards.
REPO_ROOT="$(git -C "$(dirname "${FASTA}")" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${REPO_ROOT}" && -d "${REPO_ROOT}/scripts/characterize" ]] || {
    echo "ERROR: cannot locate repo root from FASTA path ${FASTA}" >&2; exit 3; }

ESMFOLD_PY="${ESMFOLD_PYTHON:-/software/python-miniforge-25.3.0-el8-x86_64/envs/AI/bin/python}"
[[ -x "${ESMFOLD_PY}" ]] || { echo "ERROR: ESMFold python not found: ${ESMFOLD_PY}" >&2; exit 3; }

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1          # compute node: use cached weights, never phone home
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${OUT_SCORES_DIR}"
SCORES="${OUT_SCORES_DIR}/shard_${SHARD}.tsv"

DEGAP_FLAG=()
[[ "${DEGAP}" == "1" ]] && DEGAP_FLAG=(--degap)
LIMIT_FLAG=()
[[ -n "${LIMIT}" ]] && LIMIT_FLAG=(--limit "${LIMIT}")

echo "host=$(hostname)  shard=${SHARD}/${N_SHARDS}  group=${GROUP}  fasta=${FASTA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

exec "${ESMFOLD_PY}" "${REPO_ROOT}/scripts/characterize/fold_sequences.py" \
    --fasta "${FASTA}" \
    --out-structures "${OUT_STRUCT}" \
    --out-scores "${SCORES}" \
    --group "${GROUP}" \
    --n-shards "${N_SHARDS}" --shard "${SHARD}" \
    "${DEGAP_FLAG[@]}" "${LIMIT_FLAG[@]}"
