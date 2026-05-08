#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Worked example: build couplings (J) and fields (h) pruning masks for
# the CM family, then train a BM positive control that respects both,
# then sample a synthetic alignment from the trained model and render
# figures.
#
# Output lands at:
#   prune_output/<YYYY-MM-DD>_CM_<idx>/
#       <pct>_{Fij,Cij,SCA}_CM_SeqW_<theta>.npy        (J masks)
#       <pct>_{Fia,Dia}_CM_SeqW_<theta>.npy            (h masks)
#           + .manifest.json sidecars                  (from build_mask.py)
#   example_output/CM/<YYYY-MM-DD>_CM-bm-pruned_<idx>/
#       model.npy, manifest.json, command.sh           (from run_sbm.sh)
#       synthetic/align_T{0.75,1}_seed42.{npy,json}    (from sample_sbm.sh)
#       figs/{coupling_evol,correlations,pca}.pdf,
#           figs/inputs/{stats_*.npy, sources.json}    (from render_sbm.sh)
#
# This is the BM ("positive control") regime from Summary Note 3:
# m=20, lambda_J=lambda_h=0.01, N_chains=100. run_sbm.sh applies these
# defaults automatically when MODE=BM, so no manual overrides are needed.
#
# Requires the [sca] optional extra (`pip install -e ".[sca]"`).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
FULL_CM_ALG="${REPO_ROOT}/data/MSA_array/MSA_CM.npy"

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Build pruning masks from the full alignment (98% of params zeroed
#    per kind). Couplings: SCA (the conserved-correlation strategy);
#    fields: Dia (the per-site KL-divergence strategy). build_mask.py
#    creates "<path>/<run-id>/" and prints "Run dir: <abs path>" to
#    stdout; we scrape that to find the mask files.
MASK_LOG="$(mktemp -t cm_example_mask.XXXXXX)"
TRAIN_LOG="$(mktemp -t cm_example_train.XXXXXX)"
trap 'rm -f "${MASK_LOG}" "${TRAIN_LOG}"' EXIT

python build_mask.py \
    --alg "${FULL_CM_ALG}" \
    --theta 0.7 \
    --lbda 0.03 \
    --strategies "fij" "cij" "sca" "fia" "dia" \
    --ext ".npy" \
    --label "CM" \
    --path "$(pwd)/prune_output" \
    --percent 98 \
    | tee "${MASK_LOG}"

MASK_RUN_DIR="$(grep '^Run dir: ' "${MASK_LOG}" | tail -n 1 | sed 's/^Run dir: //')"
if [[ -z "${MASK_RUN_DIR}" || ! -d "${MASK_RUN_DIR}" ]]; then
    echo "error: could not locate mask run dir from build_mask.py output" >&2
    exit 1
fi

# 2. Train a BM model that uses the SCA-derived J mask and the
#    Dia-derived h mask. The BM defaults in run_sbm.sh (m=20,
#    lambda=0.01, N_chains=100) are appropriate when most parameters
#    are constrained to zero. We tee stdout so we can pull the run-dir
#    path out of the "Run dir:" lines it prints (mirrors the same
#    pattern run_sbm.sh uses internally).
bash "${REPO_ROOT}/scripts/run_sbm.sh" \
    BM "${FULL_CM_ALG}" \
    --prune-J "${MASK_RUN_DIR}/98.00_SCA_CM_SeqW_0.7.npy" \
    --prune-h "${MASK_RUN_DIR}/98.00_Dia_CM_SeqW_0.7.npy" \
    --label CM-bm-pruned \
    --results-path "$(pwd)/example_output" \
    | tee "${TRAIN_LOG}"

RUN_DIR="$(grep '^Run dir: ' "${TRAIN_LOG}" | tail -n 1 | sed 's/^Run dir: //')"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "error: could not locate run dir from training output" >&2
    exit 1
fi

# 3. Sample synthetic alignments. By default sample_sbm.sh produces
#    one alignment per temperature in {0.75, 1.0} (N=2000 each). Seed
#    defaults to the manifest's master seed.
bash "${REPO_ROOT}/scripts/sample_sbm.sh" "${RUN_DIR}"

# 4. Render figures. Default mode renders every figure whose data is
#    present: coupling_evol always; correlations + pca because the
#    sampler just wrote alignments under <RUN_DIR>/synthetic/.
bash "${REPO_ROOT}/scripts/render_sbm.sh" "${RUN_DIR}"
