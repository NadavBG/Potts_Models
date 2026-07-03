#!/usr/bin/env bash
# Warm the HuggingFace cache with facebook/esmfold_v1 (~2.5 GB) on a Midway
# LOGIN node. beagle3 compute nodes have no outbound network, so the GPU fold
# array runs with HF_HUB_OFFLINE=1 and needs these weights already cached on the
# shared home filesystem (~/.cache/huggingface, which is shared to compute nodes).
# Uses the RCC AI env python (bioM3_env's interpreter segfaults on Midway); the
# HF cache is interpreter-agnostic. Override with ESMFOLD_PYTHON.
#
# Usage (login node):
#   bash pipeline/external/prefetch_esmfold.sh
set -euo pipefail
IFS=$'\n\t'

ESMFOLD_PY="${ESMFOLD_PYTHON:-/software/python-miniforge-25.3.0-el8-x86_64/envs/AI/bin/python}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

[[ -x "${ESMFOLD_PY}" ]] || { echo "ERROR: ESMFold python not found: ${ESMFOLD_PY}" >&2; exit 2; }
mkdir -p "${HF_HOME}"

echo "python : ${ESMFOLD_PY}"
echo "HF_HOME: ${HF_HOME}"
echo "downloading facebook/esmfold_v1 weights + tokenizer (~2.5 GB) ..."

"${ESMFOLD_PY}" - <<'PY'
import os
from transformers import AutoTokenizer, EsmForProteinFolding
mid = "facebook/esmfold_v1"
print("tokenizer ...", flush=True)
AutoTokenizer.from_pretrained(mid)
print("model weights ...", flush=True)
EsmForProteinFolding.from_pretrained(mid)   # populates HF cache; no GPU needed
print("cached under", os.environ.get("HF_HOME"))
PY
echo "ESMFold weights cached."
