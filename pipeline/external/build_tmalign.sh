#!/usr/bin/env bash
# Build the Zhang-lab TMalign binary once, on a Midway LOGIN node (needs
# outbound network to fetch the source; compute nodes have none). Produces
# pipeline/bin/TMalign, which the characterize compare stage calls. The binary
# is gitignored (built per-machine); this script is the reproducible recipe.
#
# Usage (login node):
#   bash pipeline/external/build_tmalign.sh
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_DIR="${REPO_ROOT}/pipeline/bin"
OUT="${BIN_DIR}/TMalign"
SRC_URL="https://zhanggroup.org/TM-align/TMalign.cpp"

mkdir -p "${BIN_DIR}"

if [[ -x "${OUT}" ]]; then
    echo "TMalign already built: ${OUT}"
    "${OUT}" 2>&1 | head -2 || true
    exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
SRC="${tmpdir}/TMalign.cpp"

echo "fetching ${SRC_URL} ..."
if command -v wget >/dev/null 2>&1; then
    wget -q -O "${SRC}" "${SRC_URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${SRC}" "${SRC_URL}"
else
    echo "ERROR: neither wget nor curl available" >&2
    exit 3
fi
[[ -s "${SRC}" ]] || { echo "ERROR: downloaded TMalign.cpp is empty" >&2; exit 3; }

# Pick a C++ compiler; load the gcc module if g++ is not already on PATH.
if ! command -v g++ >/dev/null 2>&1; then
    module load gcc 2>/dev/null || true
fi
command -v g++ >/dev/null 2>&1 || { echo "ERROR: no g++ available (module load gcc)" >&2; exit 4; }

echo "compiling with $(g++ --version | head -1) ..."
g++ -O3 -ffast-math -o "${OUT}" "${SRC}"
chmod +x "${OUT}"
echo "built: ${OUT}"
"${OUT}" 2>&1 | head -2 || true
