#!/usr/bin/env bash
# Login-node driver for the DCAlign align step of a two-model combine run
# (spec §10.9). NOT an sbatch job — run it on a Midway login node. It pulls the
# latest code, plans the shards, submits one Slurm array task per (model, shard)
# plus a gather job chained with --dependency=afterok, and prints monitor +
# finalize instructions.
#
# Usage (on a Midway login node):
#   bash pipeline/external/run_dcalign_align.sh <run_root> [<n_shards>]
#
#   <run_root>  a built combine iteration dir, e.g.
#               combine/combine-CM-PPIC/iter-001-baseline. It MUST already
#               contain config_snapshot.yaml, models.json, query/query.fasta and
#               query/groups.json (run the combine pipeline up through build_query
#               first: `snakemake -s Snakefile.combine ... <run_root>/query/query.fasta`).
#   <n_shards>  optional override of scoring.n_shards from the config.
#
# Every query is scored under BOTH models, so the array has 2*n_shards tasks:
# task t -> model_index = t // n_shards, shard = t % n_shards.
#
# When the gather job mails END, finalize from this login node:
#   bash pipeline/external/finalize_dcalign_push.sh <run_root>
#
# Then run the cheap score step (reads the cache):
#   snakemake -s Snakefile.combine --configfile config/params_<name>.yaml \
#       --config run_root=<run_root> --cores 4 all

set -euo pipefail
IFS=$'\n\t'

# Julia's bundled mbedTLS libgit2 (on LD_LIBRARY_PATH after `module load julia`)
# can't find the system CA bundle and breaks `git` over HTTPS. Point git at the
# system bundle so pull/push work regardless of module state.
export GIT_SSL_CAINFO="${GIT_SSL_CAINFO:-/etc/pki/tls/certs/ca-bundle.crt}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <run_root> [<n_shards>]" >&2
    exit 2
fi

RUN_ROOT="$(realpath "$1")"
N_SHARDS_OVERRIDE="${2:-}"

if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "ERROR: run_root not found: ${RUN_ROOT}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARD_JOB="${SCRIPT_DIR}/sbatch_dcalign_shard.sh"
GATHER_JOB="${SCRIPT_DIR}/sbatch_dcalign_gather.sh"
for f in "${SHARD_JOB}" "${GATHER_JOB}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing sbatch script: ${f}" >&2; exit 2; }
done

# Repo + git sync (refuse on a dirty *code* tree; combine/ outputs are gitignored).
REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
echo "git sync: repo=${REPO_DIR}"
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
    echo "FATAL: working tree is dirty; refuse to submit (reproduce-on-Midway needs a clean HEAD)." >&2
    echo "  Run 'git -C ${REPO_DIR} status' and commit/stash first." >&2
    exit 7
fi
BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
git -C "${REPO_DIR}" fetch --quiet origin
if ! git -C "${REPO_DIR}" pull --ff-only --quiet origin "${BRANCH}"; then
    echo "FATAL: git pull --ff-only failed (diverged from origin/${BRANCH}?)." >&2
    exit 8
fi
echo "git sync: HEAD=$(git -C "${REPO_DIR}" rev-parse --short HEAD) (${BRANCH})"

# Env: venv (SBM) + julia + DCAlign clone + depot.
VENV="${REPO_DIR}/.venv"
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: no venv at ${VENV} (build it: uv venv && uv pip install -e .)" >&2; exit 3; }
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
module load julia/1.10.2
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/scratch/midway3/nadavbg/julia_depot}"
export DCALIGN_PATH="${DCALIGN_PATH:-$(realpath "${REPO_DIR}/../DCAlign")}"

# Preflight (catch in seconds what would otherwise fail inside a queued job).
echo "preflight: checking inputs, DCAlign clone, and julia..."
for rel in config_snapshot.yaml models.json query/query.fasta query/groups.json; do
    [[ -f "${RUN_ROOT}/${rel}" ]] || { echo "FATAL: missing ${RUN_ROOT}/${rel} (build the combine query first)." >&2; exit 5; }
done
[[ -f "${DCALIGN_PATH}/Project.toml" ]] || { echo "FATAL: DCALIGN_PATH=${DCALIGN_PATH} is not a DCAlign clone." >&2; exit 6; }
julia --version
if ! julia --project="${DCALIGN_PATH}" -e 'using DCAlign' 2>/dev/null; then
    echo "FATAL: 'using DCAlign' failed under ${DCALIGN_PATH}. Run Pkg.instantiate() in that project." >&2
    exit 6
fi
echo "preflight OK"

# Shard count: CLI arg > config scoring.n_shards.
if [[ -n "${N_SHARDS_OVERRIDE}" ]]; then
    N_SHARDS="${N_SHARDS_OVERRIDE}"
else
    N_SHARDS="$(python -c "from SBM import combine_config as cc; print(cc.load_config('${RUN_ROOT}/config_snapshot.yaml').scoring.n_shards)")"
fi
[[ "${N_SHARDS}" =~ ^[0-9]+$ && "${N_SHARDS}" -ge 1 ]] || { echo "FATAL: bad n_shards '${N_SHARDS}'" >&2; exit 2; }

# Plan: write dcalign/shards_manifest.json (sorted ids, round-robin into shards).
python "${REPO_DIR}/scripts/wf/run_dcalign_shard.py" plan --run-root "${RUN_ROOT}" --n-shards "${N_SHARDS}"

ARRAY_MAX=$(( 2 * N_SHARDS - 1 ))
CONC="${DCALIGN_MAX_CONCURRENT:-16}"

# Resource overrides for the shard array. DCALIGN_CPUS sets --cpus-per-task, which
# the wrapper exports as JULIA_NUM_THREADS, so each shard task aligns that many of
# its sequences in parallel (run_dcalign.jl threads over the shard). One Julia
# startup is amortised over all those cores, so a full node (e.g. DCALIGN_CPUS=48
# on caslake) is the efficient unit; fan across nodes with more shards + a higher
# DCALIGN_MAX_CONCURRENT. DCALIGN_MEM overrides memory; DCALIGN_TINY=1 shrinks
# walltime for a smoke (memory is NOT shrunk; see below). Defaults (no override):
# the #SBATCH 4 cpus / 8G / 8h.
SHARD_OVERRIDES=()
GATHER_OVERRIDES=()
if [[ "${DCALIGN_TINY:-0}" == "1" ]]; then
    SHARD_OVERRIDES=(--time=00:30:00 --mem="${DCALIGN_MEM:-8G}" --cpus-per-task="${DCALIGN_CPUS:-2}")
    GATHER_OVERRIDES=(--time=00:15:00 --mem=2G)
    echo "DCALIGN_TINY=1: short walltime (cpus=${DCALIGN_CPUS:-2}, mem=${DCALIGN_MEM:-8G}, time=30min)"
elif [[ -n "${DCALIGN_CPUS:-}" ]]; then
    # cpus*2 GB, floored at 8G: lambda_spec="deltan" builds DCAlign.deltan_prior at
    # shard startup (~1.8 GB dist array for PPIC's 26701-seq seed) AND N~=L inflates
    # palign's working set — measured per-task peak ~4.5 GB at cpus=2 (iter-004 smoke,
    # 2026-06-18); the old 4G cap OOM'd. The floor is a per-task need, not a thread count.
    # 8G = ~1.75x headroom and bills as 2 core-equivalents on caslake (4 GB/core) vs 16G's
    # 4. DCALIGN_MEM still overrides. (spec §10.13)
    SHARD_MEM_GB=$(( DCALIGN_CPUS * 2 )); (( SHARD_MEM_GB < 8 )) && SHARD_MEM_GB=8
    SHARD_MEM="${DCALIGN_MEM:-${SHARD_MEM_GB}G}"
    SHARD_OVERRIDES=(--cpus-per-task="${DCALIGN_CPUS}" --mem="${SHARD_MEM}")
    echo "per-shard threads: cpus-per-task=${DCALIGN_CPUS} (JULIA_NUM_THREADS), mem=${SHARD_MEM}"
fi

# Submit from RUN_ROOT/dcalign so #SBATCH --output=logs/... lands there.
DCALIGN_DIR="${RUN_ROOT}/dcalign"
mkdir -p "${DCALIGN_DIR}/logs"
cd "${DCALIGN_DIR}"

echo
echo "submitting DCAlign align: ${N_SHARDS} shards x 2 models = $((2*N_SHARDS)) tasks (max concurrent ${CONC})"
ARRAY_JID=$(sbatch --parsable --array=0-${ARRAY_MAX}%${CONC} \
    --job-name="dcalign_shard" \
    "${SHARD_OVERRIDES[@]}" \
    "${SHARD_JOB}" "${RUN_ROOT}" "${N_SHARDS}")
echo "  shard array : ${ARRAY_JID}"

GATHER_JID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JID}" \
    --job-name="dcalign_gather" \
    "${GATHER_OVERRIDES[@]}" \
    "${GATHER_JOB}" "${RUN_ROOT}")
echo "  gather      : ${GATHER_JID}  (after array ${ARRAY_JID})"

printf '%s\n%s\n' "${ARRAY_JID}" "${GATHER_JID}" > "${DCALIGN_DIR}/.shard_jids"

echo
echo "monitor:"
echo "  squeue -u \$USER"
echo "  sacct -j ${ARRAY_JID},${GATHER_JID} -X --format=JobID,JobName,Elapsed,State,ExitCode"
echo
echo "when gather mails END, finalize from this login node:"
echo "  bash ${SCRIPT_DIR}/finalize_dcalign_push.sh ${RUN_ROOT}"
echo "then run the cheap score step:"
echo "  snakemake -s Snakefile.combine --config run_root=${RUN_ROOT} --cores 4 all"
