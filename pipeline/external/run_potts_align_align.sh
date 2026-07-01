#!/usr/bin/env bash
# Login-node driver for the potts_align align step of a two-model combine run
# (iter-003, docs/POTTS_ALIGN.md). NOT an sbatch job — run it on a Midway login
# node. It plans the shards, submits one Slurm array task per shard plus a gather
# job chained with --dependency=afterok, and prints monitor + finalize
# instructions. Pure numpy — no Julia, no DCAlign clone.
#
# Usage (on a Midway login node):
#   bash pipeline/external/run_potts_align_align.sh <run_root> [<n_shards>]
#
#   <run_root>  a built combine iteration dir, e.g.
#               combine/combine-CM-PPIC-potts/iter-001-potts-align-eval. It MUST
#               already contain config_snapshot.yaml, models.json,
#               query/query.fasta and query/groups.json — built ON THE MAC
#               (`snakemake -s Snakefile.combine ... <run_root>/query/query.fasta`)
#               and rsync'd up with `scripts/sync_models.sh push`.
#   <n_shards>  optional override of scoring.n_shards from the config.
#
# FLAT layout: every in-scope (query, model) pair is round-robined into n_shards,
# so the array has exactly n_shards tasks (task t = shard t; each task loads both
# models). The array reads the committed working tree on the shared filesystem, so
# the driver refuses a dirty tree and fast-forwards to origin (`git pull --ff-only`)
# — this is how code committed + pushed on the Mac reaches Midway.
#
# When the gather job mails END, finalize from this login node:
#   bash pipeline/external/finalize_potts_align.sh <run_root>
# then, ON THE MAC (not the login node — snakemake off the login), pull the cache
# and run the cheap score+render+manifest step (reads the cache):
#   scripts/sync_models.sh pull
#   snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
#       --config run_root=<run_root> --cores 8 all

set -euo pipefail
IFS=$'\n\t'

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
SHARD_JOB="${SCRIPT_DIR}/sbatch_potts_align_shard.sh"
GATHER_JOB="${SCRIPT_DIR}/sbatch_potts_align_gather.sh"
for f in "${SHARD_JOB}" "${GATHER_JOB}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing sbatch script: ${f}" >&2; exit 2; }
done

# Repo + git sync. The array runs the committed working tree on the shared FS, so
# refuse a dirty *code* tree (combine/ outputs are gitignored) and fast-forward to
# origin — this is how code edited on the Mac (committed + pushed there) reaches
# Midway. No julia module is loaded here, so git's HTTPS CA bundle stays intact.
REPO_DIR="$(git -C "${RUN_ROOT}" rev-parse --show-toplevel)"
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
    echo "FATAL: working tree is dirty; refuse to submit (reproduce-on-Midway needs a clean HEAD)." >&2
    echo "  Commit/stash on Midway, or — if you edited on the Mac — commit + push there, then re-run." >&2
    exit 7
fi
BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
git -C "${REPO_DIR}" fetch --quiet origin
if ! git -C "${REPO_DIR}" pull --ff-only --quiet origin "${BRANCH}"; then
    echo "FATAL: git pull --ff-only failed (diverged from origin/${BRANCH}?); reconcile then re-run." >&2
    exit 7
fi
echo "git sync: repo=${REPO_DIR} HEAD=$(git -C "${REPO_DIR}" rev-parse --short HEAD) (${BRANCH})"

VENV="${REPO_DIR}/.venv"
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: no venv at ${VENV}" >&2; exit 3; }
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# Preflight: catch in seconds what would otherwise fail inside a queued job.
echo "preflight: checking inputs..."
for rel in config_snapshot.yaml models.json query/query.fasta query/groups.json; do
    [[ -f "${RUN_ROOT}/${rel}" ]] || { echo "FATAL: missing ${RUN_ROOT}/${rel} (build the combine query first)." >&2; exit 5; }
done
echo "preflight OK"

# Shard count: CLI arg > config scoring.n_shards.
if [[ -n "${N_SHARDS_OVERRIDE}" ]]; then
    N_SHARDS="${N_SHARDS_OVERRIDE}"
else
    N_SHARDS="$(python -c "from SBM import combine_config as cc; print(cc.load_config('${RUN_ROOT}/config_snapshot.yaml').scoring.n_shards)")"
fi
[[ "${N_SHARDS}" =~ ^[0-9]+$ && "${N_SHARDS}" -ge 1 ]] || { echo "FATAL: bad n_shards '${N_SHARDS}'" >&2; exit 2; }

# Plan: write potts_align/shards_manifest.json (in-scope pairs, subsample, seeds).
python "${REPO_DIR}/scripts/wf/run_potts_align_shard.py" plan --run-root "${RUN_ROOT}" --n-shards "${N_SHARDS}"

ARRAY_MAX=$(( N_SHARDS - 1 ))
CONC="${POTTS_ALIGN_MAX_CONCURRENT:-${N_SHARDS}}"

# Submit from RUN_ROOT/potts_align so #SBATCH --output=logs/... lands there.
PA_DIR="${RUN_ROOT}/potts_align"
mkdir -p "${PA_DIR}/logs"
cd "${PA_DIR}"

echo
echo "submitting potts_align: ${N_SHARDS} shards (flat) = ${N_SHARDS} tasks (max concurrent ${CONC})"
ARRAY_JID=$(sbatch --parsable --array=0-${ARRAY_MAX}%${CONC} \
    --job-name="potts_align_shard" \
    "${SHARD_JOB}" "${RUN_ROOT}")
echo "  shard array : ${ARRAY_JID}"

GATHER_JID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JID}" \
    --job-name="potts_align_gather" \
    "${GATHER_JOB}" "${RUN_ROOT}")
echo "  gather      : ${GATHER_JID}  (after array ${ARRAY_JID})"

printf '%s\n%s\n' "${ARRAY_JID}" "${GATHER_JID}" > "${PA_DIR}/.shard_jids"

echo
echo "monitor:"
echo "  squeue -u \$USER"
echo "  sacct -j ${ARRAY_JID},${GATHER_JID} -X --format=JobID,JobName,Elapsed,State,ExitCode"
echo
echo "when gather mails END, finalize from this login node:"
echo "  bash ${SCRIPT_DIR}/finalize_potts_align.sh ${RUN_ROOT}"
echo "then, ON THE MAC (not the login node), pull the cache and score+render:"
echo "  scripts/sync_models.sh pull"
echo "  snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \\"
echo "      --config run_root=${RUN_ROOT} --cores 8 all"
