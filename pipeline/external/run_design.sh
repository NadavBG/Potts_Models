#!/usr/bin/env bash
# Login-node driver for the two-model design anneal (docs/DESIGN_TWO_MODEL.md).
# NOT an sbatch job — run it on a Midway login node. It plans the shards, submits
# one Slurm array task per shard plus a gather job chained with
# --dependency=afterok, and prints monitor + finalize instructions. Pure numpy —
# no Julia.
#
# Usage (on a Midway login node):
#   bash pipeline/external/run_design.sh <run_root> [<n_shards>]
#
#   <run_root>  a combine iteration dir whose design spec is already built ON THE
#               MAC, e.g. combine/combine-CM-PPIC-potts/iter-001-potts-align-eval.
#               It MUST contain design/design_config.json (written by
#               `design_two_model.py --emit-config-only` or the pipeline's
#               design_config stage) and the models it references (results/... via
#               `scripts/sync_models.sh push`).
#   <n_shards>  optional override of design.n_shards from config_snapshot.yaml.
#
# Chains 0..n_chains-1 are round-robined into n_shards, so the array has exactly
# n_shards tasks (task t = shard t). Each chain's seed is master_seed + chain_index
# (pinned in design_config.json), so every chain is reproducible and independent.
# The array reads the committed working tree on the shared filesystem, so the
# driver refuses a dirty tree and fast-forwards to origin (git pull --ff-only) —
# this is how code committed + pushed on the Mac reaches Midway.
#
# When the gather job mails END, finalize from this login node:
#   bash pipeline/external/finalize_design.sh <run_root>
# then, ON THE MAC (not the login node — snakemake off the login), pull the
# gathered artifacts and render the figures:
#   scripts/sync_models.sh pull
#   python scripts/render_design.py --design-dir <run_root>/design --figs-dir <run_root>/figs

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
DESIGN_DIR="${RUN_ROOT}/design"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARD_JOB="${SCRIPT_DIR}/sbatch_design_shard.sh"
GATHER_JOB="${SCRIPT_DIR}/sbatch_design_gather.sh"
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

# Preflight: the design spec must exist. The referenced models are checked below
# (in the same python call that sizes the walltime).
echo "preflight: checking inputs..."
CONFIG_JSON="${DESIGN_DIR}/design_config.json"
[[ -f "${CONFIG_JSON}" ]] || { echo "FATAL: missing ${CONFIG_JSON} (build the design spec on the Mac first)." >&2; exit 5; }

# Shard count: CLI arg > config_snapshot.yaml design.n_shards. Read the raw YAML
# (not the validated schema) so a stale unrelated key in an old snapshot can't
# block the design run.
if [[ -n "${N_SHARDS_OVERRIDE}" ]]; then
    N_SHARDS="${N_SHARDS_OVERRIDE}"
else
    SNAP="${RUN_ROOT}/config_snapshot.yaml"
    [[ -f "${SNAP}" ]] || { echo "FATAL: no n_shards given and ${SNAP} absent; pass <n_shards> explicitly." >&2; exit 5; }
    N_SHARDS="$(python -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])) or {}; v=(d.get('design') or {}).get('n_shards'); print(v if v is not None else '')" "${SNAP}")"
    [[ -n "${N_SHARDS}" ]] || { echo "FATAL: config_snapshot.yaml has no design.n_shards; pass <n_shards> explicitly." >&2; exit 5; }
fi
[[ "${N_SHARDS}" =~ ^[0-9]+$ && "${N_SHARDS}" -ge 1 ]] || { echo "FATAL: bad n_shards '${N_SHARDS}'" >&2; exit 2; }

# Preflight the referenced model/seed_msa files AND size the walltime in one pass.
# Prints: "<n_chains> <steps> <polish_schedule> <chains_per_shard> <HH:MM:SS>".
# Walltime = 2 x chains_per_shard x (steps*15us + polish_seconds), floored at 30 min
# and capped at 36 h (caslake max); resume makes an under-estimate non-fatal.
# Command substitution (not process substitution) so `set -e` aborts if the python
# preflight exits nonzero (e.g. a referenced model file is missing).
PREFLIGHT_OUT="$(
python - "${CONFIG_JSON}" "${REPO_DIR}" "${N_SHARDS}" <<'PY'
import json, math, sys
from pathlib import Path

cfg_path, repo_dir, n_shards = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

# Model + seed-MSA files must exist under the repo root (repo-root-relative paths).
missing = []
for key in ("model_a_path", "model_b_path", "seed_msa_a", "seed_msa_b"):
    rel = cfg.get(key)
    if rel and not (repo_dir / rel).is_file():
        missing.append(f"{key}={rel}")
if missing:
    sys.stderr.write("FATAL: design_config.json references missing files:\n  " +
                     "\n  ".join(missing) + "\n  (run scripts/sync_models.sh push on the Mac)\n")
    raise SystemExit(6)

n_chains = int(cfg["n_chains"])
steps = int(cfg["schedule"]["n_steps"])
do_polish = bool(cfg["do_polish"])
sched = cfg["polish_schedule"]
# Conservative per-chain polish cost by schedule (docs/DESIGN_TWO_MODEL.md "Cost / SU").
polish_s = {"fast": 6.0, "auto": 220.0, "default": 220.0, "thorough": 440.0}.get(sched, 220.0) if do_polish else 0.0

chains_per_shard = math.ceil(n_chains / n_shards)
per_chain_s = steps * 15e-6 + polish_s
secs = 2.0 * chains_per_shard * per_chain_s
secs = max(1800.0, min(secs, 36 * 3600.0))       # floor 30 min, cap 36 h (caslake max)
secs = int(math.ceil(secs))
hh, rem = divmod(secs, 3600)
mm, ss = divmod(rem, 60)
print(n_chains, steps, sched if do_polish else "off", chains_per_shard, f"{hh:02d}:{mm:02d}:{ss:02d}")
PY
)"
# The python prints its 5 fields space-separated; the script header sets
# IFS=$'\n\t' (no space), so scope a space-inclusive IFS to just this read —
# otherwise the whole line lands in N_CHAINS and WALLTIME comes back empty.
IFS=$' \t\n' read -r N_CHAINS STEPS POLISH_SCHED CHAINS_PER_SHARD WALLTIME <<< "${PREFLIGHT_OUT}"
[[ -n "${WALLTIME}" ]] || { echo "FATAL: preflight produced no walltime (see errors above)." >&2; exit 6; }
echo "preflight OK: ${N_CHAINS} chains, ${STEPS} steps, polish=${POLISH_SCHED}, ${CHAINS_PER_SHARD} chains/shard (max)"

# Plan: write design/shards_manifest.json (chain -> shard assignment).
python "${REPO_DIR}/scripts/wf/run_design_shard.py" plan --run-dir "${DESIGN_DIR}" --n-shards "${N_SHARDS}"

ARRAY_MAX=$(( N_SHARDS - 1 ))
CONC="${DESIGN_MAX_CONCURRENT:-${N_SHARDS}}"

# Submit from DESIGN_DIR so #SBATCH --output=logs/... lands under design/logs.
mkdir -p "${DESIGN_DIR}/logs"
cd "${DESIGN_DIR}"

echo
echo "submitting design: ${N_SHARDS} shards = ${N_SHARDS} tasks (max concurrent ${CONC}), --time=${WALLTIME}"
ARRAY_JID=$(sbatch --parsable --array=0-${ARRAY_MAX}%${CONC} \
    --time="${WALLTIME}" \
    --job-name="design_shard" \
    "${SHARD_JOB}" "${RUN_ROOT}")
echo "  shard array : ${ARRAY_JID}"

GATHER_JID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JID}" \
    --job-name="design_gather" \
    "${GATHER_JOB}" "${RUN_ROOT}")
echo "  gather      : ${GATHER_JID}  (after array ${ARRAY_JID})"

printf '%s\n%s\n' "${ARRAY_JID}" "${GATHER_JID}" > "${DESIGN_DIR}/.shard_jids"

echo
echo "monitor:"
echo "  squeue -u \$USER"
echo "  pipeline/job_tally.sh -w 10"
echo "  sacct -j ${ARRAY_JID},${GATHER_JID} -X --format=JobID,JobName,Elapsed,State,ExitCode"
echo
echo "when gather mails END, finalize from this login node:"
echo "  bash ${SCRIPT_DIR}/finalize_design.sh ${RUN_ROOT}"
echo "then, ON THE MAC (not the login node), pull the artifacts and render:"
echo "  scripts/sync_models.sh pull"
echo "  python scripts/render_design.py --design-dir ${RUN_ROOT}/design --figs-dir ${RUN_ROOT}/figs"
