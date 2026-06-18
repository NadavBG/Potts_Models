# Pipeline runbook: MSA → models → DCAlign → combined energy

This is the operational runbook for taking two families from raw MSAs all the
way to a two-model combined-energy score using the couplings-aware DCAlign
alignment. The README and `docs/initiate_two_model_energy.md` are the reference
(what the pieces are and the math); **this file is the step-by-step sequence**.

## Where things run

The guiding principle: **run as much as possible on the Mac.** The Midway
cluster is used for exactly one thing — the DCAlign couplings-aware alignment,
which is ~700× slower than fields-Viterbi (mean ~200 s/seq on the CM/PPIC query,
which is dominated by short `N<L` queries) and must be sharded across nodes.
Everything else is Mac work.

| Step | Machine | Needs Julia/cluster? |
|---|---|---|
| Train each model (`Snakefile`) | **Mac** | no |
| Set up combine run + build query (`Snakefile.combine`, partial) | **Mac** | no |
| Push models + query to Midway (`sync_models.sh push`) | **Mac** | no |
| DCAlign alignment (sharded array + gather) | **Midway** | **yes** |
| Finalize: validate + reclaim space | **Midway** (login) | no |
| Pull the cache back (`sync_models.sh pull`) | **Mac** | no |
| Score under both models + render (`Snakefile.combine all`) | **Mac** | no |

The combine **scoring** is Julia-free: the `dcalign` score branch only *reads*
the cached `alignments.tsv` and recomputes the energy in-frame via
`potts_energy`. So once the cache is on the Mac, nothing else touches the
cluster. (The other scoring methods — `map`, `marginal`, `in_frame` — run
entirely on the Mac with no cluster step at all; this runbook is specifically
the `dcalign` path.)

## Prerequisites

- **Mac:** the project installed in its uv venv
  (`uv pip install -e ".[plotting,analysis,dev,workflow]"`); see the README
  Quick start. Figures need `lab_plotting`.
- **Midway:** a clone of this repo at `$SBM_MIDWAY_REPO`
  (default `/project/ranganathanr/nadavbg/Potts_Models`) with a built `.venv`,
  `module load julia/1.10.2`, and a DCAlign clone at `DCALIGN_PATH`. Full
  cluster-side setup, env knobs, and cost numbers are in
  `pipeline/external/README.md`.
- The DCAlign driver `git pull`s and **refuses to submit on a dirty code tree**
  (reproducibility needs a clean HEAD), so commit and push any code changes
  before Phase D. The `combine/` and `results/` outputs are gitignored and move
  by rsync, not git.

Throughout, the worked example is the two families `CM-bm-dense` and
`PPIC-dense` combined into `combine-CM-PPIC-dcalign`.

---

## Phase A — Train the two models (Mac)

One model per family, each its own single-model pipeline run:

```sh
python scripts/iter.py run CM-bm-dense "base-model"
python scripts/iter.py run PPIC-dense "baseline"
```

Each lands a trained `model.npy` under `results/<fam>/iter-NNN-<tag>/`. (See the
README "How the pipeline works" for configs and what a run dir contains.) Note
the two run-dir paths — they go into the combine config next.

## Phase B — Set up the combine run + build the query (Mac)

Mint a combine iteration dir and build only the pre-alignment artifacts the
cluster driver needs. The combine config must point `models[].run_dir` at the
two Phase-A runs **you just made** and set `scoring.method: dcalign`. The shipped
`config/params_combine-CM-PPIC-dcalign.yaml` references specific iter dirs (e.g.
`results/CM-bm-dense/iter-002-base-model`); edit `run_dir` to match the iter dirs
your Phase-A runs actually produced (the `iter-NNN` index auto-increments).

```sh
# Mint the iteration dir (prints the snakemake command; --snakefile selects the
# two-model pipeline).
python scripts/iter.py new combine-CM-PPIC-dcalign "baseline" --snakefile Snakefile.combine
RUN_ROOT=combine/combine-CM-PPIC-dcalign/iter-001-baseline

# Build ONLY the pre-align targets. snapshot_config is an independent rule, so
# name it explicitly — building query.fasta alone does not pull it in.
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 \
    $RUN_ROOT/config_snapshot.yaml $RUN_ROOT/models.json $RUN_ROOT/query/query.fasta
```

This writes `config_snapshot.yaml`, `models.json`, `query/query.fasta`, and
`query/groups.json` — the four files the DCAlign driver's preflight checks.

## Phase C — Push models + query to Midway (Mac → Midway)

The cluster shards load each model from its `results/<fam>/iter/model.npy`, and
the driver reads the combine run's `config_snapshot.yaml` / `models.json` /
`query/`. Push both trees:

```sh
scripts/sync_models.sh push          # results/ (models) + combine/ (pre-align dir)
```

`push` syncs the durable artifacts of both trees with checksum verification on
the far side. For `combine/` it sends the small config/query and any cache
present, and **excludes the heavy DCAlign scratch** (`work/`, raw `shards/`,
`logs/`, `*.tar.zst`). Details in `docs/MODEL_SYNC.md`.

## Phase D — Run the DCAlign alignment on Midway

SSH to a Midway login node, `cd` to the repo, and submit. The driver plans the
shards, submits `2*n_shards` array tasks (one per model × shard) plus a gather
job chained `--dependency=afterok`, and writes the job IDs to
`dcalign/.shard_jids`.

```sh
# Login node. Fan-out (cpus=1) is the right lever — within-shard threading
# scales poorly; see pipeline/external/README.md "Parallelism & cost".
DCALIGN_CPUS=1 DCALIGN_MAX_CONCURRENT=512 \
    bash pipeline/external/run_dcalign_align.sh $RUN_ROOT
```

Monitor while it runs:

```sh
bash pipeline/job_tally.sh -w 10                 # live tally of array tasks by state
python pipeline/dcalign_run_stats.py $RUN_ROOT   # CPU-hours, longest shard, histogram (post-hoc)
```

When the gather job emails END, finalize from the login node — validate every
job COMPLETED via `sacct`, then reclaim space (delete the ~7–8 GB/model `work/`
scratch, compress raw shards + logs):

```sh
bash pipeline/external/finalize_dcalign_push.sh $RUN_ROOT
```

The gather step writes `dcalign/cache/<model>/alignments.tsv` (one row per
sequence: aligned frame, DCAlign energy, convergence flags) and a `meta.json`
provenance sidecar per model. Those two small files per model are all that
scoring needs.

## Phase E — Pull the cache back (Midway → Mac)

```sh
scripts/sync_models.sh pull          # brings combine/<run>/dcalign/cache/<model>/alignments.tsv (+ meta.json)
```

The 15 GB of `work/` scratch never transfers (excluded); the durable cache is
~0.5 MB per run. The pull verifies checksums locally.

## Phase F — Score under both models + render (Mac)

With the cache local, run the rest of the combine pipeline. The `score` rule
reads the cache, recomputes each energy in-frame (gauge-consistent — the
in-frame recompute vs DCAlign's own energy agrees to ≤5e-7, a standing manifest
canary), and `render_combine` makes the figure:

```sh
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 all
```

Outputs under `$RUN_ROOT` (tables in `data/`, manifests in `provenance/`, figures
in `figs/`; the top level keeps only the cluster-contract files —
`config_snapshot.yaml`, `models.json`, `query/` — plus `iteration_note.md`):

- `data/scores.tsv` — tidy, one row per (sequence × model): `E_A`/`E_B`, energy, flags
- `data/scores_detail.json` — per sequence: `E_A`, `E_B`, `E_tot`, diagnostics, best frame
- `data/alignments.txt` — human-readable; each sequence threaded into both frames
- `provenance/score_manifest.json` — provenance incl. the DCAlign `meta` and the agreement canary
- `figs/two_model_energy.pdf` — `E_A` vs `E_B` scatter + marginals
- `provenance/run_manifest.json` — aggregate

For `method: dcalign`, two diagnostics also land (Blocker-1 baseline + convergence):

- `data/dcalign_vs_inframe.{tsv,json}` + `figs/dcalign_vs_inframe.pdf` — DCAlign's energy
  vs the native in-frame energy per home-pair sequence (`ΔE>0` ⇒ DCAlign worse than the
  native frame); the scatter rings the not-converged points
- `data/dcalign_convergence.{tsv,json}` + `figs/dcalign_convergence.pdf` — non-convergence
  counts per (model, group) over all alignments (most non-convergence is cross-family)

Read `data/alignments.txt` first; a native of one family sits low on its own model's
axis and high on the other's.

---

## Notes

- **The cache makes scoring re-runnable for free.** Snakemake sees
  `alignments.tsv` already present, so re-running `all` only re-does
  `score`/`render`/manifest — no cluster round-trip. (On the live
  `combine-CM-PPIC-dcalign/iter-001-baseline`, the alignment is done and Phases
  E–F are the next step.)
- **Determinism:** per-(sequence, model) DCAlign is seeded from the master
  `seed`; the alignment is cached, so the score is reproducible from the cache
  regardless of where it runs.
- **Why two pipelines.** `Snakefile` (one MSA → one model) and
  `Snakefile.combine` (two models → energies) are intentionally separate — they
  have different DAGs, different validated config schemas
  (`workflow_config.py` vs `combine_config.py`), and separate output trees
  (`results/` vs `combine/`). They are not merged.

## See also

- `pipeline/external/README.md` — cluster mechanics, env knobs, measured cost,
  cache layout.
- `docs/MODEL_SYNC.md` — the rsync wrapper: what's synced/excluded per tree, the
  per-tree `SHA256SUMS` integrity check.
- `docs/initiate_two_model_energy.md` — the two-model energy spec and the math
  behind each scoring method.
- `README.md` — the single-model pipeline, configs, and the scoring-method
  table.
