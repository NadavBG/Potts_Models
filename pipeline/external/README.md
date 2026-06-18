# DCAlign align step — Midway cluster scripts

Cluster-scale couplings-aware alignment for the two-model `combine` pipeline
(spec `docs/initiate_two_model_energy.md` §10.9). DCAlign is ~700× slower than
fields-Viterbi, so the expensive alignment is sharded over a Slurm array and
cached on disk; the `score` step then reads the cache and recomputes energies
in-frame (cheap, gauge-consistent). Per-sequence cost is heavy-tailed and
dominated by the `N<L` (query shorter than the model) regime, where `palign`
auto-bumps to 5000 sweeps: on the CM/PPIC combine query the mean is **~200 s/seq**
(almost all CM queries are `N<L`), not the 19 s spike median on raw naturals.
See "Parallelism & cost" below for the measured numbers and the recommended
fan-out launch.

This mirrors `../Make_Alignment/pipeline/external/`: a login-node driver submits
an `sbatch --array` + a gather job chained with `--dependency=afterok`, and a
finalizer validates + compresses the result.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `run_dcalign_align.sh` | login node | git-pull, preflight, plan shards, submit array + gather, write `.shard_jids` |
| `sbatch_dcalign_shard.sh` | compute node (array) | align one `(model, shard)`; resumable; no network |
| `sbatch_dcalign_gather.sh` | compute node | merge shards → `cache/<model>/alignments.tsv` + provenance |
| `finalize_dcalign_push.sh` | login node | sacct-validate, reclaim space (cache leaves Midway via rsync, not git) |

The Python entrypoints they call live in `scripts/wf/run_dcalign_shard.py`
(modes `plan` / `run`) and `scripts/wf/run_dcalign_gather.py`; the Julia driver
is `src/SBM/julia/run_dcalign.jl`.

## One run

```bash
# 0. Build the combine query first (cheap). The driver's preflight requires all
#    three of config_snapshot.yaml, models.json, query/query.fasta + query/groups.json;
#    snapshot_config is an independent rule, so name it explicitly (building only
#    query.fasta does NOT pull it in). RUN_ROOT is a combine iteration dir — mint one
#    with `scripts/iter.py new combine-CM-PPIC-dcalign "<tag>" --snakefile Snakefile.combine`.
RUN_ROOT=combine/combine-CM-PPIC-dcalign/iter-001-baseline
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 \
    $RUN_ROOT/config_snapshot.yaml $RUN_ROOT/models.json $RUN_ROOT/query/query.fasta

# 1. Submit the align step (login node).
bash pipeline/external/run_dcalign_align.sh $RUN_ROOT          # or: ... $RUN_ROOT <n_shards>

# 2. When the gather job emails END, finalize (login node): validate + reclaim space.
bash pipeline/external/finalize_dcalign_push.sh $RUN_ROOT

# 3. Score on the Mac: pull the durable cache, then run the combine pipeline locally.
#    (Scoring is Julia-free — it just reads alignments.tsv. See docs/PIPELINE.md.)
#    Run these ON THE MAC, and re-set RUN_ROOT there (it was a Midway shell var above).
RUN_ROOT=combine/combine-CM-PPIC-dcalign/iter-001-baseline     # on the Mac
scripts/sync_models.sh pull                                    # Midway -> Mac
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 all
```

## Environment assumptions (Midway)

- A built `.venv` at the repo root (SBM editable-installed: `uv venv && uv pip install -e ".[workflow]"`).
- `module load julia/1.10.2`; `JULIA_DEPOT_PATH=/scratch/midway3/nadavbg/julia_depot`.
- A DCAlign clone at `DCALIGN_PATH` (default `<repo>/../DCAlign`), instantiated
  (`julia --project=$DCALIGN_PATH -e 'using Pkg; Pkg.instantiate()'`).
- `--account=pi-ranganathanr --partition=caslake` (CPU-only).
- Driver env knobs: `DCALIGN_CPUS=N` sets `--cpus-per-task` per shard task (exported
  to Julia as `JULIA_NUM_THREADS`); `DCALIGN_MEM=NNG` overrides memory;
  `DCALIGN_MAX_CONCURRENT=N` caps array concurrency (default 16); `DCALIGN_TINY=1`
  shrinks walltime for a smoke (memory is NOT shrunk — the `deltan` seed `dist` is
  full-size regardless). Defaults (no override): 4 cpus / 16 G / 8 h.

> **git + julia gotcha:** `module load julia` puts Julia's mbedTLS `libgit2` on
> `LD_LIBRARY_PATH`, which can't find the system CA bundle and breaks `git` over
> HTTPS (`BADCERT_NOT_TRUSTED`). The login-node scripts export
> `GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt` to fix it.

## Parallelism & cost (measured on the real CM/PPIC models, 2026-06-17)

Two levers parallelise the work across the cluster (QOS caslake allows 4800 cores /
100 nodes / 1000 submitted jobs):

1. **Shard fan-out** — `2*n_shards` independent array tasks across nodes. The
   primary lever.
2. **Within-shard threading** — `run_dcalign.jl` threads a shard's sequences over
   `--cpus-per-task` (`Threads.@threads :dynamic`). Set via `DCALIGN_CPUS`.

**Threading scales poorly:** each alignment is one indivisible chunk and the cost is
heavy-tailed, so a thread stuck on a slow `N<L` sequence bounds the task. Measured
on the real models: **1.7× on 4 threads, 2.9× on 8 threads (~36% core efficiency)**.
Threading is correct (byte-identical answers, validated against single-thread) and
useful only if you would otherwise hit the 1000-job cap — which 3600 alignments do
not. **So prefer fan-out with `cpus-per-task=1`.**

**Memory (`lambda_spec="deltan"`):** every shard rebuilds the model's full seed prior at
startup — a `(N_seed,L,L)` int64 `dist` array (~1.8 GB for PPIC's 26701-seq seed) — and
`N≈L` inflates `palign`'s working set, so per-task peak exceeds 4 GB *regardless of how
few queries the shard holds* (measured OOM at >4.13 GB on a 4 GB PPIC shard, 2026-06-18).
The default floor is now **16 G/task**. Cost: caslake is 4 GB/core, so `1 core + 16 G`
bills as **4 core-equivalents** and packs ~12 tasks/node instead of 48 — same real compute,
~4× the SUs/node-hours. The smoke uses the same models + prior, so its `sacct MaxRSS` is the
true per-task peak; tune `DCALIGN_MEM` down to ~1.5× that (likely 6–8 G) before the full run
to recover packing efficiency.

**Recommended full run** (`combine-CM-PPIC-dcalign`, 1800 seqs × 2 models = 3600
alignments, `n_shards=256` → 512 one-core tasks, ~7 seqs each):

```bash
RR=combine/combine-CM-PPIC-dcalign/iter-001-baseline   # query already staged
DCALIGN_CPUS=1 DCALIGN_MAX_CONCURRENT=512 bash pipeline/external/run_dcalign_align.sh $RR
```

| Config | Service units | Wall (tunable) | Core efficiency |
|---|---|---|---|
| **fan-out** `cpus=1`, 256 shards | **~160–230 core-hours** | ~15–60 min (512 tasks at once) | ~100% |
| threaded `cpus=8`, 32 shards | ~640 core-hours (3–4× more) | similar | ~36% |

(core-hours ≈ SUs on caslake at 1 SU/core-hr — confirm against your RCC allocation.)
Estimates scale the threaded smoke (96 alignments, 17.1 billed core-hours at `cpus=8`)
by alignment count; the query distribution is identical (same seeded subsample, larger
cap). The figure step (`render_combine`) is separate and needs `lab_plotting`
(`pip install -e /home/nadavbg/lab-plotting` + `lab_plotting.install_styles()`) — or
render it on the Mac; it is not part of the alignment.

## Cache layout (under the gitignored `combine/<run>/`)

```
combine/<run>/dcalign/
  shards_manifest.json            sorted ids, round-robin into n_shards
  cache/<model>/
    meta.json                     L, q, model_sha256, maxiter, seed, pcount, lambda_spec, dcalign_commit, julia_version
    shards/shard_NNN.tsv          raw per-shard driver output (resumable; compressed by finalize)
    alignments.tsv                gathered, one row per seq_id (read by the dcalign score branch)
  logs/  .shard_jids  gather_status.json
```
