# DCAlign align step — Midway cluster scripts

Cluster-scale couplings-aware alignment for the two-model `combine` pipeline
(spec `docs/initiate_two_model_energy.md` §10.9). DCAlign is ~700× slower than
fields-Viterbi (median 19 s/seq), so the expensive alignment is sharded over a
Slurm array and cached on disk; the `score` step then reads the cache and
recomputes energies in-frame (cheap, gauge-consistent).

This mirrors `../Make_Alignment/pipeline/external/`: a login-node driver submits
an `sbatch --array` + a gather job chained with `--dependency=afterok`, and a
finalizer validates + compresses the result.

## Files

| File | Where it runs | What it does |
|---|---|---|
| `run_dcalign_align.sh` | login node | git-pull, preflight, plan shards, submit array + gather, write `.shard_jids` |
| `sbatch_dcalign_shard.sh` | compute node (array) | align one `(model, shard)`; resumable; no network |
| `sbatch_dcalign_gather.sh` | compute node | merge shards → `cache/<model>/alignments.tsv` + provenance |
| `finalize_dcalign_push.sh` | login node | sacct-validate, reclaim space, optional `--push` |

The Python entrypoints they call live in `scripts/wf/run_dcalign_shard.py`
(modes `plan` / `run`) and `scripts/wf/run_dcalign_gather.py`; the Julia driver
is `src/SBM/julia/run_dcalign.jl`.

## One run

```bash
# 0. Build the combine query first (cheap; produces config_snapshot.yaml,
#    models.json, query/). RUN_ROOT is a combine iteration dir.
RUN_ROOT=combine/combine-CM-PPIC/iter-001-dcalign
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 $RUN_ROOT/query/query.fasta $RUN_ROOT/models.json

# 1. Submit the align step (login node).
bash pipeline/external/run_dcalign_align.sh $RUN_ROOT          # or: ... $RUN_ROOT <n_shards>

# 2. When the gather job emails END, finalize (login node).
bash pipeline/external/finalize_dcalign_push.sh $RUN_ROOT      # add --push to commit the cache

# 3. Run the cheap score step (reads the cache).
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-dcalign.yaml \
    --config run_root=$RUN_ROOT --cores 4 all
```

## Environment assumptions (Midway)

- A built `.venv` at the repo root (SBM editable-installed: `uv venv && uv pip install -e ".[workflow]"`).
- `module load julia/1.10.2`; `JULIA_DEPOT_PATH=/scratch/midway3/nadavbg/julia_depot`.
- A DCAlign clone at `DCALIGN_PATH` (default `<repo>/../DCAlign`), instantiated
  (`julia --project=$DCALIGN_PATH -e 'using Pkg; Pkg.instantiate()'`).
- `--account=pi-ranganathanr --partition=caslake` (CPU-only; DCAlign is single-threaded per seq).
- Set `DCALIGN_TINY=1` for a small-resource smoke run; `DCALIGN_MAX_CONCURRENT=N`
  to cap array concurrency (default 16).

> **git + julia gotcha:** `module load julia` puts Julia's mbedTLS `libgit2` on
> `LD_LIBRARY_PATH`, which can't find the system CA bundle and breaks `git` over
> HTTPS (`BADCERT_NOT_TRUSTED`). The login-node scripts export
> `GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt` to fix it.

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
