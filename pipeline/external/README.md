# potts_align align step — Midway cluster scripts

Optional cluster scripts for pre-building the `potts_align` alignment cache of a
two-model **combine** run at scale (`scoring.method: potts_align`; spec
`docs/POTTS_ALIGN.md`). Everything here is **pure numpy** — no Julia, no external
tool. The align step only needs the cluster because aligning a large query set
(the full CM+PPIC naturals ≈ tens of core-hours, `docs/POTTS_ALIGN.md` §6.8) is
embarrassingly parallel over (query, model) pairs; a small query set is faster to
score directly on the Mac (`score_two_models.py --method potts_align`, no cache).

> DCAlign has been retired (`docs/two_model_progress.md`). The old DCAlign cluster
> scripts and their README are archived under `.archive/pipeline/external/`.

## Scripts

| script | where | what it does |
|---|---|---|
| `run_potts_align_align.sh` | login node | plan shards, submit the array + an `afterok` gather, write `.shard_jids` |
| `sbatch_potts_align_shard.sh` | compute node (array) | align one shard's `(query, model)` pairs; resumable; no network |
| `sbatch_potts_align_gather.sh` | compute node | merge shards → `cache/<model>/alignments.tsv`; errors on any missing pair / canary failure |
| `finalize_potts_align.sh` | login node | sacct-validate every task COMPLETED, then reclaim space (tar+zstd raw shards + logs) |

Python entrypoints: `scripts/wf/run_potts_align_shard.py` (`plan` / `run`) and
`scripts/wf/run_potts_align_gather.py`. Live-monitor the array with
`pipeline/job_tally.sh -w 10`.

## Flow (Mac → Midway → Mac)

```sh
# 1. Mac: build the pre-align artifacts of a combine iter dir, then push.
RR=combine/combine-CM-PPIC-potts/iter-001-<tag>
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=$RR --cores 8 \
    $RR/config_snapshot.yaml $RR/models.json $RR/query/query.fasta
scripts/sync_models.sh push        # models (results/) + the pre-align combine dir

# 2. Midway login node: shard + gather (cpus=1 fan-out; no Julia).
bash pipeline/external/run_potts_align_align.sh $RR
#    ...monitor with `pipeline/job_tally.sh -w 10`; when gather mails END:
bash pipeline/external/finalize_potts_align.sh $RR

# 3. Mac: pull the cache and run the rest of the pipeline (score reads the cache).
scripts/sync_models.sh pull
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=$RR --cores 8 all
```

The gather writes `<run_root>/potts_align/cache/<model>/alignments.tsv` (one row
per scored `(query_id, model)`: frame, energy, engine, `is_global_exact`, seed) +
a `meta.json` sidecar. Scoring reads only those and recomputes each energy
in-frame as a `≤1e-6` gauge canary — no cluster round-trip after the pull. Cost
model, the `pa_cross_subsample_*` / `n_random` knobs, and the full runbook are in
`docs/POTTS_ALIGN.md` §11.

## Cluster env

- Account/partition `pi-ranganathanr` / `caslake`; a built `.venv` at the repo
  root (`uv pip install -e ".[workflow]"` is enough — no Julia).
- Fan out at `cpus-per-task=1` over the array (each shard is a single-core numpy
  loop); size `n_shards` so each task is ~1–2 h. Compute nodes have no outbound
  network, so the shard/gather jobs do no git; the cache moves by rsync
  (`scripts/sync_models.sh`), not git.
