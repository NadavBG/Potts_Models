# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`SBM` infers the fields `h` and couplings `J` of a Potts model on a multiple-sequence alignment (MSA) by L-BFGS against MCMC-estimated statistics. The Python package wraps two C++/OpenMP MCMC kernels that are compiled at install time via scikit-build-core + CMake.

Two training regimes share the L-BFGS algorithm and differ only in parameter values (per Summary Note 3):

| Parameter | BM (positive control) | SBM (stochastic regularization) |
| --- | --- | --- |
| `m` (L-BFGS memory) | 20 | 1 |
| `lambda_J`, `lambda_h` | 0.01 | 0 |
| `N_chains` | 100 | 50 |

`N_iter=400`, zero-initialized parameters, and inference temperature `T=1` are shared. Inference temperature is always 1 — the model is meant to reproduce data statistics at T=1. Sampling synthetic alignments is a separate downstream step (`sample_sbm.sh`); it always produces both T=0.75 (low-T, mode-collapsed) and T=1.0 (the model's native fit) regardless of training mode, so every figure compares the two. Vanilla gradient descent is available via `Optimizer="GD"` (`--optimizer GD` on the CLI) but is rarely needed.

## Where things live

| If you need… | Look at |
|---|---|
| The one-command pipeline (recommended) | `Snakefile` driven by `config/params_<run_name>.yaml`; mint+run an iteration with `python scripts/iter.py run <run_name> "<tag>"`. See "Snakemake pipeline" below. |
| The pipeline config schema | `src/SBM/workflow_config.py` (`SBMRunConfig`, validated; `from_dict` rejects unknown keys) |
| The per-stage Snakemake wrappers | `scripts/wf/run_*.py` (thin; call the CLIs below) + `scripts/wf/_common.py` |
| The iteration helper | `scripts/iter.py` + `src/SBM/iteration.py` (mints `results/<run_name>/iter-NNN-<tag>/`) |
| The manual entry points (still work; called by the Snakefile too) | `scripts/run_sbm.sh` (train), `scripts/sample_sbm.sh` (synthetic MSA), `scripts/render_sbm.sh` (figures), `scripts/render_msa_stats.py` (MSA-only figure) |
| The training CLI | `scripts/train_sbm.py` (auto-names `results/<fam>/<YYYY-MM-DD>_<label>_<idx>/`; `--run-dir DIR` overrides with an exact path for the pipeline) |
| The synthetic-sampling CLI | `scripts/sample_sbm.py` (writes one `<run_dir>/synthetic/align_T<T>_seed<seed>.npy` + JSON sidecar per requested temperature; default = both 0.75 and 1.0; default N=2000) |
| The figure renderer | `scripts/render_figures.py` (writes `<run_dir>/figs/` and `<run_dir>/figs/inputs/`; the bash wrapper `render_sbm.sh` deletes `figs/` first so each call regenerates) |
| The optimizer entry point | `src/SBM/SBM_GD/SBM_proteins.py:SBM(align, options)` |
| The MCMC sampler driver (Python) | `src/SBM/utils/utils.py:Create_modAlign` |
| The C++ MCMC kernels | `src/SBM/MonteCarlo/MCMC_Potts/MonteCarlo_PottsMod.cpp` (full) and `MCMC_PottsProf/MonteCarlo_PottsProfMod.cpp` (profile-only) |
| The packed-vector encoding | `src/SBM/utils/utils.py:Wj` / `Jw` |
| The zero-sum gauge transform | `src/SBM/utils/utils.py:Zero_Sum_Gauge` |
| Statistics / reweighting | `src/SBM/utils/utils.py:CalcWeights`, `CalcStatsWeighted`, `CalcThreeCorrWeighted` |
| Plot recipes (used by `render_figures.py`) | `src/SBM/utils/utils_plot.py:plot_stats` (7 modes; `correlations` is rows=temperatures × cols=1st/2nd/3rd order, `pca` is 1×(1+N_temps)). The `mpnn` figure is separate: `src/SBM/utils/utils_mpnn_plot.py:plot_mpnn_foldability` reads `mpnn_scores.json` directly. |
| The ProteinMPNN foldability sweep | `scripts/mpnn_sweep.py` (orchestrator; called by `sample_sbm.py --mpnn-sweep`) and `src/SBM/utils/mpnn_score.py` (subprocess driver for upstream `protein_mpnn_run.py --score_only`). See `docs/MPNN_FOLDABILITY.md`. |
| Run-level provenance helpers | `src/SBM/provenance.py` |
| The pruning CLI | `pruning/build_mask.py` (auto-names a per-run subdir; `--out-file PATH` writes a single mask to an exact path for the pipeline) |
| The figure-save helpers | `scripts/lab_plotting.py` (`save_figure`, `panel_label`, `LAB_COLORS`) |
| The CM worked example (pipeline) | `config/params_CM-bm-dense.yaml` (plain BM, no pruning; the `params_CM-bm-*` variants add pruning); the legacy `pruning/CM_example.sh` (chains `run_sbm.sh` → `sample_sbm.sh` → `render_sbm.sh`) still works |
| Energy of a sequence under one/two models | `src/SBM/energy/` — `model.load_model` (loads + re-gauges), `potts.potts_energy` (in-frame, wraps `compute_energies`), `hmm.ProfileHMM` (alignment proposal: forward / Viterbi / FFBS), `score.score_sequence` / `score.score_two_models` (`method ∈ {in_frame, map, marginal, dcalign}`). Spec: `docs/initiate_two_model_energy.md` |
| The two-model scoring CLI | `scripts/score_two_models.py` (single sequence or batch FASTA → `E_A`, `E_B`, `E_tot` + diagnostics; `--method`, `--weights`, `--n-samples`, `--seed`) |
| The two-model `combine` pipeline | `Snakefile.combine` driven by `config/params_combine-*.yaml` (validated by `src/SBM/combine_config.py`); run with `python scripts/iter.py run <name> "<tag>" --snakefile Snakefile.combine`. See "Combine pipeline" below. |
| The DCAlign couplings-aware align step (cluster) | bridge `src/SBM/utils/dcalign_score.py` (mirrors `mpnn_score.py`) + Julia driver `src/SBM/julia/run_dcalign.jl` (run with `--project=<DCAlign clone>`); cluster wrappers `pipeline/external/run_dcalign_align.sh` (login driver) + `sbatch_dcalign_{shard,gather}.sh` + `finalize_dcalign_push.sh` (validate + reclaim space; no longer git-pushes); Python entrypoints `scripts/wf/run_dcalign_shard.py` (`plan`/`run`) + `run_dcalign_gather.py`. See `pipeline/external/README.md` (cluster mechanics). |
| The end-to-end MSA→DCAlign→combine runbook (Mac/Midway split) | `docs/PIPELINE.md` |
| Model transfer Mac ↔ Midway (models **and** DCAlign caches) | `scripts/sync_models.sh` (checksummed rsync; `push`/`pull`/`verify`/`status`; covers both `results/` and `combine/`). Not in git. See "Model transfer" below and `docs/MODEL_SYNC.md`. |

`src/SBM/__init__.py` is empty by design — users import submodules directly (`SBM.SBM_GD.SBM_proteins`, `SBM.utils.utils`, `SBM.provenance`).

## Data flow

```
aligned FASTA (msa_fasta)
   │
   ▼  load_fasta  (encode_msa rule → <run_dir>/inputs/msa.npy; non-canonical seqs dropped)
MSA (.npy)
   │
   ▼  CalcWeights, CalcStatsWeighted
fi, fij  ────────────────────────────────────────┐
                                                 │
W (packed h+J)  ──┐                              │
                  ▼                              │
        Create_modAlign  ──►  mc.MC / mcp.MC     │
        (artificial alignment from current model)│
                  │                              │
                  ▼                              │
            fi_mod, fij_mod                      │
                  │                              │
                  ▼   (subtract; pruning mask)   │
              gradient ◄──────────────────────────┘
                  │
                  ▼  L-BFGS or vanilla GD step
              new W → Wj/Jw → next iteration
```

After training, parameters are zero-sum gauged before averaging across replicates.

## Build, install, run

```
uv python install 3.12
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[plotting,analysis,dev,workflow]"
```

The `workflow` extra adds Snakemake + PyYAML for the pipeline (below). It installs into this same uv venv — **not** conda — and the Snakefile deliberately has no `conda:` directive.

`pip install -e .` works equivalently. Runtime deps are pinned in `pyproject.toml`; `requirements.lock` records exact pins for reproducible installs (`uv pip sync requirements.lock` then `uv pip install -e . --no-deps`).

**Where compute runs (Mac-primary).** Default to running everything on the Mac: training, sampling, figures, MPNN, MSA stats, and combine *scoring* (`map`/`marginal`/`in_frame`, and the `dcalign` score branch, which just reads a cache). The Midway cluster is used for **one** thing — the DCAlign couplings-aware *alignment* (`method: dcalign`), which is ~700× slower and is sharded over a Slurm array. That alignment writes a cache; you `sync_models.sh pull` it to the Mac and score locally. So when something here mentions Midway/`sbatch`/Julia, it almost always means the DCAlign alignment step specifically — not the rest of the pipeline. End-to-end runbook: `docs/PIPELINE.md`.

**macOS toolchain.** AppleClang has no OpenMP, so `pyproject.toml` forces `cmake/macos_llvm.cmake`, which hard-codes `/opt/homebrew/opt/llvm` and `libomp`. `brew install llvm libomp ninja cmake` is required; Intel-Mac or non-Homebrew prefixes need the toolchain file edited. On Linux, `python3-dev`, GCC/G++ with OpenMP, CMake, and Ninja are sufficient.

**Don't use conda-forge / miniforge Python.** Its `python@3.12` libpython has an ABI quirk that segfaults during numpy's `import_array()` when our C++ extension is loaded. Use uv-managed standalone CPython or Homebrew's `python@3.12`.

**C++ rebuilds.** scikit-build-core's editable install does not auto-rebuild on `.cpp` changes. After editing kernel source, run `uv pip install -e . --force-reinstall --no-deps`.

**Python.** 3.11+ (`requires-python = ">=3.11"`).

**Tests.** The energy/scoring module (`src/SBM/energy/`) has a pytest suite at `tests/test_energy.py` — run it after touching that module:

```
.venv/bin/python -m pytest tests/test_energy.py -q
```

It is pure numpy/scipy (no MCMC) and covers spec §6 (gauge invariance, in-frame base case, MAP≈marginal when unambiguous, ordering sanity, IS diagnostics) plus the **DP anchor**: the profile-HMM forward log-Z, FFBS sample frequencies, marginal-IS estimate, and Viterbi path are all checked against *brute-force enumeration of every alignment* of a tiny query — this is the gold-standard check for `hmm.py`, so no `pyhmmer`/HMMER dependency is needed.

The rest of the codebase has no unit-test suite. After non-trivial changes, run the pipeline smoke test (tiny config: 5 L-BFGS iters, 4 chains, N=50, MPNN with `skip_scoring`):

```
snakemake --configfile config/params_tiny.yaml --cores 8 all
# combine pipeline smoke test (scores a handful of CM+PPIC seqs under both models):
snakemake -s Snakefile.combine --configfile config/params_combine-tiny.yaml --cores 4 all
```

The first exercises the whole DAG — encode_msa → mask → train → sample(×2) → mpnn → render → manifest, plus the independent `msa_stats` branch — and lands deterministic outputs under `results/tiny/` (assert on `inputs/msa.npy`, `model.npy`, `manifest.json`, `synthetic/align_T*.npy`, `figs/*.pdf`, `msa_stats.pdf`, `run_manifest.json`). The legacy `bash pruning/CM_example.sh` still works but is superseded by the pipeline.

**Pruning workflow** lives in `pruning/` with its own `README.md` and `CM_example.sh`. The `"sca"` strategy depends on `pysca`, gated behind the `[sca]` optional-dependency group; `"fij"` and `"cij"` don't need it.

## Snakemake pipeline

One validated YAML config = one run. `config/params_<run_name>.yaml` drives the `Snakefile`; `src/SBM/workflow_config.py` validates it (unknown keys are an error) and the thin `scripts/wf/run_*.py` wrappers call the existing CLIs with deterministic output paths.

```
python scripts/iter.py run CM-bm-dense "baseline"     # mint iter dir + run everything
# equivalently, two steps:
python scripts/iter.py new CM-bm-dense "baseline"     # prints the snakemake command
snakemake --configfile config/params_CM-bm-dense.yaml \
          --config run_root=results/CM-bm-dense/iter-001-baseline --cores 8 all
```

- **Run dirs:** `RUN_ROOT = config.get("run_root") or results/<run_name>/`. The iteration helper mints `results/<run_name>/iter-NNN-<tag>/` (history-preserving) and updates a `latest` symlink. Re-running Snakemake against the same `run_root` overwrites in place (Snakemake re-runs only stages whose inputs changed); start a new iteration to keep the old one.
- **Rules:** `snapshot_config`, `encode_msa` (aligned FASTA `msa_fasta` → `<run_dir>/inputs/msa.npy` + `inputs/msa_manifest.json`; every MSA-consuming rule depends on this, not on the FASTA), `msa_stats` (MSA-only — no model dependency, the fix for "can't make the MSA figure without inference"), `build_mask_J`/`build_mask_h` (only when `pruning.enabled`), `train`, `sample` (one job per temperature → `synthetic/align_T<temp>.npy`), `mpnn_sweep` (only when `mpnn.enabled`), `render` (`figs/`), `run_manifest`. Target `msa_stats_only` renders just the MSA figure with no training.
- **MSA figure lands at** `<run_dir>/msa_stats.pdf` (run-dir top level, NOT under `figs/`, because `render` deletes+regenerates `figs/` each call).
- **Provenance chain:** `config_snapshot.yaml` (exact validated params) → `manifest.json` (training; input hashes incl. mask paths, options, seed, git) → `figs/inputs/sources.json` (model + synthetic sha256s per figure) + each PDF's `sbm_run_id` keyword → `run_manifest.json` (aggregate). `iteration_note.md` carries the human hypothesis.
- **Determinism:** the `sample` rule passes `--seed (master_seed + temp_index)` to reproduce the old multi-T `t_seed = seed+i` offset. `run_train.py` pins `OMP_NUM_THREADS` before importing the MCMC kernel **only when `omp_num_threads` is set** in the config; the shipped configs leave it `null`, so default runs are not bit-identical (set it to a fixed int for reproducible arrays).
- **Cluster:** every rule declares `threads` + `resources(mem_mb, runtime)`, so a Snakemake Slurm/Midway profile can be added later without touching rules. Not wired yet (runs locally).

## Combine pipeline (two-model energy)

A **combine** run consumes two already-trained models and scores a query set under both, reporting `E_A`, `E_B`, and `E_tot = w_A·E_A + w_B·E_B` (spec: `docs/initiate_two_model_energy.md`). It is a separate entity from the single-model pipeline — its own validated schema (`src/SBM/combine_config.py`, `config/params_combine-*.yaml`) and its own `Snakefile.combine` — because the single-model config is one-model-per-run.

```
python scripts/iter.py run combine-CM-PPIC "baseline" --snakefile Snakefile.combine
# or directly:
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC.yaml --cores 8 all
```

- **Rules:** `snapshot_config` → `resolve_models` (`models.json`: paths, sha256s, L, seed-MSA per model) → `build_query` (`query/query.fasta` + `query/groups.json`) → `score` (`scores.tsv` tidy long-form + `scores_detail.json` + `manifest.json`) → `render_combine` (`figs/two_model_energy.pdf`, one consolidated `E_A` vs `E_B` scatter + marginals) → `run_manifest`.
- **Methods** (`scoring.method`): `map` (**default** for combine; Viterbi/fields-MAP — the single best alignment per model, same procedure for both → comparable `E_A`/`E_B`), `marginal` (importance-sampling free energy, spec §3.1 — the only mode that yields ESS + MC stderr; warns when ESS < threshold), `in_frame` (exact Potts sum, spec §2), and `auto` (in-frame via the original MSA alignment for a sequence's home model, marginal for the other — **breaks A/B comparability, so it warns**), and `dcalign` (couplings-aware alignment by DCAlign, spec §10.9; the upgrade over fields-MAP). `map` is *fields*-MAP (ignores couplings when choosing the alignment); `dcalign` uses the full couplings. Because DCAlign is ~700× slower (median 19 s/seq), the expensive Julia alignment runs **out-of-process on the cluster** (`pipeline/external/`, sbatch array) and is **cached on disk** under `combine/<run>/dcalign/cache/<model>/alignments.tsv`; the `dcalign` score branch is then a thin cache-reader that recomputes the energy in-frame via `potts_energy` (gauge-consistent — the in-frame recompute vs DCAlign's own energy agrees ≤5e-7, a standing manifest canary). So `score_sequence` never shells out to Julia. Phase-1 uses a flat insertion prior (`lambda_spec="flat"`); the informed `deltan_prior` is the deferred phase-2 fix (spec §10.8 Blocker 1). The HMM proposal is a **self-contained numpy profile HMM** (`src/SBM/energy/hmm.py`) with match emissions from `h`, validated against brute force — no HMMER/pyhmmer dependency.
- **Reuse, don't reinvent:** in-frame energy is `SBM.utils.utils.compute_energies` (the canonical batched sum); the gauge is `Zero_Sum_Gauge` (re-applied defensively on load — idempotent). Both models are loaded in the zero-sum gauge so `E_A + E_B` is well-defined; the two keep their native lengths (CM L=96, PPIC L=91) — never trimmed/padded.
- **Efficiency:** `query.cap_per_group` (seeded subsample, drop logged) bounds the marginal cost on large naturals (e.g. ~26k PPIC seqs); per-(sequence,model) seeds derive from the master seed in stable record order, so the marginal run is reproducible.
- **Note on real-data ESS:** the fields-only proposal gives low ESS on the cross-family term for these strongly-coupled models (flagged, not hidden). For natives that is benign (the alignment posterior is sharply peaked, so marginal ≈ MAP ≈ in-frame); a genuinely poor cross-fit also reads as low ESS. `method: dcalign` (now implemented; the alignment runs on Midway, then scoring is local — see `docs/PIPELINE.md`) or annealed IS is the upgrade path.

## Model transfer (Mac ↔ Midway)

Trained models (`results/<fam>/<iter>/`, ~0.5 GB each, 4.4 GB total) are **not in
git** — too large for git/Git-LFS. They move between the Mac and Midway via
`scripts/sync_models.sh`, a checksummed `rsync` wrapper. The **same** wrapper also
moves the `combine/` DCAlign cache (Midway produces it; you pull it to the Mac to
score) — it syncs **two trees**, `results/` and `combine/`. Full doc:
`docs/MODEL_SYNC.md`; the workflow it serves is `docs/PIPELINE.md`.

- `push` (Mac→Midway) / `pull` (Midway→Mac) / `status` (diff both sides, no
  transfer) / `verify [--remote]` / `hash`. All iterate over both trees; a tree
  absent on one side is skipped (override the list with `SBM_SYNC_ROOTS`).
- **Durable-only by default.** `results/`: `model.npy`, `inputs/`,
  `synthetic/*.npy` + JSON, `masks/`, `mpnn_scores.json`, provenance JSON;
  excludes `figs/` + `mpnn_tmp/`. `combine/`: `cache/<model>/alignments.tsv` +
  `meta.json`, `query/`, config, scores/manifests; **excludes** the ~7–8 GB/model
  `work/` scratch, raw `shards/`, `logs/`, and `*.tar.zst`. `--with-figs` mirrors
  everything.
- **Two-layer integrity:** rsync's own per-file check, **plus** an independent
  per-tree manifest (`results/SHA256SUMS`, `combine/SHA256SUMS`) verified on the
  destination after transfer (a mismatched/missing file prints `FAILED` and exits
  non-zero — never silent). The rsync excludes and manifest prunes are kept in
  lock-step per tree so verify never flags a deliberately-skipped file.
- Config: `SBM_MIDWAY_HOST` (default `midway3.rcc.uchicago.edu`), `SBM_MIDWAY_REPO`
  (default `/project/ranganathanr/nadavbg/Potts_Models`), optional gitignored
  `scripts/sync_models.local.sh`. Models land at the relative `run_dir` paths the
  combine configs reference, so `resolve_models` finds them with no config change.
- Additive by default (never deletes); `--mirror` is opt-in.

## Architecture notes

### Data conventions

- Amino-acid alphabet: `"-ACDEFGHIKLMNPQRSTVWY"`, with `q = 21` and `0 = gap`. `MSA` arrays are `int` of shape `(N_sequences, L)`.
- Sequences containing any character outside the alphabet are **dropped** by `load_fasta` (mapped to `-1`, then filtered).
- `options['q']` and `options['L']` are derived from the alignment in `Init_options`; do not set them manually.
- Each run writes `results/<fam>/<YYYY-MM-DD>_<label>_<idx>/`. `run_sbm.sh` (inference) populates `model.npy`, `manifest.json`, `command.sh`. `sample_sbm.sh` adds one `synthetic/align_T<T>_seed<seed>.npy` and JSON sidecar per requested temperature (default: both T=0.75 and T=1.0) — synthetic alignments are first-class artifacts (and may eventually be tested experimentally), so they live at the run-dir top level rather than inside any figure folder. `render_sbm.sh` regenerates `figs/` on every call; `figs/inputs/` carries `stats_<align_stem>.npy` (one cached `compute_stats` output per alignment) and `sources.json` (paths + sha256s of `model.npy` and every synthetic-alignment file used — pointers only, not copies). The dir name is built by `provenance.make_run_id(label=..., parent_dir=...)`; `idx` auto-increments by scanning sibling dirs. `model.npy` is a pickled dict with the legacy keys (`J`, `h`, `W_all`, `Seeds`, `Train`, `Test`, `options0`, `options1`, …); the manifest carries the full provenance. By default `render_figures.py` produces every figure whose required data is present in the run: `coupling_evol` always (depends only on `model.npy`); `correlations` (one figure with rows=temperatures × cols=1st/2nd/3rd order) and `pca` (1×(1+N_temps) panels) if at least one synthetic alignment is available (auto-discovered under `<run_dir>/synthetic/` or supplied via `--synthetic-alignment PATH ...`); `energy`, `similarity`, `diversity`, `length` additionally if the run has `Test/Train>0` (each overlays every available temperature in one panel). Pass `--figs NAME [NAME ...]` to render an explicit subset — in that mode, requesting a figure whose data is missing is an error.

### Packed-parameter layout (`Wj` / `Jw`)

`SBM` optimizes a flat vector `W` of length `L*q + L*(L-1)/2 * q*q`, packing `h` (size `L*q`) **after** the unique upper-triangular `J[i<j]` block. The C++ MCMC indexes directly into this layout — see `MonteCarlo_PottsMod.cpp` — so any change to the encoding in `Wj`/`Jw` must be mirrored in both `.cpp` files. `Jw` symmetrizes `J[i,j,a,b] = J[j,i,b,a]` on unpack.

### Gauge

The model is over-parameterized; `Zero_Sum_Gauge` projects `(J,h)` onto the zero-sum gauge and is applied **after** training in the demo before parameters are averaged across replicas. Comparing `J`/`h` across runs without first applying this transform is meaningless.

### Statistics-matching loop

Each gradient step in `GradLogLike`:
1. Builds an artificial alignment with `Create_modAlign` (calls `mc.MC` or `mcp.MC`, performing `delta_t` Metropolis sweeps per chain via OpenMP). Inference temperature is hardcoded T=1 (the default value of `Create_modAlign`'s `temperature` argument); training never overrides it.
2. Computes `fi_mod, fij_mod` from the artificial alignment (uniform weights) and `fi, fij` from data (sequence-reweighted, optionally with pseudocount).
3. Returns `gradJ = fij_mod - fij + reg(J)` and `gradh = fi_mod - fi + reg(h)`. Pruning multiplies `gradJ` by the mask each step; `Zero Fields` / `Zero Couplings` zero the corresponding gradient.

`'Pruning Mask Couplings'` may be either a path string (loaded once) or an in-memory `int` array; `Init_Pruning` overwrites the option to the materialized mask. The original input path is preserved under `'Pruning Mask Couplings Source'` for the manifest.

### Optimizer dispatch

`SBM_proteins.Minimizer` selects the algorithm based on `options["Optimizer"]`:

- `"LBFGS"` (default) — runs `AdvanceSearch` / `UpdateHessian`. Both `Model="BM"` and `Model="SBM"` use this path; the difference between the two is the parameter values supplied (`m`, `lambda_J`, `lambda_h`, `N_chains`), not the algorithm.
- `"GD"` — opt-in vanilla gradient descent using `alpha` (default 0.2) for a decaying learning rate, or `Learning_rate` if set. Rarely useful; kept for completeness.

`run_sbm.sh` applies BM-vs-SBM defaults in shell. If you call `train_sbm.py` directly: `--N_chains` is required; `--m`, `--lambdJ`, `--lambdh` default to the SBM regime (1, 0, 0) and must be set explicitly for BM-mode runs.

### Run-level provenance

Every training run and every pruning-mask invocation writes a `manifest.json` sidecar with: git commit + dirty flag + branch, full command line, input file paths and sha256s, the entire options dict (ndarrays summarised as `{shape, dtype, sha256}`), the master seed, OMP thread count, package versions, host/platform, and start/finish timestamps. Schema version 1, defined in `src/SBM/provenance.py`. The training driver also writes `command.sh` reproducing the invocation.

The figure-side equivalent is `lab_plotting.save_figure()` (in `scripts/lab_plotting.py`), which embeds the same git/timestamp data (plus the script path and a `sbm_run_id` keyword) into the PDF metadata. It writes no copy of the source script. Don't bypass it: figures saved with bare `fig.savefig()` lose the provenance metadata.

### RNG seeding

- `--seed S` (a CLI flag on `train_sbm.py`) seeds the Python global RNG via `np.random.seed(S)` AND the C++ MCMC kernels (per-thread seed = `S + thread_id`).
- Per-replicate seeds are spawned with `np.random.SeedSequence(S).spawn(N_av)`.
- The C++ ABI is `MC(w, states, tburn, Q, seed)` — the seed is mandatory.
- Reproducibility under fixed `--seed` requires fixing `OMP_NUM_THREADS` and `N_chains` too. The manifest records both.

### ProteinMPNN foldability sweep

`bash scripts/sample_sbm.sh <run_dir> --mpnn-sweep` is an alternate sampling mode that produces a temperature ladder (default 0.1..1.0 step 0.1, 100 seqs/T) plus interpretability controls (WT, uniform random, shuffled WT, natural-MSA bootstrap) and scores them against `data/structures/1ECM.pdb` via the upstream `dauparas/ProteinMPNN` repo. Outputs land in `<run_dir>/synthetic/mpnn_sweep_seed<seed>/` (a subdir, so existing figure auto-discovery is unaffected). The figure name is `mpnn`; `render_figures.py` auto-detects the sweep dir. ProteinMPNN is not pip-installable: the user clones it next to this repo and sets `PROTEINMPNN_PATH` (or passes `--mpnn-path`); scoring is delegated via subprocess so this codebase doesn't import torch. See `docs/MPNN_FOLDABILITY.md` for setup, score interpretation, and benchmark table.

## Gotchas

- **`load_fasta` silently drops sequences** containing non-canonical residues. The reported "Final shape" may be smaller than the FASTA record count.
- **`Zero Couplings=True` switches MCMC kernels** from `MonteCarlo_Potts` to `MonteCarlo_PottsProf` and packs `W` as just `h.flatten()`. Output-handling code branches on this, e.g. `output['J_norm']` becomes `None`.
- **`'Pruning Mask Couplings'` is mutated in place** by `Init_Pruning` (path → array). The original path is stashed in `'Pruning Mask Couplings Source'` so the manifest can record it; don't reuse the options dict across `SBM(...)` calls expecting a clean re-init.
- **`model.npy` bytes are not deterministic across runs** with the same seed — the saved dict includes wall-clock `Execution times`. The arrays inside (`J`, `h`, `W_all`, `Seeds`) **are** bit-identical with the same seed + `OMP_NUM_THREADS`. Compare arrays, not pickles.
- **Conda Python segfaults** during `import_array()`. Use uv-managed CPython or Homebrew's `python@3.12`.
- **Editable install + C++ edits.** `pip install -e .` does not auto-rebuild the C++ extensions on `.cpp` changes. After editing kernel source, run `pip install -e . --force-reinstall --no-deps`.

## Project-specific code conventions to follow

- Treat `Wj`/`Jw` as the canonical (de)serialization between the optimizer and the model; do not invent a parallel encoding.
- After training, parameters meant for downstream comparison must pass through `Zero_Sum_Gauge` first.
- New C++ kernels: register them in `CMakeLists.txt` (each module gets its own `add_library(... MODULE)` + install target), keep the `set_target_properties(... PROPERTIES PREFIX "")` line so Python finds the `.so` by module name, mirror the `Wj` packing exactly, and accept a `seed` argument (per-thread seed = `seed + thread_id`).
- New `options` keys: add a default in `ParseOptions` so existing callers don't need updates, and document any in-place mutation. If the option carries a path that's loaded into memory, store the path string under a sibling `<key> Source` key so the manifest can record it.
- Anything that produces a model, mask, or other artifact lands in a per-run directory with a `manifest.json` sidecar. Use `SBM.provenance.build_run_manifest` + `save_run_manifest` rather than rolling a new format.

## Figures

This project uses the lab figure style. When writing or modifying any
plotting code:

- Activate the appropriate stylesheet before any plotting code:
  `plt.style.use("lab-paper")` for figures destined for papers/posters,
  `plt.style.use("lab-slides")` for talks. Pick by destination, not by
  guess.
- Save figures only via `lab_plotting.save_figure()`, never bare
  `fig.savefig()`. The wrapper embeds git provenance into the PDF
  metadata (it does not write a copy of the source script).
- For semantic colors (reference, negative control, fit, highlight),
  use the constants in `lab_plotting.LAB_COLORS`, not hex literals.
- Panel labels go through `lab_plotting.panel_label()`. Do not place
  bare `ax.text(...)` in the upper-left corner of axes.
- Every axis label must include units in parentheses where applicable
  (e.g. "ΔG_binding (kcal/mol)", not "delta G").
- Every plotted point with measured uncertainty needs an error bar.
  If error bars are not yet computed, leave a `# TODO: add error bars`
  comment rather than plotting bare points and forgetting.
- No spline-interpolated fit curves through data points. Fits are
  straight lines or named mathematical functions, plotted across the
  data range.
