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
| Plot recipes (used by `render_figures.py`) | `src/SBM/utils/utils_plot.py:plot_stats` (7 modes; `correlations` is rows=temperatures × cols=1st/2nd/3rd order, `pca` is 1×(1+N_temps)). |
| Run-level provenance helpers | `src/SBM/provenance.py` |
| The pruning CLI | `pruning/build_mask.py` (auto-names a per-run subdir; `--out-file PATH` writes a single mask to an exact path for the pipeline) |
| The figure-save helpers | `scripts/lab_plotting.py` (`save_figure`, `panel_label`, `LAB_COLORS`) |
| The CM worked example (pipeline) | `config/params_CM-bm-dense.yaml` (plain BM, no pruning; the `params_CM-bm-*` variants add pruning); the legacy `pruning/CM_example.sh` (chains `run_sbm.sh` → `sample_sbm.sh` → `render_sbm.sh`) still works |
| Energy of a sequence under one/two models | `src/SBM/energy/` — `model.load_model` (loads + re-gauges), `potts.potts_energy` (in-frame, wraps `compute_energies`), `hmm.ProfileHMM` (alignment proposal: forward / Viterbi / FFBS), `score.score_sequence` / `score.score_two_models` (`method ∈ {in_frame, map, marginal, potts_align}`). Progress + next steps: `docs/two_model_progress.md` |
| The two-model scoring CLI | `scripts/score_two_models.py` (single sequence or batch FASTA → `E_A`, `E_B`, `E_tot` + diagnostics; `--method`, `--weights`, `--n-samples`, `--seed`) |
| Couplings-aware aligner (the production aligner; gap-placement Potts min) | `src/SBM/energy/potts_align.py` — `enumerate_align` (exact global Potts-frame argmin for `N≤L` queries; few-gap), `pt_align` (parallel tempering w/ teleport move, the g-adaptive default high-gap engine), `sa_align`, `potts_align` (dispatch: enum if `C(L,N)≤budget` else g-adaptive PT), `PTSchedule.for_gap_count`, `perturb_frame`. Solved the combine worse-than-native residual with **no DCAlign** (it was a BP search failure). Full spec + cost model + DCAlign comparison + combine/cluster runbook: `docs/POTTS_ALIGN.md`. Cache I/O `src/SBM/utils/potts_align_cache.py`; ground-state baseline `src/SBM/energy/potts_align_baseline.py`; tests `tests/test_potts_align.py`, `tests/test_potts_align_baseline.py`. Handles cross-family when `N≤L`; **not** `N>L` (insertions). |
| The two-model `combine` pipeline | `Snakefile.combine` driven by `config/params_combine-*.yaml` (validated by `src/SBM/combine_config.py`); run with `python scripts/iter.py run <name> "<tag>" --snakefile Snakefile.combine`. See "Combine pipeline" below. |
| The post-hoc `derive` pipeline (keep only *some* params of a trained model) | `Snakefile.derive` driven by `config/params_derive-*.yaml` (validated by `src/SBM/derive_config.py`); filters an already-trained `model.npy` (fields only / couplings only / mask subset), re-gauges, lands a **normal `results/` dir** the combine pipeline consumes by `run_dir`. Core `src/SBM/derive.py` (`apply_filter`, `build_derived_dict`); reuses `pruning/build_mask.py` for the mask subset; wrappers `scripts/wf/run_derive*.py` + `run_copy_inputs.py`; tests `tests/test_derive.py`. Run with `python scripts/iter.py run <name> "<tag>" --snakefile Snakefile.derive`. See "Derive pipeline" below. |
| The potts_align combine wiring (score every (query,model) with `N≤L`) | `scoring.method: potts_align` in `config/params_combine-CM-PPIC-potts.yaml`; the `score` rule reads a cluster-built cache (or recomputes live), the `potts_align_baseline` rule reports ΔE-vs-native per home pair. Cluster wrappers `scripts/wf/run_potts_align_{shard,gather}.py` + `pipeline/external/{run_potts_align_align,sbatch_potts_align_shard,sbatch_potts_align_gather,finalize_potts_align}.sh` (pure Python, no Julia). Cost-bounding knobs (`pa_cross_subsample_*`, `query.n_random` control) + the full runbook: `docs/POTTS_ALIGN.md` §11. |
| The two-model design engine (joint annealing over `E_tot`) | `src/SBM/design/anneal.py` — `anneal_chain` (SA over (core sequence, gap placement in each frame); alignment folded into the MC so per-step cost is O(L), gap-count-independent), `AnnealSchedule`, `_sub_delta` (covers sub/insert/delete), `initial_state_from_frame` (natural-seeded starts), `polish` (warm-started `potts_align` for the authoritative argmin `E_A`/`E_B`). **Wired into the combine pipeline** as a gated `design:` stage of `Snakefile.combine` (schema `DesignConfig` in `src/SBM/combine_config.py`) — see "Combine pipeline" below; CLI `scripts/design_two_model.py` still runs it directly. Chains are seeded from a **start mix** (`start_random`/`start_natural_a`/`start_natural_b`; naturals from each model's `seed_msa`, core ≤ min(L)=91); figures color trajectories **by start type** (`src/SBM/utils/utils_design_plot.py` + `scripts/render_design.py`: trajectories, `E_A`/`E_B` phase space w/ Pareto front + landing heatmap, a final-length histogram, and a **ZAPPO-colored alignment** of the designs in both model frames), landing in `<run_root>/figs/`. Each design's in-frame **polish alignment** is also exported as `design/design_aln_{A,B}.fasta` (L=96 / L=91 gapped MSAs, uploadable to an alignment viewer). `design.execution: cluster\|auto\|local` picks where the anneal runs (**default `cluster`** = Midway, per the Mac-figures/Midway-compute split; `auto` = local iff predicted wall-time ≤ `local_budget_minutes`; `local` = Mac, ~2 min at default size); polish defaults to `fast` (warm-started, ~5.5 s/chain — the polish, not the anneal, is the cost). Snakemake wrappers `scripts/wf/run_design_{config,local,render,handoff}.py`; cluster wrappers `scripts/wf/run_design_{shard,gather}.py`; tests `tests/test_design_two_model.py`. The anneal is Mac-cheap (~min) but **defaults to Midway (`cluster`)** per the compute split; the cluster sbatch scripts (`pipeline/external/*design*`) are written. Full spec + Mac→Midway runbook: `docs/DESIGN_TWO_MODEL.md`; end-to-end ordering: `docs/RUNBOOK.md`. |
| The design characterization (fold + BLAST) | **Midway compute** (ESMFold GPU + TM-align + BLAST + merge): `scripts/characterize/characterize.py` driver + `src/SBM/characterize/` (`summary.py` merge; `fold`/`tmscore`/`blast` parsers) + `pipeline/external/*characterize*`/`*esmfold*` sbatch. **Mac renders** the figures + stats from the merged `characterize/data/summary.tsv` (pure numpy) via `src/SBM/utils/utils_characterize_plot.py` + `scripts/characterize/render_characterize.py`, wired as the gated `characterize_render` stage of `Snakefile.combine` (schema `CharacterizeConfig`). Figures land in `<run_root>/figs/`: `characterization_overview.pdf` (2×2), `tm_A_vs_B.pdf`, `fold_call_breakdown.pdf` + `characterize/data/characterization_stats.tsv`. Tests `tests/test_characterize.py` (parsers) + `tests/test_characterize_plot.py` (render). See "Characterize pipeline" below + `docs/CHARACTERIZE.md`. |
| The end-to-end runbook (two `results/` model dirs → designed, characterized seqs + figures) | **Start here: `scripts/new_combine.py <dirA> <dirB> --tag "<tag>"`** — one command that generates a validated `config/params_combine-*.yaml` (auto-picks model names, the `pa_cross_subsample_origin`=larger-N and `random_length`=min(L) knobs, from the two `results/` dirs), mints `combine/<run>/iter-NNN-<tag>/`, and writes that dir's **`RUNBOOK.txt`** — the per-run, copy-pasteable, path-interpolated step list (only the enabled stages; finalizers included). The runbook text is rendered by `src/SBM/combine_runbook.py` (`render_runbook` / `design_handoff_text`), written at scaffold time and refreshed by the `runbook` rule of `Snakefile.combine` on each `all` run (re-run `all` after editing a config to update it) and reused by the design hand-off. `docs/RUNBOOK.md` is the thin concept page + the workflow/DAG figures; `scripts/render_dag.sh` (needs `brew install graphviz`) + `scripts/render_workflow_diagram.py` regenerate `docs/workflow/combine_{workflow,rulegraph,dag}.*`. |
| Model transfer Mac ↔ Midway (models **and** the potts_align cache) | `scripts/sync_models.sh` (checksummed rsync; `push`/`pull`/`verify`/`status`; covers three trees — `results/`, `combine/`, and the content-addressed `natural_folds/` fold cache, whose `structures/` PDBs are excluded). Not in git. See "Model transfer" below and `docs/MODEL_SYNC.md`. |
| Retired DCAlign campaign (code + artifacts) | `.archive/` (gitignored, excluded from `sync_models.sh`) — all `dcalign_*` modules, the Julia drivers, the DCAlign cluster scripts, the iter-003 experiment scripts/run dirs, and `docs/{PIPELINE,ITER003_RUNBOOK}.md`. Verdict + why: `docs/two_model_progress.md`. |

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

**Where compute runs (Mac-primary).** Run everything on the Mac: training, sampling, figures, MSA stats, and combine scoring (all methods, including `potts_align`, are pure numpy). The Midway cluster is **optional** — used only to pre-build the `potts_align` alignment cache for a *large* combine query set (embarrassingly parallel over (query, model) pairs, sharded on a Slurm array, pure Python — no Julia). That align step writes a cache; you `sync_models.sh pull` it to the Mac and score locally. A small query set needs no cluster step. Runbook: `docs/POTTS_ALIGN.md` §11. (DCAlign, which previously drove the cluster step, is retired — `docs/two_model_progress.md`.)

**macOS toolchain.** AppleClang has no OpenMP, so `pyproject.toml` forces `cmake/macos_llvm.cmake`, which hard-codes `/opt/homebrew/opt/llvm` and `libomp`. `brew install llvm libomp ninja cmake` is required; Intel-Mac or non-Homebrew prefixes need the toolchain file edited. On Linux, `python3-dev`, GCC/G++ with OpenMP, CMake, and Ninja are sufficient.

**Don't use conda-forge / miniforge Python.** Its `python@3.12` libpython has an ABI quirk that segfaults during numpy's `import_array()` when our C++ extension is loaded. Use uv-managed standalone CPython or Homebrew's `python@3.12`.

**C++ rebuilds.** scikit-build-core's editable install does not auto-rebuild on `.cpp` changes. After editing kernel source, run `uv pip install -e . --force-reinstall --no-deps`.

**Python.** 3.11+ (`requires-python = ">=3.11"`).

**Tests.** The energy/scoring module (`src/SBM/energy/`) has a pytest suite at `tests/test_energy.py` — run it after touching that module:

```
.venv/bin/python -m pytest tests/test_energy.py -q
```

It is pure numpy/scipy (no MCMC) and covers spec §6 (gauge invariance, in-frame base case, MAP≈marginal when unambiguous, ordering sanity, IS diagnostics) plus the **DP anchor**: the profile-HMM forward log-Z, FFBS sample frequencies, marginal-IS estimate, and Viterbi path are all checked against *brute-force enumeration of every alignment* of a tiny query — this is the gold-standard check for `hmm.py`, so no `pyhmmer`/HMMER dependency is needed.

The rest of the codebase has no unit-test suite. After non-trivial changes, run the pipeline smoke test (tiny config: 5 L-BFGS iters, 4 chains, N=50):

```
snakemake --configfile config/params_tiny.yaml --cores 8 all
# combine pipeline smoke test (scores a handful of CM+PPIC seqs under both models):
snakemake -s Snakefile.combine --configfile config/params_combine-tiny.yaml --cores 8 all
```

The first exercises the whole DAG — encode_msa → mask → train → sample(×2) → render → manifest, plus the independent `msa_stats` branch — and lands deterministic outputs under `results/tiny/` (assert on `inputs/msa.npy`, `model.npy`, `manifest.json`, `synthetic/align_T*.npy`, `figs/*.pdf`, `msa_stats.pdf`, `run_manifest.json`). The legacy `bash pruning/CM_example.sh` still works but is superseded by the pipeline.

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
- **Rules:** `snapshot_config`, `encode_msa` (aligned FASTA `msa_fasta` → `<run_dir>/inputs/msa.npy` + `inputs/msa_manifest.json`; every MSA-consuming rule depends on this, not on the FASTA), `msa_stats` (MSA-only — no model dependency, the fix for "can't make the MSA figure without inference"), `build_mask_J`/`build_mask_h` (only when `pruning.enabled`), `train`, `sample` (one job per temperature → `synthetic/align_T<temp>.npy`), `render` (`figs/`), `run_manifest`. Target `msa_stats_only` renders just the MSA figure with no training.
- **MSA figure lands at** `<run_dir>/msa_stats.pdf` (run-dir top level, NOT under `figs/`, because `render` deletes+regenerates `figs/` each call).
- **Provenance chain:** `config_snapshot.yaml` (exact validated params) → `manifest.json` (training; input hashes incl. mask paths, options, seed, git) → `figs/inputs/sources.json` (model + synthetic sha256s per figure) + each PDF's `sbm_run_id` keyword → `run_manifest.json` (aggregate). `iteration_note.md` carries the human hypothesis.
- **Determinism:** the `sample` rule passes `--seed (master_seed + temp_index)` to reproduce the old multi-T `t_seed = seed+i` offset. `run_train.py` pins `OMP_NUM_THREADS` before importing the MCMC kernel **only when `omp_num_threads` is set** in the config; the shipped configs leave it `null`, so default runs are not bit-identical (set it to a fixed int for reproducible arrays).
- **Cluster:** every rule declares `threads` + `resources(mem_mb, runtime)`, so a Snakemake Slurm/Midway profile can be added later without touching rules. Not wired yet (runs locally).

## Combine pipeline (two-model energy)

A **combine** run consumes two already-trained models and scores a query set under both, reporting `E_A`, `E_B`, and `E_tot = w_A·E_A + w_B·E_B`. The combining weights are **not** configured — the `compute_weights` stage derives them *post-hoc from the naturals* (`w_A = m_B/(m_A+m_B)`, `w_B = m_A/(m_A+m_B)` where `m_X` = median native energy of family X under its home model, normalized `w_A+w_B=1`) so each family's median native energy contributes equally and annealing on `E_tot` isn't biased toward one family; see `src/SBM/utils/energy_weights.py` (progress + next steps: `docs/two_model_progress.md`; the production aligner is `docs/POTTS_ALIGN.md`). It is a separate entity from the single-model pipeline — its own validated schema (`src/SBM/combine_config.py`, `config/params_combine-*.yaml`) and its own `Snakefile.combine` — because the single-model config is one-model-per-run.

```
# Recommended: scaffold config + run dir + RUNBOOK.txt from two results/ dirs in one command
python scripts/new_combine.py results/CM-bm-dense/iter-002-base-model results/PPIC-dense/iter-001-baseline --tag "baseline"
# then follow the printed combine/<run>/iter-NNN-baseline/RUNBOOK.txt

# Or, with a hand-written config:
python scripts/iter.py run combine-CM-PPIC "baseline" --snakefile Snakefile.combine
# or directly:
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC.yaml --cores 8 all
```

- **Rules:** `snapshot_config` (+ a dependency-free `runbook` → `RUNBOOK.txt`, the per-run copy-paste steps, refreshed from the config on each `all` run) → `resolve_models` (`models.json`: paths, sha256s, L, seed-MSA per model) → `build_query` (`query/query.fasta` + `query/groups.json`) → `score` (`data/scores.tsv` tidy long-form + `data/scores_detail.json` + `provenance/score_manifest.json`) → `render_combine` (`figs/two_model_energy.pdf`, one consolidated `E_A` vs `E_B` scatter + marginals) → `compute_weights` (method-agnostic: derives the post-hoc `E_tot` weights from the naturals → `data/energy_weights.json` + `data/energy_weight_sweep.tsv` + `figs/energy_weights.pdf`, the weighted-median-energy-vs-`w_A` diagnostic) → `run_manifest` (`provenance/run_manifest.json`). When `scoring.method == "potts_align"`, one extra rule runs: `potts_align_baseline` (`data/potts_align_vs_inframe.{tsv,json}` + `figs/potts_align_vs_inframe.pdf`, potts_align's min energy vs the native in-frame energy per home pair — ΔE≤0 expected, reads out how often the native frame already IS the ground state), and the `score` rule declares the cluster-built `potts_align/cache/<model>/alignments.tsv` as an input (the two-phase align→score dependency). When `design.enabled`, a **design stage** runs after `compute_weights`: `design_config` (`design/design_config.json` — resolves models/weights + the seeded natural-start rows) → then, per `design.execution`, either `design_local` (`design/trajectories.npz` + `designed.tsv`/`.fasta` + `design_manifest.json`) + `design_render` (`figs/design_{trajectories,phase_space}.pdf`) **or** `design_handoff` (`design/MIDWAY_HANDOFF.txt`, then run on Midway + pull back). `execution: auto` decides by a predicted local wall-time vs `local_budget_minutes`; the estimate prints at DAG build. See `docs/DESIGN_TWO_MODEL.md`.
- **Run-dir layout:** Mac-side outputs are grouped into `data/` (tables) and `provenance/` (manifests); `figs/` holds figures. The top level keeps only the **Mac↔Midway contract** files the cluster `potts_align` scripts read by exact path — `config_snapshot.yaml`, `models.json`, `query/` — plus `iteration_note.md`, `potts_align/`, `logs/`. `sync_models.sh` selects by directory-name prune, so `data/`/`provenance/` sync automatically.
- **Methods** (`scoring.method`): `potts_align` (**the production couplings-aware aligner**: minimizes the exact in-frame Potts energy over gap placements, provably global for few gaps else parallel tempering; pure numpy; requires `N≤L`; `docs/POTTS_ALIGN.md`), `map` (**schema default**; Viterbi/fields-MAP — single best alignment per model, same procedure for both → comparable `E_A`/`E_B`, but ignores couplings), `marginal` (importance-sampling free energy — the only mode that yields ESS + MC stderr; warns when ESS < threshold), `in_frame` (exact Potts sum), and `auto` (in-frame via the original MSA alignment for a sequence's home model, marginal for the other — **breaks A/B comparability, so it warns**). For a large query set the `potts_align` alignment is pre-built out-of-process on a Slurm array and **cached on disk** under `combine/<run>/potts_align/cache/<model>/alignments.tsv`; the score branch reads the cache and recomputes the energy in-frame via `potts_energy` (gauge-consistent — the recompute vs the cached energy agrees ≤1e-6, a standing manifest canary). The HMM proposal (for `map`/`marginal`) is a **self-contained numpy profile HMM** (`src/SBM/energy/hmm.py`) with match emissions from `h`, validated against brute force — no HMMER/pyhmmer dependency.
- **Reuse, don't reinvent:** in-frame energy is `SBM.utils.utils.compute_energies` (the canonical batched sum); the gauge is `Zero_Sum_Gauge` (re-applied defensively on load — idempotent). Both models are loaded in the zero-sum gauge so `E_A + E_B` is well-defined; the two keep their native lengths (CM L=96, PPIC L=91) — never trimmed/padded.
- **Efficiency:** `query.cap_per_group` (seeded subsample, drop logged) bounds the cost on large naturals (e.g. ~26k PPIC seqs); `scoring.pa_cross_subsample_*` further bounds the expensive `potts_align` cross block; per-(sequence,model) seeds derive from the master seed in stable record order, so the run is reproducible.
- **Note on real-data ESS (`marginal` only):** the fields-only proposal gives low ESS on the cross-family term for these strongly-coupled models (flagged, not hidden). For natives that is benign (the alignment posterior is sharply peaked, so marginal ≈ MAP ≈ in-frame); a genuinely poor cross-fit also reads as low ESS. `method: potts_align` (the couplings-aware minimizer) or annealed IS is the upgrade path.

## Derive pipeline (post-hoc parameter filtering)

A **derive** run takes ONE already-trained model and writes a new model that keeps only a *subset* of its parameters — e.g. a dense model's fields with all couplings zeroed. This is how you combine models where each contributes only some of its parameters, **without retraining**: the filter is applied to the fitted arrays after the fact. It is a separate entity from the single-model and combine pipelines — its own validated schema (`src/SBM/derive_config.py`, `config/params_derive-*.yaml`) and its own `Snakefile.derive` — but it deliberately lands a **normal `results/` run dir** so the combine pipeline consumes the derived model by `run_dir` with no combine-side change.

```
python scripts/iter.py run derive-CM-profile "fields-only" --snakefile Snakefile.derive
# or directly:
snakemake -s Snakefile.derive --configfile config/params_derive-CM-profile.yaml --cores 8 all
```

- **Post-hoc ≠ retrained profile (important).** The retrain-based `config/params_CM-bm-profile.yaml` re-fits `h` to the single-site statistics with `J≡0`. The derive pipeline instead keeps the dense model's *already-fit* `h` and zeros/masks `J` — a **different energy function**, a clean ablation of the dense model's field component. Both are valid; pick deliberately. The `*-profile.yaml` retrain configs are unchanged.
- **The filter (`filter:` block).** `couplings` and `fields` are each `keep` / `zero` / a `{strategy, percent}` **MaskSpec** (reusing the pruning masks — 1=keep, `percent` = fraction removed; `theta`/`lbda`/`label`/`Dia_prior` mirror `pruning:` and are read only for a MaskSpec). Fields only = `{couplings: zero, fields: keep}`; couplings only = `{couplings: keep, fields: zero}`.
- **Rules:** `snapshot_config` → `copy_inputs` (copies the source's `inputs/msa.npy` [+ `msa_manifest.json`] so combine finds naturals + seed MSA; no re-encoding) → `build_mask_J`/`build_mask_h` (only when that block is a MaskSpec) → `derive` (`model.npy` + `manifest.json` recording the **source model sha256 + the filter** + `command.sh`) → `sample` (synthetics sampled *from the derived model*; a `J≡0` model samples independent-site sequences via the standard kernel) → `render` (`figs/`) → `run_manifest`.
- **Reuse, don't reinvent:** the mask is `pruning/build_mask.py` (same `(L,L,q,q)`/`(L,q)` 0/1 format), the filter is `J*=mask` / `h*=mask` applied to the fitted arrays, and the gauge is `Zero_Sum_Gauge` (with `J≡0` its correction to `h` vanishes, so fields-only preserves `h`; on an already-gauged source, couplings-only leaves `h` at zero). `J` is always written as a full zeros array (never `None`) so `load_model`'s shape checks pass. The derived dict drops the training-replicate `W_all`/`Seeds` (meaningless post-filter, read nowhere) and stores a single-point `J_norm` = the derived mean Frobenius coupling norm.
- **Smoke test:** `snakemake -s Snakefile.derive --configfile config/params_derive-tiny.yaml --cores 8 all` (fields-only from `results/tiny/`).

## Characterize pipeline (structural + BLAST QC of designs)

Downstream QC for the two-model **design** outputs: predict a structure for each
designed sequence (ESMFold), ask **which of the two reference folds it resembles**
(TM-align vs 1ECM = fold A / CM, 1JNT = fold B / PPIC), and **what it looks like in
sequence space** (BLAST vs SwissProt + the CM/PPIC families). Naturals from each seed
MSA are folded once as controls (the positive control: a family's naturals must match
their own reference). It **replaced** the retired ProteinMPNN foldability proxy. Full
spec: `docs/CHARACTERIZE.md`; end-to-end ordering: `docs/RUNBOOK.md`.

- **Compute is Midway-only; figures are Mac-only** (the deliberate split — see `docs/RUNBOOK.md`).
  Midway runs `scripts/characterize/characterize.py` (driver) which folds (ESMFold GPU,
  `src/SBM/characterize/fold.py`), TM-aligns (`TMalign`, `tmscore.py`), BLASTs
  (`blastp`, `blast.py`), and **merges** into the tidy tables (`summary.py` — pure
  python) under `<run_dir>/characterize/data/`. Orchestration: `pipeline/external/{run_characterize,run_esmfold_probe,sbatch_esmfold_shard,sbatch_characterize_cpu,build_tmalign,prefetch_esmfold}.sh`.
- **Mac render (authoritative figures).** `src/SBM/utils/utils_characterize_plot.py`
  (recipes; inch-budget layout, `lab_plotting`) + `scripts/characterize/render_characterize.py`
  (thin CLI, backward-compatible with the Midway driver) consume `summary.tsv` (+ optional
  `natural_summary.tsv`) — **pure numpy/matplotlib, no binaries**. Wired as the gated
  `characterize_render` stage of `Snakefile.combine` (`CharacterizeConfig.enabled` in the
  combine config). Because the input is Midway-produced and pulled via `sync_models.sh`,
  the figs are folded into `rule all` only once `summary.tsv` is on disk (checked at DAG
  build; a skip note prints otherwise — `all` is never blocked on un-pulled data);
  targeting a fig explicitly fails loudly if the table is missing. Outputs:
  `figs/characterization_overview.pdf` (2×2: fold / pLDDT / energy-vs-structure / BLAST),
  `figs/tm_A_vs_B.pdf`, `figs/fold_call_breakdown.pdf`, and the tidy
  `characterize/data/characterization_stats.tsv` (the numbers the figures cite —
  medians, fold-call counts, control-sanity PASS/FAIL, `Spearman(ΔE, ΔTM)`).
- **Reuse, don't reinvent:** `summary.read_tsv` / `summary.fold_call` / the column
  constants for I/O; `lab_plotting` + the `utils_design_plot` inch-budget helpers for
  layout. `fold_call ∈ {A, B, ambiguous, neither, na}` from the TM≥0.5 rule.
- **Sync:** the naturals fold cache is its own top-level content-addressed tree
  `natural_folds/<msa_sha8>/` (keyed by source-FASTA sha8 — a property of the MSA,
  not of any run), synced by `sync_models.sh` with its ~28k per-sequence ESMFold
  PDBs (`natural_folds/*/structures/`) **excluded** (only the distilled
  `fold_scores/*.tsv` + `tm_vs_refs/*.tsv` travel; the PDB cache stays Midway-side).
  This is the fix for the slow rsync.
- **Test:** `.venv/bin/python -m pytest tests/test_characterize.py tests/test_characterize_plot.py -q`
  (both pure-python, no binaries needed).

## Model transfer (Mac ↔ Midway)

Trained models (`results/<fam>/<iter>/`, ~0.5 GB each, 4.4 GB total) are **not in
git** — too large for git/Git-LFS. They move between the Mac and Midway via
`scripts/sync_models.sh`, a checksummed `rsync` wrapper. The **same** wrapper also
moves the `combine/` potts_align cache (a cluster align run produces it; you pull it
to the Mac to score) — it syncs **two trees**, `results/` and `combine/`. Full doc:
`docs/MODEL_SYNC.md`; the workflow it serves is `docs/POTTS_ALIGN.md` §11.

- `push` (Mac→Midway) / `pull` (Midway→Mac) / `status` (diff both sides, no
  transfer) / `verify [--remote]` / `hash`. All iterate over both trees; a tree
  absent on one side is skipped (override the list with `SBM_SYNC_ROOTS`).
- **Durable-only by default.** `results/`: `model.npy`, `inputs/`,
  `synthetic/*.npy` + JSON, `masks/`, provenance JSON;
  excludes `figs/`. `combine/`: `potts_align/cache/<model>/alignments.tsv`
  + `meta.json`, `query/`, config, scores/manifests; **excludes** the per-shard
  `shards/`/`logs/`/`work/` scratch + `*.tar.zst`, plus `.archive/` and any retired
  `*dcalign*` run (never synced). `--with-figs` mirrors everything.
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
