# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`SBM` (Stochastic Boltzmann Machine) infers the fields `h` and couplings `J` of a Potts model on a multiple-sequence alignment (MSA) by gradient descent against MCMC-estimated statistics. The Python package wraps two C++/OpenMP MCMC kernels that are compiled at install time via scikit-build-core + CMake.

## Where things live

| If you need… | Look at |
|---|---|
| The optimizer entry point | `src/SBM/SBM_GD/SBM_proteins.py:SBM(align, options)` |
| The MCMC sampler driver (Python) | `src/SBM/utils/utils.py:Create_modAlign` |
| The C++ MCMC kernels | `src/SBM/MonteCarlo/MCMC_Potts/MonteCarlo_PottsMod.cpp` (full) and `MCMC_PottsProf/MonteCarlo_PottsProfMod.cpp` (profile-only) |
| The packed-vector encoding | `src/SBM/utils/utils.py:Wj` / `Jw` |
| The zero-sum gauge transform | `src/SBM/utils/utils.py:Zero_Sum_Gauge` |
| Statistics / reweighting | `src/SBM/utils/utils.py:CalcWeights`, `CalcStatsWeighted`, `CalcThreeCorrWeighted` |
| Run-level provenance helpers | `src/SBM/provenance.py` |
| The training CLI | `scripts/train_sbm.py` (writes `results/<fam>/<run_id>/`) |
| The pruning CLI | `pruning/build_mask.py` |
| The figure helpers | `scripts/lab_plotting.py` (`save_figure`, `panel_label`, `LAB_COLORS`) |
| The CM worked example | `scripts/examples/cm-family/run.sh`, `pruning/CM_example.sh` |

`src/SBM/__init__.py` is empty by design — users import submodules directly (`SBM.SBM_GD.SBM_proteins`, `SBM.utils.utils`, `SBM.provenance`).

## Data flow

```
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
uv pip install -e ".[plotting,analysis,dev]"
```

`pip install -e .` works equivalently. Runtime deps are pinned in `pyproject.toml`; `requirements.lock` records exact pins for reproducible installs (`uv pip sync requirements.lock` then `uv pip install -e . --no-deps`).

**macOS toolchain.** AppleClang has no OpenMP, so `pyproject.toml` forces `cmake/macos_llvm.cmake`, which hard-codes `/opt/homebrew/opt/llvm` and `libomp`. `brew install llvm libomp ninja cmake` is required; Intel-Mac or non-Homebrew prefixes need the toolchain file edited. On Linux, `python3-dev`, GCC/G++ with OpenMP, CMake, and Ninja are sufficient.

**Don't use conda-forge / miniforge Python.** Its `python@3.12` libpython has an ABI quirk that segfaults during numpy's `import_array()` when our C++ extension is loaded. Use uv-managed standalone CPython or Homebrew's `python@3.12`.

**C++ rebuilds.** scikit-build-core's editable install does not auto-rebuild on `.cpp` changes. After editing kernel source, run `uv pip install -e . --force-reinstall --no-deps`.

**Python.** 3.11+ (`requires-python = ">=3.11"`).

**Tests.** There is no test suite. After non-trivial changes, run the worked example as a smoke test:

```
bash scripts/examples/cm-family/run.sh
```

It expects `data/MSA_array/MSA_CM.npy` to exist and writes a per-run directory under `results/CM/<run_id>/`.

**Pruning workflow** lives in `pruning/` with its own `README.md` and `CM_example.sh`. The `"sca"` strategy depends on `pysca`, gated behind the `[sca]` optional-dependency group; `"fij"` and `"cij"` don't need it.

## Architecture notes

### Data conventions

- Amino-acid alphabet: `"-ACDEFGHIKLMNPQRSTVWY"`, with `q = 21` and `0 = gap`. `MSA` arrays are `int` of shape `(N_sequences, L)`.
- Sequences containing any character outside the alphabet are **dropped** by `load_fasta` (mapped to `-1`, then filtered).
- `options['q']` and `options['L']` are derived from the alignment in `Init_options`; do not set them manually.
- Each training run writes `results/<fam>/<run_id>/{model.npy, manifest.json, command.sh}`. `model.npy` is a pickled dict with the legacy keys (`J`, `h`, `W_all`, `Seeds`, `Train`, `Test`, `options0`, `options1`, …); the manifest carries the full provenance. The `options0`/`options1` split inside the model dict is preserved for backward compat but no longer drives filename generation.

### Packed-parameter layout (`Wj` / `Jw`)

`SBM` optimizes a flat vector `W` of length `L*q + L*(L-1)/2 * q*q`, packing `h` (size `L*q`) **after** the unique upper-triangular `J[i<j]` block. The C++ MCMC indexes directly into this layout — see `MonteCarlo_PottsMod.cpp` — so any change to the encoding in `Wj`/`Jw` must be mirrored in both `.cpp` files. `Jw` symmetrizes `J[i,j,a,b] = J[j,i,b,a]` on unpack.

### Gauge

The model is over-parameterized; `Zero_Sum_Gauge` projects `(J,h)` onto the zero-sum gauge and is applied **after** training in the demo before parameters are averaged across replicas. Comparing `J`/`h` across runs without first applying this transform is meaningless.

### Statistics-matching loop

Each gradient step in `GradLogLike`:
1. Builds an artificial alignment with `Create_modAlign` (calls `mc.MC` or `mcp.MC`, performing `delta_t` Metropolis sweeps per chain via OpenMP).
2. Computes `fi_mod, fij_mod` from the artificial alignment (uniform weights) and `fi, fij` from data (sequence-reweighted, optionally with pseudocount).
3. Returns `gradJ = fij_mod - fij + reg(J)` and `gradh = fi_mod - fi + reg(h)`. Pruning multiplies `gradJ` by the mask each step; `Zero Fields` / `Zero Couplings` zero the corresponding gradient.

`'Pruning Mask Couplings'` may be either a path string (loaded once) or an in-memory `int` array; `Init_Pruning` overwrites the option to the materialized mask. The original input path is preserved under `'Pruning Mask Couplings Source'` for the manifest.

### Run-level provenance

Every training run and every pruning-mask invocation writes a `manifest.json` sidecar with: git commit + dirty flag + branch, full command line, input file paths and sha256s, the entire options dict (ndarrays summarised as `{shape, dtype, sha256}`), the master seed, OMP thread count, package versions, host/platform, and start/finish timestamps. Schema version 1, defined in `src/SBM/provenance.py`. The training driver also writes `command.sh` reproducing the invocation.

The figure-side equivalent is `lab_plotting.save_figure()` (in `scripts/lab_plotting.py`), which embeds the same git/timestamp data into PDF metadata and copies the calling script as a `<figure>.source.py` sidecar. Don't bypass either: figures saved with bare `fig.savefig()` lose all provenance.

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
  `fig.savefig()`. The wrapper embeds git provenance and writes a
  sidecar source script.
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
