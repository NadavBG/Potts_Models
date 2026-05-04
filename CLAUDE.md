# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`SBM` (Stochastic Boltzmann Machine) infers the fields `h` and couplings `J` of a Potts model on a multiple sequence alignment (MSA) by gradient descent against MCMC-estimated statistics. The Python package wraps two C++/OpenMP MCMC kernels that are compiled at install time via scikit-build-core + CMake.

## Build, install, run

The C++ extensions are required at runtime — `SBM.utils.utils` imports them eagerly. There is no pure-Python fallback. An editable install (`pip install -e .`) is the supported workflow; importing the source tree without building will fail.

```
pip install -r requirements.txt --no-cache-dir
pip install -e .
```

**macOS toolchain.** AppleClang has no OpenMP, so `pyproject.toml` forces `cmake/macos_llvm.cmake`, which hard-codes `/opt/homebrew/opt/llvm` and `libomp`. `brew install llvm libomp ninja` is required; an Intel-Mac or non-Homebrew prefix needs the toolchain file edited. On Linux, `python3-dev`, a GCC/G++ with OpenMP, CMake, and Ninja are sufficient.

**Python.** 3.11 is required (`requires-python = ">=3.11"`, wheel pinned to `cp311`).

**Tests.** There is no test suite. After non-trivial changes, run the demo as a smoke test:

```
bash scripts/demo-SBM-CM-family/run-SBM-CM-family.sh
```

It expects `data/MSA_array/MSA_CM.npy` to exist and writes a `.npy` pickle under `results/CM/`.

**Pruning workflow** lives in `pruning/` with its own `README.md` and `CM_example.sh`. Mask generation via the `"sca"` strategy depends on [pySCA](https://github.com/ranganathanlab/pySCA), which is **not** in `requirements.txt`; the `"fij"` and `"cij"` strategies do not need it.

## Architecture

### Layers

- `src/SBM/MonteCarlo/MCMC_Potts/` — full Potts MCMC (fields **and** couplings). Compiled C++ module `MonteCarlo_Potts`.
- `src/SBM/MonteCarlo/MCMC_PottsProf/` — profile-only MCMC (fields only, no couplings). Compiled C++ module `MonteCarlo_PottsProf`. Used when `Zero Couplings=True`.
- `src/SBM/utils/utils.py` — alignment IO (`load_fasta`, `save_fasta_from_array`), the packed-vector encoding (`Wj`/`Jw`), MCMC driver (`Create_modAlign`), reweighting/statistics (`CalcWeights`, `CalcStatsWeighted`, `CalcThreeCorrWeighted`), and the zero-sum gauge transform (`Zero_Sum_Gauge`).
- `src/SBM/SBM_GD/SBM_proteins.py` — the optimizer entry point `SBM(align, options, J0=None, h0=None)`, which dispatches to either:
  - `Model='BM'` — vanilla Boltzmann-machine gradient descent (`alpha`/`Learning_rate`),
  - `Model='SBM'` — limited-memory L-BFGS-style update (`AdvanceSearch` + `UpdateHessian`) with rank-`m` Hessian approximation.
- `pruning/build_mask.py` — standalone CLI that produces a binary `(L,L,q,q)` mask from MSA statistics (`fij`, `cij`, or pySCA `tildeC`); the mask is consumed by `SBM` via `options['Pruning Mask Couplings']`.
- `scripts/demo-SBM-CM-family/SBM-CM-family.py` — reference wrapper showing how `options` is built, how repetitions are averaged, and the output-naming convention.

### Data conventions

- Amino-acid alphabet: `"-ACDEFGHIKLMNPQRSTVWY"`, with `q = 21` and `0 = gap`. `MSA` arrays are `int` of shape `(N_sequences, L)`.
- Sequences containing any character outside the alphabet are **dropped** by `load_fasta` (mapped to `-1`, then filtered).
- `options['q']` and `options['L']` are derived from the alignment in `Init_options`; do not set them manually.
- Outputs are pickled `dict`s saved with `np.save(..., allow_pickle=True)`. Keys split into `options0` (model hyperparameters that change file naming) and `options1` (everything else); changing this split breaks the filename convention in `SBM-CM-family.py`.

### Packed-parameter layout (`Wj` / `Jw`)

`SBM` optimizes a flat vector `W` of length `L*q + L*(L-1)/2 * q*q`, packing `h` (size `L*q`) after the unique upper-triangular `J[i<j]` block. The C++ MCMC indexes directly into this layout — see `MonteCarlo_PottsMod.cpp` — so any change to the encoding in `Wj`/`Jw` must be mirrored in both `.cpp` files. `Jw` symmetrizes `J[i,j,a,b] = J[j,i,b,a]` on unpack.

### Gauge

The model is over-parameterized; `Zero_Sum_Gauge` projects `(J,h)` onto the zero-sum gauge and is applied **after** training in the demo before the parameters are averaged across replicas. Comparing `J`/`h` across runs without first applying this transform is meaningless.

### Statistics-matching loop

Each gradient step in `GradLogLike`:
1. Builds an artificial alignment with `Create_modAlign` (calls `mc.MC` or `mcp.MC`, which performs `delta_t` Metropolis sweeps per chain via OpenMP).
2. Computes `fi_mod, fij_mod` from the artificial alignment (uniform weights) and `fi, fij` from data (sequence-reweighted, optionally with pseudocount).
3. Returns `gradJ = fij_mod - fij + reg(J)` and `gradh = fi_mod - fi + reg(h)`. Pruning multiplies `gradJ` by the mask each step; `Zero Fields` / `Zero Couplings` zero the corresponding gradient.

`'Pruning Mask Couplings'` may be either a path string (loaded once) or an in-memory `int` array; `Init_Pruning` overwrites the option to the materialized mask.

## Gotchas

- **Hardcoded thread count.** Both `MonteCarlo_PottsMod.cpp` and `MonteCarlo_PottsProfMod.cpp` call `omp_set_num_threads(50)` unconditionally. This ignores `OMP_NUM_THREADS` and oversubscribes machines with fewer than 50 cores. If you change MCMC behavior, this is usually the first thing to revisit.
- **Per-thread RNG seeded from `time(nullptr)` + `thread_id`.** Two MCMC invocations within the same wall-clock second start from the same per-thread seed. Reproducibility of the *training run* relies on `options['Seed']` (logged in `output['options']['Seed']`); the *MCMC chains themselves* are not deterministic from that seed.
- **`load_fasta` silently drops sequences** containing non-canonical residues. The reported "Final shape" may be smaller than the FASTA record count.
- **`Zero Couplings=True` switches MCMC kernels** from `MonteCarlo_Potts` to `MonteCarlo_PottsProf` and packs `W` as just `h.flatten()`. Output-handling code branches on this, e.g. `output['J_norm']` becomes `None`.
- **`'Pruning Mask Couplings'` is mutated in place** by `Init_Pruning` (path → array). Do not pass the same `options` dict into a second `SBM` call expecting a clean re-init.
- **No `__init__.py` exports**: importing `SBM` itself returns essentially nothing; users import `SBM.SBM_GD.SBM_proteins` and `SBM.utils.utils` directly.

## Project-specific code conventions to follow

- Treat `Wj`/`Jw` as the canonical (de)serialization between the optimizer and the model; do not invent a parallel encoding.
- After training, parameters meant for downstream comparison must pass through `Zero_Sum_Gauge` first.
- New C++ kernels: register them in `CMakeLists.txt` (each module gets its own `add_library(... MODULE)` + install target), keep the `set_target_properties(... PROPERTIES PREFIX "")` line so Python finds the `.so` by module name, and mirror the `Wj` packing exactly.
- New `options` keys: add a default in `ParseOptions` so existing callers don't need updates, and document any in-place mutation.
