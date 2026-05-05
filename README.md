# Stochastic Boltzmann Machine (SBM)

Python 3.11+. The Monte-Carlo is written in C++ with OpenMP, compiled automatically using scikit-build-core and CMake.

SBM infers the fields `h` and couplings `J` of a Potts model on a multiple sequence alignment by gradient descent against MCMC-estimated statistics. Each training run writes a self-describing directory: the model alongside a `manifest.json` that captures the git commit, RNG seed, command line, input file hashes, package versions, and timestamps — so any result can be reproduced from its manifest.

## Installation

### System dependencies

**macOS.** Apple Clang doesn't ship with OpenMP, so the build uses Homebrew LLVM:

```sh
brew install llvm libomp ninja cmake
```

The build picks up `/opt/homebrew/opt/llvm/bin/clang++`, `libomp`, and `ninja` automatically via `cmake/macos_llvm.cmake`.

**Linux.** Install `build-essential`, `python3-dev`, `cmake`, and `ninja-build` (or equivalents for your distro).

### Python environment

Use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs:

```sh
brew install uv               # macOS; or: pip install uv
uv venv
source .venv/bin/activate
uv pip install -e ".[plotting,analysis,dev]"
pre-commit install
```

`pip` works equally well — substitute `pip install` for `uv pip install`.

The optional dependency groups are:

| Group       | What it adds                                              |
|-------------|-----------------------------------------------------------|
| `plotting`  | `seaborn`, `plotly`, `POT`, `PyGSP`                       |
| `analysis`  | `scikit-learn`                                            |
| `notebook`  | `ipykernel`, `notebook`                                   |
| `sca`       | `pysca` (only needed for the SCA pruning strategy)        |
| `dev`       | `pre-commit`, `ruff`, `pytest`                            |

**C++ rebuilds.** scikit-build-core's editable install does not auto-rebuild on `.cpp` changes. After editing C++ sources, force a rebuild:

```sh
uv pip install -e . --force-reinstall --no-deps
```

## Dataset format

SBM expects a numerical alignment as a NumPy array of shape `(N_sequences, L_positions)`. Convert a FASTA with:

```python
import numpy as np
import SBM.utils.utils as ut

MSA = ut.load_fasta("data/fasta/CM.fasta")
np.save("data/MSA_array/MSA_CM.npy", MSA)
```

`load_fasta` silently drops sequences containing characters outside `-ACDEFGHIKLMNPQRSTVWY`.

Conventional layout under `data/`:

```
data/
├── fasta/
├── MSA_array/
│   └── MSA_CM.npy
└── Ind_train/
    └── Ind_train_CM.npy   # optional pre-split train indices
```

These paths are conventional, not enforced — every script takes paths as CLI args.

## Training

```sh
python scripts/train_sbm.py CM data/MSA_array/MSA_CM.npy \
    --N_iter 400 --N_chains 70 --m 1 \
    --theta 0.3 --ParamInit zero \
    --lambdJ 0 --lambdh 0 \
    --rep 1 --N_av 1 \
    --seed 42
```

Output: `results/CM/<run_id>/`, containing

- `model.npy` — the trained model dict (`J`, `h`, replicate seeds, exec times, train/test split, …);
- `manifest.json` — full run provenance (see schema below);
- `command.sh` — a self-contained shell script that re-invokes the run.

A worked end-to-end example, including pruning, lives at `pruning/CM_example.sh`.

### Run manifest schema (v1)

```jsonc
{
  "run_id": "20260504T143012Z-7af3b",
  "schema_version": 1,
  "command_line": ["python", "scripts/train_sbm.py", "CM", "..."],
  "code":     {"git_commit": "...", "git_dirty": false, "git_branch": "main"},
  "env":      {"python": "...", "platform": "...", "hostname": "...",
               "omp_num_threads_env": "8", "omp_num_threads_used": 8,
               "package_versions": {"numpy": "...", "scipy": "...", ...}},
  "inputs":   {"msa": {"path": "...", "sha256": "..."},
               "train_indices": {"path": null, "sha256": null},
               "pruning_mask":  {"path": "...", "sha256": "..."}},
  "options":  { /* full options dict; ndarrays summarised as
                  {"_kind": "ndarray", "shape": [...], "dtype": "...",
                   "sha256": "..."} */ },
  "seed": 42,
  "started_at": "...", "finished_at": "...", "wall_seconds": 1101.4,
  "outputs":  {"model": {"path": "...", "sha256": "..."}}
}
```

Same pattern is used for pruning masks: every mask written by `pruning/build_mask.py` gets a `<mask>.manifest.json` sidecar.

## Reproducibility notes

- **RNG seed.** Pass `--seed S` for a deterministic run. The seed propagates through the Python global RNG (test/train split, parameter init) and the C++ MCMC kernel (per-thread seed = `S + thread_id`). Per-replicate seeds are spawned via `np.random.SeedSequence`.
- **OpenMP.** The C++ kernels honor `OMP_NUM_THREADS`. Two runs with the same `--seed` reproduce bit-for-bit only if the thread count is also fixed — the manifest records both the requested and actual count for diagnosis.
- **Figures.** Save figures via `lab_plotting.save_figure()`; it embeds git commit + timestamp + script path into PDF metadata and writes a sidecar `<figure>.source.py` copy of the calling script.

## Jupyter

```sh
uv pip install -e ".[notebook]"
python -m ipykernel install --user --name SBM --display-name "Python (SBM)"
```

Then select the `Python (SBM)` kernel inside Jupyter.

## Citation

If you use this code or data, please cite the associated publication.
