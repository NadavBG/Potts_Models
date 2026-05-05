# SBM — Stochastic Boltzmann Machine

A Potts-model inference tool for protein multiple-sequence alignments (MSAs).

Given an MSA, SBM learns the fields `h_i(a)` and pairwise couplings `J_ij(a,b)` of a Potts model whose single- and pairwise-residue frequencies match the data. The optimizer is L-BFGS-style gradient descent against statistics estimated from a parallel C++/OpenMP MCMC sampler. Every trained model is reproducible from a `manifest.json` written next to it (git commit, RNG seed, input hashes, package versions, full options).

## Quick start

```sh
# 1. system tools (macOS — Linux equivalents below)
brew install uv llvm libomp ninja cmake

# 2. environment
uv python install 3.12
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[plotting,analysis,dev]"

# 3. train a model on the bundled CM example (~30 s on 4 cores)
OMP_NUM_THREADS=4 python scripts/train_sbm.py CM data/MSA_array/MSA_CM.npy \
    --N_iter 100 --N_chains 50 --k_MCMC 5000 \
    --rep 1 --N_av 1 --m 1 \
    --theta 0.3 --ParamInit zero \
    --lambdJ 0 --lambdh 0 \
    --seed 42
```

The model lands in `results/CM/<run_id>/`. See `bash scripts/examples/cm-family/run.sh` for the canonical example, and `pruning/CM_example.sh` for the same with a pruning mask.

> **Heads-up: avoid conda-forge / miniforge Python.** Its `python@3.12` build has a libpython ABI quirk that segfaults during numpy's `import_array()` when our C++ extension is loaded. Use `uv python install` (above) or Homebrew's `python@3.12`.

## Inputs

SBM trains from a numerical alignment: a NumPy array of shape `(N_sequences, L_positions)`, dtype `int`, with the alphabet

```text
gap A C D E F G H I K L M N P Q R S T V W Y
 0  1 2 3 4 5 6 7 8 9 ...                  20      (q = 21)
```

To convert a FASTA:

```python
import numpy as np
import SBM.utils.utils as ut

MSA = ut.load_fasta("data/fasta/CM.fasta")    # silently drops non-canonical-residue rows
np.save("data/MSA_array/MSA_CM.npy", MSA)
```

Conventional layout (paths are conventional, not enforced — every script takes a path argument):

```text
data/
├── fasta/                         # raw inputs
├── MSA_array/MSA_<fam>.npy        # numerical alignments
└── Ind_train/Ind_train_<fam>.npy  # optional pre-split train indices
```

## Training

```sh
python scripts/train_sbm.py <fam> <MSA.npy> [options]
```

The options that matter most in practice:

| Flag | What it controls | Sensible default |
|---|---|---|
| `--N_iter` | Number of gradient-descent steps. Bigger MSA / richer model → more iters. | 400 |
| `--N_chains` | Number of MCMC chains used to estimate model statistics each step. Larger = lower-variance gradient, more wall time. | 50–100 |
| `--k_MCMC` | Metropolis sweeps per chain per step. Larger = better mixing. | 5000–100 000 |
| `--m` | L-BFGS Hessian rank (only meaningful for `--mod SBM`). | 1–20 |
| `--theta` | Sequence-reweighting similarity threshold (`1 - hamming_distance` cutoff). | 0.2–0.3 |
| `--ParamInit` | `zero`, `profile`, `random`, or `custom`. Use `profile` to start from data-derived fields. | `zero` |
| `--lambdJ`, `--lambdh` | L2 regularization on couplings / fields. | 0–0.01 |
| `--rep` | Number of *independent* runs (each goes to its own `<run_id>/`). | 1 |
| `--N_av` | Number of replicates *averaged within one run* (each gets a sub-seed via `SeedSequence`). | 1 |
| `--prune <mask.npy>` | Restrict couplings to a binary mask (see Pruning below). | none |
| `--seed S` | Master RNG seed. Required for bit-identical reproduction. | none (auto from `time()`) |
| `--results_path` | Output root. | `<repo>/results` |

Run `python scripts/train_sbm.py --help` for the full flag list, including `--TestTrain`, `--ignore_gaps`, `--train_file`, `--mod` (`BM` vs `SBM`), and `--Input_MSA` positional argument ordering. Some `options` keys (`SGD`, `Zero Fields`, `Zero Couplings`, `Precomputed_Stats`, `Infinite Mask Fields`) are only reachable from Python by calling `SBM.SBM_GD.SBM_proteins.SBM(align, options)` directly — they aren't exposed on the CLI.

## What you get

```text
results/<fam>/<run_id>/
├── model.npy        # pickled dict (see below)
├── manifest.json    # full provenance — schema in src/SBM/provenance.py
└── command.sh       # self-contained shell script that re-invokes this run
```

Loading the model:

```python
import numpy as np

m = np.load("results/CM/<run_id>/model.npy", allow_pickle=True).item()

m["J"]           # (L, L, q, q) — averaged couplings, zero-sum gauged
m["h"]           # (L, q)       — averaged fields, zero-sum gauged
m["W_all"]       # (N_av, L*q + L*(L-1)/2*q*q) — packed weights per replicate
m["Seeds"]       # per-replicate seeds (uint32)
m["Train"]       # the training-set rows used
m["Test"]        # the held-out rows (or None if --TestTrain 0)
m["J_norm"]      # Frobenius norm of J over training iterations
m["options0"]    # subset of options (kept for back-compat; full set is in manifest.json)
```

To re-run a result exactly: `bash results/<fam>/<run_id>/command.sh` from the repo root.

## Sampling synthetic sequences from a trained model

```python
import numpy as np
import SBM.utils.utils as ut

m = np.load("results/CM/<run_id>/model.npy", allow_pickle=True).item()
synthetic = ut.Create_modAlign(m, N=200, delta_t=10000, seed=0)   # (200, L) int array

# Or directly to FASTA:
ut.save_fasta_from_array("results/CM/<run_id>/model.npy",
                         "results/CM/<run_id>/synthetic.fasta",
                         Nb_seq=200)
```

`delta_t` is the number of Metropolis sweeps per chain. `seed=None` falls back to numpy's global RNG.

## Pruning workflow

For larger families, restricting couplings to a small fraction of position-pairs (chosen by a data-derived statistic) regularizes the model and speeds up training. See `pruning/README.md` for details. End-to-end:

```sh
python pruning/build_mask.py --alg data/MSA_array/MSA_CM.npy \
    --strategies "sca" \
    --percent 98 \
    --label CM --path ./prune_output

python scripts/train_sbm.py SCAPruned_CM data/MSA_array/MSA_CM.npy \
    --prune ./prune_output/98.00_SCA_CM_SeqW_0.7.npy \
    --N_iter 400 --N_chains 100 --m 20 \
    --lambdJ 0.01 --lambdh 0.01 --seed 42
```

The mask supports three strategies — `fij` (pairwise frequencies), `cij` (pairwise correlations), `sca` (conserved correlations, requires `pip install -e ".[sca]"` for pySCA). Each generated mask gets a `<mask>.manifest.json` recording the inputs and parameters.

## Reproducibility

- **`--seed S`** seeds the Python global RNG (controls test/train split, parameter init) **and** the C++ MCMC kernel (per-thread seed = `S + thread_id`). Per-replicate seeds are derived via `np.random.SeedSequence(S).spawn(N_av)`.
- Bit-identical reproduction requires fixing `OMP_NUM_THREADS` and `--N_chains` too; the manifest records both the requested and actual thread count.
- The model's `J`, `h`, `W_all`, `Seeds` arrays are bit-identical across runs with the same seed; the saved `model.npy` bytes still differ because the dict includes wall-clock `Execution times`. Compare arrays, not pickles.
- **Figures** must go through `lab_plotting.save_figure(fig, path)` (in `scripts/lab_plotting.py`); it embeds git commit + timestamp into PDF metadata and copies the calling script as a `<figure>.source.py` sidecar. Bare `fig.savefig()` loses provenance.

## Reference

### Optional dependency groups

| Group | Adds |
|---|---|
| `plotting` | `seaborn`, `plotly`, `POT`, `PyGSP` |
| `analysis` | `scikit-learn` |
| `notebook` | `ipykernel`, `notebook` |
| `sca` | `pysca` (only needed for SCA pruning) |
| `dev` | `pre-commit`, `ruff`, `pytest` |

The lock file (`requirements.lock`) was generated against `cpython-3.12.13-macos-aarch64-none` with `[plotting,analysis]`. Use `uv pip sync requirements.lock` for a deterministic install (then `uv pip install -e . --no-deps`).

### Run manifest schema (v1)

```jsonc
{
  "run_id": "20260504T143012Z-7af3b",
  "schema_version": 1,
  "command_line": ["python", "scripts/train_sbm.py", "CM", "..."],
  "code":  {"git_commit": "...", "git_dirty": false, "git_branch": "main"},
  "env":   {"python": "...", "platform": "...", "hostname": "...",
            "omp_num_threads_env": "8", "omp_num_threads_requested": 8,
            "package_versions": {"numpy": "...", "scipy": "...", ...}},
  "inputs":  {"msa": {"path": "...", "sha256": "..."},
              "train_indices": {"path": null, "sha256": null},
              "pruning_mask":  {"path": "...", "sha256": "..."}},
  "options": { /* full options dict; ndarrays summarised as
                  {"_kind": "ndarray", "shape": [...], "dtype": "...", "sha256": "..."} */ },
  "seed": 42,
  "started_at": "...", "finished_at": "...", "wall_seconds": 1101.4,
  "outputs": {"model": {"path": "...", "sha256": "..."}}
}
```

### System dependencies

**macOS.** Apple Clang doesn't ship with OpenMP — the build uses Homebrew LLVM (`/opt/homebrew/opt/llvm`) via `cmake/macos_llvm.cmake`. Intel-Mac or non-Homebrew prefixes need that toolchain file edited.

**Linux.** Install `build-essential`, `python3-dev`, `cmake`, `ninja-build`. The toolchain file is a no-op on non-Apple platforms.

### After editing C++

scikit-build-core's editable install does **not** rebuild on `.cpp` changes. Force one with:

```sh
uv pip install -e . --force-reinstall --no-deps
```

### Jupyter

```sh
uv pip install -e ".[notebook]"
python -m ipykernel install --user --name SBM --display-name "Python (SBM)"
```

Pick `Python (SBM)` as the kernel.

## Citation

If you use this code or data, please cite the associated publication.
