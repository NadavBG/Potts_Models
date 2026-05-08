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

# 3. train, then sample, then render — three independent steps
RUN=results/CM/2026-05-05_CM-example_0
OMP_NUM_THREADS=4 bash scripts/run_sbm.sh SBM data/MSA_array/MSA_CM.npy --label CM-example
bash scripts/sample_sbm.sh "$RUN"
bash scripts/render_sbm.sh "$RUN"
```

`run_sbm.sh` writes `results/CM/2026-05-05_CM-example_0/` with:

- `model.npy` — trained `J`, `h`, train/test splits, etc.
- `manifest.json` — git commit, RNG seed, input hashes, package versions
- `command.sh` — self-contained re-runner

`sample_sbm.sh` adds `synthetic/align_T<T>_seed<seed>.npy` (+ a JSON sidecar) for every requested temperature. By default it samples both T=0.75 and T=1.0 (N=2000 each); pass `--temperature T1 [T2 ...]` to override.

`render_sbm.sh` regenerates `figs/`: PDFs at the top level plus `figs/inputs/{stats_<align_stem>.npy, sources.json}` (one stats cache per alignment plus a pointer file recording the sha256 of the model and every synthetic alignment that fed the figures). With multiple alignments under `synthetic/`, `correlations.pdf` becomes a (rows = temperatures × cols = 1st/2nd/3rd order) grid and `pca.pdf` a 1×(1+N_temps) layout. Re-running blows away `figs/` and rebuilds.

Run training again and `_idx` increments to `_1`. The CM-with-pruning worked example is `bash pruning/CM_example.sh`.

The two inputs you'll usually care about for training are the **MSA path** and the optional pruning masks (`--prune-J <J_mask.npy>`, `--prune-h <h_mask.npy>`). Everything else has a default; see `bash scripts/run_sbm.sh --help`.

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

The recommended entry point is `scripts/run_sbm.sh`. It trains the model and writes a self-describing run directory; sampling and figure rendering are separate scripts so each step can be re-run independently.

```sh
bash scripts/run_sbm.sh <MODE> <MSA_NPY> [options] [-- <train_sbm.py overrides>]
```

`<MODE>` is **`BM`** or **`SBM`** (Both L-BFGS, different values for m, N_chains, L2 regularizers). The two important inputs are `<MSA_NPY>` and the optional pruning masks (`--prune-J`, `--prune-h`); everything else has a default.

The options most users will touch:

| Flag | What it does | Default |
|---|---|---|
| `--label NAME` | Label embedded in the run dir name (`<date>_<label>_<idx>`) | family name |
| `--seed N` | Master RNG seed | `42` |
| `--prune-J PATH` | Restrict couplings to a binary `(L, L, q, q)` mask (see [Pruning](#pruning-workflow)) | none |
| `--prune-h PATH` | Restrict fields to a binary `(L, q)` mask (see [Pruning](#pruning-workflow)) | none |
| `--results-path DIR` | Output root | `<repo>/results` |
| `--N_iter N` | Gradient-descent iterations | 400 |
| `--N_chains N` | MCMC chains used to estimate model statistics each step | BM=100, SBM=50 |
| `--k_MCMC N` | Metropolis sweeps per chain per step | 100 000 |
| `--TestTrain 0\|1` | Hold out 20% of the MSA as a test set | 0 |
| `--theta X` | Similarity threshold for sequence reweighting | 0.3 |
| `--rep N` | Independent replicate runs | 1 |
| `--N_av N` | Models averaged per replicate | 1 |

Anything after `--` is forwarded verbatim to `scripts/train_sbm.py`, so the long-tail flags (`--m`, `--lambdJ`, `--ParamInit`, `--ignore_gaps`, `--record_every`, …) are reachable without bloating the bash CLI. Run `bash scripts/run_sbm.sh --help` for the short list and `python scripts/train_sbm.py --help` for the long list.

A few `options` keys (`SGD`, `Zero Fields`, `Zero Couplings`, `Precomputed_Stats`, `Infinite Mask Fields`) are only reachable by calling `SBM.SBM_GD.SBM_proteins.SBM(align, options)` directly from Python.

If you prefer to drive training yourself, call `scripts/train_sbm.py` directly:

```sh
python scripts/train_sbm.py <fam> <MSA.npy> --mod SBM --label CM-example --seed 42 [...]
```

To re-render figures on an already-trained run (e.g. after editing plot code) without retraining:

```sh
bash scripts/render_sbm.sh results/CM/2026-05-05_CM-example_0
```

`render_sbm.sh` deletes and regenerates `figs/` each call. By default it renders every figure whose data is present: `coupling_evol` always; `correlations` (a single rows=temperatures × cols=1st/2nd/3rd-order grid) and `pca` (a 1×(1+N_temps) layout) if at least one synthetic alignment is available (auto-discovered under `<run_dir>/synthetic/`, or pass `--synthetic-alignment PATH ...`); `energy`, `similarity`, `diversity`, `length` additionally if the run has `Test/Train>0` (each overlays every available temperature in one panel). Pass `--figs NAME [NAME ...]` to render an explicit subset.

## What you get

```text
results/<fam>/<YYYY-MM-DD>_<label>_<idx>/
├── model.npy        # pickled dict (see below)            ─┐  from
├── manifest.json    # full provenance — schema in src/SBM/provenance.py │  run_sbm.sh
├── command.sh       # self-contained shell script that re-invokes this run ─┘
├── synthetic/       # synthetic alignments (one per temperature)
│   ├── align_T0.75_seed42.npy
│   ├── align_T0.75_seed42.json    # sampling parameters + sha256s
│   ├── align_T1_seed42.npy
│   └── align_T1_seed42.json
└── figs/            # regenerated by every render_sbm.sh call
    ├── inputs/
    │   ├── render_figures.py        # canonical copy of the renderer
    │   ├── stats_align_T0.75_seed42.npy   # cache of compute_stats per alignment
    │   ├── stats_align_T1_seed42.npy
    │   └── sources.json             # paths + sha256 of model.npy and every
    │                                # synthetic alignment that fed these figures
    ├── coupling_evol.pdf    # ‖J‖ over training iterations (no alignment needed)
    ├── correlations.pdf     # one figure: rows = temperatures × cols = 1st/2nd/3rd order
    ├── pca.pdf              # 1×(1+N_temps) panels: natural + each artificial T
    └── (energy, similarity, diversity, length — only when --TestTrain 1;
         each overlays every available temperature in one panel)
```

Each PDF embeds the git commit, run id, and timestamp in its metadata. A single canonical copy of `render_figures.py` lives at `figs/inputs/render_figures.py`; we don't write a per-figure source sidecar. `figs/inputs/sources.json` records pointers to model + synthetic-alignment paths and their sha256s without duplicating the bytes.

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

The supported workflow is `scripts/sample_sbm.sh`:

```sh
# Default: N=2000 sequences per temperature, T = {0.75, 1.0},
# delta_t = options0.k_MCMC, seed = manifest's master seed.
bash scripts/sample_sbm.sh results/CM/<run_id>

# Override anything:
bash scripts/sample_sbm.sh results/CM/<run_id> \
    --N 5000 --temperature 0.5 1.5 --label highT --seed 7
```

Each invocation writes one `synthetic/align_T<T>_seed<seed>[_<label>].npy` per temperature plus a JSON sidecar carrying the sampling parameters, the run-dir path, and sha256s of both the model and the alignment. `sample_sbm.sh` refuses to overwrite an existing file at the target path; pass `--force` if you mean it, or use `--label` / `--output` to write somewhere else.

For ad-hoc Python use, `SBM.utils.utils.Create_modAlign(model_dict, N, delta_t=..., temperature=..., seed=...)` is the underlying primitive (returns an `(N, L)` `int64` array). `seed=None` falls back to numpy's global RNG.

## Pruning workflow

For larger families, restricting parameters to a small fraction (chosen by a data-derived statistic) regularizes the model and speeds up training. Couplings $J$ and fields $h$ have independent strategies and can be pruned together or separately. See `pruning/README.md` for details. End-to-end:

```sh
python pruning/build_mask.py --alg data/MSA_array/MSA_CM.npy \
    --strategies "sca" "dia" \
    --percent 98 \
    --label CM --path ./prune_output

python scripts/train_sbm.py SCAPruned_CM data/MSA_array/MSA_CM.npy \
    --prune-J ./prune_output/<run_id>/98.00_SCA_CM_SeqW_0.7.npy \
    --prune-h ./prune_output/<run_id>/98.00_Dia_CM_SeqW_0.7.npy \
    --N_iter 400 --N_chains 100 --m 20 \
    --lambdJ 0.01 --lambdh 0.01 --seed 42
```

Strategies — couplings: `fij` (pairwise frequencies), `cij` (pairwise correlations), `sca` (conserved correlations); fields: `fia` (per-site frequencies), `dia` (per-site KL divergence). The `sca` and `dia` strategies require `pip install -e ".[sca]"` for pySCA. Each generated mask gets a `<mask>.manifest.json` recording the inputs and parameters.

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
              "pruning_mask_couplings": {"path": "...", "sha256": "..."},
              "pruning_mask_fields":    {"path": null, "sha256": null}},
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
