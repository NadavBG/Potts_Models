# SBM — Stochastic Boltzmann Machine

A Potts-model inference tool for protein multiple-sequence alignments (MSAs).

Given an MSA, SBM learns the fields `h_i(a)` and pairwise couplings `J_ij(a,b)` of a Potts model whose single- and pairwise-residue frequencies match the data. The optimizer is L-BFGS against statistics estimated from a parallel C++/OpenMP MCMC sampler.

The whole workflow — MSA statistics figure, optional pruning masks, inference, synthetic sampling, figures, and an optional ProteinMPNN foldability sweep — is driven by **one Snakemake pipeline** from **one YAML config file**. Each run lands in its own directory carrying everything needed to know exactly which parameters produced which figure.

---

## Quick start

```sh
# 1. system tools (macOS; Linux equivalents in "System dependencies" below)
brew install uv llvm libomp ninja cmake

# 2. environment (NOT conda — see the heads-up below)
uv python install 3.12
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[plotting,analysis,dev,workflow]"

# 3. run the whole pipeline from a config file
python scripts/iter.py run CM-bm-dense "first-try"
```

That last command reads `config/params_CM-bm-dense.yaml`, creates a fresh run directory `results/CM-bm-dense/iter-001-first-try/`, and builds everything into it: the MSA-statistics figure, the trained model, synthetic alignments, all the figures, the ProteinMPNN sweep, and a provenance manifest. (`CM-bm-dense` is a plain BM run with no pruning; the `params_CM-bm-*` variants under `config/` add coupling/field pruning.)

To try the fast smoke-test config first (5 iterations, tiny everything — finishes in well under a minute):

```sh
snakemake --configfile config/params_tiny.yaml --cores 8 all
```

> **Heads-up: avoid conda-forge / miniforge Python.** Its `python@3.12` build has a libpython ABI quirk that segfaults during numpy's `import_array()` when our C++ extension is loaded. Use `uv python install` (above) or Homebrew's `python@3.12`. The pipeline runs inside this uv venv; the Snakefile deliberately has no `conda:` directive.

---

## How the pipeline works

**One config = one run.** You write (or copy) a `config/params_<run_name>.yaml`, and the pipeline turns it into a run directory. The config is validated up front — unknown keys are an error — so a typo fails loudly instead of being silently ignored.

**Two ways to launch:**

```sh
# Recommended: mint a new iteration directory and run it in one step.
python scripts/iter.py run <run_name> "<short tag>"

# Or do it in two steps (mint, then run Snakemake yourself):
python scripts/iter.py new <run_name> "<short tag>"     # prints the exact command
snakemake --configfile config/params_<run_name>.yaml \
          --config run_root=results/<run_name>/iter-NNN-<tag> --cores 8 all
```

`scripts/iter.py` also has `list <run_name>` and `latest <run_name>`. By default the config is `config/params_<run_name>.yaml`; override with `--config PATH`.

**Run directories preserve history.** Each launch mints `results/<run_name>/iter-NNN-<tag>/` (the index `NNN` auto-increments) and updates a `latest` symlink. Re-running Snakemake against the *same* `run_root` overwrites in place and only re-runs stages whose inputs changed; start a new iteration to keep the previous one.

**The DAG** (run `snakemake --configfile <cfg> -n` to preview it):

```text
MSA ──┬─► msa_stats ─────────────────────────► msa_stats.pdf      (independent: no model needed)
      ├─► build_mask_J ─► masks/J_mask.npy ─┐
      ├─► build_mask_h ─► masks/h_mask.npy ─┤
      └──────────────────────────────────────┴─► train ─► model.npy + manifest.json
                                                    │
                                                    ├─► sample(T=0.75) ─► synthetic/align_T0.75.npy
                                                    ├─► sample(T=1.0)  ─► synthetic/align_T1.npy
                                                    ├─► mpnn_sweep ─► synthetic/mpnn_sweep_seed<seed>/
                                                    └─► render ─► figs/
                                                            │
   config_snapshot + every stage ──────────────────────────┴─► run_manifest.json
```

The `msa_stats` figure depends **only on the MSA** — you can make it without training a model:

```sh
snakemake --configfile config/params_<run_name>.yaml --config run_root=<dir> --cores 8 msa_stats_only
```

`build_mask_*` exist only when `pruning.enabled`, and `mpnn_sweep` only when `mpnn.enabled`.

### What a run directory contains

```text
results/<run_name>/iter-NNN-<tag>/
├── config_snapshot.yaml   # the exact, validated parameters for this run
├── iteration_note.md      # YAML front-matter (git commit, timestamp) + your hypothesis
├── msa_stats.pdf          # MSA-only statistics figure (no model needed)
├── masks/                 # only if pruning.enabled
│   ├── J_mask.npy   J_mask.manifest.json
│   └── h_mask.npy   h_mask.manifest.json
├── model.npy              # trained J, h, train/test splits, etc. (pickled dict)
├── manifest.json          # training provenance: git, seed, input hashes, options, versions
├── command.sh             # self-contained re-runner for the training step
├── train_meta.json        # small inter-stage summary
├── synthetic/
│   ├── align_T0.75.npy   align_T0.75.json     # one alignment + sidecar per temperature
│   ├── align_T1.npy      align_T1.json
│   └── mpnn_sweep_seed42/                      # only if mpnn.enabled
│       ├── align_T*.npy   control_*.npy   mpnn_scores.json   manifest.json
├── figs/                  # regenerated wholesale by every render
│   ├── coupling_evol.pdf  params.pdf  correlations.pdf  pca.pdf  (energy/similarity/diversity/length/mpnn as data allows)
│   └── inputs/
│       ├── stats_align_T*.npy   # cache of the (slow) 3-point-correlation stats, one per alignment
│       └── sources.json         # paths + sha256 of model.npy and every synthetic alignment that fed the figures
├── run_manifest.json      # one-file aggregate: config + git + per-stage timings + artifact hashes
└── logs/                  # one log per stage + logs/timings/<stage>.json
```

The MSA figure lives at the run-dir top level (`msa_stats.pdf`), **not** under `figs/`, because `render` deletes and regenerates `figs/` on every call.

### Knowing which parameters produced which figure

The provenance chain is end-to-end and needs no extra bookkeeping from you:

`config_snapshot.yaml` (exact validated params) → `manifest.json` (training: git commit, seed, input + mask sha256s, full options) → `figs/inputs/sources.json` (sha256 of the model and each synthetic alignment that fed the figures) + each PDF's metadata (git commit, timestamp, and a `sbm_run_id` keyword) → `run_manifest.json` (aggregates it all, with per-stage timings). `iteration_note.md` carries your human-written hypothesis.

---

## Writing a config

Start from `config/params_CM-bm-dense.yaml` (a full BM run with MPNN, no pruning) or the minimal `config/params_tiny.yaml`. The schema is defined and validated in `src/SBM/workflow_config.py`. Fields, with defaults:

```yaml
run_name: CM-bm-dense           # required; names results/<run_name>/...
msa_fasta: data/fasta/CM.fasta  # required; aligned FASTA, encoded to .npy by the encode_msa rule (see "Inputs")
description: ""                 # free text, copied into manifests
family: CM                      # enables CM catalytic-sector annotation in figures
seed: 42                        # master RNG seed (training + sampling)
omp_num_threads: null           # pin (e.g. 8) for bit-identical re-runs; null = OpenMP default

msa_stats:                      # the MSA-only statistics figure
  enabled: true
  theta: 0.7                    # sequence-reweighting similarity threshold
  lbda: 0.03                    # pseudocount
  Dia_prior: gap-corrected      # gap-corrected | uniform
  sector: emily                 # emily | rama | none

pruning:                        # optional; omit or enabled:false to skip
  enabled: true
  theta: 0.7
  lbda: 0.03
  label: CM
  Dia_prior: gap-corrected
  couplings: { strategy: sca, percent: 98.0 }   # J mask: strategy fij|cij|sca
  fields:    { strategy: dia, percent: 98.0 }   # h mask: strategy fia|dia

train:
  mode: BM                      # BM | SBM (see table below)
  optimizer: LBFGS              # LBFGS | GD (GD rarely needed)
  N_iter: 400
  N_chains: 100                 # BM=100, SBM=50
  m: 20                         # L-BFGS memory: BM=20, SBM=1
  lambda_J: 0.01                # L2 on couplings: BM=0.01, SBM=0
  lambda_h: 0.01                # L2 on fields:    BM=0.01, SBM=0
  theta: 0.3                    # reweighting threshold for training
  k_MCMC: 100000                # Metropolis sweeps per chain per gradient step
  TestTrain: 0                  # 1 holds out 20% as a test set (unlocks energy/length figures)
  record_every: 5               # record ‖J‖ every N iterations
  ignore_gaps: false

sample:
  N: 2000                       # synthetic sequences per temperature
  temperatures: [0.75, 1.0]     # one sampling job per temperature

figures:
  which: null                   # null = every figure whose data is present; or a list e.g. [coupling_evol, params]
  sector: emily

mpnn:                           # ProteinMPNN foldability sweep; on by default
  enabled: true
  pdb: data/structures/1ECM.pdb
  chain: A
  seed: null                    # null = inherit the run's master seed
  temperatures: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  N_per_T: 100
  controls: [wt, random, shuffled, natural]
  model_name: v_48_020
  skip_scoring: false           # true samples + writes controls but skips scoring (no torch/PROTEINMPNN_PATH needed)
```

**BM vs SBM** are the same L-BFGS algorithm with different knobs (per Summary Note 3):

| Parameter | BM (positive control) | SBM (stochastic regularization) |
| --- | --- | --- |
| `m` (L-BFGS memory) | 20 | 1 |
| `lambda_J`, `lambda_h` | 0.01 | 0 |
| `N_chains` | 100 | 50 |

The `sca` (couplings) and `dia` (fields) pruning strategies require pySCA: `uv pip install -e ".[sca]"`. The ProteinMPNN sweep with `skip_scoring: false` needs a clone of [`dauparas/ProteinMPNN`](https://github.com/dauparas/ProteinMPNN) on `PROTEINMPNN_PATH`; set `skip_scoring: true` (as the tiny config does) to exercise sampling without it. See `docs/MPNN_FOLDABILITY.md`.

---

## Inputs

SBM trains from a numerical alignment: a NumPy array of shape `(N_sequences, L_positions)`, dtype `int`, with the alphabet

```text
gap A C D E F G H I K L M N P Q R S T V W Y
 0  1 2 3 4 5 6 7 8 9 ...                  20      (q = 21)
```

The pipeline takes an **aligned FASTA** (the config's `msa_fasta` field) and encodes it to this array for you: the `encode_msa` rule writes `results/<run>/inputs/msa.npy` plus a manifest recording the input hash and any dropped sequences. To produce one by hand (e.g. for the lower-level CLIs below):

```sh
python scripts/encode_msa.py --fasta data/fasta/CM.fasta --out msa.npy
```

or programmatically (`load_fasta` logs and returns the rows it drops for non-canonical residues):

```python
import numpy as np
import SBM.utils.utils as ut

MSA, dropped = ut.load_fasta("data/fasta/CM.fasta", return_dropped=True)
np.save("msa.npy", MSA)
```

Conventional layout (the raw FASTA is the committed source of truth; encoded arrays are derived, per-run artifacts):

```text
data/
├── fasta/<fam>.fasta             # raw inputs (committed, immutable)
└── structures/<pdb>.pdb          # PDBs for the ProteinMPNN sweep
results/<run>/inputs/msa.npy      # derived integer alignment (one per run)
```

Loading a trained model:

```python
import numpy as np

m = np.load("results/<run_name>/iter-001-<tag>/model.npy", allow_pickle=True).item()
m["J"]        # (L, L, q, q) — averaged couplings, zero-sum gauged
m["h"]        # (L, q)       — averaged fields, zero-sum gauged
m["Train"]    # training-set rows used;  m["Test"] — held-out rows (or None if TestTrain 0)
m["options0"] # subset of options (full set is in manifest.json)
```

---

## Running the steps by hand

The pipeline calls a set of standalone CLIs; you can also drive them directly (this is what the old workflow did, and it still works). Each writes its own provenance.

```sh
# Encode the aligned FASTA into the integer MSA the CLIs below consume:
python scripts/encode_msa.py --fasta data/fasta/CM.fasta --out msa.npy

# MSA statistics figure (MSA only — no model):
python scripts/render_msa_stats.py --msa msa.npy --out msa_stats.pdf

# Train -> auto-named results/<fam>/<YYYY-MM-DD>_<label>_<idx>/ (use --run-dir to fix the path):
bash scripts/run_sbm.sh BM msa.npy --label CM-example --prune-J <J_mask.npy> --prune-h <h_mask.npy>

# Sample synthetic alignments from a run:
bash scripts/sample_sbm.sh results/CM/<run_id> --N 2000 --temperature 0.75 1.0

# Render figures for a run:
bash scripts/render_sbm.sh results/CM/<run_id>

# Build pruning masks:
python pruning/build_mask.py --alg msa.npy --strategies sca dia --percent-J 98 --percent-h 98 --label CM --path ./prune_output
```

`run_sbm.sh` writes `model.npy`, `manifest.json`, and `command.sh`. `<MODE>` is `BM` or `SBM`; the two inputs you usually care about are the MSA path and the optional `--prune-J` / `--prune-h` masks. Anything after `--` is forwarded to `scripts/train_sbm.py` (run `python scripts/train_sbm.py --help` for the full flag list). `sample_sbm.sh` refuses to overwrite existing samples unless you pass `--force`. The legacy worked example `bash pruning/CM_example.sh` chains these together; it is superseded by the config-driven pipeline (e.g. `config/params_CM-bm-dense.yaml`).

---

## Combining two models: energy of a sequence under both

Once you have **two trained models** (e.g. two different families), you can score any sequence under *both* and report `E_A`, `E_B`, and the combined `E_tot = w_A·E_A + w_B·E_B`. This is a separate, two-model pipeline (`Snakefile.combine`); the progress note and math are in `docs/two_model_progress.md`, and the production couplings-aware aligner is documented in `docs/POTTS_ALIGN.md`.

The catch: a raw sequence is **not in either model's frame** (the two models have different aligned lengths and are non-homologous), so it must be aligned to each model independently. By default (`method: map`) each sequence is threaded into each model's frame the **same way** — the single best alignment per model — so `E_A` and `E_B` are computed by an identical procedure and are directly comparable. If you instead want the thermodynamically-principled energy that integrates over alignment uncertainty, `method: marginal` estimates the free energy by importance sampling and reports an effective sample size (ESS); see the method table.

### Run it

The combine pipeline reuses the **same environment** as the rest of the project (no extra dependencies). It needs two already-trained run directories. Everything is driven by one config:

```sh
# one command: mint an iteration dir under combine/ and run the whole thing
python scripts/iter.py run combine-CM-PPIC "baseline" --snakefile Snakefile.combine

# or drive Snakemake directly
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC.yaml --cores 8 all

# fast smoke test first (a handful of sequences, seconds):
snakemake -s Snakefile.combine --configfile config/params_combine-tiny.yaml --cores 8 all
```

Combine runs land under **`combine/<run_name>/`** — a separate (git-ignored) tree from the single-model `results/`, so dual-model runs never mix with single-model ones.

**Where it runs.** Every method runs on your Mac. `potts_align` (the production couplings-aware aligner) is pure numpy; for a *large* query set its alignment can optionally be pre-built on a Slurm array and pulled back as a cache before scoring locally — that optional flow is `docs/POTTS_ALIGN.md` §11. A small query set needs no cluster step at all.

### Write a combine config

Copy `config/params_combine-CM-PPIC.yaml`. The schema is validated by `src/SBM/combine_config.py` (unknown keys are an error). Fields, with defaults:

```yaml
run_name: combine-CM-PPIC      # required; names combine/<run_name>/...
description: ""
seed: 42                       # required for the marginal estimator; logged
omp_num_threads: null

models:                        # required: exactly two {name, run_dir} (E_tot weights
                               # are derived post-hoc from the naturals, not set here)
  - name: CM-bm-dense          # the EXACT model variant — labels the figure axes,
    run_dir: results/CM-bm-dense/iter-002-base-model   # query groups, and manifests
    weight: 1.0
  - name: PPIC-dense
    run_dir: results/PPIC-dense/iter-001-baseline
    weight: 1.0

query:
  source: model_sets           # model_sets = each model's own natural + synthetic
  include: [natural, synthetic]#   sequences; or `fasta` + `fasta: path/to.fasta`
  fasta: null                  # used only when source: fasta
  cap_per_group: 500           # seeded subsample per group (0 = no cap); bounds the
                               #   marginal cost on large natural MSAs; drop is logged

scoring:
  method: map                  # auto | in_frame | map | marginal | potts_align (see
                               #   below). map is the schema default (same alignment
                               #   procedure for both models); potts_align is the
                               #   production couplings-aware aligner (docs/POTTS_ALIGN.md)
  n_samples: 1000              # importance-sampling draws (used only by `marginal`)
  ess_threshold: 100.0         # below this, a `marginal` estimate is flagged unreliable

figures:
  enabled: true                # one E_A-vs-E_B scatter + marginals, captioned with
                               #   each model's exact run dir + sha256
```

To score **your own sequences** instead of the models' training/synthetic sets, set `query.source: fasta` and `query.fasta: path/to/your.fasta` (a normal amino-acid FASTA; mixed lengths are fine — each sequence is aligned to each model independently).

**Methods** (`scoring.method`):

| method | what it does | when |
| --- | --- | --- |
| `map` | **(schema default)** Viterbi-align to each model, full Potts energy on that single best path; same procedure for both models | you want the single best alignment + comparable `E_A`, `E_B` |
| `potts_align` | **(production)** couplings-aware gap-placement aligner: minimizes the exact Potts energy over frames (uses the full `J`); pure numpy, provably global for few gaps | the most accurate alignment for `N ≤ L` — spec + cluster runbook in `docs/POTTS_ALIGN.md` |
| `marginal` | free energy `−log Σ_a e^(−E(x,a))` by importance sampling; reports ESS + MC stderr | the principled model-evidence; accounts for alignment ambiguity |
| `in_frame` | exact Potts sum; requires the sequence already be in the model's frame | sequences already aligned to a model |
| `auto` | `in_frame` (original MSA alignment) for a sequence's home model, `marginal` for the other | fast per-model scoring — **but** `E_A`/`E_B` use different aligners, so *not* comparable (it warns) |

`map` is the *fields*-MAP (Viterbi under the HMM, which aligns using conservation but ignores the couplings `J`), so the alignment it picks is good but not guaranteed energy-optimal. The couplings-aware upgrade is **`method: potts_align`**: it minimizes the exact in-frame Potts energy (the full `J`) over gap placements — provably global when the gap count is small, parallel tempering otherwise — and requires `N ≤ L` (raw sequence no longer than the model frame). It is pure numpy and runs on the Mac; a large query set can optionally pre-build its alignment cache on a Slurm array. Full spec, cost model, the honest comparison to the retired DCAlign approach, and the cluster runbook: `docs/POTTS_ALIGN.md`.

### What you get

```text
combine/<run_name>/iter-NNN-<tag>/
├── config_snapshot.yaml   # the exact validated config
├── models.json            # the two models: name, run_dir, sha256, length L
├── query/query.fasta      # the sequences scored (+ groups.json: origin + group per id)
├── scores.tsv             # tidy, one row per (sequence × model): energy, ess, mc_stderr, ...
├── scores_detail.json     # per sequence: E_A, E_B, E_tot, diagnostics, best alignment per model
├── alignments.txt         # HUMAN-READABLE: each sequence's best alignment under EACH model,
│                          #   stacked side-by-side, with both energies (see below)
├── manifest.json          # scoring provenance: model hashes, method, seed, ESS summary, git
├── energy_weights.json    # E_tot weights derived post-hoc from the naturals (+ energy_weight_sweep.tsv)
├── figs/two_model_energy.pdf  # E_A vs E_B scatter, captioned with the exact models used
├── figs/energy_weights.pdf    # weighted median native energy vs w_A; the crossing is the derived weight
└── run_manifest.json      # aggregate
```

`alignments.txt` is the file to read first. For every sequence it shows the raw query and how it threads into each model's frame, with both energies:

```text
### CM-bm-dense|natural|0   group=CM-bm-dense/natural   E_tot=-320.692
  query (N=94): PQDCAGMVDIRAEIDML...
  [CM-bm-dense]  E=-277.352  method=in_frame
    PQDCAGMVDIRAEIDML...-VRAKERFEAML...  (L=96)
  [PPIC-dense]  E=-43.340  method=marginal  ESS=1.0
    -PQDCAG-MVDIRAEIDML...EKM-YRDLVNYF...  (L=91)
```

The two frames are independent (different lengths, not column-aligned). A native of one family sits low on its own model's axis and high on the other's.

### Reading the ESS (only for `method: marginal`)

ESS comes out of the importance-sampling pass, so it is reported only when you run `method: marginal` (the default `map` is a deterministic single alignment with no ESS). The marginal estimate is only as good as its ESS. **A low ESS is not always a problem:** when a sequence aligns essentially one way (e.g. a native in its own family), the alignment posterior is sharply peaked, ESS is near 1 *by construction*, and the marginal energy still agrees with the MAP and in-frame energies. A low ESS on a genuinely ambiguous cross-family alignment, on the other hand, means the estimate is dominated by one lucky sample and should be treated as an upper bound — raise `n_samples`, or switch to the couplings-aware `method: potts_align` (`docs/POTTS_ALIGN.md`) or annealed importance sampling. Either way the run **warns loudly** and records the ESS in `scores.tsv`, `scores_detail.json`, and `manifest.json`; nothing is hidden.

---

## Reproducibility

- **`seed`** seeds the Python global RNG (test/train split, parameter init) **and** the C++ MCMC kernel (per-thread seed = `seed + thread_id`). Per-replicate seeds are derived via `np.random.SeedSequence(seed).spawn(N_av)`. The pipeline's per-temperature sampling uses `seed + temperature_index`.
- **Bit-identical** training arrays require fixing the thread count too. The shipped configs leave `omp_num_threads: null`, so default runs are **not** bit-reproducible; set `omp_num_threads` to a fixed integer (the pipeline then pins `OMP_NUM_THREADS` before the kernel loads) and keep `N_chains` fixed. Both are recorded in `manifest.json`.
- The model's `J`, `h`, `W_all`, `Seeds` arrays are bit-identical across runs with the same seed + thread count; the `model.npy` *bytes* still differ because the dict embeds wall-clock execution times. **Compare the arrays (or the manifest's array sha256s), not the pickle.**
- **Figures** go through `lab_plotting.save_figure(fig, path)`, which embeds the git commit, script path, and timestamp into the PDF metadata. (It no longer writes a copy of the source script next to each figure.) Don't use bare `fig.savefig()` — it loses the provenance metadata.

---

## Reference

### Optional dependency groups

| Group | Adds |
| --- | --- |
| `workflow` | `snakemake`, `pyyaml` (the pipeline) |
| `plotting` | `seaborn`, `plotly`, `POT`, `PyGSP` |
| `analysis` | `scikit-learn` |
| `notebook` | `ipykernel`, `notebook` |
| `sca` | `pysca` (only for SCA/Dia pruning) |
| `dev` | `ruff`, `pytest` |

`requirements.lock` pins exact versions (generated with `[plotting,analysis,sca,workflow]`). For a deterministic install: `uv pip sync requirements.lock` then `uv pip install -e . --no-deps`.

### Run manifest schema (v1)

```jsonc
{
  "run_id": "<run dir name>",
  "schema_version": 1,
  "command_line": ["python", "scripts/train_sbm.py", "CM", "..."],
  "code":  {"git_commit": "...", "git_dirty": false, "git_branch": "main"},
  "env":   {"python": "...", "platform": "...", "hostname": "...",
            "omp_num_threads_requested": 8, "package_versions": {"numpy": "...", ...}},
  "inputs":  {"msa": {"path": "...", "sha256": "..."},
              "pruning_mask_couplings": {"path": "...", "sha256": "..."},
              "pruning_mask_fields":    {"path": "...", "sha256": "..."}},
  "options": { /* full options dict; ndarrays summarised as {shape, dtype, sha256} */ },
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

## Citation

If you use this code or data, please cite the associated publication.
