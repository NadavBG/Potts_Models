# ProteinMPNN foldability sweep

A computational proxy for "does a sampled sequence look like it folds?"
using ProteinMPNN's conditional log-likelihood given a fixed backbone
(here, the 1ECM crystal structure for chorismate mutase). Lower negative
log-likelihood per residue means MPNN considers the sequence more
designable for that fold.

## Setup

ProteinMPNN is not pip-installable. Clone the upstream repo next to
this one and point the sweep script at it:

```bash
# From the parent of this repo:
git clone https://github.com/dauparas/ProteinMPNN.git
export PROTEINMPNN_PATH="$(realpath ./ProteinMPNN)"
```

Tell SBM where the clone lives via `PROTEINMPNN_PATH` (above) or
`--mpnn-path` on each invocation.

### Torch lives in a separate env (recommended)

The Potts_Models venv intentionally does not depend on torch. Scoring
runs in a subprocess; point that subprocess at a Python interpreter in
a dedicated env (conda, mamba, uv, …) so torch can be pinned
independently of the C++/OpenMP build env. Use `--mpnn-python` or
`PROTEINMPNN_PYTHON` to select the interpreter:

```bash
# Option 1: conda (matches upstream ProteinMPNN's docs)
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn
# Pick the torch build that matches your hardware. CPU is always safe;
# CUDA users follow https://pytorch.org/get-started/locally/.
pip install "torch>=2.2" "numpy<2.3"
conda deactivate

export PROTEINMPNN_PYTHON="$(conda run -n proteinmpnn which python)"
# Pin the env for reproducibility:
conda env export -n proteinmpnn --from-history > "$PROTEINMPNN_PATH/environment.yml"
```

```bash
# Option 2: uv venv next to the ProteinMPNN clone
cd "$PROTEINMPNN_PATH"
uv venv --python=3.12 .venv
uv pip install --python .venv/bin/python "torch>=2.2" "numpy<2.3"
uv pip freeze --python .venv/bin/python > requirements.lock
cd -
export PROTEINMPNN_PYTHON="$PROTEINMPNN_PATH/.venv/bin/python"
```

Both options produce a manifest you can check into the ProteinMPNN
clone (`environment.yml` or `requirements.lock`) so the env can be
recreated. The Potts_Models lock file (`requirements.lock` at the root
of this repo) stays clean.

If `--mpnn-python` / `PROTEINMPNN_PYTHON` are not set, the subprocess
falls back to `sys.executable` — i.e. the env that ran
`scripts/sample_sbm.py`. That env then needs torch installed.

The `data/structures/1ECM.pdb` file is checked into this repo; its
sha256 is recorded in `data/structures/README.md` and re-verified on
load.

## Run

```bash
# Defaults: 10 temperatures (0.1, 0.2, …, 1.0) × 100 sequences,
# plus WT, uniform-random, shuffled-WT, natural-MSA-bootstrap controls.
bash scripts/sample_sbm.sh results/CM/<RUN_DIR> --mpnn-sweep

# Render the figure (auto-detected by render_figures.py):
bash scripts/render_sbm.sh results/CM/<RUN_DIR> --figs mpnn
```

A smoke run that exercises sampling and the figure pipeline without
needing ProteinMPNN installed:

```bash
bash scripts/sample_sbm.sh results/CM/<RUN_DIR> --mpnn-sweep \
    --mpnn-temperatures 0.5 1.0 --mpnn-N-per-T 5 \
    --mpnn-controls wt random --mpnn-skip-scoring
bash scripts/render_sbm.sh results/CM/<RUN_DIR> --figs mpnn
```

## Outputs

For master-seed `S` from the model's training manifest:

```
<RUN_DIR>/synthetic/mpnn_sweep_seed<S>/
  align_T<T>_seed<S+i>.npy            per-T sampled MSA, schema-
  align_T<T>_seed<S+i>.json           ↳ compatible with sample_sbm.py
  control_wt.npy                      (1, L) WT MSA row
  control_random_seed<S+1000>.npy
  control_shuffled_wt_seed<S+1001>.npy
  control_natural_seed<S+1002>.npy
  mpnn_scores.json                    score table + bench (figure input)
  manifest.json                       full sweep provenance
  bench.json                          per-T sampling + scoring wall times

<RUN_DIR>/figs/mpnn.pdf                violins per T + controls + WT line
```

Per-T sample seeds use the existing `seed + i` convention from
`sample_sbm.py`; control seeds use fixed offsets (`+1000` random,
`+1001` shuffled, `+1002` natural) so they never collide with a
sample-seed range, even if the temperature ladder grows.

## Score interpretation

The y-axis of `mpnn.pdf` is **mean negative log-likelihood per scored
residue**. Lower = more native-like for that backbone.

* **WT (1ECM)** — horizontal reference line. The natural-design anchor.
* **Natural MSA bootstrap** — should be close to WT (the model is
  trained on this distribution). A large gap means the structure does
  not represent the family well.
* **Shuffled WT** — composition-only control. Score should be much
  worse than WT; if not, the model isn't using sequence identity.
* **Uniform random** — the floor; score worse than shuffled-WT.
* **Sampled T** — should track WT at low T (model concentrates on
  designable sequences) and degrade toward random as T grows.

## MSA ↔ PDB mapping

The PDB chain may be a truncation of the WT (1ECM is missing the WT's
first three residues, `TSE`). The map is built once per sweep by
pairwise-aligning the PDB chain sequence to the WT MSA row; only
matched residues are scored. Sampled non-gap residues at WT-gap MSA
columns are silently dropped from scoring (the PDB has no Cα to score
against); the per-row count is recorded in
`mpnn_scores.json["groups"][...]["extra_residue_counts"]` and
flagged on the figure when the median is non-zero.

## Model variant

Default: `v_48_020` (vanilla soluble model with 0.20 Å backbone noise).
Override with `--mpnn-model-name`. The weights file's sha256 and the
upstream repo's git commit are recorded in `manifest.json["extra"]["mpnn"]`.

## Benchmarks

Reference numbers from a representative run on a MacBook Air (Apple
Silicon, `device=cpu`), default sweep settings, taken from
`results/CM/2026-05-06_CM_0/synthetic/mpnn_sweep_seed42/bench.json`:

| Stage | Wall time | Notes |
|-------|-----------|-------|
| Sampling (10 T × 100 seqs) | 3.4 s | ~3.4 ms/seq; depends on `delta_t` (default = `options0.k_MCMC`) |
| ProteinMPNN scoring (1301 seqs) | 119 s | ~92 ms/seq; 14 subprocess calls × ~0.9 s torch startup ≈ 13 s overhead |
| Total sweep | 123 s | sampling + scoring + I/O |

The 1301 scored sequences = 10 × 100 sampled + 100 random + 100
shuffled-WT + 100 natural-MSA + 1 WT. Re-run on new hardware and read
`bench.json` to replace these numbers. Steady-state per-residue
scoring is ~80 ms on this hardware; expect ~50–200 ms/seq on Apple
Silicon CPU or GPU/MPS, and 5–10× slower on older CPUs or for larger
`L`.

## Caveats

* The score sign convention follows upstream: NLL/residue, lower is
  better. We never normalize across T; each sequence is scored
  independently.
* Scores are **not** comparable across different `--mpnn-model-name`
  values or different PDBs. Stick to one combination per analysis.
* Sweep dir reuse: re-running with the same master seed without
  `--force` refuses to clobber existing artifacts. To regenerate, pass
  `--force` (mirrors `sample_sbm.py` standard-mode behavior).
