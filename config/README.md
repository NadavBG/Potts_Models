# Pipeline configuration (`config/*.yaml`)

One YAML file describes **one pipeline run end-to-end**: input MSA → optional
pruning masks → training → synthetic sampling → figures. The `Snakefile` loads
it, `src/SBM/workflow_config.py` validates it, and
the thin `scripts/wf/run_*.py` wrappers translate each section into a call to
the underlying CLI.

```bash
# mint a fresh iteration dir and run everything
python scripts/iter.py run <run_name> "<tag>"

# or drive Snakemake directly
snakemake --configfile config/params_<run_name>.yaml --cores 8 all
```

**Three kinds of config live here.** Single-model runs (`params_<family>-*.yaml`)
train one model and are validated by `src/SBM/workflow_config.py` against the
main `Snakefile`. **Combine** runs (`params_combine-*.yaml`) score a query set
under *two already-trained* models and are validated by
`src/SBM/combine_config.py` against `Snakefile.combine`:

```bash
python scripts/iter.py run combine-CM-PPIC "baseline" --snakefile Snakefile.combine
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC.yaml --cores 8 all
```

The combine schema is documented in `combine_config.py`; keys are `models` (a
list of exactly two `{name, run_dir}`), `query` (`source` / `include` /
`cap_per_group`), `scoring` (`method` / `n_samples` / `ess_threshold`), and
`figures`.

**Derive** runs (`params_derive-*.yaml`) take *one already-trained* model and
write a new model that keeps only a subset of its parameters (e.g. fields only,
couplings zeroed) — no retraining. They are validated by
`src/SBM/derive_config.py` against `Snakefile.derive` and land a normal
`results/` dir the combine pipeline consumes by `run_dir`:

```bash
python scripts/iter.py run derive-CM-profile "fields-only" --snakefile Snakefile.derive
snakemake -s Snakefile.derive --configfile config/params_derive-CM-profile.yaml --cores 8 all
```

The derive schema is documented in `derive_config.py`; keys are `source_run_dir`
(the trained model to filter) and `filter` (`couplings` and `fields`, each
`keep` / `zero` / a `{strategy, percent}` mask), plus `sample` / `figures`.
Post-hoc filtering keeps the source's already-fit `h` and zeros/masks `J` — a
different energy function from the retrain-based `params_<family>-*-profile`
configs, which re-fit `h` with `J≡0`. The rest of this doc covers the
**single-model** schema.

**How to read this doc.** Defaults below are the *schema* defaults from
`workflow_config.py` (what you get if you omit the key). The values in the
shipped configs (`params_CM-bm-dense.yaml`, `params_tiny.yaml`) sometimes
differ — those are the BM positive-control regime, not the schema default.

**Validation is strict.** `from_dict` **rejects unknown keys** (a typo'd key is
an error, not a silent no-op) and enforces the ranges/enums noted below. The
exact validated config is round-tripped into each run's
`config_snapshot.yaml`, so a run always carries the parameters that produced it.

A few enum vocabularies are fixed in `workflow_config.py`:

| Enum | Allowed values |
| --- | --- |
| coupling pruning `strategy` | `fij`, `cij`, `sca` |
| field pruning `strategy` | `fia`, `dia` |
| `Dia_prior` | `gap-corrected`, `uniform` |
| `sector` | `emily`, `rama`, `none` |
| `train.mode` | `BM`, `SBM` |
| `train.optimizer` | `LBFGS`, `GD` |

---

## Top-level keys

| Key | Default | Required | Description / notes |
| --- | --- | --- | --- |
| `run_name` | — | **yes** | Identifier for the run; also the default results subdir (`results/<run_name>/`). Must be non-empty. |
| `msa_fasta` | — | **yes** | Path (relative to the repo root) to the input MSA as an **aligned FASTA**. The `encode_msa` rule converts it once into a run-local integer array `<run_dir>/inputs/msa.npy` (alphabet `-ACDEFGHIKLMNPQRSTVWY`, gap=0, dtype int64), and every downstream stage consumes that array. The FASTA is treated as immutable raw input; the `.npy` is a derived, regenerable artifact. See "MSA input" below. |
| `description` | `""` | no | Free-text label carried into provenance. |
| `family` | `""` | no | Protein family tag (e.g. `CM`); used only for organizing/labelling. |
| `seed` | `42` | no | **Master seed.** Seeds the Python RNG and the C++ MCMC kernels (per-thread seed = `seed + thread_id`). The `sample` rule derives a per-temperature seed `seed + temperature_index`. |
| `omp_num_threads` | `null` | no | OpenMP thread count, pinned **before** the MCMC kernel imports. **`null` ⇒ runs are *not* bit-reproducible** even with a fixed `seed` (thread count varies). Set to a fixed int (e.g. `8`) for bit-identical arrays. See "Reproducibility" below. |

Then five nested sections: `msa_stats`, `pruning`, `train`, `sample`, and
`figures`. Each may be omitted entirely (the schema default applies).

### MSA input

The pipeline starts from an **aligned FASTA** (`msa_fasta`), not a pre-encoded
array. The `encode_msa` rule runs first and writes:

- `<run_dir>/inputs/msa.npy` — the integer-encoded alignment consumed by
  `msa_stats`, the pruning mask builders, and `train`.
- `<run_dir>/inputs/msa_manifest.json` — provenance: the input FASTA's sha256,
  the output shape/hash, and the count + record IDs of any sequences dropped.

Two things to know:

- **Sequences with non-canonical residues are dropped.** Any record containing
  a character outside `-ACDEFGHIKLMNPQRSTVWY` (lowercase, `BJOUXZ`, etc.) is
  removed. The count and the dropped record IDs are logged at WARNING and
  recorded in `msa_manifest.json`, so the drop is never silent — but the
  encoded MSA may have fewer rows than the FASTA has records. (For the shipped
  `data/fasta/CM.fasta`: 1259 records → 1258 kept, 1 dropped.)
- **It must be a true alignment.** All records must have equal length; a ragged
  FASTA is a hard error, not a truncated array.

---

## `msa_stats` — the MSA-only figure

Computes the data-side statistics figure (`<run_dir>/msa_stats.pdf`:
`f_iᵃ`, `D_iᵃ`, `‖f_ij‖_F`, SCA heatmaps). **Independent of training** — it
reads only the MSA, so it can be rendered without any model
(`snakemake ... msa_stats_only`).

| Key | Default | Description / notes |
| --- | --- | --- |
| `enabled` | `true` | Set `false` to skip the MSA figure entirely. |
| `theta` | `0.7` | Sequence-reweighting threshold (fractional Hamming distance): near-identical sequences are clustered so each cluster contributes ~1 effective sequence. |
| `lbda` | `0.03` | Pseudocount for frequency/SCA estimation (additive smoothing toward the background). `0` = raw empirical frequencies. |
| `Dia_prior` | `gap-corrected` | Background distribution for the `D_iᵃ` (KL-divergence) panel. `gap-corrected` uses the alignment's empirical gap rate + standard 20-AA frequencies; `uniform` uses `1/21` per state. |
| `sector` | `emily` | Which CM sector to annotate. `emily` and `rama` are two nearly-identical hand-curated residue sets for chorismate mutase (defined in `CM_sector.py`); `none` draws no sector strip. **Only meaningful for the CM family (L=96)** — use `none` for other proteins. |

---

## `pruning` — coupling (J) and field (h) masks

Builds binary masks that zero out selected `J`/`h` parameters during training
(the gradient is multiplied by the mask each step). Masks are ranked by a
per-strategy importance score; the strongest entries are kept.

> **If `enabled: false`, the entire section is ignored** — no `build_mask_*`
> rules run and `couplings`/`fields`/`theta`/`lbda`/etc. have no effect. You can
> leave them in the file as documentation.

| Key | Default | Description / notes |
| --- | --- | --- |
| `enabled` | `false` | Master switch. When `true`, **at least one of `couplings`/`fields` must be set** (validated). |
| `theta` | `0.7` | Reweighting threshold for mask construction, applied with **identity semantics**: the code passes `1 − theta` as the distance threshold (`build_mask.py:286`), so `0.7` clusters sequences sharing ≥70% identity. (Contrast `train.theta`, which is a *distance* threshold passed directly — the two sections express the same quantity differently.) Always uses the gaps-included weighting path. |
| `lbda` | `0.03` | Pseudocount for the frequency/correlation/SCA matrices the strategies rank on. |
| `label` | `"CM"` | String embedded in the auto-generated mask filename and manifest. Provenance only — no effect on mask values. |
| `Dia_prior` | `gap-corrected` | Background for the `dia` field strategy (same meaning as in `msa_stats`). Irrelevant unless `fields.strategy: dia`. |
| `couplings` | `null` | `{ strategy: <fij\|cij\|sca>, percent: <0–100> }` or omit/`null` to leave J unpruned. |
| `fields` | `null` | `{ strategy: <fia\|dia>, percent: <0–100> }` or omit/`null` to leave h unpruned. |

**Coupling strategies** (`couplings.strategy`) — rank pairs `(i,j)`:
- `fij` — raw second-order frequencies `f_ij(a,b)`.
- `cij` — covariance `f_ij − f_i·f_j`.
- `sca` — Statistical Coupling Analysis matrix (weighted, SCA-positional). Needs `pysca` (the `[sca]` optional-dependency group); `fij`/`cij` do not.

**Field strategies** (`fields.strategy`) — rank positions `(i,a)`:
- `fia` — first-order frequencies `f_i(a)`.
- `dia` — KL divergence `D_i(a)` of `f_i(a)` from `Dia_prior` (conservation/information content).

> **`percent` is the percentage *pruned* (removed), not kept.** The mask keeps
> the top-magnitude `(100 − percent)%` of parameters and zeros the rest
> (`tokeep = total × (1 − percent/100)`, `build_mask.py:155,186`). So
> `percent: 98.0` with `strategy: sca` keeps **the strongest 2%** of couplings
> by SCA magnitude and prunes the other 98%. Higher `percent` ⇒ sparser model.

---

## `train` — the L-BFGS statistics-matching fit

Both regimes share the L-BFGS algorithm and differ only in parameter values
(Summary Note 3). `N_iter`, zero-initialized parameters, and inference
temperature `T=1` are fixed; the schema **warns** (does not fail) if `mode` and
the knobs disagree.

| Parameter | BM (positive control) | SBM (stochastic regularization) |
| --- | --- | --- |
| `m` (L-BFGS memory) | 20 | 1 |
| `lambda_J`, `lambda_h` | 0.01 | 0 |
| `N_chains` | 100 | 50 |

| Key | Default (schema) | Description / notes |
| --- | --- | --- |
| `mode` | `SBM` | `BM` or `SBM`. Selects the *intended regime* only — the actual behavior comes from the knobs below. Mismatch (e.g. `BM` with `m=1`) logs a WARN. |
| `optimizer` | `LBFGS` | `LBFGS` (used by both BM and SBM) or `GD` (vanilla gradient descent; rarely needed — uses `alpha`/`Learning_rate`). |
| `N_iter` | `400` | Number of L-BFGS iterations (outer loop). **Standard at 400** — convergence is usually reached by ~300–500. Think before changing. |
| `N_chains` | `50` | MCMC chains used to estimate model statistics each gradient step. More chains = lower-variance gradient, slower step. BM=100, SBM=50. |
| `m` | `1` | L-BFGS memory rank. Higher = better curvature approximation, more memory. BM=20, SBM=1. |
| `lambda_J` | `0.0` | L2 penalty on couplings: `gradJ += 2·lambda_J·J`. BM=0.01, SBM=0. |
| `lambda_h` | `0.0` | L2 penalty on fields: `gradh += 2·lambda_h·h`. BM=0.01, SBM=0. |
| `theta` | `0.3` | Sequence-reweighting threshold (fractional Hamming distance): sequences closer than `theta` are clustered so each cluster contributes ~1 effective sequence. Lower `theta` ⇒ stricter clustering ⇒ larger effective count. |
| `k_MCMC` | `100000` | Metropolis sweeps per chain per gradient step (the `tburn` argument to the C++ kernel). Governs how well each artificial alignment is equilibrated. **Standard at 1e5** — lowering it speeds the run but biases the gradient. |
| `TestTrain` | `0` | `0` = train on all sequences; `1` = hold out a random 20% test split (recorded in `output["Test"]`, used only for evaluation/figures, never for optimization). **Required `1` to render the `energy`/`similarity`/`diversity`/`length` figures.** |
| `record_every` | `5` | Record the coupling Frobenius norm (`J_norm`) every N iterations (feeds the `coupling_evol` figure). Cosmetic/diagnostic only. |
| `ignore_gaps` | `false` | How gaps enter the reweighting distance. `false` ⇒ gaps count as a 21st state (`pdist` Hamming over all L columns); `true` ⇒ gaps are excluded (distance over non-gap columns only). Either way the threshold is `theta`. |

Inference temperature is **always `T=1`** during training (the model is meant to
reproduce the data statistics at T=1); it is not configurable here. Sampling
temperatures are a separate downstream step — see `sample`.

---

## `sample` — synthetic alignments from the trained model

Writes one `synthetic/align_T<temp>.npy` per temperature (plus a JSON sidecar).

| Key | Default | Description / notes |
| --- | --- | --- |
| `N` | `2000` | Sequences sampled **per temperature** (not a total split across temperatures). With `temperatures: [0.75, 1.0]` and `N: 2000` you get 2000 sequences at each T. |
| `temperatures` | `[0.75, 1.0]` | Sampling temperatures (all `> 0`, must be unique). The two-T default is deliberate: **T=0.75** (low-T, mode-collapsed) vs **T=1.0** (the model's native fit). Every multi-temperature figure compares them, so changing this changes what the figures show. |

---

## `figures` — which figures to render

| Key | Default | Description / notes |
| --- | --- | --- |
| `which` | `null` | `null` = render every figure whose required data is present (auto-skip the rest). A list (e.g. `[coupling_evol, params]`) renders exactly that subset — and **errors** if a requested figure's data is missing. |
| `sector` | `emily` | Sector strip drawn on the `params` figure (`emily`/`rama`/`none`; same as `msa_stats.sector`, CM-only). |

Figure names and their data requirements:

| Figure | Requires | Notes |
| --- | --- | --- |
| `coupling_evol` | `model.npy` only | Always renderable. `J_norm` trajectory. |
| `params` | `model.npy` only | Always renderable. `h`/`J` heatmaps + sector strip. |
| `correlations` | ≥1 synthetic alignment | One figure, rows = temperatures × cols = 1st/2nd/3rd-order stats. |
| `pca` | ≥1 synthetic alignment | 1×(1+N_temps) PCA panels. |
| `energy` | `TestTrain: 1` | Energy histograms; overlays every available temperature. |
| `similarity` | `TestTrain: 1` | Sequence-similarity violins. |
| `diversity` | `TestTrain: 1` | Sequence-diversity violins. |
| `length` | `TestTrain: 1` | Sequence-length histograms. |

---

## Reproducibility

Bit-identical re-runs require **all three** of: a fixed `seed`, a fixed
`omp_num_threads`, and a fixed `train.N_chains`. The shipped configs leave
`omp_num_threads: null`, so by default the arrays inside `model.npy` are *not*
guaranteed bit-identical across machines/runs (the manifest records what was
actually used). Note that `model.npy` *bytes* are never identical even when the
arrays are — it stores wall-clock execution times. Compare arrays, not pickles.

## See also

- `src/SBM/workflow_config.py` — the validated schema (single source of truth).
- `CLAUDE.md` (repo root) — pipeline overview, data flow, gotchas.
- `pruning/README.md` — pruning strategies in depth.
- `config/params_tiny.yaml` — minimal end-to-end smoke-test config.
- `config/params_CM-bm-dense.yaml` — the CM BM positive-control worked example (no pruning; the `params_CM-bm-*` variants add coupling/field pruning).
