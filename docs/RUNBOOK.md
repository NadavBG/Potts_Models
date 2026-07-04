# End-to-end runbook: two trained models → designed, characterized sequences

This is the one-page path from **two already-trained single-model runs in
`results/`** to **designed two-model sequences with structure + BLAST
characterization and all figures**. Each step links to the doc that covers it in
depth; this page is the ordering + the Mac-vs-Midway split.

## The compute split (Mac-primary figures, Midway-primary compute)

Heavy compute runs on **Midway**; **all figures are rendered on the Mac** from the
data tables Midway produces (pulled with `scripts/sync_models.sh`). The one thing
written only on Midway is the Slurm orchestration (`pipeline/external/*.sh`).

| Step | Heavy compute (Midway) | Figures / light (Mac) |
|---|---|---|
| 0. (optional) derive | — | `Snakefile.derive` (post-hoc param filter) |
| 1. combine (natural energies + `E_tot` weights) | `potts_align` align cache (large query sets) | scoring, weights, `two_model_energy.pdf`, `energy_weights.pdf` |
| 2. design (generate sequences) | the joint-anneal Slurm array | `design_*` figures |
| 3. characterize (fold + TM + BLAST) | ESMFold (GPU) + TM-align + BLAST + merge | `characterization_*` figures |

`sync_models.sh` moves three trees — `results/` (models, seed MSAs), `combine/`
(query, potts_align cache, scores, design + characterize tables), and
`natural_folds/` (the content-addressed ESMFold cache of the naturals, keyed by
source-FASTA sha8 — a property of the MSA, not of any run). It **excludes** the
bulky regenerable scratch: the `combine/` shard/work dirs and the per-sequence
`natural_folds/*/structures/*.pdb` ESMFold cache (~28k tiny files — only the
distilled `fold_scores/*.tsv` + `tm_vs_refs/*.tsv` travel; the PDBs stay on
Midway, 0-SU to regenerate). See `docs/MODEL_SYNC.md`.

---

## Starting point

Two trained single-model runs, e.g. `results/CM-bm-dense/iter-001-…/` and
`results/PPIC-dense/iter-001-…/`, each with `model.npy` + `inputs/` (the naturals
/ seed MSA). Train them with the single-model `Snakefile` (see `CLAUDE.md`) if you
don't have them yet.

**Step 0 (optional) — derive a parameter-filtered model.** To have a model
contribute only *some* of its parameters (e.g. its fields with couplings zeroed),
run the derive pipeline; it lands a normal `results/` dir the combine step
consumes by `run_dir` (`docs`: `CLAUDE.md` "Derive pipeline").

```bash
# [MAC]
python scripts/iter.py run derive-CM-profile "fields-only" --snakefile Snakefile.derive
```

---

## Step 1 — combine: natural energies + `E_tot` weights

Point a combine config (`config/params_combine-CM-PPIC-potts.yaml`) at the two
`results/` run dirs. The combine pipeline scores each family's naturals (+
synthetics) under **both** models with the couplings-aware `potts_align` aligner,
then derives the `E_tot = w_A·E_A + w_B·E_B` weights post-hoc from the native
medians. For a large query set the `potts_align` alignment cache is pre-built on a
Midway Slurm array (`docs/POTTS_ALIGN.md §11`); a small set needs no cluster step.

```bash
# [MAC] mint the run dir, commit + push code/config to Midway
python scripts/iter.py new combine-CM-PPIC-potts "eval" --snakefile Snakefile.combine
git add -A && git commit -m "combine run config" && bash scripts/sync_models.sh push

# [MIDWAY] build the potts_align cache (large query sets only), then score
#   see docs/POTTS_ALIGN.md §11 for the align→gather array + finalize
bash scripts/sync_models.sh pull        # [MAC] pull the cache back
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=<combine_run> --cores 8 all   # [MAC] score + weights + figs
```

Outputs: `data/scores.tsv`, `data/energy_weights.json`, and the figures
`figs/two_model_energy.pdf` + `figs/energy_weights.pdf`. Full detail:
`docs/POTTS_ALIGN.md`, `docs/two_model_progress.md`.

---

## Step 2 — design: generate two-model sequences

Design is a gated stage of the *same* combine config: add a `design:` block
(`design.enabled: true`) and it anneals `E_tot` from random + CM/PPIC-natural
starts. **The shipped configs default to `design.execution: cluster`** — sequence
generation is a Midway step by policy — so `snakemake … all` on the Mac writes the
run spec + hand-off and stops; you run the Slurm array on Midway, pull back, and
render. (The anneal is Mac-cheap: set `design.execution: local` for a ~2-min local
run, or `auto` to route by predicted wall-time.) Full spec + Mac→Midway runbook:
`docs/DESIGN_TWO_MODEL.md`.

```bash
# [MAC] writes design/design_config.json + MIDWAY_HANDOFF.txt, then stops (cluster default)
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=<combine_run> --cores 8 all
git add -A && git commit -m "design run config" && bash scripts/sync_models.sh push

# [MIDWAY] run the joint-anneal Slurm array
bash pipeline/external/run_design.sh <combine_run>

# [MAC] pull the gathered trajectories back, then render the design figures
bash scripts/sync_models.sh pull
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=<combine_run> --cores 8 <combine_run>/figs/design_alignment.pdf
```

Outputs: `design/designed_sequences.fasta`, `design/designed.tsv`,
`design/trajectories.npz`, and the four `figs/design_*.pdf`.

---

## Step 3 — characterize: fold + which-fold + BLAST (Midway compute → Mac figures)

Predict a structure for every designed sequence (ESMFold, GPU), ask which of the
two reference folds it resembles (TM-align vs 1ECM = fold A / CM and 1JNT = fold B
/ PPIC), and BLAST it. Naturals from each seed MSA are folded once as controls
(cached under the top-level content-addressed `natural_folds/<msa_sha8>/`). **Compute is Midway-only**
(GPU + TM-align + BLAST binaries); the merge lands the tidy tables. Full detail +
knobs + cost: `docs/CHARACTERIZE.md`.

```bash
# [MIDWAY] one-time prep, then the 3 fold arrays + a CPU merge job (afterok)
RR=<combine_run>
bash pipeline/external/build_tmalign.sh      # -> pipeline/bin/TMalign
bash pipeline/external/prefetch_esmfold.sh   # warm the HF cache
bash pipeline/external/run_characterize.sh "$RR"   # writes characterize/data/*.tsv
bash scripts/sync_models.sh push             # send the tables (NOT the 28k PDBs) back-to-Mac side
```

Then render the **Mac-authoritative** figures. Enabling `characterize:` in the
combine config makes `snakemake … all` render them automatically once the tables
are on disk (until then it prints a skip note — the pipeline is never blocked on
un-pulled Midway data). Or target them explicitly:

```bash
# [MAC]
bash scripts/sync_models.sh pull
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml \
    --config run_root=<combine_run> --cores 8 \
    <combine_run>/figs/characterization_overview.pdf
# or directly:
.venv/bin/python scripts/characterize/render_characterize.py \
    --summary <combine_run>/characterize/data/summary.tsv \
    --natural-summary <combine_run>/characterize/data/natural_summary.tsv \
    --figs-dir <combine_run>/figs
```

Outputs: `figs/characterization_overview.pdf` (fold / pLDDT / energy-vs-structure /
BLAST), `figs/tm_A_vs_B.pdf`, `figs/fold_call_breakdown.pdf`, and the tidy
`characterize/data/characterization_stats.tsv` (the numbers the figures cite).

---

## Where each result lands

```
<combine_run>/
  data/scores.tsv, energy_weights.json          # step 1
  design/designed_sequences.fasta, designed.tsv # step 2
  characterize/data/summary.tsv, natural_summary.tsv, characterization_stats.tsv  # step 3
  figs/two_model_energy.pdf, energy_weights.pdf                 # step 1 (Mac)
  figs/design_{trajectories,phase_space,lengths,alignment}.pdf # step 2 (Mac)
  figs/characterization_overview.pdf, tm_A_vs_B.pdf, fold_call_breakdown.pdf  # step 3 (Mac)
```

See also: `CLAUDE.md` (component map), `docs/POTTS_ALIGN.md` (aligner + cluster),
`docs/DESIGN_TWO_MODEL.md` (design engine), `docs/CHARACTERIZE.md` (characterization),
`docs/MODEL_SYNC.md` (Mac↔Midway transfer), `docs/two_model_progress.md`
(project status + design decisions).
