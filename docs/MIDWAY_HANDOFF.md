# Handoff → Midway Claude

The Mac and Midway lines of work were **reunified into one `main`** (a merge commit
combining the Mac-side derive pipeline + improved design engine + ProteinMPNN
removal with your characterization module + design/characterize cluster
orchestration). Figures are now **rendered on the Mac** from the tables you
produce; heavy compute stays on Midway. `docs/RUNBOOK.md` is the end-to-end map.

Pull first: `git pull --ff-only` on Midway (refuse if the tree is dirty).

## What changed that affects you

1. **`scripts/characterize/render_characterize.py` was rewritten** into a thin CLI
   over the new `SBM.utils.utils_characterize_plot` (consolidated, lab-styled
   figures + a tidy `characterization_stats.tsv`). **The CLI is backward-compatible**
   — same `--summary` / `--natural-summary` / `--figs-dir` flags your
   `scripts/characterize/characterize.py` driver already passes, plus a new optional
   `--stats-out`. So the merge step keeps working unchanged; it now just makes the
   nicer figures.
2. **`scripts/sync_models.sh` now excludes `results/*/natural_folds/*/structures/`**
   (the ~28k per-sequence ESMFold PDBs) from both rsync and the SHA256 manifest,
   mirrored so `verify` stays in lock-step. The distilled `fold_scores/*.tsv` still
   sync. This is the fix for the slow rsync. Your first `sync_models.sh push` after
   pulling will regenerate `results/SHA256SUMS` without the PDBs — expected. The PDB
   fold caches stay Midway-side (0-SU to regenerate).
3. The design cluster scripts you wrote (`pipeline/external/*design*.sh`) and the
   characterize scripts (`run_characterize.sh`, the esmfold/tmalign/blast helpers)
   are intact in the merge. Nothing to redo there.

## What's left to finish (the scientific run)

The **profiles** combine run (`combine/combine-profiles/iter-001-profile-eval`) is
fully characterized already — that was the calibration control (both models
fields-only). Its figures are rendered on the Mac and the numbers reproduce your
`report.md` exactly (design median pLDDT 69.7, fold_call A:30/B:35/neither:31,
Spearman(ΔE, ΔTM) = −0.487, both control-sanity checks PASS).

The **coupled** run — `combine/combine-CM-PPIC-potts/iter-001-potts-align-eval` —
has 96 designed sequences (`design/designed.tsv`) but **no `characterize/` yet**.
That is the scientifically interesting one. To finish:

1. **Characterize the coupled designs on Midway** (folds the 96 designs + each
   model's naturals as controls; the natural folds are cached + reused):
   ```sh
   RR=combine/combine-CM-PPIC-potts/iter-001-potts-align-eval
   bash pipeline/external/build_tmalign.sh        # if pipeline/bin/TMalign absent
   bash pipeline/external/prefetch_esmfold.sh     # if the HF cache is cold
   bash pipeline/external/run_characterize.sh "$RR"
   ```
   Consider passing `--skip-render` through to `characterize.py` (figures are the
   Mac's job now — rendering on Midway is harmless but redundant). The Mac renders
   the authoritative figures after the pull.
2. **Push the tables back:** `bash scripts/sync_models.sh push`. Confirm `verify`
   passes (it will, with the new `structures` prune). Only the tables travel — the
   28k design/natural PDBs stay on Midway.
3. On the Mac: `sync_models.sh pull`, then `snakemake -s Snakefile.combine
   --configfile config/params_combine-CM-PPIC-potts.yaml --config run_root=$RR
   --cores 8 all` renders `figs/characterization_{overview,…}.pdf` automatically
   (the `characterize:` stage is already enabled in that config).

## Sanity checks

- `.venv/bin/python -m pytest tests/test_characterize.py tests/test_characterize_plot.py -q`
  (both pure-python; the plot test needs no binaries).
- `report.md`'s control-sanity block must PASS for the coupled run too (CM naturals
  median TM_A > TM_B; PPIC the reverse). If a family's own naturals don't match
  their own reference, the reference/mapping is wrong — stop and flag it.

Questions or a mismatch in the reunified tree → leave a note in
`docs/two_model_progress.md`.
