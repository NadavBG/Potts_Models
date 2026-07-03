# Structural + BLAST characterization of designed sequences

Downstream QC for the two-model **design** outputs: predict a 3-D structure for
each designed sequence, ask **which of the two reference folds it resembles**
(TM-score + RMSD vs 1ECM chain A = fold **A** / chorismate mutase, and 1JNT
chain A = fold **B** / PPIC / parvulin), and ask **what it looks like in
sequence space** (BLAST vs SwissProt + the CM and PPIC families, kept as
separate columns). Natural sequences from each model's seed MSA are folded too,
as calibration controls and as a reusable per-MSA reference set.

This replaces the ProteinMPNN foldability proxy (removed — it was uninformative)
with real single-sequence structure prediction.

## Predictor + environment (important)

- **ESMFold** (`facebook/esmfold_v1`, single-sequence) — the correct predictor
  for de novo designs, where an MSA is ill-defined. Confidence = mean **pLDDT**
  (0–100, read from the output PDB B-factor) + **pTM**.
- **Interpreter = the RCC `AI` env**
  (`/software/python-miniforge-25.3.0-el8-x86_64/envs/AI/bin/python`; torch
  cu128 + transformers 5.6.2 + `EsmForProteinFolding`). We do **not** use
  `bioM3_env` — its interpreter **segfaults on startup on Midway** (even
  `python --version` cores). Override with `ESMFOLD_PYTHON` if you have a better
  env. Our code is self-contained (uses the `transformers` package, no BioM3
  imports; the short-seq blastp recipe is copied from BioM3's `blast_qc.py`).
- **TM-align** built once from the Zhang-lab source; **BLAST+** from `CM_env`
  (`blastp` v2.17). Structure prediction is GPU (`beagle3` A100); TM-align +
  BLAST + merge + figures are CPU (`caslake`).

## One-command flow (Midway login node)

```sh
RR=combine/combine-profiles/iter-001-profile-eval    # a built design run dir

# 0. One-time prep (login node — needs outbound network):
bash pipeline/external/build_tmalign.sh       # -> pipeline/bin/TMalign
bash pipeline/external/prefetch_esmfold.sh    # warm HF cache (~2.5 GB)

# 1. Probe: measure ESMFold s/seq + the real SU charge before committing 28k.
bash pipeline/external/run_esmfold_probe.sh $RR 20
#    read: grep FOLD_TIMING $RR/characterize/logs/esmfold_shard_<jid>_0.log
#          accounts balance   # vs the BEFORE snapshot the script printed

# 2. Full run: 3 GPU fold arrays (designs + CM + PPIC naturals) + a caslake
#    CPU merge job (afterok) that TM-aligns, BLASTs, summarizes, and renders.
bash pipeline/external/run_characterize.sh $RR
#    monitor: squeue -u $USER   |   pipeline/job_tally.sh -w 10
```

Everything (TM-align + BLAST + merge + figures) can also run on the Mac once the
structures are pulled back; only the GPU fold needs Midway.

### Knobs (`run_characterize.sh` env vars)

| var | default | meaning |
|---|---|---|
| `ESMFOLD_MAX_CONCURRENT` | 8 | GPUs used at once per fold array (beagle3 cap 32) |
| `N_SHARDS_DESIGN` | 1 | shards for the 96 designs |
| `N_SHARDS_CM` | 4 | shards for the ~1.3k CM naturals |
| `N_SHARDS_PPIC` | 64 | shards for the ~27k PPIC naturals |

Fold shards are **resume-safe**: a record whose `<id>.pdb` exists and is already
in the shard scores TSV is skipped, so a TIME_LIMIT kill leaves a valid partial
cache and a re-submit continues.

## Outputs

```
<run_dir>/characterize/
  structures/<id>.pdb                 ESMFold designs
  structures/fold_scores/shard_*.tsv  per-shard id,group,length,plddt_mean,ptm
  data/summary.tsv                    one tidy row per design (below)
  data/natural_summary.tsv            fold+TM for the naturals (no BLAST)
  data/structure_compare.tsv          full TMalign detail (both norms, RMSD, ...)
  data/fold_scores.tsv                gathered design fold scores
  data/blast/blast_{swissprot,cmfam,ppicfam}.tsv   raw -outfmt 6 (kept separate)
  data/blast/blast_report.txt         per-design best hit per DB
  data/characterization_stats.tsv     tidy (group, metric, value) — the numbers the figures cite (Mac)
  figs/characterization_overview.pdf  consolidated 2x2: fold / pLDDT / energy-vs-structure / BLAST (Mac)
  figs/tm_A_vs_B.pdf                   standalone "which fold?" scatter (Mac)
  figs/fold_call_breakdown.pdf         fold-call composition per group (Mac)
  report.md                           human summary + control-sanity checks
  provenance/manifest.json

natural_folds/<msa_sha8>/                          content-addressed per-MSA cache (own top-level tree)
  structures/<id>.pdb + fold_scores/shard_*.tsv    (folded once per input MSA; PDBs Midway-only)
  tm_vs_refs/<refkey>.tsv + <refkey>.meta.json     (TM-aligned once per ref pair)
```

**Natural TM-align is cached, not recomputed.** The naturals dominate the
TMalign cost (~28k CM+PPIC vs 96 designs), yet each natural's TM-score is a pure
function of (its ESMFold PDB, the two reference folds) — no design dependency.
So `characterize.py` TM-aligns the designs fresh every run but reads the naturals
from a content-addressed cache at the top-level `natural_folds/<sha8>/tm_vs_refs/<refkey>.tsv`
(`<sha8>` = the source-FASTA sha8 — the fold/TM is a property of the MSA, not of any
run, so it is its own tree beside `results/` and `combine/`), keyed by
`<refkey>` = a hash of the two references' content-sha + chain. A cache **miss**
(no file, a partial file missing ids, or `--force-natural-tm`) TM-aligns and
writes the cache + a `.meta.json` provenance sidecar; a **hit** reuses it. Because
`<refkey>` changes iff a reference PDB/chain changes, a stale cache is impossible.
`structure_compare.tsv` is the union of the fresh design table and the cached
natural tables (the downstream merge is unchanged). `tm_vs_refs/` is a distinct
name from `structures/`, so `sync_models.sh` mirrors the small TSVs to the Mac
while the bulky PDBs stay Midway-side. The manifest records the `refkey` and the
per-family hit/miss.

`summary.tsv` columns: `sequence_id, group, length, plddt_mean, plddt_class,
ptm, tm_A, rmsd_A, tm_B, rmsd_B, delta_tm, fold_call, E_A, E_B, E_tot, delta_E,
start_type,` then the **separated** BLAST columns
`swissprot_top_hit/pident/evalue/annotation, cmfam_top_hit/pident,
ppicfam_top_hit/pident`.

- `tm_A`/`tm_B` are TM-scores **normalized by the reference** (both refs ~91–92
  res, so directly comparable). `fold_call ∈ {A, B, ambiguous, neither, na}`
  from the TM≥0.5 rule (Xu & Zhang 2010); `ambiguous` = both ≥0.5 within 0.05.
- pLDDT answers "does it fold at all"; TM answers "which of A/B".
- **Caveat**: 1ECM chain A is one arm of a domain-swapped dimer, so an
  isolated-monomer `tm_A` is a lower bound — flagged in `report.md`.

## Rendering on the Mac (authoritative figures)

The compute above (fold + TM + BLAST + merge) runs on Midway and lands the tidy
tables under `characterize/data/`. **The figures are made on the Mac** from those
tables — pure numpy/matplotlib, no binaries — via
`src/SBM/utils/utils_characterize_plot.py` (recipes) + `scripts/characterize/render_characterize.py`
(thin CLI), routed through `scripts/lab_plotting.py` and laid out from inch budgets.
It is wired into `Snakefile.combine` as the gated **`characterize_render`** stage
(`characterize.enabled` in the combine config): once `characterize/data/summary.tsv`
has been pulled from Midway, `snakemake … all` renders the three PDFs +
`characterization_stats.tsv` automatically; until then it is skipped from `all`
with a note (the pipeline is never blocked on un-pulled data). Target a fig
explicitly to force it (fails loudly if the table is still missing). Direct CLI:

```sh
.venv/bin/python scripts/characterize/render_characterize.py \
    --summary <run_dir>/characterize/data/summary.tsv \
    --natural-summary <run_dir>/characterize/data/natural_summary.tsv \
    --figs-dir <run_dir>/figs
```

The CLI is backward-compatible with the Midway `characterize.py` merge driver, so
that driver still renders if you let it — but the intended flow is `--skip-render`
on Midway and render on the Mac after the pull. `sync_models.sh` moves the tables
but **excludes** the bulky `natural_folds/*/structures/*.pdb` fold cache
(~28k tiny files; only the distilled `fold_scores/*.tsv` and `tm_vs_refs/*.tsv`
travel) — see `docs/MODEL_SYNC.md`.

## Correctness check (built in)

The naturals are the positive controls: `report.md` asserts median CM-natural
`tm_A > tm_B` and median PPIC-natural `tm_B > tm_A`. If a family's own naturals
don't match their own reference, the pipeline (or the reference/mapping) is
wrong. Unit tests (`tests/test_characterize.py`, pure-python) cover the TMalign
parser, the round-robin sharder, the BLAST parser, degap, and the summary merge:

```sh
.venv/bin/python -m pytest tests/test_characterize.py -q
```

## Cost (measured, 2026-07-03 probe)

- **SU = 0.** A 20-design probe (job 51397823, A100-PCIE-40GB, 5:19 wall) left
  `accounts balance` for `pi-ranganathanr` unchanged at 46,433 before *and*
  after — beagle3 GPU time does not deduct from the allocation. The 50-SU budget
  is a non-issue.
- **Throughput ≈ 9.65 s/seq** on an A100-PCIE-40GB (plus a one-time ~1.5 min
  model load per shard). Slower than ESMFold's optimized path because the AI
  env's transformers uses pure-PyTorch attention (its openfold "cpp extensions"
  want a torch other than the installed cu128) — at 0 SU this only affects
  wall-clock.
- Full fold of the ~28,140 designs+naturals ≈ **75 GPU-hours** total →
  ~9 h wall at `ESMFOLD_MAX_CONCURRENT=8`, ~4.5 h at 16, ~2.5 h at 32. One-time
  and reusable (per-MSA cache). The 96 designs alone are minutes.
```

<!-- Probe: 9.65 s/seq, 0 SU (beagle3 free), A100-PCIE-40GB — 2026-07-03. -->

