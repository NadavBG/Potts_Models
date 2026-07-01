# Two-model energy: progress & next steps

**Status (2026-07):** the alignment problem is solved (production aligner =
`potts_align`, **no DCAlign**); the next work is defining and using `E_tot`.

This doc is the running progress note for scoring a sequence under **two** fitted
Potts models. It points to the reference material and records where we are and
what comes next. It replaces the old `initiate_two_model_energy.md` spec/decision
record — that full spec (§0–§9) and the entire DCAlign campaign (§10.1–§10.20)
live in git history and, for the DCAlign code/artifacts, under `.archive/`.

## The goal

Given one amino-acid sequence and two separately-fit Potts models `A`, `B` of
different aligned lengths `L_A ≠ L_B`, compute

- `E_A(seq)`, `E_B(seq)` — the sequence's statistical energy under each model, and
- `E_tot(seq) = w_A·E_A + w_B·E_B` — the combined energy (weights explicit,
  default 1.0).

The point of the larger project is **design**: find sequences that are
simultaneously low-energy (model-typical) under multiple models.

## The one hard part — alignment — is solved

A raw sequence is **not in either model's frame** (different aligned lengths,
non-homologous), so it must be aligned to each model independently before the
Potts energy is defined. How to collapse that latent alignment was the whole
substance of this project.

**The production answer is `potts_align`** — a couplings-aware aligner that finds
the frame minimizing the exact in-frame Potts energy over gap placements (exact
enumeration when few gaps, parallel tempering otherwise). Full spec, cost model,
and the honest comparison to DCAlign: **`docs/POTTS_ALIGN.md`**. It is pure numpy,
runs on the Mac, and for a home pair provably returns the global insert-free
minimum (§5 there). Validated end-to-end in the combine pipeline at
`combine/combine-CM-PPIC-potts/iter-001-potts-align-eval`.

**DCAlign is retired.** The iter-003 campaign chased a "worse-than-native"
residual through DCAlign's insertion prior, gap penalties, `pcount`, multi-seed,
and annealing — all refuted — and established that the residual was a **BP search
failure**, not a prior/objective bias. Directly minimizing the Potts energy (what
`potts_align` does) recovers native on every worse pair (16/16; 11 provably
global) with no DCAlign at all, and a confirmatory batch showed warm-starting
DCAlign-BP at the exact frame only *loses* ground. So DCAlign is not just
unnecessary, it is worse. All DCAlign code, configs, and run outputs are under
`.archive/` (out of git and out of `sync_models.sh`); the decision trail is in
this file's git history (former §10.8–§10.20).

## Current state (what's built)

- **Scoring methods** (`SBM.energy.score.score_sequence`, `method=`): `in_frame`
  (exact base case), `map` (fields-Viterbi), `marginal` (IS free energy + ESS),
  and **`potts_align`** (the production couplings-aware aligner). `auto` at the
  CLI/pipeline layer picks in-frame-or-marginal (and warns that it breaks A/B
  comparability).
- **Combine pipeline** — `Snakefile.combine` + validated `src/SBM/combine_config.py`,
  driven by `config/params_combine-*.yaml`. Consumes two trained models, scores a
  query set under both, writes `data/scores.tsv` (tidy), `data/alignments.txt`
  (human-readable), and `figs/two_model_energy.pdf` (the `E_A`-vs-`E_B` scatter).
  Run: `python scripts/iter.py run <name> "<tag>" --snakefile Snakefile.combine`.
  Running `potts_align` locally and at cluster scale: `docs/POTTS_ALIGN.md` §11.
- **CLI** — `scripts/score_two_models.py` (single seq or FASTA → `E_A`, `E_B`,
  `E_tot` + diagnostics).
- **Tests** — `tests/test_energy.py` (gauge, base case, MAP≈marginal, ordering,
  IS diagnostics + the brute-force DP anchor for the HMM), `tests/test_potts_align.py`,
  `tests/test_potts_align_baseline.py`.

## Conventions (fixed, so this doc stands alone)

- **Energy:** `E(S) = −Σ_i h_i(S_i) − Σ_{i<j} J_ij(S_i,S_j)`; `P(S) ∝ e^{−E}`.
  **Lower = more model-typical.**
- **Gauge:** both models are loaded in the **zero-sum gauge** (`load_model`
  re-applies it idempotently). `E_A + E_B` is only meaningful in a fixed gauge;
  energy *differences within* a model are gauge-invariant, the additive constant
  and overall scale are not.
- **Native lengths kept:** each energy is computed at that model's own `L`
  (CM `L=96`, PPIC `L=91`) — never trimmed or padded to a common length.
- **Weights:** `E_tot = w_A·E_A + w_B·E_B`, default `1.0`. The per-model scales
  are **not yet calibrated** across models (see Next steps), so `E_tot` with
  default weights is a documented convention, not a calibrated quantity.

## Next steps

1. **Define `E_tot` properly — cross-model scale calibration.** `E_A` and `E_B`
   are each in their own model's (gauge-fixed but un-calibrated) energy units, so
   naively summing them is arbitrary. Decide how to make them commensurable —
   e.g. z-score each against its own native-energy distribution, or fit `w_A`,
   `w_B` so that natives of each family sit at a common reference — before reading
   `E_tot` as a real objective. This is the immediate task.
2. **Search for multi-model-satisfying sequences.** With a calibrated `E_tot`,
   look for sequences that are low-energy under *both* models (the design goal):
   MCMC / simulated annealing over sequences against `E_tot`, re-aligning with
   `potts_align` (warm-started from the previous frame — the `fast` preset, see
   `docs/POTTS_ALIGN.md` §6.8). Report whether the two constraints are jointly
   satisfiable (a Pareto / infeasibility readout), which was explicitly out of
   scope for the scoring milestone.
3. **The `N > L` insertion gap (only if needed).** `potts_align` handles `N ≤ L`
   (deletions/gaps), including cross-family pairs when `N ≤ L_other`. Sequences
   *longer* than a model (needing insertions) are out of scope. Under the "design
   ≤ min(L) = 91" direction both terms are insert-free and this never arises; only
   revisit if the design space widens.

## See also

- `docs/POTTS_ALIGN.md` — the production aligner: full spec, cost model,
  DCAlign comparison, and how to run it in the combine pipeline (§11).
- `README.md` — the combine pipeline, configs, and the scoring-method table.
- `CLAUDE.md` — where each piece lives in the repo.
- `docs/MODEL_SYNC.md` — Mac↔Midway artifact sync (models + the potts_align cache).
