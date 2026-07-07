# Training convergence: `N_chains` and the low-Meff gradient-noise trap

**TL;DR.** The number of MCMC chains (`N_chains`) is not just an estimator-variance
knob — on a low-Meff alignment it decides whether L-BFGS converges at all. The ECCM
dense model trained at `N_chains=100` failed to reproduce even its *first-order*
statistics at T=1; `N_chains=200` fixed it completely. Default is now **`N_chains=200`**
for the CM/PPIC/ECCM model set. This is a training-convergence issue, **not** a sampling
artifact — sampling the bad model *longer* makes the mismatch worse, not better.

## Symptom

`figs/correlations.pdf` first-order panel at T=1 does not lie on the diagonal: the
model's single-site frequencies are systematically *more peaked* than the data
(regression slope < 1). A genuinely variable column collapses to one residue.

Measured, ECCM dense, `N_chains=100` (`results/ECCM-dense/iter-001-baseline`):

| | corr(f_i data, f_i model) @T=1 | slope (data/model) | positions off by >0.15 |
|---|---|---|---|
| ECCM dense, nc=100 | 0.806 | 0.692 | 65 / 94 |
| PPIC dense, nc=100 | 0.999 | 0.997 | — |

Example: ECCM position 83 is R/K/I/L/V in the data (max freq 0.26) but the model puts
**0.90 on Ile**. 65 of 94 positions miss by more than 0.15.

## It is not a sampling artifact

The model's own equilibrium at T=1 genuinely does not match the data. Resampling the
*final* model (fields/couplings fixed) confirms it:

- **More sampler sweeps make it worse:** corr 0.81 (100k sweeps) → 0.78 (1M).
- **Chains started at the data drift away from it:** corr 0.97 (1k sweeps) → 0.84 (100k).
- Both directions converge to the same over-peaked equilibrium.

So "sample longer" (raising `sample.k_MCMC`/`delta_t`) is the wrong lever — it moves the
synthetic MSA *toward* the wrong equilibrium faster.

## Root cause: gradient noise, not coupling strength

Each L-BFGS step estimates the pairwise term `f_ij` of the gradient from `N_chains`
sequences. For L=94, q=21 there are ~1.9M coupling components; 100 chains is a wildly
under-sampled estimate. L-BFGS assumes an accurate, smooth gradient, so a noisy `f_ij`
prevents it from reaching the fixed point: it overshoots and the couplings drift
(visible as a late upturn — a "bump" — in `J_norm` at the end of training, versus PPIC's
flat plateau).

The fix is **more chains**, not more regularization:

| variant (single-var change off baseline) | \|J\| mean | J_norm end | corr @T=1 | slope | miss >0.15 | J_norm converged? |
|---|---|---|---|---|---|---|
| baseline (λ_J=0.01, **nc=100**) | 0.0127 | 0.508 | 0.806 | 0.692 | 65 | no — bump (+0.009, +0.009) |
| λ_J=0.02 (nc=100) | 0.0088 | 0.376 | 0.870 | 0.825 | 40 | no — unstable (spike to 0.69) |
| **nc=200 (λ_J=0.01)** | 0.0128 | 0.510 | **0.998** | **1.005** | **0** | **yes** — flat (+0.001, +0.002) |

Note that `nc=200` lands at the **same coupling magnitude** as the baseline
(|J| mean 0.0128 vs 0.0127, J_norm 0.510 vs 0.508). The couplings were never too
strong — with a cleaner gradient the fields and couplings simply balance correctly, so
the T=1 Boltzmann distribution reproduces the data. A temperature scan confirms it:
`nc=200` is best at exactly T=1.0 and degrades above it, whereas the `nc=100` baseline
only reaches slope≈1.0 at T≈1.2 (it was effectively ~20% "too cold").

Raising `lambda_J` (0.01 → 0.02) only *partially* helped: it shrank the couplings and
reduced the peaking, but left the model under-fit, still slightly too cold, and its
`J_norm` trace still thrashed. It treats the symptom.

## Why PPIC was fine at nc=100

PPIC has ~2× the effective sequences (Meff 11587 vs ECCM 5689), so its data statistics
are less noisy and 100 chains sufficed. The failure is driven by **low Meff**: fewer
effective sequences → noisier `f_ij` target and estimate → the gradient noise dominates.
Expect the same trap on any low-Meff family.

## Guidance

- **Default `N_chains=200`** for these models (set in all six
  `config/params_{ECCM,PPIC}-{dense,Fij-98,profile}.yaml`).
- **Diagnose with the T=1 first-order panel of `correlations.pdf`**, or numerically with
  `corr`/`slope` of data vs synthetic `f_i`. Slope < ~0.95 with a flat plateau that turns
  *up* at the end of `J_norm` = under-converged; add chains.
- **Do not** compensate by raising the sampler's `delta_t`/`k_MCMC` (makes it worse) or by
  leaning only on `lambda_J` (under-fits).
- **Cost:** `N_chains` scales training wall-time roughly linearly (nc=200 dense ≈ 16 min on
  the Mac at 8 cores; profile is much cheaper — no pairwise energy).
- Reproducibility unchanged: `seed` + pinned `omp_num_threads` still give bit-identical
  arrays for a fixed `N_chains`.

## Provenance

Diagnosed 2026-07-06 on `git 293bb4e`. Baseline: `results/ECCM-dense/iter-001-baseline`
(nc=100, the failing model) vs `results/PPIC-dense/iter-001-baseline` (nc=100, fine). The
`nc=200` and `λ_J=0.02` single-variable tests that established the cause were run and then
discarded; re-run the full nc=200 set as `iter-002` of each run_name.
