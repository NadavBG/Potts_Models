# Spec: Energy of a sequence under two Potts models (different lengths)

**Status:** specification / decision record for a Claude Code agent. Spec-first — do
not write implementation code until you have (1) inventoried the directory, (2)
proposed a module architecture and interfaces, and (3) confirmed the open questions
in §8 with the user.

**Drop location:** a pre-existing DCA-inference directory (already contains fitted
Potts models and, presumably, MSAs and a scorer). Follow any conventions in a local
`CLAUDE.md` / `~/.claude/rules/` if present.

---

## 0. Scope

### In scope (the only deliverable right now)
Given **one amino-acid sequence** and **two separately fit Potts (DCA) models** `A` and
`B` of **different aligned lengths** `L_A ≠ L_B`, compute:

- `E_A(seq)` — statistical energy of the sequence under model A,
- `E_B(seq)` — same under model B,
- `E_tot(seq) = w_A·E_A + w_B·E_B` — combined energy, weights explicit (default 1.0).

The only nontrivial part is that a raw sequence is **not in either model's frame**; it
must be aligned to each model independently before the Potts energy is defined. How we
collapse that latent alignment is the substantive decision (§3).

### Explicitly NOT in scope now (do not build)
- MCMC / simulated annealing over sequences (design / sampling).
- Any infeasibility / Pareto / "do two models share sequences" readout.
- Model-to-model structural or profile alignment (the two models are non-homologous;
  there is no canonical column correspondence — we never compute one).
- Calibration of `w_A, w_B` against native energy distributions (deferred).

Keep the implementation a clean **scoring module + CLI**. The alignment handling chosen
here is also the substrate for later design work, so build it correctly, but stop at
"return the number(s)".

---

## 1. Conventions (fix these once, document in code)

- **Energy convention (DCAlign / standard DCA):**
  `E(S) = −Σ_{i<j} J_ij(S_i,S_j) − Σ_i h_i(S_i)`, with `P(S) ∝ exp(−E(S))`.
  **Lower energy = better (more model-typical).** Confirm this matches the sign
  convention of the models already in the directory; convert if not.
- **Gauge:** load each model and transform to a **documented common gauge
  (zero-sum)** before any energy is reported or combined
  (Ekeberg et al. 2013, *Phys. Rev. E* 87:012707; Cocco et al. 2018, *Rep. Prog. Phys.*
  81:032601). Energy *differences* within a model are gauge-invariant; the additive
  constant and overall scale are not, which is why combining `E_A + E_B` is only
  meaningful in a fixed gauge. (Scale calibration across models is separate and
  deferred.)

---

## 2. Base case first (the unit-test anchor)

If the sequence is **already in a model's frame** (length `L_k`, gaps allowed, no
indels needed), there is no latent alignment and the energy is the direct Potts sum:

```
potts_energy(S, model) = −Σ_i h_i(S_i) − Σ_{i<j} J_ij(S_i, S_j)      # O(L^2)
```

Implement and test this first. Everything else is "produce an in-frame `S` from a raw
sequence, then call this."

---

## 3. The alignment decision: marginalize (default) vs. minimize (alternative)

A raw sequence `x` (length N, ungapped) can be threaded into model `k`'s frame many
ways. Let `a` be an alignment, `S(x,a)` the induced length-`L_k` gapped sequence, and
`E_k(x,a)` its full Potts energy (plus whatever gap/insertion penalty policy we adopt,
§3.3). We must turn the alignment-dependent `E_k(x,a)` into a single `E_k(x)`.

### 3.1 DEFAULT — (b) marginal / free energy
Integrate the alignment out:

```
Ẽ_k^F(x) = −log Σ_a exp(−E_k(x,a))
```

This is the thermodynamically correct "energy of `x` under `k`": it accounts for
alignment **uncertainty/degeneracy** (free energy = mean energy − alignment entropy),
not just the single best threading. Preferred because our sequences will often align
ambiguously, where the MAP threading is an unstable summary.

The exact sum is intractable with couplings (J connects all columns → no exact DP).
Estimate it by **importance sampling** with a tractable alignment proposal `q(a|x)`
(a profile HMM, whose alignment posterior is exact and samplable):

```
Ẽ_k^F(x) = F_prop − log E_{a~q}[ exp(−(E_k(x,a) − E_prop(x,a))) ]
         ≈ F_prop − log( (1/S) Σ_{s=1}^{S} exp(−ΔE(x, a^{(s)})) ),   a^{(s)} ~ q
```

where:
- `E_prop(x,a)` = negative log-score of alignment `a` under the profile HMM,
- `F_prop = −log Z_prop` = exact HMM alignment free energy (forward algorithm),
- `a^{(s)}` are exact draws from `q(a|x)` via forward-filtering / backward-sampling
  (FFBS; Durbin, Eddy, Krogh & Mitchison 1998, *Biological Sequence Analysis*, CUP),
- `ΔE = E_k − E_prop` is the discrepancy between the true Potts energy and the proposal
  score along the sampled alignment.

**Derivation (one line):** `Σ_a e^{−E_k} = Σ_a q(a)·e^{−E_k}/q(a) = Z_prop·E_{a~q}[e^{−ΔE}]`,
take `−log`.

**Variance control:** build the HMM emissions from the Potts fields `h` (so the proposal
already captures conservation and `ΔE` is dominated by the couplings → low variance).
Always report the **effective sample size** `ESS = (Σ w_s)^2 / Σ w_s^2`,
`w_s = exp(−ΔE(x,a^{(s)}))`. If ESS is low (strong couplings / poor proposal), the
estimate is unreliable — flag it and recommend upgrading the proposal to a Potts-aware
posterior (DCAlign; Muntoni, Pagnani, Weigt & Zamponi 2020, arXiv:2005.08500) or
annealed importance sampling (Neal 2001, *Stat. Comput.* 11:125).

**Bias note:** the partition-function estimate is unbiased; `−log` of it is biased
slightly **high** (Jensen), bias `O(1/S)`, vanishing as S grows. Document S and ESS in
every output.

### 3.2 ALTERNATIVE — (a) MAP / min
Point-estimate the alignment:

```
Ẽ_k^MAP(x) = min_a E_k(x,a)
```

Cheaper and deterministic, but sees only the single best threading (no alignment
entropy). `(a)` and `(b)` coincide only when the alignment is unambiguous. Use `(a)`
when conservation is high enough that the entropy term is negligible, or when a single
committed threading is the actual deliverable.

v1 implementation: Viterbi-align `x` to the profile HMM (MAP path under fields), then
evaluate the **full** Potts energy on that path. Caveat: this is the *fields*-MAP, not
the true full-energy MAP (the HMM ignores couplings during alignment). True Potts-MAP
needs DCAlign or a local refinement (e.g. ICM seeded from the Viterbi path) — note this
approximation in the output; do not silently call it "the MAP energy".

> **Rigor caveat that drives the default.** `(a)` is a valid deterministic potential
> only if the inner solver returns a function of `x` alone. Profile-HMM Viterbi (exact
> DP) satisfies this. **Approximate, warm-started Potts-MAP (loopy BP) does not** — its
> output depends on initialization/history. `(b)`'s sampling estimator has no such
> issue. This is why `(b)` is the default once couplings enter the alignment.

### 3.3 Gap / insertion penalty policy
Decide and document one policy:
- Match columns with gaps → scored by the Potts gap state (already in `h`,`J`).
- Query residues not placed in any match column → insertions, penalized (affine,
  DCAlign-style `H_ins`, or the HMM's native insert states).
Fold the gap/insertion bookkeeping into `E_prop` so that `ΔE` in §3.1 is dominated by
Potts couplings. State the exact mapping in code comments.

---

## 4. Interfaces (proposed — confirm/adjust in your architecture proposal)

```python
load_model(path) -> PottsModel          # fields h, couplings J, length L, alphabet,
                                         # gauge; transforms to zero-sum on load
potts_energy(S_in_frame, model) -> float # §2 base case, exact

# raw sequence -> energy under one model, handling the latent alignment
score_sequence(seq, model, method="marginal", n_samples=..., seed=...,
               proposal_hmm=...) -> ScoreResult
#   method ∈ {"marginal" (default, §3.1), "map" (§3.2)}
#   ScoreResult: energy, method, (for marginal) ess, n_samples, mc_stderr,
#                chosen/representative alignment, model_hash, gauge, timestamp

score_two_models(seq, model_A, model_B, w_A=1.0, w_B=1.0, method="marginal", ...)
#   -> {E_A, E_B, E_tot, per-model ScoreResult, weights}
```

CLI: takes a sequence (string or FASTA), two model paths, `--method`, `--weights`,
`--n-samples`, `--seed`; prints `E_A`, `E_B`, `E_tot` plus diagnostics; supports a
FASTA of many sequences (batch). Deterministic for `method="map"`; for
`method="marginal"` require an explicit `--seed` and log it.

---

## 5. Constraints and relaxation hierarchy

- **C1 (hard):** each energy is computed at that model's native length `L_k`.
  Never trim/pad to a common length.
- **C2 (hard):** both models in a documented common gauge before any energy is
  reported or combined.
- **C4 (default):** alignment is **marginalized** — `method="marginal"` (§3.1).
  - *R4a:* if HMM posterior sampling (FFBS) infrastructure is not yet available, fall
    back to `method="map"` and **label outputs as MAP energies**, do not call them
    marginal.
  - *R4b:* if the input is already in-frame, skip alignment entirely (both methods
    reduce to §2). Implement this path first.
- **C5 (default):** report `E_A`, `E_B`, and `E_tot` separately; weights explicit,
  default `1.0`.
  - *R5:* weight calibration / cross-model scale comparability is out of scope now.

---

## 6. Acceptance tests (parallel validation)

1. **Gauge:** transforming a model to zero-sum gauge shifts every energy by a single
   additive constant; energy *differences* between two sequences are unchanged. Assert.
2. **Base case:** §2 matches a hand-computed small example and any existing in-frame
   DCA scorer in the directory.
3. **MAP ≈ marginal when unambiguous:** for a natural family member that aligns with no
   indels, `Ẽ^MAP ≈ Ẽ^F` (entropy ≈ 0). Assert within tolerance.
4. **Ordering sanity (correctness only, not an infeasibility claim):** natural members
   of family `k` score lower under model `k` than (i) their column-shuffled versions and
   (ii) members of the other family. This is a unit test of the scorer, nothing more.
5. **IS diagnostics:** `score_sequence(..., method="marginal")` reports ESS and MC
   stderr; the test warns/fails when ESS falls below a configurable threshold.

---

## 7. Reproducibility

- Pin versions (HMMER / pyhmmer, numpy, etc.); record in the run log.
- Log per run: model file hashes, gauge, alphabet/gap index, weights, method,
  `n_samples`, seed, ESS, wall time.
- Deterministic Viterbi; seeded, logged FFBS sampling. Same inputs + seed → same number.
- Prefer the repo's existing env (`CM_env`) and Snakemake/CLI conventions if present.

---

## 8. Open questions — confirm before coding

1. **Model file format** in the directory (bmDCA / plmDCA / EVcouplings / raw arrays)?
   Write the loader to it; confirm sign and gauge conventions.
2. **Seed MSA per model** available? Needed to build the profile HMM used as the
   alignment proposal. If absent, ask how the user wants the proposal constructed.
3. Will query sequences arrive **ungapped** (needs alignment) or **pre-aligned to a
   frame** (base case only)? Affects how much of §3 is exercised on day one.
4. **Gap/insertion penalty policy** (§3.3): adopt DCAlign-style affine, or the HMM's
   native model? Pick one, document it.
5. Is there an **existing scorer** to validate against (test 2/4)?

---

## 9. Suggested build order

1. `load_model` + gauge transform + `potts_energy` (§2) + tests 1–2.
2. Profile-HMM proposal from each model's seed MSA (emissions seeded from `h`).
3. `method="map"` (Viterbi + full Potts eval) + test 3.
4. `method="marginal"` (FFBS sampling + IS free-energy estimator, §3.1) + test 5.
5. `score_two_models` + CLI (single + batch FASTA) + test 4.

Propose the architecture and confirm §8 before starting step 1.

---

## 10. Implementation decisions (as built)

This section is the decision record for the implementation that now lives in the
repo. It resolves §8 and documents every choice made beyond the spec.

> **Operational runbook:** for the step-by-step sequence of actually running a
> combine score (including the DCAlign path: train on the Mac → align on Midway
> → score on the Mac), see `docs/PIPELINE.md`. This section is the *why*; that is
> the *how*.

### 10.1 Answers to the §8 open questions

1. **Model file format.** Models are the project's `model.npy` — a pickled dict with
   `J` `(L,L,q,q)` and `h` `(L,q)` (float64), **already zero-sum gauged** at save time.
   Sign convention matches the spec (`E = −Σ h − Σ_{i<j} J`, lower = better). The
   in-frame energy reuses the existing `SBM.utils.utils.compute_energies` verbatim — we
   did **not** write a parallel scorer.
2. **Seed MSA per model.** Present: each run dir's `inputs/msa.npy` (the encoded training
   alignment, width `L`). The profile-HMM proposal is built from it; `load_seed_msa`
   falls back to the model pickle's `Train` array if `inputs/msa.npy` is absent.
3. **Sequence form.** Mixed, by user decision: the in-frame path is first-class *and* the
   raw-ungapped (alignment) path is fully built. `method="auto"` uses in-frame when a
   sequence is in its own model's frame and marginal otherwise.
4. **Gap/insertion policy.** The HMM's native match/insert/delete states (see §10.3).
5. **Existing scorer to validate against.** `compute_energies`; the base-case test asserts
   `potts_energy` matches it (and a hand-computed 2-site example).

### 10.2 Method ladder (all three built)

`score_sequence(seq, model, method=...)` with `method ∈ {in_frame, map, marginal}`, plus
`auto` at the pipeline/CLI layer.

- `map` — Viterbi/**fields-MAP** (labelled as such — it is *not* the full-energy MAP, per
  the spec §3.2 caveat). **This is the operational default for the combine pipeline**, by the
  user's decision: the deliverable they want is the single best alignment + its energy, and
  the *same* aligner is applied to both models, so `E_A` and `E_B` are computed identically
  and are comparable. (This overrides the spec's original marginal-default, §3.1/C4. The
  marginal remains the more principled model-evidence; on the natives ESS≈1 confirmed the
  alignment is sharp, so MAP ≈ marginal and the cheaper deterministic MAP is the better fit.)
- `marginal` — the IS free-energy estimator (spec §3.1) with the profile-HMM posterior as
  proposal, reporting `ess` and `mc_stderr`. It is `−log P(x | model)`, the Bayesian model
  evidence, and the **only** mode that produces an ESS (ESS *requires* the full FFBS pass —
  so there is no cheap "MAP + ESS"; if you have paid for FFBS you have the marginal, so use it).
- `in_frame` — the exact base case (§2); requires the sequence already be length `L`.
- `auto` — `in_frame` (the sequence's **original MSA alignment**) for its home model and
  `marginal` for the other. Faster, but `E_A` and `E_B` then come from *different* alignment
  procedures and are **not strictly comparable**, so it logs a warning. Kept for speed when
  comparability is not the goal.

**Known limitation of `map` (and the planned fix).** Fields-Viterbi chooses the alignment
using only the single-site fields `h`; it never consults the couplings `J`, so its "best"
alignment is not guaranteed to minimize the full Potts energy. Direct evidence: on a CM native
the fields-Viterbi MAP scored −259 while the sequence's own curated MSA alignment scored −264
— the fields-only aligner left ~5 units on the table. The principled fix is a **couplings-aware
aligner (DCAlign**; Muntoni et al. 2020), integrated as a subprocess against its reference
implementation (the same out-of-process pattern this repo already uses for ProteinMPNN), to be
added as a `method` once its per-sequence cost is measured. A cheap in-house ICM refinement of
the Viterbi path was considered and declined in favour of the validated DCAlign tool.

### 10.3 The proposal: a self-contained numpy profile HMM (deviation from "add pyhmmer")

The proposal `q(a|x)` is a Plan7-style profile HMM implemented in pure numpy
(`src/SBM/energy/hmm.py`), **not** a wrapper around HMMER/pyhmmer. Rationale:

- the match-state architecture is fixed by the model (one match state per column);
- match emissions must be injected from the Potts fields, `e_M(k,a) ∝ exp(h_k(a))` over the
  20 amino acids (the fields-only single-site marginal), so `ΔE` is dominated by the
  couplings (the spec's variance-control requirement); the Potts gap state is represented
  by the **delete** path, never a match emission;
- the marginal estimator needs exact **forward-filtering / backward-sampling** of the
  alignment posterior, which the search-oriented APIs do not expose.

Keeping it in numpy makes it deterministic, seed-controlled, and — decisively — checkable
against **brute-force enumeration of every alignment** for tiny `(N,L)` (the DP anchor in
`tests/test_energy.py`). That is a stronger guarantee than a pyhmmer cross-check, which
would test HMMER's model, not ours. Consequently **no pyhmmer/HMMER dependency was added**
(and no new runtime dependency at all — numpy, scipy, biopython, pandas suffice).

**Gap/insertion policy (§3.3), exact mapping.** Per-column **delete** propensities are
estimated from the seed MSA's column gap frequencies (Laplace-clamped to keep full support);
**insert** open/extend are fixed affine defaults (`tau_mi=0.05`, `tau_ii=0.5`) because the
fixed-width training alignment carries no insert evidence; insert emissions are the seed-MSA
background composition. A query residue placed in a match column sits in the frame; a delete
leaves a gap (Potts state 0); an insert residue is **not** in the frame and contributes only
to `E_prop`. The "alignment space" summed over is *defined* as the set of paths through this
HMM. Because IS is unbiased for any proposal with full support, proposal quality affects only
the variance (ESS), never correctness — so the estimate is valid even when ESS is low; ESS is
the honesty signal, not a correctness gate.

### 10.4 Two-model scoring and the pipeline

- `score_two_models(seq, A, B, w_A, w_B, ...)` → `{E_A, E_B, E_tot, per-model ScoreResult,
  weights}`; `E_tot = w_A·E_A + w_B·E_B`, weights default 1.0 (spec C5). Each model keeps its
  native length; nothing is trimmed/padded (C1). Both are loaded in the zero-sum gauge (C2).
- **CLI** `scripts/score_two_models.py`: single `--seq` or batch `--fasta`; `--method`,
  `--weights`, `--n-samples`, `--seed` (required for marginal), `--ess-threshold`. Writes a
  tidy `scores.tsv`, a per-sequence `scores_detail.json`, an `alignments.txt` report, and a
  provenance `manifest.json`.
- **Pipeline** `Snakefile.combine` + validated `src/SBM/combine_config.py` (exactly two
  models). Run via `python scripts/iter.py run <name> "<tag>" --snakefile Snakefile.combine`.
  Combine runs land under **`combine/<run_name>/`** — a separate, git-ignored tree from the
  single-model `results/`, so the two never mix.
- **Clarity Choices** The model `name` is the exact variant
  (`CM-bm-dense`, not `CM`) and labels the figure axes, the query groups, and every manifest.
  The figure is captioned with each model's exact `run_dir` + sha256. `alignments.txt` shows,
  for each sequence, the raw query and its best alignment under *each* model side-by-side
  (stacked; the frames are non-homologous and different-length, so not column-aligned), with
  both energies, the method, and the ESS.

### 10.5 Reproducibility (spec §7)

Pure numpy/scipy with a seeded `np.random.default_rng` for FFBS; deterministic forward/Viterbi.
Per-(sequence, model) seeds derive from the master seed in stable (sorted-id) record order, so
the batch is reproducible and adding/removing a sequence does not perturb the others. The
energy path does no OpenMP, so results are bit-reproducible regardless of thread count
(unlike training). `scores.tsv` was verified bit-identical across re-runs. Every run records
model hashes, gauge, method, `n_samples`, seed, and the ESS distribution in `manifest.json`.

### 10.6 Acceptance tests (spec §6) — status

All implemented in `tests/test_energy.py` and passing: (1) gauge invariance, (2) base case vs
`compute_energies` + hand example, (3) MAP ≈ marginal when unambiguous, (4) ordering sanity
(natives score lower under their own model than shuffles / the other model — verified on the
real CM/PPIC models, 4/4 each way), (5) IS diagnostics (ESS + stderr, low-ESS warning), plus
the **DP anchor**: forward log-Z, FFBS sample frequencies, marginal-IS, and Viterbi all
checked against brute-force enumeration. Input-validation tests guard gapped raw queries and
`n_samples ≤ 0`.

### 10.7 Out of scope / deferred (unchanged from §0, §5 R5)

No design/sampling over sequences, no Pareto/infeasibility readout, no model-to-model
structural alignment, and **no cross-model scale calibration of `w_A, w_B`** — `E_A` and `E_B`
are in each model's own (gauge-fixed but un-calibrated) units, so `E_tot` with default weights
is a documented convention, not a calibrated quantity. Upgrading the proposal (DCAlign /
annealed IS) for low-ESS cross-family alignments is the natural next step if those numbers
need to be quantitative.

### 10.8 DCAlign integration spike — findings (2026-06-16)

A spike confirmed that **DCAlign** (Muntoni, Pagnani, Weigt & Zamponi 2020,
arXiv:2005.08500) — the couplings-aware aligner named as the `map` upgrade in §10.2 — can
consume our pre-fit Potts models directly. Twenty raw natural queries (10 CM, 10 PPIC) were
scored under both models with DCAlign and compared against the current fields-Viterbi `map`.

**Validated — the integration is mechanically sound.**

- External `h`/`J` handoff works with no internal DCA fit: `palign(seq, J, h, Λ, :amino)`
  takes our arrays. The handoff is transpose `(L,L,q,q) → (q,q,L,L)`, gap-index remap `0 → 21`
  (residues 1..20 already coincide), zero-sum gauge, as a raw little-endian Float64 binary read
  straight into Julia (exact transform in §10.9).
- **Transfer check:** `|E_dcalign − our potts_energy recomputed on DCAlign's own alignment|`
  ≤ 5e-7 across all 32 completed alignments → the handoff and the shared zero-sum gauge are
  correct and `E_A`/`E_B` are directly comparable. DCAlign's energy sign convention matches
  ours (`E = −Σh − Σ_{i<j}J`).
- Couplings-awareness helps where expected: on ambiguous cross-family alignments DCAlign finds
  a lower-energy frame in ~half the cases (best ΔE = −9.4 units); on unambiguous natives it
  reduces to our answer (ΔE = 0.000, 100% column agreement; median agreement over all 32 was
  0.84).

**Blocker 1 — the naive config is not yet trustworthy.** With a flat insertion prior Λ (mass
on Δn=1 + pseudocount floor), default annealing, `μ=0`, and the zero-sum gauge, **3 of 16
same-family natives mis-aligned badly** — reported `converged=true` (not the decimation
fallback), yet scored worse than plain fields-Viterbi:

| sequence | E (DCAlign) | E (our `map`) | ΔE | col agreement |
| --- | --- | --- | --- | --- |
| CM_nat_04 | −132.3 | −264.4 | +132 | 0.71 |
| CM_nat_05 | −131.9 | −247.5 | +116 | 0.70 |
| CM_nat_08 | −76.4 | −186.5 | +110 | 0.55 |

A couplings-aware aligner should never do worse than fields-Viterbi on an in-frame native, so
the flat Λ is the prime suspect (the first tuning experiment, §10.9).

**Blocker 2 — cost mandates a cluster.** DCAlign: **median 19 s/seq, max 451 s** (the slow
regime is `N<L`, where `palign` auto-bumps to 5000 sweeps; our queries are mostly shorter than
`L`). Our fields-Viterbi `map`: 27 ms, flat → DCAlign is ~700× slower at the median. The
envisioned ~2258-seq × 2-model ≈ 4500-alignment job is ~24–60 h single-threaded —
embarrassingly parallel, so ~1–2 h on a 40-core node.

### 10.9 Recommended next steps — Midway-first DCAlign integration, then prior tuning

Sequencing (user decision): stand up cluster-scale capability **first** (to enable larger
experiments), **then** tune; the first tuning experiment is **fixing the prior** (Blocker 1).
The Midway execution mechanism is left to the Midway agent. This recipe is self-contained so
the (deleted) spike code is not needed.

**A `dcalign` method (out-of-process, ProteinMPNN-style).** Add `"dcalign"` to `METHODS`
(`src/SBM/energy/score.py`) and `_METHODS` (`src/SBM/combine_config.py`); add a `dcalign`
branch in `score_sequence` returning a `ScoreResult` with `representative_alignment` + energy.
Shell out to Julia exactly as we do for ProteinMPNN — mirror `src/SBM/utils/mpnn_score.py`
(`_resolve_*_path`, `MPNNContext`, `score_sequences`, result parsing, loud-failure errors) and
the `mpnn.path` / `mpnn.python` config precedent in `src/SBM/workflow_config.py`: resolve the
DCAlign repo + Julia interpreter from a `DCALIGN_PATH` env var or a `scoring.dcalign_path`
config field, invoke `subprocess.run(..., cwd=dcalign_repo)`, parse back the aligned sequence +
energy. The `score` rule (`Snakefile.combine`) is already cluster-annotated (`threads:4`,
`mem_mb=8000`, `runtime=240`).

**The validated model handoff (preserve exactly).** Our `-ACDEFGHIKLMNPQRSTVWY` (gap 0,
residues 1..20) maps to DCAlign's `A..Y=1..20, gap=21`, so only the gap moves. With
`ORDER = list(range(1,21)) + [0]`:

```python
J_dca = J.transpose(2, 3, 0, 1)[ORDER][:, ORDER]          # (L,L,q,q) -> (q,q,L,L)
h_dca = h.T[ORDER]                                         # (L,q)     -> (q,L)
path.write_bytes(J_dca.astype("<f8").tobytes(order="F"))   # Julia read! into (q,q,L,L)
```

Julia side (deps instantiate cleanly; `palign` / `Seq` are exported, decode/energy helpers are
`DCAlign.`-qualified; `HMMER_jll` / `Infernal_jll` / `PlmDCA` are only for seed-building or
fitting and are not needed when we bring our own model):

```julia
_, conv, res, _ = palign(rawseq, J, h, Λ, :amino;
                         maxiter=2000, seed=0, pcount=1e-3, verbose=false)
out = DCAlign.decodeposterior(res.pbf.P, res.seq.strseq, thP=res.alg.thP)
if !DCAlign.check_assignment(res.pbf.P, false, length(rawseq))
    _, P = DCAlign.decimate_post(res, false)               # nucleation fallback
    out = DCAlign.decodeposterior(P, res.seq.strseq, thP=res.alg.thP)
end
energy = DCAlign.compute_potts_en(J, h, out.seq, L, :amino)   # sign matches ours
```

**First experiment — fix the prior.** Replace the flat Λ (mass on Δn=1 + pseudocount floor)
with an **informed prior** via `DCAlign.deltan_prior(seed_ins, L)`. This needs a
seed-with-insertions (HMMER lowercase `.ins`), which our fixed-width `inputs/msa.npy` does not
carry — the likely source is `../Make_Alignment`'s HMMER-aligned artifacts. Fallback if no
insertion data: tune the flat-Λ shape + nonzero `μint` / `μext` gap penalties. **Success
metric:** same-family ΔE → ≈0 on CM_nat_04/05/08, and no native scoring worse than
fields-Viterbi.

**Scale on Midway.** ~4500 alignments at ~19 s each ⇒ shard the query set (one Slurm task per
shard, then gather) and/or use Julia threads within a shard (DCAlign's own example threads over
sequences). Two execution mechanisms — the Midway agent picks based on what is already set up
there:

- a **Snakemake Slurm profile** over the existing combine DAG (every rule already declares
  `threads` + `resources`), or
- **`../Make_Alignment`-style `sbatch` wrappers** in a `pipeline/external/`, driven from the
  login node with `git pull --ff-only` + `sbatch --dependency` chains.

Either way the sync is git-pull on Midway (iteration dirs under `combine/`), and Julia + the
DCAlign package are installed once on Midway (a module, or a `/scratch` conda/juliaup env, as
Make_Alignment sets up its bioconda env manually).

**Open inputs for the Midway agent:** the DCAlign clone location + Julia environment on Midway;
the seed-`.ins` source for the informed Λ; the shard count / partition / account.

### 10.10 Phase-1 implementation (2026-06-16, Midway)

Built and validated `method="dcalign"` per §10.9. **Phase-2 (informed insertion prior Λ via
`deltan_prior`, Blocker 1) is deferred** — `lambda_spec` is wired but only `"flat"` is implemented.

- **Bridge** `src/SBM/utils/dcalign_score.py` (mirrors `mpnn_score.py`): `model_to_dcalign_arrays`
  (the §10.9 transform), `dcalign_context` (resolves `DCALIGN_PATH` / `julia`, captures git commit +
  Julia version), `align_sequences` (writes `<f8` Fortran model bins + `meta.json` + `queries.fasta`,
  shells out to Julia, parses the TSV; loud `RuntimeError` on nonzero rc), and the cache I/O
  (`read_alignment_cache` / `write_alignment_cache`). **Julia driver** `src/SBM/julia/run_dcalign.jl`
  (run with `--project=<clone>`): flat Λ = mass on Δn=1 (the `Alg` ctor adds the pcount floor), the
  §10.9 `palign`→`decodeposterior`→`decimate_post`→`compute_potts_en` recipe verbatim, per-row
  flushed TSV (resumable), per-seq try/catch (empty-frame on failure, no silent drop).
- **Scoring** `score.py` `dcalign` branch is a pure cache-reader (`dcalign_frame` → `potts_energy`);
  an empty/missing frame is a loud error. `combine_config.ScoringConfig` gains
  `dcalign_path, julia, dcalign_seed, maxiter, pcount, n_shards, lambda_spec` (not added to `auto`).
  `score_two_models.py` adds `--dcalign-cache` and a manifest `dcalign` block with a per-model
  `dcalign_agreement` canary (max/median `|potts_energy − DCAlign energy|`).
- **Cluster (chosen mechanism: `Make_Alignment`-style sbatch, not a Snakemake Slurm profile):**
  `pipeline/external/run_dcalign_align.sh` (login driver: git-pull, preflight, `plan`, submit
  `--array=0-(2N-1)` shard tasks + an `afterok` gather), `sbatch_dcalign_{shard,gather}.sh`,
  `finalize_dcalign_push.sh` (sacct-validate, compress; the cache moves to the Mac by
  rsync via `scripts/sync_models.sh pull`, not git — see `docs/PIPELINE.md`). Entrypoints
  `scripts/wf/run_dcalign_shard.py` (`plan`/`run`, round-robin shards, resume-skip) +
  `run_dcalign_gather.py`. Account/partition `pi-ranganathanr`/`caslake`.
- **Validation:** Tier-0 (handoff round-trip, branch = in-frame, cache I/O, config bounds) and
  Tier-1 (end-to-end vs the real DCAlign clone: energy transfer ≤5e-7 — observed ~1e-15) pass;
  a synthetic full-flow run (`plan`→4 shards→gather→`score --method dcalign`) gives
  `dcalign_agreement` ~9e-16. **Tier-2 (real CM/PPIC models, actual sbatch submission) is now
  validated** — see §10.11. Models reached Midway via checksummed rsync (`scripts/sync_models.sh
  push`; `docs/MODEL_SYNC.md`); they are not in git (Git-LFS was removed in favor of rsync).
- **Midway env facts:** DCAlign clone at `../DCAlign` (commit `cab443f`); Julia 1.10.2 depot at
  `/scratch/midway3/nadavbg/julia_depot`; SBM in a uv `.venv` at the repo root. `module load julia`
  breaks `git` HTTPS (Julia's mbedTLS `libgit2` shadows system git) — scripts export
  `GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt`.

### 10.11 Tier-2 validation on the real models (2026-06-17, Midway)

With the CM/PPIC models rsync'd to Midway, Tier-2 was validated end-to-end on the real
models via a tiny smoke (`config/params_combine-CM-PPIC-dcalign-smoke.yaml`: cap 8/group →
48 seqs → 96 alignments, `n_shards=2`), the actual `run_dcalign_align.sh` → array → gather
→ score path.

- **Bug found + fixed (real-sbatch-only):** the shard/gather jobs ran with CWD = the driver's
  submit dir (`RUN_ROOT/dcalign`), so `load_model` on the repo-root-relative paths in
  `models.json` raised `FileNotFoundError`. The synthetic full-flow never went through the
  real driver, so Tier-1 missed it. Fix: `cd "${REPO_DIR}"` in both `sbatch_dcalign_{shard,gather}.sh`
  (RUN_ROOT is passed absolute; `#SBATCH --output` is opened by Slurm pre-`cd`, so logs are
  unaffected).
- **Validated on the real models:** agreement canary (our in-frame `potts_energy` vs DCAlign's
  own energy) max |Δ| = **4.8e-13 (CM) / 5.7e-13 (PPIC)**, 0 alignment failures; the gather →
  `score --method dcalign` → `manifest.json` chain completes (`scores.tsv` written for 48 seqs
  under both models). The `render_combine` figure is the only step that fails on Midway — it
  needs `lab_plotting` (not in the `[workflow]` venv); it is not part of the alignment and
  renders on the Mac.
- **Within-shard threading added** (`run_dcalign.jl` now `Threads.@threads :dynamic` over a
  shard, thread count from `--cpus-per-task` via `JULIA_NUM_THREADS`; write under a lock,
  BLAS pinned to 1; driver gains `DCALIGN_CPUS`/`DCALIGN_MEM`). Threaded answers are
  byte-identical to single-thread (`align_one` deepcopies `J/h/Λ`, `palign` is seeded per seq).
  **But threading scales poorly** — measured 1.7× on 4 threads, 2.9× on 8 threads (~36%): each
  alignment is one chunk and a slow `N<L` sequence bounds a thread. So the recommended cluster
  lever is **shard fan-out** (`cpus-per-task=1`, many shards), which is ~100% core-efficient and
  fits the caslake QOS (4800 cores / 1000 jobs / 65533 array) easily.
- **Cost (measured, real models):** mean ~200 s/seq — almost all combine-query sequences are
  `N<L` (raw lengths 78–96, median 91; 99.9% of CM and 26% of PPIC alignments hit the slow
  5000-sweep regime), so the 19 s spike median (raw naturals) does not apply here. **Full run
  (3600 alignments): fan-out `cpus=1`/256 shards ≈ 160–230 core-hours, ~15–60 min wall**
  (threaded `cpus=8` ≈ 640 core-hours — 3–4× more). The full query is staged at
  `combine/combine-CM-PPIC-dcalign/iter-001-baseline`; launch with
  `DCALIGN_CPUS=1 DCALIGN_MAX_CONCURRENT=512 bash pipeline/external/run_dcalign_align.sh <RR>`.
- **Phase-2 (informed insertion prior `deltan_prior`, Blocker 1) remains deferred** — the next
  tuning experiment, now that cluster-scale capability is validated.

### 10.12 DCAlign-vs-in-frame baseline rule (2026-06-18, Mac)

To make Blocker 1 a *measured* baseline (a "thing to beat" for the Phase-2 prior tuning), the
combine pipeline now produces a per-sequence comparison of DCAlign's best-attempt energy against
the **native in-frame energy**. For every query in its own model's frame (a "home pair"), the
native frame energy is `potts_energy(query, model)` and DCAlign's energy is recomputed in-frame
on its cached frame; `ΔE = E_dcalign − E_inframe`, so `ΔE > 0` means DCAlign scored the native
*worse* than the trivial native frame (the pathology). Cross-family pairs have no in-frame
reference (different length) and are skipped.

- **Code:** pure logic `src/SBM/energy/dcalign_baseline.py` (`column_agreement`, `compare_record`,
  `summarize`; unit-tested in `tests/test_energy.py`); figure `src/SBM/utils/utils_dcalign_baseline_plot.py`;
  CLI `scripts/compare_dcalign_baseline.py`; wrapper `scripts/wf/run_dcalign_baseline.py`.
- **Rule** `dcalign_baseline` in `Snakefile.combine`, defined only when `scoring.method == "dcalign"`
  and run on the Mac (pure numpy, no Julia). Outputs `data/dcalign_vs_inframe.tsv` (tidy, one row per
  home pair), `data/dcalign_vs_inframe.json` (ΔE summary per model/group + the standing cache canary),
  `provenance/dcalign_vs_inframe_manifest.json`, and (when figures enabled) `figs/dcalign_vs_inframe.pdf`
  (per-model `E_dcalign` vs `E_inframe` scatter with the `y=x` diagonal + the ΔE histogram). The
  scatter overlays the **not-converged** points (DCAlign decimation fallback) as open rings.
  `n_worse` counts `ΔE > equal_tol` (default 1.0 a.u.); the figure uses the same threshold.
- **Baseline result (flat prior, the full `iter-001-baseline` run, 1800 home pairs):** DCAlign
  scored worse than the native frame on **569/900 CM (63%, median ΔE +11.9, max +168)** but only
  **112/900 PPIC (12%, median ΔE 0)** — so Blocker 1 is pervasive on CM, not the 3-sequence anomaly
  §10.8 reported, and strongly model-dependent. The gauge/handoff canary held (max |Δ| ≤ 1.5e-12).
  This is the number the informed-prior experiment (§10.9) must drive toward ≈0.
- **Convergence is not the cause — it is converged-on-a-bad-frame.** A companion rule
  `dcalign_convergence` (`scripts/report_dcalign_convergence.py` + `convergence_by_group` in
  `dcalign_baseline.py` + figure `utils_dcalign_convergence_plot.py`) counts non-convergence per
  (model, group) over **all** alignments (home + cross). DCAlign falls back to decimation rarely and
  almost only on **cross-family** frames: **40/1800 (CM), 14/1800 (PPIC)** total, of which only
  **4 (CM) / 1 (PPIC)** are home pairs. So the 569/112 worse-than-native home pairs are overwhelmingly
  *converged* — the flat insertion prior steers `palign` to a confidently-wrong frame, which is what
  the Phase-2 `deltan_prior` must fix (not a convergence/annealing problem). Outputs
  `data/dcalign_convergence.{tsv,json}` + `figs/dcalign_convergence.pdf`.
- **Run-dir layout (tidied 2026-06-18).** Combine runs now group Mac-side outputs into `data/`
  (tables: `scores.tsv`, `scores_detail.json`, `alignments.txt`, `dcalign_vs_inframe.{tsv,json}`,
  `dcalign_convergence.{tsv,json}`) and `provenance/` (`score_manifest.json`, the two diagnostic
  manifests, `run_manifest.json`); `figs/` holds figures. The top level keeps only the
  **Mac↔Midway contract** files the cluster DCAlign scripts read by exact path —
  `config_snapshot.yaml`, `models.json`, `query/` — plus `iteration_note.md` and `dcalign/`/`logs/`.
  `scripts/sync_models.sh` selects by directory-name prune, so `data/`/`provenance/` are synced
  automatically (no sync change). The cluster scripts were intentionally **not** touched.

### 10.13 Informed insertion prior `lambda_spec="deltan"` (2026-06-18) — the Blocker-1 fix

Phase-2 of §10.9: replace the flat insertion prior (the cause of the §10.12 baseline pathology)
with an **inferred** prior, keeping the trained Potts `h`/`J` untouched. `scoring.lambda_spec`
now accepts `"deltan"` in addition to `"flat"`.

- **Why flat failed.** DCAlign's `palign` weights an insertion prior `Λ[i,j,Δn]` = the
  distribution of the number of symbols between used match columns `i,j` (its `compute_dist`
  convention: `Δn=1` = adjacent residue columns, no insertion). The flat prior
  (`build_lambda` in `src/SBM/julia/run_dcalign.jl`) put **all mass on `Δn=1` for every `(i,j)`
  pair, position-independently** — a geometry-blind prior. With only the `Alg`-ctor pcount floor +
  noise to break ties, `palign` had no positional guidance and converged on confidently-wrong
  frames (CM 569/900 home pairs worse-than-native; PPIC 112/900 — §10.12).
- **The fix (`deltan`).** Use `DCAlign.deltan_prior(seed.ins, L)` (its native prior builder; the
  exact usage in DCAlign's own `script/Run_alignment.jl`: `Λ,_,_ = deltan_prior(...)` →
  `palign(..., deepcopy(Λ), ...)`). It builds the empirical per-`(i,j)` distribution from a
  **model-frame** seed alignment.
- **Frame-matching reality (resolved).** `deltan_prior` needs an insertion-bearing seed in the
  model's *exact* L-frame. In DCAlign's native pipeline `align_seed_pfam` (hmmbuild) ties the
  prior, `L`, and the Potts frame to one HMM. **Ours is bespoke** (Make_Alignment reference
  coordinates: CM `L=96`, PPIC `L=91`), and Make_Alignment's insertion-bearing files are in the
  Pfam ~80-col frame while its model-frame files have insertions stripped — so there is no
  ready-made model-frame insertion seed. **Decision (user):** build the seed from **each model's
  own training MSA** (`inputs/msa.npy`, already in the exact L-frame), written as an all-match
  a2m. It carries no insert columns, so the prior learns the empirical **gap/deletion geometry**
  per `(i,j)` rather than literal insertions — but that alone replaces the degenerate flat prior
  with real per-position statistics.
- **Plumbing.** `SBM.utils.dcalign_score._write_seed_ins` converts the seed MSA to `seed.ins`
  (one unique-id record per row — DCAlign's `readfull` dedupes by header; gaps→`-`, residues
  uppercased via `_ints_to_str`); `align_sequences` stages it into each shard work dir when
  `lambda_spec != "flat"` (after asserting MSA width == `model.L`). `build_lambda(spec, L, in_dir)`
  gains a `"deltan"` branch = `first(DCAlign.deltan_prior(joinpath(in_dir,"seed.ins"), L))`.
  `combine_config.ScoringConfig` validates `lambda_spec ∈ {"flat","deltan"}` (default `"flat"`, so
  existing flat runs are unchanged). The cluster wrappers already thread `lambda_spec` through —
  no change. `score.py`'s `dcalign` branch is prior-agnostic (cache-reader) — no change.
- **Status / validation.** Python path validated on the real models: `seed.ins` is `1258×96`
  (CM) / `26701×91` (PPIC), all-uppercase+`-`, unique ids, width == `L`. `tests/test_energy.py`
  covers the config validation + `_write_seed_ins` (32 passed). A local Julia smoke (DCAlign
  `cab443f`, Julia 1.12.6) confirmed `build_lambda("deltan", L, in_dir)` dispatches and reaches
  `DCAlign.deltan_prior` with the correct `seed.ins`/`L`; it then fails *inside* DCAlign's
  `readfull`→`FastaIO` on a macOS-only zlib gap (`could not load symbol "gzopen64"`, the Linux LFS
  variant macOS lacks). The flat path never hits FastaIO (it uses the driver's own `read_fasta`),
  so this surfaces only for `deltan` and only on macOS. The authoritative end-to-end check is the
  Tier-1/Tier-2 run on Midway (Linux, Julia 1.10.2, where `gzopen64` exists and DCAlign's own
  pipeline uses FastaIO routinely). **iter-002** sets
  `lambda_spec: deltan` in `config/params_combine-CM-PPIC-dcalign.yaml` (+ the `-smoke` variant).
- **Non-degeneracy verified locally (the load-bearing claim).** Bypassing the macOS FastaIO gap by
  feeding the real CM seed Dict straight into DCAlign's own `compute_dist` (Julia 1.12.6, clone
  `cab443f`): the resulting Λ mode tracks column distance — adjacent `(1,2)`→Δn 1 but `(1,11)`→10,
  `(1,31)`→30, `(40,60)`→20 — and only **2.1%** of `(i,j)` pairs have mode Δn=1, versus the flat
  prior forcing Δn=1 on **100%**. So the insertion-free seed does **not** collapse back to flat;
  the gaps make Δn encode real per-`(i,j)` geometry, which is the whole point.
- **Success gate (vs the §10.12 flat baseline).** Home pairs worse-than-native (ΔE>1) CM
  569/900 → ≈0, PPIC 112/900 → ≈0, and no native scoring worse than fields-Viterbi `map`. The
  in-frame-vs-DCAlign energy canary (≤~1e-12) must still hold (the prior changes the alignment,
  not the energy recompute). **Escalation if `deltan` under-fixes CM:** build a profile HMM in the
  model frame and `hmmalign` the raw ungapped sequences to recover *literal* insertions, then
  re-run — deferred behind this measurement.
- **Memory / cost note.** `deltan_prior` allocates `dist` of shape `(N_seed, L, L)` int64 at shard
  startup; PPIC's 26701-seq seed → ~1.8 GB transient. This is **before** the first row flush, so an
  under-budgeted shard OOM-kills with zero progress. The fan-out path in
  `pipeline/external/run_dcalign_align.sh` therefore floors `--mem` at **4G** (it was `cpus*2`, i.e.
  2G for the documented `DCALIGN_CPUS=1` launch — too tight for `deltan`); `DCALIGN_MEM` still
  overrides. Each shard rebuilds the (deterministic) prior independently — redundant across a model's
  shards and on every resume, but dwarfed by the ~200 s/seq align cost; a build-once-and-stage-Λ
  design is the obvious optimization if the seed grows much larger.

### 10.14 iter-003 Phase-0 residual diagnostic — the insertion escalation is the wrong tool (2026-06-28, Mac)

§10.13's success gate named an escalation if `deltan` under-fixed CM: build a model-frame profile
HMM and `hmmalign` raw sequences to recover **literal insertions** for the prior (§10.13
"Escalation"). Before paying for that (a cross-project pull of full-length sequences from
Make_Alignment + HMMER), a **Phase-0 diagnostic gate** characterized the iter-002 worse-than-native
residual directly from the cached alignments (no new DCAlign run). Decision: **NO-GO** — the residual
is gap-placement, not insertion-shaped, so a literal-insertion seed cannot fix it.

- **Tooling (new, tested).** `SBM.energy.dcalign_residual` (pure logic: natural/synthetic
  decomposition + a geometric frame-disagreement classifier — `terminal` / `register_shift` /
  `gap_redistribution`), `scripts/analyze_dcalign_residual.py` (CLI; writes
  `<run>/analysis/residual_rows.tsv` + `residual_analysis.json` + `residual_anatomy.pdf` +
  manifest), `SBM.utils.utils_dcalign_residual_plot`. The classifier reuses
  `dcalign_baseline.compare_record`/`column_agreement`; covered in `tests/test_energy.py`
  (hand-built terminal/register/gap-redistribution frame pairs with known labels).
- **Result (iter-002, `equal_tol=1`).** 249/1800 home pairs worse-than-native: **40 natural, 209
  synthetic (84%)**. Synthetic are L-frame MCMC samples with no insertions → structurally out of
  reach for any insertion prior. **`register_shift == 0` across all 3600 alignments** — not one
  block/column displacement. Natural tail: 39/40 `gap_redistribution`, 1 `terminal`, 0
  `register_shift`; mean `col_agreement` 0.94 (a few residues placed in different columns *around
  gaps*, costly under the couplings). `insertion_free` confirmed: `max_n_residues == L` (96/91), so
  a literal-insertion seed could act only through the prior Λ, never the queries.
- **Why NO-GO.** The §10.13 escalation targets register/insertion errors; the residual has none. A
  worse-than-native frame from a couplings-aware minimizer means the **non-Potts terms (prior Λ +
  gap penalties) over-penalize the native gap arrangement** and steer DCAlign to a higher-Potts-energy
  frame. The indicated lever is a **non-Potts term** — gap-penalty (`μint`/`μext`) *or* the prior
  weight (`pcount`). A cheap objective-decomposition (Potts vs prior vs gap terms on the 40 natural
  failures) should precede any knob choice. **→ §10.15 ran that decomposition and the answer is
  `pcount`, not `μint`/`μext`.**
- **Status.** iter-003 paused after Phase-0 per user decision ("stop and record"). The Phase-1
  insertion-seed plumbing (`deltan_ins` / `seed_a2m`, HMMER `hmmbuild --hand` + `hmmalign`, raw-hit
  extraction) is **not built** — shelved unless this verdict is revisited. Artifacts:
  `combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior/analysis/`.

### 10.15 iter-003 lever hunt — Phase-A predicted `pcount`; Phase-B refuted it (the residual is a DCAlign inference failure) (2026-06-28)

§10.14's "indicated lever is `μint`/`μext`" was a guess; this is the objective-decomposition it called
for. Reading DCAlign's actual objective (the `palign` signature + `central!` in the clone's
`iterate_bplc.jl`, vs `src/SBM/julia/run_dcalign.jl`) shows it was the **wrong knob**:

- **`μint`/`μext` penalize gap *count* per column** (scalar `palign` kwargs, default `0.0`;
  `run_dcalign.jl` never passes them, so iter-002 ran at μ=0): `−μint` per interior gap column,
  `−μext` per terminal gap column. **The prior Λ decides gap *placement*.** DCAlign's reported
  `dcalign_energy` is the *pure Potts* energy (`compute_potts_en`; the in-frame recompute canary ≤5e-7
  proves it carries no μ/prior term) — so on every worse pair the **native frame already wins the
  Potts energy**, and only the prior Λ (annealed `Λ^β`) overrode it. The knob that flattens Λ so the
  lower-Potts native frame wins is **`pcount`** (the DCAlign `Alg` constructor blends
  `(1−pcount)·Λ + pcount`), which is **already plumbed** (`ScoringConfig`→`meta.json`→`palign`).
- **Tooling (extends §10.14).** `SBM.energy.dcalign_residual` gains `gap_profile` (interior/terminal
  gap split, mirroring the μint/μext regions), `lever_bucket` + `mu_floor`, an `addressability`
  aggregator, and `lever_verdict`; surfaced by `scripts/analyze_dcalign_residual.py`
  (`residual_analysis.json["addressability"]` + `lever_verdict`, a third figure panel) and covered in
  `tests/test_energy.py`. Per worse pair it buckets the lever: **`prior_only`** (equal gap counts in
  both frames ⇒ μ is *provably* neutral; only `pcount` can move it), **`mu_addressable`** (native has
  fewer gaps in some class ⇒ a μ knob *could* help — candidate only; `mu_floor = ΔE/|Δn|` is a *lower*
  bound that ignores the native-disfavouring prior), **`mu_counterproductive`**.
- **Result (iter-002).** Of 249 worse pairs: 121 `prior_only`, 128 `mu_addressable`, 0
  `mu_counterproductive`. Read by the recovery **target** (the natural ground states; synthetics are
  diagnostic): the **natural tail is 85% `prior_only`** (CM-natural **94%**, 31/33). `mu_addressable`
  is almost entirely via **μext** (126/128), concentrated in synthetics, at candidate-only floors
  (natural median μ_floor ≈ 29 a.u. — implausibly large, and a lower bound). The overall 49%
  `prior_only` that a naive reading would call "μ-shaped" is a synthetic-count artifact. Invariant
  checks: `prior_only ⟺ (Δn_int=0 ∧ Δn_ext=0)` exactly; the canonical CM pair (94/94 residues, ΔE 6.6)
  is `prior_only`; 246/249 worse pairs have equal residue count. **Verdict: tune `pcount`** (the only
  knob that touches the natural target *and* the prior_only synthetics); μ is at best a weak secondary
  candidate for some synthetics.
- **Phase-B pre-screen (built for Midway; awaiting the cluster run).** A `pcount` sweep
  (`[0.001, 0.01, 0.05, 0.1, 0.2, 0.5]`, 0.001 = the iter-002 canary) over a curated subset of
  iter-002 home pairs — **all 249 worse-than-native pairs (40 natural + 209 synthetic)** as the
  recovery set + 80 seeded currently-good controls (per model × kind) — every other DCAlign setting
  pinned to iter-002 (`deltan` prior, seed 0, maxiter 2000). The goal is DCAlign finding **each
  sequence's** energy minimum, so synthetics count equally with naturals: every worse-than-native
  pair is the aligner returning a higher-energy frame than an available reference, i.e. a real
  minimization failure. This also makes `pcount` the *universal* lever — at μ=0 the prior Λ is the
  only non-Potts term and the native frame already wins the (pure-Potts) reported energy on every
  worse pair, so the prior alone caused all 249; flattening it (`pcount`) is what can move them
  (`μ` is at most an alternative knob for the `mu_addressable` subset, with the side-effect cost).
  Decision metric: the largest `pcount` recovering the most worse pairs (reported split by kind)
  with **no** control regression; `ΔE < 0` ("beat native" — a strictly lower minimum) is reported
  separately and watched on naturals (a lower-energy but non-biological frame is good-for-min but
  may hurt downstream biological scoring).
- **macOS blocker (why it runs on Midway, not the Mac).** DCAlign *loads* locally (`using DCAlign`
  in ~1 s), but its `deltan_prior` reads `seed.ins` through `FastaIO → GZip.jl`, and **GZip 0.6.2
  ccalls `gzopen64`, a symbol macOS's zlib does not export** (`dlsym … gzopen64: symbol not found`).
  GZip 0.7.1 fixes it and works on this Mac, but **FastaIO 1.1.0 (latest) hard-caps GZip at
  `"0.5,0.6"`**, so the fix can never be selected without patching sibling source. This is the same
  class of macOS-toolchain issue that makes Midway the home of the DCAlign align step; per user
  decision the pre-screen runs there. (The `flat` prior would run locally but changes the prior,
  confounding the pcount signal — not a valid pre-screen.)
- **Tooling (new).** `scripts/pcount_presweep.py` (`build` → one cluster-ready run dir per pcount
  under `combine/combine-CM-PPIC-dcalign-pcsweep/pc<val>/`, reusing the existing
  `pipeline/external/run_dcalign_align.sh` machinery; `score` → reads the synced per-pcount caches,
  scores every curated home pair in-frame via `dcalign_residual.analyze_record`, writes
  `presweep_rows.tsv` + `presweep_summary.json` + `presweep.pdf`) and the figure
  `SBM.utils.utils_pcount_presweep_plot`. The *scoring* half is GZip-free and runs on the Mac; only
  the alignment needs Linux. *(Later generalized + renamed to `scripts/dcalign_presweep.py` /
  `SBM.utils.utils_dcalign_presweep_plot` — any `scoring.*` knob, `--scoring-key`; see §10.16. The
  archived pcsweep re-scores byte-identically under the generalized scorer.)*
- **Result (2026-06-28) — `pcount` is REFUTED; the residual is a DCAlign *inference* failure.** The
  sweep ran on Midway (a shard OOM broke the `afterok` gather, recovered with
  `run_dcalign_gather.py --allow-missing`; common fully-scored set = 286 home pairs, 226 recover +
  60 control). Canary intact: pc0.001 reproduces iter-002 (recover 226/226 worse, control 0/60
  worse). Raising `pcount` **does not recover the failing pairs** (fraction still worse 1.00 → 0.92
  across 0.001→0.5; ≤19/226 ever recovered, 1–2/36 naturals; recover median ΔE *rises* 10.6 → 36
  a.u.) and **progressively destroys the good pairs** (control fraction-worse 0.00 → 0.88, median ΔE
  0 → 49). Recommended `pcount` = 0.001 (change nothing). **Why the §10.15 prediction was wrong:**
  it assumed DCAlign returns the *argmin* of Potts+prior, so flattening the prior would let the
  lower-Potts native frame win. But DCAlign is **approximate BP + decimation**, not exact
  optimization — even with a near-flat prior it cannot reach the native frame, so the bottleneck is
  the search/convergence, and the prior was *regularizing* it (flattening hurts). This refutes
  **both** proposed levers (μint/μext §10.14 *and* pcount §10.15): the worse-than-native residual is
  DCAlign's inference failing to find the minimum for these sequences, not a prior/gap-penalty bias.
  Next options: **(A)** per-sequence `min{E_dcalign, E_native/MAP}` for home pairs — we already hold
  the witness frame, so this eliminates the home-pair residual by construction (the cross-model term
  still rides on DCAlign); **(B)** improve DCAlign's inference on the failing set (more `maxiter`,
  slower annealing `Δβ↓`/`Δt↑`, damping, multi-seed per-sequence min). Artifacts:
  `combine/combine-CM-PPIC-dcalign-pcsweep/presweep_{rows.tsv,summary.json}` + `presweep.pdf`.

### 10.16 iter-003 Phase-B — the residual is BP basin-selection; multi-seed pre-screen (2026-06-29)

§10.15 left two options. **Option (A) (per-sequence `min{E_dcalign, E_native/MAP}`) is rejected** by
the user: the naturals here are only controls — the real goal is finding the optimal mapping when the
ground state is *unknown*, so a known-frame fallback doesn't serve it. We pursue **(B): fix DCAlign's
inference.** A source audit of the clone pins the mechanism and the right knobs:

- **The failure is basin-selection in BP, not decimation.** In `iterate_bplc.jl`, BP runs with
  β-annealing (`β=1.0` start, `+Δβ` every `Δt` sweeps; `palign` defaults `Δβ=0.05`, `Δt=10`), and
  `decimate_post` is a *one-shot greedy decode that fires only when BP's argmax violates ordering
  constraints*. The failing sequences "converge cleanly," so decimation never runs — the wrong frame
  comes entirely from **which basin BP's annealed marginals land in**.
- **Knobs that change the basin** (`palign` signature `src/palign_bplc.jl`): `seed` (random message
  init + sweep order + Λ noise; default 0 — **already plumbed** as `ScoringConfig.dcalign_seed`),
  annealing slowness `Δβ`↓/`Δt`↑ (not plumbed), `damp` (not plumbed). `maxiter`/`thP` change *when*
  BP stops, **not** the basin — so the user's "raising maxiter is the wrong move" is confirmed;
  decimation granularity is not tunable and doesn't engage here.
- **This pass tests `seed` only** (zero new code). The per-sequence **min ΔE over K seeds** *is* the
  production multi-seed-min, so a positive result both validates the fix and sizes its cost (the
  recovery-vs-K curve = smallest K needed); and the per-sequence **ΔE seed-spread** diagnoses the
  next step — high spread ⇒ seeds reach different basins (annealing likely helps too); near-zero ⇒ a
  seed-robust wrong attractor (multi-seed refuted ⇒ annealing/`Δβ` is next, and needs plumbing).
- **Tooling.** `scripts/pcount_presweep.py` → **`scripts/dcalign_presweep.py`** (and
  `utils_pcount_presweep_plot` → `utils_dcalign_presweep_plot`), generalized to sweep **any**
  `scoring.*` knob: `build --scoring-key <k> --tag-prefix <p> --values …` writes one run dir per
  value with `scoring.<k>` overridden (coerced to the field's native type) + a `sweep_meta.json` so
  `score` is self-describing; `--hardest` curates the **largest-ΔE** worse pairs (vs seeded random).
  `score --aggregate {none,min,auto}`: `none` = the per-value comparison (the pcount use case);
  `min` (auto for `dcalign_seed`) = the multi-seed-min recovery-vs-K curve, per-sequence seed-spread,
  plus the **seed-0 canary** (baseline reproduces the source iter's ΔE ≤5e-7). Back-compat verified: the
  archived pcsweep re-scores byte-identically. New logic covered in `tests/test_dcalign_presweep.py`.
- **The short test (built; awaiting Midway).** `combine/combine-CM-PPIC-dcalign-seedsweep/seed{0..5}/`
  — 24 curated iter-002 home pairs (**16 hardest-by-ΔE recover**, ~4 per model×{natural,synthetic},
  with 8 controls), seeds 0–5, all else pinned to iter-002 (`deltan`, pcount 0.001, maxiter 2000). Cost
  is ~200 s/seq ×2 models, so the subset is deliberately small (~N·50 s ≈ 20 min wall if the 6 seed
  arrays run concurrently). **Midway resources are set cluster-side** — `cpus=1` per task, fan out
  over the array (the iter-002-pcsweep OOM was `cpus=4`/`mem=8G`). Decision gate: min-over-6 recovers
  a meaningful fraction of the hardest ⇒ build the in-Julia multi-seed loop and run the full set at
  the smallest K the curve justifies; recovers ≈0 with low spread ⇒ multi-seed refuted, next is
  annealing.

### 10.17 iter-003 Phase-B — multi-seed REFUTED; warm-start fixed-point probe (2026-06-29)

**Multi-seed result (the §10.16 sweep ran on Midway): refuted.** `dcalign_presweep score
--aggregate min` over seeds 0–5 (canary intact — seed-0 reproduces iter-002 to 4.6e-9):
**0/16 worse pairs recovered** at K=6, identical to 0/16 at K=1. **14/16 are byte-identical
across all 6 seeds** (median ΔE seed-spread = 0.0); the 2 movers swing widely (max spread 46 a.u.)
but their best seed still sits ΔE≈7 a.u. above native — the seed shuffles BP between *wrong*
basins, never into native's. The seeds genuinely perturb the **initial condition** (not just sweep
order): source-confirmed `Random.seed!(seed)` at `palign_bplc.jl:16` → `rand`-initialised messages
in `initialize_all!` (`iterate_bplc.jl:57,65`), and Julia 1.12.6 ⇒ task-local RNG ⇒ each seed is a
distinct, reproducible random start. So 14/16 are robust to initialisation: **stable wrong BP fixed
points.** Multi-seed-min is not the lever.

**Why the offline objective diagnostic was abandoned.** The clean "evaluate DCAlign's objective at
native vs the DCAlign frame" idea assumed a decomposable `E_Potts + E_prior`. Source audit: DCAlign
has **no closed-form per-alignment prior cost** — the reported energy (`compute_potts_en`,
`utils.jl`) is *pure Potts*, and the insertion prior Λ enters **only** inside the BP messages
(`central!`), multiplied into **every coupling term** (loops `j ∈ 1:i-2`, `i+2:L`) — i.e. Λ is
**all-pairs**, O(L²)≈4000 terms for L≈90, not a light chain gap-penalty. The implied posterior is
`P(A) ∝ exp(−E_Potts(A))·Π Λ[i,j,Δn]`, so `Φ(A) = E_Potts − Σ_pairs log Λ` is well-defined — but
reconstructing it offline means summing ~4000 floored-histogram log-terms with no DCAlign function
to validate against (a small systematic per-term error flips the sign). Untrustworthy.

**The trustworthy test instead: a warm-start fixed-point probe.** Let DCAlign's *own* inference be
the oracle — initialise BP at the native frame, run its real schedule, and watch its dynamics:

- **Stays at native** (ΔE_warm ≤ tol) ⇒ native is a reachable fixed point the random-init runs
  missed → **case A** (search/init problem; anneal or native-biased init is the lever).
- **Drifts to the worse frame** ⇒ native is not a fixed point of DCAlign's objective → **case B**
  (the objective genuinely prefers it; search-tuning is futile — accept the residual / change method).

- **In-driver, clone untouched.** Feasibility-checked: every BP primitive (`Jh`/`PBF`/`Alg`/`Data`/
  `AllVar`, `onesweep!`, `compute_en`, `decodeposterior`, `decimate_post`, `check_solution`) is
  reachable via the `DCAlign.` prefix, and `onesweep!` is standalone. So the warm start lives
  entirely in our `src/SBM/julia/run_dcalign_warmstart.jl` — a faithful replica of `palign`+`update!`
  with the random `initialize_all!` swapped for a native-frame delta init. The clone is a **pinned,
  read-only dependency** (commit `cab443ffad133e6e68eff8e50b11e8fc59178dbd`,
  `infernet-h2020/DCAlign`) on Mac and Midway alike — no fork, no Mac↔Midway divergence (the repo-mgmt
  question the user raised).
- **Tooling.** `scripts/build_dcalign_warmstart.py` stages a self-contained probe run dir (per-model
  binaries + `seed.ins` + raw queries + length-L `native.fasta`); `SBM.energy.dcalign_warmstart`
  (`analyze_warmstart_record`/`summarize_warmstart`, case-A/B verdict) +
  `scripts/analyze_dcalign_warmstart.py` + `utils_dcalign_warmstart_plot` read the pulled cache and
  compare warm-start vs native vs the iter-002 random-init frame. Tests:
  `tests/test_dcalign_warmstart.py` (62 passed overall). Validated locally: the `(x,n)` encode→decode
  round-trips exactly (5 gap patterns), `warmstart_one` runs on the real CM model (flat Λ), and the
  analysis canary — ΔE vs the random-init frame — reproduces the iter-002 residual (41.809) exactly.
- **Built (awaiting Midway).** `combine/combine-CM-PPIC-dcalign-warmstart/{CM-bm-dense,PPIC-dense}/`
  — the **same 24 home pairs** as the seed sweep (16 worse + 8 controls), `deltan`/pcount 0.001/
  maxiter 2000. The `deltan` Λ needs GZip (macOS-broken) so the probe runs on Midway; the deltan
  branch is byte-identical to the production driver's. Resources are cluster-side (`cpus=1`/task).
  Controls are a probe sanity check — warm-started at native they must stay; a `control_drift` means
  the probe, not the science, is suspect. Decision gate: most worse pairs **stay** ⇒ case A (build
  the annealing/native-init lever); most **drift** ⇒ case B (DCAlign's objective can't be made to
  prefer native by search-tuning — stop tuning DCAlign).

### 10.18 iter-003 Phase-B — CASE A confirmed; fields-MAP init refuted; anneal-from-hot sweep (2026-06-29)

**Warm-start probe RESULT: CASE A.** Controls 8/8 stayed (0 drift — probe sound). Of the 16 worse
pairs, **9 STAYED at native** (ΔE≤1) and 5 more drifted only slightly (e.g. CM-syn-140 7.6 vs 50 from
random init; PPIC-215 2.4 vs 30) — **median ΔE collapsed ~30 → 0.32 a.u.** when BP starts at native.
Only 2/16 snapped back to the production frame. So the native basin **is reachable and (near-)stable**;
the production random-init search just never lands in it. There is a small (~1–2/16) genuine case-B
tail — don't expect a full 16/16 from any lever.

**Fields-MAP init (Test 1) REFUTED — and disqualified.** `--init map` (Viterbi frame from `hmm.py`,
production-legal, all 24 insert-free so usable directly) recovered only **1/16**, median ΔE **25.4**
(≈ unchanged from production). Worse, it **regressed good cases**: PPIC-269 went 3.1 → 13. Why: the
fields-MAP frame *ignores couplings*, which is the exact signal that distinguishes native from the
wrong frame — so for ~6 pairs the MAP frame **is** the production wrong frame (CM276's MAP frame
recomputes to the production energy −253.26). A couplings-blind init can't escape the couplings-blind
basin, and a lever that can regress good pairs is unusable in production. Confirms the prediction.

**Anneal-from-hot (Test 2) — ran; did NOT close it, but the failure is diagnostic.** Key schedule fact from
the clone: DCAlign's `update!` starts at `β=1.0` and only ever *increases* β (sharpening) — it
**never anneals from a hot, smooth landscape**, so random init commits to a basin at the physical
temperature and sharpening locks it in. The fix is to start at `β₀ < 1` (smoothed, fewer basins),
equilibrate, then ramp β up to 1 — letting the **full couplings** guide the trajectory into the
native basin (which §10.18's warm start proved is reachable). `palign` can't do this, but our
warm-start driver already replicates the schedule, so a third init mode was added there (clone still
untouched):

- **Driver** `run_dcalign_warmstart.jl` generalized: `run_bp!(init_margs, beta0, …)` — `init_margs
  === nothing` ⇒ stock random init; `β` starts at `beta0` (scaling `J,h,Λ`) and ramps up by `Δβ`,
  with convergence **only accepted once `β ≥ 1`** (never decode above the physical temperature). With
  a warm init and `beta0=1.0` it's byte-identical to the §10.17 probe. RNG is now seeded
  (`Random.seed!(seed)`, mirroring `palign`) so the random-init anneal is reproducible — smoke-tested
  (same seed → identical frame).
- **Build/analysis** `build_dcalign_warmstart.py --init random --beta0-values …` writes one
  self-contained run dir per β₀ (`beta<v>/`, no init file) + `sweep_meta.json`;
  `analyze_dcalign_warmstart.py --sweep-root …` tabulates recovery-vs-β₀, picks the best β₀, and
  emits `annealsweep_summary.json` + `annealsweep.pdf` (`render_annealsweep`). `--init-kind random`
  sets the verdict wording. 15 warm-start tests pass; the random+anneal path validated on the real
  CM model (flat Λ): all 12 seqs converge, iter counts show the hot-then-cool behaviour (141 easy,
  422–625 hard).
- **Built + ran:** `combine/combine-CM-PPIC-dcalign-annealsweep/beta{0.1,0.3,0.5,1.0}/` — same 24
  home pairs, `deltan`/pcount 0.001, `maxiter 8000`, β₀=1.0 = in-sweep canary (= current behaviour).
  One 2-task sbatch array per β₀ dir, all concurrent on Midway.

**Anneal-from-hot RESULT (`annealsweep_summary.json`): 0/16 recovered at every β₀** (0.1, 0.3, 0.5,
1.0); controls clean. But the per-sequence detail (`scripts/.../anneal_summary`) is the most
informative result of the whole phase — the 16 split into **two populations**:

- **5 "soft" movers** — annealing pulls them most of the way to native but not across the 1 a.u.
  line: PPIC-17 53→**7.3**, CM-syn-140 50→**7.6**, PPIC-176 22→**7.7**, PPIC-150 36→**13**, PPIC-112
  51→**19** (best ΔE over the four β₀). Lower β₀ helps these monotonically (sweep median 35→27.9).
- **10–11 "hard" cases are byte-identical to production at every β₀** (ΔE unchanged, 27–45 a.u.) —
  the β₀=0.1 smoothed landscape does not perturb them *at all*.

**The decisive cross-tabulation (the real finding): the hard-immovable set ≈ the sequences that
STAYED at native under native-init.** CM276, CM90, CM289, CM186, syn-27, PPIC-83, PPIC-232 each sat
at ΔE≈0 from a native start (native is a *stable fixed point* for them) yet sit at ΔE 27–44 here,
identical to production, at every β₀. That is a **deep-but-narrow basin** signature: native is a
stable attractor with a *small basin of attraction*; only an init already inside it converges there,
while annealing from a random/smooth start falls into the **wide wrong basin** every time — and
temperature cannot find a narrow basin (it is entropically disfavoured at every T in [0.1, 1]). This
is *not* "the residual is inevitable": native is provably reachable (9/16 stayed). It says the lever
is **initialisation into the native basin**, not temperature — and the init must be **couplings-aware**
(fields-MAP failed precisely because it is couplings-blind). See §10.19 for the threads this opens.

### 10.19 iter-003 Phase-B — state of play + threads for tomorrow (2026-06-29 EOD)

Working hypothesis going in: **native is a deep-but-narrow BP basin.** It is provably reachable
(native-init: 9/16 stay, median ΔE 30→0.32), but no production-legal *search* we have tried lands in
it, because its basin of attraction is small. This is a tractable problem — find a couplings-aware
initialisation that lands inside the native basin — **not** an inevitable residual.

**What we tried this phase and what each ruled out (all on the same 24 curated home pairs = 16
hardest worse-than-native by ΔE + 8 controls; tooling below):**

| Lever | Knob | Result | What it rules out |
|---|---|---|---|
| pcount sweep (§10.15) | prior flattening | ≤8% recover, breaks controls | prior strength is not it |
| μint/μext (§10.14) | gap-count penalties | provably neutral on the prior_only majority | gap *count* is not it |
| multi-seed (§10.16/§10.17) | random init seed 0–5 | 0/16; 14/16 identical across 6 seeds | reseeding (same basin distribution) is not it |
| fields-MAP init (§10.18) | Viterbi warm start | 1/16, *regressed* PPIC-269 3.1→13 | couplings-**blind** init is not it (lands in the wrong basin) |
| anneal-from-hot (§10.18) | β₀ ∈ {0.1,0.3,0.5,1.0} | 0/16; 5 "soft" 50→7, 10 "hard" unmoved | temperature alone cannot find a narrow basin |
| native init (§10.17/§10.18) | warm start at ground truth | **9/16 stay**, median ΔE→0.32 | *existence proof* — the basin is real and reachable |

**The two populations (use these for targeted tests, don't re-run the whole 16):**

- **Soft movers (5):** PPIC-17, CM-syn-140, PPIC-176, PPIC-150, PPIC-112 — annealing already pulls
  these to ΔE 7–19 (from 22–53). They are *close*; a better-tuned schedule may push them across.
- **Hard immovable (≈10):** CM276, CM90, CM289, CM186, CM-syn-27, CM-syn-186, CM-syn-141, PPIC-215,
  PPIC-83, PPIC-232 — identical ΔE at every β₀ AND (for the CM ones + PPIC-83/232) native-stable. The
  narrow-basin core. These need an init *inside* the basin, not a hotter start.

**Threads for tomorrow, ordered by how directly they test the narrow-basin hypothesis:**

1. **Measure the basin width (cheapest, most diagnostic).** Warm-start at *perturbed* native — native
   with `k` randomly reassigned match/gap columns, `k ∈ {1,2,4,8}` — and see at what `k` BP stops
   returning to native. This quantifies "how close must an init be," i.e. the basin radius, and tells
   us whether *any* heuristic init has a realistic shot. Reuses the warm-start driver verbatim
   (stage perturbed frames as `init.fasta`); no new Julia. **Do this first.**
2. **Couplings-aware cheap init (the production lever if #1 says the basin isn't microscopic).**
   Produce a near-native frame *without* ground truth using the **full Potts energy** (the signal
   fields-MAP lacks): e.g. simulated annealing / greedy descent on `E_potts(frame)` over the
   alignment DOF (gap placement), or an in-frame Potts argmax pass, then warm-start BP at that frame.
   If the Potts-energy landscape over frames has native as its min (it does, by construction — native
   *is* the lower-energy frame), a Potts-only optimiser should approach it, and BP then refines. This
   is the most promising "get into the basin" route.
3. **Anneal the soft movers harder (cheap, bounded upside).** For the 5 soft movers only: slower ramp
   (smaller `Δβ`, larger `Δt` — currently 0.05/10, an ~180-sweep ramp from β₀=0.1) and longer
   equilibration *at* β₀ before ramping (right now only `Δt`=10 sweeps at the start temperature). Add
   `Δβ`/`Δt` to `meta.json` (the driver already reads them) — small change. Tests whether the soft
   tail crosses with a gentler schedule; will *not* move the hard core.
4. **Hybrid warm-start + anneal.** Warm-start at the couplings-aware init from #2 *and* run a short
   β₀<1→1 ramp from there, so BP both starts near native and is given room to settle — combines the
   two levers that each half-worked.

**Open question worth resolving early:** for the ~10 hard cases, is the wrong frame ever actually
*lower* in DCAlign's full objective (Potts + Σlog Λ) than native? The §10.17 audit showed we can't
reconstruct that objective reliably offline, but we *can* read it off DCAlign directly: warm-start at
native, run **0 sweeps**, and record `compute_en` (the free energy incl. the Λ term) at β=1; compare
to the same at the wrong frame. If native's free energy is lower, the wrong frame is purely a basin
artefact (search problem, #1–#4 apply). If higher, those specific pairs are genuine objective
disagreements (the prior really prefers the wrong frame there) and need a prior fix, not a search fix.
This cleanly partitions the hard set into "search" vs "objective" and should be run alongside #1.

**Tooling state (all built, tested, committed-ready — nothing committed yet):**

- `src/SBM/julia/run_dcalign_warmstart.jl` — `run_bp!(init_margs, beta0, …)`: random **or**
  warm-start init, β₀-anneal ramp, convergence accepted only at β≥1, RNG-seeded. Reachable-only
  `DCAlign.` calls; **clone untouched + pinned** `cab443ffad133e6e68eff8e50b11e8fc59178dbd`.
- `scripts/build_dcalign_warmstart.py` — `--init {native,map,random}`, `--beta0`, `--beta0-values`
  (sweep → one `beta<v>/` dir each + `sweep_meta.json`); stages self-contained per-model in-dirs.
- `scripts/analyze_dcalign_warmstart.py` — `--run-dir` (single) or `--sweep-root` (recovery-vs-β₀
  table + best-β₀ + verdict); `--init-kind {native,map,random}` for wording.
- `src/SBM/energy/dcalign_warmstart.py` (+ `tests/test_dcalign_warmstart.py`, 15 tests) — three
  energies/pair (native, warm-start, production random-init), case-A/B classifier, sweep verdict.
- `src/SBM/utils/utils_dcalign_warmstart_plot.py` — `render_warmstart` + `render_annealsweep`.
- `pipeline/external/sbatch_dcalign_warmstart.sh` — 2-task array (one per model), enforces the clone
  pin, cpus=4/mem=12G/time=2h. Run dirs: `combine/combine-CM-PPIC-dcalign-{warmstart,mapinit,annealsweep}/`.
- **macOS constraint (unchanged):** the `deltan` Λ needs GZip (broken on macOS), so every real run is
  Midway-side; the Mac validates mechanics with flat Λ. Pull caches with `sync_models.sh`, analyse
  locally.

### 10.20 iter-003 Phase-C — the residual is a DCAlign search failure; direct Potts-energy minimization solves it (2026-06-30, Mac)

§10.19's working hypothesis (deep-but-narrow native basin; the lever is a *couplings-aware* init,
not temperature) is **confirmed and made constructive.** The §10.18 fields-MAP init failed because it
is couplings-*blind*; the fix is to minimize the **full Potts energy** over the alignment directly.
For an insert-free home pair the alignment is just a choice of which `g = L − N` columns are gaps —
`C(L, g)` monotone frames — and for the worse-than-native residual `g` is *small*, so this is often
**exactly enumerable**. New, tested, all-numpy module `src/SBM/energy/potts_align.py`
(`enumerate_align` = exact global Potts-frame argmin via batched `potts_energies`; `sa_align` =
multi-restart simulated annealing with an incremental two-column ΔE for the high-gap cases, warm-
started from production-legal heuristic frames; `potts_align` dispatches enumerate-else-SA;
`perturb_frame` for the basin-width probe). Tests `tests/test_potts_align.py` anchor SA against
exact enumeration on a tiny model and check the incremental ΔE against a from-scratch recompute
(the brute-force-oracle philosophy of `test_energy.py`).

- **Result on the 24 curated home pairs (`scripts/analyze_potts_align.py`,
  `combine/combine-CM-PPIC-dcalign-pottsalign/`):** **13/16 worse pairs reach native-or-better with
  no DCAlign at all; controls 8/8.** The **11 enumerable cases (≤3 gaps, including all 8 PPIC pairs)
  recover EXACTLY** — native is the *provable* global Potts minimum (ΔE_best ≈ 0), and for PPIC-176 a
  frame strictly **beats** native (ΔE −1.24, rigorous: the MSA "native" frame is not always the global
  Potts min). The 1-gap PPIC pairs are a **91-frame** search that DCAlign's BP placed ΔE 22–53 above
  the 8 ms enumerated optimum — i.e. **the residual on the tractable cases is purely a BP search
  failure**, not an objective/prior bias (refuting the §10.14/§10.15 prior-tuning framing for good).
- **The 3 stragglers** (CM-186, CM-289, CM-syn-186 — the 9/13/14-gap cases, `C(L,g)` up to ~10¹⁵)
  plateau at ΔE 5–17 above native under warm-started single-move SA, but **all beat DCAlign's iter-002
  frame** (27–44). These are the genuine hard combinatorial core (the §10.19 "hard immovable" set,
  which §10.18 also showed is native-*stable*). Warm-starting SA from the fields-MAP and DCAlign frames
  (vs random) was decisive: it took the high-gap median from worse-than-DCAlign to better-than-DCAlign
  and recovered two cases (CM-syn-140, CM-syn-141) exactly.
- **Decision (user):** the production aligner returns the **global Potts minimum** (lowest-energy
  frame, regardless of whether it matches the MSA "native" frame); the rare beats-native cases are
  surfaced as a diagnostic. This is the fully-unsupervised "find the right frame for an unknown
  sequence" criterion. iter-003's tuning question is **answered**: drop the DCAlign prior/search debate
  — minimize the Potts energy over gap placements (exact when feasible, warm-started SA otherwise).
- **Confirmatory Midway batch (built + staged, awaiting the cluster run):**
  `combine/combine-CM-PPIC-dcalign-pottsinit/` (`scripts/build_potts_align_warmstart.py` +
  `MIDWAY_RUN.md`). **M1** (`sa-beta{1,0.5}`) warm-starts DCAlign-BP at the Potts-align frame (pure
  refine + hybrid anneal) — does BP close the last 3, now that the init is ΔE 5–17 (vs DCAlign's 27–44)
  from native? **M3** (`perturb-k{1,2,4,8}`) is the basin-width probe (native ± k reassigned columns).
  **M4** (`diag-{native,dcalign}`) reads DCAlign's 0-sweep `compute_en` at each frame via a new scalar
  `n_diag_sweeps` flag in `run_dcalign_warmstart.jl` — **note (§10.17 confirmed):** `compute_en` is the
  *Potts* energy of the argmax-decoded frame, **not** a Λ-inclusive free energy, so the literal
  "free-energy split" is not available; this is a gauge cross-check, validated on the Mac with flat Λ to
  match numpy `potts_energy` at **8e-13**. The DCAlign clone stays pinned + unmodified
  (`cab443ffad133e6e68eff8e50b11e8fc59178dbd`). Analyze the pulled caches with
  `scripts/analyze_potts_init_batch.py`.
- **Why this matters beyond the residual:** the result reframes the whole combine alignment step. The
  couplings-aware aligner that the campaign needed is a **direct Potts-energy minimizer**, not DCAlign's
  BP+prior — which is ~700× slower and demonstrably fails near-trivial searches. For home/low-gap
  sequences the minimizer is exact and all-Mac (no Julia, no cluster). The open generalization is
  sequences with insertions (`N > L`), where the gap-placement enumeration does not directly apply and
  DCAlign's insertion machinery (or an extended SA move set) is still the candidate.
