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
  `finalize_dcalign_push.sh` (sacct-validate, compress, opt-in `--push`). Entrypoints
  `scripts/wf/run_dcalign_shard.py` (`plan`/`run`, round-robin shards, resume-skip) +
  `run_dcalign_gather.py`. Account/partition `pi-ranganathanr`/`caslake`.
- **Validation:** Tier-0 (handoff round-trip, branch = in-frame, cache I/O, config bounds) and
  Tier-1 (end-to-end vs the real DCAlign clone: energy transfer ≤5e-7 — observed ~1e-15) pass;
  a synthetic full-flow run (`plan`→4 shards→gather→`score --method dcalign`) gives
  `dcalign_agreement` ~9e-16. **Tier-2** (real CM/PPIC models, actual sbatch submission) is
  pending the Git-LFS model handoff (Mac-side commit; `.gitattributes` + `.gitignore` prepared).
- **Midway env facts:** DCAlign clone at `../DCAlign` (commit `cab443f`); Julia 1.10.2 depot at
  `/scratch/midway3/nadavbg/julia_depot`; SBM in a uv `.venv` at the repo root. `module load julia`
  breaks `git` HTTPS (Julia's mbedTLS `libgit2` shadows system git) — scripts export
  `GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt`.
