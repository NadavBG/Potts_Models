# iter-003 runbook: score the combine query set with `potts_align` (cluster)

**For the agent doing the wiring (Midway-side).** This specifies, exactly, how to
wire and run iter-003 of `combine/combine-CM-PPIC-dcalign`. The goal is to
**evaluate the new gap-placement aligner** (`src/SBM/energy/potts_align.py`,
spec `docs/POTTS_ALIGN.md`) as the combine alignment method — **DCAlign is not used
at all** in iter-003.

## What iter-003 computes (the method semantics)

Score **every (query, model) pair** under the combine query set with `potts_align`,
**subject to `N ≤ L`** (the gap-placement aligner needs the query no longer than the
model frame):

- **Home term** (query under its origin model): the query is in that model's
  `L`-frame, so stripped it is always `N ≤ L` → always scored.
- **Cross term** (query under the *other* model): scored **iff** `N ≤ L_other`.
  PPIC→CM is always `N ≤ 96` (scored). CM→PPIC is scored only when `N_CM ≤ 91`
  (≈ ⅓ of CM queries; the rest have `N > 91`, need insertions, and are **skipped** —
  emit a row with `energy = NaN`, `note="N>L: insertions needed, out of scope"`).

For each scored pair return the **global Potts-energy minimum frame**: exact
enumeration when `C(L, N) ≤ enum_max_frames` (g ≤ 3), else g-adaptive parallel
tempering (`PTSchedule.for_gap_count(g)` — §6.7/§6.8 of `docs/POTTS_ALIGN.md`). No
DCAlign cache is read or written.

### Two query-set choices that cap the cost at ~100 SU (decided with the user)

- **Subsample the PPIC→CM cross block.** That block (every PPIC natural scored under
  the CM frame, all `g ≥ 5` → PT) is the *entire* cost driver (241 of 262 core-h). It
  is **subsampled to `N_cross_ppic = 8000` of 26 701** (a *seeded* random subset —
  log the seed; preserves the gap-count distribution, so the cluster pattern in the
  figure is unbiased). All **home** terms (both models, full) and the small cross
  CM→PPIC block are scored in full. ~8000 cross points is far more than enough to see
  where PPIC sequences cluster under CM in the `E_A`-vs-`E_B` scatter.
- **Add a random-sequence negative control.** Generate **`N_random = 500` random
  length-91 sequences** (residues 1..20 uniform i.i.d., *seeded* — log it) as a new
  query group `random/N91`. Score each under **both** models: under PPIC it is
  in-frame (`N = L = 91`, `g = 0`, instant); under CM it is `g = 5` cross (PT). These
  are a negative control — random (non-evolved) sequences should sit at *high* energy
  under both models, well separated from the naturals, validating the energy scale and
  the figure. They appear as their own colour in `two_model_energy.pdf` automatically
  (the renderer colours by `groups.json`).

## Cost (why the cluster) — ~100 SU with the choices above

Per-pair cost is set by the schedule, **not** by `g` (§6.8): ~instant for `g ≤ 3`,
~32 s for `g = 4..12`, ~80 s for `g ≥ 13`, at ~4.7·10⁵ moves/s/core (measured,
`L`-independent, so identical at `L = 94`). Breakdown (measured gap distributions):

| block | seqs | core-h |
| --- | --- | --- |
| home CM (under CM) | 1258 | 7.1 |
| home PPIC (under PPIC) | 26 701 | 12.3 |
| cross CM→PPIC (`N≤91` only) | 483 | 1.6 |
| **cross PPIC→CM (subsampled)** | **8000** of 26 701 | **72** |
| random N=91 control (both models) | 500 | 4.4 |
| **total** | | **≈ 97 SU** |

(The un-subsampled cross PPIC→CM would be ~241 core-h → ~262 SU total; the subsample
is what brings it to ~100.) Embarrassingly parallel over pairs → **~12 min on a
500-task array.** `query.cap_per_group` can shrink it further if wanted.

> **Measured cost correction (2026-06-30, Midway wiring).** The per-pair rate above
> (~32 s for a `g=4..12` PT pair) was the Mac estimate; a caslake core measured
> **~74 s** for a `g=5` PT pair (~2× slower). At that rate the fixed PT work (home
> CM/PPIC + CM→PPIC cross + random→CM) is ~53 core-h and each subsampled PPIC→CM
> pair costs ~0.02 core-h, so `pa_cross_subsample_n` was cut from 8000 to **2000**
> to hold the run near **~94 SU** (2000 cross points is still plenty to see the
> PPIC-under-CM cluster). The shipped config
> (`config/params_combine-CM-PPIC-potts.yaml`) uses `pa_cross_subsample_n: 2000`,
> `n_shards: 128` (≈0.75 h/shard, 3 h wall). Raise the subsample for a denser cross
> cloud at proportionally more SU.

## Mac-side code to add (portable; no Slurm) — implement these first

1. **`src/SBM/energy/score.py`** — add `"potts_align"` to `METHODS` and a branch:
   ```python
   if method == "potts_align":
       if seq.size > model.L:
           raise ValueError(f"potts_align needs N<=L; N={seq.size} > L={model.L}")
       res = potts_align(seq, model, seed=seed, sequence_id=...)  # seq = raw, gap-free
       return ScoreResult(energy=res.best_energy, method="potts_align",
                          model_name=model.name, gauge=model.gauge,
                          model_sha256=model.sha256,
                          representative_alignment=ints_to_seq(res.best_frame),
                          notes=f"global Potts-min; engine={res.method}; exact={res.is_global_exact}")
   ```
   `potts_align` **requires `seed`** (logged); thread the per-(query,model) seed in.
2. **`scripts/score_two_models.py`** — in `_score_one`, for `method=="potts_align"`:
   `raw = strip_gaps(record.ints)`; if `raw.size <= model.L` → `score_sequence(raw,
   model, method="potts_align", seed=<derived>)`; else emit the skipped NaN row. Derive
   per-(query,model) seeds from the master seed in stable `(id, model)` order (mirror
   the marginal seed derivation already there). **Do not** require/read a DCAlign cache.
3. **`src/SBM/combine_config.py`** — add `"potts_align"` to `_METHODS`. No new required
   fields (the g-adaptive `PTSchedule` is internal); optionally expose
   `scoring.pt_n_blocks` / `pt_teleport_frac` overrides if you want them in the config.
4. **`config/params_combine-CM-PPIC-potts.yaml`** — copy the dcalign config, set
   `scoring.method: potts_align`, keep `query.cap_per_group` (set per the cost decision),
   drop the dcalign-only keys.

Test locally on a handful of pairs before sharding: `score_two_models.py --method
potts_align` on 2–3 CM + PPIC sequences, assert home `E ≤` the DCAlign-frame energy
from iter-002 (the residual is gone) and that `N>L` cross pairs are skipped, not crashed.

## Cluster sharding (Midway) — mirror the DCAlign shard model, but pure Python

`potts_align` is **pure numpy** — no Julia, no DCAlign, no GZip, no `module load
julia`. Reuse the `.venv`. Mirror `scripts/wf/run_dcalign_shard.py` (plan/run) +
`run_dcalign_gather.py`, replacing the Julia call with `potts_align`:

- **Build the query set** (`plan`, on the login node): `assemble_query_records`
  (honoring `cap_per_group`), then **append the random control group** — `N_random=500`
  length-91 sequences from `np.random.default_rng(master_seed).integers(1, 21,
  size=(500, 91))`, each a `QueryRecord(id=f"random|N91|{i}", group="random/N91",
  origin_model="", ints=<row>)` (`origin_model=""` ⇒ no home term; both models are
  "cross", and both are `N=91 ≤ L`). Log the seed.
- **Unit of work = a (query_id, model_name) pair to score.** The in-scope predicate:
  *home* pair (`origin_model == model`, always `N ≤ L`) → score; *cross* pair
  (`N ≤ L_other`) → score **except** that the **PPIC→CM cross is restricted to a seeded
  8000-id subset** (`rng(master_seed, "ppic_cross").choice(ppic_ids, 8000,
  replace=False)`); `N > L_other` → skip (NaN row). `plan` enumerates the in-scope
  pairs in **sorted, stable** order, round-robins into `n_shards`, writes
  `<run_root>/potts_align/shards_manifest.json` (record the cross-subsample id list +
  seed there, so the run is reproducible and the gather can assert coverage).
- **Slurm array** (`sbatch --array=0-<n_shards-1>%<conc>`): each task loads both
  models once, scores its pair-slice with `potts_align`, **flushes per row** to
  `<run_root>/potts_align/cache/shard_NNN.tsv` (resume contract: skip ids already
  present — same as the dcalign shard). Columns: `query_id, model, n_residues, gaps,
  energy, engine(enumerate|pt), is_global_exact, frame, seed`.
- **Resources**: `cpus-per-task=1` (PT is a Python loop; one core per task, fan out
  over the array — do **not** ask for many cpus), `mem=2G`, `time` = (pairs-per-shard
  × ~80 s worst-case) + margin. Size `n_shards` so each task is ~1–2 h (≈ 500 shards
  for the uncapped set). No `afterok` GZip/OOM hazards (the dcalign deltan-prior OOM
  does not apply — there is no seed.ins).
- **`gather`**: merge `shard_*.tsv` → `<run_root>/data/scores.tsv` (tidy long form,
  same schema the render step reads) + `scores_detail.json` + a provenance manifest
  (git commit, master seed, per-shard sha256s, `potts_align` schedule, package
  versions). Then run the existing `render_combine` for the `E_A` vs `E_B` figure.

## Reproducibility + validation gates

- Every `potts_align` call takes an explicit per-(query,model) seed derived from the
  one master `seed` in stable order — record the master seed and the derivation in the
  manifest (hard project rule).
- **Canary**: re-score any one home pair locally on the Mac with the same seed; the
  energy must match the cluster shard exactly (pure numpy → bit-reproducible per seed,
  independent of core count, unlike the OMP-sensitive MCMC kernels).
- **Result gate**: home-term `ΔE = E_potts − E_native ≤ 0` for ~all queries (the
  worse-than-native residual is gone — that is the iter-003 point); flag any home pair
  with `ΔE > 1` (the hardest `g≥13` tail may leave a few a.u. — expected, §6.8).
- **Figure** (`render_combine` → `figs/two_model_energy.pdf`, same as iter-002): the
  `E_A` (CM) vs `E_B` (PPIC) scatter with marginals. Expect the **naturals to cluster
  low** under their home model and the **random N=91 control to sit clearly high**
  under both — if the random cloud overlaps the naturals, something is wrong (wrong
  gauge, or PT not converging). This control is the iter-003 sanity check on the energy
  scale.
- **No DCAlign anywhere** in the iter-003 outputs (the whole point is to evaluate the
  new method standalone).

## Hand-off summary for Midway Claude

1. Implement the 4 Mac-side pieces above; smoke-test `--method potts_align` on a few
   pairs in the `.venv`.
2. Write `run_potts_align_shard.py` (plan/run) + `run_potts_align_gather.py` (pure
   Python, mirror the dcalign wrappers; no Julia) and an `sbatch_potts_align_shard.sh`
   (cpus=1, mem=2G; `--array` sized to `n_shards`).
3. `python scripts/iter.py run combine-CM-PPIC-potts "potts-align-eval" --snakefile
   Snakefile.combine` (after adding the `potts_align` align+score rules), or drive the
   shard/gather scripts directly.
4. Confirm the validation gates; pull `data/scores.tsv` to the Mac for analysis.
