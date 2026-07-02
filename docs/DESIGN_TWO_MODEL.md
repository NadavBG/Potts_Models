# Two-model sequence design by joint simulated annealing

Search for sequences that are simultaneously low-energy under two Potts models
(CM, `L=96`, and PPIC, `L=91`) by annealing over `E_tot = w_A·E_A + w_B·E_B`.
This is Next-step #2 of `docs/two_model_progress.md`; it builds on the calibrated
weights (`src/SBM/utils/energy_weights.py`) and the couplings-aware aligner
(`docs/POTTS_ALIGN.md`).

- Engine: `src/SBM/design/anneal.py`
- CLI: `scripts/design_two_model.py` (runs end-to-end on the Mac)
- **Pipeline stage:** a `design:` block in a `combine` config (`src/SBM/combine_config.py`
  `DesignConfig`) drives it as part of `Snakefile.combine` — the recommended entry point
  (see "Run it via the combine pipeline" below).
- Figures: `src/SBM/utils/utils_design_plot.py` + `scripts/render_design.py`
- Snakemake wrappers: `scripts/wf/run_design_{config,local,render,handoff}.py`
- Cluster wrappers: `scripts/wf/run_design_{shard,gather}.py`
- Tests: `tests/test_design_two_model.py`

## Why joint annealing (the key design choice)

`E_A`/`E_B` are defined as the *argmin over gap placements* (`potts_align`). A
length-91 query against CM is a search over `C(96, 5) ≈ 6.1e7` frames → parallel
tempering, ~8 s per model per candidate (`POTTS_ALIGN.md §6.8`). Re-aligning at
every Metropolis step (~16 s/step) makes a real anneal infeasible, and its cost
scales with the (run-dependent) gap count — so the per-run SU is unpredictable.

**Resolution:** fold the alignment *into* the Monte Carlo. The state is

    (core sequence x of length N ≤ min(L_A,L_B)=91,
     gap placement of x in the CM frame,
     gap placement of x in the PPIC frame)

and every proposal is an O(L) incremental energy update — **~10–15 µs/step** on the
real CM/PPIC models (the bare incremental kernel is ~9 µs; RNG + move dispatch bring
the end-to-end step to ~15 µs), **independent of gap count**. A 500k-step chain is
~5–8 s; 96 chains run in ~2 min of anneal on the 8-core Mac (the polish adds a few
more; a full 96-chain `fast`-polish run measured ~5 min). SU is predictable and tiny.
The `execution: auto` estimator uses 15 µs/step.

**Honest caveat.** At finite T the alignment degrees of freedom sample a *thermal
free-energy over frames*, not the exact argmin; the two coincide as T→0.1
(`beta_end`). This is an annealing **optimizer** (the insert/delete moves are
birth/death moves, not corrected for exact reversible-jump detailed balance). The
authoritative per-model energies come from a final real `potts_align` **polish**
on each finished sequence, warm-started from the joint-MC frame so it can only
match-or-improve it (a cold high-gap PT search over `C(L,g)` frames often lands
*worse* than the annealed frame — verified, and the reason the polish is
warm-started).

## The engine

State invariant (test-enforced): `frame_X[occ_X] == x` in order, `occ_X` strictly
monotone, `N = len(x) ≤ 91`. Five moves, each O(L) per frame, accepted by the
Metropolis rule `de<=0 or rand < exp(-beta·de)`:

| Move | What changes | ΔE |
|---|---|---|
| **substitute** | one core residue `a→b` (both frames, at `occ_A[k]`/`occ_B[k]`) | `w_A·ΔE_A + w_B·ΔE_B` |
| **slide A / slide B** | move a residue into an adjacent/teleported gap column in *one* frame (the alignment DOF) | reuses `potts_align._try_move` at `beta·w_X` |
| **insert** | add a core residue, filling a gap in *each* frame; `N→N+1` | `w_A·ΔE_A + w_B·ΔE_B` |
| **delete** | remove a core residue, vacating its column in both frames; `N→N-1` | `w_A·ΔE_A + w_B·ΔE_B` |

Substitute/insert/delete share one primitive — `_sub_delta`, "change column `c`
from state `s0` to `s1`" (delete = `→GAP`, insert = `GAP→`) — which mirrors
`potts_align._move_delta` (diagonal self-term carried to match `compute_energies`).

- **Length cap.** Insertion needs a free gap in *both* frames; at `N=91` the
  shorter (PPIC) frame is gap-free, so insertion is always rejected → `N` never
  exceeds 91. Deletions are floored at `schedule.min_length` (default 70).
- **Schedule.** `AnnealSchedule`: geometric `beta` from `beta_start=1.0` to
  `beta_end=10.0` (T: 1→0.1) over `n_steps` (default 500k); move mixture default
  sub 0.70 / slide_A 0.10 / slide_B 0.10 / insert 0.05 / delete 0.05;
  `teleport_frac=0.3`; `record_every=1000`.
- **Start mix (Pareto coverage).** Chains are seeded from three kinds of start,
  counted independently: `start_random` (a random sequence, as before),
  `start_natural_a` (a **model-A / CM natural**), `start_natural_b` (a **model-B /
  PPIC natural**); `n_chains` is their sum. Natural starts come from each model's
  `seed_msa` (in `models.json`), filtered to core length `≤ min(L_A,L_B)=91` so they
  fit both frames — a home-family natural places its core at the native columns in the
  home frame (home energy starts native) and left-packs it in the other (relaxed by the
  slide moves; `initial_state_from_frame`). The selection is *seeded* and pinned in
  `design_config.json`, so local and cluster runs reproduce identical starts. Each
  chain records its `start_type`, and the figures **color trajectories by it** — random
  black, CM natural orange, PPIC natural blue (coherent with the family clouds). This
  seeds both native basins so the Pareto front is mapped from both ends, not just from
  random sequences.
- **Seeding.** Chain `i` uses `master_seed + i` (`np.random.default_rng`), blocked by
  start type (random, then natural_A, then natural_B), so the type is a pure function of
  the index. Refuses to run unseeded; same seed ⇒ bit-identical trajectory.
- **Drift canary.** Chain-end running `E_A`/`E_B` re-verified against a from-scratch
  `potts_energy(frame)` (`abs_tol 1e-6`), fall back to exact on mismatch — same as
  `potts_align`.

## Outputs

Under `<run_root>/design/` (a stable path, overwritten in place):

- `trajectories.npz` — shared `steps`, `temperatures`; per-chain `(n_chains, R)`
  arrays `E_tot`, `E_A`, `E_B`, `n_residues`; per-chain `start_type`, `seed`, and
  finals (`final_E_*_mc`, `final_E_*_polish`, `final_n_residues`).
- `designed_sequences.fasta` — one core (ungapped) design per chain, energies +
  `start=` in the header.
- `designed.tsv` — tidy: `chain, seed, start_type, N_final, E_{A,B,tot}_mc,
  E_{A,B,tot}_polish, polish_exact_{A,B}, accept_rate, model_A, model_B`.
- `design_manifest.json` — git/seeds/weights/start-mix/model-sha256s/schedule/summary
  via `SBM.provenance`; `design_config.json` — the self-contained cluster contract
  (incl. the pinned natural-start rows); `command.sh`.
- `design_aln_A.fasta` / `design_aln_B.fasta` — each design threaded into model A's
  (CM, `L=96`) and model B's (PPIC, `L=91`) frame with gaps — the **polish argmin
  alignment**. Equal-length gapped records = a valid MSA per model, so they drop
  straight into an alignment viewer (e.g. `alnviz`) with **ZAPPO** coloring. Headers
  carry `model=`, `start=`, `N=`, `E=`.

The four figures land in **`<run_root>/figs/`** (beside `two_model_energy.pdf`), via
`render_design.py` (the pipeline calls it with `--figs-dir <run_root>/figs`).

### Figures

1. **`design_trajectories.pdf`** — `E_tot` vs step for every chain (colored by start
   type, best chain in `reference` red), with the shared annealing temperature strip above.
2. **`design_phase_space.pdf`** — (A) the trajectories as arrowed paths in the
   `E_A`–`E_B` plane over the natural clouds (from the combine `data/scores.tsv`),
   colored by start type, with the Pareto front of the final designs marked
   (`reference` red) and an `E_tot` iso-line through the best; (B) a heatmap of where
   the cold-phase states land (the basins).
3. **`design_lengths.pdf`** — histogram of the final design length `N`, stacked by
   start type.
4. **`design_alignment.pdf`** — the designs in both model frames (side-by-side panels,
   CM and PPIC), **ZAPPO-colored** cells with residue letters, a left strip colored by
   start type, and a column ruler. This is a static overview (the `design_aln_{A,B}.fasta`
   are the interactive-viewer input). `--no-letters` renders color-only (faster).

All layouts are computed from `rcParams` inch budgets and routed through
`scripts/lab_plotting.py` (ZAPPO uses the standard Jalview scheme, a domain convention).

## Run it via the combine pipeline (recommended)

Design is a gated stage of `Snakefile.combine`: add a `design:` block to the combine
config (validated by `DesignConfig`) and it runs after `compute_weights`, landing
outputs at stable paths (no timestamped scratch dir). Example (`config/params_combine-CM-PPIC-potts.yaml`):

```yaml
design:
  enabled: true
  start_random: 48         # random-sequence starts
  start_natural_a: 24      # CM (model A) natural starts   } seed both native basins
  start_natural_b: 24      # PPIC (model B) natural starts  } → wider Pareto front
  steps: 500000
  seed: 0
  min_length: 70           # deletion floor on N (≤ min L = 91)
  polish: true
  polish_schedule: fast    # warm-started; keeps a run to minutes (see Cost / SU)
  execution: auto          # auto | local | cluster (see below)
  local_budget_minutes: 30
  n_shards: 64             # Slurm array size if it goes CLUSTER
```

```bash
snakemake -s Snakefile.combine --configfile config/params_combine-CM-PPIC-potts.yaml --cores 8 all
```

- Outputs land in `<run_root>/design/` (`trajectories.npz`, `designed.tsv`,
  `designed_sequences.fasta`, `design_manifest.json`, `design_config.json`) and the two
  figures in **`<run_root>/figs/`** (beside `two_model_energy.pdf`).
- The stage prints its cost estimate at DAG build (`[design] estimated N min … → LOCAL/CLUSTER`).

### `execution`: run here or on Midway

The anneal is always Mac-cheap; the final warm-started `potts_align` polish is the only
variable cost (high-gap frames). One knob decides where the anneal runs:

- **`local`** — run the whole thing on the Mac now.
- **`cluster`** — write `design/design_config.json` + `design/MIDWAY_HANDOFF.txt` and
  **stop** (run the array on Midway, pull back, re-run to render — see the runbook below).
- **`auto`** (default) — the Snakefile predicts the local wall-time
  `n_chains × (steps·15µs + polish_per_chain) / --cores` and runs **local** iff it is
  `≤ local_budget_minutes`, else behaves as **cluster**. `polish_per_chain` is a
  conservative per-schedule constant (fast ≈ 6 s, default/auto ≈ 220 s, thorough ≈ 440 s).

So the default-size run (96 chains, `fast` polish ≈ 2 min predicted) runs on the Mac; you
only land on Midway by raising the chain count/steps or the polish depth past the budget.

## Run it directly (CLI, for one-off exploration)

```bash
# default 96 chains (48 random / 24 CM / 24 PPIC) × 500k steps; minutes on 8 cores
python scripts/design_two_model.py \
    --combine-run combine/combine-CM-PPIC-potts/iter-001-potts-align-eval \
    --start-random 48 --start-natural-a 24 --start-natural-b 24 \
    --steps 500000 --seed 0 --jobs 8
python scripts/render_design.py \
    --design-dir combine/combine-CM-PPIC-potts/iter-001-potts-align-eval/design

# fast smoke (all random, no polish)
python scripts/design_two_model.py --start-random 4 --start-natural-a 0 --start-natural-b 0 \
    --steps 5000 --seed 0 --no-polish
```

Models (`models[0]`=CM as A, `models[1]`=PPIC as B) and weights are read from the
combine run's `models.json` + `data/energy_weights.json`; override with `--model-a/-b`
+ `--w-a/-b`. Outputs land under `<combine_run>/design/` (a stable path, overwritten in
place; rides `sync_models.sh`'s `combine/` tree).

**Polish cost note.** A design that shrinks to the length floor gives the CM frame many
gaps (e.g. `N=70` ⇒ `g=26`), so a cold high-gap PT search would be minutes per chain — the
`default`/`thorough` schedules. The **default is `fast`**: because the polish is
warm-started from the joint-MC frame it can only match-or-improve it, so a light ladder
(~2.5 s/model) suffices for the routine run. Reserve `default`/`thorough` for a
high-quality polish at cluster scale. Use `--no-polish` for a quick look.

## Cost / SU

- Joint-MC: ~15 µs/step (real models) ⇒ a 500k-step chain ≈ 7.5 s (`1 SU ≈ 1
  core-hour`, `POTTS_ALIGN.md §4.1`). 96 chains ≈ 0.2 core-h ≈ **0.2 SU** for the anneal.
- **Polish is the variable, dominant cost** (not the anneal). Measured on a real
  N≈75 design (CM `g≈21`): `fast` ≈ **2.5 s/model** (~5.5 s/chain), but `default`/
  `thorough` are **minutes per chain** (a cold-scale PT over `C(96,75)≈10²⁰` frames).
  This is the whole reason a firstlook 48-chain `default`-polish run took ~18 min while
  the anneal alone was ~20 s. `fast` is the default; a 96-chain `fast` run is ~2 min on
  the Mac. Reserve `default`/`thorough` (cluster) for a high-quality final polish.
- Because per-step cost is gap-count-independent, a **fixed** step budget gives a
  **predictable** anneal SU; the `execution: auto` gate bounds the polish cost too, by
  comparing a per-schedule estimate to `local_budget_minutes`.

## Mac → Midway → Mac runbook (optional; only to scale up)

The default-size run is Mac-feasible, so Midway is *optional* — use it only to push
chains/steps far higher. The split mirrors the `potts_align` cluster path
(`POTTS_ALIGN.md §11`); chains are embarrassingly parallel, one Slurm array task
per shard, pure numpy (no Julia).

1. **Mac writes the run spec.** Either set `design.execution: cluster` (bump
   `start_*`/`steps`) and run the pipeline — it writes `design/design_config.json` +
   `design/MIDWAY_HANDOFF.txt` and stops — or emit the config directly:
   ```bash
   python scripts/design_two_model.py --emit-config-only \
       --combine-run <combine_run> \
       --start-random 1000 --start-natural-a 500 --start-natural-b 500 \
       --steps 500000 --seed 0
   git add -A && git commit   # code + config reach Midway via git pull
   bash scripts/sync_models.sh push
   ```
   This writes `<combine_run>/design/design_config.json` (with the pinned natural-start
   rows) and runs no chains.
2. **Midway runs the array** (login node → `plan`, `sbatch --array` of shard `run`,
   `afterok` gather). See the **Midway Claude TODO** below.
3. **Mac pulls + renders.** In cluster mode `all` stops at the hand-off, so target the
   figures explicitly (the `design_render` rule is defined in both modes and runs once the
   gathered `trajectories.npz` is on disk), or render directly:
   ```bash
   bash scripts/sync_models.sh pull
   snakemake -s Snakefile.combine --configfile <config> --config run_root=<combine_run> \
       --cores 8 <combine_run>/figs/design_alignment.pdf
   # or, without snakemake:
   python scripts/render_design.py --design-dir <combine_run>/design --figs-dir <combine_run>/figs
   ```

The Mac-side cluster wrappers already exist and are tested locally:
- `scripts/wf/run_design_shard.py plan --run-dir <D> --n-shards N` → `shards_manifest.json`
- `scripts/wf/run_design_shard.py run --run-dir <D> --shard K` → `shards/shard_<K>.jsonl`
  (per-chain seed `master_seed+i`; **skips chains already in the file** = resume-by-flush)
- `scripts/wf/run_design_gather.py --run-dir <D>` → `trajectories.npz` +
  `designed_sequences.fasta` + `designed.tsv` + `design_manifest.json` +
  `gather_status.json`. Two gates: all planned chains present (unless
  `--allow-missing`), and the warm-started polish never worse than the MC frame
  (`E_polish ≤ E_mc`).

### Midway Claude TODO (needs cluster knowledge + testing — not written here)

Write four shell scripts under `pipeline/external/`, mirroring the
`*_potts_align_*` ones one-to-one (same accounting, resume, `afterok` chaining):

- **`sbatch_design_shard.sh`** — the array task: `#SBATCH --account=pi-ranganathanr
  --partition=caslake --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
  --time=<sized>`, `export OMP_NUM_THREADS=1` (the loop is serial numpy — fan out
  over the array, not threads), then
  `python scripts/wf/run_design_shard.py run --run-dir "$D" --shard "$SLURM_ARRAY_TASK_ID"`.
  Size `--time` from `chains_per_shard × (steps × ~15 µs + polish_seconds)`; the
  polish term dominates and depends on `--polish-schedule` (bound it, or run with
  `--no-polish` and polish the survivors in a second pass).
- **`sbatch_design_gather.sh`** — `--cpus-per-task=1 --mem=4G --time=00:30:00`,
  runs `run_design_gather.py`.
- **`run_design.sh`** — login-node driver: refuse a dirty tree, `git pull
  --ff-only`, preflight `design_config.json`, `run_design_shard.py plan`,
  `sbatch --parsable --array=0-<N-1>%<conc>` for shards + `sbatch
  --dependency=afterok:<arrayid>` for gather, write `.shard_jids`.
- **`finalize_design.sh`** — after the gather END mail: `sacct`-validate all jobs
  COMPLETED, confirm the run outputs exist, tar+zstd the raw `shards/` to reclaim
  space (the Mac only needs the gathered artifacts).

Also add the design paths to `scripts/sync_models.sh`'s **combine** include list
(the file is not in git): sync `design/{design_config.json, shards_manifest.json,
trajectories.npz, designed_sequences.fasta, design_aln_A.fasta, design_aln_B.fasta,
designed.tsv, design_manifest.json, gather_status.json}`; **exclude** `design/shards/`
(raw per-shard scratch), matching
how the `potts_align/shards/` scratch is excluded. The figures live in
`<run_root>/figs/` and are regenerable on the Mac.
