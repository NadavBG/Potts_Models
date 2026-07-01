# Couplings-aware Potts-energy alignment by gap-placement minimization

**Module:** `src/SBM/energy/potts_align.py` · **Tests:** `tests/test_potts_align.py`
· **Scoring branch:** `SBM.energy.score.score_sequence(method="potts_align")` ·
**Combine pipeline:** `Snakefile.combine` with `scoring.method: potts_align` ·
**Progress/context:** `docs/two_model_progress.md`

This is the **production aligner** for the two-model energy work: it finds the
best alignment of a raw sequence to a Potts model by minimizing the exact Potts
energy over gap placements — **no DCAlign** (the DCAlign campaign that this
replaced is archived; the verdict is recorded in `docs/two_model_progress.md`).
This document specifies it in enough detail to re-implement from scratch, and
explains precisely how it relates to (and differs from) DCAlign. Read §1–§2 for
the framing, §3–§7 for the replicable mechanics, §8 for the honest comparison to
DCAlign and the scope limits, §11 for running it in the combine pipeline (locally
and at cluster scale).

---

## 1. What problem this solves (and what it does *not*)

The combine pipeline scores a raw amino-acid sequence under a Potts model by first
**aligning** it to the model's frame (the model has `L` columns; the raw sequence
has `N` residues). The general alignment problem — matches, deletions (gaps in the
model), **and insertions** (extra residues that fit no column), under the full
pairwise couplings — is what DCAlign solves approximately (§8). This module solves
a **strict, easier sub-case**:

> **Insert-free gap placement.** The query has `N ≤ L` residues and *no
> insertions*: every residue maps to a distinct model column, in order. The only
> freedom is **which `g = L − N` of the `L` columns are gaps.** The objective is the
> exact in-frame Potts energy — *no* insertion prior.

This is the case that actually arises for the home-pair residual: those sequences
come from each model's own `L`-frame (natural = the model's MSA; synthetic = MCMC
samples from the model), so they are insert-free by construction (confirmed in
§10.14: `max_n_residues == L`, zero register shifts), with few gaps. The sub-case
matters because it is small enough to often be solved **exactly**.

It does **not** handle insertions (`N > L`), so it does not replace DCAlign for the
cross-family term (a length-96 CM sequence scored under a length-91 PPIC model must
insert). See §9.

---

## 2. Conventions and the energy

Alphabet `"-ACDEFGHIKLMNPQRSTVWY"`: **gap = 0**, the 20 amino acids = 1..20,
`q = 21` (`SBM.energy.encoding`). A **frame** is a length-`L` integer array; an
**insert-free** frame has exactly `N` non-gap entries holding the query residues in
order, the other `g = L − N` entries being `0`.

The Potts energy (package convention, zero-sum gauge; `SBM.energy.potts.potts_energy`
→ `SBM.utils.utils.compute_energies`):

```
E(S) = −Σ_i h[i, S_i] − ½ Σ_{i,j} J[i, j, S_i, S_j]
```

with `J` symmetric (`J[i,j,a,b] = J[j,i,b,a]`) and `J[i,i,·,·] = 0`, so the double
sum equals `Σ_{i<j}`. **Lower `E` = more model-typical.** Gaps are a real
alphabet symbol with their own `h[i,0]` and `J[i,j,0,·]` — *placing a gap is a
scored event*, which is exactly why the placement of the `g` gaps changes the
energy and is worth optimizing.

Both models are loaded in the zero-sum gauge (`load_model` re-applies it
idempotently); energies are only comparable in a fixed gauge.

---

## 3. The size of the search space — `C(L, N)`

The number of insert-free frames is **`C(L, N)`** — `L` choose `N`, the
binomial coefficient: the number of distinct ways to choose which `N` of the
`L` columns are occupied by residues (the rest being the `g = L − N` gaps). Because
choosing the `N` occupied columns is the same as choosing the `g` gap columns,

```text
C(L, N) = C(L, g) = L! / (N! · g!) = (number of monotone insert-free frames)
```

It is **not** the order of work but the *whole search space*: every one of these
frames is a candidate alignment, and the minimum-energy one is the answer. The
count is tiny for few gaps and explodes combinatorially as `g` grows:

| g = L−N | C(96, g) | what it is |
| --- | --- | --- |
| 1 | 96 | the 1-gap PPIC/CM pairs |
| 2 | 4 560 | |
| 3 | 142 880 | largest enumerated in the §7 run |
| 4 | 3.3·10⁶ | |
| 7 | 1.2·10¹⁰ | |
| 9 | 1.3·10¹² | CM-syn-186 (a straggler) |
| 13 | 4.0·10¹⁵ | CM-186 |
| 14 | 2.4·10¹⁶ | CM-289 |

## 4. The two algorithms and how they are chosen

`potts_align(query, model, *, seed, schedule, init_frames=None, fallback="pt")`
dispatches on the space size:

- if `C(L, N) ≤ schedule.enum_max_frames` (default `2·10⁵`) → **exact enumeration**
  (`enumerate_align`), returning the *provably global* minimum;
- else → an **approximate** search: **parallel tempering** (`pt_align`, the default,
  `fallback="pt"`; §6.7) or single-temperature **simulated annealing** (`sa_align`,
  `fallback="sa"`; §6.1–§6.6). Either branch emits a `logging.WARNING` naming the gap
  count and that the result is not provably global, so an approximation is never
  returned silently.

The default budget `2·10⁵` (≈ `g ≤ 3` at `L=96`) is a *memory* floor of the current
implementation, not the feasibility limit — see §4.1.

### 4.1 Cost by gap count, and the exactness wall

Two separate ceilings govern how far exact enumeration can be pushed:

1. **Memory.** `enumerate_align` materializes the full `(C(L,g), L)` int64 frame
   array up front. At `L=96` that is ~2.5 GB for `g=4` and ~47 GB for `g=5`, so the
   in-memory version caps out near `g=4`. Going higher needs a *streaming/sharded*
   enumeration (generate combinations in chunks, never materialize all) — a code
   change, not done yet.
2. **Compute.** At the measured single-core rate of the current numpy energy kernel
   (~1.5·10⁴ frames/s), the wall-clock and Midway SU cost (1 SU ≈ 1 core-hour) is:

   | g | C(96,g) | 1 core | 500 cores | SU (core-h) | verdict |
   | --- | --- | --- | --- | --- | --- |
   | 5 | 6.1·10⁷ | 1.1 h | 8 s | 1 | trivial |
   | 6 | 9.3·10⁸ | 17 h | 2 min | 17 | trivial |
   | 7 | 1.2·10¹⁰ | 9 d | 27 min | 221 | cheap cluster job |
   | 8 | 1.3·10¹¹ | 102 d | 5 h | 2 500 | affordable cluster job |
   | 9 | 1.3·10¹² | 2.7 yr | 2 d | 24 000 | expensive (a big chunk of an allocation, for ONE sequence) |
   | 10 | 1.1·10¹³ | 24 yr | 17 d | 209 000 | infeasible |
   | 13 | 4.0·10¹⁵ | 8 500 yr | 17 yr | 7.5·10⁷ | **impossible** |
   | 14 | 2.4·10¹⁶ | 51 000 yr | 101 yr | 4.4·10⁸ | **impossible** |

A compiled or Gray-code-incremental energy kernel (the incremental ΔE of §6.3 is
~`L`× cheaper than a full recompute; the project already has a C++/OpenMP energy
path) would shift the compute wall right by ~2 orders of magnitude — making `g ≤ 10`
comfortable — but it cannot rescue `g = 13, 14`: a 1000× speedup on `10¹⁶` frames is
still `10¹³` evaluations (decades). **So exact enumeration has a hard wall around
`g ≈ 9–10`; the two worst stragglers (CM-186 `g=13`, CM-289 `g=14`) are
combinatorially out of reach for brute force, on any cluster.**

The principled route to *exact* answers for the high-gap tail is **branch-and-bound
/ integer programming** (the problem is a binary quadratic program with ordering
constraints — an ILP/QUBO solver prunes the space instead of enumerating it),
which can solve large `g` exactly when the bound is tight (not built). The
*good-enough* route — **parallel tempering with the teleport move** (§6.7, §6.2) — is
built and is the default approximate engine, with a cost set by the schedule, not by
`g` (§6.8). It reaches native for the full `g = 4..14` range on the curated set and
the great majority of the high-`g` natural tail. Note PT gives no exactness
*guarantee* (only the enumerated `g ≤ 3` cases are provably global); a residual
handful of the hardest `g = 13–14` naturals may sit a few a.u. above native even at
the `thorough` budget — still far below DCAlign's ΔE 27–44.

---

## 5. Exact enumeration (`enumerate_align`) — the headline path

Replicable steps:

1. Validate the query (1-D, residues in `1..20`, `N ≤ L`); compute `n = C(L, N)`.
   If `n > max_frames`, raise `EnumerationInfeasible`.
2. Enumerate every monotone placement: the occupied-column sets are
   `itertools.combinations(range(L), N)` — there are exactly `n` of them, each a
   sorted length-`N` index tuple. Materialize an `(n, N)` integer array `occ`.
3. Build the `(n, L)` frame matrix: zeros, then scatter the query residues into the
   occupied columns of each row (`np.put_along_axis(frames, occ, query, axis=1)`).
4. Score all frames with the canonical batched energy, in chunks of `chunk`
   (default 4096) to bound peak memory (`compute_energies` materializes an
   `(L, L, chunk)` array ≈ 0.3 GB at `L=96`): `energies[s:e] = potts_energies(frames[s:e], model)`.
5. `best = argmin(energies)`; `topk` = the `k` lowest-energy frames, energy-ascending.

Because every monotone insert-free frame is evaluated with the exact energy, the
result is the **global** minimum over the insert-free space (`is_global_exact =
True`). Cost is `O(C(L,g) · L²)` and is the dominant term only for `g` up to ~3–4.

---

## 6. Simulated annealing (`sa_align`) — the high-gap fallback

For `g` too large to enumerate. Metropolis SA over gap placements, with multiple
seeded restarts; the per-step energy change is computed incrementally in `O(L)`
(not `O(L²)`), which is what makes thousands of steps × dozens of restarts cheap.

### 6.1 State: the placement vector `cols`

Work in `cols` — a **strictly increasing** length-`N` array of the occupied column
indices (residue `r` sits in column `cols[r]`), not the length-`L` frame.
Monotonicity is then a free invariant and a single move is `O(1)` to propose. The
frame is reconstructed from `cols` only at the end.

### 6.2 Move set: ±1 shift + non-local teleport

Each Metropolis step picks a residue `r ∈ {0..N−1}` uniformly and proposes one of
two moves (both keep monotonicity; both are a single two-column ΔE, §6.3):

- **±1 shift** (local): `dst = cols[r] ± 1`, legal iff `lo < dst < hi` where
  `lo = cols[r−1]` (or `−1`) and `hi = cols[r+1]` (or `L`). Refines a placement.
- **Teleport** (non-local, with probability `teleport_frac`, default 0.3): `dst` =
  a *uniformly random empty column strictly between the neighbours* `(lo, hi)`. For
  high `g` that span is wide, so a teleport is a long jump.

The teleport is **load-bearing for high-gap cases**. With ±1 shifts only, moving a
residue across a wide gap requires a path of intermediate frames; if any is a high
energy barrier the residue is *topologically trapped*, and even heavy PT plateaus
(measured: a real `g=13` CM natural stuck at ΔE +88 above native under ±1-only PT at
every budget). Teleport steps over the barrier in one move — the same sequence drops
to ΔE +13 at a fraction of the budget, and `g=14` naturals that needed ~200 s of
±1-only PT to reach native reach it in ~30 s with teleport. At `teleport_frac=0` the
RNG stream is byte-identical to pure-±1 (so the SA path and its tests are unchanged);
PT defaults to `0.3`. Cold replicas reject most teleports (high ΔE × high β) and so
self-restrict to ±1 refinement — the fraction is self-regulating across the ladder.

### 6.3 Incremental ΔE (the load-bearing piece)

Moving residue `a = query[r]` from column `src = cols[r]` (which becomes a gap) to
the empty column `dst` (which becomes `a`) is a **two-column** state change: `src`
goes `a → 0` and `dst` goes `0 → a`. Let `f` be the frame *before* the move
(`f[src]=a`, `f[dst]=0`). With `U(S) = Σ_c h[c,f_c] + ½ Σ_{c,d} J[c,d,f_c,f_d]` so
`E = −U` and `ΔE = −ΔU`:

**Fields:**
```
Δh = (h[src,0] − h[src,a]) + (h[dst,a] − h[dst,0])
```

**Couplings** — only bonds touching `src` or `dst` change. Split into: `src`'s
bonds to all other columns `k ∉ {src,dst}` (at `src`: `a→0`), `dst`'s bonds to all
`k ∉ {src,dst}` (at `dst`: `0→a`), the single `src–dst` bond (joint `(a,0)→(0,a)`),
and the two self terms (zero when `J[i,i]=0`, kept for safety):
```
s_k       = J[src,k,0,f_k] − J[src,k,a,f_k]          # length-L vector over k
t_k       = J[dst,k,a,f_k] − J[dst,k,0,f_k]
sum_src   = Σ_k s_k − s_src − s_dst                  # drop k∈{src,dst}
sum_dst   = Σ_k t_k − t_src − t_dst
pair      = J[src,dst,0,a] − J[src,dst,a,0]          # the src–dst bond, counted once
self_src  = ½(J[src,src,0,0] − J[src,src,a,a])
self_dst  = ½(J[dst,dst,a,a] − J[dst,dst,0,0])

ΔU = Δh + sum_src + sum_dst + pair + self_src + self_dst
ΔE = −ΔU
```

The `src–dst` bond is excluded from both `sum_src` and `sum_dst` (by dropping
`s_dst` and `t_src`) and re-added exactly once by `pair`, with the correct joint
after-state. This is `_move_delta` verbatim and is checked against a from-scratch
`potts_energy` recompute to `1e-9` in `test_incremental_delta_matches_full_recompute`.

### 6.4 One restart (`_anneal_once`)

```
rng        = default_rng(restart_seed)
cols       = sort(random N-subset of [0,L))           # or init_cols (warm start)
energy     = potts_energy(frame(cols))                # one exact call
betas      = geomspace(beta_start, beta_end, n_steps) # β RISES (cools) over the run
for step in range(n_steps):
    propose a legal ±1 move (else continue)
    de = move_delta(...)
    if de <= 0 or rand() < exp(-betas[step]·de):       # Metropolis at inverse-temp β
        apply move; energy += de; track running best
best_energy = potts_energy(best_frame)                 # re-verify; drift canary (abs_tol 1e-6)
```

`β` is the **inverse** temperature and *increases* from `beta_start` to `beta_end`
(annealing = cooling), matching the project/DCAlign convention. The final exact
recompute guards against float drift in the incremental accumulator.

### 6.5 Multi-restart and warm starts (`sa_align`)

- Per-restart seeds are spawned from one master seed:
  `seeds = SeedSequence(seed).generate_state(n_restarts + len(init_frames))`. The
  master `seed` is **required and logged** (project rule).
- `init_frames` (optional) warm-starts the first restarts at production-legal
  heuristic frames (fields-MAP Viterbi frame, and/or DCAlign's own frame); the
  remaining `n_restarts` start from random monotone placements. Warm starts are the
  decisive lever for the high-gap cases (they took the high-gap median from
  worse-than-DCAlign to better-than-DCAlign; see §7).
- Return the global-best frame over all restarts, plus the `topk` **distinct**
  restart-best frames (energy-ascending) for a downstream "min over K".

### 6.6 Default schedule (`SASchedule`, all fields logged)

`n_restarts=16`, `n_steps=5000`, `beta_start=0.5`, `beta_end=8.0`, `topk=4`,
`enum_max_frames=2·10⁵`.

### 6.7 Parallel tempering (`pt_align`) — the default high-gap engine

Single-temperature SA gets stuck in the deep-but-narrow wrong basins of the
high-gap cases. **Parallel tempering** (replica exchange) crosses them and is the
default `potts_align` fallback. It reuses the exact same move set and incremental ΔE
(`_try_move`); the only additions are multiple temperatures and swaps:

1. **Replicas.** `n_replicas` copies run the `±1`-shift Metropolis dynamics, each at
   a *fixed* inverse temperature from a geometric ladder
   `betas = geomspace(beta_min, beta_max, n_replicas)` (hot → cold). Hot replicas
   (`β≈0.1`) cross energy barriers freely; cold replicas (`β≈8`) refine.
2. **Sweep block.** Each block, every replica runs `sweep_moves` proposed moves
   (default `= N`) at its own `β`, tracking the global-best frame seen.
3. **Replica-exchange swaps.** After each block, adjacent replicas `(k, k+1)` attempt
   to **swap their whole configurations** (the temperatures stay fixed by slot), with
   the replica-exchange Metropolis rule

   ```text
   P_swap = min(1, exp((β_k − β_{k+1}) · (E_k − E_{k+1})))
   ```

   Pairing alternates even/odd parity each block. The effect: a low-energy
   configuration discovered by a hot replica migrates down the ladder to be refined
   cold. This is what lets PT reach a narrow basin a fixed-temperature walk would
   never enter.
4. **Warm starts + multi-restart.** `init_frames` seed the *coldest* replicas (where
   refinement happens); `n_restarts` independent PT runs add robustness. The global
   best over all replicas and restarts is returned (with the same drift-canary
   re-verification and `topk` collection as SA).

**Default schedule (`PTSchedule`):** `n_replicas=12`, `beta_min=0.05`, `beta_max=10.0`,
`n_blocks=3000`, `sweep_moves=None` (⇒ `N`), `n_restarts=5`, `teleport_frac=0.3`,
`topk=4`. PT parallelizes trivially (replicas and restarts are independent).

### 6.8 Cost of the approximate route, and schedule presets

**The cost is set by the schedule, not by `g`.** Unlike enumeration (exponential in
`g`, §4.1), one PT run does a *fixed* number of Metropolis moves:

```text
moves = n_restarts · n_blocks · n_replicas · sweep_moves   (sweep_moves defaults to N)
time  ≈ moves / R,   R ≈ 4.7·10⁵ moves/s   (measured, single Mac core)
```

`R` is **independent of `L`** — at `L = 91, 94, 96` the per-move cost is 2.14 / 2.11 /
2.09 µs (each move is one `O(L)` ΔE plus bookkeeping, and `sweep_moves = N ≈ L`
cancels the `L`-growth). So **every number here transfers to `L = 94` unchanged.**
What grows with `g` is the *budget needed to reach native*, not the cost per unit
budget. Per-sequence cost (at `N ≈ 85`):

| preset | n_rep · n_blk · n_rst | moves | time/seq | use |
| --- | --- | --- | --- | --- |
| `fast` | 10 · 1500 · 3 | 3.8·10⁶ | ~8 s | warm-started refine; the design inner loop |
| **default** `PTSchedule()` | 12 · 3000 · 5 | 1.5·10⁷ | **~32 s** | `g = 4..12` |
| `thorough` `PTSchedule.thorough()` | 14 · 6000 · 6 | 4.3·10⁷ | ~80 s | `g ≥ 13`; one important sequence |

**The intelligent default is `g`-adaptive.** Because `g` is always known (`L − N`)
and the *cost* of a schedule is `g`-independent while the *budget needed* grows with
`g`, `potts_align` auto-selects the schedule by gap count
(`PTSchedule.for_gap_count(g)`: `thorough` for `g ≥ 13`, else the default) — no
ground truth needed. Measured (`teleport_frac=0.3`): `g ≤ 3` enumerate exactly
(instant); `g = 4–12` reach native at the default (~32 s); `g = 13–14` reach native
at `thorough` (~80 s) — where the default alone leaves a real residual (a `g=13` CM
natural: default ΔE +65, `thorough` 0.0). A residual handful of the very hardest may
still sit a few a.u. above native even at `thorough`, but far below DCAlign's ΔE
27–44. Without teleport these needed ~200 s and some never converged (§6.2).

**Sizing "≈10,000 MCMC steps to minimize one sequence's energy."** One replica does
`n_blocks · sweep_moves` moves per restart; `n_blocks ≈ 10⁴` is ~`10⁴ · 85 ≈ 8.5·10⁵`
moves per replica, i.e. a deep single-sequence optimisation — the `thorough` end. For
a **design loop** that re-aligns after each of ~10⁴ sequence edits, do **not** cold-
align each step: warm-start from the previous frame and run the `fast` preset (~8 s),
or fewer blocks still, since one residue edit perturbs the alignment only locally
(`init_frames=[previous_frame]`). The cost model above lets you trade blocks for
wall-time directly.

**Cost to align all the naturals (home term).** Gap-count distribution (measured):
CM `L=96`, 1258 naturals — 51% `g≤3` (enumerate, instant), 49% `g≥4` (PT); PPIC
`L=91`, 26 701 naturals — 95.5% `g≤3`, 4.5% `g≥4`. So ≈ 617 (CM) + ≈ 1200 (PPIC) ≈
**1800 sequences hit PT** at ~32 s each ≈ **16 core-hours**, plus the ~26 k enumerable
ones (sub-second each) ≈ a couple more. Total ≈ **18 SU** — ~2.3 h on the 8-core Mac,
or minutes on a Slurm array (the work is embarrassingly parallel over sequences).

---

## 7. The basin-width helper (`perturb_frame`) and results

`perturb_frame(native_frame, k, *, rng)` returns a length-`L` insert-free frame
differing from `native_frame` by reassigning `k` columns: vacate `k` occupied
columns, fill `k` previously-gap columns, re-thread the residues in order (so `N`
and monotone order are preserved). Used by the Midway basin-width probe (warm-start
BP at native ± k columns; reuses the warm-start driver verbatim).

**Result on the 24 curated home pairs** (the one-off eval driver
`analyze_potts_align.py` and its output `combine-CM-PPIC-dcalign-pottsalign/` are
archived under `.archive/`, since the comparison was against DCAlign frames;
master seed 0; `ΔE = E_best − E_native`):

| group | count | outcome |
| --- | --- | --- |
| enumerable (`g ≤ 3`, all 8 PPIC + 3 CM) | 11/16 | recover **exactly** — native is the provable global Potts min (`ΔE ≈ 0`); PPIC-176 **beats** native (`ΔE −1.24`) |
| parallel-tempering (`g = 4..14`) | 5/16 | reach native-or-better (CM-186 `g=13`, CM-289 `g=14` cracked by PT) |
| controls | 8/8 | unchanged |

So **16/16** worse-than-native pairs reach native-or-better with **no DCAlign at
all** — 11 provably global by enumeration, the 5 high-gap cases by parallel tempering
(§6.7; the `g=14` case needs the heavier PT schedule, ~50 s). With the older
single-temperature SA only 13/16 were reached (the `g = 9, 13, 14` tail plateaued at
ΔE 5–17, still beating DCAlign's 27–44); parallel tempering crosses those basins. A
Midway confirmatory batch (archived; see `docs/two_model_progress.md`) showed
warm-starting DCAlign-BP at the Potts-align frames recovers *fewer* (10/16) — BP only
drifts off the exact frame — so the direct minimizer strictly dominates and BP is
dropped from the loop.

---

## 8. How this relates to DCAlign (and why it is not "beating DCA")

**DCAlign** (Muntoni, Pagnani, Weigt, Zamponi, *Phys. Rev. E* 102, 062409 (2020);
DCAlign v1.0, *Bioinformatics* 39(9):btad537 (2023)) solves the **general**
alignment problem. Its variables are, per model column `i`, a match/gap bit
`x_i ∈ {0,1}` and a pointer `n_i` into the query; the pointer differences
`Δn_ij = n_j − n_i` encode matches, deletions, **and insertions** (`Δn > 1` = extra
query residues between matched columns). It maximizes (their Eq. 1)

```
argmax_(x,n)  exp(−β·ℋ(x,n)) · Π_{ij} P_ij(Δn_ij)^β
```

where `ℋ` is the **full Potts energy** (so yes, DCAlign *is* minimizing a Potts
energy over alignments) and `P_ij(Δn)` is an empirical **insertion prior** learned
from the seed MSA.

**Why the general problem is hard.** With only fields (a profile HMM), the score
decomposes along the chain and Viterbi/forward DP finds the exact optimal alignment
in polynomial time. Adding couplings `J_ij` between *all* column pairs makes the
optimal state at column `i` depend on every other column simultaneously — the
Markov factorization DP needs is gone, and there is no exact polynomial algorithm.
DCAlign's contribution is an **approximate** solver: belief propagation +
decimation on the alignment factor graph, annealed in `β`. Approximate ⇒ no
global-optimum guarantee ⇒ it can converge to a wrong fixed point. That failure
mode is precisely the combine residual we chased through §10.12–§10.19.

**What this module does differently — and why it is easier, not smarter.** It
restricts to a sub-case where the hardness evaporates:

| | DCAlign | this module |
|---|---|---|
| insertions | yes (`Δn > 1`) | **no** — deletions only |
| objective | Potts **+** insertion prior `P_ij` | **pure** Potts |
| alignment space | all indel alignments (huge) | choose `g` gap columns: `C(L,g)` |
| method | approximate BP + decimation + annealing | **exact enumeration** (or SA) |
| guarantee | none (approximate) | **provably global** when enumerated |

For the residual sequences `g` is 1–3, so `C(L,g)` is 91 to ~140,000 — small enough
to **evaluate every candidate exactly and take the minimum**. Exhaustive search
trivially beats an approximate solver on a tiny instance; that is a statement about
*instance size*, not about out-inferring DCAlign. The analogy: DCAlign is a
general approximate MAP solver; we noticed our particular instances have only a
handful of binary choices and enumerated all of them. We did **not** invent a
better alignment algorithm.

Two further honest points:

- **The comparison is fair but within the insert-free space.** DCAlign's output
  frames in our cache are also insert-free, length-`L`, scored by the identical
  in-frame energy (§10.14), so both search the same space — and on it, enumeration
  finds the exact min while DCAlign's BP lands in a worse local optimum.
- **We optimize a simpler (cleaner, for this goal) objective.** DCAlign minimizes
  Potts + prior; the prior can pull it off the pure-Potts minimum (a feature for
  general alignment — it encodes real indel geometry — but a distraction for our
  "return the lowest-energy frame" goal, which the user adopted as the production
  policy). We minimize pure Potts, which is exactly that goal.

The genuinely non-trivial work was **diagnostic**, not algorithmic: the iter-003
campaign (refuting pcount, μ, multi-seed, fields-MAP init, anneal-from-hot) is what
established that the residual is insert-free with few gaps and is a *search*
failure — which is what told us the right tool is brute force, not more BP tuning.

---

## 9. Scope and limitations

- **No insertions (`N > L`).** The gap-placement enumeration/PT only applies when the
  query fits the frame (`N ≤ L`). It *does* handle **cross-family** pairs when
  `N ≤ L_other` (a PPIC query, `N ≤ 91`, in the CM `L=96` frame is just a `g ≥ 5`
  gap placement — validated). Only `N > L` is out of scope (CM queries with `N > 91`
  scored under PPIC need insertions). The combine pipeline scores every pair with
  `N ≤ L` and skips the rest with a NaN row (§11). Closing the `N > L` gap needs an
  insertion-capable move set — deferred; under the "design ≤ min(L)=91" direction
  both terms are insert-free and this never arises (`docs/two_model_progress.md`).
- **High gap counts.** For `g` beyond the enumeration budget the result is the SA
  approximation, which on the 3 high-gap CM stragglers (`g = 9–14`) plateaus
  `ΔE 5–17` above native (still better than DCAlign). Block moves or simulated
  tempering could push these further; not yet needed.
- **The MSA "native" frame is not always the global Potts min.** Enumeration found
  frames strictly lower than the curated native frame (e.g. PPIC-176, `ΔE −1.24`).
  The production policy returns the global Potts minimum and surfaces such
  beats-native cases as a diagnostic.

---

## 10. API and how to run

```python
from SBM.energy.model import load_model
from SBM.energy.potts_align import potts_align, SASchedule

model = load_model("results/PPIC-dense/iter-001-baseline/model.npy", name="PPIC")
raw = ...  # 1-D int array of residues 1..20 (gaps stripped), length N ≤ L
res = potts_align(raw, model, seed=0, schedule=SASchedule())
res.best_frame      # length-L insert-free frame: the global (or SA) Potts min
res.best_energy     # its exact in-frame Potts energy
res.is_global_exact # True iff the whole insert-free space was enumerated
res.method          # "enumerate" | "sa"
res.topk_frames     # K best distinct frames (for a multi-start min)
```

`PottsAlignResult` carries no `E_native`/`ΔE` — the aligner never sees the
ground-truth frame; any comparison to a reference frame is the caller's job. The
score-layer wrapper is `SBM.energy.score.score_sequence(seq, model,
method="potts_align", seed=...)` (§11), and the combine `potts_align_baseline`
rule (`SBM.energy.potts_align_baseline`) reports ΔE-vs-native per home pair.

```
.venv/bin/python -m pytest tests/test_potts_align.py tests/test_potts_align_baseline.py -q
```

**Tests (the brute-force-anchor philosophy of `tests/test_energy.py`):**
SA's best energy equals the exact `enumerate_align` minimum on a tiny model
(enumeration is the oracle); the incremental ΔE matches a from-scratch recompute;
determinism (same seed → identical frame); output invariants (length-`L`, `N`
non-gap, monotone, in-alphabet); top-K sorted/distinct; loud `ValueError`s on a
gapped query, `N > L`, or a missing seed.

---

## 11. Running it in the combine pipeline (local + at cluster scale)

`potts_align` is wired into the two-model `combine` pipeline as
`scoring.method: potts_align` (`config/params_combine-CM-PPIC-potts.yaml`;
`-potts-tiny` is the smoke config). It scores **every (query, model) pair with
`N ≤ L`** and returns the global insert-free Potts-min frame; cross pairs with
`N > L_other` (insertions needed) are emitted as a NaN row (`note="N>L …"`), not
crashed. Both are computed by the *same* aligner, so `E_A` and `E_B` are
comparable. The score branch is pure numpy — no Julia, no external tool.

```
python scripts/iter.py run combine-CM-PPIC-potts "<tag>" --snakefile Snakefile.combine
```

**Two knobs bound the cost** (`scoring.*`, validated by `combine_config.py`):

- `pa_cross_subsample_{origin,under,n}` — restrict the expensive cross block
  (e.g. every PPIC natural scored under the CM frame, all `g ≥ 5` → PT) to a
  *seeded* random subset of `n` ids. This is the run's cost driver; the subsample
  preserves the gap-count distribution so the figure cluster is unbiased.
- `query.{n_random,random_length}` — append `n_random` random length-`random_length`
  sequences (uniform iid residues, seeded) as a negative-control group. They must
  sit at high energy under both models, well clear of the naturals — the sanity
  check on the energy scale in `two_model_energy.pdf`.

**Local vs. cluster.** Everything runs on the Mac. Per-pair cost is set by the PT
schedule, not by `g` (§6.8): instant for `g ≤ 3`, ~tens of seconds for the PT
cases. For a large query set (the full CM+PPIC naturals ≈ tens of core-hours,
§6.8) the alignment can be pre-built out-of-process on a Slurm array — pure Python,
no `module load julia`:

1. **Build the query on the Mac** (`snapshot_config`, `resolve_models`,
   `build_query` targets of `Snakefile.combine`) and, if using Midway, push
   models + query with `scripts/sync_models.sh push`.
2. **Shard + gather** on the cluster with `pipeline/external/run_potts_align_align.sh`
   (`plan` → `sbatch --array` of `run_potts_align_shard.py` at `cpus=1` → gather via
   `run_potts_align_gather.py`), writing `<run_root>/potts_align/cache/<model>/alignments.tsv`.
3. **Pull the cache** (`scripts/sync_models.sh pull`) and run `Snakefile.combine all`
   on the Mac. The `score` rule reads the cache and recomputes each energy in-frame
   as a `≤1e-6` gauge canary; nothing else touches the cluster.

The `score` rule *declares* the cache as an input when `method: potts_align`, so
Snakemake refuses to score until the align+gather has filled it (the two-phase
dependency is explicit in the DAG). When scoring a small query set, omit the
cluster step entirely: `score_two_models.py --method potts_align --seed S` (no
`--potts-align-cache`) recomputes each pair live in-process.
