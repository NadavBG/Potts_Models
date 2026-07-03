# Artifact transfer: Mac ↔ Midway (checksummed rsync)

`scripts/sync_models.sh` syncs the large, gitignored artifacts that don't belong
in git between the Mac and Midway, wrapping `rsync` with an independent SHA-256
verification pass so a corrupted or truncated transfer fails loudly instead of
silently producing a wrong file on the far side. It covers **three trees**:

- **`results/`** — trained models (`model.npy` ~47 MB; ~4.4 GB across all
  families). Meant to exist on both machines so the cluster alignment can read
  them.
- **`combine/`** — two-model runs, including the **`potts_align` cache**
  (`potts_align/cache/<model>/alignments.tsv`) the Mac reads to score. When a
  large query set is aligned on the cluster, it produces the cache; you pull it
  back and score on the Mac. (A small query set is scored directly on the Mac
  with no cache — `docs/POTTS_ALIGN.md` §11.)
- **`natural_folds/`** — the content-addressed ESMFold cache of the naturals
  (`natural_folds/<msa_sha8>/`, keyed by source-FASTA sha8; `docs/CHARACTERIZE.md`).
  The fold is a property of the FASTA content, not of any run, so it is its own
  tree rather than nested under a model. Only the distilled `fold_scores/` +
  `tm_vs_refs/` TSVs travel; the bulky per-sequence PDBs stay Midway-side.

Each tree gets its own `<tree>/SHA256SUMS` manifest. A tree absent on one side
is skipped (a fresh Mac has no `combine/` until a run); a command fails only if
no tree is present. Override the tree list with
`SBM_SYNC_ROOTS="results ..."`. The Mac-primary workflow that motivates all this
is `docs/POTTS_ALIGN.md` §11 (running `potts_align` locally and at cluster scale).

> **Retired DCAlign runs are never synced.** They live under `.archive/` on the
> Mac (gitignored); `.archive/` and any `combine/*dcalign*` dir are excluded from
> both the rsync transfer and the SHA256SUMS manifests, so a `pull` can't restore
> them from Midway. See `docs/two_model_progress.md`.

## Why not Git-LFS

The repo was briefly *configured* for Git-LFS (`.gitattributes` tracked
`results/**/*.npy`) but nothing was ever committed to it. LFS is the wrong tool
here: each model run is ~0.5 GB, which blows past GitHub LFS free quotas (1 GB
storage / 1 GB monthly bandwidth) and bloats every clone. rsync moves only the
bytes that changed, costs nothing, and keeps the binaries out of git history.

## What gets synced — `results/` (models)

By default, **durable artifacts only** — the small files needed to reproduce or
score a model, not the regenerable figure caches:

| Synced (durable) | Skipped (regenerable / scratch) |
|---|---|
| `model.npy` | `figs/` (PDFs + `figs/inputs/stats_*.npy`, ~0.4 GB/run) |
| `inputs/msa.npy` + `inputs/*.json` | `__pycache__/`, `.snakemake/`, `*.pyc`, `.DS_Store` |
| `synthetic/align_T*.npy` + `*.json` | |
| `masks/*.npy` + `*.json` | |
| `manifest.json`, `run_manifest.json`, `train_meta.json` | |

Durable-only is ~50–60 MB/run vs ~0.5 GB/run. Figures regenerate from `model.npy`
+ `synthetic/` via `scripts/render_sbm.sh` / `render_figures.py`. Pass `--with-figs`
to mirror everything (e.g. archiving a finished run). The naturals fold cache is a
separate tree (below), no longer nested under `results/`.

## What gets synced — `natural_folds/` (fold cache)

The per-MSA ESMFold cache (`docs/CHARACTERIZE.md`) lives in its own top-level tree,
`natural_folds/<msa_sha8>/`, content-addressed by the source-FASTA sha8 — it is a
property of the FASTA, not of any model run or combine run, so a model and a combine
that share an MSA share the fold.

| Synced (durable) | Skipped (regenerable / scratch) |
|---|---|
| `fold_scores/*.tsv` (distilled pLDDT/pTM per natural) | `structures/*.pdb` (one ESMFold PDB per natural — ~28k tiny files) |
| `tm_vs_refs/<refkey>.tsv` + `.meta.json` (cached TM-scores) | `__pycache__/`, `.snakemake/`, `*.pyc`, `.DS_Store` |

**`natural_folds/*/structures/` is excluded.** One PDB per natural sequence — tens
of thousands of tiny files (e.g. ~27k for PPIC) that otherwise dominate the rsync
stat/checksum time. Only the distilled TSVs sync; the PDB cache stays Midway-side
(0-SU to regenerate on the GPU, and the Mac only needs the scores to render figures).
Excluded by a `structures`-basename prune in `rsync_excludes()`, `find_durable()`,
**and** `build_remote_manifest()` — all three gate on `--with-figs`, mirrored so
`verify` stays in lock-step. `--with-figs` includes the full PDB archive. (The
`combine/` tree separately keeps its `characterize/structures/` — only the ~96 design
PDBs.)

## What gets synced — `combine/` (potts_align cache)

A combine run's cluster align output is dominated by per-shard scratch that is
regenerable. Sync keeps only the small durable cache + run metadata and drops the
rest:

| Synced (durable) | Skipped (scratch / regenerable) |
|---|---|
| `potts_align/cache/<model>/alignments.tsv` (the gathered result) | `potts_align/shards/` (raw per-shard TSVs, merged into `alignments.tsv`) |
| `potts_align/cache/<model>/meta.json` (provenance) | `potts_align/logs/` + top-level `logs/` (machine-local job logs) |
| `config_snapshot.yaml`, `models.json`, `query/` | `*.tar.zst` (the finalizer's archives) |
| `potts_align/{shards_manifest,gather_status}.json`, `.shard_jids` | `figs/` (regenerable; `--with-figs` to include) |
| `data/` (scores) + `provenance/` (manifests), after scoring | `.archive/` and any `combine/*dcalign*` dir (retired) |

The whole point: aligning a large query set runs on Midway, but only ~0.5 MB/run
needs to come back for the Mac to score. The score step reads `alignments.tsv`
and recomputes energies in-frame — no cluster. The end-to-end sequence is in
`docs/POTTS_ALIGN.md` §11.

The exclude patterns (`work/`, `shards/`, `logs/`, `*.tar.zst`, `.archive/`,
`*dcalign*`) and the manifest prunes are kept in lock-step inside
`sync_models.sh` so the post-transfer verify never flags a manifested file that
was deliberately skipped.

## The checksum guarantee

Two independent layers protect against silent corruption:

1. **rsync's own per-file rolling+MD5 check** during transfer.
2. **An independent `<tree>/SHA256SUMS` manifest per synced tree, verified on the
   destination after the transfer.** This proves every durable file's *content*
   matches the source — independent of rsync's transfer logic. It also catches a
   partial sync or a manifested file that rsync failed to deliver: an entry that
   did not land (or landed corrupted) prints `FAILED` and the command exits
   non-zero. (It checks only files *in* the manifest, so it will not flag an
   extra file that an exclude rule failed to drop — that is a wasted-bandwidth
   concern, not a corruption one.)

Each manifest (`results/SHA256SUMS`, `combine/SHA256SUMS`, `natural_folds/SHA256SUMS`) is standard
`sha256sum` format with repo-root-relative paths (e.g.
`results/CM-bm-dense/iter-002-base-model/model.npy`), so it verifies identically
on macOS (`sha256sum` or `shasum -a 256`) and Linux. Each lives under its
gitignored tree, travels with the rsync, and is a standing record you can
re-check any time with `verify`.

## First-time setup

1. **Midway clone.** The repo must be cloned on Midway at the remote path
   (default `/project/ranganathanr/nadavbg/Potts_Models`), with `results/`
   writable. The combine pipeline reads models by their relative `run_dir`
   (e.g. `results/CM-bm-dense/iter-002-base-model`), so models synced into
   `<repo>/results/` resolve directly with no config change.

2. **GNU rsync on the Mac (recommended).** macOS ships `openrsync`, which lacks
   some flags; the script warns and falls back to portable flags. For a smoother
   transfer: `brew install rsync` (the script auto-prefers `/opt/homebrew/bin/rsync`).

3. **Optional local config.** Only if your host/path differ from the defaults:
   ```bash
   cp scripts/sync_models.local.sh.example scripts/sync_models.local.sh   # gitignored
   ```
   Or override per-invocation with `--host` / `--repo`, or the `SBM_MIDWAY_HOST`
   / `SBM_MIDWAY_REPO` / `SBM_RSYNC` env vars.

## Usage

```bash
# Preview what a push would transfer (no transfer, no verify):
scripts/sync_models.sh push --dry-run

# Mac -> Midway, then verify the checksums on Midway:
scripts/sync_models.sh push

# Midway -> Mac, then verify the checksums locally:
scripts/sync_models.sh pull

# Are the two machines in sync? (builds manifests both sides, diffs them; no transfer)
scripts/sync_models.sh status

# Re-verify an existing tree against its manifest:
scripts/sync_models.sh verify            # local
scripts/sync_models.sh verify --remote   # on Midway

# (Re)build the local manifest without transferring:
scripts/sync_models.sh hash
```

### Authentication (Midway / Duo)

A `push` does two remote operations (transfer, then verify) and `pull`/`status`
likewise; each would normally trigger its own password + Duo prompt. To avoid
that, every command opens **one shared SSH connection** (OpenSSH `ControlMaster`
multiplexing) and reuses it, so **you authenticate exactly once per command**.
The shared connection is torn down when the command finishes.

### Flags

- `--dry-run` — rsync `-n`; show the transfer plan, skip verify.
- `--no-verify` — transfer only; skip the checksum verify. Use when you want a
  fast push/pull and will verify later (`verify` / `verify --remote`). You still
  authenticate once for the transfer.
- `--with-figs` — also sync `figs/` (full mirror).
- `--mirror` — add rsync `--delete`: delete destination files absent from the
  synced set (prompts first; `--yes` to skip the prompt). **Off by default** —
  normal `push`/`pull` are additive and never delete, so a model present on only
  one side is preserved until you explicitly mirror. Note `--delete` does **not**
  remove excluded dirs (`figs/`), so `--mirror` without `--with-figs`
  mirrors the *durable* set, not the full tree.
- `--host HOST`, `--repo PATH` — override the Midway host / repo root.

## Removing a model

`push` and `pull` never delete (additive by design). To drop a model from both
machines, delete its run dir on each side manually, or use `--mirror` from the
side you consider authoritative (it deletes destination files absent from the
source — review the `--dry-run` output first).

## Next steps

- Run `scripts/sync_models.sh push --dry-run` once to confirm the file list and
  destination, then `push` for real.
- Where this fits end to end — push models to Midway, align a large query set
  there with `potts_align`, pull the cache back, score on the Mac — is
  `docs/POTTS_ALIGN.md` §11 (cluster mechanics in `pipeline/external/README.md`).
- If `/project` quota becomes tight, point `SBM_MIDWAY_REPO`'s `results/` at a
  scratch-backed dir via a symlink on Midway; the script only cares about the
  final path, not whether it is a real dir or a symlink.
