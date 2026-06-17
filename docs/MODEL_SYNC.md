# Model transfer: Mac ↔ Midway (checksummed rsync)

Trained models are large binary `.npy` blobs (`model.npy` ~47 MB; ~4.4 GB across
all families). They do **not** live in git. They are synced between the Mac and
Midway with `scripts/sync_models.sh`, which wraps `rsync` and adds an independent
SHA-256 verification pass so a corrupted or truncated transfer fails loudly
instead of silently producing a wrong model on the cluster.

Every model is meant to exist on **both** machines, so a larger cross-model
comparison can run on Midway while the originals stay on the Mac.

## Why not Git-LFS

The repo was briefly *configured* for Git-LFS (`.gitattributes` tracked
`results/**/*.npy`) but nothing was ever committed to it. LFS is the wrong tool
here: each model run is ~0.5 GB, which blows past GitHub LFS free quotas (1 GB
storage / 1 GB monthly bandwidth) and bloats every clone. rsync moves only the
bytes that changed, costs nothing, and keeps the binaries out of git history.

## What gets synced

By default, **durable artifacts only** — the small files needed to reproduce or
score a model, not the regenerable figure caches:

| Synced (durable) | Skipped (regenerable / scratch) |
|---|---|
| `model.npy` | `figs/` (PDFs + `figs/inputs/stats_*.npy`, ~0.4 GB/run) |
| `inputs/msa.npy` + `inputs/*.json` | `synthetic/mpnn_sweep_*/mpnn_tmp/` |
| `synthetic/align_T*.npy` + `*.json` | `__pycache__/`, `.snakemake/`, `*.pyc`, `.DS_Store` |
| `synthetic/.../mpnn_scores.json` | |
| `masks/*.npy` + `*.json` | |
| `manifest.json`, `run_manifest.json`, `train_meta.json` | |

Durable-only is ~50–60 MB/run vs ~0.5 GB/run. Figures regenerate from `model.npy`
+ `synthetic/` via `scripts/render_sbm.sh` / `render_figures.py`. Pass `--with-figs`
to mirror everything (e.g. archiving a finished run).

## The checksum guarantee

Two independent layers protect against silent corruption:

1. **rsync's own per-file rolling+MD5 check** during transfer.
2. **An independent `results/SHA256SUMS` manifest, verified on the destination
   after the transfer.** This proves every durable file's *content* matches the
   source — independent of rsync's transfer logic. It also catches a partial
   sync or a manifested file that rsync failed to deliver: an entry that did not
   land (or landed corrupted) prints `FAILED` and the command exits non-zero.
   (It checks only files *in* the manifest, so it will not flag an extra file
   that an exclude rule failed to drop — that is a wasted-bandwidth concern, not
   a corruption one.)

`results/SHA256SUMS` is standard `sha256sum` format with repo-root-relative paths
(e.g. `results/CM-bm-dense/iter-002-base-model/model.npy`), so it verifies
identically on macOS (`sha256sum` or `shasum -a 256`) and Linux. It lives under
the gitignored `results/`, travels with the rsync, and is a standing record you
can re-check any time with `verify`.

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
- `--with-figs` — also sync `figs/` and `mpnn_tmp/` (full mirror).
- `--mirror` — add rsync `--delete`: delete destination files absent from the
  synced set (prompts first; `--yes` to skip the prompt). **Off by default** —
  normal `push`/`pull` are additive and never delete, so a model present on only
  one side is preserved until you explicitly mirror. Note `--delete` does **not**
  remove excluded dirs (`figs/`, `mpnn_tmp/`), so `--mirror` without `--with-figs`
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
- This push unblocks **Tier-2** of the two-model energy work (real CM/PPIC models
  on Midway for actual `sbatch` DCAlign submission) — see
  `docs/initiate_two_model_energy.md` §10.9 and `pipeline/external/README.md`.
- If `/project` quota becomes tight, point `SBM_MIDWAY_REPO`'s `results/` at a
  scratch-backed dir via a symlink on Midway; the script only cares about the
  final path, not whether it is a real dir or a symlink.
