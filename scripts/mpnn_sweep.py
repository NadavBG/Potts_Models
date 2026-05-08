"""ProteinMPNN foldability sweep — orchestrator.

Driven by ``scripts/sample_sbm.py``'s ``--mpnn-sweep`` dispatch. Samples
N sequences from a trained model at each of a list of temperatures,
builds interpretability controls (WT, uniform random, shuffled WT,
natural MSA bootstrap), and (optionally) scores everything with the
upstream ``dauparas/ProteinMPNN`` ``--score_only`` mode.

Output layout (under ``<run_dir>/synthetic/mpnn_sweep_seed<seed>/``)::

    align_T<T>_seed<seed>.npy        # per-T sampled alignment, schema
    align_T<T>_seed<seed>.json       # ↳ matches scripts/sample_sbm.py sidecar
    control_wt.npy                   # (1, L) WT MSA row
    control_random_seed<seed+1000>.npy
    control_shuffled_wt_seed<seed+1001>.npy
    control_natural_seed<seed+1002>.npy
    mpnn_scores.json                 # per-group score table + bench
    manifest.json                    # full sweep provenance
    bench.json                       # per-T sample + score wall times

Design notes:

* Per-T sample seed = master_seed + T_index (existing convention from
  ``sample_sbm.py``); control seeds use fixed offsets +1000/+1001/+1002
  so they never collide with a sample-seed range, even with up to 1000
  temperatures.
* The PDB ↔ MSA map comes from a pairwise alignment of the PDB chain
  sequence to the WT MSA row (first record in ``data/fasta/CM.fasta``
  by default; override with ``--mpnn-wt-fasta``). The PDB may be a
  truncation of the WT (e.g. 1ECM lacks the WT's first 3 residues).
* Scoring is delegated to ``protein_mpnn_run.py --score_only`` via
  subprocess; this avoids binding the project to ProteinMPNN's
  internal ``tied_featurize`` API. ``--mpnn-skip-scoring`` produces
  the alignments and the manifest without running MPNN — useful for
  smoke tests and for hosts without the upstream clone installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

import SBM.provenance as provenance
import SBM.utils.utils as ut

log = logging.getLogger(__name__)

#: Default temperature ladder for the sweep — 10 evenly-spaced Ts in
#: (0, 1.0]. The training temperature is T=1.0; T<1 sharpens the
#: distribution toward MAP modes.
_DEFAULT_TEMPERATURES: tuple[float, ...] = tuple(
    round(0.1 * (i + 1), 2) for i in range(10)
)

#: Default number of synthetic sequences per temperature.
_DEFAULT_N_PER_T: int = 100

#: Default control set. ``wt`` is the natural-design anchor; ``random``
#: is the floor; ``shuffled`` is the composition-only control;
#: ``natural`` is the "is the model close to natural?" anchor.
_DEFAULT_CONTROLS: tuple[str, ...] = ("wt", "random", "shuffled", "natural")

#: Default PDB. Lives at ``data/structures/1ECM.pdb`` in this repo.
_DEFAULT_PDB_RELPATH: str = "data/structures/1ECM.pdb"

#: Default WT FASTA — the project-wide CM alignment's first record is
#: the 1ECM reference (``>pdb|1ECM|A|...``).
_DEFAULT_WT_FASTA_RELPATH: str = "data/fasta/CM.fasta"

#: Control seed offsets, large enough to never collide with a per-T
#: sample seed range that grows with the temperature list.
_CONTROL_SEED_OFFSETS: dict[str, int] = {
    "random": 1000,
    "shuffled": 1001,
    "natural": 1002,
}

#: Natural-bootstrap N defaults to the per-T sample N so violins are
#: visually comparable.


# ── WT extraction ───────────────────────────────────────────────────────


def _read_first_fasta_record(path: Path) -> tuple[str, str]:
    """Return (header, sequence) of the first record in a FASTA file.

    Hand-rolled parser to avoid pulling Biopython at sweep-load time
    (it is loaded lazily inside ``mpnn_score``).
    """
    if not path.is_file():
        raise FileNotFoundError(f"WT FASTA not found at {path}")
    with open(path, encoding="utf-8") as f:
        header: str | None = None
        seq_parts: list[str] = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    return header, "".join(seq_parts)
                header = line[1:]
            elif header is not None:
                seq_parts.append(line)
        if header is None:
            raise ValueError(f"no records in {path}")
        return header, "".join(seq_parts)


def _encode_msa_string(
    s: str, *, alphabet: str = "-ACDEFGHIKLMNPQRSTVWY"
) -> np.ndarray:
    """Encode a string to int64 indices in the project alphabet.

    Non-canonical characters → 0 (gap); a warning is logged for any
    character that isn't in the alphabet, since silently mapping
    arbitrary text to gap could mask data-cleaning bugs.
    """
    table = {c: i for i, c in enumerate(alphabet)}
    out = np.empty(len(s), dtype=np.int64)
    unknown: set[str] = set()
    for i, c in enumerate(s):
        if c in table:
            out[i] = table[c]
        else:
            out[i] = 0
            unknown.add(c)
    if unknown:
        log.warning(
            "WT FASTA contains non-canonical characters %s — mapped to gap",
            sorted(unknown),
        )
    return out


def _wt_msa_row(wt_fasta: Path, expected_L: int) -> np.ndarray:
    """Encode the first record in the WT FASTA, validating L."""
    header, seq = _read_first_fasta_record(wt_fasta)
    if len(seq) != expected_L:
        raise ValueError(
            f"WT FASTA record length {len(seq)} != model L {expected_L}. "
            f"First record header: {header!r}. "
            f"If this run uses a different reference, pass --mpnn-wt-fasta."
        )
    log.info("WT anchor: %r (len %d)", header.split()[0], len(seq))
    return _encode_msa_string(seq)


# ── Sampling ────────────────────────────────────────────────────────────


def sample_temperatures(
    model: dict,
    temperatures: list[float],
    n_per_t: int,
    master_seed: int,
    delta_t: int,
) -> dict[float, dict[str, Any]]:
    """Sample one alignment per temperature.

    Mirrors ``sample_sbm.py``'s per-T loop: ``t_seed = master_seed + i``,
    ``np.random.seed(t_seed)`` for reproducibility, then call
    ``Create_modAlign``.

    Returns a dict keyed by temperature with values
    ``{"align": ndarray(N, L), "seed": int, "wall_seconds": float,
       "started_at": str, "finished_at": str}``.
    """
    out: dict[float, dict[str, Any]] = {}
    for i, T in enumerate(temperatures):
        t_seed = master_seed + i
        log.info(
            "sampling T=%g, N=%d, seed=%d, delta_t=%d", T, n_per_t, t_seed, delta_t
        )
        np.random.seed(t_seed)
        started = dt.datetime.now(dt.timezone.utc)
        t0 = time.perf_counter()
        align = ut.Create_modAlign(
            model, n_per_t, delta_t=delta_t, temperature=T, seed=t_seed
        )
        wall = time.perf_counter() - t0
        finished = dt.datetime.now(dt.timezone.utc)
        out[float(T)] = {
            "align": align,
            "seed": t_seed,
            "wall_seconds": float(wall),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
        }
    return out


# ── Controls ────────────────────────────────────────────────────────────


def build_random_control(
    *, n: int, L: int, scorable_cols: np.ndarray, wt_msa: np.ndarray, seed: int
) -> np.ndarray:
    """Uniform random over 20 AAs at scorable columns; WT residues elsewhere.

    Non-scorable columns (WT-gap columns) are filled with the WT row's
    own value (i.e. gap). This mirrors what the structure "sees": only
    the scorable cols feed MPNN.
    """
    rng = np.random.default_rng(seed)
    out = np.tile(wt_msa, (n, 1)).astype(np.int64)
    rand_aa = rng.integers(low=1, high=21, size=(n, scorable_cols.size), dtype=np.int64)
    out[:, scorable_cols] = rand_aa
    return out


def build_shuffled_wt_control(*, n: int, wt_msa: np.ndarray, seed: int) -> np.ndarray:
    """Per-row permutation of WT non-gap residues at non-gap columns.

    Composition is preserved per row; structural information is
    destroyed. Gap columns stay as gaps.
    """
    rng = np.random.default_rng(seed)
    nongap_cols = np.where(wt_msa != 0)[0]
    nongap_aas = wt_msa[nongap_cols]
    out = np.tile(wt_msa, (n, 1)).astype(np.int64)
    for i in range(n):
        perm = rng.permutation(nongap_aas)
        out[i, nongap_cols] = perm
    return out


def build_natural_bootstrap(
    *, n: int, msa_natural: np.ndarray | None, seed: int
) -> np.ndarray:
    """Sample with replacement from the natural training MSA."""
    if msa_natural is None or msa_natural.size == 0:
        raise RuntimeError(
            "model['align'] is missing or empty; cannot bootstrap natural "
            "sequences. Drop 'natural' from --mpnn-controls, or re-train so "
            "the natural alignment is stored in model.npy."
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, msa_natural.shape[0], size=n)
    return msa_natural[idx].astype(np.int64)


def build_controls(
    *,
    requested: list[str],
    n_per_group: int,
    L: int,
    wt_msa: np.ndarray,
    msa_natural: np.ndarray | None,
    scorable_cols: np.ndarray,
    master_seed: int,
) -> dict[str, dict[str, Any]]:
    """Materialize each requested control as a (N, L) int64 alignment.

    Returns a dict keyed by control name. ``"none"`` short-circuits
    (returns an empty dict). Each entry is
    ``{"align": ndarray, "seed": int | None, "kind": "control"}``.
    """
    if "none" in requested:
        if len(requested) > 1:
            raise ValueError(
                "--mpnn-controls 'none' is exclusive; cannot combine with other "
                f"control names: {requested}"
            )
        return {}
    out: dict[str, dict[str, Any]] = {}
    if "wt" in requested:
        out["wt"] = {"align": wt_msa.reshape(1, -1).astype(np.int64), "seed": None}
    if "random" in requested:
        s = master_seed + _CONTROL_SEED_OFFSETS["random"]
        out["random"] = {
            "align": build_random_control(
                n=n_per_group,
                L=L,
                scorable_cols=scorable_cols,
                wt_msa=wt_msa,
                seed=s,
            ),
            "seed": s,
        }
    if "shuffled" in requested:
        s = master_seed + _CONTROL_SEED_OFFSETS["shuffled"]
        out["shuffled"] = {
            "align": build_shuffled_wt_control(n=n_per_group, wt_msa=wt_msa, seed=s),
            "seed": s,
        }
    if "natural" in requested:
        s = master_seed + _CONTROL_SEED_OFFSETS["natural"]
        out["natural"] = {
            "align": build_natural_bootstrap(
                n=n_per_group, msa_natural=msa_natural, seed=s
            ),
            "seed": s,
        }
    return out


# ── Disk layout ─────────────────────────────────────────────────────────


def _format_temperature(T: float) -> str:
    """Same convention as ``sample_sbm.py._format_temperature``."""
    return f"{T:.10g}"


def _sweep_dir(run_dir: Path, master_seed: int) -> Path:
    return run_dir / "synthetic" / f"mpnn_sweep_seed{master_seed}"


def _per_T_paths(sweep_dir: Path, T: float, t_seed: int) -> tuple[Path, Path]:
    stem = f"align_T{_format_temperature(T)}_seed{t_seed}"
    return sweep_dir / f"{stem}.npy", sweep_dir / f"{stem}.json"


def _control_path(sweep_dir: Path, name: str, seed: int | None) -> Path:
    if name == "wt":
        return sweep_dir / "control_wt.npy"
    return sweep_dir / f"control_{name}_seed{seed}.npy"


def write_alignments(
    sweep_dir: Path,
    *,
    samples: dict[float, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    run_dir: Path,
    model_path: Path,
    master_seed: int,
    mode: str,
    delta_t: int,
) -> dict[str, Path]:
    """Materialize the alignment .npy files + per-T sidecars on disk.

    Per-T sidecars use the same schema as ``scripts/sample_sbm.py`` so a
    user pointing the standard renderer at one of these files would Just
    Work. Returns a name → path index for the manifest.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    model_sha = provenance.file_sha256(model_path)
    code_block = {
        "git_commit": provenance.git_commit(),
        "git_dirty": provenance.git_dirty(),
        "git_branch": provenance.git_branch(),
    }

    for i, (T, entry) in enumerate(sorted(samples.items())):
        align: np.ndarray = entry["align"]
        t_seed: int = entry["seed"]
        npy_path, json_path = _per_T_paths(sweep_dir, T, t_seed)
        np.save(npy_path, align)
        sidecar = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "alignment_path": str(npy_path),
            "alignment_sha256": provenance.file_sha256(npy_path),
            "alignment_shape": list(align.shape),
            "alignment_dtype": str(align.dtype),
            "mode": mode,
            "N": int(align.shape[0]),
            "temperature": float(T),
            "delta_t": int(delta_t),
            "seed": int(t_seed),
            "master_seed": int(master_seed),
            "temperature_index": i,
            "label": "mpnn_sweep",
            "started_at": entry["started_at"],
            "finished_at": entry["finished_at"],
            "wall_seconds": float(entry["wall_seconds"]),
            "code": code_block,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written[f"sample_T{_format_temperature(T)}"] = npy_path

    for name, entry in controls.items():
        path = _control_path(sweep_dir, name, entry["seed"])
        np.save(path, entry["align"])
        written[f"control_{name}"] = path

    return written


# ── Manifest assembly ───────────────────────────────────────────────────


def write_sweep_manifest(
    sweep_dir: Path,
    *,
    run_dir: Path,
    args: argparse.Namespace,
    samples: dict[float, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    written: dict[str, Path],
    pdb_path: Path | None,
    mpnn_meta: dict[str, Any] | None,
    msa_to_pdb: dict[str, Any] | None,
    bench: dict[str, Any],
    started_at: dt.datetime,
    finished_at: dt.datetime,
    master_seed: int,
    mode: str,
    delta_t: int,
) -> Path:
    """Write ``manifest.json`` for the sweep and ``bench.json`` next to it."""
    inputs: dict[str, Path | str | None] = {
        "model": run_dir / "model.npy",
        "wt_fasta": Path(args.mpnn_wt_fasta) if args.mpnn_wt_fasta else None,
        "pdb": pdb_path,
    }
    sample_meta = {
        f"T{_format_temperature(T)}": {
            "seed": entry["seed"],
            "alignment_path": str(written[f"sample_T{_format_temperature(T)}"]),
            "wall_seconds": float(entry["wall_seconds"]),
            "shape": list(entry["align"].shape),
        }
        for T, entry in sorted(samples.items())
    }
    control_meta = {
        name: {
            "seed": entry["seed"],
            "alignment_path": str(written[f"control_{name}"]),
            "shape": list(entry["align"].shape),
        }
        for name, entry in controls.items()
    }
    extra: dict[str, Any] = {
        "sweep_kind": "mpnn_foldability",
        "temperatures": [float(T) for T in sorted(samples.keys())],
        "n_per_T": int(args.mpnn_N_per_T),
        "controls": list(controls.keys()),
        "samples": sample_meta,
        "control_artifacts": control_meta,
        "msa_to_pdb_map": msa_to_pdb,
        "mpnn": mpnn_meta,
        "bench": bench,
    }
    manifest = provenance.build_run_manifest(
        run_id=sweep_dir.name,
        command_line=provenance.current_command_line(),
        inputs=inputs,
        options={
            "Model": mode,
            "delta_t": delta_t,
            "master_seed": master_seed,
            "mpnn_skip_scoring": bool(args.mpnn_skip_scoring),
            "mpnn_device": args.mpnn_device,
        },
        seed=master_seed,
        started_at=started_at,
        finished_at=finished_at,
        output_path=sweep_dir,
        omp_threads_requested=provenance.omp_threads_requested(),
        extra=extra,
    )
    manifest_path = sweep_dir / "manifest.json"
    provenance.save_run_manifest(manifest, manifest_path)

    bench_path = sweep_dir / "bench.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return manifest_path


def write_scores_json(
    sweep_dir: Path,
    *,
    samples: dict[float, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    sample_scores: dict[float, dict[str, Any]] | None,
    control_scores: dict[str, dict[str, Any]] | None,
    L: int,
    scorable_cols: np.ndarray,
    pdb_meta: dict[str, Any] | None,
    mpnn_meta: dict[str, Any] | None,
    bench: dict[str, Any],
) -> Path:
    """Write the consolidated ``mpnn_scores.json`` for the figure plotter."""

    def _group_for_sample(T: float) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": "sample",
            "temperature": float(T),
            "seed": int(samples[T]["seed"]),
            "n": int(samples[T]["align"].shape[0]),
        }
        if sample_scores is not None and T in sample_scores:
            s = sample_scores[T]
            entry["scores"] = [float(x) for x in s["scores"]]
            entry["extra_residue_counts"] = [int(x) for x in s["extra_residue_counts"]]
            entry["wall_seconds"] = float(s.get("wall_seconds", 0.0))
        return entry

    def _group_for_control(name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": "control",
            "name": name,
            "seed": (
                int(controls[name]["seed"])
                if controls[name]["seed"] is not None
                else None
            ),
            "n": int(controls[name]["align"].shape[0]),
        }
        if control_scores is not None and name in control_scores:
            s = control_scores[name]
            entry["scores"] = [float(x) for x in s["scores"]]
            entry["extra_residue_counts"] = [int(x) for x in s["extra_residue_counts"]]
            entry["wall_seconds"] = float(s.get("wall_seconds", 0.0))
        return entry

    groups: dict[str, dict[str, Any]] = {}
    for T in sorted(samples.keys()):
        groups[f"T{_format_temperature(T)}"] = _group_for_sample(T)
    for name in controls.keys():
        groups[name] = _group_for_control(name)

    payload = {
        "schema_version": 1,
        "alphabet": "-ACDEFGHIKLMNPQRSTVWY",
        "L": int(L),
        "scorable_cols": [int(c) for c in scorable_cols],
        "pdb": pdb_meta,
        "mpnn": mpnn_meta,
        "groups": groups,
        "bench": bench,
    }
    out_path = sweep_dir / "mpnn_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path


# ── Pre-existing-sweep guard ────────────────────────────────────────────


_TEMP_SUFFIX_RE = re.compile(r"^align_T[\d.eE+-]+_seed-?\d+\.(npy|json)$")


def _existing_artifacts(sweep_dir: Path) -> list[Path]:
    """Return the union of files we'd write that already exist."""
    if not sweep_dir.is_dir():
        return []
    candidates: list[Path] = []
    for p in sweep_dir.iterdir():
        if p.is_file() and (
            _TEMP_SUFFIX_RE.match(p.name)
            or p.name.startswith("control_")
            or p.name in {"manifest.json", "bench.json", "mpnn_scores.json"}
        ):
            candidates.append(p)
    return candidates


# ── Entry point ─────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    """Execute one ProteinMPNN foldability sweep.

    Wired in from ``scripts/sample_sbm.py`` when ``--mpnn-sweep`` is set.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"error: {run_dir} is not a directory")

    repo_root = Path(__file__).resolve().parent.parent

    model_path = run_dir / "model.npy"
    manifest_path = run_dir / "manifest.json"
    if not model_path.is_file():
        raise SystemExit(f"error: no model at {model_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"error: no manifest at {manifest_path}")
    model = np.load(model_path, allow_pickle=True).item()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Mode (BM/SBM) — same fallback chain as sample_sbm.py.
    raw_mode = manifest.get("options", {}).get("Model")
    if raw_mode is None:
        raw_mode = model.get("options0", {}).get("Model")
    if raw_mode is None:
        raise SystemExit("error: could not determine Model (BM/SBM)")
    mode = str(raw_mode).upper()

    # Master seed (CLI > manifest).
    if args.seed is not None:
        master_seed = int(args.seed)
    else:
        manifest_seed = manifest.get("seed")
        if manifest_seed is None:
            raise SystemExit(
                "error: no seed in manifest and --seed not given; refusing to "
                "sample with an unseeded RNG"
            )
        master_seed = int(manifest_seed)

    # delta_t
    delta_t = (
        int(args.delta_t)
        if args.delta_t is not None
        else int(model["options0"]["k_MCMC"])
    )

    temperatures: list[float] = (
        list(args.mpnn_temperatures)
        if args.mpnn_temperatures
        else list(_DEFAULT_TEMPERATURES)
    )
    # Validate up front so a typo doesn't waste a sampling pass before
    # the error surfaces. Each check matches a sample seed = master_seed
    # + i ordering, so the duplicate check uses the formatted form to
    # catch values that round to the same on-disk filename.
    for T in temperatures:
        if not (T > 0):
            raise SystemExit(f"error: --mpnn-temperatures must be positive; got {T}")
    formatted = [_format_temperature(T) for T in temperatures]
    if len(set(formatted)) != len(formatted):
        raise SystemExit(
            "error: --mpnn-temperatures contains duplicates after "
            f"formatting: {formatted}"
        )
    n_per_t = int(args.mpnn_N_per_T)
    requested_controls: list[str] = list(args.mpnn_controls)
    # Validate control list before any sampling, since ``build_controls``
    # would only catch ``none`` mixing after the per-T sampling loop ran.
    if "none" in requested_controls and len(requested_controls) > 1:
        raise SystemExit(
            "error: --mpnn-controls 'none' is exclusive; cannot combine "
            f"with other control names: {requested_controls}"
        )

    # Sweep dir + collision check.
    sweep_dir = _sweep_dir(run_dir, master_seed)
    if not args.force:
        existing = _existing_artifacts(sweep_dir)
        if existing:
            shown = ", ".join(p.name for p in existing[:5])
            more = " …" if len(existing) > 5 else ""
            raise SystemExit(
                f"error: refusing to overwrite existing sweep artifacts in "
                f"{sweep_dir} ({shown}{more}). Pass --force to overwrite."
            )

    # WT MSA row.
    wt_fasta = (
        Path(args.mpnn_wt_fasta).expanduser().resolve()
        if args.mpnn_wt_fasta
        else (repo_root / _DEFAULT_WT_FASTA_RELPATH).resolve()
    )
    L = int(np.asarray(model["h"]).shape[0])
    wt_msa = _wt_msa_row(wt_fasta, expected_L=L)

    # PDB.
    pdb_path = (
        Path(args.mpnn_pdb).expanduser().resolve()
        if args.mpnn_pdb
        else (repo_root / _DEFAULT_PDB_RELPATH).resolve()
    )
    if not pdb_path.is_file():
        raise SystemExit(
            f"error: PDB not found at {pdb_path}. Place 1ECM.pdb under "
            "data/structures/ or pass --mpnn-pdb."
        )

    # Build the MSA→PDB map up front: even with --mpnn-skip-scoring, we
    # need ``scorable_cols`` to build the random control properly.
    from SBM.utils import mpnn_score

    pdb_seq, pdb_resnums = mpnn_score.read_pdb_chain_seq(
        pdb_path, chain=args.mpnn_chain
    )
    scorable_cols, pdb_idx_for_col = mpnn_score.build_msa_to_pdb_map(wt_msa, pdb_seq)

    started_at = dt.datetime.now(dt.timezone.utc)

    # ── Sample.
    log.info(
        "sampling: %d temperatures × %d sequences (master_seed=%d)",
        len(temperatures),
        n_per_t,
        master_seed,
    )
    samples = sample_temperatures(model, temperatures, n_per_t, master_seed, delta_t)

    # ── Controls.
    # Only convert ``model['align']`` when 'natural' is actually
    # requested — np.asarray(None) produces an opaque TypeError.
    raw_natural = model.get("align")
    if "natural" in requested_controls and raw_natural is not None:
        msa_natural: np.ndarray | None = np.asarray(raw_natural, dtype=np.int64)
    else:
        msa_natural = None
    controls = build_controls(
        requested=requested_controls,
        n_per_group=n_per_t,
        L=L,
        wt_msa=wt_msa,
        msa_natural=msa_natural,
        scorable_cols=scorable_cols,
        master_seed=master_seed,
    )

    # ── Write alignments before scoring so the disk artifacts exist
    # even if MPNN errors halfway through.
    written = write_alignments(
        sweep_dir,
        samples=samples,
        controls=controls,
        run_dir=run_dir,
        model_path=model_path,
        master_seed=master_seed,
        mode=mode,
        delta_t=delta_t,
    )

    # ── Score (optional).
    sample_scores: dict[float, dict[str, Any]] | None = None
    control_scores: dict[str, dict[str, Any]] | None = None
    mpnn_meta: dict[str, Any] | None = None
    pdb_meta: dict[str, Any] = {
        "path": str(pdb_path),
        "sha256": provenance.file_sha256(pdb_path),
        "chain": args.mpnn_chain,
        "n_residues": len(pdb_seq),
        "first_resnum": int(pdb_resnums[0]) if pdb_resnums else None,
        "last_resnum": int(pdb_resnums[-1]) if pdb_resnums else None,
    }
    bench: dict[str, Any] = {
        "sample_seconds_per_T": {
            _format_temperature(T): float(samples[T]["wall_seconds"])
            for T in sorted(samples.keys())
        },
        "sample_seconds_total": float(sum(s["wall_seconds"] for s in samples.values())),
    }

    if not args.mpnn_skip_scoring:
        mpnn_path = (
            Path(args.mpnn_path).expanduser().resolve() if args.mpnn_path else None
        )
        mpnn_python = (
            Path(args.mpnn_python).expanduser().resolve() if args.mpnn_python else None
        )
        ctx = mpnn_score.mpnn_context(
            mpnn_path=mpnn_path,
            model_name=args.mpnn_model_name,
            device=args.mpnn_device,
            python_executable=mpnn_python,
            backbone_noise=float(args.mpnn_backbone_noise),
        )
        mpnn_meta = {
            "path": str(ctx.mpnn_path),
            "model_name": ctx.model_name,
            "weights_path": str(ctx.weights_path),
            "weights_sha256": ctx.weights_sha256,
            "git_commit": ctx.mpnn_git_commit,
            "device": ctx.device,
            "python_executable": str(ctx.python_executable),
            "backbone_noise": ctx.backbone_noise,
        }

        # Score samples.
        sample_scores = {}
        score_seconds_per_T: dict[str, float] = {}
        for T in sorted(samples.keys()):
            align = samples[T]["align"]
            sequences = mpnn_score.align_to_pdb_strings(
                align,
                scorable_cols=scorable_cols,
                pdb_idx_for_col=pdb_idx_for_col,
                n_pdb=len(pdb_seq),
                pdb_default_seq=pdb_seq,
            )
            stem = f"score_T{_format_temperature(T)}"
            scores, extras = mpnn_score.score_sequences(
                ctx,
                pdb_path,
                args.mpnn_chain,
                sequences,
                out_dir=sweep_dir / "mpnn_tmp" / stem,
            )
            sample_scores[T] = {
                "scores": scores,
                "extra_residue_counts": mpnn_score.count_extra_residues(align, wt_msa),
                "wall_seconds": float(extras["wall_seconds"]),
            }
            score_seconds_per_T[_format_temperature(T)] = float(extras["wall_seconds"])
        bench["score_seconds_per_T"] = score_seconds_per_T
        bench["score_seconds_total_samples"] = float(sum(score_seconds_per_T.values()))

        # Score controls.
        control_scores = {}
        score_seconds_per_control: dict[str, float] = {}
        for name, entry in controls.items():
            align = entry["align"]
            sequences = mpnn_score.align_to_pdb_strings(
                align,
                scorable_cols=scorable_cols,
                pdb_idx_for_col=pdb_idx_for_col,
                n_pdb=len(pdb_seq),
                pdb_default_seq=pdb_seq,
            )
            scores, extras = mpnn_score.score_sequences(
                ctx,
                pdb_path,
                args.mpnn_chain,
                sequences,
                out_dir=sweep_dir / "mpnn_tmp" / f"score_{name}",
            )
            control_scores[name] = {
                "scores": scores,
                "extra_residue_counts": mpnn_score.count_extra_residues(align, wt_msa),
                "wall_seconds": float(extras["wall_seconds"]),
            }
            score_seconds_per_control[name] = float(extras["wall_seconds"])
        bench["score_seconds_per_control"] = score_seconds_per_control
        bench["score_seconds_total_controls"] = float(
            sum(score_seconds_per_control.values())
        )
        bench["score_seconds_total"] = (
            bench["score_seconds_total_samples"] + bench["score_seconds_total_controls"]
        )

    finished_at = dt.datetime.now(dt.timezone.utc)
    bench["sweep_wall_seconds"] = (finished_at - started_at).total_seconds()

    # ── Provenance / score JSON.
    write_scores_json(
        sweep_dir,
        samples=samples,
        controls=controls,
        sample_scores=sample_scores,
        control_scores=control_scores,
        L=L,
        scorable_cols=scorable_cols,
        pdb_meta=pdb_meta if not args.mpnn_skip_scoring else None,
        mpnn_meta=mpnn_meta,
        bench=bench,
    )

    msa_to_pdb_meta = {
        "n_scorable": int(scorable_cols.size),
        "n_wt_nongap": int(np.sum(wt_msa != 0)),
        "n_pdb_residues": int(len(pdb_seq)),
        "scorable_cols_first10": [int(c) for c in scorable_cols[:10]],
    }
    write_sweep_manifest(
        sweep_dir,
        run_dir=run_dir,
        args=args,
        samples=samples,
        controls=controls,
        written=written,
        pdb_path=pdb_path,
        mpnn_meta=mpnn_meta,
        msa_to_pdb=msa_to_pdb_meta,
        bench=bench,
        started_at=started_at,
        finished_at=finished_at,
        master_seed=master_seed,
        mode=mode,
        delta_t=delta_t,
    )

    log.info("sweep written: %s", sweep_dir)
    print(f"Sweep: {sweep_dir}")
    return 0
