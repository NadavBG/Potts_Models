"""Sample a synthetic alignment from a trained SBM/BM model.

Reads ``<run_dir>/model.npy`` and ``<run_dir>/manifest.json``, runs
``SBM.utils.utils.Create_modAlign`` (the canonical sampler used in
training), and writes the result to ``<run_dir>/synthetic/`` with a
JSON sidecar that records every sampling parameter.

Mode-aware defaults: BM samples at T=0.75 and SBM at T=1.0 (Summary
Note 3); both can be overridden with ``--temperature``. The number of
sequences defaults to the size of the training alignment so downstream
similarity / diversity histograms are directly comparable.

Usage::

    python scripts/sample_sbm.py <run_dir> [options]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
import SBM.utils.utils as ut

log = logging.getLogger(__name__)

#: Mode → default sampling temperature. Matches Summary Note 3.
_MODE_TEMPERATURE: dict[str, float] = {"BM": 0.75, "SBM": 1.0}


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_model(run_dir: Path) -> dict:
    path = run_dir / "model.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"no model at {path} — pass a run dir produced by " "scripts/train_sbm.py"
        )
    return np.load(path, allow_pickle=True).item()


def _resolve_mode(model: dict, manifest: dict) -> str:
    """BM or SBM, normalised to upper case. Manifest is authoritative;
    model.npy's options0 is the fallback for hand-built run dirs."""
    raw = manifest.get("options", {}).get("Model")
    if raw is None:
        raw = model.get("options0", {}).get("Model")
    if raw is None:
        raise ValueError(
            "could not determine Model (BM/SBM) from manifest or model.npy"
        )
    return str(raw).upper()


def _format_temperature(T: float) -> str:
    """Filename-safe, lossless-within-reason temperature token.

    ``%g`` with high precision: ``1.0`` → ``'1'``, ``0.75`` → ``'0.75'``,
    ``0.001`` → ``'0.001'``. Avoids the collision-by-rounding hazard that a
    ``.2f`` formatter has at small ``T``. Mirrors ``_format_temperature``
    in ``render_figures.py``; both must agree so auto-discovery there can
    match filenames written here.
    """
    return f"{T:.10g}"


def _default_output_path(
    run_dir: Path, *, temperature: float, seed: int, label: str | None
) -> Path:
    parts = [f"align_T{_format_temperature(temperature)}", f"seed{seed}"]
    if label:
        parts.append(label)
    return run_dir / "synthetic" / ("_".join(parts) + ".npy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample a synthetic alignment from a trained SBM/BM model."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="path to a run directory produced by scripts/train_sbm.py",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="number of synthetic sequences (default: size of training MSA)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "sampling temperature (default: 0.75 for BM, 1.0 for SBM, read from "
            "the run's manifest)"
        ),
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=None,
        help="Metropolis sweeps per chain (default: options0.k_MCMC from model.npy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master RNG seed (default: master seed from manifest)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="suffix added to the default filename",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="full output .npy path (overrides the default location under synthetic/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite existing alignment / sidecar at the output path. "
            "Default is to refuse so accidental re-invocation does not "
            "destroy a prior sample."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")

    started_at = dt.datetime.now(dt.timezone.utc)

    model = _load_model(run_dir)
    manifest = _load_manifest(run_dir)
    mode = _resolve_mode(model, manifest)

    N = args.N if args.N is not None else int(model["align"].shape[0])
    temperature = (
        args.temperature
        if args.temperature is not None
        else _MODE_TEMPERATURE.get(mode, 1.0)
    )
    delta_t = (
        args.delta_t if args.delta_t is not None else int(model["options0"]["k_MCMC"])
    )
    manifest_seed = manifest.get("seed")
    if args.seed is not None:
        seed = int(args.seed)
    elif manifest_seed is not None:
        seed = int(manifest_seed)
    else:
        raise ValueError(
            "no seed in manifest and --seed not given; refusing to sample with "
            "an unseeded RNG (reproducibility)"
        )

    if args.output is not None:
        out_path = args.output.resolve()
        # np.save appends .npy if missing; mirror that explicitly so the
        # path we hash + record in the sidecar matches the file on disk.
        if out_path.suffix != ".npy":
            out_path = (
                out_path.with_suffix(out_path.suffix + ".npy")
                if out_path.suffix
                else out_path.with_suffix(".npy")
            )
    else:
        out_path = _default_output_path(
            run_dir, temperature=temperature, seed=seed, label=args.label
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_path.with_suffix(".json")
    if not args.force:
        existing = [p for p in (out_path, sidecar_path) if p.exists()]
        if existing:
            shown = ", ".join(str(p) for p in existing)
            parser.error(
                f"refusing to overwrite existing file(s): {shown}. "
                "Pass --force to overwrite, --label to disambiguate, or "
                "--output to point elsewhere."
            )

    log.info(
        "sampling: mode=%s, N=%d, T=%s, delta_t=%d, seed=%d",
        mode,
        N,
        _format_temperature(temperature),
        delta_t,
        seed,
    )

    # Seed the global RNG so any internal np.random use inside Create_modAlign
    # (and the C++ kernels via the seed path) is reproducible.
    np.random.seed(seed)
    align = ut.Create_modAlign(
        model, N, delta_t=delta_t, temperature=temperature, seed=seed
    )

    np.save(out_path, align)
    finished_at = dt.datetime.now(dt.timezone.utc)

    sidecar = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "model_path": str(run_dir / "model.npy"),
        "model_sha256": provenance.file_sha256(run_dir / "model.npy"),
        "alignment_path": str(out_path),
        "alignment_sha256": provenance.file_sha256(out_path),
        "alignment_shape": list(align.shape),
        "alignment_dtype": str(align.dtype),
        "mode": mode,
        "N": N,
        "temperature": temperature,
        "delta_t": delta_t,
        "seed": seed,
        "label": args.label,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_seconds": (finished_at - started_at).total_seconds(),
        "code": {
            "git_commit": provenance.git_commit(),
            "git_dirty": provenance.git_dirty(),
            "git_branch": provenance.git_branch(),
        },
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log.info("wrote alignment: %s", out_path)
    log.info("wrote sidecar:   %s", sidecar_path)
    print(f"Sampled: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
