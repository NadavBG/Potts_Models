"""Sample synthetic alignments from a trained SBM/BM model.

Reads ``<run_dir>/model.npy`` and ``<run_dir>/manifest.json``, runs
``SBM.utils.utils.Create_modAlign`` (the canonical sampler used in
training), and writes one alignment per requested temperature to
``<run_dir>/synthetic/`` with a JSON sidecar per file.

By default this samples *both* T=0.75 and T=1.0 — every downstream
analysis in this project compares the two regardless of whether the
model is BM or SBM, so producing them in one shot is the workflow
default. Pass ``--temperature T1 [T2 ...]`` to override the list.

The number of sequences defaults to 2000, independent of the training
MSA size.

Usage::

    python scripts/sample_sbm.py <run_dir> [options]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
import SBM.utils.utils as ut

# scripts/mpnn_sweep.py is a sibling, not a package module. Add this
# script's directory to sys.path so the plain `import mpnn_sweep` works
# regardless of the cwd from which sample_sbm.py is invoked. Mirrors the
# pattern in scripts/render_figures.py for `lab_plotting`.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

log = logging.getLogger(__name__)

#: Default temperature schedule. The project doctrine is that both T=0.75
#: (low-T, mode-collapse-friendly) and T=1.0 (the model's training
#: temperature) are useful for every run, so we sample both unless told
#: otherwise. Override per-call with ``--temperature T1 [T2 ...]``.
_DEFAULT_TEMPERATURES: tuple[float, ...] = (0.75, 1.0)

#: Default number of synthetic sequences. Decoupled from the training MSA
#: size so independent histograms (similarity / diversity / length) have
#: enough samples regardless of how small the training alignment is.
_DEFAULT_N: int = 2000


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
    model.npy's options0 is the fallback for hand-built run dirs.

    Recorded in each sidecar for provenance; no longer used to pick a
    sampling temperature.
    """
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
    ``0.001`` → ``'0.001'``. Avoids the collision-by-rounding hazard that
    a ``.2f`` formatter has at small ``T``. The renderer reads each
    alignment's sampling temperature from its sidecar JSON, not from
    the filename, so this formatter only governs filename layout here.
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
        help=f"number of synthetic sequences (default: {_DEFAULT_N})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "one or more sampling temperatures; each writes its own .npy + "
            ".json under <run_dir>/synthetic/. Default: "
            + " ".join(_format_temperature(t) for t in _DEFAULT_TEMPERATURES)
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

    # ── ProteinMPNN foldability-sweep mode ─────────────────────────────
    # An alternate code path (orchestrated by scripts/mpnn_sweep.py) that
    # samples N sequences at each of a temperature ladder, builds
    # interpretability controls (WT, random, shuffled WT, natural MSA
    # bootstrap), and (optionally) scores all of them with the upstream
    # ProteinMPNN repository. Outputs land under
    # <run_dir>/synthetic/mpnn_sweep_seed<seed>/. Disabled unless
    # --mpnn-sweep is set; mixing with the standard sampling flags is a
    # hard error to avoid surprising clobbers.
    mpnn_group = parser.add_argument_group("ProteinMPNN sweep (alternate mode)")
    mpnn_group.add_argument(
        "--mpnn-sweep",
        action="store_true",
        help=(
            "Sample N sequences at each --mpnn-temperature, build "
            "interpretability controls, and score everything against a "
            "PDB with ProteinMPNN. Mutually exclusive with the standard "
            "--temperature/--N/--output/--label sampling flags."
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-temperatures",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help="temperature ladder for the sweep (default: 0.1 0.2 ... 1.0)",
    )
    mpnn_group.add_argument(
        "--mpnn-N-per-T",
        type=int,
        default=100,
        help="sequences per temperature and per control group (default: 100)",
    )
    mpnn_group.add_argument(
        "--mpnn-controls",
        nargs="+",
        choices=["wt", "random", "shuffled", "natural", "none"],
        default=["wt", "random", "shuffled", "natural"],
        help=(
            "interpretability controls to score alongside the temperature "
            "ladder. 'none' disables all controls. (default: all four)"
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-pdb",
        type=Path,
        default=None,
        help="PDB to score against (default: data/structures/1ECM.pdb)",
    )
    mpnn_group.add_argument(
        "--mpnn-chain",
        type=str,
        default="A",
        help="PDB chain id to score (default: A)",
    )
    mpnn_group.add_argument(
        "--mpnn-wt-fasta",
        type=Path,
        default=None,
        help=(
            "FASTA whose first record is the WT MSA-aligned reference "
            "sequence (default: data/fasta/CM.fasta — its first record "
            "is the 1ECM reference)"
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-path",
        type=Path,
        default=None,
        help=(
            "path to a clone of github.com/dauparas/ProteinMPNN. "
            "Defaults to the PROTEINMPNN_PATH environment variable. "
            "Ignored under --mpnn-skip-scoring."
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-python",
        type=Path,
        default=None,
        help=(
            "Python interpreter used to invoke ProteinMPNN's "
            "protein_mpnn_run.py (the env where torch is installed). "
            "Defaults to the PROTEINMPNN_PYTHON environment variable, "
            "else sys.executable. Set this to point at a separate "
            "conda/venv env so torch does not need to live in the "
            "Potts_Models env. Ignored under --mpnn-skip-scoring."
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-model-name",
        type=str,
        default="v_48_020",
        help=(
            "ProteinMPNN weights basename in vanilla_model_weights/ "
            "(default: v_48_020 — the soluble model with 0.20 Å noise)"
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help=(
            "torch device for ProteinMPNN. Default: auto-detect " "(cuda → mps → cpu)."
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-backbone-noise",
        type=float,
        default=0.0,
        help=(
            "augmentation epsilon passed to ProteinMPNN at scoring time. "
            "0.0 reproduces the deterministic-backbone score; nonzero "
            "values match a noise-augmented training regime."
        ),
    )
    mpnn_group.add_argument(
        "--mpnn-skip-scoring",
        action="store_true",
        help=(
            "produce sample alignments + controls + manifest without "
            "running ProteinMPNN. Useful for smoke tests or hosts "
            "without an MPNN clone."
        ),
    )

    args = parser.parse_args(argv)

    # Dispatch into the MPNN sweep before any of the standard-sampling
    # validation runs, so the standard flags' defaults don't trigger
    # spurious errors. Mixing modes is rejected up front because the
    # two paths write to different locations and have non-overlapping
    # parameter semantics.
    if args.mpnn_sweep:
        conflicts = [
            name
            for name, val in (
                ("--N", args.N),
                ("--temperature", args.temperature),
                ("--output", args.output),
                ("--label", args.label),
            )
            if val is not None
        ]
        if conflicts:
            parser.error(
                "--mpnn-sweep is mutually exclusive with "
                f"{', '.join(conflicts)}. Use --mpnn-temperatures and "
                "--mpnn-N-per-T to control the sweep, and --mpnn-pdb / "
                "--mpnn-path / --mpnn-skip-scoring to configure scoring."
            )
        import mpnn_sweep  # noqa: E402  (sibling script; sys.path set above)

        return mpnn_sweep.run(args)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")

    model = _load_model(run_dir)
    manifest = _load_manifest(run_dir)
    mode = _resolve_mode(model, manifest)

    N = args.N if args.N is not None else _DEFAULT_N
    temperatures: list[float] = (
        list(args.temperature)
        if args.temperature is not None
        else list(_DEFAULT_TEMPERATURES)
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

    if args.output is not None and (
        args.temperature is None or len(args.temperature) > 1
    ):
        # --output picks a fixed path; with multiple temperatures we'd
        # write to it once per T, clobbering on each iteration. Require
        # the user to pin a single T explicitly so they're aware they're
        # opting out of the dual-T workflow.
        parser.error(
            "--output requires a single --temperature value (got "
            + (
                "none — defaults to two temperatures"
                if args.temperature is None
                else f"{len(args.temperature)})"
            )
            + "). Pass --temperature T explicitly, or omit --output to "
            "let sample_sbm.py write one .npy per default T."
        )

    written: list[Path] = []
    for i, temperature in enumerate(temperatures):
        # Per-T seed: each temperature gets its own deterministic seed
        # derived from the master, so chains at different T are
        # independent draws (not just the same configuration nudged by
        # acceptance ratio). seed + i is enough — it keeps "remove one T,
        # the rest are bit-identical" reproducibility.
        t_seed = seed + i

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
                run_dir, temperature=temperature, seed=t_seed, label=args.label
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
            t_seed,
        )

        started_at = dt.datetime.now(dt.timezone.utc)
        # Seed the global RNG so any internal np.random use inside
        # Create_modAlign (and the C++ kernels via the seed path) is
        # reproducible. The per-T seed produces independent chains.
        np.random.seed(t_seed)
        align = ut.Create_modAlign(
            model, N, delta_t=delta_t, temperature=temperature, seed=t_seed
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
            "seed": t_seed,
            "master_seed": seed,
            "temperature_index": i,
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
        written.append(out_path)

    if len(written) > 1:
        log.info(
            "sampled %d temperatures: %s",
            len(written),
            ", ".join(p.name for p in written),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
