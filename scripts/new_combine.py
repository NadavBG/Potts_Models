#!/usr/bin/env python3
"""Scaffold a two-model combine run from two ``results/`` model dirs — one command.

Replaces the copy-paste-a-template-and-edit dance. Given two trained single-model
run dirs it:

    1. validates each has ``model.npy`` + ``inputs/msa.npy`` (loud otherwise);
    2. infers each model's name from its dir path;
    3. auto-picks the two error-prone potts_align knobs from the real data —
       ``pa_cross_subsample_origin`` = the larger-N family (the PT cost driver),
       and ``query.random_length`` = min(L_A, L_B) (must be <= both to score
       under both models);
    4. builds a full CombineRunConfig, validates it (round-trips through
       ``combine_config.from_dict`` — the same check the pipeline uses), and writes
       a clean ``config/params_combine-<run_name>.yaml`` (short header, no embedded
       runbook);
    5. mints ``combine/<run_name>/iter-NNN-<tag>/``; and
    6. writes that dir's fully-interpolated ``RUNBOOK.txt``.

Usage:

    python scripts/new_combine.py <results_dir_A> <results_dir_B> --tag "<tag>"

Common flags: ``--method map`` (no cluster), ``--no-design``,
``--no-characterize``, ``--design-local``, ``--run-name NAME``, ``--config-only``,
``--force``. Then open the printed ``RUNBOOK.txt`` and follow it block by block.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from SBM import combine_config as cc
from SBM import combine_runbook, iteration

_ITER_RE = re.compile(r"^iter-\d+-")


def _model_name(run_dir: Path) -> str:
    """Model name = the component above an ``iter-*`` dir, else the leaf.

    ``results/CM-bm-dense/iter-002-base-model`` -> ``CM-bm-dense``;
    ``results/derive-CM-profile`` -> ``derive-CM-profile``.
    """
    if _ITER_RE.match(run_dir.name):
        return run_dir.parent.name
    return run_dir.name


def _validate_model_dir(run_dir: Path) -> tuple[int, int]:
    """Return ``(L, N)`` for a trained run dir, exiting loudly on any problem.

    ``L`` and ``N`` are read from the MSA header (shape ``(N, L)``) via mmap —
    the alignment width IS the model length — so we never unpickle the ~0.5 GB
    model dict just to size the config; we only assert ``model.npy`` exists.
    """
    if not run_dir.is_dir():
        sys.exit(f"error: not a directory: {run_dir}")
    model_path = run_dir / "model.npy"
    msa_path = run_dir / "inputs" / "msa.npy"
    if not model_path.is_file():
        sys.exit(f"error: no model.npy in {run_dir} (is this a trained results/ run dir?)")
    if not msa_path.is_file():
        sys.exit(
            f"error: no inputs/msa.npy in {run_dir} "
            "(combine needs the naturals / seed MSA)"
        )
    msa = np.load(msa_path, mmap_mode="r")
    if msa.ndim != 2:
        sys.exit(f"error: {msa_path} is not a 2-D (N, L) alignment (shape {msa.shape})")
    n_seqs, length = int(msa.shape[0]), int(msa.shape[1])
    return length, n_seqs


def _build_config_dict(args: argparse.Namespace) -> tuple[dict, str, str]:
    """Assemble the raw config dict + return ``(dict, name_a, name_b)``."""
    dir_a, dir_b = Path(args.dir_a), Path(args.dir_b)
    la, na = _validate_model_dir(dir_a)
    lb, nb = _validate_model_dir(dir_b)
    name_a = args.name_a or _model_name(dir_a)
    name_b = args.name_b or _model_name(dir_b)
    if name_a == name_b:
        sys.exit(
            f"error: both models resolve to the name {name_a!r}; "
            "pass distinct --name-a / --name-b"
        )

    run_name = args.run_name or f"combine-{name_a}-{name_b}"
    min_l = min(la, lb)
    # Cross-subsample the LARGER family's cross block (the PT cost driver).
    if na >= nb:
        origin, under = name_a, name_b
    else:
        origin, under = name_b, name_a

    scoring: dict = {"method": args.method}
    query: dict = {"source": "model_sets", "include": ["natural"]}
    if args.method == "potts_align":
        scoring.update(
            n_shards=512,
            pa_cross_subsample_origin=origin,
            pa_cross_subsample_under=under,
            pa_cross_subsample_n=2000,
        )
        query.update(cap_per_group=0, n_random=500, random_length=min_l)
    else:
        # Non-potts methods score in-process; cap large naturals, skip the
        # potts-only random control.
        query.update(cap_per_group=300, n_random=0, random_length=0)

    data: dict = {
        "run_name": run_name,
        "description": f"Cross-score {name_a} and {name_b} under both models",
        "seed": args.seed,
        "omp_num_threads": None,
        "models": [
            {"name": name_a, "run_dir": str(dir_a)},
            {"name": name_b, "run_dir": str(dir_b)},
        ],
        "query": query,
        "scoring": scoring,
        "figures": {"enabled": True},
    }
    if not args.no_design:
        data["design"] = {
            "enabled": True,
            "start_random": 48,
            "start_natural_a": 24,
            "start_natural_b": 24,
            "steps": 500_000,
            "seed": 0,
            "min_length": 70,
            "polish": True,
            "polish_schedule": "fast",
            "execution": "local" if args.design_local else "cluster",
            "n_shards": 64,
        }
    if not args.no_characterize:
        data["characterize"] = {"enabled": True}
    return data, name_a, name_b


_CONFIG_HEADER = (
    "Two-model combine config generated by scripts/new_combine.py.\n"
    "The full run steps (score -> design -> characterize) with every path filled\n"
    "in live in the minted run dir's RUNBOOK.txt (regenerated on each `snakemake\n"
    "... all`). Edit fields here freely; unknown keys are rejected on load."
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scaffold a two-model combine run from two results/ dirs."
    )
    p.add_argument("dir_a", help="first trained results/ run dir (model A)")
    p.add_argument("dir_b", help="second trained results/ run dir (model B)")
    p.add_argument("--tag", required=True, help="short hypothesis/label for the iteration dir")
    p.add_argument("--run-name", default=None, help="run name (default: combine-<A>-<B>)")
    p.add_argument("--name-a", default=None, help="override model A's name")
    p.add_argument("--name-b", default=None, help="override model B's name")
    p.add_argument(
        "--method", default="potts_align", choices=list(cc._METHODS),
        help="scoring.method (default: potts_align)",
    )
    p.add_argument("--seed", type=int, default=42, help="run seed (default: 42)")
    p.add_argument("--no-design", action="store_true", help="disable the design stage")
    p.add_argument(
        "--design-local", action="store_true",
        help="run the design anneal locally (design.execution: local) instead of on Midway",
    )
    p.add_argument("--no-characterize", action="store_true", help="disable the characterize stage")
    p.add_argument(
        "--config-only", action="store_true",
        help="only write the config/ YAML; do not mint a run dir or runbook",
    )
    p.add_argument(
        "--force", action="store_true",
        help="overwrite an existing config even if it differs (an identical one is reused). "
        "Each run mints a new iteration dir regardless.",
    )
    args = p.parse_args(argv)

    raw, _name_a, _name_b = _build_config_dict(args)
    # Validate with the same checker the pipeline uses (rejects unknown keys,
    # enforces two unique models, checks cross-subsample names) BEFORE writing.
    try:
        cfg = cc.from_dict(raw)
    except cc.ConfigError as exc:
        sys.exit(f"error: generated config failed validation: {exc}")

    config_path = Path("config") / f"params_{cfg.run_name}.yaml"
    if config_path.exists() and not args.force:
        # An identical existing config is reused (so a second iteration of the same
        # run needs no --force); a DIFFERING one is refused rather than clobbered.
        try:
            existing = cc.load_config(config_path)
        except cc.ConfigError as exc:
            sys.exit(f"error: {config_path} exists but is invalid ({exc}); pass --force to overwrite")
        if existing.as_dict() != cfg.as_dict():
            sys.exit(
                f"error: {config_path} exists and differs from the generated config; "
                "pass --force to overwrite it, or --run-name NAME for a separate config"
            )
        print(f"Using existing {config_path} (identical)")
    else:
        cc.dump_config(cfg, config_path, header=_CONFIG_HEADER)
        print(f"Wrote {config_path}")

    if args.config_only:
        print("Mint + run with:")
        print(
            f'  python scripts/iter.py run {cfg.run_name} "{args.tag}" '
            f"--snakefile Snakefile.combine"
        )
        return 0

    iter_path = iteration.start_iteration(
        cfg.run_name, args.tag, config_path=config_path, results_root="combine"
    )
    iteration.update_latest(iter_path)
    print(f"Created {iter_path}")

    runbook_path = iter_path / "RUNBOOK.txt"
    runbook_path.write_text(
        combine_runbook.render_runbook(cfg, str(iter_path), str(config_path)),
        encoding="utf-8",
    )
    print(f"Wrote {runbook_path}")
    print(f"\nNext: open {runbook_path} and follow it block by block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
