#!/usr/bin/env python3
"""Mint and run pipeline iterations.

    python scripts/iter.py new  <run_name> "<tag>" [--config PATH]
    python scripts/iter.py run  <run_name> "<tag>" [--config PATH] [--cores N] [-- <snakemake args>]
    python scripts/iter.py list <run_name>
    python scripts/iter.py latest <run_name>

`new` creates results/<run_name>/iter-NNN-<tag>/ and prints the Snakemake
command to run it. `run` does both (mint + invoke Snakemake). The config
defaults to config/params_<run_name>.yaml.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from SBM import iteration


def _default_config(run_name: str) -> Path:
    return Path("config") / f"params_{run_name}.yaml"


def _mint(args: argparse.Namespace) -> tuple[Path, Path]:
    cfg = Path(args.config) if args.config else _default_config(args.run_name)
    if not cfg.is_file():
        sys.exit(f"error: config not found: {cfg}")
    iter_path = iteration.start_iteration(args.run_name, args.tag, config_path=cfg)
    iteration.update_latest(iter_path)
    print(f"Created {iter_path}")
    return iter_path, cfg


def _snakemake_cmd(cfg: Path, iter_path: Path, cores: int, extra: list[str]) -> list[str]:
    return [
        sys.executable, "-m", "snakemake",
        "--configfile", str(cfg),
        "--config", f"run_root={iter_path}",
        "--cores", str(cores),
        "all", *extra,
    ]


def _cmd_new(args: argparse.Namespace) -> int:
    iter_path, cfg = _mint(args)
    print("Run it with:")
    print("  " + " ".join(_snakemake_cmd(cfg, iter_path, args.cores, [])))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    iter_path, cfg = _mint(args)
    cmd = _snakemake_cmd(cfg, iter_path, args.cores, args.snakemake_args)
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def _cmd_list(args: argparse.Namespace) -> int:
    iters = iteration.list_iters(args.run_name)
    latest = iteration.latest_iter(args.run_name)
    if not iters:
        print(f"(no iterations under results/{args.run_name}/)")
        return 0
    for p in iters:
        marker = "  <- latest" if latest and p.name == latest.name else ""
        print(f"{p}{marker}")
    return 0


def _cmd_latest(args: argparse.Namespace) -> int:
    p = iteration.latest_iter(args.run_name)
    print(p if p is not None else "(none)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mint and run pipeline iterations.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("new", "run"):
        p = sub.add_parser(name, help=f"{name} an iteration")
        p.add_argument("run_name", help="run name (also the config stem)")
        p.add_argument("tag", help="short hypothesis/label for the iteration dir")
        p.add_argument("--config", default=None, help="config path (default: config/params_<run_name>.yaml)")
        p.add_argument("--cores", type=int, default=8, help="cores for Snakemake (default: 8)")
        if name == "run":
            p.add_argument(
                "snakemake_args",
                nargs="*",
                help="extra args forwarded to snakemake (e.g. -n, --rerun-incomplete)",
            )
            p.set_defaults(func=_cmd_run)
        else:
            p.set_defaults(func=_cmd_new)

    for name in ("list", "latest"):
        p = sub.add_parser(name, help=f"{name} iterations for a run")
        p.add_argument("run_name", help="run name")
        p.set_defaults(func=_cmd_list if name == "list" else _cmd_latest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
