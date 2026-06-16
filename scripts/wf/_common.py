"""Shared helpers for the Snakemake ``script:`` wrappers in this directory.

Each ``run_*.py`` wrapper is run by Snakemake with a ``snakemake`` object
injected into its namespace. These helpers (1) make the sibling
``scripts/`` directory importable so wrappers can ``import train_sbm`` /
``sample_sbm`` / ``render_figures`` / ``build_mask``, (2) turn the raw
``snakemake.config`` dict into a validated :class:`SBMRunConfig`, and (3)
route stdout/stderr/logging to the rule's per-stage log file and record a
timing JSON.

NOTE: do not add ``from __future__ import annotations`` to the wrappers —
Snakemake prepends boilerplate to ``script:`` files, so a future-import on
line 1 raises ``SyntaxError``. (This module is imported, not run as a
script, so it is exempt, but the wrappers are not.)
"""

import atexit
import json
import logging
import sys
import time
from pathlib import Path

# Make `scripts/` (the parent of this `wf/` dir) importable so wrappers can
# import the existing CLIs as modules. `build_mask` lives under `pruning/`.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (_SCRIPTS_DIR, _REPO_ROOT / "pruning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from SBM import combine_config as cc  # noqa: E402
from SBM import workflow_config as wc  # noqa: E402

#: Keys accepted on the Snakemake CLI/config that are not part of the
#: validated run schema and must be stripped before validation.
_SNAKEFILE_ONLY_KEYS = {"run_root"}


class _Tee:
    """Write to several streams at once (console + per-stage log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def load_cfg_from_snakemake(snakemake) -> wc.SBMRunConfig:
    """Validate ``snakemake.config`` into an :class:`SBMRunConfig`."""
    raw = {k: v for k, v in dict(snakemake.config).items() if k not in _SNAKEFILE_ONLY_KEYS}
    return wc.from_dict(raw)


def load_combine_cfg_from_snakemake(snakemake) -> cc.CombineRunConfig:
    """Validate ``snakemake.config`` into a :class:`CombineRunConfig` (Snakefile.combine)."""
    raw = {k: v for k, v in dict(snakemake.config).items() if k not in _SNAKEFILE_ONLY_KEYS}
    return cc.from_dict(raw)


def setup_stage_logging(snakemake, stage_name: str, level: int = logging.INFO) -> logging.Logger:
    """Tee stdout/stderr to the rule's log file and configure logging.

    Registers an atexit hook that writes ``{RUN_ROOT}/logs/timings/
    {stage_name}.json`` with the wall-clock elapsed seconds, so the
    aggregate run manifest can report per-stage timings.
    """
    log_path = Path(snakemake.log[0]) if len(snakemake.log) else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 (lives for the job)
        sys.stdout = _Tee(sys.__stdout__, handle)
        sys.stderr = _Tee(sys.__stderr__, handle)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logger = logging.getLogger(stage_name)
    logger.info("stage %r starting", stage_name)
    start = time.monotonic()

    def _write_timing() -> None:
        if log_path is None:
            return
        timings_dir = log_path.parent / "timings"
        timings_dir.mkdir(parents=True, exist_ok=True)
        (timings_dir / f"{stage_name}.json").write_text(
            json.dumps(
                {"stage": stage_name, "elapsed_sec": round(time.monotonic() - start, 3)},
                indent=2,
            ),
            encoding="utf-8",
        )

    atexit.register(_write_timing)
    return logger
