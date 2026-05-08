"""Per-run provenance for SBM training and mask generation.

Every trained model and every pruning mask should be reproducible from a
``manifest.json`` written alongside the artifact. This module captures
the bits a future reader needs: git commit + dirty flag, command line,
input file hashes, full options dict, RNG seed, OMP thread count,
package versions, host, and timestamps.

Mirrors the figure-side pattern in ``scripts/lab_plotting.py`` (which
embeds the same provenance into PDF metadata via ``save_figure``).
The git helpers are duplicated rather than shared because the script
lives outside the package and we don't want a runtime import cycle.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json as _json
import os as _os
import platform as _platform
import secrets as _secrets
import shutil as _shutil
import socket as _socket
import subprocess as _subprocess
import sys as _sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

#: Packages whose versions are recorded in every manifest. Anything the
#: user installs with ``[plotting,analysis]`` extras isn't listed because
#: training itself doesn't depend on them.
_DEFAULT_TRACKED_PACKAGES = (
    "SBM",
    "numpy",
    "scipy",
    "biopython",
    "tqdm",
    "more-itertools",
    "matplotlib",
    "pandas",
)


# ── Git ─────────────────────────────────────────────────────────────────


def git_commit() -> str | None:
    """Current HEAD commit hash, or None if not in a git repo."""
    if _shutil.which("git") is None:
        return None
    try:
        out = _subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes."""
    if _shutil.which("git") is None:
        return False
    try:
        out = _subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def git_branch() -> str | None:
    """Current branch name, or None if detached / no repo."""
    if _shutil.which("git") is None:
        return None
    try:
        out = _subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired):
        return None
    branch = out.stdout.strip()
    return branch if branch and branch != "HEAD" else None


# ── Hashing ─────────────────────────────────────────────────────────────


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file. Returns a 64-char hex digest."""
    h = _hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """Hash an ndarray's bytes. Casts to a contiguous view first so two
    arrays with identical contents but different strides hash identically.
    """
    arr = np.ascontiguousarray(arr)
    return _hashlib.sha256(arr.tobytes()).hexdigest()


def array_summary(arr: np.ndarray) -> dict[str, Any]:
    """Manifest-friendly summary of an ndarray: shape, dtype, sha256."""
    return {
        "_kind": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": array_sha256(arr),
    }


# ── Environment ─────────────────────────────────────────────────────────


def package_versions(
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Look up installed versions for the named distributions.

    Missing packages map to ``"unknown"`` rather than raising. The default
    set covers SBM's runtime dependencies; pass ``names`` to widen it.
    """
    if names is None:
        names = _DEFAULT_TRACKED_PACKAGES
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


def env_block(omp_threads_requested: int | None = None) -> dict[str, Any]:
    """Snapshot of the runtime: Python, platform, host, OMP threads,
    package versions.

    ``omp_threads_requested`` is the value the *caller* asked for (typically
    the parsed ``OMP_NUM_THREADS`` env var). It is **not** necessarily what
    OpenMP actually used at runtime — OpenMP can clamp under
    ``OMP_DYNAMIC`` or platform limits. The C++ kernels do not currently
    report ``omp_get_max_threads()`` back, so we record only the request.
    Bit-identical reproduction across machines requires both runs to land
    on the same actual thread count, which usually means setting
    ``OMP_NUM_THREADS`` to a value the hardware can satisfy on each.
    """
    return {
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "hostname": _socket.gethostname(),
        "omp_num_threads_env": _os.environ.get("OMP_NUM_THREADS"),
        "omp_num_threads_requested": omp_threads_requested,
        "package_versions": package_versions(),
    }


# ── Options sanitization ────────────────────────────────────────────────


def _sanitize(value: Any) -> Any:
    """Recursively convert a value to a JSON-safe form.

    - ndarrays → ``array_summary`` (shape/dtype/sha256, not the data)
    - Path → str
    - dict/list/tuple → recurse
    - bytes → hex
    - everything else → passes through if JSON-serializable, else ``str()``
    """
    if isinstance(value, np.ndarray):
        return array_summary(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def sanitize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Top-level wrapper around _sanitize for an options dict."""
    return _sanitize(options)


# ── Manifest assembly ───────────────────────────────────────────────────


def make_run_id(
    when: _dt.datetime | None = None,
    *,
    label: str | None = None,
    parent_dir: Path | str | None = None,
) -> str:
    """Construct a run id.

    Two formats:

    1. **Human-readable** (when ``label`` is given): ``YYYY-MM-DD_<label>_<idx>``,
       e.g. ``2026-05-05_CM-example_0``. ``idx`` is the next free integer
       under ``parent_dir`` matching ``<date>_<label>_*``; if ``parent_dir``
       is None or empty, ``idx`` starts at 0. Best-effort: a race between
       two simultaneous runs can pick the same idx, but for a research
       workflow the wall-clock spacing makes this unlikely.

    2. **Legacy timestamp** (when ``label`` is None):
       ``YYYYMMDDTHHMMSSZ-xxxxx`` — sortable UTC timestamp + 5-hex random
       suffix. Kept so existing callers don't break.

    UTC throughout. ``label`` should be filesystem-safe (no slashes); the
    function does not sanitize.
    """
    when = when or _dt.datetime.now(_dt.timezone.utc)
    if label is None:
        stamp = when.strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{_secrets.token_hex(3)[:5]}"
    base = f"{when:%Y-%m-%d}_{label}"
    if parent_dir is None:
        return f"{base}_0"
    parent = Path(parent_dir)
    if not parent.is_dir():
        return f"{base}_0"
    used: list[int] = []
    prefix = f"{base}_"
    for entry in parent.iterdir():
        if entry.name.startswith(prefix):
            suffix = entry.name[len(prefix) :]
            if suffix.isdigit():
                used.append(int(suffix))
    next_idx = (max(used) + 1) if used else 0
    return f"{base}_{next_idx}"


def _input_entry(path: Path | str | None) -> dict[str, Any]:
    """Manifest entry for an input file path. Records path + sha256.
    Returns ``{"path": None, "sha256": None}`` if path is None or missing.
    """
    if path is None:
        return {"path": None, "sha256": None}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "sha256": None, "missing": True}
    return {"path": str(p), "sha256": file_sha256(p)}


def build_run_manifest(
    *,
    run_id: str,
    command_line: list[str] | None,
    inputs: dict[str, Path | str | None],
    options: dict[str, Any],
    seed: int | None,
    started_at: _dt.datetime,
    finished_at: _dt.datetime,
    output_path: Path | str | None = None,
    omp_threads_requested: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict ready to be JSON-dumped.

    ``inputs`` maps a label (e.g. ``"msa"``, ``"train_indices"``,
    ``"pruning_mask_couplings"``, ``"pruning_mask_fields"``) to a file
    path or None. Each is hashed.

    ``output_path``, if given and existing, is also hashed so a run's
    manifest commits to the bytes it produced.
    """
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "command_line": list(command_line) if command_line is not None else None,
        "code": {
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "git_branch": git_branch(),
        },
        "env": env_block(omp_threads_requested=omp_threads_requested),
        "inputs": {label: _input_entry(p) for label, p in inputs.items()},
        "options": sanitize_options(options),
        "seed": seed,
        "started_at": started_at.astimezone(_dt.timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(_dt.timezone.utc).isoformat(),
        "wall_seconds": (finished_at - started_at).total_seconds(),
        "outputs": {
            "model": _input_entry(output_path) if output_path is not None else None,
        },
    }
    if extra:
        manifest["extra"] = _sanitize(extra)
    return manifest


def save_run_manifest(manifest: dict[str, Any], path: Path | str) -> Path:
    """Write the manifest to ``path`` as pretty-printed JSON. Creates
    parent directories. Returns the resolved path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path.resolve()


def load_run_manifest(path: Path | str) -> dict[str, Any]:
    """Read a manifest written by ``save_run_manifest``."""
    with open(path, encoding="utf-8") as f:
        return _json.load(f)


def write_command_sh(
    command_line: list[str], path: Path | str, *, cwd: Path | str | None = None
) -> Path:
    """Write a shell script that re-invokes ``command_line``.

    Args are quoted with ``shlex.quote`` so paths with spaces survive.
    A leading ``cd`` is emitted if ``cwd`` is given so the script works
    when invoked from elsewhere.
    """
    import shlex

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(shlex.quote(a) for a in command_line)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if cwd is not None:
        lines.append(f"cd {shlex.quote(str(cwd))}")
        lines.append("")
    lines.append(f"exec {quoted}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


# ── Helpers for callers ─────────────────────────────────────────────────


def omp_threads_requested() -> int | None:
    """Best-effort read of the OpenMP thread count the user asked for.

    Returns ``int(OMP_NUM_THREADS)`` if the env var is set, else None.
    This is **not** necessarily the count OpenMP actually used at runtime
    — the C++ kernels do not currently report ``omp_get_max_threads()``
    back. Recorded in the manifest under ``env.omp_num_threads_requested``.
    """
    raw = _os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def current_command_line() -> list[str]:
    """``sys.argv`` with the executable prepended, suitable for re-running."""
    return [_sys.executable, *_sys.argv]
