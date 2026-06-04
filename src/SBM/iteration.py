"""Iteration directories for pipeline runs.

A run lives at ``results/<run_name>/iter-NNN-<tag>/`` so re-running a config
with a tweaked parameter preserves history instead of overwriting. The
index ``NNN`` is chosen by scanning siblings *here* (outside the Snakemake
DAG), so by the time Snakemake runs against an explicit ``run_root`` the
output paths are fully determined.

This is the project's analogue of ``Make_Alignment``'s iteration workflow,
minus the git-tracking layer (``results/`` is gitignored because models are
large): provenance lives in each iteration's ``manifest.json`` /
``config_snapshot.yaml`` / ``run_manifest.json`` instead.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import yaml

from SBM import provenance

_ITER_RE = re.compile(r"^iter-(\d+)-")
_DEFAULT_RESULTS_ROOT = Path("results")


def slugify(tag: str, *, max_len: int = 40) -> str:
    """kebab-case a human tag for use in an ``iter-NNN-<tag>`` dir name."""
    slug = re.sub(r"[^a-z0-9]+", "-", tag.strip().lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "run"


def next_iter_index(run_dir: Path) -> int:
    """The next ``NNN`` for ``run_dir`` (1 if none exist yet)."""
    if not run_dir.is_dir():
        return 1
    indices = [
        int(m.group(1))
        for child in run_dir.iterdir()
        if child.is_dir() and (m := _ITER_RE.match(child.name))
    ]
    return (max(indices) + 1) if indices else 1


def latest_iter(run_name: str, *, results_root: Path | str = _DEFAULT_RESULTS_ROOT) -> Path | None:
    """The most recent ``iter-NNN-*`` dir for ``run_name`` (highest NNN)."""
    run_dir = Path(results_root) / run_name
    if not run_dir.is_dir():
        return None
    iters = sorted(
        (c for c in run_dir.iterdir() if c.is_dir() and _ITER_RE.match(c.name)),
        key=lambda c: int(_ITER_RE.match(c.name).group(1)),
    )
    return iters[-1] if iters else None


def list_iters(run_name: str, *, results_root: Path | str = _DEFAULT_RESULTS_ROOT) -> list[Path]:
    """All ``iter-NNN-*`` dirs for ``run_name``, in index order."""
    run_dir = Path(results_root) / run_name
    if not run_dir.is_dir():
        return []
    return sorted(
        (c for c in run_dir.iterdir() if c.is_dir() and _ITER_RE.match(c.name)),
        key=lambda c: int(_ITER_RE.match(c.name).group(1)),
    )


def update_latest(iter_path: Path) -> Path:
    """Point ``<run_name>/latest`` at ``iter_path`` (relative symlink)."""
    link = iter_path.parent / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(iter_path.name)
    return link


def start_iteration(
    run_name: str,
    tag: str,
    *,
    config_path: Path | str | None = None,
    results_root: Path | str = _DEFAULT_RESULTS_ROOT,
) -> Path:
    """Create ``results/<run_name>/iter-NNN-<tag>/`` and its iteration note.

    Returns the new iteration directory. Does not run the pipeline; the
    caller passes ``run_root=<this path>`` to Snakemake.
    """
    run_dir = Path(results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the parent (current latest) BEFORE creating the new dir, else
    # the new dir would be found as its own parent.
    parent = latest_iter(run_name, results_root=results_root)
    idx = next_iter_index(run_dir)
    iter_id = f"iter-{idx:03d}-{slugify(tag)}"
    iter_path = run_dir / iter_id
    iter_path.mkdir(parents=True, exist_ok=False)
    frontmatter = {
        "iter_id": iter_id,
        "parent_iter": parent.name if parent is not None else None,
        "git_commit": provenance.git_commit(),
        "git_dirty": provenance.git_dirty(),
        "git_branch": provenance.git_branch(),
        "config_path": str(config_path) if config_path else None,
        "started": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "started",
    }
    note = (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False)
        + "---\n\n"
        + f"# {iter_id}\n\n"
        + "## Hypothesis\n\n"
        + f"{tag}\n\n"
        + "## Notes\n\n"
    )
    (iter_path / "iteration_note.md").write_text(note, encoding="utf-8")
    return iter_path
