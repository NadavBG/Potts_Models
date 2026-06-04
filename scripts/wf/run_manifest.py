"""Snakemake wrapper: aggregate per-stage provenance into run_manifest.json.

One file at the run-dir top level that answers "which params produced which
figure": it points at the frozen config snapshot, the training manifest, the
figure sources.json, and records per-stage timings + git state.
"""

import json
from pathlib import Path

from _common import load_cfg_from_snakemake, setup_stage_logging

from SBM import provenance

setup_stage_logging(snakemake, "run_manifest")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

run_root = Path(snakemake.params.run_root)  # noqa: F821


def _file_ref(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": provenance.file_sha256(path)}


timings_dir = run_root / "logs" / "timings"
timings: dict[str, float] = {}
if timings_dir.is_dir():
    for tf in sorted(timings_dir.glob("*.json")):
        rec = json.loads(tf.read_text(encoding="utf-8"))
        timings[rec.get("stage", tf.stem)] = rec.get("elapsed_sec")

synthetic = sorted((run_root / "synthetic").glob("align_T*.npy")) if (run_root / "synthetic").is_dir() else []

manifest = {
    "schema_version": 1,
    "run_name": cfg.run_name,
    "run_root": str(run_root),
    "description": cfg.description,
    "seed": cfg.seed,
    "code": {
        "git_commit": provenance.git_commit(),
        "git_dirty": provenance.git_dirty(),
        "git_branch": provenance.git_branch(),
    },
    "env": provenance.env_block(provenance.omp_threads_requested()),
    "artifacts": {
        "config_snapshot": _file_ref(run_root / "config_snapshot.yaml"),
        "model": _file_ref(run_root / "model.npy"),
        "train_manifest": _file_ref(run_root / "manifest.json"),
        "msa_stats_figure": _file_ref(run_root / "msa_stats.pdf"),
        "figure_sources": _file_ref(run_root / "figs" / "inputs" / "sources.json"),
        "synthetic_alignments": [_file_ref(p) for p in synthetic],
    },
    "stage_timings_sec": timings,
}

out = Path(snakemake.output[0])  # noqa: F821
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
