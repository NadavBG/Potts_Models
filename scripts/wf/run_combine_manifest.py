"""Snakemake wrapper: aggregate combine-run provenance into run_manifest.json."""

import json
from pathlib import Path

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM import provenance

setup_stage_logging(snakemake, "run_manifest")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821
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

manifest = {
    "schema_version": 1,
    "run_name": cfg.run_name,
    "run_kind": "combine",
    "run_root": str(run_root),
    "description": cfg.description,
    "seed": cfg.seed,
    "models": [{"name": m.name, "run_dir": m.run_dir} for m in cfg.models],
    "scoring": {"method": cfg.scoring.method, "n_samples": cfg.scoring.n_samples,
                "ess_threshold": cfg.scoring.ess_threshold},
    "code": {
        "git_commit": provenance.git_commit(),
        "git_dirty": provenance.git_dirty(),
        "git_branch": provenance.git_branch(),
    },
    "env": provenance.env_block(provenance.omp_threads_requested()),
    "artifacts": {
        # Top-level contract files (cluster reads these) + the human note.
        "config_snapshot": _file_ref(run_root / "config_snapshot.yaml"),
        "models": _file_ref(run_root / "models.json"),
        "query_fasta": _file_ref(run_root / "query" / "query.fasta"),
        # Tables live under data/, manifests under provenance/ (run-dir tidy).
        "scores": _file_ref(run_root / "data" / "scores.tsv"),
        "scores_detail": _file_ref(run_root / "data" / "scores_detail.json"),
        "alignments": _file_ref(run_root / "data" / "alignments.txt"),
        "score_manifest": _file_ref(run_root / "provenance" / "score_manifest.json"),
        # E_tot combining weights derived from the naturals (compute_weights).
        "energy_weights": _file_ref(run_root / "data" / "energy_weights.json"),
        "energy_weight_sweep": _file_ref(run_root / "data" / "energy_weight_sweep.tsv"),
        "energy_weights_figure": _file_ref(run_root / "figs" / "energy_weights.pdf"),
        "energy_figure": _file_ref(run_root / "figs" / "two_model_energy.pdf"),
        # potts_align ground-state-recovery baseline (only present for
        # method=potts_align; None otherwise).
        "potts_align_baseline_table": _file_ref(run_root / "data" / "potts_align_vs_inframe.tsv"),
        "potts_align_baseline_summary": _file_ref(run_root / "data" / "potts_align_vs_inframe.json"),
        "potts_align_baseline_figure": _file_ref(run_root / "figs" / "potts_align_vs_inframe.pdf"),
    },
    "stage_timings_sec": timings,
}

Path(snakemake.output[0]).write_text(  # noqa: F821
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
