"""Snakemake wrapper: derive a filtered model from the source model.

Loads the source ``model.npy``, applies the configured couplings/fields filter
(keep / zero / mask), re-gauges, and writes a new ``model.npy`` in the standard
dict layout plus a ``manifest.json`` (recording the source model hash + the
filter — the derivation lineage) and a ``command.sh``.
"""

import datetime as dt
from pathlib import Path

import numpy as np

from _common import load_derive_cfg_from_snakemake, setup_stage_logging

from SBM import derive, provenance
from SBM.derive_config import KEEP, ZERO

log = setup_stage_logging(snakemake, "derive")  # noqa: F821 (injected by Snakemake)
cfg = load_derive_cfg_from_snakemake(snakemake)  # noqa: F821
run_root = Path(snakemake.params.run_root)  # noqa: F821

source_model_path = Path(cfg.source_model_path)
if not source_model_path.is_file():
    raise FileNotFoundError(
        f"no source model at {source_model_path} — train it first "
        "(the derive pipeline filters an existing model, it does not train)"
    )
source = np.load(source_model_path, allow_pickle=True).item()
J = np.asarray(source["J"], dtype=np.float64)
h = np.asarray(source["h"], dtype=np.float64)

# Translate the per-block filter directive into apply_filter arguments.
coup, flds = cfg.filter.couplings, cfg.filter.fields
zero_J = coup == ZERO
zero_h = flds == ZERO
mask_J_path = getattr(snakemake.input, "mask_J", None)  # noqa: F821 (only when a MaskSpec)
mask_h_path = getattr(snakemake.input, "mask_h", None)  # noqa: F821
mask_J = np.load(mask_J_path) if mask_J_path is not None else None
mask_h = np.load(mask_h_path) if mask_h_path is not None else None


def _describe(block) -> object:
    if block in (KEEP, ZERO):
        return block
    return {"strategy": block.strategy, "percent": block.percent}


log.info(
    "deriving from %s: couplings=%s, fields=%s",
    cfg.source_run_dir,
    _describe(coup),
    _describe(flds),
)

started = dt.datetime.now(dt.timezone.utc)
J_new, h_new = derive.apply_filter(
    J, h, zero_J=zero_J, mask_J=mask_J, zero_h=zero_h, mask_h=mask_h
)
provenance_note = {
    "Derived From": str(cfg.source_run_dir),
    "Source Model SHA256": provenance.file_sha256(source_model_path),
    "Filter Couplings": _describe(coup),
    "Filter Fields": _describe(flds),
}
out_dict = derive.build_derived_dict(source, J_new, h_new, provenance_note=provenance_note)

model_path = Path(snakemake.output.model)  # noqa: F821
np.save(model_path, out_dict)
finished = dt.datetime.now(dt.timezone.utc)
log.info("wrote derived model: %s", model_path)

manifest = provenance.build_run_manifest(
    run_id=run_root.name,
    command_line=provenance.current_command_line(),
    inputs={
        "source_model": str(source_model_path),
        "coupling_mask": (str(mask_J_path) if mask_J_path is not None else None),
        "field_mask": (str(mask_h_path) if mask_h_path is not None else None),
    },
    options={
        "source_run_dir": cfg.source_run_dir,
        "family": cfg.family,
        "filter": {
            "couplings": _describe(coup),
            "fields": _describe(flds),
            "theta": cfg.filter.theta,
            "lbda": cfg.filter.lbda,
            "label": cfg.filter.label,
            "Dia_prior": cfg.filter.Dia_prior,
        },
    },
    seed=cfg.seed,
    started_at=started,
    finished_at=finished,
    output_path=model_path,
    omp_threads_requested=provenance.omp_threads_requested(),
    extra={"derived": True, "source_model": str(source_model_path)},
)
provenance.save_run_manifest(manifest, snakemake.output.manifest)  # noqa: F821
provenance.write_command_sh(
    provenance.current_command_line(),
    snakemake.output.command,  # noqa: F821
    cwd=Path.cwd(),
)
