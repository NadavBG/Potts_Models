"""Snakemake wrapper: resolve the two referenced models into a models.json.

Records, per model, its label, run dir, model.npy path + sha256, aligned length
L, seed-MSA path, and weight — the provenance pointer the score / render stages
and the aggregate manifest rely on. Fails loudly if a referenced model is absent.
"""

import json
from pathlib import Path

import numpy as np

from _common import load_combine_cfg_from_snakemake, setup_stage_logging

from SBM import provenance
from SBM.energy.model import seed_msa_path

log = setup_stage_logging(snakemake, "resolve_models")  # noqa: F821
cfg = load_combine_cfg_from_snakemake(snakemake)  # noqa: F821

entries = []
for ref in cfg.models:
    model_path = Path(ref.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model {ref.name!r}: no model.npy at {model_path} — train it first "
            f"(run_dir={ref.run_dir})"
        )
    h = np.load(model_path, allow_pickle=True).item()["h"]
    smsa = seed_msa_path(model_path)
    entries.append({
        "name": ref.name,
        "run_dir": str(ref.run_dir),
        "model_path": str(model_path),
        "model_sha256": provenance.file_sha256(model_path),
        "L": int(h.shape[0]),
        "q": int(h.shape[1]),
        "seed_msa": str(smsa) if smsa is not None else None,
        "weight": ref.weight,
    })
    log.info("resolved model %r: L=%d, %s", ref.name, h.shape[0], model_path)

Path(snakemake.output[0]).write_text(  # noqa: F821
    json.dumps({"schema_version": 1, "models": entries}, indent=2) + "\n", encoding="utf-8"
)
