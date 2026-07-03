"""Snakemake wrapper: copy the source model's encoded MSA into the derive run dir.

A derived model dir must carry its own ``inputs/msa.npy`` so the combine
pipeline finds the naturals + seed MSA (``resolve_models`` / ``datasets``) and
so a coupling/field mask can be built from the same statistics the source model
was trained on. No re-encoding: the source's ``inputs/msa.npy`` is copied
verbatim. The ``msa_manifest.json`` sidecar is copied if present, else a small
stub recording the provenance of the copy is written so the declared output
always exists.
"""

import json
import logging
import shutil
from pathlib import Path

from _common import load_derive_cfg_from_snakemake, setup_stage_logging

from SBM import provenance

log = setup_stage_logging(snakemake, "copy_inputs")  # noqa: F821 (injected by Snakemake)
cfg = load_derive_cfg_from_snakemake(snakemake)  # noqa: F821

src_msa = Path(cfg.source_msa_path)
if not src_msa.is_file():
    raise FileNotFoundError(
        f"source model dir has no encoded MSA at {src_msa}; the derive pipeline "
        "copies the source's inputs/msa.npy (train the source model first)"
    )

out_npy = Path(snakemake.output.npy)  # noqa: F821
out_manifest = Path(snakemake.output.manifest)  # noqa: F821
out_npy.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src_msa, out_npy)
log.info("copied seed MSA: %s -> %s", src_msa, out_npy)

src_manifest = src_msa.parent / "msa_manifest.json"
if src_manifest.is_file():
    shutil.copyfile(src_manifest, out_manifest)
    log.info("copied MSA manifest: %s", src_manifest)
else:
    log.warning("source has no msa_manifest.json; writing a stub for %s", out_manifest)
    out_manifest.write_text(
        json.dumps(
            {
                "note": "copied from source model dir; source had no msa_manifest.json",
                "source_msa": str(src_msa),
                "source_msa_sha256": provenance.file_sha256(src_msa),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
