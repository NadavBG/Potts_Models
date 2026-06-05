"""Snakemake wrapper: encode the aligned input FASTA into an integer MSA.

Raw FASTA in (immutable input) → run-local ``inputs/msa.npy`` out, plus an
``inputs/msa_manifest.json`` sidecar recording the input hash, the output
hash, and exactly which sequences were dropped for non-canonical residues.
Thin wrapper over ``scripts/encode_msa.py`` (which does the work), so the
manual encoder and the pipeline share one implementation.
"""

from _common import load_cfg_from_snakemake, setup_stage_logging

import encode_msa

setup_stage_logging(snakemake, "encode_msa")  # noqa: F821 (injected by Snakemake)
cfg = load_cfg_from_snakemake(snakemake)  # noqa: F821

rc = encode_msa.main(
    [
        "--fasta", str(snakemake.input.fasta),  # noqa: F821
        "--out", str(snakemake.output.npy),  # noqa: F821
        "--manifest", str(snakemake.output.manifest),  # noqa: F821
        "--run-id", f"{cfg.run_name}/encode_msa",
    ]
)
if rc:
    raise SystemExit(rc)
