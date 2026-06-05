#! /usr/bin/env python3
"""Encode an aligned FASTA into the project's integer MSA ``.npy``.

The FASTA is the immutable source of truth; this CLI produces the derived
integer array that every other tool consumes (``train_sbm.py``,
``build_mask.py``, ``render_msa_stats.py``). Sequences carrying residues
outside the canonical alphabet (``-ACDEFGHIKLMNPQRSTVWY``) are dropped;
the count and record IDs are logged at WARNING and recorded in a
manifest sidecar so the drop is never silent.

Usage::

    python scripts/encode_msa.py --fasta data/fasta/CM.fasta --out msa.npy
        [--manifest PATH] [--run-id STR]

The Snakemake ``encode_msa`` rule calls this via ``scripts/wf/run_encode_msa.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np

from SBM import provenance
from SBM.utils.utils import MSA_ALPHABET, load_fasta

log = logging.getLogger("encode_msa")


def encode(fasta: Path, out_npy: Path, manifest_path: Path, run_id: str) -> dict:
    """Encode ``fasta`` → ``out_npy`` and write ``manifest_path``. Returns the manifest."""
    started = dt.datetime.now(dt.timezone.utc)
    msa, dropped_ids = load_fasta(str(fasta), dtype=np.int64, return_dropped=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, msa)
    finished = dt.datetime.now(dt.timezone.utc)

    n_kept = int(msa.shape[0])
    manifest = provenance.build_run_manifest(
        run_id=run_id,
        command_line=provenance.current_command_line(),
        inputs={"msa_fasta": fasta},
        options={"alphabet": MSA_ALPHABET, "dtype": str(msa.dtype)},
        seed=None,  # encoding is deterministic; no RNG
        started_at=started,
        finished_at=finished,
        output_path=out_npy,
        extra={
            "n_records": n_kept + len(dropped_ids),
            "n_kept": n_kept,
            "n_dropped": len(dropped_ids),
            "dropped_record_ids": dropped_ids,
            "L": int(msa.shape[1]),
        },
    )
    provenance.save_run_manifest(manifest, manifest_path)
    log.info("encoded %s -> %s (%d x %d)", fasta, out_npy, n_kept, msa.shape[1])
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, required=True, help="input aligned FASTA")
    parser.add_argument("--out", type=Path, required=True, help="output integer MSA .npy")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest sidecar path (default: <out>.manifest.json)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run_id recorded in the manifest (default: derived from --out)",
    )
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_npy = args.out if str(args.out).endswith(".npy") else args.out.with_suffix(".npy")
    manifest_path = args.manifest or out_npy.with_suffix(".manifest.json")
    run_id = args.run_id or f"encode_msa/{out_npy.stem}"
    encode(args.fasta, out_npy, manifest_path, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
