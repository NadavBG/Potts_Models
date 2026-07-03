"""TM-align wrapper: structural comparison of a predicted model to a reference.

Runs the Zhang-lab ``TMalign`` binary (built once by
``pipeline/external/build_tmalign.sh``) and parses its stdout into a
:class:`TMResult` (TM-score normalized by each chain, RMSD, aligned length,
sequence identity). Pure stdlib so it imports in the CPU ``.venv`` and tests.

Convention: call ``run_tmalign(binary, query_pdb, ref_pdb)`` so **Chain_1 =
query** (the design/natural model) and **Chain_2 = reference** (1ECM/1JNT).
The headline "TM to fold A/B" is ``tm_ref`` — TM-score normalized by the
reference length — because both references are ~91–92 residues, so TM_A and
TM_B share a normalization scale and are directly comparable. ``tm_query``
(normalized by the shorter design) is retained for detail.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

# ── Reference single-chain extraction ───────────────────────────────────────


def extract_chain(
    pdb_in: Path | str, chain_id: str, pdb_out: Path | str
) -> int:
    """Write ATOM records of a single chain (first model) to ``pdb_out``.

    Drops HETATM, waters, other chains, and (for NMR ensembles) every model
    after the first. Keeps only the blank / "A" altloc so TMalign sees one
    conformer per atom. Returns the number of CA atoms written (= residues).
    """
    pdb_out = Path(pdb_out)
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    n_ca = 0
    with open(pdb_in, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break  # first model only
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            kept.append(line)
            if line[12:16].strip() == "CA":
                n_ca += 1
    if n_ca == 0:
        raise ValueError(
            f"no chain '{chain_id}' CA atoms found in {pdb_in}"
        )
    kept.append("TER\n")
    kept.append("END\n")
    pdb_out.write_text("".join(kept), encoding="utf-8")
    return n_ca


# ── TMalign stdout parsing ──────────────────────────────────────────────────


class TMResult(NamedTuple):
    """Parsed TMalign output for one (query, reference) pair."""

    tm_query: float  # TM-score normalized by Chain_1 (query) length
    tm_ref: float  # TM-score normalized by Chain_2 (reference) length
    rmsd: float  # RMSD over aligned residues (Angstrom)
    aligned_len: int  # number of aligned residue pairs
    seq_id: float  # n_identical / n_aligned over the alignment


_ALIGNED_RE = re.compile(
    r"Aligned length\s*=\s*(\d+),\s*RMSD\s*=\s*([-\d.]+),"
    r"\s*Seq_ID\s*=\s*n_identical/n_aligned\s*=\s*([-\d.]+)"
)
_TM_RE = re.compile(
    r"TM-score\s*=\s*([-\d.]+)\s*\(if normalized by length of Chain_(\d)"
)


def parse_tmalign_stdout(text: str) -> TMResult:
    """Parse TMalign stdout text into a :class:`TMResult`.

    Raises ``ValueError`` if the expected lines are absent (a failed or
    truncated run must be loud, not silently zero).
    """
    m = _ALIGNED_RE.search(text)
    if m is None:
        raise ValueError("could not parse 'Aligned length=' line from TMalign output")
    aligned_len = int(m.group(1))
    rmsd = float(m.group(2))
    seq_id = float(m.group(3))

    tm_by_chain: dict[int, float] = {}
    for score, chain in _TM_RE.findall(text):
        tm_by_chain[int(chain)] = float(score)
    if 1 not in tm_by_chain or 2 not in tm_by_chain:
        raise ValueError(
            "could not parse both Chain_1 and Chain_2 TM-scores from TMalign output"
        )
    return TMResult(
        tm_query=tm_by_chain[1],
        tm_ref=tm_by_chain[2],
        rmsd=rmsd,
        aligned_len=aligned_len,
        seq_id=seq_id,
    )


def run_tmalign(
    binary: Path | str, query_pdb: Path | str, ref_pdb: Path | str
) -> TMResult:
    """Run ``TMalign query_pdb ref_pdb`` and return the parsed result."""
    proc = subprocess.run(
        [str(binary), str(query_pdb), str(ref_pdb)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TMalign failed (exit {proc.returncode}) on {query_pdb} vs {ref_pdb}:\n"
            f"{proc.stderr}"
        )
    return parse_tmalign_stdout(proc.stdout)
