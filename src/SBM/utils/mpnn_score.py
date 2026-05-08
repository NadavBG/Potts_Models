"""ProteinMPNN-based foldability scoring of synthetic alignments.

This module is a thin wrapper around the upstream
``dauparas/ProteinMPNN`` repository's ``protein_mpnn_run.py --score_only``
mode. ProteinMPNN is not pip-installable (the upstream is a flat
directory of Python files, no ``pyproject.toml``); we treat it as an
external tool and invoke it via subprocess, so this module never imports
any ProteinMPNN code itself. The upstream tool brings its own ``torch``
install.

The MSA alphabet here is the project-wide
``"-ACDEFGHIKLMNPQRSTVWY"`` (gap = 0). When mapping an MSA row to a
PDB-length sequence for MPNN, gaps at scored positions are imputed as
``"X"`` (MPNN handles unknowns), and the WT-anchor PDB column ↔ MSA
column map is built from a pairwise alignment between the PDB chain
sequence and the MSA WT row — the PDB may be a truncation of the WT
(e.g. 1ECM is missing the WT's first 3 residues, ``TSE``).

Usage::

    ctx = mpnn_context(mpnn_path=Path("../ProteinMPNN"), model_name="v_48_020")
    pdb_seq, pdb_chain = read_pdb_chain_seq(pdb_path, chain="A")
    scorable_cols, pdb_idx = build_msa_to_pdb_map(wt_msa_arr, pdb_seq)
    sequences = align_to_pdb_strings(align, scorable_cols, n_pdb=len(pdb_seq))
    scores, extras = score_sequences(ctx, pdb_path, "A", sequences, out_dir=tmp)
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

#: Project-wide MSA alphabet, gap at index 0. Mirrors ``utils.py``.
ALPHABET: str = "-ACDEFGHIKLMNPQRSTVWY"

#: ProteinMPNN's internal alphabet (no gap; ``X`` for unknown).
#: Upstream: ``protein_mpnn_utils.py``, the ``ALPHABET`` constant.
MPNN_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MPNNContext:
    """Pointer + provenance for an upstream ProteinMPNN install.

    Carries everything ``score_sequences`` needs (paths, model name,
    device hint) and everything we want recorded in the sweep manifest
    (weights sha256, upstream git commit). Does **not** load any
    PyTorch model — scoring is delegated to a subprocess that loads
    its own model.
    """

    mpnn_path: Path
    model_name: str
    weights_path: Path
    weights_sha256: str
    mpnn_git_commit: str | None
    # ``None`` means "let upstream auto-detect" (cuda → mps → cpu); a
    # string pins the subprocess via ``--device``. We never probe torch
    # in this (parent) process — that would couple the Potts_Models env
    # to torch and would also lie when the parent and the scoring
    # subprocess run in different envs.
    device: str | None
    python_executable: Path
    backbone_noise: float = 0.0


# ── Path / device / hashing helpers ─────────────────────────────────────


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_at(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None
    return out.stdout.strip() or None


def _resolve_mpnn_path(explicit: Path | None) -> Path:
    """Return a validated ProteinMPNN clone path.

    Priority: ``explicit`` (from CLI) > ``PROTEINMPNN_PATH`` env >
    raise. Validates by checking for ``protein_mpnn_run.py`` and
    ``protein_mpnn_utils.py`` at the root.
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("PROTEINMPNN_PATH")
        if env is None:
            raise RuntimeError(
                "ProteinMPNN path not supplied. Pass --mpnn-path or set "
                "PROTEINMPNN_PATH=/path/to/ProteinMPNN (a clone of "
                "https://github.com/dauparas/ProteinMPNN)."
            )
        p = Path(env).expanduser().resolve()
    if not (p / "protein_mpnn_run.py").is_file():
        raise FileNotFoundError(
            f"{p} does not look like a ProteinMPNN clone "
            "(no protein_mpnn_run.py). Expected a checkout of "
            "https://github.com/dauparas/ProteinMPNN."
        )
    if not (p / "protein_mpnn_utils.py").is_file():
        raise FileNotFoundError(
            f"{p}/protein_mpnn_utils.py is missing — clone may be incomplete."
        )
    return p


def _resolve_python_executable(explicit: Path | None) -> Path:
    """Return the Python interpreter to use for ProteinMPNN subprocesses.

    Priority: ``explicit`` (from CLI) > ``PROTEINMPNN_PYTHON`` env >
    ``sys.executable``. Validates that the path exists and is
    executable, but does not probe for ``torch`` — that is the
    subprocess's job (and would couple this codebase to torch).
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("PROTEINMPNN_PYTHON")
        candidate = (
            Path(env).expanduser().resolve() if env else Path(sys.executable).resolve()
        )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"ProteinMPNN python interpreter not found at {candidate}. "
            "Pass --mpnn-python or set PROTEINMPNN_PYTHON to a Python "
            "executable in an env that has torch installed."
        )
    if not os.access(candidate, os.X_OK):
        raise PermissionError(
            f"{candidate} is not executable; cannot use as the "
            "ProteinMPNN subprocess interpreter."
        )
    return candidate


def mpnn_context(
    mpnn_path: Path | None = None,
    *,
    model_name: str = "v_48_020",
    device: str | None = None,
    python_executable: Path | None = None,
    backbone_noise: float = 0.0,
) -> MPNNContext:
    """Validate a ProteinMPNN install and return a context object.

    ``model_name`` is the basename of the weights file in
    ``<mpnn_path>/vanilla_model_weights/``. ``v_48_020`` is the
    soluble-protein model with 0.20 Å training noise (the most-cited
    default for foldability scoring).

    ``device`` is passed through verbatim: ``None`` (default) means the
    subprocess auto-detects (upstream's ``cuda → mps → cpu`` ladder); a
    string is forwarded as ``--device <value>``. We deliberately do not
    probe torch in this (parent) process.
    """
    repo = _resolve_mpnn_path(mpnn_path)
    weights_path = repo / "vanilla_model_weights" / f"{model_name}.pt"
    if not weights_path.is_file():
        available = sorted(
            p.stem for p in (repo / "vanilla_model_weights").glob("*.pt")
        )
        raise FileNotFoundError(
            f"weights file not found at {weights_path}. "
            f"Available models in this clone: {available}"
        )
    return MPNNContext(
        mpnn_path=repo,
        model_name=model_name,
        weights_path=weights_path,
        weights_sha256=_file_sha256(weights_path),
        mpnn_git_commit=_git_commit_at(repo),
        device=device,
        python_executable=_resolve_python_executable(python_executable),
        backbone_noise=backbone_noise,
    )


# ── PDB / WT alignment ──────────────────────────────────────────────────


def read_pdb_chain_seq(pdb_path: Path, chain: str = "A") -> tuple[str, list[int]]:
    """Return the chain's residue 1-letter sequence and PDB residue numbers.

    Uses Biopython's ``PDBParser``. Skips heteroatoms (``hetflag != ' '``);
    keeps only standard residues. ``X`` is substituted for non-canonical
    residues so the alignment downstream still works (rare in deposited
    structures).
    """
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("query", str(pdb_path))
    for model in struct:
        if chain not in [c.id for c in model]:
            continue
        ch = model[chain]
        residues = [r for r in ch if r.id[0] == " "]
        seq = "".join(seq1(r.resname, custom_map={}, undef_code="X") for r in residues)
        resnums = [int(r.id[1]) for r in residues]
        return seq, resnums
    raise ValueError(f"chain {chain!r} not found in {pdb_path}")


def msa_row_to_str(row: np.ndarray) -> str:
    """Decode an integer MSA row to a string using ``ALPHABET``."""
    if row.ndim != 1:
        raise ValueError(f"expected 1-D row, got shape {row.shape}")
    return "".join(ALPHABET[int(c)] for c in row)


def build_msa_to_pdb_map(
    wt_msa: np.ndarray,
    pdb_seq: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map MSA columns to PDB residue indices via pairwise alignment.

    The PDB chain sequence may be a truncation (or have missing density)
    relative to the WT MSA row — for 1ECM/CM the PDB lacks the WT's
    first three residues, ``TSE``. We can't assume
    ``len(pdb_seq) == np.sum(wt_msa != 0)``.

    Returns
    -------
    scorable_cols
        1-D ``int64`` array of MSA columns that align to a PDB residue.
    pdb_idx_for_col
        1-D ``int64`` array, same length as ``scorable_cols``: the
        PDB-residue index (0-based, into the chain sequence returned by
        ``read_pdb_chain_seq``) corresponding to each scorable MSA column.

    Both arrays have length equal to the number of matched residues
    between WT and PDB.
    """
    from Bio import Align

    wt_str = msa_row_to_str(wt_msa)
    nongap_cols = np.where(wt_msa != 0)[0]
    wt_nogap = "".join(wt_str[c] for c in nongap_cols)

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    # Simple match/mismatch: we just want substring-or-near-substring
    # alignment of pdb_seq inside wt_nogap. Strong gap penalty so we
    # don't fragment the alignment over the occasional residue mismatch.
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    alignments = aligner.align(wt_nogap, pdb_seq)
    if len(alignments) == 0:
        raise RuntimeError(
            "could not align PDB chain to WT MSA row — "
            f"WT non-gap (len {len(wt_nogap)}) vs PDB (len {len(pdb_seq)})."
        )
    aln = alignments[0]
    # Iterate aligned blocks: aln.aligned is a pair of tuples, one per
    # sequence, each a list of (start, end) ranges that correspond to
    # the matched/mismatched aligned blocks (gaps are the spaces between).
    wt_blocks, pdb_blocks = aln.aligned

    scorable_cols: list[int] = []
    pdb_idx_for_col: list[int] = []
    for (w0, w1), (p0, p1) in zip(wt_blocks, pdb_blocks):
        if (w1 - w0) != (p1 - p0):
            # Should not happen for global alignments where both ranges
            # come from the same matched span, but be defensive.
            raise RuntimeError(
                "alignment block length mismatch — got "
                f"WT[{w0}:{w1}] vs PDB[{p0}:{p1}]"
            )
        for k in range(w1 - w0):
            wt_pos = w0 + k  # index into wt_nogap
            pdb_pos = p0 + k
            # Keep only matches (skip mismatches) so a single divergent
            # residue does not pull a column into the scorable set with
            # a wrong amino-acid identity.
            if wt_nogap[wt_pos] == pdb_seq[pdb_pos]:
                scorable_cols.append(int(nongap_cols[wt_pos]))
                pdb_idx_for_col.append(int(pdb_pos))

    if not scorable_cols:
        raise RuntimeError(
            "no matched residues between PDB and WT MSA row. "
            "Check the chain id and that the PDB matches the MSA family."
        )
    log.info(
        "MSA↔PDB map: %d scorable columns / %d WT non-gap / %d PDB residues",
        len(scorable_cols),
        len(nongap_cols),
        len(pdb_seq),
    )
    return (
        np.asarray(scorable_cols, dtype=np.int64),
        np.asarray(pdb_idx_for_col, dtype=np.int64),
    )


def align_to_pdb_strings(
    align: np.ndarray,
    scorable_cols: np.ndarray,
    pdb_idx_for_col: np.ndarray,
    *,
    n_pdb: int,
    pdb_default_seq: str | None = None,
) -> list[str]:
    """Encode each MSA row as a PDB-length sequence string for MPNN scoring.

    For each row, we walk ``scorable_cols`` and place the row's residue
    at the corresponding PDB index. Positions in the PDB chain that
    don't have a matched MSA column (e.g. PDB-internal gaps where WT
    diverges) are filled from ``pdb_default_seq`` if given, else "X".
    Sampled gap (MSA index 0) at a scorable col → "X".
    """
    if align.ndim != 2:
        raise ValueError(f"expected 2-D alignment, got shape {align.shape}")
    if scorable_cols.shape != pdb_idx_for_col.shape:
        raise ValueError(
            "scorable_cols and pdb_idx_for_col must have the same shape; "
            f"got {scorable_cols.shape} vs {pdb_idx_for_col.shape}"
        )
    default = list(pdb_default_seq) if pdb_default_seq is not None else ["X"] * n_pdb
    if len(default) != n_pdb:
        raise ValueError(f"pdb_default_seq length {len(default)} != n_pdb {n_pdb}")

    out: list[str] = []
    for row in align:
        chars = list(default)  # copy per row
        for col, pdb_idx in zip(scorable_cols, pdb_idx_for_col):
            aa_idx = int(row[col])
            chars[pdb_idx] = "X" if aa_idx == 0 else ALPHABET[aa_idx]
        out.append("".join(chars))
    return out


def count_extra_residues(align: np.ndarray, wt_msa: np.ndarray) -> np.ndarray:
    """Per-row count of non-gap residues at WT-gap columns.

    These are positions where the model placed a residue but the PDB
    has nothing to score against. They are silently dropped from the
    score; this counter is recorded so a high rate flags up that the
    sample is "outside" the structural support.
    """
    wt_gap_cols = np.where(wt_msa == 0)[0]
    if wt_gap_cols.size == 0:
        return np.zeros(align.shape[0], dtype=np.int64)
    return (align[:, wt_gap_cols] != 0).sum(axis=1).astype(np.int64)


# ── Scoring via subprocess to upstream protein_mpnn_run.py ──────────────


def score_sequences(
    ctx: MPNNContext,
    pdb_path: Path,
    chain: str,
    sequences: Sequence[str],
    *,
    out_dir: Path,
) -> tuple[np.ndarray, dict]:
    """Run upstream ``protein_mpnn_run.py --score_only`` over a list of seqs.

    Each entry in ``sequences`` must be a single string of length equal
    to the PDB chain length (one letter per residue, ``X`` for unknown).
    The upstream script iterates entries in an outer loop (one forward
    pass per entry). ``--num_seq_per_target`` and upstream's
    ``--batch_size`` are pinned to ``1`` here — they control *replicate*
    forward passes of the same sequence (relevant only with nonzero
    backbone noise), not fasta-fanout, and any other combination trips a
    divide-to-zero at upstream line 60.

    Returns
    -------
    scores
        ``float64`` array of shape ``(len(sequences),)`` — mean negative
        log-likelihood per residue under MPNN's conditional distribution
        given the backbone (lower = more designable).
    extras
        ``{"wall_seconds": float, "wall_per_seq": float, "stdout": str,
        "stderr": str, "score_files": [str]}``.

    Notes
    -----
    The upstream CLI is the documented public surface (Dauparas et al.
    2022 supplement; ``protein_mpnn_run.py --help``). If the upstream
    repo changes the ``--score_only`` output layout, only this function
    needs adjustment.
    """
    pdb_path = Path(pdb_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any leftover ``score_only/`` from a prior run with a
    # different sequence count: ``_read_mpnn_scores`` enforces an exact
    # file count match, so stale files from a longer earlier sweep
    # would surface as a misleading "expected N got M" error rather
    # than a clean re-score.
    score_dir = out_dir / "score_only"
    if score_dir.exists():
        shutil.rmtree(score_dir)

    # Write the input fasta. One record per sequence; MPNN reads this
    # via --path_to_fasta and pairs each entry against pdb_path.
    fasta_path = out_dir / "sequences_to_score.fa"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")

    # The upstream script wants a JSONL describing chain layout for the
    # PDB. With a simple one-chain target it is enough to omit
    # ``--jsonl_path`` and pass ``--pdb_path`` + ``--pdb_path_chains``;
    # the script then builds the chain_id_dict internally.
    cmd = [
        str(ctx.python_executable),
        str(ctx.mpnn_path / "protein_mpnn_run.py"),
        "--pdb_path",
        str(pdb_path),
        "--pdb_path_chains",
        chain,
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        "1",
        "--batch_size",
        "1",
        "--score_only",
        "1",
        "--path_to_fasta",
        str(fasta_path),
        "--model_name",
        ctx.model_name,
        "--backbone_noise",
        f"{ctx.backbone_noise}",
    ]
    if ctx.device is not None:
        cmd.extend(["--device", ctx.device])

    log.info(
        "ProteinMPNN scoring: %d sequence(s) (one forward pass per entry)",
        len(sequences),
    )
    log.debug("running: %s", " ".join(cmd))
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ctx.mpnn_path),
    )
    finished = time.perf_counter()

    if proc.returncode != 0:
        raise RuntimeError(
            f"protein_mpnn_run.py failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    # Upstream writes one .npz per fasta entry under
    # <out_folder>/score_only/, named after the pdb stem and entry index.
    # Tolerate either .npz or .npy and either flat or per-entry layouts.
    if not score_dir.is_dir():
        raise RuntimeError(
            f"protein_mpnn_run.py reported success but {score_dir} is missing.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    score_files = sorted(list(score_dir.glob("*.npz")) + list(score_dir.glob("*.npy")))
    if not score_files:
        raise RuntimeError(f"no score files under {score_dir}")

    scores = _read_mpnn_scores(score_files, n_seqs=len(sequences))

    return (
        scores,
        {
            "wall_seconds": float(finished - started),
            "wall_per_seq": float((finished - started) / max(len(sequences), 1)),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "score_files": [str(p) for p in score_files],
        },
    )


def _read_mpnn_scores(score_files: list[Path], *, n_seqs: int) -> np.ndarray:
    """Concatenate per-file scores into a length-n_seqs array.

    The upstream layout has shifted between releases. We try, in order:

    1. A single .npz containing key ``score`` of shape ``(n_seqs,)``.
    2. A single .npz with key ``score`` of shape ``(n_seqs, K)`` —
       collapse along axis=1 by mean (MPNN per-decoding-order replicas).
    3. One .npz per fasta entry, each with key ``score`` (scalar or 1-D).
    """
    # Case 1/2: single bundle.
    if len(score_files) == 1 and score_files[0].suffix == ".npz":
        with np.load(score_files[0], allow_pickle=False) as data:
            keys = list(data.keys())
            score_key = next((k for k in ("score", "scores") if k in keys), None)
            if score_key is None:
                raise RuntimeError(
                    f"unexpected score-file keys in {score_files[0]}: {keys}"
                )
            arr = np.asarray(data[score_key])
        if arr.ndim == 1 and arr.size == n_seqs:
            return arr.astype(np.float64)
        if arr.ndim == 2 and arr.shape[0] == n_seqs:
            return arr.mean(axis=1).astype(np.float64)
        raise RuntimeError(
            f"score array shape {arr.shape} doesn't match n_seqs={n_seqs} "
            f"in {score_files[0]}"
        )

    # Case 3: one file per entry. Sort by trailing integer in stem so we
    # match the order in which the fasta entries were written.
    #
    # Upstream's --score_only with --path_to_fasta also writes a
    # ``<pdb_stem>_pdb.npz`` containing the score of the PDB-native
    # sequence (chain residues read from the PDB itself, not from our
    # fasta). It is not one of the entries we asked about; drop it.
    score_files = [p for p in score_files if not p.stem.endswith("_pdb")]

    def _entry_index(p: Path) -> int:
        # filenames look like <pdb_stem>_seq_<i>.npz or pdb_stem_<i>.npz
        # Take the last '_'-separated integer fragment.
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    score_files_sorted = sorted(score_files, key=_entry_index)
    if len(score_files_sorted) != n_seqs:
        raise RuntimeError(
            f"expected {n_seqs} score files, got {len(score_files_sorted)}: "
            + ", ".join(p.name for p in score_files_sorted[:5])
            + (" …" if len(score_files_sorted) > 5 else "")
        )
    out = np.empty(n_seqs, dtype=np.float64)
    for i, p in enumerate(score_files_sorted):
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as data:
                keys = list(data.keys())
                score_key = next((k for k in ("score", "scores") if k in keys), None)
                if score_key is None:
                    raise RuntimeError(f"unexpected keys in {p}: {keys}")
                arr = np.asarray(data[score_key])
        else:
            arr = np.load(p, allow_pickle=False)
        if arr.ndim == 0:
            out[i] = float(arr)
        elif arr.ndim == 1:
            out[i] = float(arr.mean())
        else:
            raise RuntimeError(f"unexpected score array shape {arr.shape} in {p}")
    return out
