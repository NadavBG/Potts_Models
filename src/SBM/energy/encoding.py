"""Residue ↔ integer conversions for the Potts alphabet.

The canonical alphabet is ``SBM.utils.utils.MSA_ALPHABET`` =
``"-ACDEFGHIKLMNPQRSTVWY"`` (gap = 0, the 20 amino acids = 1..20, q = 21),
shared by the encoded MSAs, the fitted models, and the synthetic alignments.
These helpers move between that integer encoding and amino-acid strings, and
strip the gap state to recover the *raw, ungapped* query a profile HMM
re-threads into a model frame.
"""

from __future__ import annotations

import numpy as np

from SBM.utils.utils import MSA_ALPHABET

#: gap is index 0; the 20 amino acids occupy 1..20.
GAP = 0
Q = len(MSA_ALPHABET)
_AA_TO_INT = {aa: i for i, aa in enumerate(MSA_ALPHABET)}


def seq_to_ints(seq: str) -> np.ndarray:
    """Map an amino-acid string to an int array over :data:`MSA_ALPHABET`.

    Raises ``ValueError`` on any character outside the alphabet (including
    lowercase / ambiguity codes) — unlike ``load_fasta``, which drops whole
    sequences, a single query is scored or rejected loudly, never silently.
    """
    try:
        return np.array([_AA_TO_INT[ch] for ch in seq], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"sequence contains a character outside {MSA_ALPHABET!r}: {exc.args[0]!r}"
        ) from exc


def ints_to_seq(arr: np.ndarray) -> str:
    """Inverse of :func:`seq_to_ints`: int array → amino-acid string."""
    return "".join(MSA_ALPHABET[int(i)] for i in np.asarray(arr).ravel())


def strip_gaps(row: np.ndarray) -> np.ndarray:
    """Drop gap states (0) from an integer sequence, leaving residues in 1..20.

    This is how an in-frame natural/synthetic sequence (length ``L``, gapped)
    becomes the *raw ungapped* query that a profile HMM re-aligns to a model's
    frame — the only honest input for cross-family scoring.
    """
    row = np.asarray(row, dtype=np.int64)
    return row[row != GAP]
