"""In-frame Potts energy — the spec §2 base case.

If a sequence is already in a model's frame (length ``L``, gaps allowed, no
indels needed) there is no latent alignment and the energy is the direct Potts
sum. This is a thin, validated wrapper over ``SBM.utils.utils.compute_energies``
(``src/SBM/utils/utils.py``), the package's existing batched implementation of
``E(S) = −Σ_i h_i(S_i) − ½ Σ_ij J_ij(S_i, S_j)``; we do not re-derive the sum.
"""

from __future__ import annotations

import numpy as np

from SBM.utils.utils import compute_energies

from .model import PottsModel


def _check_in_frame(S: np.ndarray, model: PottsModel) -> None:
    if S.shape[-1] != model.L:
        raise ValueError(
            f"in-frame energy needs a length-{model.L} sequence for model "
            f"{model.name!r}; got length {S.shape[-1]}. Use score_sequence(...) "
            "with method='map'/'marginal' to align a raw sequence first."
        )
    if S.size and (S.min() < 0 or S.max() >= model.q):
        raise ValueError(
            f"sequence has states outside [0, {model.q}) for model {model.name!r}"
        )


def potts_energy(S_in_frame: np.ndarray, model: PottsModel) -> float:
    """Exact Potts energy of one in-frame integer sequence (length ``L``)."""
    S = np.asarray(S_in_frame, dtype=np.int64)
    if S.ndim != 1:
        raise ValueError(f"potts_energy expects a 1-D sequence, got shape {S.shape}")
    _check_in_frame(S, model)
    return float(compute_energies(S, model.h, model.J)[0])


def potts_energies(S_batch: np.ndarray, model: PottsModel) -> np.ndarray:
    """Exact Potts energies for a batch of in-frame sequences, shape ``(N, L)``.

    One vectorized ``compute_energies`` call — this is the path used to score
    the ``S`` importance-sampling alignments of a single query at once.
    """
    S = np.asarray(S_batch, dtype=np.int64)
    if S.ndim != 2:
        raise ValueError(f"potts_energies expects a 2-D (N, L) batch, got shape {S.shape}")
    _check_in_frame(S, model)
    return compute_energies(S, model.h, model.J)
