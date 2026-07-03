"""Post-hoc parameter filtering: derive a new Potts model from a trained one.

Loads the ``(J, h)`` of a trained model and keeps a subset of the parameters —
fields only (``J`` zeroed), couplings only (``h`` zeroed), or a mask-selected
subset of either — then re-applies the zero-sum gauge. Pure numpy: no MCMC, no
training. The derived model is written as an ordinary ``model.npy`` dict so the
sampler, figure renderer, and combine pipeline consume it unchanged.

Filtering keeps the dense model's *already-fit* ``h`` and zeros/masks ``J``;
this is a different energy function from a retrained profile model (whose ``h``
is re-fit to single-site statistics with ``J≡0``). See ``docs`` / ``CLAUDE.md``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from SBM.utils.utils import Zero_Sum_Gauge

log = logging.getLogger(__name__)

#: Training-replicate / timing artifacts that are meaningless for a filtered
#: model and are read nowhere downstream; dropped rather than carried stale.
_DROP_KEYS = ("W_all", "Seeds", "Execution times")


def apply_filter(
    J: np.ndarray,
    h: np.ndarray,
    *,
    zero_J: bool = False,
    mask_J: np.ndarray | None = None,
    zero_h: bool = False,
    mask_h: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the filtered ``(J, h)``, re-gauged to zero-sum.

    Per block, at most one action applies (mask wins over the ``zero`` flag is
    disallowed — pass exactly one):

    * couplings: ``zero_J`` → ``J := 0``; else ``mask_J`` given → ``J *= mask_J``
      (``mask_J`` is a ``(L, L, q, q)`` 0/1 array, 1 = keep); else keep.
    * fields: ``zero_h`` → ``h := 0``; else ``mask_h`` given → ``h *= mask_h``
      (``mask_h`` is a ``(L, q)`` 0/1 array); else keep.

    ``J`` is always returned as a full ``(L, L, q, q)`` array (zeros, never
    ``None``) so ``SBM.energy.model.load_model``'s shape checks pass. With
    ``J ≡ 0`` the gauge's J-derived correction to ``h`` vanishes, so a
    fields-only model preserves ``h`` up to its own per-site centering.
    """
    if zero_J and mask_J is not None:
        raise ValueError("apply_filter: pass either zero_J or mask_J for couplings, not both")
    if zero_h and mask_h is not None:
        raise ValueError("apply_filter: pass either zero_h or mask_h for fields, not both")

    J_out = np.array(J, dtype=np.float64, copy=True)
    h_out = np.array(h, dtype=np.float64, copy=True)

    if zero_J:
        J_out = np.zeros_like(J_out)
    elif mask_J is not None:
        m = np.asarray(mask_J)
        if m.shape != J_out.shape:
            raise ValueError(f"coupling mask shape {m.shape} != J shape {J_out.shape}")
        J_out = J_out * m

    if zero_h:
        h_out = np.zeros_like(h_out)
    elif mask_h is not None:
        m = np.asarray(mask_h)
        if m.shape != h_out.shape:
            raise ValueError(f"field mask shape {m.shape} != h shape {h_out.shape}")
        h_out = h_out * m

    J_zg, h_zg = Zero_Sum_Gauge(J_out, h_out)
    return np.asarray(J_zg, dtype=np.float64), np.asarray(h_zg, dtype=np.float64)


def build_derived_dict(
    source: dict[str, Any],
    J_new: np.ndarray,
    h_new: np.ndarray,
    *,
    provenance_note: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the derived ``model.npy`` dict.

    Copies the source dict (so ``options0``/``options1``/``Train``/``Test``/
    ``align`` carry through — same MSA), overwrites the load-bearing ``J``/``h``,
    replaces the training-trajectory ``J_norm`` with a single truthful point (the
    derived model's mean Frobenius coupling norm; 0 for fields-only), drops stale
    replicate artifacts, and records the derivation in ``options1``.
    """
    out = dict(source)
    out["J"] = np.asarray(J_new, dtype=np.float64)
    out["h"] = np.asarray(h_new, dtype=np.float64)

    # J_norm is a per-iteration training trajectory (utils_plot Coupling_evol
    # reads it as (N_av, 1 + N_records); the leading column is a placeholder).
    # There was no training here, so store one final point = the derived
    # model's actual mean Frobenius coupling norm.
    fro = float(np.mean(np.linalg.norm(out["J"], ord="fro", axis=(2, 3))))
    out["J_norm"] = np.array([[0.0, fro]])
    out["J_norm_iters"] = [0]

    for key in _DROP_KEYS:
        out.pop(key, None)

    opts1 = dict(out.get("options1", {}) or {})
    opts1.update(provenance_note)
    out["options1"] = opts1
    return out
