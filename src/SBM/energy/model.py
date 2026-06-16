"""Load a fitted Potts model and pin it to a documented common gauge.

A trained ``model.npy`` is a pickled dict (see ``scripts/train_sbm.py``) whose
``J`` ``(L, L, q, q)`` and ``h`` ``(L, q)`` arrays are already zero-sum gauged
before saving. :func:`load_model` re-applies the (idempotent) zero-sum gauge
anyway so the invariant holds regardless of how the file was produced — energy
*differences* within a model are gauge-invariant, but the additive constant and
overall scale are not, so combining ``E_A + E_B`` is only meaningful once both
models sit in the same fixed gauge (spec §1, §5 C2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SBM import provenance
from SBM.utils.utils import MSA_ALPHABET, Zero_Sum_Gauge


@dataclass(frozen=True, eq=False)
class PottsModel:
    """A fitted Potts model in the zero-sum gauge.

    ``J[i, j, a, b]`` and ``h[i, a]`` follow the package convention
    ``E(S) = −Σ_i h_i(S_i) − Σ_{i<j} J_ij(S_i, S_j)`` with ``P(S) ∝ exp(−E)``
    (lower energy = more model-typical). ``J`` is symmetric: ``J[i,j,a,b] = J[j,i,b,a]``.
    """

    name: str
    J: np.ndarray  # (L, L, q, q), float64
    h: np.ndarray  # (L, q), float64
    L: int
    q: int
    alphabet: str
    gauge: str
    sha256: str
    source: str


def load_model(path: Path | str, *, name: str | None = None) -> PottsModel:
    """Load ``model.npy`` into a :class:`PottsModel`, re-gauged to zero-sum.

    ``name`` defaults to the run-directory name (``model.npy``'s parent), which
    is what the combine pipeline uses to label each model (``CM``, ``PPIC``).
    """
    path = Path(path)
    raw = np.load(path, allow_pickle=True).item()
    h = np.asarray(raw["h"], dtype=np.float64)
    J = np.asarray(raw["J"], dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"{path}: expected h with shape (L, q), got {h.shape}")
    if J.ndim != 4 or J.shape[0] != J.shape[1] or J.shape[2] != J.shape[3]:
        raise ValueError(f"{path}: expected J with shape (L, L, q, q), got {J.shape}")
    L, q = h.shape
    if J.shape != (L, L, q, q):
        raise ValueError(f"{path}: J shape {J.shape} inconsistent with h shape {h.shape}")
    if q != len(MSA_ALPHABET):
        raise ValueError(
            f"{path}: model has q={q}, expected {len(MSA_ALPHABET)} for alphabet "
            f"{MSA_ALPHABET!r}; refusing to score against a mismatched alphabet"
        )
    J_zg, h_zg = Zero_Sum_Gauge(J, h)
    return PottsModel(
        name=name if name is not None else path.parent.name,
        J=J_zg,
        h=h_zg,
        L=int(L),
        q=int(q),
        alphabet=MSA_ALPHABET,
        gauge="zero_sum",
        sha256=provenance.file_sha256(path),
        source=str(path),
    )


def seed_msa_path(model_path: Path | str) -> Path | None:
    """The encoded seed MSA sitting next to a ``model.npy`` (``inputs/msa.npy``)."""
    candidate = Path(model_path).parent / "inputs" / "msa.npy"
    return candidate if candidate.is_file() else None


def load_seed_msa(model_path: Path | str) -> np.ndarray:
    """Seed MSA used to build the profile-HMM proposal (width ``L``).

    Prefers the run dir's ``inputs/msa.npy`` (small); falls back to the model
    pickle's ``Train`` array. Raises if neither is available.
    """
    path = seed_msa_path(model_path)
    if path is not None:
        return np.load(path)
    raw = np.load(Path(model_path), allow_pickle=True).item()
    train = raw.get("Train")
    if train is None or np.asarray(train).size == 0:
        raise FileNotFoundError(
            f"no seed MSA for {model_path}: neither inputs/msa.npy nor a 'Train' "
            "array in the model pickle"
        )
    return np.asarray(train, dtype=np.int64)
