"""Score a raw sequence under one or two Potts models (spec §3, §4).

Three methods collapse the latent alignment of a raw, ungapped query into a
single energy:

- ``"in_frame"`` — the query is already in the model's frame (length ``L``);
  return the exact Potts energy (spec §2 base case, ``R4b``).
- ``"map"`` — Viterbi-align to the profile HMM, evaluate the full Potts energy on
  that single path. This is the *fields*-MAP, not the true full-energy MAP
  (spec §3.2) — labelled as such in the result, never silently called "MAP".
- ``"marginal"`` (default) — integrate the alignment out by importance sampling
  with the profile-HMM posterior as proposal (spec §3.1). Reports ESS and Monte
  Carlo standard error; warns loudly when ESS is low (the estimate is unreliable).
- ``"dcalign"`` — couplings-aware alignment by DCAlign (spec §10.9). The
  expensive Julia alignment runs out-of-process and is cached on disk
  (:mod:`SBM.utils.dcalign_score`); this branch is a thin cache-reader that
  recomputes the energy in-frame on the cached frame, so it stays pure (no Julia
  here) and gauge-consistent with ``map``/``in_frame``.
- ``"potts_align"`` — couplings-aware gap-placement aligner (iter-003, spec
  ``docs/POTTS_ALIGN.md``). Minimizes the exact in-frame Potts energy over the
  ``C(L, N)`` insert-free frames (exact enumeration for few gaps, else parallel
  tempering) and returns the global-minimum frame. Pure numpy; requires a raw
  gap-free ``seq`` with ``N <= L`` and an explicit ``seed``.

Energies are only comparable / additive across models in a fixed gauge; both
models are loaded in the zero-sum gauge (see :mod:`SBM.energy.model`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .encoding import GAP, ints_to_seq, seq_to_ints
from .hmm import ProfileHMM
from .model import PottsModel
from .potts import potts_energies, potts_energy
from .potts_align import potts_align

log = logging.getLogger(__name__)

METHODS = ("in_frame", "map", "marginal", "dcalign", "potts_align")
DEFAULT_N_SAMPLES = 1000
DEFAULT_ESS_THRESHOLD = 100.0


@dataclass(frozen=True)
class ScoreResult:
    """Energy of one sequence under one model, plus method diagnostics."""

    energy: float
    method: str
    model_name: str
    gauge: str
    model_sha256: str
    n_samples: int | None = None
    ess: float | None = None
    # MC standard error of the IS estimate — the *variance* of the log-mean-exp
    # only. It does NOT include the one-sided O(1/S) Jensen bias of −log Ẑ; at
    # low ESS that bias can exceed mc_stderr, hence the low-ESS warning.
    mc_stderr: float | None = None
    seed: int | None = None
    # For marginal: the highest-importance-weight sampled frame (the alignment
    # that dominates the estimate), NOT necessarily the minimum-energy frame.
    representative_alignment: str | None = None
    notes: str = ""


def _marginal_energy(
    x: np.ndarray,
    model: PottsModel,
    hmm: ProfileHMM,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """Importance-sampling free-energy estimate ``Ẽ^F`` (spec §3.1).

    Returns ``(energy, ess, mc_stderr, representative_frame)``.

    ``Ẽ^F = F_prop − log E_{a~q}[exp(−ΔE)]`` with ``ΔE = E_k − E_prop``,
    ``F_prop = −log Z_prop``. Frames are scored in one batched Potts call.
    """
    paths = hmm.sample_paths(x, n_samples, rng)
    frames = np.stack([hmm.path_to_frame(p, x) for p in paths])  # (S, L)
    e_potts = potts_energies(frames, model)  # E_k(x, a^(s))
    e_prop = -np.array([hmm.path_logscore(p, x) for p in paths])  # E_prop = −log q_joint
    delta = e_potts - e_prop  # ΔE
    log_weights = -delta  # log w_s

    log_z_prop = hmm.forward_logZ(x)
    log_mean_w = logsumexp(log_weights) - np.log(n_samples)
    energy = -log_z_prop - log_mean_w  # = F_prop − log mean(w)

    # ESS and MC stderr from the (shifted, stable) weights. With cv² the squared
    # coefficient of variation of w: ESS = S/(1+cv²); stderr(log mean w) ≈ √(cv²/S).
    shift = float(log_weights.max())
    w = np.exp(log_weights - shift)
    mean_w, mean_w2 = float(w.mean()), float((w**2).mean())
    cv2 = max(mean_w2 / mean_w**2 - 1.0, 0.0)
    ess = n_samples / (1.0 + cv2)
    mc_stderr = float(np.sqrt(cv2 / n_samples))

    representative = frames[int(np.argmax(log_weights))]
    return float(energy), float(ess), mc_stderr, representative


def score_sequence(
    seq: np.ndarray,
    model: PottsModel,
    *,
    method: str = "marginal",
    hmm: ProfileHMM | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int | None = None,
    ess_threshold: float = DEFAULT_ESS_THRESHOLD,
    dcalign_frame: str | None = None,
    dcalign_notes: str = "",
) -> ScoreResult:
    """Energy of one query (integer array) under ``model`` via ``method``.

    For ``"in_frame"`` ``seq`` must be length ``L`` (gaps allowed). For ``"map"``
    / ``"marginal"`` ``seq`` is a raw ungapped query (residues 1..20) and ``hmm``
    (built once via :meth:`ProfileHMM.from_model`) is required. ``marginal``
    requires an explicit ``seed`` for reproducibility. For ``"dcalign"`` ``seq``
    is ignored and ``dcalign_frame`` (a length-``L`` amino-acid frame string, gap
    ``-``, from the on-disk DCAlign cache) is required; the energy is recomputed
    in-frame on that frame, so this branch is pure (no Julia call here). For
    ``"potts_align"`` ``seq`` is a raw gap-free query with ``N <= L`` and an
    explicit ``seed`` is required; the energy is the global insert-free Potts
    minimum found by :func:`SBM.energy.potts_align.potts_align`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    seq = np.asarray(seq, dtype=np.int64)

    if method == "in_frame":
        energy = potts_energy(seq, model)
        return ScoreResult(
            energy=energy, method="in_frame", model_name=model.name,
            gauge=model.gauge, model_sha256=model.sha256,
            representative_alignment=ints_to_seq(seq),
        )

    if method == "dcalign":
        # The expensive couplings-aware alignment ran out-of-process; we only
        # recompute the energy on the cached frame. A missing/empty frame means
        # DCAlign failed for this id — a loud error, never a silent skip.
        if not dcalign_frame:
            raise ValueError(
                f"method='dcalign' needs a non-empty dcalign_frame for model {model.name!r}; "
                "an empty frame means DCAlign failed to align this sequence (see the cache)."
            )
        frame = seq_to_ints(dcalign_frame)
        if frame.size != model.L:
            raise ValueError(
                f"dcalign_frame length {frame.size} != model L={model.L} for {model.name!r}"
            )
        energy = potts_energy(frame, model)
        notes = "couplings-aware (DCAlign); energy recomputed in-frame via potts_energy"
        if dcalign_notes:
            notes += f"; {dcalign_notes}"
        return ScoreResult(
            energy=energy, method="dcalign", model_name=model.name,
            gauge=model.gauge, model_sha256=model.sha256,
            representative_alignment=ints_to_seq(frame), notes=notes,
        )

    if method == "potts_align":
        # Couplings-aware gap-placement minimization (iter-003, docs/POTTS_ALIGN.md):
        # return the global insert-free Potts-energy minimum frame. Pure numpy; the
        # dispatcher enumerates exactly for few gaps and falls back to g-adaptive
        # parallel tempering otherwise. seq is a raw gap-free query with N <= L.
        # (The checks below duplicate potts_align's own _validate_query for a
        # clearer score-layer error; potts_align re-enforces as defense in depth.)
        if seed is None:
            raise ValueError("method='potts_align' requires an explicit seed (reproducibility)")
        if seq.size and np.any(seq == GAP):
            raise ValueError(
                "method='potts_align' needs a gap-free raw query (residues 1..20); the input "
                "contains gaps. Strip them first with SBM.energy.encoding.strip_gaps."
            )
        if seq.size > model.L:
            raise ValueError(
                f"method='potts_align' needs N<=L (insert-free); N={seq.size} > L={model.L} "
                f"for model {model.name!r}. N>L pairs need insertions and are out of scope."
            )
        res = potts_align(seq, model, seed=seed)
        return ScoreResult(
            energy=res.best_energy, method="potts_align", model_name=model.name,
            gauge=model.gauge, model_sha256=model.sha256, seed=seed,
            representative_alignment=ints_to_seq(res.best_frame),
            notes=f"global Potts-min; engine={res.method}; exact={res.is_global_exact}",
        )

    if hmm is None:
        raise ValueError(f"method={method!r} needs a ProfileHMM (build via ProfileHMM.from_model)")
    if seq.size and np.any(seq == GAP):
        raise ValueError(
            f"method={method!r} needs a gap-free raw query (residues 1..20); the input "
            "contains gaps. Strip them first with SBM.energy.encoding.strip_gaps."
        )

    if method == "map":
        path = hmm.viterbi(seq)
        frame = hmm.path_to_frame(path, seq)
        energy = potts_energy(frame, model)
        return ScoreResult(
            energy=energy, method="map", model_name=model.name,
            gauge=model.gauge, model_sha256=model.sha256,
            representative_alignment=ints_to_seq(frame),
            notes="fields-MAP (Viterbi under HMM emissions from h); not the full-energy MAP",
        )

    # marginal
    if seed is None:
        raise ValueError("method='marginal' requires an explicit seed (reproducibility)")
    if n_samples < 1:
        raise ValueError(f"method='marginal' needs n_samples >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    energy, ess, mc_stderr, frame = _marginal_energy(
        seq, model, hmm, n_samples=n_samples, rng=rng
    )
    notes = "marginal IS free-energy; −log of an unbiased Ẑ is biased high O(1/S)"
    if ess < ess_threshold:
        log.warning(
            "low ESS=%.1f (< %.1f) for model %r: marginal energy unreliable; "
            "raise n_samples or upgrade the proposal (DCAlign / annealed IS)",
            ess, ess_threshold, model.name,
        )
        notes += f"; LOW ESS={ess:.1f} < {ess_threshold:.0f} — estimate unreliable"
    return ScoreResult(
        energy=energy, method="marginal", model_name=model.name,
        gauge=model.gauge, model_sha256=model.sha256,
        n_samples=n_samples, ess=ess, mc_stderr=mc_stderr, seed=seed,
        representative_alignment=ints_to_seq(frame), notes=notes,
    )


def score_two_models(
    seq: np.ndarray,
    model_A: PottsModel,
    model_B: PottsModel,
    *,
    w_A: float = 1.0,
    w_B: float = 1.0,
    method: str = "marginal",
    hmm_A: ProfileHMM | None = None,
    hmm_B: ProfileHMM | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int | None = None,
    ess_threshold: float = DEFAULT_ESS_THRESHOLD,
) -> dict:
    """Score one query under both models; return ``E_A``, ``E_B``, ``E_tot``.

    ``E_tot = w_A·E_A + w_B·E_B`` (weights explicit, default 1.0; spec §5 C5).
    The two models keep their native lengths — the query is aligned to each
    independently (spec §5 C1); nothing is trimmed or padded to a common length.
    """
    res_A = score_sequence(
        seq, model_A, method=method, hmm=hmm_A,
        n_samples=n_samples, seed=seed, ess_threshold=ess_threshold,
    )
    # Offset B's seed so the two models' IS draws are independent yet reproducible.
    seed_B = None if seed is None else seed + 1
    res_B = score_sequence(
        seq, model_B, method=method, hmm=hmm_B,
        n_samples=n_samples, seed=seed_B, ess_threshold=ess_threshold,
    )
    e_tot = w_A * res_A.energy + w_B * res_B.energy
    return {
        "E_A": res_A.energy,
        "E_B": res_B.energy,
        "E_tot": e_tot,
        "result_A": res_A,
        "result_B": res_B,
        "weights": {"w_A": w_A, "w_B": w_B},
    }
