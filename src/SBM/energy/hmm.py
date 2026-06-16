"""Profile-HMM alignment proposal for the marginal-energy estimator (spec §3.1).

A raw, ungapped query ``x`` can be threaded into a model's ``L``-column frame
many ways. The marginal energy integrates that latent alignment out; the exact
sum is intractable with couplings, so we estimate it by importance sampling
with a tractable proposal ``q(a|x)`` whose alignment posterior is *exact* and
*samplable* — a profile HMM (Durbin, Eddy, Krogh & Mitchison 1998,
*Biological Sequence Analysis*, CUP).

This is a self-contained, log-space Plan7-style profile HMM rather than a wrapper
around HMMER/pyhmmer, for three reasons: (1) the match-state architecture is
fixed by the model (one match state per column), (2) emissions must be injected
from the Potts fields ``h`` so ``ΔE`` is dominated by the couplings (variance
control), and (3) we need exact forward-filtering / backward-sampling of the
alignment posterior, which the search-oriented APIs do not expose. Keeping it in
numpy makes it deterministic, seed-controlled, and checkable against brute-force
enumeration (see ``tests/test_energy.py``).

**Alignment space.** The set of "alignments" of ``x`` to the model is *defined*
as the set of paths through this profile HMM. Each column ``k`` is visited
exactly once, as a match ``M_k`` (consumes one query residue → that residue sits
in frame position ``k``) or a delete ``D_k`` (→ gap, state 0). Query residues
placed in insert states ``I_k`` are *not* in the frame; they contribute only to
the proposal score ``E_prop`` (the §3.3 gap/insertion policy), so the Potts
energy ``E_k`` is a function of the match/delete assignment alone.

**Emissions / transitions.** Match emissions are ``e_M(k,a) ∝ exp(h_k(a))`` over
the 20 amino acids (the fields-only single-site marginal; the Potts gap state is
represented by the delete path, never a match emission). Insert emissions are
the seed-MSA background composition. Per-column delete propensities are estimated
from the seed MSA's gap frequencies; insert open/extend are fixed affine defaults
(``tau_mi``/``tau_ii``) because the fixed-width training alignment carries no
insert evidence. Proposal quality only affects the importance-sampling variance
(ESS), never the unbiasedness of the partition-function estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import GAP
from .model import PottsModel

NEG_INF = -np.inf


@dataclass(frozen=True)
class AlignmentPath:
    """One alignment of a query to the model frame: an ordered list of core
    states ``(state, k)`` with ``state in {'M','I','D'}`` and column ``k`` (the
    Begin/End states are implicit). Emitting states (M, I) consume query residues
    left-to-right; deletes (D) consume none.
    """

    states: tuple[tuple[str, int], ...]


def _logaddexp(values: list[float]) -> float:
    """Numerically stable log-sum-exp of a short list (handles all -inf)."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.logaddexp.reduce(arr)) if arr.size else NEG_INF


def _column_gap_frequencies(seed_msa: np.ndarray, L: int, *, eps: float) -> np.ndarray:
    """Per-column delete propensity from the seed MSA, clamped to (eps, 1-eps).

    Clamping keeps every alignment in the proposal's support (no log(0)), which
    the importance-sampling estimator requires.
    """
    pdel = np.full(L + 1, eps)  # index 1..L used; pdel[0] unused
    if seed_msa is not None and seed_msa.size:
        if seed_msa.shape[1] != L:
            raise ValueError(
                f"seed MSA width {seed_msa.shape[1]} != model length {L}"
            )
        freq = np.mean(seed_msa == GAP, axis=0)
        pdel[1:] = np.clip(freq, eps, 1.0 - eps)
    return pdel


def _background_log_emission(seed_msa: np.ndarray, q: int, *, eps: float) -> np.ndarray:
    """Insert-state log-emission over residues 1..q-1 from seed-MSA composition."""
    counts = np.full(q, eps)
    if seed_msa is not None and seed_msa.size:
        obs = np.bincount(seed_msa.ravel(), minlength=q).astype(np.float64)
        counts[1:] += obs[1:]  # ignore the gap state for the residue background
    counts[GAP] = 0.0
    probs = counts / counts.sum()
    with np.errstate(divide="ignore"):
        log = np.log(probs)
    return log  # log[GAP] = -inf


def _match_log_emission(h: np.ndarray) -> np.ndarray:
    """Match log-emission ``log softmax(h[k, 1:])`` over residues; gap → -inf.

    Returns shape ``(L+1, q)`` with rows 1..L populated (row 0 = Begin, unused).
    """
    L, q = h.shape
    log_e = np.full((L + 1, q), NEG_INF)
    fields = h[:, 1:]  # the 20 amino acids; the gap field is not an emission
    log_norm = np.logaddexp.reduce(fields, axis=1)  # (L,)
    log_e[1:, 1:] = fields - log_norm[:, None]
    return log_e


@dataclass(frozen=True)
class ProfileHMM:
    """A profile HMM proposal built from a Potts model's fields + seed MSA."""

    L: int
    q: int
    log_e_match: np.ndarray  # (L+1, q)
    log_e_insert: np.ndarray  # (q,)
    # transition log-probs, indexed by source node k = 0..L (node 0 = Begin/I0)
    log_mm: np.ndarray
    log_mi: np.ndarray
    log_md: np.ndarray
    log_me: np.ndarray
    log_im: np.ndarray
    log_ii: np.ndarray
    log_ie: np.ndarray
    log_dm: np.ndarray
    log_dd: np.ndarray
    log_de: np.ndarray

    @classmethod
    def from_model(
        cls,
        model: PottsModel,
        seed_msa: np.ndarray,
        *,
        tau_mi: float = 0.05,
        tau_ii: float = 0.5,
        eps: float = 1e-3,
    ) -> "ProfileHMM":
        """Build the proposal HMM from ``model.h`` and a seed MSA (width ``L``)."""
        L, q = model.L, model.q
        pdel = _column_gap_frequencies(seed_msa, L, eps=eps)
        log_e_match = _match_log_emission(model.h)
        log_e_insert = _background_log_emission(seed_msa, q, eps=eps)

        def _z() -> np.ndarray:
            return np.full(L + 1, NEG_INF)

        mm, mi, md, me = _z(), _z(), _z(), _z()
        im, ii, ie = _z(), _z(), _z()
        dm, dd, de = _z(), _z(), _z()
        log = np.log
        for k in range(L):  # transitions into column k+1
            pd = pdel[k + 1]
            rem = 1.0 - tau_mi
            mi[k] = log(tau_mi)
            md[k] = log(rem * pd)
            mm[k] = log(rem * (1.0 - pd))
            ii[k] = log(tau_ii)
            im[k] = log(1.0 - tau_ii)
            if k >= 1:  # D_0 does not exist
                dd[k] = log(pd)
                dm[k] = log(1.0 - pd)
        # Last column: only inserts and exits to End.
        mi[L] = log(tau_mi)
        me[L] = log(1.0 - tau_mi)
        ii[L] = log(tau_ii)
        ie[L] = log(1.0 - tau_ii)
        de[L] = 0.0  # D_L -> End w.p. 1
        return cls(
            L=L, q=q, log_e_match=log_e_match, log_e_insert=log_e_insert,
            log_mm=mm, log_mi=mi, log_md=md, log_me=me,
            log_im=im, log_ii=ii, log_ie=ie,
            log_dm=dm, log_dd=dd, log_de=de,
        )

    # ── core dynamic program ────────────────────────────────────────────
    def _forward_tables(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-space forward tables ``(fM, fI, fD)``, each shape ``(N+1, L+1)``."""
        N, L = len(x), self.L
        fM = np.full((N + 1, L + 1), NEG_INF)
        fI = np.full((N + 1, L + 1), NEG_INF)
        fD = np.full((N + 1, L + 1), NEG_INF)
        fM[0, 0] = 0.0  # Begin
        for i in range(N + 1):
            r = int(x[i - 1]) if i >= 1 else -1
            for k in range(L + 1):
                if i >= 1 and k >= 1:
                    fM[i, k] = self.log_e_match[k, r] + _logaddexp([
                        fM[i - 1, k - 1] + self.log_mm[k - 1],
                        fI[i - 1, k - 1] + self.log_im[k - 1],
                        fD[i - 1, k - 1] + self.log_dm[k - 1],
                    ])
                if i >= 1:
                    fI[i, k] = self.log_e_insert[r] + _logaddexp([
                        fM[i - 1, k] + self.log_mi[k],
                        fI[i - 1, k] + self.log_ii[k],
                    ])
            for k in range(1, L + 1):  # deletes: same row, ascending column
                fD[i, k] = _logaddexp([
                    fM[i, k - 1] + self.log_md[k - 1],
                    fD[i, k - 1] + self.log_dd[k - 1],
                ])
        return fM, fI, fD

    def forward_logZ(self, x: np.ndarray) -> float:
        """Exact log partition ``log Z_prop = log P_HMM(x)`` (forward algorithm).

        ``F_prop = −forward_logZ`` is the proposal's alignment free energy.
        """
        x = np.asarray(x, dtype=np.int64)
        fM, fI, fD = self._forward_tables(x)
        N, L = len(x), self.L
        return _logaddexp([
            fM[N, L] + self.log_me[L],
            fI[N, L] + self.log_ie[L],
            fD[N, L] + self.log_de[L],
        ])

    def viterbi(self, x: np.ndarray) -> AlignmentPath:
        """MAP alignment path under the proposal (max-product + backtrace).

        Note (spec §3.2): evaluating the full Potts energy on this path gives the
        *fields*-MAP energy, not the true full-energy MAP — the proposal ignores
        couplings during alignment. Callers must label it as such.
        """
        x = np.asarray(x, dtype=np.int64)
        N, L = len(x), self.L
        vM = np.full((N + 1, L + 1), NEG_INF)
        vI = np.full((N + 1, L + 1), NEG_INF)
        vD = np.full((N + 1, L + 1), NEG_INF)
        bp: dict[tuple[str, int, int], tuple[str, int]] = {}
        vM[0, 0] = 0.0

        def _best(cands: list[tuple[float, str, int]]) -> tuple[float, str | None, int]:
            best = max(cands, key=lambda c: c[0]) if cands else (NEG_INF, None, -1)
            return best

        for i in range(N + 1):
            r = int(x[i - 1]) if i >= 1 else -1
            for k in range(L + 1):
                if i >= 1 and k >= 1:
                    val, s, kk = _best([
                        (vM[i - 1, k - 1] + self.log_mm[k - 1], "M", k - 1),
                        (vI[i - 1, k - 1] + self.log_im[k - 1], "I", k - 1),
                        (vD[i - 1, k - 1] + self.log_dm[k - 1], "D", k - 1),
                    ])
                    vM[i, k] = self.log_e_match[k, r] + val
                    if s is not None:
                        bp[("M", k, i)] = (s, kk)
                if i >= 1:
                    val, s, kk = _best([
                        (vM[i - 1, k] + self.log_mi[k], "M", k),
                        (vI[i - 1, k] + self.log_ii[k], "I", k),
                    ])
                    vI[i, k] = self.log_e_insert[r] + val
                    if s is not None:
                        bp[("I", k, i)] = (s, kk)
            for k in range(1, L + 1):
                val, s, kk = _best([
                    (vM[i, k - 1] + self.log_md[k - 1], "M", k - 1),
                    (vD[i, k - 1] + self.log_dd[k - 1], "D", k - 1),
                ])
                vD[i, k] = val
                if s is not None:
                    bp[("D", k, i)] = (s, kk)

        term, s_end, _ = _best([
            (vM[N, L] + self.log_me[L], "M", L),
            (vI[N, L] + self.log_ie[L], "I", L),
            (vD[N, L] + self.log_de[L], "D", L),
        ])
        if not np.isfinite(term):
            raise ValueError("query has no finite-probability alignment under the proposal")
        return AlignmentPath(self._backtrace(bp, s_end, x))

    def _backtrace(self, bp, s_end, x) -> tuple[tuple[str, int], ...]:
        N, L = len(x), self.L
        s, k, i = s_end, L, N
        rev: list[tuple[str, int]] = []
        while not (s == "M" and k == 0):
            rev.append((s, k))
            ps, pk = bp[(s, k, i)]
            if s in ("M", "I"):
                i -= 1
            s, k = ps, pk
        return tuple(reversed(rev))

    def sample_paths(self, x: np.ndarray, n: int, rng: np.random.Generator) -> list[AlignmentPath]:
        """Draw ``n`` exact samples from the alignment posterior ``q(a|x)`` (FFBS).

        Forward-filter once, then backward-sample each path proportionally to the
        posterior. Seeded via ``rng`` for reproducibility.
        """
        x = np.asarray(x, dtype=np.int64)
        fM, fI, fD = self._forward_tables(x)
        return [self._sample_one(x, fM, fI, fD, rng) for _ in range(n)]

    def _sample_one(self, x, fM, fI, fD, rng) -> AlignmentPath:
        N, L = len(x), self.L

        def _choose(cands: list[tuple[float, str, int]]) -> tuple[str, int]:
            weights = np.array([c[0] for c in cands], dtype=np.float64)
            finite = np.isfinite(weights)
            if not finite.any():
                raise ValueError("backward sampling hit a zero-probability state")
            weights = weights - weights[finite].max()
            probs = np.where(finite, np.exp(weights), 0.0)
            probs /= probs.sum()
            idx = rng.choice(len(cands), p=probs)
            return cands[idx][1], cands[idx][2]

        s, k = _choose([
            (fM[N, L] + self.log_me[L], "M", L),
            (fI[N, L] + self.log_ie[L], "I", L),
            (fD[N, L] + self.log_de[L], "D", L),
        ])
        i = N
        rev: list[tuple[str, int]] = [(s, k)]
        while not (s == "M" and k == 0):
            if s == "M":
                cands = [
                    (fM[i - 1, k - 1] + self.log_mm[k - 1], "M", k - 1),
                    (fI[i - 1, k - 1] + self.log_im[k - 1], "I", k - 1),
                    (fD[i - 1, k - 1] + self.log_dm[k - 1], "D", k - 1),
                ]
                ps, pk = _choose(cands)
                i -= 1
            elif s == "I":
                cands = [
                    (fM[i - 1, k] + self.log_mi[k], "M", k),
                    (fI[i - 1, k] + self.log_ii[k], "I", k),
                ]
                ps, pk = _choose(cands)
                i -= 1
            else:  # delete: no emission, step left a column
                cands = [
                    (fM[i, k - 1] + self.log_md[k - 1], "M", k - 1),
                    (fD[i, k - 1] + self.log_dd[k - 1], "D", k - 1),
                ]
                ps, pk = _choose(cands)
            s, k = ps, pk
            if not (s == "M" and k == 0):
                rev.append((s, k))
        return AlignmentPath(tuple(reversed(rev)))

    # ── path → frame / score ────────────────────────────────────────────
    def path_to_frame(self, path: AlignmentPath, x: np.ndarray) -> np.ndarray:
        """Map an alignment path to an in-frame, length-``L`` gapped int sequence.

        Match states place the consumed residue in their column; deletes leave a
        gap; insert states' residues do not appear in the frame.
        """
        x = np.asarray(x, dtype=np.int64)
        S = np.zeros(self.L, dtype=np.int64)  # all gap by default
        p = 0
        for state, k in path.states:
            if state == "M":
                S[k - 1] = x[p]
                p += 1
            elif state == "I":
                p += 1
            # delete: leave S[k-1] = GAP
        if p != len(x):
            raise ValueError(f"path consumes {p} residues but query has {len(x)}")
        return S

    def path_logscore(self, path: AlignmentPath, x: np.ndarray) -> float:
        """``log q_joint(x, a)`` — the proposal's joint log-prob of query+path.

        ``E_prop(x, a) = −path_logscore`` (spec §3.1). Sums transition log-probs
        (Begin → … → End) and emission log-probs of the residues each M/I emits.
        """
        x = np.asarray(x, dtype=np.int64)
        states = path.states
        total = 0.0
        p = 0
        prev = ("M", 0)  # Begin
        for state, k in states:
            total += self._trans_logprob(prev, (state, k))
            if state == "M":
                total += self.log_e_match[k, int(x[p])]
                p += 1
            elif state == "I":
                total += self.log_e_insert[int(x[p])]
                p += 1
            prev = (state, k)
        total += self._end_logprob(prev)
        return total

    def _trans_logprob(self, src: tuple[str, int], dst: tuple[str, int]) -> float:
        s1, k1 = src
        s2, k2 = dst
        if s1 == "M" and s2 == "M" and k2 == k1 + 1:
            return self.log_mm[k1]
        if s1 == "M" and s2 == "I" and k2 == k1:
            return self.log_mi[k1]
        if s1 == "M" and s2 == "D" and k2 == k1 + 1:
            return self.log_md[k1]
        if s1 == "I" and s2 == "M" and k2 == k1 + 1:
            return self.log_im[k1]
        if s1 == "I" and s2 == "I" and k2 == k1:
            return self.log_ii[k1]
        if s1 == "D" and s2 == "M" and k2 == k1 + 1:
            return self.log_dm[k1]
        if s1 == "D" and s2 == "D" and k2 == k1 + 1:
            return self.log_dd[k1]
        return NEG_INF

    def _end_logprob(self, src: tuple[str, int]) -> float:
        s, k = src
        if k != self.L:
            return NEG_INF
        return {"M": self.log_me, "I": self.log_ie, "D": self.log_de}[s][k]
