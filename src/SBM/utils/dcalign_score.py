"""Couplings-aware alignment via DCAlign (combine spec §10.9).

DCAlign (Muntoni, Pagnani, Weigt & Zamponi 2020, arXiv:2005.08500) is a
couplings-aware Julia aligner — the planned ``map`` upgrade that the
fields-only Viterbi aligner cannot match on ambiguous frames. Like
ProteinMPNN (:mod:`SBM.utils.mpnn_score`), it is an external tool we never
import in-process: alignment is delegated to a subprocess that runs
``src/SBM/julia/run_dcalign.jl`` against a DCAlign clone, with our pre-fit Potts
model handed over as raw little-endian ``Float64`` binaries (the validated
transform below). The subprocess writes a TSV cache of
``(seq_id → aligned length-L frame + DCAlign's own energy + diagnostics)``.

This module is a *bridge*, not a scorer. The ``dcalign`` scoring branch
(:func:`SBM.energy.score.score_sequence`) is a thin cache-reader that recomputes
the energy in-frame via our :func:`SBM.energy.potts.potts_energy` on the cached
frame — gauge-consistent with ``map``/``in_frame`` (the spike showed agreement
``≤ 5e-7``, kept as a standing canary). So ``score_sequence`` never shells out to
Julia; only the (expensive, cluster-side) cache build does.

The model handoff (spec §10.9, validated by the spike). Our alphabet
``-ACDEFGHIKLMNPQRSTVWY`` (gap 0, residues 1..20) maps to DCAlign's
``A..Y = 1..20, gap = 21`` — only the gap moves. With
``ORDER = list(range(1, 21)) + [0]``::

    J_dca = J.transpose(2, 3, 0, 1)[ORDER][:, ORDER]   # (L,L,q,q) -> (q,q,L,L)
    h_dca = h.T[ORDER]                                  # (L,q)     -> (q,L)
    path.write_bytes(J_dca.astype("<f8").tobytes(order="F"))  # Julia reads (q,q,L,L)

Both arrays go out in the zero-sum gauge (models are loaded that way), so
DCAlign's ``compute_potts_en`` and our ``potts_energy`` agree up to fp noise.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid importing the model module (and the MCMC kernel) at load
    from SBM.energy.model import PottsModel

log = logging.getLogger(__name__)

#: Project alphabet, gap at index 0. Mirrors ``SBM.utils.utils.MSA_ALPHABET``;
#: kept local so the bridge (and the §10.9 transform test) stay decoupled from
#: the compiled MCMC kernel that ``SBM.utils.utils`` pulls in.
ALPHABET: str = "-ACDEFGHIKLMNPQRSTVWY"

#: Permutation taking our (gap=0, residues 1..20) order to DCAlign's
#: (residues 1..20, gap=21) order. ``J.transpose(2,3,0,1)[ORDER][:, ORDER]``.
ORDER: list[int] = list(range(1, 21)) + [0]

#: TSV columns written by the Julia driver and read back here. Stable contract.
TSV_COLUMNS: tuple[str, ...] = (
    "seq_id",
    "aligned_frame",
    "dcalign_energy",
    "converged",
    "used_decimation",
    "n_iter",
)
TSV_HEADER: str = "\t".join(TSV_COLUMNS)


@dataclasses.dataclass(frozen=True)
class DCAlignContext:
    """Pointer + provenance for a DCAlign clone and a Julia interpreter.

    Carries everything :func:`align_sequences` needs to launch the driver and
    everything we want recorded in the run manifest (clone path + git commit,
    Julia version, the algorithm params). Loads no Julia code in-process.
    """

    dcalign_path: Path
    julia: Path
    driver_jl: Path
    julia_project: Path
    dcalign_git_commit: str | None
    julia_version: str | None
    maxiter: int = 2000
    seed: int = 0
    pcount: float = 1e-3
    threads: int = 1


@dataclasses.dataclass(frozen=True)
class DCAlignResult:
    """One sequence's DCAlign alignment + DCAlign's own reported energy.

    ``aligned_frame`` is the length-``L`` frame as an amino-acid string (gap
    ``-``); an **empty** string marks a sequence DCAlign failed on (recorded,
    not dropped — it becomes a loud error at score time only if that id is
    needed). ``dcalign_energy`` is DCAlign's ``compute_potts_en`` value (NaN on
    failure); we recompute the authoritative energy in-frame at score time and
    compare the two as a gauge/handoff canary.
    """

    seq_id: str
    aligned_frame: str
    dcalign_energy: float
    converged: bool
    used_decimation: bool
    n_iter: int

    @property
    def ok(self) -> bool:
        return bool(self.aligned_frame)


# ── encoding helpers (local; mirror SBM.energy.encoding without the kernel) ──


def _ints_to_str(arr: np.ndarray) -> str:
    return "".join(ALPHABET[int(i)] for i in np.asarray(arr).ravel())


# ── path / version / hashing resolution (mirror mpnn_score) ──────────────────


def _git_commit_at(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def _resolve_dcalign_path(explicit: Path | None) -> Path:
    """Return a validated DCAlign clone path.

    Priority: ``explicit`` (from CLI/config) > ``DCALIGN_PATH`` env > raise.
    Validates by checking for ``Project.toml`` and a ``src/`` directory.
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("DCALIGN_PATH")
        if env is None:
            raise RuntimeError(
                "DCAlign path not supplied. Pass scoring.dcalign_path / --dcalign-path "
                "or set DCALIGN_PATH=/path/to/DCAlign (a clone of "
                "https://github.com/infernet-h2020/DCAlign)."
            )
        p = Path(env).expanduser().resolve()
    if not (p / "Project.toml").is_file():
        raise FileNotFoundError(
            f"{p} does not look like a DCAlign clone (no Project.toml). "
            "Expected a checkout of https://github.com/infernet-h2020/DCAlign."
        )
    if not (p / "src").is_dir():
        raise FileNotFoundError(f"{p}/src is missing — DCAlign clone may be incomplete.")
    return p


def _resolve_julia(explicit: Path | None) -> Path:
    """Return the Julia interpreter to launch DCAlign with.

    Priority: ``explicit`` > ``JULIA_BINARY`` env > ``shutil.which('julia')`` >
    raise. (On Midway, ``module load julia/1.10.2`` puts ``julia`` on PATH.)
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("JULIA_BINARY")
        if env:
            candidate = Path(env).expanduser().resolve()
        else:
            found = shutil.which("julia")
            if found is None:
                raise RuntimeError(
                    "julia not found. Pass scoring.julia / --julia, set JULIA_BINARY, "
                    "or put julia on PATH (Midway: `module load julia/1.10.2`)."
                )
            candidate = Path(found).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"julia interpreter not found at {candidate}.")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"{candidate} is not executable; cannot use as the julia interpreter.")
    return candidate


def _julia_version(julia: Path) -> str | None:
    try:
        out = subprocess.run(
            [str(julia), "--version"], capture_output=True, text=True, check=True, timeout=30
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def dcalign_context(
    dcalign_path: Path | None = None,
    *,
    julia: Path | None = None,
    maxiter: int = 2000,
    seed: int = 0,
    pcount: float = 1e-3,
    threads: int = 1,
) -> DCAlignContext:
    """Validate a DCAlign install + Julia interpreter and return a context.

    The driver script (``src/SBM/julia/run_dcalign.jl``) ships in *this* repo and
    is run with ``--project=<DCAlign clone>`` so it resolves DCAlign from the
    clone's environment. We never import Julia here.
    """
    repo = _resolve_dcalign_path(dcalign_path)
    jl = _resolve_julia(julia)
    driver = Path(__file__).resolve().parent.parent / "julia" / "run_dcalign.jl"
    if not driver.is_file():
        raise FileNotFoundError(f"DCAlign driver script missing at {driver}")
    if maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {maxiter}")
    if pcount <= 0:
        raise ValueError(f"pcount must be > 0, got {pcount}")
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    return DCAlignContext(
        dcalign_path=repo,
        julia=jl,
        driver_jl=driver,
        julia_project=repo,
        dcalign_git_commit=_git_commit_at(repo),
        julia_version=_julia_version(jl),
        maxiter=maxiter,
        seed=seed,
        pcount=pcount,
        threads=threads,
    )


# ── model handoff ────────────────────────────────────────────────────────────


def model_to_dcalign_arrays(model: "PottsModel") -> tuple[np.ndarray, np.ndarray]:
    """Transform our ``(J, h)`` into DCAlign's ``(q,q,L,L)`` / ``(q,L)`` layout.

    Pure index permutation (spec §10.9): our gap-at-0 alphabet becomes DCAlign's
    gap-at-21 alphabet, and ``J`` is transposed to coupling-major. Round-trips
    exactly (see ``tests/test_energy.py``).
    """
    J = np.asarray(model.J, dtype=np.float64)  # (L, L, q, q)
    h = np.asarray(model.h, dtype=np.float64)  # (L, q)
    J_dca = J.transpose(2, 3, 0, 1)[ORDER][:, ORDER]  # (q, q, L, L)
    h_dca = h.T[ORDER]  # (q, L)
    return J_dca, h_dca


def _write_queries(seqs: Sequence[np.ndarray], seq_ids: Sequence[str], path: Path) -> None:
    lines: list[str] = []
    for sid, s in zip(seq_ids, seqs):
        arr = np.asarray(s, dtype=np.int64)
        if arr.size and (arr.min() <= 0 or arr.max() > 20):
            raise ValueError(
                f"query {sid!r} must be a raw gap-free sequence (residues 1..20); "
                f"got values in [{int(arr.min())}, {int(arr.max())}]. Strip gaps first."
            )
        lines.append(f">{sid}")
        lines.append(_ints_to_str(arr))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_seed_ins(msa: np.ndarray, path: Path) -> None:
    """Write the model-frame seed MSA as a DCAlign a2m seed (``seed.ins``).

    Each row of the width-``L`` integer MSA becomes one FASTA record with
    residues uppercased and gaps as ``-`` (via :func:`_ints_to_str`); this is the
    input ``DCAlign.deltan_prior`` reads to build the ``lambda_spec="deltan"``
    prior (combine spec §10.13). The fixed-width MSA carries no insert columns,
    so there are no lowercase residues and the prior captures the empirical
    per-(i,j) gap/deletion geometry rather than literal insertions. Record ids
    must be unique — DCAlign's ``readfull`` dedupes by the first header token, so
    a shared id would silently collapse the seed to one sequence.
    """
    msa = np.asarray(msa, dtype=np.int64)
    if msa.ndim != 2:
        raise ValueError(f"seed MSA must be 2-D (N, L); got shape {msa.shape}")
    if msa.size and (msa.min() < 0 or msa.max() > 20):
        # _ints_to_str indexes ALPHABET (len 21); a stray -1 (load_fasta's
        # non-canonical sentinel) would silently map to 'Y'. Fail loud instead.
        raise ValueError(
            f"seed MSA values must be in 0..20 (gap=0, residues 1..20); "
            f"got [{int(msa.min())}, {int(msa.max())}]"
        )
    lines: list[str] = []
    for i, row in enumerate(msa):
        lines.append(f">seed{i}")
        lines.append(_ints_to_str(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── alignment via the Julia subprocess ───────────────────────────────────────


def align_sequences(
    ctx: DCAlignContext,
    model: "PottsModel",
    seqs: Sequence[np.ndarray],
    seq_ids: Sequence[str],
    *,
    out_dir: Path,
    out_tsv: Path | None = None,
    lambda_spec: str = "flat",
) -> list[DCAlignResult]:
    """Align ``seqs`` to ``model`` with DCAlign, returning one result per id.

    Writes the model binaries + ``meta.json`` + ``queries.fasta`` into
    ``out_dir`` and launches the Julia driver (cwd = the DCAlign clone). The
    driver appends one TSV row per sequence to ``out_tsv`` (default
    ``out_dir/dcalign_out.tsv``), flushing each row so a killed run leaves a
    valid partial cache. On nonzero exit we raise ``RuntimeError`` with the full
    stdout+stderr (loud, no fallback — the ProteinMPNN policy).

    ``seqs`` are raw, gap-free integer arrays (residues 1..20). ``out_tsv`` may
    already contain rows for *other* ids (append-mode resume); we parse it after
    the run and return exactly the rows for the requested ``seq_ids``.
    """
    if len(seqs) != len(seq_ids):
        raise ValueError(f"len(seqs)={len(seqs)} != len(seq_ids)={len(seq_ids)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = Path(out_tsv) if out_tsv is not None else out_dir / "dcalign_out.tsv"

    _write_queries(seqs, seq_ids, out_dir / "queries.fasta")
    J_dca, h_dca = model_to_dcalign_arrays(model)
    (out_dir / "model_J.bin").write_bytes(J_dca.astype("<f8").tobytes(order="F"))
    (out_dir / "model_h.bin").write_bytes(h_dca.astype("<f8").tobytes(order="F"))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "L": int(model.L),
                "q": int(model.q),
                "maxiter": int(ctx.maxiter),
                "seed": int(ctx.seed),
                "pcount": float(ctx.pcount),
                "lambda_spec": str(lambda_spec),
                "alphabet": ALPHABET,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if lambda_spec != "flat":
        # The empirical "deltan" prior is built (Julia-side) from the model-frame
        # seed MSA; stage it as seed.ins next to the model binaries (spec §10.13).
        # Local import: keep the bridge decoupled from the model module (and the
        # MCMC kernel it pulls in) at import time — this runs cluster-side only.
        from SBM.energy.model import load_seed_msa

        msa = load_seed_msa(model.source)
        if msa.ndim != 2 or msa.shape[1] != model.L:
            raise ValueError(
                f"seed MSA for {model.name!r} has shape {msa.shape}, expected "
                f"(N, L={model.L}); the deltan prior must be built in the model's "
                "exact frame"
            )
        _write_seed_ins(msa, out_dir / "seed.ins")

    cmd = [
        str(ctx.julia),
        f"--project={ctx.julia_project}",
        str(ctx.driver_jl),
        str(out_dir),
        str(out_tsv),
    ]
    env = dict(os.environ)
    env.setdefault("JULIA_NUM_THREADS", str(ctx.threads))
    log.info("DCAlign: aligning %d sequence(s) under model %r (L=%d)", len(seqs), model.name, model.L)
    log.debug("running: %s (cwd=%s)", " ".join(cmd), ctx.dcalign_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ctx.dcalign_path), env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_dcalign.jl failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not out_tsv.is_file():
        raise RuntimeError(
            f"run_dcalign.jl reported success but {out_tsv} is missing.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    by_id = read_alignment_cache(out_tsv)
    results: list[DCAlignResult] = []
    missing: list[str] = []
    for sid in seq_ids:
        if sid not in by_id:
            missing.append(sid)
        else:
            results.append(by_id[sid])
    if missing:
        raise RuntimeError(
            f"run_dcalign.jl did not emit rows for {len(missing)} requested id(s): "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return results


# ── TSV cache I/O (shared format with the Julia driver) ──────────────────────


def _parse_bool(token: str) -> bool:
    return token.strip().lower() in ("1", "true", "t", "yes")


def _parse_row(line: str) -> DCAlignResult:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(TSV_COLUMNS):
        raise ValueError(
            f"DCAlign TSV row has {len(parts)} fields, expected {len(TSV_COLUMNS)} "
            f"({TSV_HEADER}); offending row: {line!r}"
        )
    seq_id, frame, energy, converged, used_dec, n_iter = parts
    e_str = energy.strip().lower()
    e_val = math.nan if e_str in ("", "nan") else float(energy)
    return DCAlignResult(
        seq_id=seq_id,
        aligned_frame=frame,
        dcalign_energy=e_val,
        converged=_parse_bool(converged),
        used_decimation=_parse_bool(used_dec),
        n_iter=int(n_iter) if n_iter.strip() else 0,
    )


def format_row(res: DCAlignResult) -> str:
    """Inverse of :func:`_parse_row` — one TSV line for ``res`` (round-trips)."""
    energy = "nan" if math.isnan(res.dcalign_energy) else repr(res.dcalign_energy)
    return "\t".join(
        (
            res.seq_id,
            res.aligned_frame,
            energy,
            "true" if res.converged else "false",
            "true" if res.used_decimation else "false",
            str(res.n_iter),
        )
    )


def write_alignment_cache(path: Path | str, results: list[DCAlignResult]) -> None:
    """Write a gathered ``alignments.tsv`` (header + one row per result)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [TSV_HEADER] + [format_row(r) for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_alignment_cache(tsv_path: Path | str) -> dict[str, DCAlignResult]:
    """Read a DCAlign TSV (shard or gathered ``alignments.tsv``) into a dict.

    Keyed by ``seq_id``. Tolerates an optional header line. Raises on a
    duplicate id (the index→energy mapping must be unambiguous).
    """
    tsv_path = Path(tsv_path)
    out: dict[str, DCAlignResult] = {}
    with open(tsv_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.startswith(TSV_COLUMNS[0] + "\t"):  # header
                continue
            res = _parse_row(line)
            if res.seq_id in out:
                raise ValueError(f"duplicate seq_id {res.seq_id!r} in DCAlign cache {tsv_path}")
            out[res.seq_id] = res
    return out
