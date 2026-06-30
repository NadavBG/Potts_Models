"""E1 (iter-003 §10.20): couplings-aware Potts-energy alignment of the curated set.

The all-Mac, ground-truth-free production candidate. For each curated home pair
it aligns the *raw ungapped* query to its home model by minimizing the exact
in-frame Potts energy (exact enumeration where the gap count is small, multi-
restart SA otherwise — :mod:`SBM.energy.potts_align`), then compares that frame's
energy to the native frame and to DCAlign's iter-002 frame:

    delta_e_best   = E(potts-align frame) − E(native)   (≤0 ⇒ found native-or-better)
    delta_e_dcalign = E(DCAlign frame)    − E(native)   (the residual to beat)

A ``delta_e_best ≤ 0`` means the couplings-aware minimizer reached (or beat) the
native frame *without ever seeing it* — i.e. the production lever works with no
BP at all for that sequence. ``is_global_exact`` flags the enumerated (provably
global) cases. The aligner never sees the native frame; the ΔE comparison is
done here, in the analysis layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.encoding import GAP, seq_to_ints
from SBM.energy.hmm import ProfileHMM
from SBM.energy.model import load_model, load_seed_msa
from SBM.energy.potts import potts_energy
from SBM.energy.potts_align import SASchedule, potts_align
from SBM.utils.dcalign_score import read_alignment_cache

log = logging.getLogger(__name__)

EQUAL_TOL = 1.0  # a.u.; matches dcalign_baseline.DEFAULT_EQUAL_TOL


def classify(delta_e_best: float, tol: float) -> str:
    if delta_e_best < -tol:
        return "beats_native"
    if delta_e_best <= tol:
        return "is_native"
    return "wrong"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"))
    p.add_argument("--roles", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-seedsweep/roles.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-pottsalign"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-restarts", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--enum-max-frames", type=int, default=200_000)
    p.add_argument("--equal-tol", type=float, default=EQUAL_TOL)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = dt.datetime.now(dt.timezone.utc)
    src = args.src_run_dir
    sched = SASchedule(n_restarts=args.n_restarts, n_steps=args.n_steps,
                       enum_max_frames=args.enum_max_frames)

    model_entries = json.loads((src / "models.json").read_text())["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    caches = {
        m["name"]: read_alignment_cache(src / "dcalign" / "cache" / m["name"] / "alignments.tsv")
        for m in model_entries
    }

    roles = json.loads(args.roles.read_text())
    records = datasets.read_query_fasta(src / "query" / "query.fasta", src / "query" / "groups.json")
    by_id = {r.id: r for r in records}
    curated = sorted(sid for sid in roles if sid in by_id and by_id[sid].origin_model in models)
    # Stable per-id seeds derived from the one master seed (logged).
    id_seeds = {sid: int(s) for sid, s in
                zip(curated, np.random.SeedSequence(args.seed).generate_state(len(curated)))}
    hmms: dict[str, ProfileHMM] = {}  # built lazily; only the SA cases use the warm starts

    rows = []
    for sid in curated:
        r = by_id[sid]
        model = models[r.origin_model]
        native = np.asarray(r.ints, dtype=np.int64)
        raw = native[native != GAP]
        e_native = potts_energy(native, model)

        dca = caches[r.origin_model].get(sid)
        if dca is not None and dca.aligned_frame:
            dca_frame = seq_to_ints(dca.aligned_frame)
            e_dca = potts_energy(dca_frame, model)
            delta_dca = e_dca - e_native
        else:
            dca_frame, e_dca, delta_dca = None, float("nan"), float("nan")

        # Warm-start the SA branch from production-legal heuristic frames
        # (fields-MAP + DCAlign's own frame); enumeration ignores them.
        if r.origin_model not in hmms:
            hmms[r.origin_model] = ProfileHMM.from_model(model, load_seed_msa(model.source))
        hmm = hmms[r.origin_model]
        map_frame = hmm.path_to_frame(hmm.viterbi(raw), raw)
        init_frames = [map_frame] + ([dca_frame] if dca_frame is not None else [])

        res = potts_align(raw, model, seed=id_seeds[sid], schedule=sched,
                          sequence_id=sid, init_frames=init_frames)
        delta_best = res.best_energy - e_native

        rows.append({
            "sequence_id": sid, "model": r.origin_model, "group": r.group,
            "role": roles[sid], "n_residues": int(raw.size), "L": int(model.L),
            "n_gaps": int(model.L - raw.size), "method": res.method,
            "is_global_exact": res.is_global_exact, "n_frames": res.n_frames_evaluated,
            "e_native": e_native, "e_best": res.best_energy, "e_dcalign": e_dca,
            "delta_e_best": delta_best, "delta_e_dcalign": delta_dca,
            "outcome": classify(delta_best, args.equal_tol), "seed": id_seeds[sid],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(rows, args.out_dir / "potts_align_rows.tsv")
    summary = _summarize(rows, args.equal_tol)
    (args.out_dir / "potts_align_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_table(rows, summary)

    manifest = provenance.build_run_manifest(
        run_id="analyze_potts_align", command_line=provenance.current_command_line(),
        inputs={"models_json": src / "models.json", "roles": args.roles,
                "query_fasta": src / "query" / "query.fasta"},
        options={"seed": args.seed, "schedule": sched.as_dict(), "equal_tol": args.equal_tol,
                 "id_seeds": id_seeds, "src_run_dir": str(src)},
        seed=args.seed, started_at=started, finished_at=dt.datetime.now(dt.timezone.utc),
        output_path=args.out_dir / "potts_align_rows.tsv")
    provenance.save_run_manifest(manifest, args.out_dir / "potts_align_manifest.json")
    return 0


def _write_tsv(rows: list[dict], path: Path) -> None:
    cols = list(rows[0].keys())
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(rows: list[dict], tol: float) -> dict:
    def stats(subset: list[dict]) -> dict:
        n = len(subset)
        if not n:
            return {"n": 0}
        recovered = sum(1 for r in subset if r["delta_e_best"] <= tol)
        beats = sum(1 for r in subset if r["delta_e_best"] < -tol)
        # delta_e_dcalign is NaN for any pair lacking a DCAlign frame (e.g. controls);
        # filter explicitly and report the comparison count rather than letting an
        # all-NaN slice emit a RuntimeWarning and silently produce NaN.
        dca = np.array([r["delta_e_dcalign"] for r in subset], dtype=float)
        dca_finite = dca[~np.isnan(dca)]
        return {
            "n": n,
            "n_recovered": recovered, "frac_recovered": recovered / n,
            "n_beats_native": beats,
            "n_exact": sum(1 for r in subset if r["is_global_exact"]),
            "median_delta_e_best": float(np.median([r["delta_e_best"] for r in subset])),
            "n_dcalign_compared": int(dca_finite.size),
            "median_delta_e_dcalign": (float(np.median(dca_finite))
                                       if dca_finite.size else float("nan")),
        }

    recover = [r for r in rows if r["role"] == "recover"]
    control = [r for r in rows if r["role"] == "control"]
    return {
        "equal_tol": tol, "n_total": len(rows),
        "delta_e_convention": "delta_e_* = E_* - E_native; recovered iff delta_e_best <= tol",
        "recover": {"overall": stats(recover),
                    "enumerated": stats([r for r in recover if r["is_global_exact"]]),
                    "sa": stats([r for r in recover if not r["is_global_exact"]])},
        "control": stats(control),
    }


def _print_table(rows: list[dict], summary: dict) -> None:
    print(f"\n{'id':<30}{'role':<8}{'meth':<6}{'gaps':>5}{'dE_best':>9}{'dE_dca':>9}  outcome")
    for r in sorted(rows, key=lambda x: (x["role"], x["model"], x["sequence_id"])):
        print(f"{r['sequence_id']:<30}{r['role']:<8}{r['method']:<6}{r['n_gaps']:>5}"
              f"{r['delta_e_best']:>9.2f}{r['delta_e_dcalign']:>9.2f}  {r['outcome']}")
    rec = summary["recover"]["overall"]
    print(f"\nRECOVER {rec['n_recovered']}/{rec['n']} reach native-or-better "
          f"(median dE_best {rec['median_delta_e_best']:.2f} vs dcalign "
          f"{rec['median_delta_e_dcalign']:.2f}); {rec['n_beats_native']} beat native; "
          f"{rec['n_exact']} provably global. Controls "
          f"{summary['control'].get('n_recovered', 0)}/{summary['control'].get('n', 0)}.")


if __name__ == "__main__":
    raise SystemExit(main())
