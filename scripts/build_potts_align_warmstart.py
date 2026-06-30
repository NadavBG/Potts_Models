"""Stage the iter-003 §10.20 Midway batch: warm-start DCAlign-BP at the
couplings-aware Potts-align frame (M1), the basin-width probe (M3), and the
compute_en readout (M4).

The Mac-side aligner (:mod:`SBM.energy.potts_align`) already recovers the native
(global-Potts-min) frame for the low-gap home pairs by *exact enumeration* and the
mid-gap ones by warm-started SA. This batch runs the remaining cluster tests in
one shot (deltan Λ ⇒ Linux only — the real runs are Midway):

* **sastart** (M1): warm-start BP at the production aligner's best frame, at
  β₀=1.0 (pure refine) and β₀=0.5 (hybrid anneal). Does BP keep/refine it to
  native — in particular for the 3 high-gap stragglers (CM-186/289/syn-186) whose
  SA frame is far closer to native than DCAlign's own?
* **perturbed** (M3): warm-start BP at native with ``k`` columns reassigned
  (k∈{1,2,4,8}); the k at which BP stops returning to native is the basin radius.
* **diag** (M4): 0-sweep ``DCAlign.compute_en`` at the native frame and at
  DCAlign's iter-002 frame — the Potts energy DCAlign assigns each (a gauge
  cross-check; NOT a Λ free energy, §10.17).

Each ``<out-root>/<subdir>/`` is a self-contained warm-start run dir (per-model
in-dirs with binaries + seed.ins + queries + init.fasta + meta) that Midway runs
with ``pipeline/external/sbatch_dcalign_warmstart.sh``. Reuses
:func:`build_dcalign_warmstart.stage_model_indir`. The DCAlign clone stays pinned
+ unmodified; the only new Julia is the ``n_diag_sweeps`` readout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
from build_dcalign_warmstart import _git_commit, _resolve_clone, load_roles, stage_model_indir
from SBM.energy import datasets
from SBM.energy.encoding import GAP, seq_to_ints
from SBM.energy.hmm import ProfileHMM
from SBM.energy.model import load_model, load_seed_msa
from SBM.energy.potts_align import SASchedule, perturb_frame, potts_align
from SBM.utils.dcalign_score import _write_queries, read_alignment_cache

log = logging.getLogger(__name__)

DEFAULT_LAMBDA_SPEC = "deltan"
DEFAULT_MAXITER = 2000
DEFAULT_PCOUNT = 1e-3


def _stage_run(
    run_dir: Path, per_model: dict, models: dict, init_frames: dict[str, np.ndarray],
    *, models_json_text: str, roles: dict, curated_ids: list[str], raw_by_id: dict,
    groups: dict, init_mode: str, beta0: float, lambda_spec: str, maxiter: int,
    pcount: float, seed: int, clone_commit: str | None, src: Path, extra_meta: dict | None,
    started: dt.datetime, note: str,
) -> None:
    """One self-contained warm-start run dir (per-model in-dirs + top-level contract files)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for name, recs in per_model.items():
        present = [r for r in recs if r.id in init_frames]
        if not present:
            continue
        staged.append(stage_model_indir(
            models[name], present, run_dir / name, {r.id: init_frames[r.id] for r in present},
            maxiter=maxiter, seed=seed, pcount=pcount, lambda_spec=lambda_spec,
            init_mode=init_mode, beta0=beta0, extra_meta=extra_meta))
    ids = [sid for sid in curated_ids if sid in init_frames]
    # Filtered models.json: only the models actually staged (in order), so the
    # sbatch 2-model array maps task->model correctly even for CM-only dirs (the
    # high-k perturbation dirs have no eligible PPIC sequences). Submit such a dir
    # with --array=0 (one task); see MIDWAY_RUN.md.
    staged_names = {s["model"] for s in staged}
    all_entries = json.loads(models_json_text)["models"]
    kept = [m for m in all_entries if m["name"] in staged_names]
    (run_dir / "models.json").write_text(json.dumps({"models": kept}, indent=2) + "\n",
                                         encoding="utf-8")
    (run_dir / "roles.json").write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")
    (run_dir / "query").mkdir(exist_ok=True)
    _write_queries([raw_by_id[s] for s in ids], ids, run_dir / "query" / "query.fasta")
    (run_dir / "query" / "groups.json").write_text(
        json.dumps({s: groups[s] for s in ids if s in groups}, indent=2) + "\n", encoding="utf-8")
    options = {"init_mode": init_mode, "beta0": beta0, "lambda_spec": lambda_spec,
               "maxiter": maxiter, "pcount": pcount, "seed": seed, "extra_meta": extra_meta,
               "n_queries": len(ids), "src_run_dir": str(src),
               "dcalign_clone_commit": clone_commit, "staged": staged}
    manifest = provenance.build_run_manifest(
        run_id="build_potts_align_warmstart", command_line=provenance.current_command_line(),
        inputs={"models_json": src / "models.json", "query_fasta": src / "query" / "query.fasta"},
        options=options, seed=seed, started_at=started,
        finished_at=dt.datetime.now(dt.timezone.utc), output_path=run_dir / "models.json")
    provenance.save_run_manifest(manifest, run_dir / "warmstart_manifest.json")
    (run_dir / "iteration_note.md").write_text(note, encoding="utf-8")
    log.info("staged %s (%d queries) -> %s", init_mode, len(ids), run_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"))
    p.add_argument("--roles", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-seedsweep/roles.json"))
    p.add_argument("--out-root", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-pottsinit"))
    p.add_argument("--modes", nargs="+", default=["sastart", "perturbed", "diag"],
                   choices=["sastart", "perturbed", "diag"])
    p.add_argument("--beta0-values", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--perturb-k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--dcalign-path", type=Path, default=None)
    p.add_argument("--lambda-spec", default=DEFAULT_LAMBDA_SPEC, choices=("deltan", "flat"))
    p.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    p.add_argument("--pcount", type=float, default=DEFAULT_PCOUNT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sa-restarts", type=int, default=64)
    p.add_argument("--sa-steps", type=int, default=20000)
    p.add_argument("--enum-max-frames", type=int, default=200_000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = dt.datetime.now(dt.timezone.utc)
    src = args.src_run_dir
    sched = SASchedule(n_restarts=args.sa_restarts, n_steps=args.sa_steps,
                       enum_max_frames=args.enum_max_frames)
    models_json_text = (src / "models.json").read_text(encoding="utf-8")
    model_entries = json.loads(models_json_text)["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}
    caches = {m["name"]: read_alignment_cache(
        src / "dcalign" / "cache" / m["name"] / "alignments.tsv") for m in model_entries}
    roles = load_roles(args.roles)
    records = datasets.read_query_fasta(src / "query" / "query.fasta", src / "query" / "groups.json")
    by_id = {r.id: r for r in records}
    curated_ids = sorted(s for s in roles if s in by_id and by_id[s].origin_model in models)
    per_model: dict[str, list] = {name: [] for name in models}
    for sid in curated_ids:
        per_model[by_id[sid].origin_model].append(by_id[sid])
    raw_by_id = {s: by_id[s].ints[by_id[s].ints != GAP] for s in curated_ids}
    groups = json.loads((src / "query" / "groups.json").read_text(encoding="utf-8"))
    clone_commit = _git_commit(_resolve_clone(args.dcalign_path)) \
        if _resolve_clone(args.dcalign_path) else None
    common = dict(models_json_text=models_json_text, roles=roles, curated_ids=curated_ids,
                  raw_by_id=raw_by_id, groups=groups, lambda_spec=args.lambda_spec,
                  maxiter=args.maxiter, pcount=args.pcount, seed=args.seed,
                  clone_commit=clone_commit, src=src, started=started)
    id_seeds = {s: int(v) for s, v in
                zip(curated_ids, np.random.SeedSequence(args.seed).generate_state(len(curated_ids)))}
    built = []

    if "sastart" in args.modes:
        hmms: dict[str, ProfileHMM] = {}
        sa_frames = {}
        for sid in curated_ids:
            r = by_id[sid]
            model = models[r.origin_model]
            if r.origin_model not in hmms:
                hmms[r.origin_model] = ProfileHMM.from_model(model, load_seed_msa(model.source))
            raw = raw_by_id[sid]
            warm = [hmms[r.origin_model].path_to_frame(hmms[r.origin_model].viterbi(raw), raw)]
            dca = caches[r.origin_model].get(sid)
            if dca and dca.aligned_frame:
                warm.append(seq_to_ints(dca.aligned_frame))
            sa_frames[sid] = potts_align(raw, model, seed=id_seeds[sid], schedule=sched,
                                         sequence_id=sid, init_frames=warm).best_frame
        for b in args.beta0_values:
            rd = args.out_root / f"sa-beta{b:g}"
            _stage_run(rd, per_model, models, sa_frames, init_mode="potts-align", beta0=b,
                       extra_meta=None,
                       note=_note("potts-align", b, args, clone_commit, src), **common)
            built.append((f"sastart beta0={b:g}", rd))

    if "perturbed" in args.modes:
        recover_ids = [s for s in curated_ids if roles[s] == "recover"]
        for k in args.perturb_k_values:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, k]).generate_state(1)[0])
            frames = {}
            for sid in recover_ids:
                native = np.asarray(by_id[sid].ints, dtype=np.int64)
                n_occ = int(np.count_nonzero(native != GAP))
                if k <= min(n_occ, native.size - n_occ):
                    frames[sid] = perturb_frame(native, k, rng=rng)
            if not frames:
                log.warning("perturb k=%d: no eligible sequences (insufficient gaps); skipping", k)
                continue
            rd = args.out_root / f"perturb-k{k}"
            _stage_run(rd, per_model, models, frames, init_mode=f"perturbed-k{k}", beta0=1.0,
                       extra_meta=None,
                       note=_note(f"perturbed-k{k}", 1.0, args, clone_commit, src), **common)
            built.append((f"perturbed k={k}", rd))

    if "diag" in args.modes:
        native_frames = {s: np.asarray(by_id[s].ints, dtype=np.int64) for s in curated_ids}
        dca_frames = {s: seq_to_ints(caches[by_id[s].origin_model][s].aligned_frame)
                      for s in curated_ids
                      if caches[by_id[s].origin_model].get(s) and
                      caches[by_id[s].origin_model][s].aligned_frame}
        for label, frames in (("native", native_frames), ("dcalign", dca_frames)):
            rd = args.out_root / f"diag-{label}"
            _stage_run(rd, per_model, models, frames, init_mode=f"diag-{label}", beta0=1.0,
                       extra_meta={"n_diag_sweeps": 0},
                       note=_note(f"diag-{label}", 1.0, args, clone_commit, src), **common)
            built.append((f"diag {label}", rd))

    (args.out_root).mkdir(parents=True, exist_ok=True)
    (args.out_root / "batch_meta.json").write_text(json.dumps({
        "modes": args.modes, "beta0_values": args.beta0_values,
        "perturb_k_values": args.perturb_k_values, "sa_schedule": sched.as_dict(),
        "lambda_spec": args.lambda_spec, "src_run_dir": str(src),
        "dcalign_clone_commit": clone_commit,
        "run_dirs": [str(rd) for _, rd in built],
    }, indent=2) + "\n", encoding="utf-8")
    print("\n=== staged Midway batch ===")
    for label, rd in built:
        print(f"  {label:<20} -> {rd}")
    print(f"\nclone pin: {clone_commit}")
    print("Hand-off (see MIDWAY_RUN.md): sync_models.sh push; on Midway submit each dir with "
          "pipeline/external/sbatch_dcalign_warmstart.sh; sync_models.sh pull; analyze on Mac.")
    return 0


def _note(init_mode: str, beta0: float, args, clone_commit, src: Path) -> str:
    return (
        f"# DCAlign warm-start run (init={init_mode}, beta0={beta0:g})\n\n"
        f"Part of the iter-003 §10.20 Midway batch (M1 sastart / M3 perturbed / M4 diag).\n"
        f"Source frames/models: `{src}`. Prior `{args.lambda_spec}`, pcount={args.pcount}, "
        f"maxiter={args.maxiter}.\n\n"
        f"DCAlign clone PINNED + UNMODIFIED (`{clone_commit}`); warm-start driver "
        f"`src/SBM/julia/run_dcalign_warmstart.jl`.\n"
        f"Submit `pipeline/external/sbatch_dcalign_warmstart.sh <this dir>` from the Midway login "
        f"node (compute-node only).\n")


if __name__ == "__main__":
    raise SystemExit(main())
