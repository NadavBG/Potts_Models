"""Build the DCAlign warm-start fixed-point probe (iter-003 Phase-B, §10.x).

The probe asks, per worse-than-native home pair: if DCAlign's belief propagation
is *initialised at the native frame* (instead of randomly), does it STAY there
(native is a stable fixed point — BP just never lands in its basin from random
starts → **case A**, a search/init problem → anneal) or does it flow to the same
worse frame the production run found (native is not a fixed point of DCAlign's
objective → **case B**, the objective genuinely prefers the other frame → stop
tuning)? The alignment runs the warm-start driver
(``src/SBM/julia/run_dcalign_warmstart.jl``), which calls DCAlign as a pinned,
unmodified library — the clone is never edited (no Mac↔Midway fork divergence).

This script stages a **self-contained** probe run dir: per-model in-dirs with the
model binaries (~30 MB each), ``seed.ins`` (deltan prior), the raw queries and the
length-L native frames to warm-start from. Midway then just runs ``julia
run_dcalign_warmstart.jl <in_dir> <out_tsv>`` per model — no cluster-side staging
logic to port (the binaries are small enough to push). Resource params
(cpus/mem/array) are set Midway-side, not here.

By default the curated set is read from the multi-seed sweep's ``roles.json`` so
the warm-start probe runs on the *same* 24 sequences (16 hardest worse-than-native
+ 8 already-recovered controls) as the seed sweep — directly comparable. Frames,
models and the seed MSA come from the canonical iter-002 source run.

Usage::

    python scripts/build_dcalign_warmstart.py \
        --src-run-dir combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior \
        --roles combine/combine-CM-PPIC-dcalign-seedsweep/roles.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

import SBM.provenance as provenance
from SBM.energy import datasets
from SBM.energy.encoding import GAP
from SBM.energy.hmm import ProfileHMM
from SBM.energy.model import load_model, load_seed_msa
from SBM.utils.dcalign_score import (
    ALPHABET,
    _ints_to_str,
    _write_queries,
    _write_seed_ins,
    model_to_dcalign_arrays,
)

log = logging.getLogger(__name__)

#: Pinned DCAlign defaults for the probe (match the iter-002 production run).
DEFAULT_MAXITER = 2000
DEFAULT_PCOUNT = 1e-3
DEFAULT_LAMBDA_SPEC = "deltan"
DEFAULT_SEED = 0


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def _resolve_clone(explicit: Path | None) -> Path | None:
    import os
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("DCALIGN_PATH")
    return Path(env).expanduser() if env else None


def load_roles(path: Path) -> dict[str, str]:
    roles = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"{path} is not a non-empty id->role map")
    return roles


def stage_model_indir(
    model, records: list[datasets.QueryRecord], in_dir: Path,
    init_frames: dict[str, np.ndarray] | None, *, maxiter: int, seed: int, pcount: float,
    lambda_spec: str, init_mode: str, beta0: float,
) -> dict:
    """Write one self-contained in-dir for ``model``'s home pairs.

    Mirrors :func:`SBM.utils.dcalign_score.align_sequences` staging (model
    binaries + ``seed.ins`` + ``queries.fasta``). If ``init_frames`` is given,
    also writes ``init.fasta`` — the length-L frame BP warm-starts from (native or
    fields-MAP per ``init_mode``), which must be insert-free (non-gap count == raw
    query length; the (x,n) warm start cannot represent inserts). If ``init_frames
    is None`` (``init_mode="random"``), no init file is written and the driver
    starts BP from a random init (the anneal-from-hot sweep, ``beta0 < 1``).
    """
    in_dir.mkdir(parents=True, exist_ok=True)
    raw_seqs, init_lines, ids = [], [], []
    for r in records:
        if r.ints.size != model.L:
            raise ValueError(
                f"{r.id}: native frame length {r.ints.size} != L={model.L} for {model.name}")
        raw = r.ints[r.ints != GAP]  # gap-free residues, in order (the query)
        raw_seqs.append(raw)
        ids.append(r.id)
        if init_frames is not None:
            frame = np.asarray(init_frames[r.id], dtype=np.int64)
            n_match = int(np.count_nonzero(frame != GAP))
            if frame.size != model.L or n_match != raw.size:
                raise ValueError(
                    f"{r.id}: {init_mode} init frame must be length L={model.L} and insert-free "
                    f"({raw.size} residues); got length {frame.size} with {n_match} residues")
            init_lines.append(f">{r.id}\n{_ints_to_str(frame)}")

    _write_queries(raw_seqs, ids, in_dir / "queries.fasta")
    if init_frames is not None:
        (in_dir / "init.fasta").write_text("\n".join(init_lines) + "\n", encoding="utf-8")

    J_dca, h_dca = model_to_dcalign_arrays(model)
    (in_dir / "model_J.bin").write_bytes(J_dca.astype("<f8").tobytes(order="F"))
    (in_dir / "model_h.bin").write_bytes(h_dca.astype("<f8").tobytes(order="F"))

    if lambda_spec != "flat":
        msa = load_seed_msa(model.source)
        if msa.ndim != 2 or msa.shape[1] != model.L:
            raise ValueError(
                f"seed MSA for {model.name!r} has shape {msa.shape}, expected (N, L={model.L})")
        _write_seed_ins(msa, in_dir / "seed.ins")

    (in_dir / "meta.json").write_text(
        json.dumps({
            "L": int(model.L), "q": int(model.q), "maxiter": int(maxiter),
            "seed": int(seed), "pcount": float(pcount), "lambda_spec": str(lambda_spec),
            "init_mode": str(init_mode), "beta0": float(beta0), "alphabet": ALPHABET,
        }, indent=2),
        encoding="utf-8",
    )
    log.info("staged %d home pair(s) for %r -> %s", len(ids), model.name, in_dir)
    return {"model": model.name, "n_queries": len(ids), "ids": ids,
            "model_sha256": model.sha256, "in_dir": str(in_dir)}


def _iteration_note(args, beta0: float, clone_commit: str | None, src: Path,
                    n_curated: int, n_recover: int, n_control: int) -> str:
    """The per-run-dir iteration_note.md, worded for the init mode."""
    if args.init == "random":
        title = f"DCAlign anneal-from-hot run (random init, beta0={beta0:g})"
        init_desc = (f"random BP messages, annealed from beta0={beta0:g} UP to the physical "
                     "temperature (beta>=1) — the couplings-aware search the §10.17 native "
                     "warm-start showed is needed (multi-seed and fields-MAP init both failed)")
        reading = ("Reading: from a random start on the beta0-smoothed landscape cooled to beta=1, "
                   "does BP REACH native quality (delta_E~0)? Across the sweep, the smallest beta0 "
                   "that recovers most worse pairs is the production anneal schedule.")
    elif args.init == "map":
        title = "DCAlign fields-MAP-init test"
        init_desc = "the fields-MAP / Viterbi frame (production-legal — no ground truth)"
        reading = ("Reading: BP from the fields-MAP frame. REACHES native quality for most worse "
                   "pairs => MAP-init is the production recipe; else fall back to annealing.")
    else:
        title = "DCAlign warm-start fixed-point probe"
        init_desc = "the native home-model frame"
        reading = ("Reading: BP from native. STAYS at native => case A (a reachable fixed point the "
                   "random-init runs missed); flows to the worse frame => case B (stop tuning).")
    return (
        f"# {title}\n\n"
        f"Init: {init_desc}.\n"
        f"Curated set ({n_curated}: {n_recover} hardest worse-than-native + {n_control} controls) "
        f"reused from the multi-seed sweep for comparability.\n"
        f"Source frames/models: `{src}`. Prior `{args.lambda_spec}`, pcount={args.pcount}, "
        f"maxiter={args.maxiter}, beta0={beta0:g}.\n\n"
        f"DCAlign clone PINNED + UNMODIFIED (commit `{clone_commit}`, infernet-h2020/DCAlign); the "
        f"warm-start lives in our driver `src/SBM/julia/run_dcalign_warmstart.jl` (clone not edited "
        f"=> no Mac<->Midway divergence). The sbatch enforces the pin.\n\n"
        f"COMPUTE NODE ONLY — submit `pipeline/external/sbatch_dcalign_warmstart.sh` from the login "
        f"node (2-task array per model, cpus=4/mem=12G/time=2h). Do NOT run the driver on the login "
        f"node.\n\n{reading}\n"
    )


def _print_handoff(args, clone_commit, out_root: Path, is_sweep: bool, runs: list) -> None:
    print(f"\n=== built (init={args.init}, {'sweep' if is_sweep else 'single'}) ===")
    for beta0, rd in runs:
        print(f"  beta0={beta0:g} -> {rd}")
    print(f"\nclone pin: {clone_commit} (the sbatch enforces this on Midway)\n")
    print("Hand-off — submit from the Midway LOGIN node (it dispatches to compute nodes):")
    print("  bash scripts/sync_models.sh push")
    if is_sweep:
        print("  # on Midway — one 2-task array per beta0 dir, all concurrent:")
        print(f"  for d in {out_root}/beta*/ ; do "
              f"sbatch pipeline/external/sbatch_dcalign_warmstart.sh \"$d\" ; done")
        print("  squeue --me                                # wait for all to clear, then on the Mac:")
        print("  bash scripts/sync_models.sh pull")
        print(f"  .venv/bin/python scripts/analyze_dcalign_warmstart.py "
              f"--sweep-root {out_root} --init-kind random")
    else:
        print(f"  sbatch pipeline/external/sbatch_dcalign_warmstart.sh {out_root}")
        print("  squeue --me ; bash scripts/sync_models.sh pull   # then, on the Mac:")
        print(f"  .venv/bin/python scripts/analyze_dcalign_warmstart.py "
              f"--run-dir {out_root} --init-kind {args.init}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-run-dir", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign/iter-002-nonuniform-prior"),
                   help="canonical source: models.json + query/ + dcalign cache (default iter-002)")
    p.add_argument("--roles", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-seedsweep/roles.json"),
                   help="id->role curated set (default: reuse the seed sweep's, for comparability)")
    p.add_argument("--out-root", type=Path,
                   default=Path("combine/combine-CM-PPIC-dcalign-warmstart"),
                   help="probe run dir to create")
    p.add_argument("--dcalign-path", type=Path, default=None,
                   help="DCAlign clone (for recording the pinned commit); else $DCALIGN_PATH")
    p.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    p.add_argument("--pcount", type=float, default=DEFAULT_PCOUNT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--lambda-spec", default=DEFAULT_LAMBDA_SPEC, choices=("deltan", "flat"))
    p.add_argument("--init", default="native", choices=("native", "map", "random"),
                   help="BP init: native (home-model frame, the fixed-point probe), map "
                        "(fields-MAP/Viterbi frame), or random (no init file — the anneal-from-hot "
                        "sweep, paired with --beta0/--beta0-values < 1)")
    p.add_argument("--beta0", type=float, default=1.0,
                   help="starting inverse-temperature for the BP anneal (1.0 = DCAlign's behaviour; "
                        "<1 starts on the smoothed landscape and ramps up)")
    p.add_argument("--beta0-values", type=float, nargs="+", default=None,
                   help="sweep these beta0 values -> one self-contained run dir per value under "
                        "<out-root>/beta<v>/ (overrides --beta0)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = dt.datetime.now(dt.timezone.utc)

    src = args.src_run_dir
    models_json = src / "models.json"
    model_entries = json.loads(models_json.read_text(encoding="utf-8"))["models"]
    models = {m["name"]: load_model(m["model_path"], name=m["name"]) for m in model_entries}

    roles = load_roles(args.roles)
    records = datasets.read_query_fasta(src / "query" / "query.fasta", src / "query" / "groups.json")
    by_id = {r.id: r for r in records}
    missing = [sid for sid in roles if sid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} curated id(s) not in {src}/query: {missing[:5]}")

    # Group curated home pairs by their home model (skip any without a home model).
    per_model: dict[str, list[datasets.QueryRecord]] = {name: [] for name in models}
    skipped = []
    for sid in roles:
        r = by_id[sid]
        if r.origin_model in per_model:
            per_model[r.origin_model].append(r)
        else:
            skipped.append(sid)
    if skipped:
        log.warning("skipped %d curated id(s) with no home model: %s", len(skipped), skipped[:5])

    # Per-id init frame (length-L, insert-free) for native/map; random init writes
    # no init file (init_frames stays None). The MAP path is built once per model.
    init_frames: dict[str, np.ndarray] | None
    if args.init == "random":
        init_frames = None
    else:
        init_frames = {}
        for name, recs in per_model.items():
            if not recs:
                continue
            if args.init == "map":
                hmm = ProfileHMM.from_model(models[name], load_seed_msa(models[name].source))
            for r in recs:
                if args.init == "native":
                    init_frames[r.id] = r.ints
                else:
                    raw_ints = r.ints[r.ints != GAP]
                    init_frames[r.id] = hmm.path_to_frame(hmm.viterbi(raw_ints), raw_ints)

    curated_ids = [sid for sid in roles if sid in by_id and by_id[sid].origin_model in models]
    raw = [by_id[sid].ints[by_id[sid].ints != GAP] for sid in curated_ids]
    groups = json.loads((src / "query" / "groups.json").read_text(encoding="utf-8"))
    clone = _resolve_clone(args.dcalign_path)
    clone_commit = _git_commit(clone) if clone is not None else None
    n_recover = sum(1 for v in roles.values() if v == "recover")
    n_control = sum(1 for v in roles.values() if v == "control")

    def _stage_run(run_dir: Path, beta0: float) -> None:
        """Stage one self-contained run dir (per-model in-dirs + provenance) at beta0."""
        run_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        for name, recs in per_model.items():
            if not recs:
                log.warning("model %r has no curated home pairs; skipping", name)
                continue
            staged.append(stage_model_indir(
                models[name], recs, run_dir / name, init_frames,
                maxiter=args.maxiter, seed=args.seed, pcount=args.pcount,
                lambda_spec=args.lambda_spec, init_mode=args.init, beta0=beta0))
        (run_dir / "models.json").write_text(models_json.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "roles.json").write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")
        (run_dir / "query").mkdir(exist_ok=True)
        _write_queries(raw, curated_ids, run_dir / "query" / "query.fasta")
        (run_dir / "query" / "groups.json").write_text(
            json.dumps({sid: groups[sid] for sid in curated_ids if sid in groups}, indent=2) + "\n",
            encoding="utf-8")
        options = {
            "maxiter": args.maxiter, "pcount": args.pcount, "seed": args.seed,
            "init_mode": args.init, "beta0": beta0, "lambda_spec": args.lambda_spec,
            "src_run_dir": str(src), "roles": str(args.roles), "dcalign_clone_commit": clone_commit,
            "dcalign_clone_remote": "https://github.com/infernet-h2020/DCAlign",
            "n_curated": len(curated_ids), "n_recover": n_recover, "n_control": n_control,
            "staged": staged,
        }
        manifest = provenance.build_run_manifest(
            run_id="build_dcalign_warmstart", command_line=provenance.current_command_line(),
            inputs={"models_json": models_json, "roles": args.roles,
                    "query_fasta": src / "query" / "query.fasta"},
            options=options, seed=args.seed, started_at=started_at,
            finished_at=dt.datetime.now(dt.timezone.utc), output_path=run_dir / "models.json",
            extra={"model_sha256": {m["name"]: m.get("model_sha256") for m in model_entries}})
        provenance.save_run_manifest(manifest, run_dir / "warmstart_manifest.json")
        (run_dir / "iteration_note.md").write_text(
            _iteration_note(args, beta0, clone_commit, src, len(curated_ids), n_recover, n_control),
            encoding="utf-8")

    beta0_list = args.beta0_values if args.beta0_values is not None else [args.beta0]
    is_sweep = args.beta0_values is not None
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    runs = [(b, (out_root / f"beta{b:g}") if is_sweep else out_root) for b in beta0_list]
    for beta0, rd in runs:
        _stage_run(rd, beta0)

    if is_sweep:
        (out_root / "sweep_meta.json").write_text(json.dumps({
            "kind": "annealsweep", "init": args.init, "beta0_values": beta0_list,
            "maxiter": args.maxiter, "pcount": args.pcount, "lambda_spec": args.lambda_spec,
            "src_run_dir": str(src), "dcalign_clone_commit": clone_commit,
            "run_dirs": {f"{b:g}": str(rd) for b, rd in runs},
        }, indent=2) + "\n", encoding="utf-8")

    _print_handoff(args, clone_commit, out_root, is_sweep, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
