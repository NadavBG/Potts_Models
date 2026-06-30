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
    init_frames: dict[str, np.ndarray], *, maxiter: int, seed: int, pcount: float,
    lambda_spec: str, init_mode: str,
) -> dict:
    """Write one self-contained warm-start in-dir for ``model``'s home pairs.

    Mirrors :func:`SBM.utils.dcalign_score.align_sequences` staging (model
    binaries + ``seed.ins`` + ``queries.fasta``) and adds ``init.fasta`` — the
    length-L frame BP warm-starts from (native or fields-MAP per ``init_mode``),
    one per query id. The init must be insert-free (non-gap count == raw query
    length); the (x,n) warm start cannot represent inserts.
    """
    in_dir.mkdir(parents=True, exist_ok=True)
    raw_seqs, init_lines, ids = [], [], []
    for r in records:
        if r.ints.size != model.L:
            raise ValueError(
                f"{r.id}: native frame length {r.ints.size} != L={model.L} for {model.name}")
        raw = r.ints[r.ints != GAP]  # gap-free residues, in order (the query)
        frame = np.asarray(init_frames[r.id], dtype=np.int64)
        n_match = int(np.count_nonzero(frame != GAP))
        if frame.size != model.L or n_match != raw.size:
            raise ValueError(
                f"{r.id}: {init_mode} init frame must be length L={model.L} and insert-free "
                f"({raw.size} residues); got length {frame.size} with {n_match} residues")
        raw_seqs.append(raw)
        init_lines.append(f">{r.id}\n{_ints_to_str(frame)}")
        ids.append(r.id)

    _write_queries(raw_seqs, ids, in_dir / "queries.fasta")
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
            "init_mode": str(init_mode), "alphabet": ALPHABET,
        }, indent=2),
        encoding="utf-8",
    )
    log.info("staged %d home pair(s) for %r -> %s", len(ids), model.name, in_dir)
    return {"model": model.name, "n_queries": len(ids), "ids": ids,
            "model_sha256": model.sha256, "in_dir": str(in_dir)}


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
    p.add_argument("--init", default="native", choices=("native", "map"),
                   help="warm-start frame: native (home-model frame, the fixed-point probe) or "
                        "map (fields-MAP/Viterbi frame, the production-usable init test)")
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

    # Build the per-id warm-start init frame (length-L, insert-free): the native
    # home-model frame, or the fields-MAP/Viterbi frame (production-legal — no
    # ground truth). The MAP path is built once per model from h + seed MSA.
    init_frames: dict[str, np.ndarray] = {}
    for name, recs in per_model.items():
        if not recs:
            continue
        if args.init == "map":
            hmm = ProfileHMM.from_model(models[name], load_seed_msa(models[name].source))
        for r in recs:
            if args.init == "native":
                init_frames[r.id] = r.ints
            else:
                raw = r.ints[r.ints != GAP]
                init_frames[r.id] = hmm.path_to_frame(hmm.viterbi(raw), raw)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    staged = []
    for name, recs in per_model.items():
        if not recs:
            log.warning("model %r has no curated home pairs; skipping", name)
            continue
        staged.append(stage_model_indir(
            models[name], recs, out_root / name, init_frames,
            maxiter=args.maxiter, seed=args.seed, pcount=args.pcount,
            lambda_spec=args.lambda_spec, init_mode=args.init))

    # Run-dir level provenance + contract files.
    (out_root / "models.json").write_text(models_json.read_text(encoding="utf-8"), encoding="utf-8")
    (out_root / "roles.json").write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")
    (out_root / "query").mkdir(exist_ok=True)
    # Canonical query set = raw queries + groups for the curated ids (provenance).
    curated_ids = [sid for sid in roles if sid in by_id and by_id[sid].origin_model in models]
    raw = [by_id[sid].ints[by_id[sid].ints != GAP] for sid in curated_ids]
    _write_queries(raw, curated_ids, out_root / "query" / "query.fasta")
    groups = json.loads((src / "query" / "groups.json").read_text(encoding="utf-8"))
    (out_root / "query" / "groups.json").write_text(
        json.dumps({sid: groups[sid] for sid in curated_ids if sid in groups}, indent=2) + "\n",
        encoding="utf-8")

    clone = _resolve_clone(args.dcalign_path)
    clone_commit = _git_commit(clone) if clone is not None else None
    finished_at = dt.datetime.now(dt.timezone.utc)
    options = {
        "maxiter": args.maxiter, "pcount": args.pcount, "seed": args.seed,
        "init_mode": args.init,
        "lambda_spec": args.lambda_spec, "src_run_dir": str(src), "roles": str(args.roles),
        "dcalign_clone_commit": clone_commit,
        "dcalign_clone_remote": "https://github.com/infernet-h2020/DCAlign",
        "n_curated": len(curated_ids), "n_recover": sum(1 for v in roles.values() if v == "recover"),
        "n_control": sum(1 for v in roles.values() if v == "control"),
        "staged": staged,
    }
    manifest = provenance.build_run_manifest(
        run_id="build_dcalign_warmstart", command_line=provenance.current_command_line(),
        inputs={"models_json": models_json, "roles": args.roles,
                "query_fasta": src / "query" / "query.fasta"},
        options=options, seed=args.seed, started_at=started_at, finished_at=finished_at,
        output_path=out_root / "models.json",
        extra={"model_sha256": {m["name"]: m.get("model_sha256") for m in model_entries}},
    )
    provenance.save_run_manifest(manifest, out_root / "warmstart_manifest.json")

    is_map = args.init == "map"
    title = ("DCAlign fields-MAP-init test (production lever)" if is_map
             else "DCAlign warm-start fixed-point probe")
    init_desc = ("the fields-MAP / Viterbi frame (production-legal — no ground truth; the §10.17 "
                 "fixed-point probe showed a good *near*-native start reaches native quality)"
                 if is_map else "the native home-model frame")
    reading = (
        ("Reading the result: BP starts at the fields-MAP frame and runs DCAlign's real schedule. "
         "If it REACHES native quality (ΔE≈0) for most worse pairs, MAP-init + couplings-aware BP is "
         "the production recipe — adopt it as the dcalign align mode and run the full combine. If it "
         "does not, the MAP start is too far from native's basin → fall back to an annealing sweep.")
        if is_map else
        ("Reading the result: BP starts at the native frame and runs DCAlign's real schedule. "
         "If it STAYS at native (ΔE≈0) the native frame is a stable fixed point the production "
         "random-init runs missed (case A → anneal / better init). If it flows to the cached worse "
         "frame, native is not a fixed point of DCAlign's objective (case B → stop tuning)."))
    note = (
        f"# {title}\n\n"
        f"Init frame: {init_desc}.\n"
        f"Curated set ({len(curated_ids)}: "
        f"{options['n_recover']} hardest worse-than-native + {options['n_control']} controls) "
        f"reused from the multi-seed sweep for direct comparability.\n"
        f"Source frames/models: `{src}`. Prior: `{args.lambda_spec}`, pcount={args.pcount}, "
        f"maxiter={args.maxiter}.\n\n"
        f"DCAlign clone is a PINNED, UNMODIFIED dependency — commit `{clone_commit}` "
        f"(infernet-h2020/DCAlign). The warm-start lives entirely in our driver "
        f"`src/SBM/julia/run_dcalign_warmstart.jl`; the clone is not edited, so the Mac and "
        f"Midway clones cannot diverge. Verify the Midway clone is at the same commit before running.\n\n"
        f"COMPUTE NODE ONLY — do NOT run the driver on the Midway login node. Submit the array job "
        f"`pipeline/external/sbatch_dcalign_warmstart.sh` from the login node; it dispatches a 2-task "
        f"array (one per model, cpus=4/mem=12G/time=2h, runs concurrently) to compute nodes and enforces "
        f"the clone pin. Each task is real compute (12 seqs, ~10 min at cpus=4, ~4.5 GB peak like the "
        f"production deltan align). Edit the #SBATCH lines there if resources need tuning.\n\n"
        f"{reading}\n"
    )
    (out_root / "iteration_note.md").write_text(note, encoding="utf-8")

    # Midway hand-off (resources are cluster-side: cpus=1 per task, fan over the array;
    # the iter-002-pcsweep OOM was cpus=4/mem=8G).
    print(f"\n=== built warm-start probe (init={args.init}) ===")
    for s in staged:
        print(f"  {s['model']}: {s['n_queries']} home pairs -> {s['in_dir']}")
    print(f"\nclone pin: {clone_commit} (the sbatch enforces this on Midway)\n")
    print("Hand-off — submit the array from the Midway LOGIN node (it dispatches to compute nodes):")
    print("  bash scripts/sync_models.sh push                       # Mac -> Midway")
    print("  ssh $SBM_MIDWAY_HOST ; cd <repo> ; mkdir -p logs       # then on Midway:")
    print(f"  sbatch pipeline/external/sbatch_dcalign_warmstart.sh {out_root}")
    print("  #   2-task array (CM, PPIC) cpus=4/mem=12G/time=2h, concurrent; ~10 min each.")
    print("  squeue --me                                            # wait until both tasks clear")
    print("  bash scripts/sync_models.sh pull                       # Midway -> Mac, then:")
    print(f"  .venv/bin/python scripts/analyze_dcalign_warmstart.py --run-dir {out_root} "
          f"--init-kind {args.init}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
