#!/usr/bin/env python
"""Design sequences that are jointly low-energy under two Potts models.

Runs a batch of independent joint-annealing trajectories (``SBM.design.anneal``)
against ``E_tot = w_A·E_A + w_B·E_B`` and writes the trajectories, the final
designed sequences (with a real ``potts_align`` "polish" of each), a tidy table,
and a provenance manifest.

Models and the calibrated combining weights are read from an existing *combine*
run directory (its ``models.json`` and ``data/energy_weights.json``); everything
lands under ``<combine_run>/design/`` (a stable path — the combine pipeline drives
this) so it reuses those artifacts and rides ``scripts/sync_models.sh``'s
``combine/`` tree. Pass ``--model-a/-b`` + ``--w-a/-b`` to override.

Trajectories can be seeded from three kinds of start (``--start-*``): a **random**
sequence, a **CM natural** (model A), or a **PPIC natural** (model B). Natural
starts are drawn from each model's ``seed_msa`` (in ``models.json``), filtered to a
core length ``<= min(L_A, L_B)`` so they fit both frames; the selection is seeded
from the master seed and recorded in ``design_config.json`` so a local run and a
cluster run reproduce the same starts.

Example (fast Mac smoke, all-random, no polish):
    python scripts/design_two_model.py \\
        --combine-run combine/combine-CM-PPIC-potts/iter-001-potts-align-eval \\
        --start-random 4 --start-natural-a 0 --start-natural-b 0 \\
        --steps 5000 --seed 0 --no-polish
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from SBM import provenance
from SBM.design.anneal import (
    AnnealSchedule,
    ChainResult,
    anneal_chain,
    initial_state_from_frame,
)
from SBM.energy.encoding import GAP, ints_to_seq
from SBM.energy.model import load_model
from SBM.energy.potts_align import PTSchedule

log = logging.getLogger("design_two_model")

DEFAULT_COMBINE_RUN = "combine/combine-CM-PPIC-potts/iter-001-potts-align-eval"

# Polish schedules: "auto" lets potts_align pick per gap count (PTSchedule.for_gap_count);
# "fast" is a light ladder — the default. Because the polish is warm-started from the
# joint-MC frame it can only match-or-improve it, so a light ladder suffices and keeps a
# full run to minutes (a "default"/"thorough" cold-scale PT on a high-gap frame is minutes
# *per chain*). Use "default"/"thorough" for a high-quality polish at cluster scale.
_POLISH_SCHEDULES = {
    "auto": None,
    "fast": PTSchedule(n_replicas=8, n_blocks=800, n_restarts=2),
    "default": PTSchedule(),
    "thorough": PTSchedule.thorough(),
}

# Salt for the natural-start row selection RNG (kept distinct from the per-chain seeds).
_START_SELECT_SALT = 0x5EED


# --------------------------------------------------------------------------- #
# Config resolution (models, weights, natural-start selection)
# --------------------------------------------------------------------------- #

def _pick_natural_rows(msa_path: str | None, lo: int, cap: int, k: int,
                       rng: np.random.Generator, tag: str) -> list[int]:
    """Seeded selection of ``k`` seed-MSA rows whose ungapped core is in ``[lo, cap]``.

    ``cap = min(L_A, L_B)`` (fits both frames); ``lo = min_length`` (the deletion floor),
    so a natural start never begins below the documented ``N >= min_length`` invariant."""
    if not msa_path:
        raise SystemExit(f"natural_{tag} starts requested but models.json has no seed_msa "
                         f"for model {tag}")
    msa = np.load(msa_path)
    core = (msa != GAP).sum(axis=1)
    eligible = np.nonzero((core >= lo) & (core <= cap))[0]
    if eligible.size == 0:
        raise SystemExit(f"no {tag} naturals with core length in [{lo}, {cap}] "
                         f"(min_length .. min model L)")
    replace = k > eligible.size
    if replace:
        log.warning("only %d %s naturals have core in [%d, %d]; sampling %d WITH replacement",
                    eligible.size, tag, lo, cap, k)
    return [int(r) for r in rng.choice(eligible, size=k, replace=replace)]


def resolve_design_config(
    *,
    combine_run: str | None,
    schedule: AnnealSchedule,
    master_seed: int,
    start_random: int,
    start_natural_a: int,
    start_natural_b: int,
    do_polish: bool,
    polish_schedule: str,
    model_a: str | None = None,
    model_b: str | None = None,
    name_a: str | None = None,
    name_b: str | None = None,
    w_a: float | None = None,
    w_b: float | None = None,
) -> dict:
    """Build the self-contained ``design_config.json`` dict (Mac↔Midway contract).

    Resolves the two model paths/names, the combining weights, the two ``seed_msa``
    paths, and the *seeded* natural-start row selection (so local and cluster runs
    reproduce identical starts). ``schema_version`` 2 adds the start-mix fields.
    """
    if min(start_random, start_natural_a, start_natural_b) < 0:
        raise SystemExit("start counts must be >= 0")
    n_chains = start_random + start_natural_a + start_natural_b
    if n_chains < 1:
        raise SystemExit("need at least one chain (start_random + start_natural_a + start_natural_b >= 1)")

    info: dict = {"model_a_path": model_a, "model_b_path": model_b, "name_a": name_a,
                  "name_b": name_b, "w_a": w_a, "w_b": w_b, "seed_msa_a": None,
                  "seed_msa_b": None, "weights_source": None}
    L_a = L_b = None
    if combine_run:
        cr = Path(combine_run)
        models = json.loads((cr / "models.json").read_text(encoding="utf-8"))["models"]
        a, b = models[0], models[1]
        info["model_a_path"] = info["model_a_path"] or a["model_path"]
        info["model_b_path"] = info["model_b_path"] or b["model_path"]
        info["name_a"] = info["name_a"] or a["name"]
        info["name_b"] = info["name_b"] or b["name"]
        info["seed_msa_a"], info["seed_msa_b"] = a.get("seed_msa"), b.get("seed_msa")
        L_a, L_b = int(a["L"]), int(b["L"])
        if info["w_a"] is None or info["w_b"] is None:
            weights_json = cr / "data" / "energy_weights.json"
            w = json.loads(weights_json.read_text(encoding="utf-8"))
            info["w_a"] = info["w_a"] if info["w_a"] is not None else float(w["w_A"])
            info["w_b"] = info["w_b"] if info["w_b"] is not None else float(w["w_B"])
            info["weights_source"] = str(weights_json)
    missing = [k for k in ("model_a_path", "model_b_path", "w_a", "w_b") if info.get(k) is None]
    if missing:
        raise SystemExit(
            f"could not resolve {missing}; pass --combine-run or --model-a/--model-b + --w-a/--w-b")

    natural_a_rows: list[int] = []
    natural_b_rows: list[int] = []
    if start_natural_a or start_natural_b:
        if not combine_run:
            raise SystemExit("natural starts require --combine-run (naturals come from "
                             "models.json seed_msa)")
        cap = min(L_a, L_b)
        lo = schedule.min_length     # natural starts respect the deletion floor
        sel_rng = np.random.default_rng([master_seed, _START_SELECT_SALT])
        if start_natural_a:
            natural_a_rows = _pick_natural_rows(info["seed_msa_a"], lo, cap, start_natural_a, sel_rng, "A")
        if start_natural_b:
            natural_b_rows = _pick_natural_rows(info["seed_msa_b"], lo, cap, start_natural_b, sel_rng, "B")

    return {
        "schema_version": 2,
        "model_a_path": info["model_a_path"], "name_a": info["name_a"],
        "model_b_path": info["model_b_path"], "name_b": info["name_b"],
        "w_a": info["w_a"], "w_b": info["w_b"],
        "weights_source": info["weights_source"], "combine_run": combine_run,
        "seed_msa_a": info["seed_msa_a"], "seed_msa_b": info["seed_msa_b"],
        "master_seed": master_seed,
        "do_polish": do_polish, "polish_schedule": polish_schedule,
        "schedule": schedule.as_dict(),
        "start_random": start_random,
        "start_natural_a": start_natural_a,
        "start_natural_b": start_natural_b,
        "natural_a_rows": natural_a_rows,
        "natural_b_rows": natural_b_rows,
        "n_chains": n_chains,
    }


# --------------------------------------------------------------------------- #
# Per-chain start construction (shared by the local + cluster runners)
# --------------------------------------------------------------------------- #

def chain_start_spec(config: dict, i: int) -> tuple[str, str | None, int | None]:
    """``(start_type, home, seed_msa_row)`` for chain ``i`` in the blocked layout.

    Chains are blocked by type — random first, then natural_A, then natural_B — so the
    per-chain seed ``master_seed + i`` stays stable and the start type is a pure
    function of the index (and the config, which pins the chosen rows)."""
    nr, na = config["start_random"], config["start_natural_a"]
    if i < nr:
        return "random", None, None
    if i < nr + na:
        return "natural_A", "A", config["natural_a_rows"][i - nr]
    return "natural_B", "B", config["natural_b_rows"][i - nr - na]


def build_initial_state(config, i, model_A, model_B, msa_a, msa_b):
    """``(start_type, DesignState | None)`` for chain ``i`` (None ⇒ random start)."""
    start_type, home, row = chain_start_spec(config, i)
    if start_type == "random":
        return start_type, None
    msa = msa_a if home == "A" else msa_b
    state = initial_state_from_frame(msa[row], model_A, model_B, home=home)
    return start_type, state


def _load_seed_msas(config: dict):
    msa_a = np.load(config["seed_msa_a"]) if config.get("start_natural_a", 0) else None
    msa_b = np.load(config["seed_msa_b"]) if config.get("start_natural_b", 0) else None
    return msa_a, msa_b


# --------------------------------------------------------------------------- #
# Running the chains
# --------------------------------------------------------------------------- #

# Worker globals (populated by _worker_init in the parallel path).
_W: dict = {}


def _worker_init(config: dict) -> None:
    _W.update(
        config=config,
        model_A=load_model(config["model_a_path"], name=config["name_a"]),
        model_B=load_model(config["model_b_path"], name=config["name_b"]),
        sched=AnnealSchedule(**config["schedule"]),
        polish_pt=_POLISH_SCHEDULES[config["polish_schedule"]],
    )
    _W["msa_a"], _W["msa_b"] = _load_seed_msas(config)


def _worker_run(i: int) -> ChainResult:
    c = _W["config"]
    start_type, init = build_initial_state(c, i, _W["model_A"], _W["model_B"],
                                           _W["msa_a"], _W["msa_b"])
    return anneal_chain(_W["model_A"], _W["model_B"], c["w_a"], c["w_b"], _W["sched"],
                        seed=c["master_seed"] + i, chain_index=i, do_polish=c["do_polish"],
                        polish_pt_schedule=_W["polish_pt"], init_state=init, start_type=start_type)


def _run_chains(config: dict, jobs: int) -> list[ChainResult]:
    n = config["n_chains"]
    if jobs <= 1:
        model_A = load_model(config["model_a_path"], name=config["name_a"])
        model_B = load_model(config["model_b_path"], name=config["name_b"])
        msa_a, msa_b = _load_seed_msas(config)
        sched = AnnealSchedule(**config["schedule"])
        polish_pt = _POLISH_SCHEDULES[config["polish_schedule"]]
        results = []
        for i in range(n):
            start_type, init = build_initial_state(config, i, model_A, model_B, msa_a, msa_b)
            log.info("chain %d/%d (seed=%d, start=%s) ...", i + 1, n, config["master_seed"] + i,
                     start_type)
            results.append(anneal_chain(model_A, model_B, config["w_a"], config["w_b"], sched,
                                        seed=config["master_seed"] + i, chain_index=i,
                                        do_polish=config["do_polish"], polish_pt_schedule=polish_pt,
                                        init_state=init, start_type=start_type))
        return results
    from concurrent.futures import ProcessPoolExecutor
    log.info("running %d chains across %d worker processes ...", n, jobs)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(config,)) as ex:
        results = list(ex.map(_worker_run, range(n)))
    return sorted(results, key=lambda r: r.chain_index)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def _stack_trajectories(results: list[ChainResult]) -> dict[str, np.ndarray]:
    """All chains share the schedule, hence identical (steps, temperatures)."""
    ref = results[0]
    return {
        "steps": ref.steps,
        "temperatures": ref.temperatures,
        "E_tot": np.vstack([r.E_tot for r in results]),
        "E_A": np.vstack([r.E_A for r in results]),
        "E_B": np.vstack([r.E_B for r in results]),
        "n_residues": np.vstack([r.n_residues for r in results]),
        "chain_index": np.array([r.chain_index for r in results], dtype=np.int64),
        "seed": np.array([r.seed for r in results], dtype=np.int64),
        "start_type": np.array([r.start_type for r in results], dtype="<U16"),
        "final_E_A_mc": np.array([r.E_A_mc for r in results]),
        "final_E_B_mc": np.array([r.E_B_mc for r in results]),
        "final_E_tot_mc": np.array([r.E_tot_mc for r in results]),
        "final_E_A_polish": np.array([np.nan if r.E_A_polish is None else r.E_A_polish for r in results]),
        "final_E_B_polish": np.array([np.nan if r.E_B_polish is None else r.E_B_polish for r in results]),
        "final_E_tot_polish": np.array([np.nan if r.E_tot_polish is None else r.E_tot_polish for r in results]),
        "final_n_residues": np.array([r.final_n_residues for r in results], dtype=np.int64),
    }


def _write_fasta(results: list[ChainResult], path: Path) -> None:
    lines = []
    for r in results:
        pol = "" if r.E_tot_polish is None else f" E_tot_polish={r.E_tot_polish:.4f}"
        lines.append(f">design_chain{r.chain_index:04d} start={r.start_type} "
                     f"N={r.final_n_residues} E_tot_mc={r.E_tot_mc:.4f} "
                     f"E_A_mc={r.E_A_mc:.4f} E_B_mc={r.E_B_mc:.4f}{pol}")
        lines.append(r.final_sequence)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table(results: list[ChainResult], name_A: str, name_B: str, path: Path) -> pd.DataFrame:
    rows = [{
        "chain": r.chain_index, "seed": r.seed, "start_type": r.start_type,
        "N_final": r.final_n_residues,
        "E_A_mc": r.E_A_mc, "E_B_mc": r.E_B_mc, "E_tot_mc": r.E_tot_mc,
        "E_A_polish": r.E_A_polish, "E_B_polish": r.E_B_polish, "E_tot_polish": r.E_tot_polish,
        "polish_exact_A": r.polish_exact_A, "polish_exact_B": r.polish_exact_B,
        "accept_rate": r.accept_rate, "model_A": name_A, "model_B": name_B,
    } for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return df


def _write_alignments(results: list[ChainResult], model_A, model_B, out_dir: Path) -> None:
    """Two aligned FASTAs (one per model frame): each design threaded into that model's
    frame with gaps (the polish argmin alignment). Length L_A / L_B, so they are a valid
    MSA per model — drop straight into an alignment viewer (e.g. alnviz, ZAPPO coloring)."""
    for tag, model, attr in (("A", model_A, "aln_frame_A"), ("B", model_B, "aln_frame_B")):
        lines = []
        for r in results:
            e = getattr(r, f"E_{tag}_polish")
            e = getattr(r, f"E_{tag}_mc") if e is None else e
            lines.append(f">design_chain{r.chain_index:04d} model={model.name} frame={tag} "
                         f"start={r.start_type} N={r.final_n_residues} E={e:.4f}")
            lines.append(ints_to_seq(np.asarray(getattr(r, attr), dtype=np.int64)))
        (out_dir / f"design_aln_{tag}.fasta").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, *, config: dict, results, models, started, finished, out_dir) -> None:
    model_A, model_B = models
    sched = AnnealSchedule(**config["schedule"])
    best = min(results, key=lambda r: r.E_tot_mc)
    extra = {
        "weights": {"w_A": config["w_a"], "w_B": config["w_b"], "source": config.get("weights_source")},
        "models": {
            "A": {"name": model_A.name, "L": model_A.L, "sha256": model_A.sha256, "source": model_A.source},
            "B": {"name": model_B.name, "L": model_B.L, "sha256": model_B.sha256, "source": model_B.source},
        },
        "schedule": sched.as_dict(),
        "n_chains": config["n_chains"],
        "start_mix": {"random": config["start_random"], "natural_A": config["start_natural_a"],
                      "natural_B": config["start_natural_b"]},
        "chain_seeds": [r.seed for r in results],
        "polish": {"enabled": config["do_polish"], "schedule": config["polish_schedule"]},
        "summary": {
            "E_tot_mc_median": float(np.median([r.E_tot_mc for r in results])),
            "E_tot_mc_min": float(best.E_tot_mc),
            "best_chain": best.chain_index,
            "N_final_median": float(np.median([r.final_n_residues for r in results])),
            "N_final_min": int(min(r.final_n_residues for r in results)),
            "N_final_max": int(max(r.final_n_residues for r in results)),
            "accept_rate_median": float(np.median([r.accept_rate for r in results])),
        },
    }
    inputs = {"model_a": config["model_a_path"], "model_b": config["model_b_path"],
              "energy_weights": config.get("weights_source")}
    if config.get("start_natural_a"):
        inputs["seed_msa_a"] = config["seed_msa_a"]
    if config.get("start_natural_b"):
        inputs["seed_msa_b"] = config["seed_msa_b"]
    manifest = provenance.build_run_manifest(
        run_id=out_dir.name,
        command_line=provenance.current_command_line(),
        inputs=inputs,
        options={"master_seed": config["master_seed"], "do_polish": config["do_polish"],
                 "start_random": config["start_random"],
                 "start_natural_a": config["start_natural_a"],
                 "start_natural_b": config["start_natural_b"], **sched.as_dict()},
        seed=config["master_seed"],
        started_at=started, finished_at=finished,
        omp_threads_requested=provenance.omp_threads_requested(),
        extra=extra,
    )
    provenance.save_run_manifest(manifest, path)


def run_from_config(config: dict, out_dir: Path | str, jobs: int) -> pd.DataFrame:
    """Run every chain in ``config`` and write the run's outputs into ``out_dir``.

    The shared execution entry point: the CLI (``main``) and the Snakemake local
    wrapper (``scripts/wf/run_design_local.py``) both call this with a resolved config."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    results = _run_chains(config, jobs)
    finished = dt.datetime.now(dt.timezone.utc)

    model_A = load_model(config["model_a_path"], name=config["name_a"])
    model_B = load_model(config["model_b_path"], name=config["name_b"])
    np.savez_compressed(out_dir / "trajectories.npz", **_stack_trajectories(results))
    _write_fasta(results, out_dir / "designed_sequences.fasta")
    _write_alignments(results, model_A, model_B, out_dir)
    df = _write_table(results, model_A.name, model_B.name, out_dir / "designed.tsv")
    _write_manifest(out_dir / "design_manifest.json", config=config, results=results,
                    models=(model_A, model_B), started=started, finished=finished, out_dir=out_dir)
    provenance.write_command_sh(provenance.current_command_line(), out_dir / "command.sh")

    ecol = "E_tot_polish" if config["do_polish"] else "E_tot_mc"
    log.info("done: %d chains in %.1fs | median %s=%.2f, best=%.2f | N_final in [%d, %d]",
             len(results), (finished - started).total_seconds(), ecol,
             float(df[ecol].median()), float(df[ecol].min()),
             int(df["N_final"].min()), int(df["N_final"].max()))
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("models & weights")
    src.add_argument("--combine-run", default=DEFAULT_COMBINE_RUN,
                     help="combine run dir with models.json + data/energy_weights.json "
                          f"(default: {DEFAULT_COMBINE_RUN})")
    src.add_argument("--model-a", default=None, help="override model A path (model.npy)")
    src.add_argument("--model-b", default=None, help="override model B path")
    src.add_argument("--name-a", default=None, help="override model A name")
    src.add_argument("--name-b", default=None, help="override model B name")
    src.add_argument("--w-a", type=float, default=None, help="override weight w_A")
    src.add_argument("--w-b", type=float, default=None, help="override weight w_B")

    run = p.add_argument_group("run")
    run.add_argument("--start-random", type=int, default=48, help="chains seeded from a random sequence")
    run.add_argument("--start-natural-a", type=int, default=24,
                     help="chains seeded from a model-A (CM) natural")
    run.add_argument("--start-natural-b", type=int, default=24,
                     help="chains seeded from a model-B (PPIC) natural")
    run.add_argument("--steps", type=int, default=500_000, help="Metropolis steps per chain")
    run.add_argument("--seed", type=int, default=0, help="master seed (chain i uses seed+i)")
    run.add_argument("--jobs", type=int, default=1, help="worker processes (>=2 parallelizes chains)")

    sch = p.add_argument_group("schedule")
    sch.add_argument("--beta-start", type=float, default=1.0, help="start inverse temp (T=1/beta)")
    sch.add_argument("--beta-end", type=float, default=10.0, help="end inverse temp (default T:1->0.1)")
    sch.add_argument("--record-every", type=int, default=1000, help="trajectory sub-sampling stride")
    sch.add_argument("--min-length", type=int, default=70, help="deletion floor on N")
    sch.add_argument("--teleport-frac", type=float, default=0.3, help="non-local slide fraction")
    # Defaults are the insert-biased "colaware" recipe (docs/DESIGN_TWO_MODEL.md search study).
    sch.add_argument("--p-sub", type=float, default=0.50)
    sch.add_argument("--p-slide-a", type=float, default=0.10)
    sch.add_argument("--p-slide-b", type=float, default=0.10)
    sch.add_argument("--p-insert", type=float, default=0.25)
    sch.add_argument("--p-delete", type=float, default=0.05)
    sch.add_argument("--move-kind", choices=["metropolis", "heatbath", "colaware"], default="colaware",
                     help="substitute/insert proposal (default 'colaware'): 'metropolis' (uniform), "
                          "'heatbath' (Gibbs substitute + conditional-residue insert), or "
                          "'colaware' (heatbath + column-aware insert pairing)")

    out = p.add_argument_group("polish & output")
    out.add_argument("--polish", dest="polish", action="store_true", default=True,
                     help="final real potts_align on each design (default on)")
    out.add_argument("--no-polish", dest="polish", action="store_false")
    out.add_argument("--polish-schedule", choices=sorted(_POLISH_SCHEDULES), default="fast",
                     help="warm-started polish depth (default: fast — Mac-feasible in minutes)")
    out.add_argument("--out-dir", default=None,
                     help="output dir (default <combine_run>/design, stable/overwritten)")
    out.add_argument("--emit-config-only", action="store_true",
                     help="resolve inputs, write design_config.json for a cluster run, then exit")
    out.add_argument("--from-config", default=None,
                     help="run chains from an existing design_config.json (skip resolution); "
                          "used by the Snakemake local wrapper so worker processes re-import "
                          "this CLI as __main__ rather than the injected wrapper script")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    # Run from an already-resolved config (Snakemake local wrapper path): no re-resolution,
    # so the seeded natural selection is identical to what the config pinned.
    if args.from_config:
        cfg_path = Path(args.from_config)
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        out_dir = Path(args.out_dir) if args.out_dir else cfg_path.parent
        log.info("running from config %s -> %s (%d chains, %d jobs)",
                 cfg_path, out_dir, config["n_chains"], args.jobs)
        run_from_config(config, out_dir, args.jobs)
        return 0

    sched = AnnealSchedule(
        n_steps=args.steps, beta_start=args.beta_start, beta_end=args.beta_end,
        p_sub=args.p_sub, p_slide_A=args.p_slide_a, p_slide_B=args.p_slide_b,
        p_insert=args.p_insert, p_delete=args.p_delete,
        teleport_frac=args.teleport_frac, min_length=args.min_length,
        record_every=args.record_every, move_kind=args.move_kind,
    )
    sched.move_probs()  # validate probabilities early

    config = resolve_design_config(
        combine_run=args.combine_run, schedule=sched, master_seed=args.seed,
        start_random=args.start_random, start_natural_a=args.start_natural_a,
        start_natural_b=args.start_natural_b, do_polish=args.polish,
        polish_schedule=args.polish_schedule, model_a=args.model_a, model_b=args.model_b,
        name_a=args.name_a, name_b=args.name_b, w_a=args.w_a, w_b=args.w_b,
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.combine_run) / "design"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output dir: %s", out_dir)
    log.info("models: %s vs %s | weights w_A=%.4f w_B=%.4f | starts: %d random / %d CM / %d PPIC",
             config["name_a"], config["name_b"], config["w_a"], config["w_b"],
             config["start_random"], config["start_natural_a"], config["start_natural_b"])

    (out_dir / "design_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if args.emit_config_only:
        provenance.write_command_sh(provenance.current_command_line(), out_dir / "command.sh")
        log.info("emitted design_config.json (no chains run) -> %s", out_dir)
        return 0

    run_from_config(config, out_dir, args.jobs)
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
