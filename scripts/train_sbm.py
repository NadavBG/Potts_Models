"""SBM training driver.

Reads a numerical MSA (`.npy`), runs ``SBM.SBM_GD.SBM_proteins.SBM`` once
or more (averaged across replicates), and writes a self-describing run
directory:

    <results_path>/<fam>/<run_id>/
        model.npy        # the existing pickled-dict model artifact
        manifest.json    # full provenance: git, seeds, options, hashes,
                         # package versions, timestamps
        command.sh       # one-liner that re-invokes this run

Each (rep × N_chains) combination produces one such run directory.
"""

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path

import numpy as np  # type: ignore

import SBM
import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
from SBM import provenance

log = logging.getLogger(__name__)

ROOT = Path(SBM.__file__).resolve().parents[2]
data_dir = ROOT / "data"
results_dir = ROOT / "results"


def _hash_input_array(path: Path | str | None) -> dict:
    """For an MSA / train-indices file, record path + sha256 + shape.

    Shape/dtype is best-effort: if the file isn't a plain ``.npy`` (e.g.
    the user pointed at a pickled or unsupported format), we still record
    path + sha256 and log a warning. The hash is the load-bearing field.
    """
    if path is None:
        return {"path": None, "sha256": None}
    p = Path(path)
    entry = {"path": str(p), "sha256": provenance.file_sha256(p)}
    try:
        arr = np.load(p, allow_pickle=False)
    except (ValueError, OSError) as exc:
        log.warning("could not read shape/dtype for %s: %s", p, exc)
        return entry
    entry["shape"] = list(arr.shape)
    entry["dtype"] = str(arr.dtype)
    return entry


def _spawn_seeds(seed: int | None, n: int) -> list[int]:
    """Derive ``n`` per-replicate seeds from a master seed using
    ``np.random.SeedSequence``. If ``seed`` is None, return ``[None]*n``
    so SBM auto-generates each one (existing behavior)."""
    if seed is None:
        return [None] * n
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def run_SBM(
    Input_MSA,
    fam,
    Model,
    train_file,
    N_iter,
    m,
    N_chains_list,
    Nb_rep,
    Nb_av,
    k_MCMC,
    TestTrain,
    ParamInit,
    lambdJ,
    lambdh,
    theta,
    ignore_gaps,
    prune_file,
    results_path,
    seed,
    label,
    optimizer,
    record_every,
):
    if results_path is None:
        results_path = results_dir
    else:
        results_path = Path(results_path)
    fam = str(fam)

    msa_entry = _hash_input_array(Input_MSA)
    train_entry = _hash_input_array(train_file)
    prune_entry = _hash_input_array(prune_file)

    for rep in range(Nb_rep):
        for N_chains in N_chains_list:
            replicate_seeds = _spawn_seeds(seed, Nb_av)
            run_started = _dt.datetime.now(_dt.timezone.utc)

            W_rep = np.array([[]])
            Jnorm_rep = np.array([[]])
            Seeds_rep = np.zeros(Nb_av, dtype=np.int64)
            Extime_rep = np.zeros(Nb_av)
            for n_av in range(Nb_av):
                print("AVG: ", n_av)
                align = np.load(str(Input_MSA))
                if train_file is not None:
                    ind_train = np.load(train_file)
                    print(
                        "Database size: ",
                        align.shape,
                        " & Training set size: ",
                        len(ind_train),
                    )
                else:
                    ind_train = None
                    print("Database size: ", align.shape)

                options = dict(
                    [
                        ("Model", Model),
                        ("Optimizer", optimizer),
                        ("N_iter", N_iter),
                        ("N_chains", N_chains),
                        ("m", m),
                        ("skip_log", 1),
                        ("theta", theta),
                        ("ignore_gaps_weighting", ignore_gaps),
                        ("k_MCMC", k_MCMC),
                        ("Record_every", record_every),
                        ("lambda_h", lambdh),
                        ("lambda_J", lambdJ),
                        ("Pruning", prune_file is not None),
                        ("Pruning Mask Couplings", prune_file),
                        ("Param_init", ParamInit),
                        ("Test/Train", TestTrain == 1),
                        ("Train sequences", ind_train),
                        ("Weights", None),
                        ("SGD", None),
                        ("Seed", replicate_seeds[n_av]),
                        ("Zero Fields", False),
                        ("Store Parameters", None),
                    ]
                )

                output = sbm.SBM(align, options)

                J_out, h_out = ut.Zero_Sum_Gauge(output["J"], output["h"])
                W_out = ut.Wj(J_out, h_out)
                W_rep = np.concatenate(
                    (W_rep, np.expand_dims(W_out, axis=0)), axis=int((n_av == 0))
                )

                Jnorm_rep = np.concatenate(
                    (Jnorm_rep, np.expand_dims(output["J_norm"], axis=0)),
                    axis=int((n_av == 0)),
                )

                Seeds_rep[n_av] = output["options"]["Seed"]
                Extime_rep[n_av] = output["Execution time"]

            run_finished = _dt.datetime.now(_dt.timezone.utc)
            W_av = np.mean(W_rep, axis=0)
            J_av, h_av = ut.Jw(W_av, output["options"]["q"])
            output_av = {
                "J": J_av,
                "h": h_av,
                "W_all": W_rep,
                "Seeds": Seeds_rep,
                "Execution times": Extime_rep,
                "J_norm": Jnorm_rep,
                # All replicates share N_iter and Record_every, so
                # J_norm_iters is identical across them; store one copy.
                "J_norm_iters": list(output.get("J_norm_iters", [])),
                "align": output["align"],
                "Test": output["Test"],
                "Train": output["Train"],
            }

            # ── Backwards-compatible options0 / options1 split ──────────
            output_av["options0"] = {
                "Model": output["options"]["Model"],
                "Optimizer": output["options"]["Optimizer"],
                "N_iter": output["options"]["N_iter"],
                "N_chains": output["options"]["N_chains"],
                "m": output["options"]["m"],
                "theta": output["options"]["theta"],
                "k_MCMC": output["options"]["k_MCMC"],
                "Record_every": output["options"]["Record_every"],
                "lambda_h": output["options"]["lambda_h"],
                "lambda_J": output["options"]["lambda_J"],
                "Param_init": output["options"]["Param_init"],
            }
            output_av["options1"] = {
                "skip_log": output["options"]["skip_log"],
                "Pruning": output["options"]["Pruning"],
                "Pruning Mask Couplings": output["options"]["Pruning Mask Couplings"],
                "Test/Train": output["options"]["Test/Train"],
                "Train sequences": output["options"]["Train sequences"],
                "Weights": output["options"]["Weights"],
                "SGD": output["options"]["SGD"],
                "Seed": output["options"]["Seed"],
                "Zero Fields": output["options"]["Zero Fields"],
                "Store Parameters": output["options"]["Store Parameters"],
                "Learning_rate": output["options"]["Learning_rate"],
                "Pruning_perc": output["options"]["Pruning_perc"],
                "Shuffle Columns": output["options"]["Shuffle Columns"],
                "q": output["options"]["q"],
                "L": output["options"]["L"],
            }

            # ── Per-run directory + provenance ──────────────────────────
            # Run-id format is <YYYY-MM-DD>_<label>_<idx>; label defaults to
            # the family name. The legacy timestamp+random format is still
            # available via make_run_id(label=None) for non-CLI callers.
            fam_dir = results_path / fam
            fam_dir.mkdir(parents=True, exist_ok=True)
            run_id = provenance.make_run_id(
                run_started, label=label or fam, parent_dir=fam_dir
            )
            run_dir = fam_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = run_dir / "model.npy"
            np.save(model_path, output_av)

            # Manifest captures the FULL options dict (incl. the 6 keys
            # dropped from options0/options1) plus per-replicate seeds and
            # exec times, plus input file hashes and library versions.
            full_options = dict(output["options"])
            # Restore the user-supplied mask path (Init_Pruning materializes
            # the mask in-place, then stashes the source under a new key).
            mask_source = full_options.pop("Pruning Mask Couplings Source", None)
            manifest = provenance.build_run_manifest(
                run_id=run_id,
                command_line=sys.argv,
                inputs={
                    "msa": Input_MSA,
                    "train_indices": train_file,
                    "pruning_mask": mask_source,
                },
                options=full_options,
                seed=seed,
                started_at=run_started,
                finished_at=run_finished,
                output_path=model_path,
                omp_threads_requested=provenance.omp_threads_requested(),
                extra={
                    "rep_index": rep,
                    "Nb_rep": Nb_rep,
                    "Nb_av": Nb_av,
                    "replicate_seeds_actual": [int(s) for s in Seeds_rep],
                    "replicate_seeds_planned": replicate_seeds,
                    "replicate_exec_seconds": Extime_rep.tolist(),
                    # Echo the input-array shapes/hashes (we rehash so the
                    # entry has shape/dtype too, on top of the path-only
                    # form provenance.build_run_manifest writes by default).
                    "inputs_detail": {
                        "msa": msa_entry,
                        "train_indices": train_entry,
                        "pruning_mask": prune_entry,
                    },
                },
            )
            provenance.save_run_manifest(manifest, run_dir / "manifest.json")
            provenance.write_command_sh(
                [sys.executable, *sys.argv],
                run_dir / "command.sh",
                cwd=Path.cwd(),
            )
            print(f"Run written: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SBM parameters.")
    parser.add_argument("fam", help="Protein family name in a numpy format")
    parser.add_argument(
        "--train_file", type=str, default=None, help="Ind_train filename"
    )
    parser.add_argument(
        "--TestTrain",
        type=int,
        default=0,
        help="1 to hold out 20%% of the MSA as a test set, 0 to train on all rows",
    )
    parser.add_argument("--rep", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--N_av", type=int, default=1, help="Number of averaged models")
    parser.add_argument(
        "--mod",
        type=lambda s: s.upper(),
        default="SBM",
        choices=["BM", "SBM"],
        help="Model regime (case-insensitive). Selects a label, not algorithm — "
        "BM and SBM both run L-BFGS; the mode-specific knobs (--m, --lambdJ, "
        "--lambdh, --N_chains) must still be set explicitly when calling this "
        "script directly. run_sbm.sh applies the per-mode defaults from "
        "Summary Note 3.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        choices=["LBFGS", "GD"],
        help=(
            "Optimization algorithm. LBFGS is the default for both BM and SBM "
            "(they differ in m, lambda_J/h, and N_chains, not in the algorithm). "
            "GD selects vanilla gradient descent (uses --alpha / --learning_rate "
            "instead of --m); rarely needed."
        ),
    )
    parser.add_argument("--N_iter", type=int, default=400, help="Number of iterations")
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="L-BFGS memory rank. Recommended: BM=20, SBM=1 (Summary Note 3). "
        "Default 1 matches the SBM regime.",
    )
    parser.add_argument(
        "--N_chains",
        type=int,
        nargs="+",
        required=True,
        help="MCMC chains per gradient step. Required. One value, or several "
        "to sweep (each spawns its own run dir). Recommended: BM=100, SBM=50.",
    )
    parser.add_argument(
        "--ParamInit", type=str, default="Zero", help="Init of fields and couplings"
    )
    parser.add_argument(
        "--k_MCMC", type=int, default=100000, help="Number of MCMC steps"
    )
    parser.add_argument(
        "--record_every",
        type=int,
        default=5,
        help=(
            "Record J_norm every N L-BFGS iterations (default: 5). A final "
            "recording at iteration N_iter is added unconditionally so the "
            "trajectory ends at training completion."
        ),
    )
    parser.add_argument(
        "--lambdJ",
        type=float,
        default=0,
        help="L2 regularization on couplings. Recommended: BM=0.01, SBM=0. "
        "Default 0 matches the SBM regime.",
    )
    parser.add_argument(
        "--lambdh",
        type=float,
        default=0,
        help="L2 regularization on fields. Recommended: BM=0.01, SBM=0. "
        "Default 0 matches the SBM regime.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.3,
        help="threshold to compute the effective number of sequences",
    )
    parser.add_argument(
        "--ignore_gaps",
        help="ignore gaps when calculating similarity for sequence weights",
        action="store_true",
    )
    parser.add_argument(
        "--prune", type=str, default=None, help="prune parameters based on input mask"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="path to results directory. Default is <SBM Repo>/results/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master RNG seed; per-replicate seeds are spawned via SeedSequence",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="label embedded in the run dir name (default: family name)",
    )
    parser.add_argument("Input_MSA")

    args = parser.parse_args()
    run_SBM(
        args.Input_MSA,
        args.fam,
        args.mod,
        args.train_file,
        args.N_iter,
        args.m,
        args.N_chains,
        args.rep,
        args.N_av,
        args.k_MCMC,
        args.TestTrain,
        args.ParamInit,
        args.lambdJ,
        args.lambdh,
        args.theta,
        args.ignore_gaps,
        args.prune,
        args.results_path,
        args.seed,
        args.label,
        args.optimizer,
        args.record_every,
    )
