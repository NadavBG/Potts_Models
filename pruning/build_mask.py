import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pysca.scaTools as sca
import scipy.io as sio

from SBM import provenance
from SBM.utils.utils import CalcWeights

DEFAULT_OUTFILE_PARENT = Path(__file__).resolve().parent / "masks"

BACKGROUND_FREQS_GAPLESS = np.array(
    [
        0.073,
        0.025,
        0.050,
        0.061,
        0.042,
        0.072,
        0.023,
        0.053,
        0.064,
        0.089,
        0.023,
        0.043,
        0.052,
        0.040,
        0.052,
        0.073,
        0.056,
        0.063,
        0.013,
        0.033,
    ]
)


def calcSCAMat(
    alg, seqw=1, lbda=0, freq0=np.ones(21) / 21, norm=None, include_gaps=True
):
    """
    This is basically a rewrite of the scaMat function that (1) includes
    gap characters in the LxLxQxQ matrix, and (2) returns the full LxLxQxQ
    matrix instead of the compressed LxL SCA matrix.
    """
    N_seq, N_pos = alg.shape
    N_aa = freq0.shape[0]
    if include_gaps and 0 in alg:
        alg = alg.copy() + 1
    assert (
        np.max(alg) <= N_aa
    ), "Background frequency distribution has size mismatch to alignment"

    if isinstance(seqw, int) and seqw == 1:
        seqw = np.ones((1, N_seq))

    freq1, freq2, _ = sca.freq(
        alg,
        Naa=N_aa,
        seqw=seqw,
        lbda=lbda,
        freq0=np.ones(N_aa) / (N_aa + int(not include_gaps)),
    )
    Wpos, _, _ = sca.posWeights(alg, seqw, lbda, N_aa, freq0)
    tildeC = np.outer(Wpos, Wpos) * (freq2 - np.outer(freq1, freq1))
    tildeC = tildeC.reshape(N_pos, N_aa, N_pos, N_aa).transpose(0, 2, 1, 3)
    if norm is None:
        return tildeC
    # Optionally, get the matrix norm, for example if you want to compare with
    # the output of pySCA
    Cnorm = np.zeros((N_pos, N_pos))
    for i in range(N_pos):
        for j in range(i, N_pos):
            st = int(include_gaps)
            u, s, vt = np.linalg.svd(
                tildeC[
                    i, j, st:, st:
                ]  # ignore gaps while norming for consistency with SCA
            )
            if norm == "spec":
                Cnorm[i, j] = s[0]
            else:
                Cnorm[i, j] = np.sqrt(sum(s**2))  # frob norm
    Cnorm += np.triu(Cnorm, 1).T
    return Cnorm


def write_file(outfile, prune_mat, verbose=False):
    if outfile[-4:] == ".npy":
        np.save(outfile, prune_mat)
    elif outfile[:-4] == ".mat":
        prune_mat = prune_mat.transpose(
            2, 3, 0, 1
        )  # swap indices for consistency with MATLAB code
        sio.savemat(outfile, {"pruneJ": prune_mat})
    else:
        raise Exception("Filetype not supported")
    return


def write_mask_manifest(
    outfile, *, alg_file, strategy, pct, theta, lbda, Dia_prior, started_at
):
    """Write `<outfile>.manifest.json` with mask provenance.

    Captures the input MSA (path + sha256 + shape), the strategy / theta /
    lbda / percent that drove this mask, the Dia prior (only meaningful
    for ``strategy="Dia"`` but recorded uniformly), git state, package
    versions, and timestamps. The output mask file itself is hashed too,
    so a manifest commits to the bytes it sits next to.
    """
    finished_at = _dt.datetime.now(_dt.timezone.utc)
    manifest = provenance.build_run_manifest(
        run_id=provenance.make_run_id(started_at),
        command_line=[sys.executable, *sys.argv],
        inputs={"msa": alg_file},
        options={
            "strategy": strategy,
            "percent": pct,
            "theta": theta,
            "lbda": lbda,
            "Dia_prior": Dia_prior,
        },
        seed=None,  # mask generation is deterministic from inputs
        started_at=started_at,
        finished_at=finished_at,
        output_path=outfile,
        omp_threads_requested=provenance.omp_threads_requested(),
    )
    provenance.save_run_manifest(manifest, Path(outfile).with_suffix(".manifest.json"))


def partition_params(
    prune_vals, pcts, partial_outfile, outfile_path, *, manifest_meta=None
):
    """Generate one binary mask per requested percent and write it.

    ``manifest_meta`` is a dict with keys ``alg_file``, ``strategy``,
    ``theta``, ``lbda``, ``started_at``; if given, a manifest sidecar is
    written next to each output file.
    """
    N_pos = prune_vals.shape[0]
    triu = np.triu_indices(N_pos, k=1)  # ignore diagonal
    prune_vals_triu = np.zeros(prune_vals.shape)
    prune_vals_triu[triu] = prune_vals[triu]
    idx = np.argsort(abs(prune_vals_triu).flatten())[::-1]  # descending order
    for pct in pcts:
        tokeep_idx = int((N_pos**2 * 21**2 / 2) * (1 - pct / 100))
        bin_prune_mat = np.zeros(prune_vals.size, dtype="int")
        bin_prune_mat[idx[:tokeep_idx]] = 1
        bin_prune_mat = bin_prune_mat.reshape(prune_vals.shape)
        for ii in range(N_pos):
            for jj in range(ii + 1, N_pos):
                bin_prune_mat[jj, ii] = bin_prune_mat[ii, jj].T
        outfile = "%s/%.2f_%s" % (outfile_path, pct, partial_outfile)
        write_file(outfile, bin_prune_mat)
        if manifest_meta is not None:
            write_mask_manifest(outfile, pct=pct, **manifest_meta)


def partition_fields_params(
    prune_vals, pcts, partial_outfile, outfile_path, *, manifest_meta=None
):
    """Generate one binary (L, q) field mask per requested percent and write it.

    Mirrors :func:`partition_params` but for fields: there is no symmetry
    constraint and no diagonal to ignore, so all (i, a) entries are
    ranked together. Output is always ``.npy`` (no ``.mat`` route is
    needed for fields).
    """
    idx = np.argsort(np.abs(prune_vals).flatten())[::-1]  # descending order
    for pct in pcts:
        tokeep_idx = int(prune_vals.size * (1 - pct / 100))
        bin_mask = np.zeros(prune_vals.size, dtype="int")
        bin_mask[idx[:tokeep_idx]] = 1
        bin_mask = bin_mask.reshape(prune_vals.shape)
        outfile = "%s/%.2f_%s" % (outfile_path, pct, partial_outfile)
        np.save(outfile, bin_mask)
        if manifest_meta is not None:
            write_mask_manifest(outfile, pct=pct, **manifest_meta)


def main(
    alg_file,
    theta=0.7,
    lbda=0.03,
    strategies=["fij", "cij", "sca"],
    output_type=".npy",
    output_label="CM",
    outfile_path=".",  # parent dir; a per-run subdir is created under it
    pct_J=[95],
    pct_h=[95],
    Dia_prior="gap-corrected",
):
    started_at = _dt.datetime.now(_dt.timezone.utc)
    parent_dir = Path(outfile_path)
    parent_dir.mkdir(parents=True, exist_ok=True)
    run_dir = parent_dir / provenance.make_run_id(
        when=started_at, label=output_label, parent_dir=parent_dir
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run dir: {run_dir.resolve()}")
    outfile_path = str(run_dir)
    # read in the file
    alg = None
    if alg_file[-4:] == ".npy":
        alg = np.load(alg_file)
    elif alg_file[-4:] == ".mat":
        alg = sio.loadmat(alg_file) - 1
    else:
        alg = sca.readAlg(alg_file)[1]
        # get rid of any non-canonical AAs
        alg = sca.lett2num(alg, code="-ACDEFGHIKLMNPQRSTVWY")
        alg = alg[~(alg == 0).any(axis=1), :] - 1

    def _meta(strategy):
        return {
            "alg_file": alg_file,
            "strategy": strategy,
            "theta": theta,
            "lbda": lbda,
            "Dia_prior": Dia_prior,
            "started_at": started_at,
        }

    # process inputs
    if not isinstance(pct_J, list):
        pct_J = [pct_J]
    if not isinstance(pct_h, list):
        pct_h = [pct_h]
    # 100% short-circuit: write a single all-zero J mask without computing
    # weights. Only meaningful for couplings — for fields, partition_fields_params
    # naturally produces an all-zero mask at pct=100, so no special case needed.
    if 100 in pct_J:
        prune_mat = np.zeros((alg.shape[1], alg.shape[1], 21, 21), dtype="int")
        outfile = "%s/%.2fp_%s_%s_SeqW_%.1f%s" % (
            outfile_path,
            100,
            "Fij",
            output_label,
            theta,
            output_type,
        )
        write_file(outfile, prune_mat)
        write_mask_manifest(outfile, pct=100.0, **_meta("Fij"))
        pct_J.remove(100)

    if len(pct_J) == 0 and len(pct_h) == 0:
        return

        # get sequence weights; necessary for all pruning types
    seqw, neff = CalcWeights(alg, 1 - theta, False)
    seqwn = seqw / neff
    # calculate background frequencies with gaps for correlation-based pruning
    bg_gaps = (1 - lbda) * np.sum(seqwn * (alg == 0).sum(axis=1)) / alg.shape[
        1
    ] + lbda * (1 / 21)
    freqs0 = np.hstack([[bg_gaps], (1 - bg_gaps) * BACKGROUND_FREQS_GAPLESS])

    strategies = set(strategies)
    outfile_base = "%s_SeqW_%.1f%s" % (output_label, theta, output_type)
    # Field masks are always saved as .npy; the .mat route is J-only and
    # exists for legacy MATLAB consumers that don't read field masks.
    outfile_base_fields = "%s_SeqW_%.1f.npy" % (output_label, theta)
    if "cij" in strategies:
        f1, f2, _ = sca.freq(
            alg + 1, seqw=seqw, lbda=lbda, freq0=freqs0, Naa=freqs0.size
        )
        prune_vals = f2 - np.outer(f1, f1)
        prune_vals = prune_vals.reshape(alg.shape[1], 21, alg.shape[1], 21).transpose(
            0, 2, 1, 3
        )
        partition_params(
            prune_vals,
            pct_J,
            "%s_%s" % ("Cij", outfile_base),
            outfile_path,
            manifest_meta=_meta("Cij"),
        )
    if "fij" in strategies:
        _, prune_vals, _ = sca.freq(alg + 1, seqw=seqw, Naa=21, lbda=0)
        prune_vals = prune_vals.reshape(alg.shape[1], 21, alg.shape[1], 21).transpose(
            0, 2, 1, 3
        )
        partition_params(
            prune_vals,
            pct_J,
            "%s_%s" % ("Fij", outfile_base),
            outfile_path,
            manifest_meta=_meta("Fij"),
        )
    if "sca" in strategies or "cij" in strategies:
        prune_vals = calcSCAMat(
            alg, seqw=seqw, lbda=lbda, freq0=freqs0, norm=None, include_gaps=True
        )
        partition_params(
            prune_vals,
            pct_J,
            "%s_%s" % ("SCA", outfile_base),
            outfile_path,
            manifest_meta=_meta("SCA"),
        )
    if "fia" in strategies:
        # First-order frequency-based fields mask (parallel to "fij" for J).
        f1, _, _ = sca.freq(alg + 1, seqw=seqw, Naa=21, lbda=0)
        prune_vals = f1.reshape(alg.shape[1], 21)
        partition_fields_params(
            prune_vals,
            pct_h,
            "%s_%s" % ("Fia", outfile_base_fields),
            outfile_path,
            manifest_meta=_meta("Fia"),
        )
    if "dia" in strategies:
        # Per-site KL-divergence fields mask (parallel to "sca" for J).
        # By default uses the gap-corrected background `freqs0` so the
        # divergence reflects deviation from the natural amino-acid
        # distribution at the alignment's gap rate; --Dia-prior=uniform
        # selects np.ones(21)/21 instead.
        if Dia_prior == "uniform":
            Dia_freq0 = np.ones(21) / 21
        else:
            Dia_freq0 = freqs0
        _, Dia, _ = sca.posWeights(alg + 1, seqw, lbda, 21, Dia_freq0)
        prune_vals = Dia.reshape(alg.shape[1], 21)
        partition_fields_params(
            prune_vals,
            pct_h,
            "%s_%s" % ("Dia", outfile_base_fields),
            outfile_path,
            manifest_meta=_meta("Dia"),
        )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pruning masks for input to sBM."
    )
    parser.add_argument(
        "-a", "--alg", type=str, required=True, help="Path to alignment file."
    )
    parser.add_argument(
        "-t",
        "--theta",
        type=float,
        default=0.7,
        help="similarity threshold to reweight sequences",
    )
    parser.add_argument(
        "-l",
        "--lbda",
        type=float,
        default=0.03,
        help="pseudocount to add for SCA/correlation calculations",
    )
    parser.add_argument(
        "-s",
        "--strategies",
        nargs="+",
        type=str,
        default=["sca"],
        help=(
            "types of pruning files to generate. Couplings (J): any of "
            "'fij', 'cij', 'sca'. Fields (h): any of 'fia', 'dia'. "
            "Strategies are independent — mix and match freely."
        ),
    )
    parser.add_argument(
        "--Dia-prior",
        type=str,
        default="gap-corrected",
        choices=["gap-corrected", "uniform"],
        help=(
            "Background distribution for the Dia (KL-divergence) fields "
            "strategy. 'gap-corrected' (default) uses the alignment's "
            "estimated gap rate plus standard 20-AA frequencies; "
            "'uniform' uses np.ones(21)/21."
        ),
    )
    parser.add_argument(
        "-x",
        "--ext",
        type=str,
        default=".npy",
        help="file format of output files. File types .npy and .mat are supported.",
    )
    parser.add_argument(
        "-b",
        "--label",
        type=str,
        default="CM",
        help="Label for output file (e.g. protein name)",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=str(DEFAULT_OUTFILE_PARENT),
        help=(
            "parent directory under which a per-run subdir is created. "
            "Default: pruning/masks/ (resolved relative to this script)."
        ),
    )
    parser.add_argument(
        "--percent-J",
        dest="percent_J",
        nargs="+",
        type=float,
        default=[95.0],
        help=(
            "set of percents of couplings (J) to remove. Used by the "
            "'fij', 'cij', and 'sca' strategies. Multiple values produce "
            "one mask per percent."
        ),
    )
    parser.add_argument(
        "--percent-h",
        dest="percent_h",
        nargs="+",
        type=float,
        default=[95.0],
        help=(
            "set of percents of fields (h) to remove. Used by the "
            "'fia' and 'dia' strategies. Multiple values produce one "
            "mask per percent."
        ),
    )

    args = parser.parse_args()
    main(
        args.alg,
        theta=args.theta,
        lbda=args.lbda,
        strategies=[x.lower() for x in args.strategies],
        output_type=args.ext,
        output_label=args.label,
        outfile_path=args.path,
        pct_J=args.percent_J,
        pct_h=args.percent_h,
        Dia_prior=args.Dia_prior,
    )
