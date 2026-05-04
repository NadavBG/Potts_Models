import numpy as np
import scipy.io as sio
import pysca.scaTools as sca
import argparse
from SBM.utils.utils import CalcWeights

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


def partition_params(prune_vals, pcts, partial_outfile, outfile_path):
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


def main(
    alg_file,
    theta=0.7,
    lbda=0.03,
    strategies=["fij", "cij", "sca"],
    output_type=".npy",
    output_label="CM",
    outfile_path=".",  # folder to save files to
    pct=[95],
):
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

    # process inputs
    if not isinstance(pct, list):
        pct = [pct]
    if 100 in pct:
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
        pct.remove(100)

    if len(pct) == 0:
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
    if "cij" in strategies:
        f1, f2, _ = sca.freq(
            alg + 1, seqw=seqw, lbda=lbda, freq0=freqs0, Naa=freqs0.size
        )
        prune_vals = f2 - np.outer(f1, f1)
        prune_vals = prune_vals.reshape(alg.shape[1], 21, alg.shape[1], 21).transpose(
            0, 2, 1, 3
        )
        partition_params(prune_vals, pct, "%s_%s" % ("Cij", outfile_base), outfile_path)
    if "fij" in strategies:
        _, prune_vals, _ = sca.freq(alg + 1, seqw=seqw, Naa=21, lbda=0)
        prune_vals = prune_vals.reshape(alg.shape[1], 21, alg.shape[1], 21).transpose(
            0, 2, 1, 3
        )
        partition_params(prune_vals, pct, "%s_%s" % ("Fij", outfile_base), outfile_path)
    if "sca" in strategies or "cij" in strategies:
        prune_vals = calcSCAMat(
            alg, seqw=seqw, lbda=lbda, freq0=freqs0, norm=None, include_gaps=True
        )
        partition_params(prune_vals, pct, "%s_%s" % ("SCA", outfile_base), outfile_path)
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
        help="types of pruning files to generate. any combination of 'fij', 'cij', or 'sca'",
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
        "-p", "--path", type=str, default=".", help="path to output directory"
    )
    parser.add_argument(
        "-c",
        "--percent",
        nargs="+",
        type=float,
        default=95.0,
        help="set of percents of parameters to remove",
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
        pct=args.percent,
    )
