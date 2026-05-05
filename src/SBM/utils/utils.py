#! /usr/bin/env python3
"""
@author: Marion CHAUVEAU

:On:  October 2022
"""

####################### MODULES #######################
import itertools as it
import SBM.MonteCarlo.MCMC_Potts.MonteCarlo_Potts as mc  # type: ignore
import SBM.MonteCarlo.MCMC_PottsProf.MonteCarlo_PottsProf as mcp  # type: ignore
import numpy as np  # type: ignore
from Bio import SeqIO  # type: ignore
from tqdm import tqdm  # type: ignore
from scipy.spatial.distance import squareform, pdist  # type: ignore
import csv as csv

##########################################################
####################### LOAD FILES #######################


def csv_to_fasta(csv_path, fasta_path):
    """
    Function to convert a CSV file to a FASTA file.

    Args:
    - csv_path (str): The path to the input CSV file.
    - fasta_path (str): The path to the output FASTA file.

    Returns: None
    """
    list_seq = []
    list_name = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            list_seq.append(row["sequence"])
            list_name.append(row["sequence_id"])
    ofile = open(fasta_path, "w")
    for i in range(len(list_seq)):
        ofile.write(">" + list_name[i] + "\n" + list_seq[i] + "\n")
    ofile.close()


def load_fasta(file):
    """
    Load a FASTA file and convert sequences into a numerical numpy array,
    much faster than the original version.
    """
    # Define amino acid mapping
    code = "-ACDEFGHIKLMNPQRSTVWY"
    AA_to_num = {aa: i for i, aa in enumerate(code)}

    # Unknown or invalid characters → -1
    invalid_chars = set("BJOUXZabcdefghijklmnopqrstuvwxyz")
    for ch in invalid_chars:
        AA_to_num[ch] = -1

    # Parse once
    records = list(SeqIO.parse(file, "fasta"))
    n_seq = len(records)
    if n_seq == 0:
        raise ValueError("No sequences found in FASTA file.")

    seq_len = len(records[0].seq)
    print(f"Nb of sequences: {n_seq}, sequence length: {seq_len}")

    # Preallocate numpy array (int8 is enough for 0–20 + -1)
    MSA = np.empty((n_seq, seq_len), dtype=np.int8)

    # Fill array efficiently
    for i, record in enumerate(records):
        seq = str(record.seq)
        MSA[i] = [AA_to_num.get(ch, -1) for ch in seq]

    # Remove erroneous sequences (containing -1)
    valid_mask = np.all(MSA != -1, axis=1)
    MSA = MSA[valid_mask]

    print(f"Final shape: {MSA.shape}")
    return MSA


def save_fasta_from_array(Model_file, fasta_file, Nb_seq=100):
    output_mod = np.load(Model_file, allow_pickle=True)[()]
    sequences = Create_modAlign(
        output_mod, Nb_seq, output_mod["options0"]["k_MCMC"], temperature=1
    )

    CODE = "-ACDEFGHIKLMNPQRSTVWY"
    M = sequences.shape[0]

    with open(fasta_file, "w") as f:
        for i in range(M):
            f.write(
                f">SBM N_chains={output_mod['options0']['N_chains']}|sequence {i+1}\n"
            )
            sequence = "".join([CODE[int(j)] for j in sequences[i]])
            f.write(sequence + "\n")


##########################################################

####################### CREATE ARTICIAL ALIGNEMENT #######################


def Create_modAlign(output, N, delta_t=None, ITER="", temperature=1, seed=None):
    """
        Function to create a alignment based on the provided parameters
        using the C_MonteCarlo module implemented in cython

        Args:
        - output (dict): A dictionary containing 'h' and 'J' values.
        - N (int): The value of N.
        - delta_t (int): Number of MCMC steps
        - ITER (str): Optional parameter (if several 'h' and 'J' values are stored in the output dictionary
        we can choose 'h' and 'J' values at a specific iteration)
    - temperature (float): Optional parameter to specify the temperature at which sampling should proceed
        - seed (int): Master RNG seed for the C++ MCMC kernel. If None, a fresh
        seed is drawn from the (already-seeded) global numpy RNG so callers
        that previously relied on np.random.seed continue to be reproducible.

        Returns:
        - numpy.array: A 2D numpy array with the created alignment.
    """
    if delta_t is None:
        delta_t = output["options0"]["k_MCMC"]

    if output["h" + str(ITER)] is not None:
        h = np.copy(output["h" + str(ITER)]) / temperature
        L, q = h.shape
    else:
        h = None

    if output["J" + str(ITER)] is not None:
        J = np.copy(output["J" + str(ITER)]) / temperature
        L, q = J.shape[0], J.shape[2]
    else:
        J = None
    w = np.array(Wj(J, h))
    states = np.random.randint(q, size=(N, L)).astype("int32")
    if seed is None:
        # Drawn from the global RNG so reproducibility tracks np.random.seed.
        seed = int(np.random.randint(0, 2**63 - 1, dtype=np.int64))
    mcmc_seed = int(seed) & ((1 << 64) - 1)  # C++ ABI takes uint64
    if J is None:
        mcp.MC(w, states, int(delta_t), int(q), mcmc_seed)
    else:
        mc.MC(w, states, int(delta_t), int(q), mcmc_seed)
    MSA = np.copy(states)
    return np.array(MSA, dtype="int64")


def Wj(J=None, h=None):
    """
    Function to translate J (L*L*q*q) and h (L*q)  into a vector of L*q + (L-1)*L*q*q/2 variables.

    Parameters:
    J : numpy array
            A 4D array representing J with dimensions (L, L, q, q).
    h : numpy array
            An array representing h with dimensions (L, q).

    Returns:
    W : numpy array
            A 1D numpy array of independent variables derived from J and h.
    """
    if J is not None:
        L = J.shape[0]
        if L > 1:
            q = J.shape[2]
            W = np.zeros(
                int(
                    (q * L + q * q * L * (L - 1) / 2),
                )
            )
            x = np.array([[i, j] for i, j in it.combinations(range(L), 2)])
            for a in range(q):
                for b in range(q):
                    W[
                        (
                            q**2
                            * (
                                (x[:, 0]) * (2 * L - x[:, 0] - 1) / 2
                                + (x[:, 1] - x[:, 0] - 1)
                            )
                            + (a) * q
                            + b
                        ).astype(int)
                    ] = J[x[:, 0], x[:, 1], a, b]
            x = np.array(range(L))
            for a in range(q):
                W[(q**2 * L * (L - 1) / 2 + q * x[:] + a).astype(int)] = h[x[:], a]
    else:
        W = h.flatten()
    return W


def Jw(W, q, Couplings=True):
    L = int(
        ((q * q - 2 * q) + ((2 * q - q * q) ** 2 + 8 * W.shape[0] * q * q) ** (1 / 2))
        / 2
        / q
        / q
    )

    if L > 1 and Couplings:
        J = np.zeros((L, L, q, q))
        h = np.zeros((L, q))
        x = np.array([[i, j] for i, j in it.combinations(range(L), 2)])
        for a in range(q):
            for b in range(q):
                J[x[:, 0], x[:, 1], a, b] = W[
                    (
                        q**2
                        * (
                            (x[:, 0]) * (2 * L - x[:, 0] - 1) / 2
                            + (x[:, 1] - x[:, 0] - 1)
                        )
                        + (a) * q
                        + b
                    ).astype(int)
                ]
                J[x[:, 1], x[:, 0], b, a] = J[x[:, 0], x[:, 1], a, b]
        x = np.array(range(L))
        for a in range(q):
            h[x[:], a] = W[(q**2 * L * (L - 1) / 2 + q * x[:] + a).astype(int)]
        return J, h
    else:
        L = len(W) // q
        h = np.reshape(W, (L, q))
        return None, h


def states_rand(samples):
    """
    Function to randomize states in the input samples

    Args:
    - samples (numpy.array): A 2D numpy array representing the input samples.

    Returns:
    - numpy.array: A 2D numpy array representing the states after randomization without preserving correlations.
    """
    # output states randomised (without correlations)
    Cop = np.copy(samples)
    np.apply_along_axis(np.random.shuffle, 0, Cop)
    return Cop


##########################################################

####################### Post processing model #######################


def avg_over_runs(ws):
    ws_mean = np.mean(ws, axis=0)
    ws_std = np.std(ws, axis=0)
    # Count outliers for each run
    outliers = np.sum((ws - ws_mean) > ws_std, axis=1)
    # Find runs with outlier count more than 1 std above mean
    exclude = np.where((outliers - np.mean(outliers)) > np.std(outliers))[0]
    w_av = np.mean(np.delete(ws, exclude, axis=0), axis=0)
    return w_av


##########################################################
####################### COMPUTE STATISTICS #######################


def CalcWeights(align, theta, ignore_gaps=True):
    """
    Function to compute the weights and effective count for a given alignment and threshold.

    Parameters:
    align : numpy array
            The alignment data.
    theta : float
            The threshold value for distance (if dist < theta sequences are considered to be "the same")

    Returns:
    W : numpy array
            An array of weights calculated using the Hamming distance.
    N_eff : float
            The effective count derived from the sum of weights.
    """
    W = 0
    if ignore_gaps:
        counts = np.sum(squareform(compute_diversity(align)) < theta, axis=0)
        W = 1 / (counts + (counts == 0).astype("int"))
    else:  # consistent with MATLAB code, SCA code
        W = np.sum(squareform(pdist(align, "hamming")) < 0.3, axis=0)
        W = 1 / np.array([max(1, x) for x in W])
    N_eff = sum(W)
    return W, N_eff


def CalcStatsWeighted(q, MSA, p=None):
    """
    Function to calculate the statistics of a given weighted multiple sequence alignment,
    including the frequencies and pairwise frequencies.

    Parameters:
    q : int
            The number of amino acids.
    MSA : numpy array
            A 2D numpy array representing the Multiple Sequence Alignment.
    p : numpy array
            An array representing the weights. If None, it is set to an array of equal weights.

    Returns:
    fi : numpy array
            A 2D numpy array with the calculated frequencies.
    fij : numpy array
            A 4D numpy array with the pairwise frequencies.
    """
    if p is None:
        p = np.zeros(MSA.shape[0]) + 1 / MSA.shape[0]
    L = MSA.shape[1]
    fi = np.zeros([L, q])
    x = np.array([i for i in range(L)])
    for m in range(MSA.shape[0]):
        fi[x[:], MSA[m, x[:]]] += p[m]

    fij = np.zeros([L, L, q, q])
    x = np.array([[i, j] for i, j in it.product(range(L), range(L))])

    for m in range(MSA.shape[0]):
        fij[x[:, 0], x[:, 1], MSA[m, x[:, 0]], MSA[m, x[:, 1]]] += p[m]
    return fi, fij


def CalcThreeCorrWeighted(MSA, fi, fij, p=None, ind_L=None):
    """
    Function to calculate the three-point correlation of a given MSA.

    Args:
    - MSA (numpy.array): A 2D numpy array representing the Multiple Sequence Alignment.
    - fi (numpy.array): A numpy array with frequencies fi(a)
    - fij (numpy.array): A numpy array with pairwise frequencies fij(a,b)
    - p (numpy.array): An optional parameter representing weights. If None, it is set to an array of equal weights.
    - ind_L (numpy.array): An optional array representing indices for which we compute the three-point correlations.
    If None, it is set to all indices.

    Returns:
    - numpy.array: A 6D numpy array representing the three-point correlation.

    """
    L, q = fi.shape
    Np = MSA.shape[0]
    if p is None:
        p = np.zeros(Np) + 1 / Np
    if ind_L is None:
        ind_L = np.arange(L)
    nL = len(ind_L)
    fijk = np.zeros((nL, nL, nL, q, q, q))
    x = np.array([[i, j, k] for i, j, k in it.product(ind_L, ind_L, ind_L)])
    x2 = np.array(
        [[i, j, k] for i, j, k in it.product(range(nL), range(nL), range(nL))]
    )
    for m in tqdm(range(Np)):
        fijk[
            x2[:, 0],
            x2[:, 1],
            x2[:, 2],
            MSA[m, x[:, 0]],
            MSA[m, x[:, 1]],
            MSA[m, x[:, 2]],
        ] += p[m]
    fij_l, fi_l = fij[ind_L], fi[ind_L]
    fij_l = fij_l[:, ind_L]
    C3 = fijk - (fij_l.reshape(nL, nL, 1, q, q, 1) * fi_l.reshape(1, 1, nL, 1, 1, q))
    C3 -= fij_l.reshape(nL, 1, nL, q, 1, q) * fi_l.reshape(1, nL, 1, 1, q, 1)
    C3 -= fij_l.reshape(1, nL, nL, 1, q, q) * fi_l.reshape(nL, 1, 1, q, 1, 1)
    C3 += 2 * (
        fi_l.reshape(nL, 1, 1, q, 1, 1)
        * fi_l.reshape(1, nL, 1, 1, q, 1)
        * fi_l.reshape(1, 1, nL, 1, 1, q)
    )
    return C3


def compute_stats(output, align_mod):
    """
    Function to compute various statistics (frequency, pairwise frequency, and three-point correlation)
    for the test, training and artificial sets stored in the output dictionary.

    Args:
    - output (dict): A dictionary containing various data including 'Test', 'options', 'Train', and 'align_mod'.
    - align_mod (numpy.array): A 2D numpy array representing the alignment data.

    Returns:
    - dict: A dictionary containing different statistics calculated from the input data.
    """
    Stats = {}
    # options = output['options']
    train_align = output["Train"]
    test_align = output["Test"]
    if test_align is None:
        test_align = np.copy(train_align)
    M = min(train_align.shape[0], test_align.shape[0], align_mod.shape[0])
    train_align = train_align[
        np.sort(np.random.choice(train_align.shape[0], M, replace=False))
    ]
    align_mod = align_mod[
        np.sort(np.random.choice(align_mod.shape[0], M, replace=False))
    ]
    test_align = test_align[
        np.sort(np.random.choice(test_align.shape[0], M, replace=False))
    ]

    ind_L = np.random.choice(output["options1"]["L"], 10, replace=False)
    # Artificial stats
    art = {}
    W, N_eff = CalcWeights(align_mod, output["options0"]["theta"])
    fi_s, fij_s = CalcStatsWeighted(output["options1"]["q"], align_mod, W / N_eff)
    C3_s = CalcThreeCorrWeighted(align_mod, fi_s, fij_s, p=W / N_eff, ind_L=ind_L)
    art["Freq"] = fi_s
    art["Pair_freq"] = CalcCorr2(fi_s, fij_s)  # fij_s#
    art["Three_corr"] = C3_s

    # Train stats
    train = {}
    W, N_eff = CalcWeights(train_align, output["options0"]["theta"])
    fi, fij = CalcStatsWeighted(output["options1"]["q"], train_align, W / N_eff)
    C3 = CalcThreeCorrWeighted(train_align, fi, fij, p=W / N_eff, ind_L=ind_L)
    train["Freq"] = fi
    train["Pair_freq"] = CalcCorr2(fi, fij)  # fij#
    train["Three_corr"] = C3

    # Test stats
    test = {}
    W, N_eff = CalcWeights(test_align, output["options0"]["theta"])
    fi, fij = CalcStatsWeighted(output["options1"]["q"], test_align, W / N_eff)
    C3 = CalcThreeCorrWeighted(test_align, fi, fij, p=W / N_eff, ind_L=ind_L)
    test["Freq"] = fi
    test["Pair_freq"] = CalcCorr2(fi, fij)  # fij #
    test["Three_corr"] = C3

    Stats["Train"] = train
    Stats["Test"] = test
    Stats["Artificial"] = art

    return Stats


def CalcContingency(q, MSA):
    # input MSA in amino acid form
    # output the unweighted freqs fi, co-occurnces fij and correlations Cij=fij-fi*fj
    L = MSA.shape[1]
    fi = np.zeros([L, q])
    x = np.array([i for i in range(L)])
    for m in range(MSA.shape[0]):
        fi[x[:], MSA[m, x[:]]] += 1

    fij = np.zeros([L, L, q, q])
    x = np.array([[i, j] for i, j in it.product(range(L), range(L))])

    for m in range(MSA.shape[0]):
        fij[x[:, 0], x[:, 1], MSA[m, x[:, 0]], MSA[m, x[:, 1]]] += 1

    return fi, fij


def shuff_column(align):
    align_rand = np.zeros(align.shape)
    for i in range(align.shape[1]):
        col = np.copy(align[:, i])
        np.random.shuffle(col)
        align_rand[:, i] = col
    align_rand = align_rand.astype("int")
    return align_rand


def Zero_Sum_Gauge(J=None, h=None):
    """
    Function to apply a zero-sum gauge transformation to J and h matrices.

    Args:
    - J (numpy.array): A 4D numpy with couplings.
    - h (numpy.array): A 2D numpy array with fields.

    Returns:
    - J_zg (numpy.array): Updated J matrix after applying the zero-sum gauge transformation.
    - h_zg (numpy.array): Updated h vector after applying the zero-sum gauge transformation.
    """
    if J is None:
        L, q = h.shape
        J = np.zeros((L, L, q, q))
        h_zg = np.copy(h)
        h_zg -= np.expand_dims(np.mean(h, axis=1), axis=1)
        h_zg += np.sum(
            np.mean(J, axis=3) - np.expand_dims(np.mean(J, axis=(2, 3)), axis=2), axis=1
        )
        return None, h_zg

    if h is None:
        L, q = J.shape[0], J.shape[2]
        J_zg = np.copy(J)
        J_zg -= np.expand_dims(np.mean(J, axis=2), axis=2)
        J_zg -= np.expand_dims(np.mean(J, axis=3), axis=3)
        J_zg += np.expand_dims(np.mean(J, axis=(2, 3)), axis=(2, 3))
        return J_zg, None

    J_zg = np.copy(J)
    h_zg = np.copy(h)

    h_zg -= np.expand_dims(np.mean(h, axis=1), axis=1)
    h_zg += np.sum(
        np.mean(J, axis=3) - np.expand_dims(np.mean(J, axis=(2, 3)), axis=2), axis=1
    )

    J_zg -= np.expand_dims(np.mean(J, axis=2), axis=2)
    J_zg -= np.expand_dims(np.mean(J, axis=3), axis=3)
    J_zg += np.expand_dims(np.mean(J, axis=(2, 3)), axis=(2, 3))

    return J_zg, h_zg


def compute_energies(seqs, h, J=None):
    """
    Function to compute energies for an alignment based on the provided parameters provided h and J values.

    Args:
    - align (numpy.array): A 2D or 1D numpy array representing the input alignment.
    - h (numpy.array): A 2D numpy array representing the h values (fields).
    - J (numpy.array): A 4D numpy array representing the J values (couplings).

    Returns:
    - numpy.array: A 1D numpy array representing the computed energies for the input alignment.
    """
    if len(seqs.shape) == 2:
        L = seqs.shape[1]
    elif len(seqs.shape) == 1:
        L = seqs.shape[0]
        seqs = seqs.reshape((1, L))
    if J is None:
        J = np.zeros((L, L, h.shape[1], h.shape[1]))
    energy = np.sum(np.array([h[i, seqs[:, i]] for i in range(L)]), axis=0)
    energy = energy + (
        np.sum(
            np.array(
                [[J[i, j, seqs[:, i], seqs[:, j]] for j in range(L)] for i in range(L)]
            ),
            axis=(0, 1),
        )
        / 2
    )
    return -energy


def compute_similarities(Gen1, Gen2=None, N_aa=20):
    "Calculate distance to nearest sequence in Gen2 without tacking into acount the gaps"
    N1 = Gen1.shape[0]
    Gen1_2d = alg2bin(Gen1, N_aa=N_aa)

    if Gen2 is None:
        Sim = np.zeros(N1)
        for i in range(N1):
            a2d = Gen1_2d[i : i + 1]
            simMat = a2d.dot(Gen1_2d.T)
            SUM = Gen1[i] + Gen1
            norm = np.sum(SUM != 0, axis=1)
            d = 1 - simMat[0] / norm
            Sim[i] = np.amin(np.sort(d)[1:])
    else:
        Sim = np.zeros(N1)
        Gen2_2d = alg2bin(Gen2, N_aa=N_aa)
        for i in range(N1):
            a2d = Gen1_2d[i : i + 1]
            simMat = a2d.dot(Gen2_2d.T)
            SUM = Gen1[i] + Gen2
            norm = np.sum(SUM != 0, axis=1)
            d = 1 - simMat[0] / norm
            Sim[i] = np.amin(d)
    return Sim


def compute_diversity(alg, N_aa=20):
    "Calculate distance between sequences without tacking into acount the gaps"
    Nseq = alg.shape[0]
    X2d = alg2bin(alg, N_aa=N_aa)
    simMat = X2d.dot(X2d.T)
    Dist = simMat[np.triu_indices(Nseq, k=1)]
    NORM = np.zeros(Dist.size)
    idx = 0
    for i in range(Nseq - 1):
        a = alg[i]
        align_rm = alg[i + 1 :]
        SUM = a + align_rm
        NORM[idx : idx + align_rm.shape[0]] = np.sum(SUM != 0, axis=1)
        idx += align_rm.shape[0]
    Dist = 1 - Dist / NORM
    return Dist


def CalcCorr2(fi, fij):
    """
    Function to calculate pairwise correlations based on the provided fi and fij values.

    Args:
    fi (numpy.array): A 2D numpy array representing the frequencies.
    fij (numpy.array): A 4D numpy array representing the pairwise frequencies.

    Returns:
    numpy.array: A 4D numpy array representing the calculated pairwise correlations.
    """
    L, q = fi.shape
    Cij = fij - (fi.reshape([L, 1, q, 1]) * fi.reshape([1, L, 1, q]))
    for i in range(L):
        Cij[i, i, :, :] = 0
    return Cij


##########################################################

####################### PCA #######################


def PCA_comparison(COG_samp, COG_model, Pears=0, Mask=1):
    assert COG_samp.shape[1] == COG_model.shape[1]

    if Pears != 0:
        Cov_Ising = np.corrcoef(COG_samp.T) * Mask
    else:
        Cov_Ising = np.cov(COG_samp.T) * Mask

    W_cov, V_cov = np.linalg.eigh(Cov_Ising)

    ind_sort = np.argsort(W_cov)[::-1]
    W_cov = W_cov[ind_sort]
    V_cov = V_cov[:, ind_sort]

    w1, w2 = W_cov[0], W_cov[1]
    v1, v2 = V_cov[:, 0], V_cov[:, 1]

    v1_norm, v2_norm = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)

    ProjX_samp, ProjY_samp = COG_samp @ v1_norm, COG_samp @ v2_norm
    ProjX_model, ProjY_model = COG_model @ v1_norm, COG_model @ v2_norm

    conserved_var = (w1 + w2) / np.sum(W_cov)
    print("Conserved Var. ", conserved_var * 100, "%")

    X_samp = np.concatenate(
        (np.expand_dims(ProjX_samp, axis=1), np.expand_dims(ProjY_samp, axis=1)), axis=1
    )
    X_model = np.concatenate(
        (np.expand_dims(ProjX_model, axis=1), np.expand_dims(ProjY_model, axis=1)),
        axis=1,
    )

    return X_samp, X_model


##########################################################

####################### OTHER FUNCTIONS #######################


def MSA_from_seqlist(seq_list):
    code = "-ACDEFGHIKLMNPQRSTVWY"
    AA_to_num = dict([(code[i], i) for i in range(len(code))])
    errs = "BJOUXZabcdefghijklmonpqrstuvwxyz"
    AA_to_num.update(dict([(errs[i], -1) for i in range(len(errs))]))
    for s in range(len(seq_list)):
        chars = [*seq_list[s]]
        seq = np.array([AA_to_num[chars[i]] for i in range(len(chars))])
        if s == 0:
            MSA = np.expand_dims(seq, axis=0)
        else:
            MSA = np.concatenate((MSA, np.expand_dims(seq, axis=0)), axis=0)
    if np.min(MSA) < 0:
        MSA = np.delete(MSA, (np.sum(MSA == -1, axis=1)).nonzero()[0][0], axis=0)
    return MSA


def alg2bin(alg, N_aa=20):
    """
    Function to convert an alignment into a binary representation.

    Args:
    - alg (numpy.array): A 2D numpy array representing the input alignment.
    - N_aa (int): The number of amino acids. Default value is 20.

    Returns:
    - numpy.array: A 2D numpy array representing the binary representation of the input alignment.
    """
    [N_seq, N_pos] = alg.shape
    Abin_tensor = np.zeros((N_aa, N_pos, N_seq))
    for ia in range(N_aa):
        Abin_tensor[ia, :, :] = (alg == ia + 1).T
    Abin = Abin_tensor.reshape(N_aa * N_pos, N_seq, order="F").T
    return Abin


def averaged_model(file_names, fam, Model, ITER=""):
    """
    Function to compute the averaged model based on the provided file names, family, and Model.
    It saves the results with the appropriate naming convention.

    Args:
    - file_names (list): A list of file names.
    - fam (str): A string representing the family.
    - Model (str): A string representing the model.
    - ITER (str): An optional string representing the iteration.

    Returns: None
    """
    AVG_model = np.load("results/" + fam + "/" + file_names[0], allow_pickle=True)[()]
    J_avg = np.zeros(AVG_model["J" + str(ITER)].shape)
    h_avg = np.zeros(AVG_model["h" + str(ITER)].shape)
    for f in file_names:
        mod = np.load("results/" + fam + "/" + f, allow_pickle=True)[()]
        J_avg += mod["J" + str(ITER)]
        h_avg += mod["h" + str(ITER)]
    AVG_model["J"] = J_avg / len(file_names)
    AVG_model["h"] = h_avg / len(file_names)
    if Model == "SBM":
        if ITER == "":
            nb_it = AVG_model["options"]["N_iter"]
        else:
            nb_it = ITER
        # np.save('results/Article/'+Model+'/'+fam+'/'+fam+'_avgMod_m'+str(AVG_model['options']['m'])+'Ns'+str(AVG_model['options']['n_states'])+'Ni'+str(nb_it)+'.npy',AVG_model)
        np.save(
            "results/"
            + fam
            + "/"
            + "TMF_avgMod_m"
            + str(AVG_model["options"]["m"])
            + "Ns"
            + str(AVG_model["options"]["N_chains"])
            + "Ni"
            + str(nb_it)
            + ".npy",
            AVG_model,
        )


##########################################################
