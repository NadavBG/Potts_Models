"""
Created on Fri Oct 23 00:25:00 2020

@author: yaakov KLEEORIN & Marion CHAUVEAU
"""

####################### MODULES #######################

import numpy as np
from tqdm import tqdm
import time
import more_itertools as mit
import SBM.utils.utils as ut

##########################################################

####################### OPTIONS #######################


def ParseOptions(options):
    assert options["Model"] in ["BM", "SBM"]
    Opt = [
        ("N_iter", 300),  # nb of GD iterations
        ("N_chains", 1000),  # nb of states used to compute statistics
        ("m", 1),  # Rank of the Hessian matrix (only for SBM)
        ("theta", 0.2),
        (
            "ignore_gaps_weighting",
            True,
        ),  # ignore gaps when calculating sequence weights
        ("k_MCMC", 10000),
        ("PseudoCount", False),  # the default pseudo count is 1/Neff
        ("alpha", 0.2),  # Learning rate for the BM method
        ("Learning_rate", None),
        ("lambda_h", 0),  # regularization for the fields
        ("lambda_J", 0),  # regularization for the couplings
        ("regul", "L2"),
        ("Pruning", False),
        ("Pruning_perc", 0.9),
        ("Pruning Mask Couplings", None),
        ("Infinite Mask Fields", None),  # To forbid certain a.a at certain positions
        ("Param_init", "profile"),  # Zero, Profile, Custom
        (
            "Test/Train",
            True,
        ),  # If True and 'Train sequences' is None: the MSA is randomly splitted in a 80% training set / 20% test set
        ("Train sequences", None),  # indices of sequences used for training
        ("Precomputed_Stats", None),
        ("Weights", None),
        ("Shuffle Columns", False),
        ("SGD", None),  # for classic stochastic gradient descent
        ("Seed", None),
        ("Zero Fields", False),  # True to impose zero fields
        ("Zero Couplings", False),  # True to impose zero couplings
        (
            "Store Parameters",
            None,
        ),  # if not None store couplings and fields every ** iterations ('Store Couplings', **)
    ]
    for k, v in Opt:
        if k not in options.keys():
            if options["Model"] == "SBM":
                if k not in ["alpha", "Learning rate"]:
                    options[k] = v
            else:
                if k not in ["m"]:
                    options[k] = v
    return options


##############################################################################

####################### FUNCTIONS for Initialization #######################


def Init_options(options, align):
    options = ParseOptions(options)

    options["q"] = np.max(align) + 1
    options["L"] = align.shape[1]

    ############# SEED #############
    if options["Seed"] is None:
        t = time.time()
        options["Seed"] = int(t)
    np.random.seed(options["Seed"])
    ################################

    return options


def Init_TestTrain(options, align):
    N = align.shape[0]

    ########## TEST/TRAIN ##########
    if options["Test/Train"]:
        if options["Train sequences"] is None:
            ind_train = np.random.choice(N, int(0.80 * N), replace=False)
        else:
            ind_train = options["Train sequences"]
        train_align = align[ind_train]
        test_align = align[np.delete(np.arange(N), ind_train)]
        # sim_test = ut.compute_similarities(test_align,train_align)
        # test_align = test_align[(sim_test>0.2)]
    else:
        train_align = align
        test_align = None

    if options["Shuffle Columns"]:
        print("Shuffle Columns...")
        train_align = ut.shuff_column(train_align)
    ################################
    return train_align, test_align


def Init_SGD(options, train_align):
    ########## SGD OPTIONS #########
    if options["SGD"] is not None:
        assert options["SGD"] <= train_align.shape[0]
        ind = np.arange(train_align.shape[0])
        np.random.shuffle(ind)
        options["Batches"] = list(mit.chunked(ind, options["SGD"]))
        options["Num_batch"] = 0
    ################################

    if options["SGD"] is not None:
        align_subsamp = train_align
    else:
        align_subsamp = None

    return align_subsamp


def Init_statistics(options, train_align):
    ###### EVALUATE GOAL STATS #####
    print("Compute the statistics from the database....")
    if options["Weights"] is None:
        W, N_eff = ut.CalcWeights(
            train_align, options["theta"], options["ignore_gaps_weighting"]
        )
    else:
        assert len(options["Weights"]) == train_align.shape[0]
        W = options["Weights"]
        N_eff = np.sum(W)
    print("Training size: ", train_align.shape[0])
    print("Effective training size: ", N_eff)

    if options["Precomputed_Stats"] is None:
        fi, fij = ut.CalcStatsWeighted(options["q"], train_align, W / N_eff)
    else:
        fi, fij = (
            options["Precomputed_Stats"]["fi"],
            options["Precomputed_Stats"]["fij"],
        )
    ################################

    return fi, fij, N_eff


def Init_Pruning(options, fij):
    if options["Pruning"]:
        # Preserve the original input (path string or None) for provenance,
        # since the next assignment overwrites the option with the array.
        options.setdefault(
            "Pruning Mask Couplings Source", options["Pruning Mask Couplings"]
        )
        if options["Pruning Mask Couplings"] is None:
            # Nombre d'éléments à mettre à zéro
            total = fij.size
            n_zero = int(options["Pruning_perc"] * total)

            # Indices des plus petits éléments
            flat_indices = np.argpartition(fij.flatten(), n_zero)[:n_zero]

            # Création du masque
            Mask = np.ones(fij.size, dtype=int)
            Mask[flat_indices] = 0
            Mask = Mask.reshape(fij.shape)
        else:
            Mask = np.load(options["Pruning Mask Couplings"])

        options["Pruning Mask Couplings"] = Mask.astype("int")
        options["Pruning_perc"] = 1 - np.sum(Mask) / Mask.size
        print("Pruning pct: ", 1 - np.sum(Mask) / Mask.size)


def Init_Param(options, J0, h0, N_eff, fi):
    ########### PARAM INIT #########
    if options["Param_init"].lower() == "zero":
        Jinit = np.zeros((options["L"], options["L"], options["q"], options["q"]))
        hinit = np.zeros((options["L"], options["q"]))
    elif options["Param_init"].lower() == "profile":
        Jinit = np.zeros((options["L"], options["L"], options["q"], options["q"]))
        alpha = 1 / N_eff
        fi_init = (1 - alpha) * fi + alpha / options["q"]
        hinit = np.log(fi_init)  # - np.log(1 - fi_init)
    elif options["Param_init"].lower() == "custom":
        # assert J0 is not None and h0 is not None
        Jinit = J0
        hinit = h0
    elif options["Param_init"].lower() == "random":
        ma = 1
        Jinit = np.random.uniform(
            -ma, ma, (options["L"], options["L"], options["q"], options["q"])
        )
        hinit = np.random.uniform(-ma, ma, (options["L"], options["q"]))
    else:
        print('This "Param_init" option is not available')
        assert 0 == 1
    ################################

    ################################
    if options["Pruning"]:
        Jinit *= options["Pruning Mask Couplings"]
    if options["Infinite Mask Fields"] is not None:
        hinit[~options["Infinite Mask Fields"]] = -1e4
    if options["Zero Fields"]:
        hinit *= 0
    if options["Zero Couplings"]:
        Jinit = None

    w0 = ut.Wj(Jinit, hinit)

    ################################

    return w0


def add_PseudoCount(options, fi, fij, N_eff):
    alpha = 1 / N_eff
    fi_pc = (1 - alpha) * fi + alpha / options["q"]
    fij_pc = (1 - alpha) * fij + alpha / options["q"] ** 2
    return fi_pc, fij_pc


##############################################################################

####################### FUNCTIONS OUTSIDE THE MINIMIZER #######################


def SBM(align, options, J0=None, h0=None):
    ###### SBM Initialization ######
    options = Init_options(options, align)
    train_align, test_align = Init_TestTrain(options, align)
    align_subsamp = Init_SGD(options, train_align)
    fi, fij, N_eff = Init_statistics(options, train_align)
    Init_Pruning(options, fij)
    w0 = Init_Param(options, J0, h0, N_eff, fi)

    if options["PseudoCount"]:
        print("Adding pseudo count on statistics")
        fi, fij = add_PseudoCount(options, fi, fij, N_eff)

    ################################

    ###### OBJECTIVE FUNCTION ######
    lamJ, lamh = options["lambda_J"], options["lambda_h"]

    def f(x):
        return GradLogLike(x, lamJ, lamh, fi, fij, options, align_subsamp=align_subsamp)

    ################################

    ####### GRADIENT DESCENT #######
    Ex_time = time.time()
    w, output = Minimizer(f, w0, options)
    output["Execution time"] = time.time() - Ex_time
    print("Execution time: ", time.time() - Ex_time)
    ################################

    output["options"] = options
    if options["Zero Couplings"]:
        J, h = ut.Jw(w, options["q"], Couplings=False)
        output["J_norm"] = None
    else:
        J, h = ut.Jw(w, options["q"])
        output["h_norm"] = None

    output["J"] = J
    output["h"] = h

    output["align"] = align
    output["Test"] = test_align
    output["Train"] = train_align

    return output


def GradLogLike(w, lambdaJ, lambdah, fi, fij, options, align_subsamp=None):
    if options["Zero Couplings"]:
        J, h = ut.Jw(w, options["q"], Couplings=False)
    else:
        J, h = ut.Jw(w, options["q"])

    ########## MODEL STATS #########
    if options["Zero Fields"]:
        h *= 0

    align_mod = ut.Create_modAlign(
        {"J": J, "h": h}, options["N_chains"], delta_t=options["k_MCMC"]
    )
    p = np.zeros(options["N_chains"]) + 1 / options["N_chains"]
    fi_mod, fij_mod = ut.CalcStatsWeighted(options["q"], align_mod, p)
    ################################

    ########## SGD OPTIONS #########
    if options["SGD"] is not None:
        Batch = options["Batches"][options["Num_batch"]]
        sub = align_subsamp[Batch]
        if options["Num_batch"] == len(options["Batches"]) - 1:
            options["Num_batch"] = 0
        else:
            options["Num_batch"] = options["Num_batch"] + 1
        W, N_eff = ut.CalcWeights(
            sub, options["theta"], options["ignore_gaps_weighting"]
        )
        fi, fij = ut.CalcStatsWeighted(options["q"], sub, W / N_eff)
    ################################

    ####### COMPUTE GRADIENTS ######
    if options["regul"].lower() == "l2":
        gradh = fi_mod - fi + 2 * lambdah * h
        if J is not None:
            gradJ = fij_mod - fij + 2 * lambdaJ * J
    elif options["regul"].lower() == "l1":
        gradh = fi_mod - fi + lambdah
        if J is not None:
            gradJ = fij_mod - fij + lambdaJ
    elif options["regul"].lower() == "both":
        gradh = fi_mod - fi + lambdah[0] + 2 * lambdah[1] * h
        if J is not None:
            gradJ = fij_mod - fij + lambdaJ[0] + 2 * lambdaJ[1] * J
    ################################

    if options["Pruning"]:
        gradJ *= options["Pruning Mask Couplings"]
    if options["Infinite Mask Fields"] is not None:
        gradh *= options["Infinite Mask Fields"]
    if options["Zero Fields"]:
        gradh *= 0
    if options["Zero Couplings"]:
        gradJ = None

    grad = ut.Wj(gradJ, gradh)
    return grad


##########################################################

####################### FUNCTIONS INSIDE THE MINIMIZER #######################


def Minimizer(fun, x0, options):
    x = np.copy(x0)
    output = {"skipping": 0, "J_norm": 0, "h_norm": 0}
    if options["Store Parameters"] is not None:
        output["Trajectory"] = {"w_0": x0}

    for i in tqdm(range(options["N_iter"])):
        ########## SBM METHOD #########
        if options["Model"] == "SBM":
            if i == 0:
                g = fun(x)
                h = -g
                s = np.zeros((x.shape[0], options["m"]))
                y = np.zeros((x.shape[0], options["m"]))
                ys = np.zeros((options["m"]))
                diag = 1
                gtd = np.dot(-g, h)
                ind = np.zeros(options["m"]) - 1
                ind[0] = 0
                t = 1 / np.sum(g**2) ** 0.5
            else:
                t = 1
            x, h, g, gtd, s, y, ys, diag, ind, output["skipping"] = AdvanceSearch(
                x, t, h, g, fun, gtd, s, y, ys, diag, ind, output["skipping"], options
            )
        ################################

        ########## BM METHOD ###########
        else:
            if options["Learning_rate"] is not None:
                t = options["Learning_rate"]
            else:
                t = 1 / ((i + 1) ** options["alpha"])
            grad = fun(x)
            x -= t * grad
        ################################

        ################################
        idx = i + 1
        if options["Store Parameters"] is not None:
            if options["Store Parameters"] == 1:
                output["Trajectory"]["w_" + str(idx)] = np.copy(x)
            else:
                if idx < 10 and idx % 2 == 0:
                    output["Trajectory"]["w_" + str(idx)] = np.copy(x)
                elif idx < 100 and idx % 10 == 0:
                    output["Trajectory"]["w_" + str(idx)] = np.copy(x)
                elif idx < 1000 and idx % 50 == 0:
                    output["Trajectory"]["w_" + str(idx)] = np.copy(x)
                elif idx % options["Store Parameters"] == 0:
                    output["Trajectory"]["w_" + str(idx)] = np.copy(x)

        if i % 100 == 0:
            if not options["Zero Couplings"]:
                J, h_field = ut.Jw(x, options["q"])
                J_norm = np.mean(np.linalg.norm(J, "fro", axis=(2, 3)))
                output["J_norm"] = np.append(output["J_norm"], np.round(J_norm, 3))
        if options["Zero Couplings"]:
            J, h_field = ut.Jw(x, options["q"], Couplings=False)
            J, h_field = ut.Zero_Sum_Gauge(J, h_field)
            h_norm = np.mean(h_field[:, 1])  # np.sqrt(np.sum(h_field**2,axis = 1)))
            output["h_norm"] = np.append(output["h_norm"], h_norm)

    return x, output


def AdvanceSearch(x, t, h, g, fun, gtd, s, y, ys, diag, ind, skipping, options):
    count = 0
    while True:
        x_out = x + t * h
        # calculate the gradient
        g_out = fun(x_out)
        # and use it to update the h=hessian*gradient
        h_out, s_out, y_out, ys_out, diag_out, ind_out, skipping_out = UpdateHessian(
            g_out, g_out - g, x_out - x, s, y, ys, diag, ind, skipping, options
        )
        gtd_out = np.dot(-g_out, h_out)
        # sometimes this can be an irrelevant step, retry it if it strays too far
        if abs(gtd_out) < abs(70 * gtd):
            break
        else:
            print(abs(gtd_out), abs(70 * gtd))
            count += 1
            if count == 10:
                print("too much irrelevant steps")
                break
    # hessian_save = h/(g_out+(g_out==0).astype('int'))
    return (
        x_out,
        h_out,
        g_out,
        gtd_out,
        s_out,
        y_out,
        ys_out,
        diag_out,
        ind_out,
        skipping_out,
    )


def UpdateHessian(g, y, s, s_out, y_out, ys_out, diag, ind, skipping, options):
    ys = np.dot(y, s)
    # if this is a meaningful step
    if ys > 10 ** (-10):
        y_out[:, ind == max(ind)] = y.reshape(-1, 1)
        s_out[:, ind == max(ind)] = s.reshape(-1, 1)
        ys_out[ind == max(ind)] = ys
        diag = ys / np.dot(y, y)
    # or if not meaningful, the update will be skipped
    else:
        skipping = skipping + 1
    # here the hessian*gradient is calculated
    h_out = -g
    order = np.argsort(ind[ind > -1])
    alpha = np.zeros(order.shape[0])
    beta = np.zeros(order.shape[0])
    for i in order[::-1]:
        alpha[i] = np.dot(s_out[:, i], h_out) / ys_out[i]
        h_out = h_out - alpha[i] * y_out[:, i]
    h_out = diag * h_out
    for i in order:
        beta[i] = np.dot(y_out[:, i], h_out) / ys_out[i]
        h_out = h_out + s_out[:, i] * (alpha[i] - beta[i])

    # update the memory steps (indices) only if it is meaningful
    if ys > 10 ** (-10):
        if ind[options["m"] - 1] == -1:
            ind = ind
            ind[(ind == max(ind)).nonzero()[0] + 1] = max(ind) + 1
        else:
            ind = np.roll(ind, 1)
    return h_out, s_out, y_out, ys_out, diag, ind, skipping
