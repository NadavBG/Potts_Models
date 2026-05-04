"""
@author: Marion CHAUVEAU

:On:  October 2022
"""

####################### MODULES #######################

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import SBM.utils.utils as ut
from matplotlib.colors import Normalize  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from matplotlib import cm  # type: ignore

##########################################################

####################### PLOT STATISTICS #######################


def plot_stats(output, Stats, plot="Freq", ma=None):
    if plot == "Freq":
        if ma is None:
            ma = 1
        fig = plt.figure(figsize=(12, 4))
        fig.add_subplot(1, 2, 1)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Freq"].flatten(), Stats["Artificial"]["Freq"].flatten()
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="1st order statistics\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Freq"].flatten(),
            Stats["Artificial"]["Freq"].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Artificial set", fontsize=14)
        plt.plot([0, ma], [0, ma], color="black")
        # plt.xticks(fontsize = 16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Artificial VS Test")

        fig.add_subplot(1, 2, 2)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Freq"].flatten(), Stats["Train"]["Freq"].flatten()
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="1st order statistics\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Freq"].flatten(),
            Stats["Train"]["Freq"].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Training set", fontsize=14)
        plt.plot([0, ma], [0, ma], color="black")
        # plt.xticks(fontsize = 16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Train VS Test")

        fig.tight_layout()

    if plot == "Pair_freq":
        if ma is None:
            ma = 0.4
        fig = plt.figure(figsize=(12, 4))
        fig.add_subplot(1, 2, 1)
        ind = np.triu_indices(output["align_mod"].shape[1], 1)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Pair_freq"][ind].flatten(),
                Stats["Artificial"]["Pair_freq"][ind].flatten(),
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="Pairwise Corr\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Pair_freq"][ind].flatten(),
            Stats["Artificial"]["Pair_freq"][ind].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Artificial set", fontsize=14)
        # plt.plot([0,ma],[0,ma])
        plt.plot([-ma, ma], [-ma, ma], color="black")
        # plt.xticks(fontsize = 16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Artificial VS Test")

        fig.add_subplot(1, 2, 2)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Pair_freq"][ind].flatten(),
                Stats["Train"]["Pair_freq"][ind].flatten(),
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="Pairwise Corr\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Pair_freq"][ind].flatten(),
            Stats["Train"]["Pair_freq"][ind].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Training set", fontsize=14)
        # plt.plot([0,ma],[0,ma])
        plt.plot([-ma, ma], [-ma, ma], color="black")
        # plt.xticks(fontsize = 16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Train VS Test")

        fig.tight_layout()

    if plot == "Corr3":
        if ma is None:
            ma = 0.1
        fig = plt.figure(figsize=(12, 4))
        fig.add_subplot(1, 2, 1)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Three_corr"].flatten(),
                Stats["Artificial"]["Three_corr"].flatten(),
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="3rd order correlations\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Three_corr"].flatten(),
            Stats["Artificial"]["Three_corr"].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Artificial set", fontsize=14)
        plt.plot(
            [
                np.amin(Stats["Test"]["Three_corr"]),
                np.amax(Stats["Test"]["Three_corr"]),
            ],
            [
                np.amin(Stats["Test"]["Three_corr"]),
                np.amax(Stats["Test"]["Three_corr"]),
            ],
            color="black",
        )
        # plt.xticks(fontsize = 14)
        # plt.plot([-ma,ma],[-ma,ma])
        plt.legend(loc="upper left", fontsize=12)
        plt.grid()
        plt.title("Artificial VS Test")

        fig.add_subplot(1, 2, 2)
        Pears = np.round(
            np.corrcoef(
                Stats["Test"]["Three_corr"].flatten(),
                Stats["Train"]["Three_corr"].flatten(),
            )[0, 1],
            2,
        )
        plt.plot(
            [],
            [],
            "o",
            markersize=3,
            color="white",
            label="3rd order correlations\n Pearson: " + str(Pears),
        )
        plt.plot(
            Stats["Test"]["Three_corr"].flatten(),
            Stats["Train"]["Three_corr"].flatten(),
            "o",
            markersize=3,
            color="grey",
        )
        plt.xlabel("Test set", fontsize=14)
        plt.ylabel("Training set", fontsize=14)
        plt.plot(
            [
                np.amin(Stats["Test"]["Three_corr"]),
                np.amax(Stats["Test"]["Three_corr"]),
            ],
            [
                np.amin(Stats["Test"]["Three_corr"]),
                np.amax(Stats["Test"]["Three_corr"]),
            ],
            color="black",
        )
        # plt.xticks(fontsize = 14)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid()
        plt.title("Train VS Test")

        fig.tight_layout()

    if plot == "PCA":
        axis_font = {"size": "17"}
        Max = 0.15
        align_nat = output["align"]
        M = min(align_nat.shape[0], output["align_mod"].shape[0])
        sub_align_PCA = align_nat[
            np.random.choice(align_nat.shape[0], M, replace=False)
        ]
        sub_align_mod = output["align_mod"][
            np.random.choice(output["align_mod"].shape[0], M, replace=False)
        ]

        bin_align = ut.alg2bin(sub_align_PCA, N_aa=20)
        bin_align_mod = ut.alg2bin(sub_align_mod, N_aa=20)

        X, X_mod = ut.PCA_comparison(bin_align, bin_align_mod, Pears=0, Mask=1)

        shift = 0.4
        ma1, mi1 = np.amax(X[:, 0]) + shift, np.amin(X[:, 0]) - shift
        ma2, mi2 = np.amax(X[:, 1]) + shift, np.amin(X[:, 1]) - shift
        # Wass_dist = ot.sliced_wasserstein_distance(X, X_mod, n_projections=500)
        # print('Dist:',Wass_dist)

        density_scatter(X[:, 0], X[:, 1], Max=Max, markersize=18)
        plt.xlim([mi1, ma1])
        plt.ylim([mi2, ma2])
        plt.xlabel("PC 1", **axis_font)
        plt.ylabel("PC 2", **axis_font)
        plt.title("Natural sequences", **axis_font)
        plt.grid(color="gray", linestyle=(0, (5, 10)))
        plt.gca().spines[["right", "top", "left", "bottom"]].set_visible(False)

        density_scatter(X_mod[:, 0], X_mod[:, 1], Max=Max, markersize=18)
        plt.xlim([mi1, ma1])
        plt.ylim([mi2, ma2])
        plt.xlabel("PC 1", **axis_font)
        plt.ylabel("PC 2", **axis_font)
        plt.title("Artificial sequences", **axis_font)
        plt.grid(color="gray", linestyle=(0, (5, 10)))
        plt.gca().spines[["right", "top", "left", "bottom"]].set_visible(False)

    if plot == "Energy":
        fig = plt.figure(figsize=(8, 4))
        Bins = 60
        # Random Sequences
        rand = np.round(np.random.random(output["align_mod"].shape)).astype("int32")

        Erand_SBM = ut.compute_energies(rand, output["h"], output["J"])
        Etest_SBM = ut.compute_energies(output["Test"], output["h"], output["J"])
        Etrain_SBM = ut.compute_energies(output["Train"], output["h"], output["J"])
        Emod_SBM = ut.compute_energies(output["align_mod"], output["h"], output["J"])
        Mean_SBM = np.mean(Etrain_SBM)
        STD_SBM = np.std(Etrain_SBM)
        Erand_SBM, Etest_SBM, Etrain_SBM, Emod_SBM = (
            (Erand_SBM - Mean_SBM) / STD_SBM,
            (Etest_SBM - Mean_SBM) / STD_SBM,
            (Etrain_SBM - Mean_SBM) / STD_SBM,
            (Emod_SBM - Mean_SBM) / STD_SBM,
        )

        fig.add_subplot(1, 1, 1)
        c1, c2, c3 = (
            "rgb(0.279,0.681,0.901)",
            "rgb(0.616,0.341,0.157)",
            "rgb(0.092,0.239,0.404)",
        )
        mi = np.amin(np.concatenate((Etest_SBM, Etrain_SBM, Emod_SBM, Erand_SBM)))
        ma = np.amax(np.concatenate((Etest_SBM, Etrain_SBM, Emod_SBM, Erand_SBM)))
        plt.hist(
            Etest_SBM,
            Bins,
            range=(mi - 0.5, ma + 0.5),
            alpha=0.4,
            label="Test",
            color=c1,
            density=True,
        )
        plt.hist(
            Emod_SBM,
            Bins,
            range=(mi - 0.5, ma + 0.5),
            alpha=0.4,
            label="Artificial",
            color=c2,
            density=True,
        )
        plt.hist(
            Etrain_SBM,
            Bins,
            range=(mi - 0.5, ma + 0.5),
            alpha=0.4,
            label="Train",
            color=c3,
            density=True,
        )
        plt.hist(
            Erand_SBM,
            Bins,
            range=(mi - 0.5, ma + 0.5),
            alpha=0.4,
            label="Random",
            color="grey",
            density=True,
        )
        plt.legend()
        plt.xlabel("Statistical energy")
        plt.ylabel("probability den.")
        plt.grid()

        fig.tight_layout()

    if plot == "Similarity":
        fig = plt.figure(figsize=(8, 4))
        Bins = 80  # 25
        # align_train = ut.RemoveCloseSeqs(output['Train'],0.2)$
        Sim_SBM = ut.compute_similarities(output["align_mod"], output["Train"])
        Sim_train = ut.compute_similarities(output["Train"])
        Sim_test = ut.compute_similarities(output["Test"], output["Train"])

        fig.add_subplot(1, 1, 1)
        plt.hist(Sim_test, Bins, range=(0, 1), alpha=0.4, density=True, label="Test")
        plt.hist(
            Sim_SBM, Bins, range=(0, 1), alpha=0.4, density=True, label="Artificial"
        )
        plt.hist(Sim_train, Bins, range=(0, 1), alpha=0.4, density=True, label="Train")
        plt.xlabel("Distance to closest natural seq")
        plt.ylabel("probability den.")
        plt.legend()
        plt.grid()
        fig.tight_layout()

    if plot == "Diversity":
        Bins = 60
        fig = plt.figure(figsize=(8, 4))
        Div_SBM = ut.compute_diversity(output["align_mod"])
        Div_train = ut.compute_diversity(output["Train"])
        Div_test = ut.compute_diversity(output["Test"])

        fig.add_subplot(1, 1, 1)
        plt.hist(Div_train, Bins, range=(0, 1), label="Train", alpha=0.3, density=True)
        plt.hist(Div_test, Bins, range=(0, 1), label="Test", alpha=0.3, density=True)
        plt.hist(
            Div_SBM, Bins, range=(0, 1), alpha=0.3, label="Artificial", density=True
        )
        plt.legend()
        plt.xlabel("Diversity")
        plt.ylabel("probability den.")
        plt.grid()
        fig.tight_layout()

    if plot == "Length":
        Bins = 80
        fig = plt.figure(figsize=(8, 4))
        Length_SBM = np.sum(output["align_mod"], axis=1)
        Length_train = np.sum(output["Train"], axis=1)
        Length_test = np.sum(output["Test"], axis=1)

        fig.add_subplot(1, 1, 1)
        mi, ma = (
            np.amin(np.concatenate((Length_SBM, Length_train, Length_test))),
            np.amax(np.concatenate((Length_SBM, Length_train, Length_test))),
        )
        plt.hist(
            Length_train, Bins, range=(mi, ma), label="Train", alpha=0.3, density=True
        )
        plt.hist(
            Length_test, Bins, range=(mi, ma), label="Test", alpha=0.3, density=True
        )
        plt.hist(
            Length_SBM,
            Bins,
            range=(mi, ma),
            alpha=0.3,
            label="Artificial",
            density=True,
        )
        plt.legend()
        plt.xlabel("Genome Length")
        plt.ylabel("probability den.")
        plt.grid()
        fig.tight_layout()

    if plot == "Coupling_evol":
        fig = plt.figure(figsize=(5, 4))
        plt.plot(output["J_norm"], "o", markersize=2, color="tab:blue")
        plt.xlabel("Iterations")
        plt.ylabel("Couplings norm")
        # plt.xticks(fontsize = 16)
        plt.grid()
        # plt.yscale('log')
        plt.title(
            "SBM, n_states="
            + str(output["options"]["n_states"])
            + " m="
            + str(output["options"]["m"])
        )


def density_scatter(x, y, Max, markersize=10):
    """
    Scatter plot colored by 2d histogram
    """
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # print(np.min(z),np.max(z))
    z = np.concatenate((np.array([0]), z, np.array([Max])))
    x = np.concatenate((np.array([-10]), x, np.array([-10])))
    y = np.concatenate((np.array([-10]), y, np.array([-10])))
    # print(x.shape,y.shape,z.shape)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=markersize, cmap="magma")

    norm = Normalize(vmin=0, vmax=Max)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax)
    # cbar.ax.set_ylabel('Density')q
    return ax
