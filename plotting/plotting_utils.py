import numpy as np
import matplotlib.pyplot as plt

import utils

colors = ["#0000cc"]
plt.rc("font", **{"size": 16})


# can add latex font style
# plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
# plt.rc("text", usetex=True)
def dup(a):
    return np.append(a, a[-1])


def make_hist(
    reference,
    generated,
    xlabel="",
    ylabel="a.u.",
    logy=False,
    binning=None,
    label_loc="best",
    normalize=True,
    fname="",
    leg_font=16,
):

    if binning is None:
        binning = np.linspace(
            np.quantile(reference, 0.0), np.quantile(reference, 1), 50
        )

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(5, 4.5),
        gridspec_kw={"hspace": 0.0, "height_ratios": (3, 1)},
        sharex=True,
    )

    # Geant4 lines
    dist_ref, binning = np.histogram(
        reference,
        bins=binning,
        density=False,
    )
    # Normalize (guard against empty histogram)
    ref_sum = dist_ref.sum()
    dist_ref_normalized = dist_ref / ref_sum if ref_sum > 0 else dist_ref.astype(float)
    den_counts = np.sqrt(dist_ref) # 0 where N=0
    # Poisson error on normalized histogram: sigma(p_i) = sqrt(N_i) / N_tot
    # (equivalent to p_i / sqrt(N_i) but safe for N_i=0)
    dist_ref_error = np.divide(
        den_counts,
        ref_sum,
        out=np.zeros_like(dist_ref_normalized, dtype=float),
        where=ref_sum > 0,
    )
    # Relative error used for ratio panel: sigma(p_i) / p_i = 1/sqrt(N_i)
    dist_ref_ratio_error = np.divide(
        dist_ref_error,
        dist_ref_normalized,
        out=np.zeros_like(dist_ref_error),
        where=dist_ref_normalized != 0,
    )

    ax[0].step(
        binning,
        dup(dist_ref_normalized),
        label="Geant4",
        linestyle="-",
        alpha=0.8,
        linewidth=1.0,
        color="k",
        where="post",
    )
    ax[0].fill_between(
        binning,
        dup(dist_ref_normalized + dist_ref_error),
        dup(dist_ref_normalized - dist_ref_error),
        step="post",
        color="k",
        alpha=0.2,
    )
    ax[1].fill_between(
        binning,
        dup(1 - dist_ref_ratio_error),
        dup(1 + dist_ref_ratio_error),
        step="post",
        color="k",
        alpha=0.2,
    )

    # Generator lines
    if generated is not None:
        dist_gen, binning = np.histogram(generated, bins=binning, density=False)
        dist_gen_normalized = dist_gen / dist_gen.sum()
        dist_gen_error = dist_gen_normalized / np.sqrt(dist_gen)
        ratio = dist_gen_normalized / dist_ref_normalized
        ratio_err = dist_gen_error / dist_ref_normalized
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        ax[0].step(
            binning,
            dup(dist_gen_normalized),
            label="Model",
            where="post",
            linewidth=1.0,
            alpha=1.0,
            color=colors[0],
            linestyle="-",
        )
        ax[0].fill_between(
            binning,
            dup(dist_gen_normalized + dist_gen_error),
            dup(dist_gen_normalized - dist_gen_error),
            step="post",
        color=colors[0],
            alpha=0.2,
        )
        ax[1].step(
            binning,
            dup(ratio),
            linewidth=1.0,
            alpha=1.0,
            color=colors[0],
            where="post",
        )
        ax[1].fill_between(
            binning,
            dup(ratio - ratio_err),
            dup(ratio + ratio_err),
            step="post",
            color=colors[0],
            alpha=0.2,
        )
        ax[1].hlines(
            1.0,
            binning[0],
            binning[-1],
            linewidth=1.0,
            alpha=0.8,
            linestyle="-",
            color="k",
        )
        sep_power = utils._separation_power(dist_ref, dist_gen, binning)

    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(binning[0], binning[-1])

    if logy:
        ax[0].set_yscale("log")
    ax[1].axhline(0.7, c="k", ls="--", lw=0.5)
    ax[1].axhline(1.3, c="k", ls="--", lw=0.5)
    ax[0].set_ylabel(ylabel, fontsize=leg_font)
    ax[1].set_xlabel(xlabel, fontsize=leg_font)
    ax[1].set_ylabel(r"$\frac{\text{Model}}{\text{Geant4}}$", fontsize=leg_font)
    ax[0].legend(
        loc=label_loc,
        frameon=False,
        handlelength=1.2,
        title_fontsize=leg_font,
        fontsize=leg_font,
    )
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

    # should probably make them pdfs at some point
    # plt.savefig(fname, dpi=300, format="pdf")
    if len(fname) > 0:
        fig.savefig(fname)
    plt.close(fig)
    return sep_power if generated is not None else 0
