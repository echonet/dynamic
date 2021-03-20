#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 1."""

import os

import matplotlib
import matplotlib.pyplot as plt

import echonet


def main(root=os.path.join("output", "video"),
         fig_root=os.path.join("figure", "hyperparameter"),
         FRAMES=(1, 8, 16, 32, 64, 96, None),
         PERIOD=(1, 2, 4, 6, 8)
         ):
    """Generate plots for Extended Data Fig. 1."""

    echonet.utils.latexify()
    os.makedirs(fig_root, exist_ok=True)

    # Parameters for plotting length sweep
    MAX = FRAMES[-2]
    START = 1    # Starting point for normal range
    TERM0 = 104  # Ending point for normal range
    BREAK = 112  # Location for break
    TERM1 = 120  # Starting point for "all" section
    ALL = 128    # Location of "all" point
    END = 135    # Ending point for "all" section
    RATIO = (BREAK - START) / (END - BREAK)

    # Set up figure
    fig = plt.figure(figsize=(3 + 2.5 + 1.5, 2.75))
    outer = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[3, 2.5, 1.50])
    ax = plt.subplot(outer[2])   # Legend
    ax2 = plt.subplot(outer[1])  # Period plot
    gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], width_ratios=[RATIO, 1], wspace=0.020)  # Length plot

    # Plot legend
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"],
                              matplotlib.colors.TABLEAU_COLORS):
        ax.plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax.plot([float("nan")], [float("nan")], "-", color="k", label="Pretrained")
    ax.plot([float("nan")], [float("nan")], "--", color="k", label="Random")
    ax.set_title("")
    ax.axis("off")
    ax.legend(loc="center")

    # Plot length sweep (panel a)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)
    print("FRAMES")
    for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"],
                              matplotlib.colors.TABLEAU_COLORS):
        for pretrained in [True, False]:
            loss = [load(root, model, frames, 1, pretrained) for frames in FRAMES]
            print(model, pretrained)
            print("    ".join(list(map(lambda x: "{:.1f}".format(x) if x is not None else None, loss))))

            l0 = loss[-2]
            l1 = loss[-1]
            ax0.plot(FRAMES[:-1] + (TERM0,),
                     loss[:-1] + [l0 + (l1 - l0) * (TERM0 - MAX) / (ALL - MAX)],
                     "-" if pretrained else "--", color=color)
            ax1.plot([TERM1, ALL],
                     [l0 + (l1 - l0) * (TERM1 - MAX) / (ALL - MAX)] + [loss[-1]],
                     "-" if pretrained else "--", color=color)
            ax0.scatter(list(map(lambda x: x if x is not None else ALL, FRAMES)), loss, color=color, s=4)
            ax1.scatter(list(map(lambda x: x if x is not None else ALL, FRAMES)), loss, color=color, s=4)

    ax0.set_xticks(list(map(lambda x: x if x is not None else ALL, FRAMES)))
    ax1.set_xticks(list(map(lambda x: x if x is not None else ALL, FRAMES)))
    ax0.set_xticklabels(list(map(lambda x: x if x is not None else "All", FRAMES)))
    ax1.set_xticklabels(list(map(lambda x: x if x is not None else "All", FRAMES)))

    # https://stackoverflow.com/questions/5656798/python-matplotlib-is-there-a-way-to-make-a-discontinuous-axis/43684155
    # zoom-in / limit the view to different portions of the data
    ax0.set_xlim(START, BREAK)  # most of the data
    ax1.set_xlim(BREAK, END)

    # hide the spines between ax and ax2
    ax0.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax1.get_yaxis().set_visible(False)

    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False, linewidth=1)
    x0, x1, y0, y1 = ax0.axis()
    scale = (y1 - y0) / (x1 - x0) / 2
    ax0.plot((1 - scale * d, 1 + scale * d), (-d, +d), **kwargs)  # top-left diagonal
    ax0.plot((1 - scale * d, 1 + scale * d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

    kwargs.update(transform=ax1.transAxes)  # switch to the bottom 1xes
    x0, x1, y0, y1 = ax1.axis()
    scale = (y1 - y0) / (x1 - x0) / 2
    ax1.plot((-scale * d, scale * d), (-d, +d), **kwargs)  # top-right diagonal
    ax1.plot((-scale * d, scale * d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # ax0.xaxis.label.set_transform(matplotlib.transforms.blended_transform_factory(
    #        matplotlib.transforms.IdentityTransform(), fig.transFigure # specify x, y transform
    #        )) # changed from default blend (IdentityTransform(), a[0].transAxes)
    ax0.xaxis.label.set_position((0.6, 0.0))
    ax0.text(-0.05, 1.10, "(a)", transform=ax0.transAxes)
    ax0.set_xlabel("Clip length (frames)")
    ax0.set_ylabel("Validation Loss")

    # Plot period sweep (panel b)
    print("PERIOD")
    for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
        for pretrained in [True, False]:
            loss = [load(root, model, 64 // period, period, pretrained) for period in PERIOD]
            print(model, pretrained)
            print("    ".join(list(map(lambda x: "{:.1f}".format(x) if x is not None else None, loss))))

            ax2.plot(PERIOD, loss, "-" if pretrained else "--", marker=".", color=color)
    ax2.set_xticks(PERIOD)
    ax2.text(-0.05, 1.10, "(b)", transform=ax2.transAxes)
    ax2.set_xlabel("Sampling Period (frames)")
    ax2.set_ylabel("Validation Loss")

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_root, "hyperparameter.pdf"))
    plt.savefig(os.path.join(fig_root, "hyperparameter.eps"))
    plt.savefig(os.path.join(fig_root, "hyperparameter.png"))
    plt.close(fig)


def load(root, model, frames, period, pretrained):
    """Loads best validation loss for specified hyperparameter choice."""
    pretrained = ("pretrained" if pretrained else "random")
    f = os.path.join(
        root,
        "{}_{}_{}_{}".format(model, frames, period, pretrained),
        "log.csv")
    with open(f, "r") as f:
        for line in f:
            if "Best validation loss " in line:
                return float(line.split()[3])

    raise ValueError("File missing information.")


if __name__ == "__main__":
    main()
