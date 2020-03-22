#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 3."""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt

import echonet


def main():
    """Generate plots for Extended Data Fig. 3."""

    # Select paths and hyperparameter to plot
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default="output")
    parser.add_argument("fig", nargs="?", default=os.path.join("figure", "loss"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--period", type=int, default=2)
    args = parser.parse_args()

    # Set up figure
    echonet.utils.latexify()
    os.makedirs(args.fig, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    gs = matplotlib.gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[2.75, 2.75, 1.50])

    # Plot EF loss curve
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    for pretrained in [True]:
        for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
            loss = load(os.path.join(args.dir, "video", "{}_{}_{}_{}".format(model, args.frames, args.period, "pretrained" if pretrained else "random"), "log.csv"))
            ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "-" if pretrained else "--", color=color)
            ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "-" if pretrained else "--", color=color)

    plt.axis([0, max(len(loss["train"]), len(loss["val"])), 0, max(max(loss["train"]), max(loss["val"]))])
    ax0.text(-0.25, 1.00, "(a)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(b)", transform=ax1.transAxes)
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 15, 30, 45])
    ax1.set_xticks([0, 15, 30, 45])
    ax0.set_ylabel("Training MSE Loss")
    ax1.set_ylabel("Validation MSE Loss")

    # Plot segmentation loss curve
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1], sharey=ax0)
    pretrained = False
    for (model, color) in zip(["deeplabv3_resnet50"], list(matplotlib.colors.TABLEAU_COLORS)[3:]):
        loss = load(os.path.join(args.dir, "segmentation", "{}_{}".format(model, "pretrained" if pretrained else "random"), "log.csv"))
        ax0.plot(range(1, 1 + len(loss["train"])), loss["train"], "--", color=color)
        ax1.plot(range(1, 1 + len(loss["val"])), loss["val"], "--", color=color)

    ax0.text(-0.25, 1.00, "(c)", transform=ax0.transAxes)
    ax1.text(-0.25, 1.00, "(d)", transform=ax1.transAxes)
    ax0.set_ylim([0, 0.13])
    ax0.set_xlabel("Epochs")
    ax1.set_xlabel("Epochs")
    ax0.set_xticks([0, 25, 50])
    ax1.set_xticks([0, 25, 50])
    ax0.set_ylabel("Training Cross Entropy Loss")
    ax1.set_ylabel("Validation Cross Entropy Loss")

    # Legend
    ax = fig.add_subplot(gs[:, 2])
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3", "EchoNet-Dynamic (Seg)"], matplotlib.colors.TABLEAU_COLORS):
        ax.plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax.set_title("")
    ax.axis("off")
    ax.legend(loc="center")

    plt.tight_layout()
    plt.savefig(os.path.join(args.fig, "loss.pdf"))
    plt.savefig(os.path.join(args.fig, "loss.eps"))
    plt.savefig(os.path.join(args.fig, "loss.png"))
    plt.close(fig)


def load(filename):
    """Loads losses from specified file."""

    losses = {"train": [], "val": []}
    with open(filename, "r") as f:
        for line in f:
            line = line.split(",")
            if len(line) < 4:
                continue
            epoch, split, loss, *_ = line
            epoch = int(epoch)
            loss = float(loss)
            assert(split in ["train", "val"])
            if epoch == len(losses[split]):
                losses[split].append(loss)
            elif epoch == len(losses[split]) - 1:
                losses[split][-1] = loss
            else:
                raise ValueError("File has uninterpretable formatting.")
    return losses


if __name__ == "__main__":
    main()
