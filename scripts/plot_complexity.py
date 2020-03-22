#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 4."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import echonet


def main(root=os.path.join("timing", "video"),
         fig_root=os.path.join("figure", "complexity"),
         FRAMES=(1, 8, 16, 32, 64, 96),
         pretrained=True):
    """Generate plots for Extended Data Fig. 4."""

    echonet.utils.latexify()

    os.makedirs(fig_root, exist_ok=True)
    fig = plt.figure(figsize=(6.50, 2.50))
    gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[2.5, 2.5, 1.50])
    ax = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))

    # Create legend
    for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], matplotlib.colors.TABLEAU_COLORS):
        ax[2].plot([float("nan")], [float("nan")], "-", color=color, label=model)
    ax[2].set_title("")
    ax[2].axis("off")
    ax[2].legend(loc="center")

    for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
        for split in ["val"]:  # ["val", "train"]:
            print(model, split)
            data = [load(root, model, frames, 1, pretrained, split) for frames in FRAMES]
            time = np.array(list(map(lambda x: x[0], data)))
            n = np.array(list(map(lambda x: x[1], data)))
            mem_allocated = np.array(list(map(lambda x: x[2], data)))
            # mem_cached = np.array(list(map(lambda x: x[3], data)))
            batch_size = np.array(list(map(lambda x: x[4], data)))

            # Plot Time (panel a)
            ax[0].plot(FRAMES, time / n, "-" if pretrained else "--", marker=".", color=color, linewidth=(1 if split == "train" else None))
            print("Time:\n" + "\n".join(map(lambda x: "{:8d}: {:f}".format(*x), zip(FRAMES, time / n))))

            # Plot Memory (panel b)
            ax[1].plot(FRAMES, mem_allocated / batch_size / 1e9, "-" if pretrained else "--", marker=".", color=color, linewidth=(1 if split == "train" else None))
            print("Memory:\n" + "\n".join(map(lambda x: "{:8d}: {:f}".format(*x), zip(FRAMES, mem_allocated / batch_size / 1e9))))
            print()

    # Labels for panel a
    ax[0].set_xticks(FRAMES)
    ax[0].text(-0.05, 1.10, "(a)", transform=ax[0].transAxes)
    ax[0].set_xlabel("Clip length (frames)")
    ax[0].set_ylabel("Time Per Clip (seconds)")

    # Labels for panel b
    ax[1].set_xticks(FRAMES)
    ax[1].text(-0.05, 1.10, "(b)", transform=ax[1].transAxes)
    ax[1].set_xlabel("Clip length (frames)")
    ax[1].set_ylabel("Memory Per Clip (GB)")

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_root, "complexity.pdf"))
    plt.savefig(os.path.join(fig_root, "complexity.eps"))
    plt.close(fig)


def load(root, model, frames, period, pretrained, split):
    """Loads runtime and memory usage for specified hyperparameter choice."""
    with open(os.path.join(root, "{}_{}_{}_{}".format(model, frames, period, "pretrained" if pretrained else "random"), "log.csv"), "r") as f:
        for line in f:
            line = line.split(",")
            if len(line) < 4:
                # Skip lines that are not csv (these lines log information)
                continue
            if line[1] == split:
                *_, time, n, mem_allocated, mem_cached, batch_size = line
                time = float(time)
                n = int(n)
                mem_allocated = int(mem_allocated)
                mem_cached = int(mem_cached)
                batch_size = int(batch_size)
                return time, n, mem_allocated, mem_cached, batch_size
    raise ValueError("File missing information.")


if __name__ == "__main__":
    main()
