#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import echonet
import os
import numpy as np

root = os.path.join("timing", "video")
fig_root = os.path.join("figure", "complexity")

echonet.utils.latexify()

os.makedirs(fig_root, exist_ok=True)

def load(model, frames, period, pretrained, split):
    with open(os.path.join(root, "{}_{}_{}_{}".format(model, frames, period, "pretrained" if pretrained else "random"), "log.csv"), "r") as f:
        for l in f:
            l = l.split(",")
            if len(l) < 4:
                continue
            if l[1] == split:
                *_, time, n, mem_allocated, mem_cached, batch_size = l
                time = float(time)
                n = int(n)
                mem_allocated = int(mem_allocated)
                mem_cached = int(mem_cached)
                batch_size = int(batch_size)
                return time, n, mem_allocated, mem_cached, batch_size

fig = plt.figure(figsize=(6.50, 2.50))
gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[2.5, 2.5, 1.50])
ax = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))

for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], matplotlib.colors.TABLEAU_COLORS):
    ax[2].plot([float("nan")], [float("nan")], "-", color=color, label=model)
ax[2].set_title("")
ax[2].axis("off")
ax[2].legend(loc="center")

FRAMES = [1, 8, 16, 32, 64, 96]
pretrained = True
for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
    # for split in ["val", "train"]:
    for split in ["val"]:
        print(model)
        print(split)
        print()
        data = [load(model, frames, 1, pretrained, split) for frames in FRAMES]
        time = np.array(list(map(lambda x: x[0], data)))
        n = np.array(list(map(lambda x: x[1], data)))
        mem_allocated = np.array(list(map(lambda x: x[2], data)))
        mem_cached = np.array(list(map(lambda x: x[3], data)))
        batch_size = np.array(list(map(lambda x: x[4], data)))

        ax[0].plot(FRAMES, time / n, "-" if pretrained else "--",  marker=".", color=color, linewidth=(1 if split == "train" else None))
        print("\n".join(map(lambda x: "{:2d}: {:f}".format(*x), zip(FRAMES, time / n))))
        print()

        ax[1].plot(FRAMES, mem_allocated / batch_size / 1e9, "-" if pretrained else "--",  marker=".", color=color, linewidth=(1 if split == "train" else None))
        print("\n".join(map(lambda x: "{:2d}: {:f}".format(*x), zip(FRAMES, mem_allocated / batch_size / 1e9))))
        print()

ax[0].set_xticks(FRAMES)
ax[0].text(-0.05, 1.10, "(a)", transform=ax[0].transAxes)
ax[0].set_xlabel("Clip length (frames)")
ax[0].set_ylabel("Time Per Clip (seconds)")

ax[1].set_xticks(FRAMES)
ax[1].text(-0.05, 1.10, "(b)", transform=ax[1].transAxes)
ax[1].set_xlabel("Clip length (frames)")
ax[1].set_ylabel("Memory Per Clip (GB)")

plt.tight_layout()
plt.savefig(os.path.join(fig_root, "complexity.pdf"))
plt.savefig(os.path.join(fig_root, "complexity.eps"))
plt.close(fig)
