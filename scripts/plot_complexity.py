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

fig = plt.figure(figsize=(1.75, 2.75))
for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], matplotlib.colors.TABLEAU_COLORS):
    plt.plot([float("nan")], [float("nan")], "-", color=color, label=model)
plt.title("")
plt.axis("off")
plt.legend(loc="center")
plt.tight_layout()
plt.savefig(os.path.join(fig_root, "legend.pdf"))
plt.savefig(os.path.join(fig_root, "legend.png"))
plt.close(fig)

FRAMES = [1, 8, 16, 32, 64, 96]
fig = [plt.figure(figsize=(2.5, 2.75)) for _ in range(3)]
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

        plt.figure(fig[0].number)
        plt.plot(FRAMES, time / n, "-" if pretrained else "--",  marker=".", color=color, linewidth=(1 if split == "train" else None))
        print("\n".join(map(lambda x: "{:2d}: {:f}".format(*x), zip(FRAMES, time / n))))
        print()

        plt.figure(fig[1].number)
        plt.plot(FRAMES, mem_allocated / batch_size / 1e9, "-" if pretrained else "--",  marker=".", color=color, linewidth=(1 if split == "train" else None))
        print("\n".join(map(lambda x: "{:2d}: {:f}".format(*x), zip(FRAMES, mem_allocated / batch_size / 1e9))))
        print()

plt.figure(fig[0].number)
plt.xticks(FRAMES)
plt.text(0.05, 0.95, "(a)", transform=fig[0].transFigure)
plt.xlabel("Clip length (frames)")
plt.ylabel("Time Per Clip (seconds)")
plt.tight_layout()
plt.savefig(os.path.join(fig_root, "time.pdf"))
plt.savefig(os.path.join(fig_root, "time.png"))

plt.figure(fig[1].number)
plt.xticks(FRAMES)
plt.text(0.05, 0.95, "(b)", transform=fig[1].transFigure)
plt.xlabel("Clip length (frames)")
plt.ylabel("Memory Per Clip (GB)")
plt.tight_layout()
plt.savefig(os.path.join(fig_root, "mem.pdf"))
plt.savefig(os.path.join(fig_root, "mem.png"))

for f in fig:
    plt.close(f)
