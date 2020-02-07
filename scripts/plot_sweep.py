#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import echonet
import os
import math

root = os.path.join("output", "video")
fig_root = os.path.join("figure", "hyperparameter")

echonet.utils.latexify()

os.makedirs(fig_root, exist_ok=True)

def load(model, frames, period, pretrained):
    try:
        with open(os.path.join(root, "{}_{}_{}_{}".format(model, frames, period, "pretrained" if pretrained else "random"), "log.csv"), "r") as f:
            for l in f:
                if "Best validation loss " in l:
                    return float(l.split()[3])
    except:
        pass

fig = plt.figure(figsize=(1.75, 2.75))

for (model, color) in zip(["EchoNet-Dynamic (EF)", "R3D", "MC3"], matplotlib.colors.TABLEAU_COLORS):
    plt.plot([float("nan")], [float("nan")], "-", color=color, label=model)
plt.plot([float("nan")], [float("nan")], "-", color="k", label="Pretrained")
plt.plot([float("nan")], [float("nan")], "--", color="k", mfc="none", label="Random")
plt.title("")
plt.axis("off")
plt.legend(loc="center")
plt.tight_layout()
plt.savefig(os.path.join(fig_root, "legend.pdf"))
plt.savefig(os.path.join(fig_root, "legend.eps"))
plt.savefig(os.path.join(fig_root, "legend.png"))
plt.close(fig)

print("FRAMES")
FRAMES = [1, 8, 16, 32, 64, 96, None]
MAX = FRAMES[-2]
START = 1
TERM0 = 104
BREAK = 112
TERM1 = 120
ALL = 128
END = 135
RATIO = (BREAK - START) / (END - BREAK)
fig = plt.figure(figsize=(3.0, 2.75))
gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[RATIO, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharey=ax0)
for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
    for pretrained in [True, False]:
        loss = [load(model, frames, 1, pretrained) for frames in FRAMES]
        print(model, pretrained)
        print(list(map(lambda x: "{:.1f}".format(x) if x is not None else None, loss)))

        l0 = loss[-2]
        l1 = loss[-1]
        ax0.plot(FRAMES[:-1] + [TERM0], loss[:-1] + [l0 + (l1 - l0) * (TERM0 - MAX) / (ALL - MAX)], "-" if pretrained else "--",  color=color)
        ax1.plot([TERM1, ALL], [l0 + (l1 - l0) * (TERM1 - MAX) / (ALL - MAX)] + [loss[-1]], "-" if pretrained else "--",  color=color)
        ax0.scatter(list(map(lambda x: x if x is not None else ALL, FRAMES)), loss, color=color, s=4)
        ax1.scatter(list(map(lambda x: x if x is not None else ALL, FRAMES)), loss, color=color, s=4)

ax0.set_xticks(list(map(lambda x: x if x is not None else ALL, FRAMES)))
ax1.set_xticks(list(map(lambda x: x if x is not None else ALL, FRAMES)))
ax0.set_xticklabels(list(map(lambda x: x if x is not None else "All", FRAMES)))
ax1.set_xticklabels(list(map(lambda x: x if x is not None else "All", FRAMES)))

#https://stackoverflow.com/questions/5656798/python-matplotlib-is-there-a-way-to-make-a-discontinuous-axis/43684155
# zoom-in / limit the view to different portions of the data
ax0.set_xlim(START, BREAK) # most of the data
ax1.set_xlim(BREAK, END)
# 
# # hide the spines between ax and ax2
ax0.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.get_yaxis().set_visible(False)
# 

d = 0.015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False, linewidth=1)
x0, x1, y0, y1 = ax0.axis()
scale = (y1 - y0)/ (x1 - x0) / 2
ax0.plot((1 - scale * d,1 + scale * d),(-d,+d), **kwargs) # top-left diagonal
ax0.plot((1 - scale * d,1 + scale * d),(1-d,1+d), **kwargs) # bottom-left diagonal

kwargs.update(transform=ax1.transAxes) # switch to the bottom 1xes
x0, x1, y0, y1 = ax1.axis()
scale = (y1 - y0)/ (x1 - x0) / 2
ax1.plot((-scale * d,scale * d),(-d,+d), **kwargs) # top-right diagonal
ax1.plot((-scale * d,scale * d),(1-d,1+d), **kwargs) # bottom-right diagonal


# ax0.xaxis.label.set_transform(matplotlib.transforms.blended_transform_factory(
#        matplotlib.transforms.IdentityTransform(), fig.transFigure # specify x, y transform
#        )) # changed from default blend (IdentityTransform(), a[0].transAxes)
ax0.xaxis.label.set_position((0.6, 0.0))
ax0.text(0.05, 0.95, "(a)", transform=fig.transFigure)
ax0.set_xlabel("Clip length (frames)")

ax0.set_ylabel("Validation Loss")
gs.tight_layout(fig)
gs.update(wspace=0.020) # set the spacing between axes.
plt.savefig(os.path.join(fig_root, "length.pdf"))
plt.savefig(os.path.join(fig_root, "length.eps"))
plt.savefig(os.path.join(fig_root, "length.png"))
plt.close(fig)

print("PERIOD")
PERIOD = [1, 2, 4, 6, 8]
fig = plt.figure(figsize=(2.5, 2.75))

for (model, color) in zip(["r2plus1d_18", "r3d_18", "mc3_18"], matplotlib.colors.TABLEAU_COLORS):
    for pretrained in [True, False]:
        loss = [load(model, 64 // period, period, pretrained) for period in PERIOD]
        print(model, pretrained)
        print(list(map(lambda x: "{:.1f}".format(x) if x is not None else None, loss)))

        plt.plot(PERIOD, loss, "-" if pretrained else "--",  marker=".", color=color)
plt.xticks(PERIOD)
plt.text(0.05, 0.95, "(b)", transform=fig.transFigure)
plt.xlabel("Sampling Period (frames)")
plt.ylabel("Validation Loss")
plt.tight_layout()
plt.savefig(os.path.join(fig_root, "period.pdf"))
plt.savefig(os.path.join(fig_root, "period.eps"))
plt.savefig(os.path.join(fig_root, "period.png"))
plt.close(fig)

