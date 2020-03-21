#!/usr/bin/env python3

"""Code to generate plots for Extended Data Fig. 6."""

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sklearn
import torch
import torchvision

import echonet


def main(fig_root=os.path.join("figure", "noise"),
         video_output=os.path.join("output", "video", "r2plus1d_18_32_2_pretrained"),
         seg_output=os.path.join("output", "segmentation", "deeplabv3_resnet50_random"),
         NOISE=(0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Generate plots for Extended Data Fig. 6."""

    device = torch.device("cuda")

    filename = os.path.join(fig_root, "data.pkl")  # Cache of results
    try:
        # Attempt to load cache
        with open(filename, "rb") as f:
            Y, YHAT, INTER, UNION = pickle.load(f)
    except FileNotFoundError:
        # Generate results if no cache available
        os.makedirs(fig_root, exist_ok=True)

        # Load trained video model
        model_v = torchvision.models.video.r2plus1d_18()
        model_v.fc = torch.nn.Linear(model_v.fc.in_features, 1)
        if device.type == "cuda":
            model_v = torch.nn.DataParallel(model_v)
        model_v.to(device)

        checkpoint = torch.load(os.path.join(video_output, "checkpoint.pt"))
        model_v.load_state_dict(checkpoint['state_dict'])

        # Load trained segmentation model
        model_s = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)
        model_s.classifier[-1] = torch.nn.Conv2d(model_s.classifier[-1].in_channels, 1, kernel_size=model_s.classifier[-1].kernel_size)
        if device.type == "cuda":
            model_s = torch.nn.DataParallel(model_s)
        model_s.to(device)

        checkpoint = torch.load(os.path.join(seg_output, "checkpoint.pt"))
        model_s.load_state_dict(checkpoint['state_dict'])

        # Run simulation
        dice = []
        mse = []
        r2 = []
        Y = []
        YHAT = []
        INTER = []
        UNION = []
        for noise in NOISE:
            Y.append([])
            YHAT.append([])
            INTER.append([])
            UNION.append([])

            dataset = echonet.datasets.Echo(split="test", noise=noise)
            PIL.Image.fromarray(dataset[0][0][:, 0, :, :].astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(fig_root, "noise_{}.tif".format(round(100 * noise))))

            mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

            tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
            kwargs = {
                "target_type": tasks,
                "mean": mean,
                "std": std,
                "noise": noise
            }
            dataset = echonet.datasets.Echo(split="test", **kwargs)

            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=16, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"))

            loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model_s, dataloader, "test", None, device)
            inter = np.concatenate((large_inter, small_inter)).sum()
            union = np.concatenate((large_union, small_union)).sum()
            dice.append(2 * inter / (union + inter))

            INTER[-1].extend(large_inter.tolist() + small_inter.tolist())
            UNION[-1].extend(large_union.tolist() + small_union.tolist())

            kwargs = {"target_type": "EF",
                      "mean": mean,
                      "std": std,
                      "length": 32,
                      "period": 2,
                      "noise": noise
                      }

            dataset = echonet.datasets.Echo(split="test", **kwargs)

            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=16, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda"))
            loss, yhat, y = echonet.utils.video.run_epoch(model_v, dataloader, "test", None, device)
            mse.append(loss)
            r2.append(sklearn.metrics.r2_score(y, yhat))
            Y[-1].extend(y.tolist())
            YHAT[-1].extend(yhat.tolist())

        # Save results in cache
        with open(filename, "wb") as f:
            pickle.dump((Y, YHAT, INTER, UNION), f)

    # Set up plot
    echonet.utils.latexify()

    NOISE = list(map(lambda x: round(100 * x), NOISE))
    fig = plt.figure(figsize=(6.50, 4.75))
    gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[2.0, 2.0, 0.75])
    ax = (plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2]))

    # Plot EF prediction results (R^2)
    r2 = [sklearn.metrics.r2_score(y, yhat) for (y, yhat) in zip(Y, YHAT)]
    ax[0].plot(NOISE, r2, color="k", linewidth=1, marker=".")
    ax[0].set_xticks([])
    ax[0].set_ylabel("R$^2$")
    l, h = min(r2), max(r2)
    l, h = l - 0.1 * (h - l), h + 0.1 * (h - l)
    ax[0].axis([min(NOISE) - 5, max(NOISE) + 5, 0, 1])

    # Plot segmentation results (DSC)
    dice = [echonet.utils.dice_similarity_coefficient(inter, union) for (inter, union) in zip(INTER, UNION)]
    ax[1].plot(NOISE, dice, color="k", linewidth=1, marker=".")
    ax[1].set_xlabel("Pixels Removed (%)")
    ax[1].set_ylabel("DSC")
    l, h = min(dice), max(dice)
    l, h = l - 0.1 * (h - l), h + 0.1 * (h - l)
    ax[1].axis([min(NOISE) - 5, max(NOISE) + 5, 0, 1])

    # Add example images below
    for noise in NOISE:
        image = matplotlib.image.imread(os.path.join(fig_root, "noise_{}.tif".format(noise)))
        imagebox = matplotlib.offsetbox.OffsetImage(image, zoom=0.4)
        ab = matplotlib.offsetbox.AnnotationBbox(imagebox, (noise, 0.0), frameon=False)
        ax[2].add_artist(ab)
        ax[2].axis("off")
    ax[2].axis([min(NOISE) - 5, max(NOISE) + 5, -1, 1])

    fig.tight_layout()
    plt.savefig(os.path.join(fig_root, "noise.pdf"), dpi=1200)
    plt.savefig(os.path.join(fig_root, "noise.eps"), dpi=300)
    plt.savefig(os.path.join(fig_root, "noise.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
