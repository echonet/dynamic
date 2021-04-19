"""Functions for training and running EF prediction."""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet


def run(num_epochs=45,
        modelname="r2plus1d_18",
        tasks="EF",
        frames=32,
        period=2,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        num_workers=5,
        batch_size=20,
        seed=0,
        lr_step_period=15,
        run_test=False):
    """Trains/tests EF prediction model.

    Args:
        num_epochs (int, optional): Number of epochs during training
            Defaults to 45.
        modelname (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<modelname>)
            Defaults to ``r2plus1d_18''.
        tasks (str, optional): Name of task to predict. Options are the headers
            of FileList.csv.
            Defaults to ``EF''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        output (str or None, optional): Name of directory to place outputs
            Defaults to None (replaced by output/video/<modelname>_<pretrained/random>/).
        device (str or None, optional): Name of device to run on. See
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            for options. If ``None'', defaults to ``cuda'' if available, and ``cpu'' otherwise.
            Defaults to ``None''.
        n_train_patients (str or None, optional): Number of training patients. Used to ablations
            on number of training patients. If ``None'', all patients used.
            Defaults to ``None''.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 5.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 20.
        seed (int, optional): Seed for random number generator.
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            If ``None'', learning rate is not decayed.
            Defaults to 15.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    train_dataset = echonet.datasets.Echo(split="train", **kwargs, pad=12)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloaders[phase], phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

        if run_test:
            for split in ["val", "test"]:
                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y
