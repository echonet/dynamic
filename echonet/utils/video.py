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
        modelname="r3d_18",
        tasks="EF",
        frames=16,
        period=4,
        pretrained=True,
        output=None,
        device=None,
        n_train_patients=None,
        seed=0,
        num_workers=5,
        batch_size=20,
        lr_step_period=None,
        run_test=False,
        run_extra_tests=False):

    ### Seed RNGs ###
    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(modelname, frames, period, "pretrained" if pretrained else "random"))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output, exist_ok=True)

    model = torchvision.models.video.__dict__[modelname](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    train_dataset = echonet.datasets.Echo(split="train", **kwargs, pad=12)
    if n_train_patients is not None and len(train_dataset) > n_train_patients:
        indices = np.random.choice(len(train_dataset), n_train_patients, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        echonet.datasets.Echo(split="val", **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
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
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloaders[phase], phase, optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(yhat, y),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()
            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(yhat, y),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

        if run_extra_tests:
            ds = echonet.datasets.Echo(split="nsc", **kwargs, clips="all")
            test_dataloader = torch.utils.data.DataLoader(
                ds, batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
            loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, "test", None, device, save_all=True, blocks=100)

            with open(os.path.join(output, "nsc_predictions.csv"), "w") as g:
                for (filename, pred) in zip(ds.fnames, yhat):
                    for (i, p) in enumerate(pred):
                        g.write("{},{},{:.4f}\n".format(filename, i, p))

            ds = echonet.datasets.Echo(split="clinical_test", **kwargs, clips="all")
            test_dataloader = torch.utils.data.DataLoader(
                ds, batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
            loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, "test", None, device, save_all=True, blocks=100)

            with open(os.path.join(output, "clinical_test_predictions.csv"), "w") as g:
                for (filename, pred) in zip(ds.fnames, yhat):
                    for (i, p) in enumerate(pred):
                        g.write("{},{},{:.4f}\n".format(filename, i, p))

            ds = echonet.datasets.Echo(split="full", **kwargs, clips="all")
            os.makedirs(os.path.join(output, "full"), exist_ok=True)
            for (block, start) in enumerate(range(0, len(ds), 1000)):
                print("Block #{}".format(block), flush=True)
                if not os.path.isfile(os.path.join(output, "full", "full_predictions_{}.csv".format(block))):
                    test_dataloader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(ds, range(start, min(start + 1000, len(ds)))),
                        batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                    loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, "test", None, device, save_all=True, blocks=100)

                    with open(os.path.join(output, "full", "full_predictions_{}.csv".format(block)), "w") as g:
                        for (filename, pred) in zip(ds.fnames, yhat):
                            for (i, p) in enumerate(pred):
                                g.write("{},{},{:.4f}\n".format(filename, i, p))
            with open(os.path.join(output, "full_predictions.csv"), "w") as g:
                for (block, start) in enumerate(range(0, len(ds), 1000)):
                    with open(os.path.join(output, "full", "full_predictions_{}.csv".format(block)), "r") as h:
                        for l in h:
                            g.write(l)

        if run_test:
            for split in ["val", "test"]:
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, split, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                ds = echonet.datasets.Echo(split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, split, None, device, save_all=True, blocks=100)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

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
                # plt.gca().set_axisbelow(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

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


def run_epoch(model, dataloader, phase, optim, device, save_all=False, blocks=None):

    criterion = torch.nn.MSELoss()  # Standard L2 loss

    runningloss = 0.0

    model.train(phase == 'train')

    counter = 0
    summer = 0
    summer_squared = 0

    yhat = []
    y = []

    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                summer += outcome.sum()
                summer_squared += (outcome ** 2).sum()

                if blocks is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + blocks), ...]) for j in range(0, X.shape[0], blocks)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = criterion(outputs.view(-1), outcome)

                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                runningloss += loss.item() * X.size(0)
                counter += X.size(0)

                epoch_loss = runningloss / counter

                # str(i, runningloss, epoch_loss,  str(((summer_squared) / counter - (summer / counter)**2).item()))

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(epoch_loss, loss.item(), summer_squared / counter - (summer / counter) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return epoch_loss, yhat, y
