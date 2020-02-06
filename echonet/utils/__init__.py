import torch
import tqdm
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('agg')

from . import video
from . import segmentation


def loadvideo(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    v = np.zeros((frame_count, frame_width, frame_height, 3), np.float32)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v


def savevideo(filename, array, fps=1):
    c, f, height, width = array.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(f):
        out.write(array[:, i, :, :].transpose((1, 2, 0)))


def get_mean_and_std(dataset, samples=10):
    if len(dataset) > samples:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    n = 0
    mean = 0.
    std = 0.
    for (i, (x, *_)) in enumerate(tqdm.tqdm(dataloader)):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        mean += torch.sum(x, dim=1).numpy()
        std += torch.sum(x ** 2, dim=1).numpy()
    mean /= n
    std = np.sqrt(std / n - mean ** 2)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def bootstrap(a, b, func, samples=10000):
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for i in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


# Based on https://nipunbatra.github.io/blog/2014/latexify.html
def latexify():
    params = {'backend': 'pdf',
              'axes.titlesize':  8,
              'axes.labelsize':  8,
              'font.size':       8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    return 2 * sum(inter) / (sum(union) + sum(inter))


__all__ = [video, segmentation, loadvideo, savevideo, get_mean_and_std, bootstrap, latexify, dice_similarity_coefficient]
