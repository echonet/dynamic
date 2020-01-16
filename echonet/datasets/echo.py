import echonet
import pathlib
import torch.utils.data
import os
import numpy as np
import cv2
import collections
import skimage.draw
import math

class Echo(torch.utils.data.Dataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=4,
                 max_length=250,
                 crops=1,
                 pad=None,
                 noise=None,
                 segmentation=None,
                 target_transform=None,
                 external_test_location = None
                ):
        """length = None means take all possible"""

        if root is None:
            root = echonet.config.DATA_DIR

        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.crops = crops
        self.pad = pad
        self.noise = noise
        self.segmentation = segmentation
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        
        self.fnames, self.outcome = [], []
        
        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        elif split == "clinical_test":
            self.fnames = sorted(os.listdir(self.folder / "ProcessedStrainStudyA4c"))
        else:
            with open(self.folder / "FileList.csv") as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                splitIndex = self.header.index("Split")

                for (i, line) in enumerate(f):
                    lineSplit = line.strip().split(',')

                    fileName = lineSplit[filenameIndex]
                    fileMode = lineSplit[splitIndex].lower()

                    if (split == "all" or split == fileMode) and os.path.exists(self.folder / "Videos" / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(self.folder / "VolumeTracings.csv") as f:
                header = f.readline().strip().split(",")

                for (i, line) in enumerate(f):
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 and f != "0X4F55DC7F6080587E.avi" for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.folder, "Videos", self.fnames[index])
        video = echonet.utils.loadvideo(video)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        assert(type(self.mean) == type(self.std))
        if isinstance(self.mean, int) or isinstance(self.mean, float):
            video = (video - self.mean) / self.std
        else:
            video = (video - self.mean.reshape(3, 1, 1, 1)) / self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        if self.length == None:
            length = f // self.period
        else:
            length = self.length

        length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        if self.crops == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.crops)

        target = []
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t == "LargeTrace" or t == "SmallTrace":
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if self.segmentation is not None:
             seg = np.load(os.path.join(self.segmentation, os.path.splitext(self.fnames[index])[0] + ".npy"))
             video[2, :seg.shape[0], :, :] = seg

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select random crops
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.crops == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target

    def __len__(self):
        return len(self.fnames)

def _defaultdict_of_lists():
    return collections.defaultdict(list)
