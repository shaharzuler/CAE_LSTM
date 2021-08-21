import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class AutoEncoderDataset(Dataset):
    def __init__(self, path, phase, img_size, split_ratio=0.75, shuffle=False):
        self.H, self.W, self.C = img_size
        self.phase: str = phase
        self.path: str = path
        filenames = [name for name in os.listdir(path) if name.endswith((".jpg", ".jpeg", ".png"))]
        if shuffle:
            random.shuffle(filenames)
        num_train = int(split_ratio * len(filenames))
        self.train_filenames: list = filenames[:num_train]
        self.val_filenames: list = filenames[num_train:]

    def __len__(self):
        return len(self.train_filenames) if self.phase=="train" else len(self.val_filenames)

    def __getitem__(self, idx):
        filenames = self.train_filenames if self.phase=="train" else self.val_filenames
        sample_path = os.path.join(self.path, filenames[idx])
        img = cv2.imread(sample_path)
        if img is None:
            raise ValueError("opencv couldn't open image stored in {}".format(sample_path), sample_path)
        if self.C == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.H, self.W))
        img = torch.Tensor(img)
        if self.C == 3:
            img = img.permute(2, 0, 1)
        elif self.C == 1:
            img = img.unsqueeze(dim=0)
        img = img / 255.
        return img
