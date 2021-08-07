from typing import Tuple, List

import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from models.CAE_LSTM_AutoEncoder import CAELSTMAutoEncoder


class LSTMDataset(Dataset):
    def __init__(self, csv_path: str, seq_length: int, phase: str, img_size: Tuple[int, int, int], trained_encoder_path: str):
        self.H, self.W, self.C = img_size
        self.csv_path = csv_path
        self.seq_length = seq_length

        self.raw_data = pd.read_csv(csv_path)
        self.raw_data = self.split_dataset(phase)
        self.raw_numeric_data = self.raw_data[["steer", "gas", "break"]]
        self.image_paths = self.raw_data[["img_path"]]

        self.num_rows = len(self.raw_data.index)

        self.encoder = self.get_encoder(trained_encoder_path)

    def get_encoder(self, trained_encoder_path: str) -> pl.LightningModule:
        encoder = CAELSTMAutoEncoder(img_size=(self.H, self.W, self.C)).load_from_checkpoint(trained_encoder_path, img_size=(self.H, self.W, self.C))
        encoder.eval()
        return encoder

    def split_dataset(self, phase: str) -> pd.DataFrame:
        num_raw_rows = len(self.raw_data.index)
        num_train_rows = int(num_raw_rows * 0.7)
        if phase == "train":
            raw_data = self.raw_data[:num_train_rows]
        elif phase == "val":
            raw_data = self.raw_data[num_train_rows:]
        return raw_data

    def __len__(self) -> int:
        return self.num_rows - self.seq_length - 1

    def __getitem__(self, idx: int) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        img_features_x = self.get_x_image_features(idx)
        state_feat_x = torch.tensor(self.raw_numeric_data[idx:idx + self.seq_length].values)
        x = [img_features_x, state_feat_x[:, 0].unsqueeze(1), state_feat_x[:, 1].unsqueeze(1), state_feat_x[:, 2].unsqueeze(1)]

        img_features_y = self.get_x_image_features(idx + 1)
        state_feat_y = torch.tensor(self.raw_numeric_data[idx + 1:idx + self.seq_length + 1].values)  # torch.tensor(self.raw_numeric_data.iloc[[idx + self.seq_length]].values)
        y = [img_features_y, state_feat_y[:, 0].unsqueeze(1), state_feat_y[:, 1].unsqueeze(1), state_feat_y[:, 2].unsqueeze(1)]
        return x, y

    def get_x_image_features(self, idx: int) -> torch.tensor:
        img_features_x = torch.zeros(self.seq_length, 32)
        for n, i in enumerate(range(idx, idx + self.seq_length)):
            path = self.image_paths.iloc[[i]].values[0][0]
            img = self.get_preprocess_image(path)
            embedding = self.get_image_features(img)
            img_features_x[n, :] = embedding
        return img_features_x

    def get_preprocess_image(self, path: str) -> torch.tensor:
        img = cv2.imread(path)
        if self.C == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.H, self.W))
        img = torch.Tensor(img)
        if self.C == 3:
            img = img.permute(2, 0, 1)
        elif self.C == 1:
            img = img.unsqueeze(dim=0)
        img = img / 255.
        img = img.unsqueeze(0)
        return img

    def get_image_features(self, img: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            image_features = self.encoder(img)
            return image_features.squeeze()
