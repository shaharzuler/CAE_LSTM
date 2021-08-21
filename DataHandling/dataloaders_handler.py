import os
import pickle
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from DataHandling.autoencoder_dataset import AutoEncoderDataset
from DataHandling.lstm_dataset import LSTMDataset


class DataloadersHandler:
    def __init__(self, data_root_folder: str):
        self.configure_paths(data_root_folder)

    def configure_paths(self, data_root_folder: str) -> None:
        self.data_root_folder: str = data_root_folder
        self.dataloaders_path: str = os.path.join(data_root_folder, "dataloaders")
        self.autoencoder_dataloader_path: str = os.path.join(self.dataloaders_path, "autoencoder")
        self.seq_dataloader_path: str = os.path.join(self.dataloaders_path, "lstm")
        self.csv_path: str = os.path.join(data_root_folder, "raw_data.csv")

    def create_lstm_dataloaders(self, img_size: Tuple[int, int, int], trained_encoder_path: str) -> Tuple[DataLoader, DataLoader]:  # img_size should be of format (H,W,C)
        if trained_encoder_path is None:
            trained_encoder_path = self.get_trained_autoencoder_path(self.data_root_folder)
        train_dataset = LSTMDataset(self.csv_path, 4, "train", img_size=img_size, trained_encoder_path=trained_encoder_path)
        val_dataset = LSTMDataset(self.csv_path, 4, "val", img_size=img_size, trained_encoder_path=trained_encoder_path)

        train_loader = DataLoader(train_dataset, batch_size=8, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers=8)

        self.save_dataloaders(self.seq_dataloader_path, train_loader, val_loader)
        return train_loader, val_loader

    @staticmethod
    def get_trained_autoencoder_path(data_root_folder) -> str:
        autoencoder_training_logs_path = os.path.join(data_root_folder, "autoencoder_logs", "lightning_logs")
        version_names = os.listdir(autoencoder_training_logs_path)
        version_numbers = [int(name.split("_")[-1]) for name in version_names]
        latest_num = np.argmax(version_numbers)
        latest_version = "version_{}".format(str(latest_num))
        checkpoints_path = os.path.join(autoencoder_training_logs_path, latest_version, "checkpoints")
        checkpoints_file_name = os.listdir(checkpoints_path)[0]  # assuming one exists because pl deletes old ckpts
        return os.path.join(checkpoints_path, checkpoints_file_name)

    def create_autoencoder_dataloaders(self, img_size: Tuple[int, int, int]) -> Tuple[DataLoader, DataLoader]:  # img_size should be of format (H,W,C)
        train_dataset = AutoEncoderDataset(self.data_root_folder, phase="train", img_size=img_size)
        val_dataset = AutoEncoderDataset(self.data_root_folder, phase="val", img_size=img_size)

        train_loader = DataLoader(train_dataset, batch_size=24)
        val_loader = DataLoader(val_dataset, batch_size=24)

        self.save_dataloaders(self.autoencoder_dataloader_path, train_loader, val_loader)
        return train_loader, val_loader

    def save_dataloaders(self, dataloader_path: str, train_loader: DataLoader, val_loader: DataLoader) -> None:
        os.makedirs(dataloader_path, exist_ok=True)
        train_dl_path = os.path.join(dataloader_path, "train_dl.pkl")
        val_dl_path = os.path.join(dataloader_path, "val_dl.pkl")

        with open(train_dl_path, "wb") as fp:
            pickle.dump(train_loader, fp)
        with open(val_dl_path, "wb") as fp:
            pickle.dump(val_loader, fp)
