import argparse
import os
import sys
from typing import Tuple

import pytorch_lightning as pl

from DataHandling.dataloaders_handler import DataloadersHandler
from models.CAE_LSTM_AutoEncoder import CAELSTMAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-img_size', dest="img_size", type=str, help='A tuple wrapped as an integer in the format "H,W,C"')
parser.add_argument('-root_folder', dest='root_folder', help='Path to root folder where data and experiments results are located')


def train_autoencoder(img_size: Tuple[int, int, int], root_folder: str) -> None:
    dataloaders_handler = DataloadersHandler(data_root_folder=root_folder)
    train_loader, val_loader = dataloaders_handler.create_autoencoder_dataloaders(img_size)
    model = CAELSTMAutoEncoder(img_size=img_size)
    trainer = pl.Trainer(gpus=1, default_root_dir=os.path.join(root_folder, "autoencoder_logs"))
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    img_size = tuple([int(i) for i in args.img_size.split(",")])
    train_autoencoder(img_size, args.root_folder)
