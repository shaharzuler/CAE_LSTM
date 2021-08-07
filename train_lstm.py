import argparse
import os
from typing import Tuple

import pytorch_lightning as pl

from DataHandling.dataloaders_handler import DataloadersHandler
from models.CAE_LSTM_seq import CAELSTMSeq

parser = argparse.ArgumentParser()
parser.add_argument('-img_size', dest="img_size", type=str, help='A tuple wrapped as an integer in the format "H,W,C"')
parser.add_argument('-data_root_folder', dest='data_root_folder', help='Path to root folder where data and experiments results are located')
parser.add_argument('-trained_encoder_path', dest='trained_encoder_path', default=None,
                    help='Path to trained autoencoder checkpoints. If not specified, most updated checkpoints in the root dir will be chosen')


def train_lstm(img_size: Tuple[int, int, int], data_root_folder: str, trained_encoder_path: str) -> None:
    dataloaders_handler = DataloadersHandler(data_root_folder=data_root_folder)
    train_loader, val_loader = dataloaders_handler.create_lstm_dataloaders(img_size, trained_encoder_path)
    model = CAELSTMSeq()
    trainer = pl.Trainer(gpus=1, default_root_dir=os.path.join(data_root_folder, "lstm_logs"))
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    img_size = tuple([int(i) for i in args.img_size.split(",")])
    train_lstm(img_size, args.data_root_folder, args.trained_encoder_path)
