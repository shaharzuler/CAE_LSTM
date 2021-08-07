import argparse
from typing import Tuple

import cv2
import gym
import numpy as np
import pytorch_lightning as pl
import torch

from DataHandling.dataloaders_handler import DataloadersHandler
from models.CAE_LSTM_AutoEncoder import CAELSTMAutoEncoder
from models.CAE_LSTM_seq import CAELSTMSeq

parser = argparse.ArgumentParser()
parser.add_argument('-img_size', dest="img_size", type=str, help='A tuple wrapped as an integer in the format "H,W,C"')
parser.add_argument('-root_folder', dest='root_folder', help='Path to root folder where data and experiments results are located')
parser.add_argument('-trained_encoder_path', dest='trained_encoder_path', default=None, help='Path to trained autoencoder checkpoints.')
parser.add_argument('-trained_lstm_path', dest='trained_lstm_path', help='Path to trained lstm checkpoints.')

args = parser.parse_args()


class CaeLstmInfer:
    def __init__(self, model: pl.LightningModule, img_size: Tuple[int, int, int], root_folder: str, trained_encoder_path: str, trained_lstm_path: str):
        self.model = model
        self.img_size = img_size
        self.root_folder = root_folder
        self.trained_encoder_path = trained_encoder_path
        self.trained_lstm_path = trained_lstm_path

    def get_x_from_state(self, state: np.ndarray) -> torch.tensor:
        img = self.preprocess_image(state)
        image_features = self.get_image_features(img)
        return image_features

    @staticmethod
    def preprocess_image(img: np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img / 128. - 1.
        img = (img + 1.0) / 2.0
        img = cv2.resize(img.squeeze(), (96, 96))
        img = torch.Tensor(img)
        img = img.unsqueeze(dim=0)
        img = img.unsqueeze(0)
        return img

    def get_image_features(self, img: torch.tensor) -> torch.tensor:
        encoder = self.get_encoder()
        with torch.no_grad():
            image_features = encoder(img)
            return image_features.unsqueeze(0)

    def get_encoder(self) -> pl.LightningModule:
        if self.trained_encoder_path is None:
            self.trained_encoder_path = DataloadersHandler.get_trained_autoencoder_path(self.root_folder)
        encoder = CAELSTMAutoEncoder(img_size=img_size).load_from_checkpoint(self.trained_encoder_path, img_size=img_size)
        encoder.eval()
        return encoder

    def predict(self, state) -> np.ndarray:
        image_features = self.get_x_from_state(state)
        x = [image_features, torch.tensor([[[0.]]]), torch.tensor([[[1.]]]), torch.tensor([[[0.8]]])]
        pred = self.model(x)
        action = pred[0, 0, -3:]
        return action.cpu().numpy()


if __name__ == "__main__":
    img_size = tuple([int(i) for i in args.img_size.split(",")])
    model = CAELSTMSeq().load_from_checkpoint(args.trained_lstm_path).eval()
    cae_lstm_infer = CaeLstmInfer(model, img_size, args.root_folder, args.trained_encoder_path, args.trained_lstm_path)
    env = gym.make('CarRacing-v0')
    with torch.no_grad():
        for i_ep in range(10):
            score = 0
            state = env.reset()
            for t in range(1000):
                action = cae_lstm_infer.predict(state)
                state_, reward, done, die = env.step(action)
                env.render()
                state = state_
                if done or die:
                    break
