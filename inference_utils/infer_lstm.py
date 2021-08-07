import argparse
import os
import pickle
import random
import sys

import torch

sys.path.append(os.getcwd())
from models.CAE_LSTM_seq import CAELSTMSeq

parser = argparse.ArgumentParser()
parser.add_argument('-img_size', dest="img_size", type=str, help='A tuple wrapped as an integer in the format "H,W,C"')
parser.add_argument('-root_folder', dest='root_folder', help='Path to root folder where data and experiments results are located')
parser.add_argument('-trained_encoder_path', dest='trained_encoder_path', help='Path to trained autoencoder checkpoints.')
parser.add_argument('-trained_lstm_path', dest='trained_lstm_path', help='Path to trained lstm checkpoints.')
parser.add_argument('-dataloader_path', dest='dataloader_path', type=str, default=None,
                    help='path to dataloader pkl file. If None, default location will be root_folder/dataloaders/autoencoder/val_dl.pkl')
args = parser.parse_args()


def load_dataset(root_folder, dataloader_path):
    if dataloader_path is None:
        dataloader_path = os.path.join(root_folder, "dataloaders", "lstm", "val_dl.pkl")
    with open(dataloader_path, "rb") as fp:
        pred_dl = pickle.load(fp)
    return pred_dl.dataset


def load_model(trained_lstm_path):
    model = CAELSTMSeq()
    model = model.load_from_checkpoint(trained_lstm_path)
    model.eval()
    return model


def pred_random_sample(val_dataset, model):
    i = random.randint(0, len(val_dataset))
    x, y = val_dataset[i]
    x = [xi.unsqueeze(0) for xi in x]
    with torch.no_grad():
        pred = model(x)
    return y, pred


img_size = tuple([int(i) for i in args.img_size.split(",")])
model = load_model(args.trained_lstm_path)
csv_path = os.path.join(args.root_folder, "raw_data.csv")
val_dataset = load_dataset(args.root_folder, args.dataloader_path)
y, pred = pred_random_sample(val_dataset, model)
print("y", y)
print("pred", pred)
