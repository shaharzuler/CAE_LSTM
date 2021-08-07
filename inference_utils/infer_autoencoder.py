import argparse
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.getcwd())
from models.CAE_LSTM_AutoEncoder import CAELSTMAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-img_size', dest="img_size", type=str, help='A tuple wrapped as an integer in the format "H,W,C"')
parser.add_argument('-data_root_folder', dest='data_root_folder', help='Path to root folder where data and experiments results are located')
parser.add_argument('-trained_encoder_path', dest='trained_encoder_path', default=None,
                    help='Path to trained autoencoder checkpoints.')
parser.add_argument('-dataloader_path', dest='dataloader_path', type=str, default=None,
                    help='path to dataloader pkl file. If None, default location will be data_root_folder/dataloaders/autoencoder/val_dl.pkl')

args = parser.parse_args()

def load_dataset(data_root_folder, dataloader_path):
    if dataloader_path is None:
        dataloader_path = os.path.join(data_root_folder, "dataloaders", "autoencoder", "val_dl.pkl")
    with open(dataloader_path, "rb") as fp:
        pred_dl = pickle.load(fp)
    return pred_dl.dataset

def load_model(img_size, trained_encoder_path):
    model = CAELSTMAutoEncoder(img_size=img_size)
    model = model.load_from_checkpoint(trained_encoder_path, img_size=img_size)
    model.debug = True
    model.eval()
    return model


def encode_random_images(num_images, val_dataset, model):
    imgs = []
    for j in range(num_images):
        i = random.randint(0, len(val_dataset)-1)
        input_img = val_dataset[i].unsqueeze(0)
        with torch.no_grad():
            loss, embedding, x_hat = model(input_img)
        print(loss, embedding.squeeze())
        imgs.append(input_img.squeeze(dim=1).permute(1, 2, 0))
        imgs.append(x_hat.squeeze(dim=1).permute(1, 2, 0))
    return imgs


def plot(num_images, imgs):
    fig = plt.figure()
    columns = 2
    rows = num_images
    for i, img in zip(range(1, columns * rows + 1), imgs):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()





img_size = tuple([int(i) for i in args.img_size.split(",")])
val_dataset = load_dataset(args.data_root_folder, args.dataloader_path)
model = load_model(img_size, args.trained_encoder_path)
num_images = 3
imgs = encode_random_images(num_images, val_dataset, model)
plot(num_images, imgs)
