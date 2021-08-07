from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class CAELSTMSeq(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.img_emb_dim: int = 35
        self.other_states_dim: List[int] = [0]
        self.input_size: int = self.img_emb_dim + sum(np.array(self.other_states_dim))
        self.hidden_size = 250
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=1)
        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

    def forward(self, x: List[torch.tensor], activations=(None, torch.tanh, torch.sigmoid, torch.sigmoid)):
        sizes = [xi.shape[2] for xi in x]

        x = torch.cat(x, dim=2).float()
        x, (h_n, c_n) = self.lstm(x)  # input should be shape: batch, length, input_size
        pred = self.fc(x)

        for i in range(len(sizes)):
            if activations[i] is None:
                pass
            else:
                pred[:, :, int(np.sum(sizes[:i])):np.sum(sizes[:i + 1])] = activations[i](pred[:, :, int(np.sum(sizes[:i])):np.sum(sizes[:i + 1])])

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self.general_step(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.general_step(val_batch)
        self.log('val_loss', loss)

    def general_step(self, batch):
        x, y = batch
        y = torch.cat(y, dim=2).float().squeeze(1)
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss
