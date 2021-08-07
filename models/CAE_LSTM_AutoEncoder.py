import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class CAELSTMAutoEncoder(pl.LightningModule):
    def __init__(self, img_size, debug=False):
        super().__init__()
        self.debug = debug
        self.H, self.W, self.C = img_size
        self.conv1 = torch.nn.Conv2d(in_channels=self.C, out_channels=32, kernel_size=5, padding=2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(in_features=int(self.H / 8 * self.W / 8) * 128, out_features=1024)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=32)

        self.fc3 = torch.nn.Linear(in_features=32, out_features=1024)
        self.fc4 = torch.nn.Linear(in_features=1024, out_features=int(self.H / 8 * self.W / 8 * 128))
        self.max_unpool2d = torch.nn.MaxUnpool2d(kernel_size=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=self.C, kernel_size=5, padding=2)

    def encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, i1 = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, i2 = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x, i3 = self.max_pool(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x, (i1, i2, i3)

    def decoder(self, x, indices):
        i1, i2, i3 = indices
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = x.view(-1, 128, int(self.H / 8), int(self.W / 8))
        x = self.max_unpool2d(x, i3)
        x = self.conv_transpose1(x)
        x = F.relu(x)
        x = self.max_unpool2d(x, i2)
        x = self.conv_transpose2(x)
        x = F.relu(x)
        x = self.max_unpool2d(x, i1)
        x = self.conv_transpose3(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        if self.debug:
            loss, embedding, x_hat = self.general_step(x)
            return loss, embedding, x_hat
        else:
            embedding, _ = self.encoder(x)
            return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def general_step(self, batch):
        x = batch
        z, indices = self.encoder(x)
        x_hat = self.decoder(z, indices)
        loss = F.mse_loss(x_hat, x)
        return loss, z, x_hat

    def training_step(self, train_batch, batch_idx):
        loss, _, _ = self.general_step(train_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, _, _ = self.general_step(val_batch)
        self.log('val_loss', loss)
        return loss
