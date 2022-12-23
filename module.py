import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6,20,5)
        self.lstm = nn.LSTM(input_size=20, hidden_size=50,batch_first=True)
        self.linear = nn.Linear(in_features=50, out_features=1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = x.permute(0,2,1)
        # x = x.reshape(0,2,1)
        x = self.lstm(x)[1][0]
        x = torch.sigmoid(self.linear(x))
        x = x.squeeze()
        # print(x.shape)
        return x

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.float()
        y = y.float()
        # print(x.shape, y.shape)
        y_hat = self.model(x)
        # y_hat = torch.squeeze(y_hat).long()
        loss = F.binary_cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer