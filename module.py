import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class AgeGaitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6,20,3,padding="same")
        # self.conv2 = nn.Conv1d(50,50,3,padding="same")
        # self.lstm = nn.LSTM(input_size=20, hidden_size=30,batch_first=True)
        self.Flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=2000, out_features=200)
        # self.linear = nn.Linear(in_features=1280, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = x.permute(0,2,1)
        x = self.Flatten(x)
        # print(x.shape)
        # x = x.reshape(0,2,1)
        # out, _ = self.lstm(x)
        x = torch.relu(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        # print(x.shape,"xxx")
        # x = self.linear(x)
        x = x.squeeze(dim=1)
        # print(x.shape)
        return x

class AgeGaitModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.valid_acc = torchmetrics.Accuracy(task='binary')
        self.model = model
        
    def training_epoch_end(self, outputs):
        self.train_acc.reset()
        self.valid_acc.reset()
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.float()
        y = y.float()
        # print(x.shape, y.shape)
        y_hat = self.model(x)
        # y_hat = torch.squeeze(y_hat).long()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc(y_hat,y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        # print(x.shape, y.shape)
        y_hat = self.model(x)
        # print(y_hat,y)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_acc", self.valid_acc(y_hat,y),on_step=False,on_epoch=True,prog_bar=True)
                                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-04)
        return optimizer