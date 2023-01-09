
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import AutomaticExtractionDataset, read_data
from module import AgeGaitModule,AgeGaitModel
import torch.utils.data as data
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")


path = "OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter"
label_file = "IDGenderAgelist.csv"
# datasest = AutomaticExtractionDataset(path, label_file)
train_set, valid_set = read_data(path, label_file)
    
print("train size", len(train_set),"valid size", len(valid_set))
train_loader = DataLoader(train_set,batch_size=8,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=8,shuffle=True)

# model
module = AgeGaitModule(AgeGaitModel())

# train model
trainer = pl.Trainer(logger=logger,max_epochs=50)
trainer.fit(model=module, train_dataloaders=train_loader,val_dataloaders=valid_loader)

# autoencoder = LitAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()

# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()