
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import AutomaticExtractionDataset
from module import LitAutoEncoder,Encoder

path = "OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter"
label_file = "IDGenderAgelist.csv"
dataset = AutomaticExtractionDataset(path, label_file)
train_loader = DataLoader(dataset,batch_size=4)

# model
autoencoder = LitAutoEncoder(Encoder())

# train model
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# autoencoder = LitAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()

# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()