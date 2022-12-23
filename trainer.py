
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import AutomaticExtractionDataset
from module import AgeGaitModule,AgeGaitModel

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")


path = "OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter"
label_file = "IDGenderAgelist.csv"
dataset = AutomaticExtractionDataset(path, label_file)
train_loader = DataLoader(dataset,batch_size=4)

# model
module = AgeGaitModule(AgeGaitModel())

# train model
trainer = pl.Trainer(logger=logger)
trainer.fit(model=module, train_dataloaders=train_loader)

# autoencoder = LitAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()

# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()