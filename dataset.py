import os
import pandas as pd
from torch.utils.data import Dataset

class AutomaticExtractionDataset(Dataset):
    def __init__(self, data_folder, label_file):
        label_file = pd.read_csv(label_file,index_col="ID")
        self.datas= []
        self.labels = []
        self.ids = []
        for file in os.listdir(data_folder):
            if "seq0" in file:
                try:
                    id = file.split('_')[1][2:]
                    self.ids.append(id)
                    self.datas.append(os.path.join(data_folder,file))
                    self.labels.append(label_file["Gender(0:Female;1:Male)"][int(id)])
                except:
                    print(file)
                          
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = pd.read_csv(self.datas[idx],skiprows=2,header=None).to_numpy()
        label = self.labels[idx]
        return sample[:150],label

if __name__ == '__main__':
    path = "OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter"
    label_file = "IDGenderAgelist.csv"
    dataset = AutomaticExtractionDataset(path, label_file)
    print(len(dataset))
    print(dataset[0][0].shape)