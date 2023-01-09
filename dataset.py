import os
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.utils.data as data

def read_data(data_folder,label_file,test_rate=0.2):
    label_file = pd.read_csv(label_file,index_col="ID")
    ids = [file.split('_')[1][2:] for file in os.listdir(data_folder)]
    print("number people:",len(ids))
    # use 20% of training data for validation
    train_set_size = int(len(ids) * 0.8)
    valid_set_size = len(ids) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set_id, valid_set_id = data.random_split(ids, [train_set_size, valid_set_size], generator=seed)
    train_set = [file for file in os.listdir(data_folder) if file.split('_')[1][2:] in train_set_id ]
    valid_set = [file for file in os.listdir(data_folder) if file.split('_')[1][2:] in valid_set_id ]
    print("train set size:",len(train_set),len(train_set_id))
    print("valid set size:",len(valid_set),len(valid_set_id))
    return AutomaticExtractionDataset(data_folder,train_set,label_file), AutomaticExtractionDataset(data_folder,valid_set,label_file) 


class AutomaticExtractionDataset(Dataset):
    def __init__(self, data_folder,file_list, label_file):
        
        self.datas= []
        self.labels = []
        self.ids = []
        for file in file_list:
            if 1:#"seq1" in file:
                try:
                    id = file.split('_')[1][2:]
                    self.ids.append(id)
                    seq = self.read_seq(os.path.join(data_folder,file))
                    # print(len(seq),'xxxxxxxx')
                    for i in range(len(seq)//50-6):
                        # print(i*50,(i+6)*50)
                        self.datas.append(seq[i*50:(i+6)*50])
                        self.labels.append(label_file["Gender(0:Female;1:Male)"][int(id)])
                except Exception as e:
                    # traceback.print_exc()
                    print(file,e)
                          
    def __len__(self):
        return len(self.labels)

    def read_seq(self,file):
        return pd.read_csv(file,skiprows=2,header=None).to_numpy()
        
    def __getitem__(self, idx):
        sample = self.datas[idx]
        # sample2 = pd.read_csv(self.datas[idx].replace("seq0","seq1"),skiprows=2,header=None).to_numpy()
        # print(sample.shape,1)
        sample = np.fft.fft(sample,n=100,axis=0)
        # sample2 = np.fft.fft(sample2,n=64,axis=0)
        # sample = np.concatenate((sample,sample2),axis=1)
        # print(sample.shape,2)
        label = self.labels[idx]
        return sample,label

if __name__ == '__main__':
    path = "OU-IneritialGaitData/AutomaticExtractionData_IMUZCenter"
    label_file = "IDGenderAgelist.csv"
    read_data(path, label_file)
    