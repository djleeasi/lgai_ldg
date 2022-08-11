import pandas as pd
from torch.utils.data import Dataset
import torch
import pickle
from .LDGlib import shufflearrays

class ProcessDataset(Dataset):
    def __init__ (self, dataparams, mode):
        with open(dataparams.DATA_DIR_TRAIN, 'rb')as f:
            dataarr= pickle.load(f)
        split = int(len(dataarr[0])*dataparams.TRAINRATIO)
        dataarr = shufflearrays([dataarr[0], dataarr[1]], dataparams.seed)
        if mode == "train":
          self.Xs = dataarr[0][:split]
          self.Ys = dataarr[1][:split]
        else:
          self.Xs = dataarr[0][split:]
          self.Ys = dataarr[1][split:]
        self.Xs[:,1,3] = 0
        self.Xs[:,1,4] = 0
        with open(dataparams.DATA_DIR_MINMAX, 'rb')as f:
            self.minmax= pickle.load(f)
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32), torch.tensor(self.Ys[idx], dtype = torch.float32)



class TestDataset(Dataset):
    def __init__ (self,dataparams):
        with open(dataparams.DATA_DIR_TEST, 'rb')as f:
            dataarr= pickle.load(f)
        self.Xs = dataarr
        self.Xs[:,1,3] = 0
        self.Xs[:,1,4] = 0
        with open(dataparams.DATA_DIR_MINMAX, 'rb')as f:
            self.minmax= pickle.load(f)
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32)


