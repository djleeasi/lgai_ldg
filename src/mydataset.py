import pandas as pd
from torch.utils.data import Dataset
import torch
import pickle

class ProcessDataset(Dataset):
    def __init__ (self, x, y, mode = True):
        self.Xs = x
        self.Ys = y
        if mode:    
            self.Xs[:,1,3] = 0
            self.Xs[:,1,4] = 0
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32), torch.tensor(self.Ys[idx], dtype = torch.float32)

class TestDataset(Dataset):
    def __init__ (self,x, mode = True):
        self.Xs = x
        if mode:    
            self.Xs[:,1,3] = 0
            self.Xs[:,1,4] = 0
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32)


