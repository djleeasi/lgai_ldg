from torch.utils.data import Dataset
import torch

class ProcessDataset(Dataset):
    def __init__ (self, x, y, mode = True):
        self.Xs = x
        self.Ys = y
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32), torch.tensor(self.Ys[idx], dtype = torch.float32)

class TestDataset(Dataset):
    def __init__ (self,x):
        self.Xs = x
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32)


