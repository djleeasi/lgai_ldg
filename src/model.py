import torch.nn as nn
from torch.nn.functional import softmax as softmax
from numpy import Inf
import ipdb 

#-------------------------------------------------------------------------------------------------------------------
class TheModel(nn.Module):
    def __init__(self, modelparams):
        super().__init__()
        self.device = modelparams.device
        self.minloss = Inf
        HIDDEN_SIZE = modelparams.HIDDEN_SIZE
        self.layers = nn.Sequential(
                                    nn.BatchNorm1d(modelparams.VECTOR_SIZE),
                                    nn.Linear(modelparams.VECTOR_SIZE, HIDDEN_SIZE),
                                    nn.ReLU(HIDDEN_SIZE),
                                    nn.BatchNorm1d(HIDDEN_SIZE),
                                    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE*2),
                                    nn.ReLU(HIDDEN_SIZE*2),                                    
                                    nn.BatchNorm1d(HIDDEN_SIZE*2),
                                    nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE),
                                    nn.ReLU(HIDDEN_SIZE),                                    
                                    nn.BatchNorm1d(HIDDEN_SIZE),
                                    nn.Linear(HIDDEN_SIZE, modelparams.OUTPUT_CLASS)
                                    ).to(device= self.device)
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                if layer.weight.dim() > 1:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    def forward(self, x):
        x = x.to(device=self.device)
        x = self.layers(x)
        return x
