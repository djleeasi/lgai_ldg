import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
from numpy import Inf
import ipdb 

#-------------------------------------------------------------------------------------------------------------------
class TheModel(nn.Module):
    def __init__(self, modelparams):
        self.device = modelparams.device
        self.minloss = Inf
        self.drop = modelparams.DROPP
        super().__init__()
        self.LSTM = nn.LSTM(input_size = modelparams.VECTOR_SIZE,
            hidden_size= modelparams.HIDDEN_SIZE,
            batch_first=True,
            bidirectional = True
            )
        self.attention = Attention(modelparams.HIDDEN_SIZE, modelparams.ATTENTION_SIZE)
        self.linear1 = nn.Linear(in_features = 2*modelparams.HIDDEN_SIZE, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=14)
        self.activation1 = nn.LeakyReLU(negative_slope=0.5)
        # self.activation2 = nn.LeakyReLU(negative_slope=0.3)

        self.batch1 = nn.BatchNorm1d(256)
        self.batch2 = nn.BatchNorm1d(14)

        # nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.xavier_normal_(self.linear2.weight, gain = 1.0)

    def forward(self, x):
        x = x.to(device=self.device)
        batch = x.shape[0]
        x,(h,c)= self.LSTM(x)
        x, alpha = self.attention(x, False)
        x = self.linear1(x)
        x = nn.functional.dropout(x, self.drop)
        x = self.batch1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = nn.functional.dropout(x, self.drop)
        x = self.batch2(x)
        return x 

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.W_omega = nn.Parameter(torch.FloatTensor(2*hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.FloatTensor(attention_size))
        self.u_omega = nn.Parameter(torch.FloatTensor(attention_size))
        
        nn.init.xavier_normal_(self.W_omega, gain=1)
        nn.init.zeros_(self.b_omega)
        nn.init.normal_(self.u_omega, std=1e-2)

    def forward(self, x, time_major):
        if time_major:
            x = x.permute(1,0,2)
        v = torch.sigmoid(torch.tensordot(x, self.W_omega, dims=1) + self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alpha = nn.functional.softmax(vu.squeeze(), dim=1)
        output = (x * alpha.unsqueeze(-1)).mean(1)
        return output, alpha
