import torch
import torch.nn as nn
from numpy import Inf
import ipdb 
from src.hy_params import modelhyper
#-------------------------------------------------------------------------------------------------------------------

class SimcldEncoder(nn.Module):
    def __init__(self, modelparams:modelhyper):
        super(SimcldEncoder, self).__init__()
        e_seqlength = modelparams.E_SEQLENGTH
        d_model = modelparams.E_INPUTSIZE
        device = modelparams.DEVICE
        dropout = modelparams.E_DROPOUT
        nhead = modelparams.E_NHEAD
        dim_feedforward = modelparams.E_DIM_FEEDFORWARD
        e_layer_num = modelparams.E_LAYER_NUM
        self.d_model = d_model
        e_layers = list()
        for number in range(e_layer_num):
            e_layers.append(
                    torch.nn.TransformerEncoderLayer(d_model = d_model, nhead= nhead, dim_feedforward= dim_feedforward, dropout = dropout,norm_first=False , batch_first= True, activation='relu')
            )
        self.posi_embed = nn.parameter.Parameter(torch.randn(1,e_seqlength,d_model)).to(device)
        self.multi_embed = nn.parameter.Parameter(torch.randn(1,e_seqlength,d_model)).to(device)
        self.encoderlayers = nn.Sequential(*e_layers).to(device)
    def forward(self,x):#x:(batch,seqlength,1)
        # ipdb.set_trace()
        batch_size = x.size(0)
        x = x.repeat(1,1,self.d_model)
        x= x*self.multi_embed.repeat(x.size(0),1,1) + self.posi_embed.repeat(x.size(0),1,1)
        x = self.encoderlayers(x)
        # return x[:,0,:].reshape(batch_size, -1)
        return x.reshape(batch_size,-1)
class ALinearLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(ALinearLayer, self).__init__()
        a = int(input_size)
        b = int(output_size)
        self.layerstack = nn.Sequential(
            nn.Linear(a,b),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(b)
        ).to(modelhyper.DEVICE)
    def forward(self,x):
        return self.layerstack(x)
class Linear14(nn.Module):
    def __init__(self, modelparams:modelhyper):
        super(Linear14, self).__init__()
        hs = int(modelparams.E_INPUTSIZE)
        l_dropout = modelparams.L_DROPOUT
        linearlist = list()
        for ys in range(14):
            linearlist.append(
                nn.Sequential(
                    nn.BatchNorm1d(hs),
                    ALinearLayer(hs, hs/modelparams.E_NHEAD, l_dropout),
                    #ALinearLayer(hs/4, hs/8, l_dropout),
                    nn.Linear(int(hs/modelparams.E_NHEAD),1)
                )
            )
        self.classifiers = nn.ModuleList(linearlist).to(modelhyper.DEVICE)
    def forward(self,x):#batch, input차원*d_modez
        result = list()
        for classifier in self.classifiers:
            result.append(classifier(x))#batch, 1
        result = torch.stack(result, dim = 1)
        return result.squeeze(-1)
class TheModel(nn.Module):
    def __init__(self, modelparams:modelhyper):
        super(TheModel, self).__init__()
        self.encoder = SimcldEncoder(modelparams)
        # self.decoder = Linear14(modelparams)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(modelparams.E_SEQLENGTH*modelparams.E_INPUTSIZE),
            nn.Linear(modelparams.E_SEQLENGTH*modelparams.E_INPUTSIZE,128 ),
            nn.ReLU(),
            nn.Dropout(modelhyper.L_DROPOUT),
            nn.BatchNorm1d(128),
            nn.Linear(128,14),
        ).to(modelhyper.DEVICE)
    def forward(self, x):
        batch_size = x.size(0)
        representation = self.encoder(x)
        resultvector = self.decoder(representation)
        return resultvector, representation