from typing import OrderedDict
import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
from numpy import Inf

#-------------------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, modelparams):
        self.device = modelparams.device
        super().__init__()
        self.LSTM = nn.LSTM(input_size = modelparams.VECTOR_SIZE,
            hidden_size= modelparams.HIDDEN_SIZE,
            batch_first=True
            ).to(device = self.device)
    def forward(self, x):
        x = x.to(device=self.device)#x shape: batch*process_size*vector_size
        output, final_status = self.LSTM(x)
        return output, final_status# output shape: batch x process_size x hidden_size

class TheModel(nn.Module):
    def __init__(self,modelparams):
        self.modeldata = modelparams.MODELDATA#additional data
        self.device = modelparams.device
        hidden_size = modelparams.HIDDEN_SIZE
        super().__init__()
        self.encoder=Encoder(modelparams)
        self.decode_embed = nn.Parameter(torch.randn(1,hidden_size).to(device=self.device), requires_grad = True)
        self.hiddenstate = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                         nn.Dropout(modelparams.DROPP),
                                         nn.BatchNorm1d(hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_size, modelparams.OUTPUT_CLASS),
                                        nn.BatchNorm1d(modelparams.OUTPUT_CLASS)
                                         ).to(device = self.device)
        
    def forward(self, x):#x: vector(vector_size= RNN 의 한 timestep 에 넣어주는 vector.)
        x = x.to(device=self.device)
        batch_size = x.shape[0]
        encoder_result, _ = self.encoder(x)#encoder result: batchxprocess_sizexhidden_size
        decode_duplicated = self.decode_embed.unsqueeze(0).repeat(batch_size,1,1)    
        attention_score = torch.matmul(encoder_result,decode_duplicated.transpose(1,2))#decoder_embed를 column vector로 바꿔주었다. final shape = batch*sentence size* 1
        attention_distribution = softmax(attention_score, dim=1)
        attention_value = torch.sum(encoder_result * attention_distribution, dim=1).unsqueeze(1)#아다마르 곱 후 unsqueeze 로 batch*1*hidden_size 로 바꿔준다
        att_dec_concat = torch.cat((attention_value, decode_duplicated), dim = 2).squeeze(1)#batch*hidden_size 두배
        final_vector = self.hiddenstate(att_dec_concat)
        return final_vector
