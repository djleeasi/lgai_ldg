import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
import ipdb 

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

# class Attention(nn.Module):
#     def __init__(self, hidden_size, attention_size):
#         super(Attention, self).__init__()
#         # input: (batch, timestep, hidden)
#         # 각 hidden 마다 attention size로 mapping 시킬 가중치(W_omega)가 필요하기 때문에 (hidden size, attention size)의 가중치 생성 
#         # (batch, timestep, hidden) -> (batch, timestep, attention size) 
#         # bias:(attention size)를 더해준다. 
#         # (batch, timestep, attention size) - > (batch, timestep)
#         # 으로 만들기 위해  (attention size) 의 가중치를 만들어 준다(u_omega)
#         self.W_omega = nn.Parameter(torch.FloatTensor(hidden_size, attention_size))
#         self.b_omega = nn.Parameter(torch.FloatTensor(attention_size))
#         self.u_omega = nn.Parameter(torch.FloatTensor(attention_size))
        
#         nn.init.xavier_normal_(self.W_omega, gain=1)
#         nn.init.zeros_(self.b_omega)
#         nn.init.normal_(self.u_omega, std=1e-2)

#     def forward(self, x, time_major):
#         # 전부 time_major = True로 설정되어 있고
#         # 그 경우 들어오는 input의 shape은 다음과 같다. input: (time, batch, hidden)이다. 
#         if time_major:
#             x = x.permute(1,0,2)
#             # (time, batch, hidden)에서 permute를 통해 (batch, time, hidden)로 shape을 변경해준다. 

#         # torch.tensorodt 행렬곱셈 dims =1 연산에 서 행렬곱을 통해 사라지는 차원의 수를 선택한다. 
#         # 예를 들어 (2,3,3)와 (3,3)이 dims= 1 로 설정되어있는 경우 2x3x3와 3x3이 행렬곱을 하여 (2,3,3) 의 output이 나오게 된다.
#         # dims =2 일 경우 [2x(3x3)]과 [(3x3)]을 연산하여 [2]의 shape이 나온다.
#         # (batch, timestep, {hidden}) *({hidden},attention) {} 사라지게 한 차원 
#         # ->  (batch, timestep, attention) + (attention) 
#         # 위 연산을 거친 후 activation function(sigmoid)를 걸어준다. 
#         # 변환된 값과 가중치와의 행렬 곱을 통해 다음의 shape의 값을 얻는다.
#         # (batch, timestep, {attention}) *({attention})
#         # (batch, timestep) 
#         v = F.sigmoid(torch.tensordot(x, self.W_omega, dims=1) + self.b_omega)
#         vu = torch.tensordot(v, self.u_omega, dims=1)
#         # .squeeze는 1인 차원을 축소시켜준다. 
#         # 다음 timestep 차원으로 softmax를 걸어 timestep별로 0~1사이의 값을 갖는 attention score를 계산한다. 
#         alpha = F.softmax(vu.squeeze(), dim=1)
#         # unsqueeze로 (batch, timestep) -> (batch, timestep,1) 로 차원을 늘려준다. 
#         # (batch, timestep, hidden)
#         # (batch, timestep, hidden) -> (batch, hidden)
#         output = (x * alpha.unsqueeze(-1)).mean(1)
#         return output, alpha

class TheModel(nn.Module):
    def __init__(self,modelparams):
        self.modeldata = modelparams.MODELDATA
        self.device = modelparams.device
        hidden_size = modelparams.HIDDEN_SIZE
        super().__init__()
        self.encoder=Encoder(modelparams)
        self.attention = Attention(modelparams)
        self.decode_embed = nn.Parameter(torch.randn(1,hidden_size).to(device=self.device), requires_grad = True)
        self.hiddenstate = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                         nn.Dropout(modelparams.DROPP),
                                         nn.BatchNorm1d(hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_size, modelparams.OUTPUT_CLASS),
                                        nn.BatchNorm1d(modelparams.OUTPUT_CLASS)
                                         ).to(device = self.device)
        
    def forward(self, x):#x: vector(vector_size= RNN 의 한 timestep 에 넣어주는 vector.)
        ipdb.set_trace()
        x = x.to(device=self.device)
        batch_size = x.shape[0]
        encoder_result, _ = self.encoder(x) #encoder result: batch, time, hidden

        decode_duplicated = self.decode_embed.unsqueeze(0).repeat(batch_size,1,1)    
        attention_score = torch.matmul(encoder_result,decode_duplicated.transpose(1,2))#decoder_embed를 column vector로 바꿔주었다. final shape = batch*sentence size* 1
        attention_distribution = softmax(attention_score, dim=1)
        attention_value = torch.sum(encoder_result * attention_distribution, dim=1).unsqueeze(1)#아다마르 곱 후 unsqueeze 로 batch*1*hidden_size 로 바꿔준다
        att_dec_concat = torch.cat((attention_value, decode_duplicated), dim = 2).squeeze(1)#batch*hidden_size 두배
        final_vector = self.hiddenstate(att_dec_concat)
        return final_vector # batch, label 
