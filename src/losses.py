import torch 
import torch.nn as nn
import numpy as np
import ipdb
class SimCLR_Loss():
    def __init__(self, batch_size, temperature):
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)#Computationally 비싸서 이렇게 하는 듯?
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def __call__(self, z_i, z_j):#shape: z_i = z_j = batch*128??

        N = 2 * self.batch_size#N=batchsize*2

        z = torch.cat((z_i, z_j), dim=0)#

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature#(i+j)^2=NxN 행렬. 대각선: 자기 자신과의 sim, batch size만큼 더함/뺌:: 같은 원본 image의 다른 transformation
        # ipdb.set_trace()
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)#분자
        negative_samples = sim[self.mask].reshape(N, -1)#대각선성분, 대응성분 0으로 만든 뒤 줄세우기
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float() 어케?
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)#Nx(N+1)이 되고, 마지막 1 dimentsion의 첫 한 값만 증폭시켜야 함. 따라서 정답 라벨이 죄다 0인것.
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss