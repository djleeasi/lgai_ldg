#데이콘 운영진측에서 밝힌 custom loss. 단, 평균값은 주어진 data에서 가져온다.
#https://dacon.io/competitions/official/235927/overview/rules
import torch 
import pickle
#model specific imports
#-----------------------------------------------------------
from .hy_params import datahyper, modelhyper
#------------------------------------------------------------

class customloss():
    def __init__(self, gtarray, unnorm = False):
        #gtarray = y label
        #deprecated
        dataparams = datahyper()
        modelparams = modelhyper()
        gttensor = torch.tensor(gtarray).to(device = modelparams.device)
        if unnorm == True:
            with open(dataparams.DATA_DIR_MINMAX, 'rb')as f:
                minmax= pickle.load(f)
                self.min = torch.tensor(minmax[2]).to(device = modelparams.device)
                self.max = torch.tensor(minmax[3]).to(device = modelparams.device)
                gttensor = self.origin(gttensor)
        self.gtmean = torch.mean(torch.abs(gttensor),0)
        self.MSELoss = torch.nn.MSELoss(reduction = 'none')
    def __call__(self, gt, pred):
        torchmse = torch.mean(self.MSELoss(self.origin(gt), self.origin(pred)), 0)#shape: 1xOUTPUT SIZE        
        rmse = torch.sqrt(torchmse)
        nrmse = torch.div(rmse,self.gtmean)
        return 1.2*torch.sum(nrmse[:8])+1*torch.sum(nrmse[8:])
    def origin(self,tensor):
        return tensor*(self.max-self.min)+self.min



