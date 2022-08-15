#데이콘 운영진측에서 밝힌 custom loss. 단, 평균값은 주어진 data에서 가져온다.
#https://dacon.io/competitions/official/235927/overview/rules
from cProfile import label
import torch 
import pickle
from scipy.stats import norm
#model specific imports
#-----------------------------------------------------------
from .hy_params import datahyper, modelhyper
#------------------------------------------------------------

class customloss():
    def __init__(self, gtarray, unnorm = False):
        #gtarray = y label
        #deprecated
        self.unnorm = unnorm
        dataparams = datahyper()
        self.modelparams = modelhyper()
        gttensor = torch.tensor(gtarray).to(device = self.modelparams.device)
        if self.unnorm == True:
            with open(dataparams.DATA_DIR_MINMAX, 'rb')as f:
                minmax= pickle.load(f)
                self.min = torch.tensor(minmax[2]).to(device = self.modelparams.device)
                self.max = torch.tensor(minmax[3]).to(device = self.modelparams.device)
                gttensor = self.origin(gttensor)
        self.gtmean = torch.mean(torch.abs(gttensor),0)
        self.MSELoss = torch.nn.MSELoss(reduction = 'none')
    def __call__(self, gt, pred):
        
        torchmse = torch.mean(self.MSELoss(self.origin(gt), self.origin(pred)), 0)#shape: 1xOUTPUT SIZE        
        rmse = torch.sqrt(torchmse)
        nrmse = torch.div(rmse,self.gtmean)
        return 1.2*torch.sum(nrmse[:8])+1*torch.sum(nrmse[8:])
    def origin(self,tensor):
        if self.unnorm == True:
            return tensor*(self.max-self.min)+self.min
        else:
            return tensor

class customloss_entropy(customloss):#useable on train only
    def __init__(self, gtarray, unnorm = False):
        super().__init__(gtarray, unnorm)
        gttensor = torch.tensor(gtarray, requires_grad=False) 
        labelmeans = torch.mean(gttensor,0)
        labelstd = torch.std(gttensor,0)
        self.labeldist = norm(labelmeans,labelstd)
        
    def __call__(self,gt,pred):
        raw_mse_torch = self.MSELoss(self.origin(gt), self.origin(pred))
        label_probs = self.labeldist.pdf(gt.detach().cpu().numpy())
        label_probs = torch.tensor(label_probs, requires_grad=False).to(device = self.modelparams.device)
        label_entropy = 1/(label_probs+1e-6)
        #print(label_entropy)
        #raise Exception
        """
        try:
            max_than_inf = torch.max(label_entropy[label_entropy < float('Inf')])#TODO: Implement elegant way to express continius prob
            label_entropy[label_entropy == float('Inf')] = max_than_inf
        except:
            print(label_probs)
            print(gt)
            print(label_entropy)
            raise Exception
        """
        entropy_mse_torch = raw_mse_torch*label_entropy
        torchmse = torch.mean(entropy_mse_torch,0)
        rmse = torch.sqrt(torchmse)
        nrmse = torch.div(rmse,self.gtmean)
        return 1.2*torch.sum(nrmse[:8])+1*torch.sum(nrmse[8:])



