"""
    rawdata의 전처리를 담당하는 함수들+loss. TODO: loss를 다른 파일로 독립시킬지 생각해보기
"""
import numpy as np 
import torch 
from sklearn.model_selection import KFold
from scipy.stats import norm #정규분포를 위해
import ipdb
import copy
#model specific imports-------------------------------
from .hy_params import datahyper, modelhyper

def k_fold(data, label, k_num=9):
    # kf = StratifiedKFold(n_splits=k_num, shuffle=True)
    # kf = KFold(n_splits=k_num, shuffle=True)
    kf = KFold(n_splits=k_num) 
    train_list = list()
    valid_list = list()
    for train_idx, valid_idx in kf.split(data, label):
        x_train, x_valid = data[train_idx], data[valid_idx]
        y_train, y_valid = label[train_idx], label[valid_idx]
        train_list.append((x_train, y_train))
        valid_list.append((x_valid, y_valid))

    return train_list, valid_list

def preXRnn(xData:np.ndarray,stageNum:np.ndarray) -> tuple: 
    """
        xData: (batchsize, vector), stageNum:(stage,), return: totalX, xMin, xMax \n
        RNN에 넣기 위해 2차원(batchsize, vector) MInmax normalization 한 후 3차원 (batchsize, stage, vector/stageNum) 으로 바꿔주는 함수.\n
        ex)vector = 8, stageNum = [2,3,3] 이면 totalX.shape = (batchsize, 3, 3) 이 된다. 크기가 안 맞는 부분은 0으로 pad.
    """
    maxNum = np.max(stageNum)
    initX = xData[:,:stageNum[0]]
    stNum = stageNum[0]
    initX, xMax, xMin = minMaxNormalization(initX) 
    initX = np.pad(initX,((0,0),(0,maxNum-np.shape(initX)[-1])), 'constant', constant_values=(0))
    totalX = np.expand_dims(initX, axis=1)
    for i in range(1,len(stageNum),1):
        stageX = xData[:,stNum:stNum+stageNum[i]] 
        stNum += stageNum[i]
        stageX, sMax,sMin = minMaxNormalization(stageX)
        xMax, xMin = np.append(xMax,sMax), np.append(xMin,sMin)
        padX = np.pad(stageX,((0,0),(0,maxNum - np.shape(stageX)[-1])), 'constant', constant_values=(0))
        padX = np.expand_dims(padX, axis=1)
        totalX = np.concatenate((totalX,padX),axis =1)
    return totalX, xMin, xMax

def preYRnn(yData:np.ndarray) -> tuple:
    """
        yData: (batchsize, vector), return: yData, yMin, yMax \n
        주어진 2차원 행렬을 minmax normalization 하여 출력
    """
    yData, yMax, yMin = minMaxNormalization(yData) 
    return yData, yMin, yMax

def prevXRnn(xData,xMin,xMax,stageNum):
    """
    preXRnn 과 같지만, min-max 정보를 외부에서 가져옴
    """
    maxNum = np.max(stageNum)
    initX = xData[:,:stageNum[0]]
    stNum = stageNum[0]
    initX = testNormalization(initX,xMin[:stageNum[0]],xMax[:stageNum[0]])
    initX = np.pad(initX,((0,0),(0,maxNum-np.shape(initX)[-1])), 'constant', constant_values=(0))
    totalX = np.expand_dims(initX, axis=1)
    for i in range(1,len(stageNum),1):
        stageX = xData[:,stNum:stNum+stageNum[i]] 
        stageX = testNormalization(stageX,xMin[stNum:stNum+stageNum[i]],xMax[stNum:stNum+stageNum[i]])
        stNum += stageNum[i]
        padX = np.pad(stageX,((0,0),(0,maxNum - np.shape(stageX)[-1])), 'constant', constant_values=(0))
        padX = np.expand_dims(padX, axis=1)
        totalX = np.concatenate((totalX,padX),axis =1)
    return totalX

def prevYRnn(yData, yMin,yMax):
    """
    preYRnn 과 같지만, minm-max 정보를 외부에서 가져옴
    """
    yData  = testNormalization(yData, yMin, yMax) 
    return yData

def testNormalization(array,tmin,tmax):
    ndim = np.shape(array)[-1]
    max = tmax
    min = tmin
    array = (array-min)/(max-min) 
    return array

def minMaxNormalization(array):
    """
        array -> (array, max, min) \n
        주어진 ndarray를 minmax nomalization 하여 내보냄.\n
        array.shape =  (row,column) 에 대해 max.shape = (column,)
    """
    ndim = np.shape(array)[-1]
    max = np.max(array, axis=0)
    min = np.min(array, axis=0)
    for i in range(len(max)):
        if max[i] == min[i]:
            raise ZeroDivisionError(f"min == max in {i}")
    array = (array-min)/(max-min) 
    return array, max, min #TODO: max가 먼저 오는 이유?

#----------------------loss functions from now--------------------
def wRMSE(yhat, y):
    """
        Daycon에서 사용하는 것과 같게 가중치를 먹인 mseloss. 완전1대1 대응하지는 않음.
    """
    rmse = torch.sqrt(torch.mean((yhat-y)**2,axis = 0))
    wrmse = 1.2 * torch.sum(rmse[:8]) + 1.0 * torch.sum(rmse[8:14])
    return wrmse

class customloss():
    def __init__(self, yhatarray, *minmaxs):
        """
        #데이콘 운영진측에서 밝힌 custom loss. 단, 평균값은 주어진 data에서 가져온다.\n
        #운영진측에서 밝힌대로 loss 를 계산해야 하기 때문에, 만약 data가 nomalized 된 상태일 경우 unnorm한다. \n
        #참고: https://dacon.io/competitions/official/235927/overview/rules
        """
        modelparams = modelhyper()
        if minmaxs == None:
            self.isnorm = False #주어진 데이터가 nomalized 인지 여부
        else:
            self.isnorm = True
        yhattensor = torch.tensor(yhatarray, requires_grad=False).to( modelparams.DEVICE)
        if self.isnorm == True:
            self.min = torch.tensor(minmaxs[0][0]).to( modelparams.DEVICE)#mean
            self.max = torch.tensor(minmaxs[0][1]).to( modelparams.DEVICE)#std
            yhattensor = self.unnormalize(yhattensor)
        #이 시점에서 모든 data는 원복됨
        self.yhatmean = torch.mean(torch.abs(yhattensor),0)
        self.yhatstd = torch.std(yhattensor,0)
        self.MSELoss = torch.nn.MSELoss(reduction = 'none')
    def __call__(self, yhat, y):
        if self.isnorm == True:
            torchmse = torch.mean(self.MSELoss(self.unnormalize(yhat), self.unnormalize(y)), 0)#shape: 1xOUTPUT SIZE
        else:
            torchmse = torch.mean(self.MSELoss(yhat, y), 0)#shape: 1xOUTPUT SIZE
        rmse = torch.sqrt(torchmse)
        nrmse = torch.div(rmse,self.yhatmean)
        return 1.2*torch.sum(nrmse[:8])+1*torch.sum(nrmse[8:])
    def unnormalize(self,tensor):
        return tensor*(self.max)+self.min

class customloss_entropy(customloss):#useable on train only
    """
    customloss에 라벨의 정규분포 상에서의 빈도수에 따른 보정값(현재: 확률의 역수) 를 취한 loss
    """
    def __init__(self, yhatarray, minmaxs = None):
        super().__init__(yhatarray, minmaxs)
        self.labeldist = norm(self.yhatmean.cpu().numpy(), self.yhatstd.cpu().numpy())
    def __call__(self,yhat,y):
        if self.isnorm == True:
            raw_mse_torch = self.MSELoss(self.unnormalize(yhat), self.unnormalize(y))#shape: BATCHSIZExOUTPUT SIZE
        else:
            raw_mse_torch = self.MSELoss(yhat, y)
        label_probs = self.labeldist.pdf(yhat.detach().cpu().numpy())
        label_probs = torch.tensor(label_probs, requires_grad=False).to( self.yhatmean.device)
        label_entropy = -torch.log2(label_probs+1e-6)+1
        entropy_mse_torch = raw_mse_torch*label_entropy
        torchmse = torch.mean(entropy_mse_torch,0)
        rmse = torch.sqrt(torchmse)
        nrmse = torch.div(rmse,self.yhatmean)
        return 1.2*torch.sum(nrmse[:8])+1*torch.sum(nrmse[8:])