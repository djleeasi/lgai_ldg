import numpy as np 
import torch 
from sklearn.model_selection import KFold

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


def wRMSE(yhat, y):
    rmse = torch.sqrt(torch.mean((yhat-y)**2,axis = 0))
    wrmse = 1.2 * torch.sum(rmse[:8]) + 1.0 * torch.sum(rmse[8:14])
    return wrmse

def preXRnn(xData,stageNum):
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

def preYRnn(yData):
    yData, yMax, yMin = minMaxNormalization(yData) 
    return yData, yMin, yMax

def prevXRnn(xData,xMin,xMax,stageNum):
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
    yData  = testNormalization(yData, yMin, yMax) 
    return yData

def testNormalization(array,tmin,tmax):
    ndim = np.shape(array)[-1]
    max = tmax
    min = tmin
    array = (array-min)/(max-min) 
    return array

def minMaxNormalization(array):
    ndim = np.shape(array)[-1]
    max = np.max(array, axis=0)
    min = np.min(array, axis=0)
    for i in range(len(max)):
        if max[i] == min[i]:
            raise ZeroDivisionError(f"min == max in {i}")
    array = (array-min)/(max-min) 
    return array, max, min
