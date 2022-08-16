from src.model import *
from src.mydataset import ProcessDataset
from src.hy_params import modelhyper, datahyper, trainhyper
from src.prepro import *
#import packages
import torch
import pickle
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from os.path import exists
from torch.utils.data import DataLoader
import pickle
import ipdb

def main():
    modelparams = modelhyper()
    dataparams = datahyper()
    trainparams = trainhyper()
    foldNum = modelparams.KFOLD_NUM
    with open(dataparams.DATA_DIR_TRAIN,'rb')as f:   
        x_data, y_data, stages = pickle.load(f)
    #raw 데이터셋을 shuffle
    shuffle_idx = np.arange(x_data.shape[0])
    np.random.shuffle(shuffle_idx)#TODO: 좋은 결과를 복원할 수 있게 seed 저장방법 찾기 IDEA: numpy의 randomGenerator를 random seed 로 initialize
    x_data = x_data[shuffle_idx,:]
    y_data = y_data[shuffle_idx,:]
    train_list, valid_list = k_fold(x_data, y_data, foldNum)


    for fold in range(foldNum):
        print('FOLD', 1+fold)
        train_x, train_y = train_list[fold][0], train_list[fold][1]
        valid_x, valid_y = valid_list[fold][0], valid_list[fold][1]
        train_x, train_x_min, train_x_max = preXRnn(train_x, stages)
        train_y, train_y_min, train_y_max = preYRnn(train_y)
        valid_x = prevXRnn(valid_x, train_x_min, train_x_max,stages)
        valid_y = prevYRnn(valid_y, train_y_min, train_y_max)
        PARAM_DIR = dataparams.DATA_DIR_PARAMETER + modelparams.MODELNAME + f'{fold}.pt'
        MinMax_path = dataparams.DATA_DIR_MM + modelparams.MODELNAME +f'{fold}.pickle'
        with open(MinMax_path, 'wb')as f:
            pickle.dump((train_x_min, train_x_max, train_y_min, train_y_max), f)
        model = TheModel(modelparams)
        model = model.to(model.device)
        train_loader = DataLoader(ProcessDataset(train_x, train_y), trainparams.BATCH_SIZE, shuffle = True)
        test_loader = DataLoader(ProcessDataset(valid_x, valid_y), 2048, shuffle = True)
        train(trainparams, train_loader,test_loader, model, train_y_min, train_y_max, PARAM_DIR, modelparams.Norm)


def validate(test_loader, model, mult, add,Norm):
    # criterion = torch.nn.MSELoss()
    test_losses = []
    total=0
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputdata, target) in enumerate(tqdm(test_loader)):
            target = target.to(device=model.device) 
            output = model(inputdata).to(device=model.device)
            # if Norm:
            #     loss = criterion((output*mult)+add, (target*mult)+add)
            # else:
            #     loss = criterion(output, target)
            loss = wRMSE(output, target)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
    model.train()
    return test_loss



def train(trainparams, data_loader, test_loader, model, tmin, tmax, PARAM_DIR,Norm):
    NUM_EPOCHES = trainparams.NUM_EPOCHES
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainparams.LR, weight_decay = trainparams.WD)
    # testingloss = weightedRMSE
    mult = tmax - tmin 
    mult = torch.tensor(mult).to(device = model.device)
    add = torch.tensor(tmin).to(device = model.device)
    for epoch in range(NUM_EPOCHES):
        train_losses = []
        total=0
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        for i, (inputdata, target) in enumerate(tqdm(data_loader)):
            target = target.to(device=model.device)        
            optimizer.zero_grad()
            output = model(inputdata).to(device=model.device)

            # if Norm:
            #     loss = criterion((output*mult)+add, (target*mult)+add)
            # else:
            #     loss = criterion(output, target)
            loss = wRMSE(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        epoch_train_loss = np.mean(train_losses)
        validation_loss = validate(test_loader,model, mult, add,Norm)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}, validation_loss: {validation_loss}')

        # Save Model
        if model.minloss > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(model.minloss, validation_loss))
            model.minloss = validation_loss
            torch.save(model.state_dict(), PARAM_DIR)
            
if __name__ == '__main__':
    main()