from src.model import *
from src.mydataset import ProcessDataset
from src.hy_params import modelhyper, datahyper, trainhyper
from src.prepro import *
from src.LDGlib import shufflearrays
from src.losses import SimCLR_Loss
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
    datas = pd.read_csv('data/rawdata/train.csv')
    datas = datas.drop(columns='ID')
    datas = datas.drop(columns = ['X_10', 'X_11', 'X_04', 'X_23', 'X_47', 'X_48'])
    x_data = datas.iloc[:, :50].values
    y_data = datas.iloc[:, 50:].values
    shuffle_idx = np.arange(x_data.shape[0])
    np.random.shuffle(shuffle_idx)#TODO: 좋은 결과를 복원할 수 있게 seed 저장방법 찾기 IDEA: numpy의 randomGenerator를 random seed 로 initialize
    x_data = x_data[shuffle_idx,:]
    y_data = y_data[shuffle_idx,:]
    train_list, valid_list = k_fold(x_data, y_data, foldNum)

    for fold in range(foldNum):
        print('FOLD', 1+fold)
        train_x, train_y = train_list[fold][0], train_list[fold][1]
        valid_x, valid_y = valid_list[fold][0], valid_list[fold][1]
        train_x_std = np.std(train_x, axis = 0)+1e-20
        train_y_std = np.std(train_y, axis = 0)+1e-20
        train_x_mean = np.mean(train_x, axis = 0)
        train_y_mean = np.mean(train_y, axis = 0)
        train_x = (train_x-train_x_mean)/train_x_std
        train_y = (train_y-train_y_mean)/train_y_std
        valid_x = (valid_x-train_x_mean)/train_x_std
        valid_y = (valid_y-train_y_mean)/train_y_std
        PARAM_DIR = dataparams.DATA_DIR_PARAMETER + modelparams.MODELNAME + f'{fold}.pt'
        with open(PARAM_DIR+'ms', 'wb') as f:
            pickle.dump((train_x_mean, train_x_std, train_y_mean, train_y_std), f)
        model = TheModel(modelparams)
        model = model.to(model.device)
        train_loader = DataLoader(ProcessDataset(train_x, train_y, mode =  False), trainparams.BATCH_SIZE, shuffle = True)
        test_loader = DataLoader(ProcessDataset(valid_x, valid_y, mode = False), 2048, shuffle = True)
        train(trainparams, train_loader,test_loader, model, PARAM_DIR, modelparams.Norm)

def validate(test_loader, model, loss_func):
    test_losses = []
    total=0
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputdata, target) in enumerate(tqdm(test_loader)):
            target = target.to(device=model.device) 
            output = model(inputdata).to(device=model.device)
            loss = loss_func(output, target)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
    model.train()
    return test_loss

def train(trainparams, data_loader, test_loader, model, PARAM_DIR,Norm):
    NUM_EPOCHES = trainparams.NUM_EPOCHES
    loss_func_train = loss_func_validation = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainparams.LR, weight_decay = trainparams.WD, eps= 1e-7)
    for epoch in range(NUM_EPOCHES):
        train_losses = []
        total=0
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        for i, (inputdata, target) in enumerate(tqdm(data_loader)):
            target = target.to(device=model.device)        
            optimizer.zero_grad()
            output = model(inputdata).to(device=model.device)
            loss = loss_func_train(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        epoch_train_loss = np.mean(train_losses)
        validation_loss = validate(test_loader,model, loss_func_validation)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}, validation_loss: {validation_loss}')

        # Save Model
        if model.minloss > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(model.minloss, validation_loss))
            model.minloss = validation_loss
            torch.save(model.state_dict(), PARAM_DIR)
if __name__ == '__main__':
    main()