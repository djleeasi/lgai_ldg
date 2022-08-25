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
    np.random.shuffle(shuffle_idx)
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

        # train_y_mean *= 0
        # train_y_std *= 0
        # train_y_std += 1# Norm 안함

        train_x = (train_x-train_x_mean)/train_x_std
        train_y = (train_y-train_y_mean)/train_y_std
        valid_x = (valid_x-train_x_mean)/train_x_std
        valid_y = (valid_y-train_y_mean)/train_y_std
        PARAM_DIR = dataparams.DATA_DIR_PARAMETER + modelparams.MODELNAME + f'{fold}.pt'
        with open(PARAM_DIR+'ms', 'wb') as f:
            pickle.dump((train_x_mean, train_x_std, train_y_mean, train_y_std), f)
        model = TheModel(modelparams)
        model = model.to(model.device)
        train_loader = DataLoader(ProcessDataset(train_x, train_y, mode =  False), trainparams.BATCH_SIZE, shuffle = True, drop_last= True)
        test_loader = DataLoader(ProcessDataset(valid_x, valid_y, mode = False), 2048, shuffle = True)
        train(trainparams, train_loader,test_loader, model, PARAM_DIR, (train_y_mean, train_y_std))

def validate(test_loader, model, loss_func):
    test_losses = []
    total=0
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputdata, target) in enumerate(tqdm(test_loader)):
            target = target.to(device=model.device)
            output, _ = model(inputdata)
            loss = loss_func(output, target)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
    model.train()
    return test_loss

def train(trainparams, data_loader, test_loader, model, PARAM_DIR, stdmeans):
    
    NUM_EPOCHES = trainparams.NUM_EPOCHES

    loss_func_train = loss_func_validation = customloss(data_loader.dataset.Ys, stdmeans)
    loss_func_contrastive = SimCLR_Loss(trainparams.BATCH_SIZE, trainparams.TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainparams.LR, weight_decay = trainparams.WD, eps= 1e-7)
    for epoch in range(NUM_EPOCHES):
        train_losses = []
        contrastive_losses = []
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        for i, (inputdata, target) in enumerate(tqdm(data_loader)):
            target = target.to(device=model.device)
            inputdata = inputdata.to(model.device)     
            optimizer.zero_grad()
            output1, representation1 = model(inputdata)
            output2, representation2 = model(inputdata)
            loss1 = loss_func_train(output1, target)
            loss2 = loss_func_train(output2, target)
            contrastive_loss = loss_func_contrastive(representation1, representation2)
            ratio = 0.5
            preloss = ratio*(loss1+loss2) 
            w_closs = (1-ratio)*contrastive_loss
            loss = preloss + w_closs
            loss.backward()
            optimizer.step()
            train_losses.append(preloss.item())
            contrastive_losses.append(w_closs.item())
        epoch_train_loss = np.mean(train_losses)
        epoch_contrastive_loss = np.mean(contrastive_losses)
        validation_loss = validate(test_loader, model, loss_func_validation)
        print(f'Epoch {epoch+1}') 
        print(f'train prediction_loss : {epoch_train_loss}, train contrastive loss:{epoch_contrastive_loss}, validation_loss: {validation_loss}')
        torch.save(model.state_dict(), PARAM_DIR)
if __name__ == '__main__':
    main()