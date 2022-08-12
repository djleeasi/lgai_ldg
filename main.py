from src.model import *
from src.mydataset import ProcessDataset
from src.hy_params import modelhyper, datahyper, trainhyper
from src.customloss import customloss
#import packages
import torch
import pickle
import pandas as pd 
from sklearn.model_selection import KFold
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
    FOLDER_DIR = dataparams.DATA_DIR_PARAMETER
    foldNum = modelparams.KFOLD_NUM
    with open(dataparams.DATA_DIR_TRAIN,'rb')as f:   
        x_data, y_data = pickle.load(f)
        
    with open(dataparams.DATA_DIR_MINMAX,'rb')as f:   
        _, _, y_min, y_max = pickle.load(f)
         
    train_list, test_list = k_fold(x_data, y_data, foldNum)

    for fold in range(foldNum):
        PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + f'{fold}.pt'
        model = TheModel(modelparams)
        model = model.to(model.device)
        train_loader = DataLoader(ProcessDataset(train_list[fold][0], train_list[fold][1]), trainparams.BATCH_SIZE, shuffle = True)
        test_loader = DataLoader(ProcessDataset(test_list[fold][0], test_list[fold][1]), 2048, shuffle = True)
        train(trainparams, train_loader,test_loader, model, y_min, y_max, PARAM_DIR)


def validate(test_loader, model, tmin, tmax, criterion):
    test_losses = []
    total=0
    model.eval()
    test_loss = 0
    #minmax normalization 원복 코드
    mult= tmax - tmin 
    mult = torch.tensor(mult).to(device = model.device)
    add = torch.tensor(tmin).to(device = model.device)
    #---
    with torch.no_grad():
        for i, (inputdata, target) in enumerate(tqdm(test_loader)):
            target = target.to(device=model.device) 
            output = model(inputdata).to(device=model.device)
            loss = criterion(output,target)
            test_losses.append(loss.item())
            total += target.size(0)
        test_loss = np.mean(test_losses)
    model.train()
    return test_loss



def train(trainparams, data_loader, test_loader, model, tmin, tmax, PARAM_DIR):
    criterion_validation = customloss(test_loader.dataset.Ys)

    NUM_EPOCHES = trainparams.NUM_EPOCHES
    criterion = customloss(data_loader.dataset.Ys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainparams.LR, weight_decay = trainparams.WD)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    #optimizer = torch.optim.Adam(model.parameters(), lr=trainparams.LR)
    
    for epoch in range(NUM_EPOCHES):
        train_losses = []
        total=0
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        for i, (inputdata, target) in enumerate(tqdm(data_loader)):
            target = target.to(device=model.device)        
            optimizer.zero_grad()
            output = model(inputdata).to(device=model.device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            total += target.size(0)
        epoch_train_loss = np.mean(train_losses)
        validation_loss = validate(test_loader, model, tmin, tmax, criterion_validation)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}, validation_loss: {validation_loss}')

        # Save Model
        if model.modeldata['minloss'] > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(model.modeldata['minloss'], validation_loss))
            model.modeldata['minloss'] =  validation_loss
            model.modeldata['bestparam'] = model.state_dict()#potential pointer threat. but it's not an issue because the parameter is saved as a file rightafetr
            with open(PARAM_DIR,'wb') as f:
                pickle.dump(model.modeldata,f)
            
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

if __name__ == '__main__':
    main()