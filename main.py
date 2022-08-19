from src.model import *
from src.mydataset import ProcessDataset
from src.hy_params import modelhyper, datahyper, trainhyper
from src.prepro import *
from src.LDGlib import *
#import packages
import torch
import pickle 
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from os.path import exists
from torch.utils.data import DataLoader, RandomSampler
import pickle
import ipdb

def main():
    modelparams = modelhyper()
    dataparams = datahyper()
    trainparams = trainhyper()
    foldNum = modelparams.KFOLD_NUM
    #raw 데이터셋을 shuffle
    #ipdb.set_trace()
    with open(dataparams.DATA_DIR_TRAIN,'rb')as f:   
        x_data, y_data, stages = pickle.load(f)
    x_n, x_an = x_data[0], x_data[1]
    indexlist = list(range(8))+list(range(10,52))
    x_n = x_n[:, indexlist]
    x_an = x_an[:,indexlist]
    y_n, y_an = y_data[0], y_data[1]
    train_list_n, valid_list_n = k_fold(x_n, y_n, foldNum)
    train_list_an, valid_list_an = k_fold(x_an, y_an, foldNum)
    
    for fold in range(foldNum):
        print('FOLD', 1+fold)
        train_x_n, train_y_n = train_list_n[fold][0], train_list_n[fold][1]
        train_x_an, train_y_an = train_list_an[fold][0], train_list_an[fold][1]
        valid_x = np.concatenate((
                                    valid_list_n[fold][0], 
                                    valid_list_an[fold][0]))
        valid_y = np.concatenate((
                                    valid_list_n[fold][1],
                                    valid_list_an[fold][1]))
        PARAM_DIR = dataparams.DATA_DIR_PARAMETER + modelparams.MODELNAME + f'{fold}.pt'
        model = TheModel(modelparams)
        model = model.to(model.device)
        train_n_set = ProcessDataset(train_x_n, train_y_n, mode = False)
        train_an_set = ProcessDataset(train_x_an, train_y_an, mode= False)
        train_n_batch = round(trainparams.BATCH_SIZE*trainparams.N_RATIO)
        train_an_batch = round(trainparams.BATCH_SIZE*(1-trainparams.N_RATIO))
        batchtotal = train_n_batch+train_an_batch
        print(train_an_batch, train_n_batch, batchtotal)
        train_loader_n = DataLoader(
                                    train_n_set,
                                    train_n_batch,
                                    sampler = RandomSampler(train_n_set, replacement=True, num_samples = batchtotal)
                                    )
        train_loader_an = DataLoader(
                                    train_an_set,
                                    train_an_batch,
                                    sampler = RandomSampler(train_an_set, replacement=True, num_samples = batchtotal)
                                    )
        #ipdb.set_trace()
        test_loader = DataLoader(ProcessDataset(valid_x, valid_y, mode = False), 2048, shuffle = True)
        train(trainparams, train_loader_n, train_loader_an, test_loader, model, PARAM_DIR)

def validate(test_loader, model, loss_func):
    test_losses = []
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

def train(trainparams, data_loader_n, data_loader_an, test_loader, model, PARAM_DIR):
    NUM_EPOCHES = trainparams.NUM_EPOCHES
    loss_func_train = nn.L1Loss()
    loss_func_validation = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainparams.LR, weight_decay = trainparams.WD)
    for epoch in range(NUM_EPOCHES):
        train_losses = []
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        ipdb.set_trace()
        for i, (inputdata_n, target_n) in enumerate(tqdm(data_loader_n)):
            for inputdata_an, target_an in data_loader_an:
                inputdata = torch.cat((inputdata_n, inputdata_an))
                target = torch.cat((target_n, target_an))
                target = target.to(device = model.device)
                optimizer.zero_grad()
                output = model(inputdata).to(device=model.device)
                loss = loss_func_train(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pass
        epoch_train_loss = np.mean(train_losses)
        validation_loss = validate(test_loader, model, loss_func_validation)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}, validation_loss: {validation_loss}')
        # Save Model
        if model.minloss > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(model.minloss, validation_loss))
            model.minloss = validation_loss
            torch.save(model.state_dict(), PARAM_DIR)
    torch.save(model.state_dict(), PARAM_DIR+'k')

if __name__ == '__main__':
    main()