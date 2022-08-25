from src.model import *
from src.mydataset import ProcessDataset
from src.hy_params import modelhyper, datahyper, trainhyper
from src.prepro import *
from src.LDGlib import *
from src.losses import SimCLR_Loss
#import packages
import torch
import pickle 
# from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from os.path import exists
from torch.utils.data import DataLoader
import pickle
import ipdb

def main():
    with open(datahyper.DATA_DIR_TRAIN,'rb')as f:
        x_data, y_data, stages = pickle.load(f)
    # x_n, x_an = x_data[0], x_data[1] KFOLD할때 써라
    # y_n, y_an = y_data[0], y_data[1]
    # train_list_n, valid_list_n = k_fold(x_n, y_n, foldNum)
    # train_list_an, valid_list_an = k_fold(x_an, y_an, foldNum)

    # for fold in range(foldNum):
    #     print('FOLD', 1+fold)
    #     train_x_n, train_y_n = train_list_n[fold][0], train_list_n[fold][1]
    #     train_x_an, train_y_an = train_list_an[fold][0], train_list_an[fold][1]
    #     valid_x = np.concatenate((
    #                                 valid_list_n[fold][0], 
    #                                 valid_list_an[fold][0]))
    #     valid_y = np.concatenate((
    #                                 valid_list_n[fold][1],
    #                                 valid_list_an[fold][1]))
    #     train_x_n = preXRnn_nn(train_x_n, stages)
    #     train_x_an = preXRnn_nn(train_x_an, stages)
    #     valid_x = preXRnn_nn(valid_x, stages)
    x_data, y_data = shufflearrays([x_data, y_data], datahyper.SEED)
    train_len = round(len(x_data)*datahyper.TRAINRATIO)
    train_x = x_data[:train_len]
    valid_x = x_data[train_len:]
    train_y = y_data[:train_len]
    valid_y = y_data[train_len:]
    train_x_std = np.std(train_x, axis = 0)+1e-6
    train_y_std = np.std(train_y, axis = 0)+1e-6
    train_x_mean = np.mean(train_x, axis = 0)
    train_y_mean = np.mean(train_y, axis = 0)
    train_x = (train_x-train_x_mean)/train_x_std
    train_y = (train_y-train_y_mean)/train_y_std
    valid_x = (valid_x-train_x_mean)/train_x_std
    valid_y = (valid_y-train_y_mean)/train_y_std
    PARAM_DIR = datahyper.DATA_DIR_PARAMETER + modelhyper.MODELNAME + '.pt'
    with open(PARAM_DIR+'k', 'wb') as f:
        pickle.dump((train_x_mean, train_x_std, train_y_mean, train_y_std), f)
    model = TheModel(modelhyper)
    model = model.to(modelhyper.DEVICE)
    train_loader = DataLoader(ProcessDataset(train_x, train_y), trainhyper.BATCH_SIZE, shuffle = True, drop_last= True)
    test_loader = DataLoader(ProcessDataset(valid_x, valid_y), max(500, valid_y.shape[0]), shuffle = False)
    train(trainhyper, train_loader, test_loader, model, PARAM_DIR, (train_y_mean, train_y_std))

def validate(test_loader, model, loss_func):
    test_losses = []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (inputdata, target) in enumerate(tqdm(test_loader)):
            inputdata = inputdata.unsqueeze(-1).to(modelhyper.DEVICE)
            target = target.to(device=modelhyper.DEVICE) 
            output, _ = model(inputdata)
            loss = loss_func(output, target)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
    model.train()
    return test_loss

def train(trainhyper, data_loader, test_loader, model, PARAM_DIR, stdmeans):
    NUM_EPOCHES = trainhyper.NUM_EPOCHES
    minloss = float('inf')
    loss_func_train = customloss(data_loader.dataset.Ys, stdmeans)
    loss_func_contrastive = SimCLR_Loss(trainhyper.BATCH_SIZE, trainhyper.TEMPERATURE)
    loss_func_validation = customloss(data_loader.dataset.Ys, stdmeans)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainhyper.LR, weight_decay = trainhyper.WD)
    for epoch in range(NUM_EPOCHES):
        train_losses = list()
        contrastive_losses = list()
        print(f"[Epoch {epoch+1} / {NUM_EPOCHES}]")
        model.train()
        for i, (inputdata, target) in enumerate(tqdm(data_loader)):
            inputdata = inputdata.unsqueeze(-1).to(modelhyper.DEVICE)
            target = target.to(modelhyper.DEVICE)
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
        if minloss > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(minloss, validation_loss))
            minloss = validation_loss
        torch.save(model.state_dict(), PARAM_DIR)
if __name__ == '__main__':
    main()