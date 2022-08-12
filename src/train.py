#import from src
from .model import TheModel
from .mydataset import ProcessDataset, TestDataset
from .hy_params import modelhyper, datahyper, trainhyper
from .modelload import loadmodel
from .customloss import customloss
#import packages

import argparse
import torch
import numpy as np
from tqdm import tqdm
from os.path import exists
from torch.utils.data import DataLoader
import pickle


FOLDER_DIR = './parameters/'
modelparams = modelhyper()
dataparams = datahyper()
trainparams = trainhyper()

PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + '.pt'

def validate(test_loader, model, criterion):
    test_losses = []
    total=0
    model.eval()
    test_loss = 0
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



def train(trainparams, data_loader, test_loader, model):
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
        validation_loss = validate(test_loader,model,criterion_validation)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}, validation_loss(unnormalized): {validation_loss}')

        # Save Model
        if model.modeldata['minloss'] > validation_loss:
            print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(model.modeldata['minloss'], validation_loss))
            model.modeldata['minloss'] =  validation_loss
            model.modeldata['bestparam'] = model.state_dict()#potential pointer threat. but it's not an issue because the parameter is saved as a file rightafetr
            with open(PARAM_DIR,'wb') as f:
                pickle.dump(model.modeldata,f)
            
"""
if __name__ == '__main__':
    Unused
    parser = argparse.ArgumentParser(description='LGAI LDG version 3-train')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--learning_rate', type=float, default=4e-7, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train for (default: 5)")
    parser.add_argument('--weight_decay', type=float, default=4e-5, help="weight decay for (default: 0.0004)")
    args = parser.parse_args()
    # instantiate model
    
"""

def execute_train():
    model = TheModel(modelparams)
    loadmodel(model,PARAM_DIR)
    model = model.to(model.device)
    train_loader = DataLoader(ProcessDataset(dataparams,'train'), trainparams.BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(ProcessDataset(dataparams,'validation'), 3000, shuffle = True)
    # Training The Model
    train(trainparams, train_loader,test_loader, model)