from src.model import TheModel
from src.mydataset import TestDataset
from src.hy_params import datahyper, modelhyper
from src.modelload import loadmodel
#------------------
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 
from os.path import exists
import pickle
import ipdb 
#-------------------
def result():
    FOLDER_DIR = './parameters/'
    modelparams = modelhyper()
    dataparams = datahyper()
    foldNum = modelparams.KFOLD_NUM

    with open(dataparams.DATA_DIR_MINMAX,'rb')as f:   
        _, _, y_min, y_max = pickle.load(f)
    with open(dataparams.DATA_DIR_TEST,'rb')as f:   
        testset = pickle.load(f)
    test_outputs = np.zeros((39608,14))
    for fold in range(foldNum):
        PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + f'{fold}.pt'
        test_loader = DataLoader(TestDataset(testset), 2048, shuffle = False)
        model = TheModel(modelparams)
        if loadmodel(model,PARAM_DIR) == True:
            print("loaded model for result generation")
        else:
            raise Exception("parameter file does not exist")
        model = model.to(model.device)
        test_outputs += final_test(test_loader, model).to(device = 'cpu').numpy()
    test_outputs = test_outputs/foldNum
    min = y_min
    max = y_max
    test_result = (test_outputs*(max-min))+min
    sub_csv = pd.read_csv(dataparams.DATA_DIR_SUBMISSION)
    sub_csv.loc[:,'Y_01':] = test_result
    sub_csv.to_csv(dataparams.DATA_DIR_RESULT + modelparams.MODELNAME + '.csv',index=False)

#--------------------------------
def final_test(test_loader, model):
    test_outputs = None
    total=0
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, inputdata in enumerate(tqdm(test_loader)):
            output = model(inputdata).to(device=model.device)
            total += inputdata.size(0)
            if test_outputs == None:
                test_outputs = output
            else:
                test_outputs = torch.cat((test_outputs, output), 0)
    model.train()
    return test_outputs

if __name__ == '__main__':
    result()
