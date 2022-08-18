import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from os.path import exists
import pickle
import ipdb 
import copy
#------model specific imports-------------------
from src.model import TheModel
from src.mydataset import TestDataset
from src.hy_params import datahyper, modelhyper
from src.prepro import *


def result():
    modelparams = modelhyper()
    dataparams = datahyper()
    foldNum = modelparams.KFOLD_NUM
    with open(dataparams.DATA_DIR_TRAIN,'rb')as f:   
        _, train_y, stages = pickle.load(f)
    datas = pd.read_csv('data/rawdata/test.csv')
    datas = datas.drop(columns='ID')
    testset_raw = datas.iloc[:, :56].values
    test_outputs = np.zeros((testset_raw.shape[0], train_y.shape[1]))
    del train_y
    for fold in range(foldNum):
        # fold 별 min max 불러오기
        FOLDER_DIR = dataparams.DATA_DIR_PARAMETER
        PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + f'{fold}.pt'
        testset = copy.deepcopy(testset_raw)
        print(np.mean(testset, axis = 0))
        test_loader = DataLoader(TestDataset(testset, mode=False), 2048, shuffle = False)
        model = TheModel(modelparams)
        if exists(PARAM_DIR):
            model.load_state_dict(torch.load(PARAM_DIR))
            print(PARAM_DIR + " exists")
        else:
            raise Exception("parameter file does not exist")
        model = model.to(model.device)
        test_output = final_test(test_loader, model).to(device = 'cpu').numpy()
        test_outputs += test_output

    test_result = test_outputs/foldNum
    sub_csv = pd.read_csv(dataparams.DATA_DIR_SUBMISSION)
    sub_csv.loc[:,'Y_01':] = test_result
    sub_csv.to_csv(dataparams.DATA_DIR_RESULT + modelparams.MODELNAME + '.csv',index=False)

#--------------------------------
def final_test(test_loader, model):
    test_outputs = None
    total=0
    model.eval()
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
