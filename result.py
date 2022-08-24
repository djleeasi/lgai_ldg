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
    # foldNum = modelparams.KFOLD_NUM
    with open(dataparams.DATA_DIR_TRAIN,'rb')as f:   
        _, _, stages = pickle.load(f)
    with open(dataparams.DATA_DIR_TEST,'rb')as f:   
        testset_raw = pickle.load(f)
    # testset_raw = preXRnn_nn(testset_raw, stages)
    test_outputs = np.zeros((testset_raw.shape[0], 14))

    for fold in range(1):
        FOLDER_DIR = dataparams.DATA_DIR_PARAMETER
        PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + '.pt'
        #std 등등 불러오기
        with open(PARAM_DIR+'k','rb')as f:
            x_mean, x_std, y_mean, y_std  = pickle.load(f)
        testset = copy.deepcopy(testset_raw)
        testset = (testset-x_mean)/x_std
        test_loader = DataLoader(TestDataset(testset), 1000, shuffle = False)
        model = TheModel(modelparams)
        if exists(PARAM_DIR):
            model.load_state_dict(torch.load(PARAM_DIR))
            print(PARAM_DIR + " exists")
        else:
            raise Exception("parameter file does not exist")
        model = model.to(modelhyper.DEVICE)
        output = final_test(test_loader, model).to(device = 'cpu').numpy()
        test_output = (output*y_std)+y_mean
        test_outputs += test_output

    test_result = test_outputs
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
            inputdata = inputdata.unsqueeze(-1).to(modelhyper.DEVICE)
            output, _= model(inputdata)
            total += inputdata.size(0)
            if test_outputs == None:
                test_outputs = output
            else:
                test_outputs = torch.cat((test_outputs, output), 0)
    model.train()
    return test_outputs

if __name__ == '__main__':
    result()
