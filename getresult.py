import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from os.path import exists
#-------------------
from src.model import TheModel
from src.mydataset import TestDataset
from src.hy_params import datahyper, modelhyper
#-------------------

modelparams = modelhyper()
dataparams = datahyper()
FOLDER_DIR = dataparams.DATA_DIR_PARAMETER
PARAM_DIR = FOLDER_DIR + modelparams.MODELNAME + '.pt'

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


def execute_getresult():
    model = TheModel(modelparams)
    test_loader = DataLoader(TestDataset(dataparams), 3000, shuffle = False)
    if exists(PARAM_DIR):
        model.load_state_dict(torch.load(PARAM_DIR))
        print(PARAM_DIR + " exists")
    else:
        raise Exception("parameter file does not exist")
    model = model.to(model.device)
    test_outputs = final_test(test_loader, model).to(device = 'cpu').numpy()
    min = test_loader.dataset.minmax[2]
    max = test_loader.dataset.minmax[3]
    test_result = (test_outputs*(max-min))+min
    sub_csv = pd.read_csv(dataparams.DATA_DIR_SUBMISSION)
    sub_csv.loc[:,'Y_01':] = test_result
    sub_csv.to_csv(dataparams.DATA_DIR_RESULT + modelparams.MODELNAME + '.csv',index=False)

#----------------------------
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='LGAI LDG version 3-train')
    # parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    # parser.add_argument('--learning_rate', type=float, default=4e-7, help="Learning rate (default: 0.001)")
    # parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train for (default: 5)")
    # parser.add_argument('--weight_decay', type=float, default=4e-5, help="weight decay for (default: 0.0004)")
    # args = parser.parse_args()
    execute_getresult()
    # instantiate model