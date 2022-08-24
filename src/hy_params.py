import torch
#model specific imports
class modelhyper():
    MODELNAME = 'SimCLD'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    E_SEQLENGTH = 52
    E_INPUTSIZE = 24#무조건 E_NHEAD 로 나눠져야 함
    E_DROPOUT = 0.1
    E_NHEAD = 4
    E_DIM_FEEDFORWARD = 256
    E_LAYER_NUM = 10
    L_DROPOUT = 0.51



class datahyper():
    TRAINRATIO = 0.8
    DATA_DIR_TRAIN = './data/preprocessed/train.pickle'
    DATA_DIR_TEST = './data/preprocessed/test.pickle'
    DATA_DIR_MM = './parameters/MinMax/'
    DATA_DIR_SUBMISSION = './data/rawdata/sample_submission.csv'
    DATA_DIR_RESULT = './result/'
    DATA_DIR_PARAMETER = './parameters/'
    SEED = 4242

class trainhyper():
    LR = 5e-5
    WD = 0.3
    TEMPERATURE = 0.05
    NUM_EPOCHES = 1000
    BATCH_SIZE = 64# 작은 batchsize가 좋다.
