import torch
#model specific imports
class modelhyper():
    def __init__(self):
        self.MODELNAME = "LINEAR_BASELINE_BS_"
        self.HIDDEN_SIZE = 128
        self.VECTOR_SIZE = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.OUTPUT_CLASS = 14
        self.KFOLD_NUM = 5
        self.Norm = False

class datahyper():
    def __init__(self):
        #ratio of training compared to total dataset
        self.TRAINRATIO = 0.8
        self.DATA_DIR_TRAIN = './data/preprocessed/ntrain.pickle'
        self.DATA_DIR_TEST = './data/preprocessed/test.pickle'
        self.DATA_DIR_SUBMISSION = './data/rawdata/sample_submission.csv'
        self.DATA_DIR_RESULT = './result/'
        self.DATA_DIR_PARAMETER = './parameters/'
        self.seed = 4231

class trainhyper():
    def __init__(self):
        self.LR = 0.001
        self.WD = 0
        self.NUM_EPOCHES = 50
        self.BATCH_SIZE = 320 #작은 batchsize가 좋다.
