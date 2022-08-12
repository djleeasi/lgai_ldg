#model specific imports
import torch
#----------
class modelhyper():
    def __init__(self):
        self.MODELNAME = "ATTLSTM_"
        self.MODELVERSION = '1.1'
        self.HIDDEN_SIZE = 512
        self.ATTENTION_SIZE = 64
        self.VECTOR_SIZE = 7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DROPP = 0.5
        self.OUTPUT_CLASS = 14
        self.KFOLD_NUM = 10 

class datahyper():
    def __init__(self):
        #ratio of training compared to total dataset
        self.TRAINRATIO = 0.8
        self.DATA_DIR_TRAIN = './data/preprocessed/Norm_train.pickle'
        self.DATA_DIR_MINMAX = './data/preprocessed/train_minmax.pickle'
        self.DATA_DIR_TEST = './data/preprocessed/Norm_test.pickle'
        self.DATA_DIR_SUBMISSION = './data/rawdata/sample_submission.csv'
        self.DATA_DIR_RESULT = './result/'
        self.DATA_DIR_PARAMETER = './parameters/'
        self.seed = 213564

class trainhyper():
    def __init__(self):
        self.LR = 4e-3
        self.WD = 0
        self.NUM_EPOCHES = 300
        self.BATCH_SIZE = 2048
