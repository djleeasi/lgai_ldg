import torch
#model specific imports
class modelhyper():
    def __init__(self):
        self.MODELNAME = "LINEAR_BASELINE_"
        self.HIDDEN_SIZE = 128
        self.VECTOR_SIZE = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.OUTPUT_CLASS = 14
        self.KFOLD_NUM = 5
        self.Norm = False

class datahyper():
    def __init__(self):
        #ratio of training compared to total datase
        self.DATA_DIR_TRAIN = './data/preprocessed/train.pickle'
        self.DATA_DIR_TEST = './data/preprocessed/test.pickle'
        self.DATA_DIR_SUBMISSION = './data/rawdata/sample_submission.csv'
        self.DATA_DIR_MM = './parameters/MinMax/'
        self.DATA_DIR_RESULT = './result/'
        self.DATA_DIR_PARAMETER = './parameters/'
        self.seed = 213564#currently not used

class trainhyper():
    def __init__(self):
        self.LR = 0.001
        self.WD = 0
        self.NUM_EPOCHES = 100
        self.BATCH_SIZE = 320#작은 batchsize가 좋다.
