#모델 설명: 재선의 attention_LSTMT(real unnormalized Y)+K-fold +동준의 custom loss
#model specific imports
import torch
import numpy 
#----------
class modelhyper():
    def __init__(self):
        self.MODELNAME = "ATTLSTM_Unnorm_"
        self.MODELVERSION = '1.1'
        self.HIDDEN_SIZE = 512
        self.ATTENTION_SIZE = 64
        self.VECTOR_SIZE = 7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DROPP = 0.65
        self.OUTPUT_CLASS = 14
        self.KFOLD_NUM = 5
        self.MODELDATA = {#will be replaced with saved data
            'minloss':numpy.Inf,
            'bestparam':'Not Yet',
        }
        self.RESTORE = False #temporary variable. to be replaced with argparse


class datahyper():
    def __init__(self):
        #ratio of training compared to total dataset
        self.TRAINRATIO = 0.8
        self.DATA_DIR_TRAIN = './data/preprocessed/Norm_xonly.pickle'
        self.DATA_DIR_MINMAX = './data/preprocessed/train_minmax.pickle'
        self.DATA_DIR_TEST = './data/preprocessed/Norm_test.pickle'
        self.DATA_DIR_SUBMISSION = './data/rawdata/sample_submission.csv'
        self.DATA_DIR_RESULT = './result/'
        self.DATA_DIR_PARAMETER = './parameters/'
        self.seed = 213564

class trainhyper():
    def __init__(self):
        self.LR = 5e-3
        self.WD = 8e-9
        self.NUM_EPOCHES = 500
        self.BATCH_SIZE = 2048
