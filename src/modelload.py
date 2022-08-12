#refer to hy_params.py for modeldata specification.
from os.path import exists
import pickle
def loadmodel(model,PARAM_DIR):
    if exists(PARAM_DIR):
        with open(PARAM_DIR,'rb') as f:
            modeldata = pickle.load(f)
        model.load_state_dict(modeldata['bestparam'])
        model.modeldata = modeldata
        print(PARAM_DIR + " exists")
        return True
    else: 
        print(PARAM_DIR + "does not exists")
        return False