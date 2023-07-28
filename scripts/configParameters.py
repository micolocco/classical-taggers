# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

grid_n = None # For the grid search 
KaonCombiner = False
optimized = True

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = True

# NN parameters
model_name = 'Prova'
learning_rate = 0.0001
n_epochs=100
earlyStop = 5

# Change to False if no training is needed
training = False
# In calibration mode, all tracks are needed 
calibration = False