# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

grid_n = None # For the grid search. Set to None if no grid is used
KaonCombiner = False
optimized = True

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = True

# NN parameters
test_split = 0.1 # test set percentage of dataset
learning_rate = 0.001
n_epochs = 500
earlyStop = 30
seed = 11
model_name = f'M_seed{seed}_{learning_rate}_{n_epochs}_{earlyStop}'


# Change to False if no training is needed
training = True
# In calibration mode, all tracks are needed 
calibration = False