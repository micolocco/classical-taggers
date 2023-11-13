# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

# sample type: openVELO or closedVELO
sample_type = 'closedVELO'

grid_n = None # For the grid search. Set to None if no grid is used
KaonCombiner = False
optimized = True

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = False

# NN parameters
test_split = 0.3 # test set percentage of dataset
learning_rate = 0.001
n_epochs = 500
earlyStop = 75 #75 #100
seed = 42 #11 
model_name = f'test_{test_split}'


# Change to False if no training is needed
training = True
# In calibration mode, all tracks are needed 
calibration = False