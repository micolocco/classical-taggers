# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

# sample type: openVELO or closedVELO
sample_type = 'openVELO'

grid_n = None # For the grid search. Set to None if no grid is used
KaonCombiner = False
optimized = True

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = True

# NN parameters
train_split = 0.8 # test set percentage of dataset
learning_rate = 0.0001
n_epochs = 500
patience = 25 #75 #100
min_delta = 0.00
seed = 45 #11 Need to change see?
activation_function= 'ELU' #ReLU, ELU
model_name = f'test{train_split}_{activation_function}_{learning_rate}_{patience}_earlyNew'


# Change to False if no training is needed
training = True
# In calibration mode, all tracks are needed 
calibration = False