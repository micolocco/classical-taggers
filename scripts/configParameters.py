# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

# sample type: openVELO or closedVELO, MC2024_withUT
sample_type = 'withUT_MC_2024'

grid_n = None # For the grid search. Set to None if no grid is used
KaonCombiner = False
optimized = False

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = True
# Set file from which to read the pre-selection cuts
cut_file = 'test_cut'

# NN parameters
train_split = 0.6 # test set percentage of dataset
learning_rate = 0.0001
n_epochs = 1 #500
patience = 25 #75 #100
min_delta = 0.00
seed = 45 #11 Need to change see?
activation_function= 'ELU' #ReLU, ELU
model_name = f'test'

# Change to False if no training is needed
training = False
# To be change to the specific date-folder for the model wanted in case no training is needed
target_dir = None
