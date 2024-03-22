# path to the workspace folder and to the ROOT NTuple input
repoPath = "/work/celani/flavour_tagging/classical-taggers/"

# sample type: openVELO or closedVELO, MC2024_withUT
sample_type = 'withUT_MC_2024'

grid_n = None # For the grid search. Set to None if no grid is used
KaonCombiner = False
optimized = False

# If preSelected = False, the decision tree cuts are applied and tracks are pre-selected accordingly. Otherwise NTuples with pre-selcted tracks already exist.
preSelected = False
# Set file from which to read the pre-selection cuts
cut_file = 'test_cut'

# NN parameters
train_val_split = 0.6 # test set percentage of dataset
learning_rate = 0.0008
n_epochs = 20 #500
patience = 25 #75 #100
min_delta = 0.00
seed = 23 #11 Need to change see?
activation_function= 'ReLU' #ReLU, ELU
model_name = f'OSKaon'

# Change to False if no training is needed
training = True
# To be changed to the specific date-folder for the input model in case no training is needed.
# Format must be of type "dd_mm_yyyy:hhmmss" ex. "31_01_2024:171750" (note the "")
target_dir = "31_01_2024:171750"
