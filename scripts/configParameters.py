# path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

grid_n = None # For the grid search 
KaonCombiner = False
optimized = True

# NN parameters
model_name = 'Prova'
learning_rate = 0.0001
n_epochs=100
earlyStop = 30

# Change to False if no training is needed
training = True