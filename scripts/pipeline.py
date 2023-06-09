from dir_checker import check_directories
import sys
import numpy as np
import pyTorchTraining as pyTrain
from NNModel import NeuralNetwork
import torch

# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

if len(sys.argv)>3: # For the grid search 
    grid_n = sys.argv[3]
    if len(sys.argv)>4:
        KaonCombiner = True
else: 
    optimized = True
# Read from file path to the workspace folder and to the ROOT NTuple input
repoPath = "/ceph/users/molocco/classical-taggers/"

# Path to the ROOT file
inputPath = f"{repoPath}root/{eventType}/{tagger}/selected.root:DecayTree"
# Path to where the scaler parameters will be saved
scalerPath = f"{repoPath}scaler/{eventType}/{tagger}/std_scaler.bin"

# Check whether the specified directories exist, otherwise create them
directory_list = ["calibrationPlots", "plots", "results", "root", "scaler", "csv", "cuts", "modelSave"]
check_directories(repoPath, directory_list, eventType, tagger)

# Definition of the features for the NN and the selection variables   
features = ["B_Tr_T_cos_diff_Phi",
        "B_Tr_T_PhiDistance",
        "B_Tr_T_PT",
        "B_Tr_T_CHI2DOF",
        "B_Tr_T_BPVIP",
        "B_Tr_T_GHOSTPROB",
        "B_Tr_T_BPVIPCHI2",
        "diff_P",
        "B_Tr_T_EtaDistance",
        "P_proj",
        "EVIP"]

selection_variables = ["entry", "label" , "B_Tr_T_PROBNN_K", "B_Tr_T_P" , "B_Tr_T_ISMUON", "B_Tr_T_BPVX" , "B_BPVX" , 
                        "B_Tr_T_BPVY" ,"B_BPVY" ,"B_Tr_T_BPVZ" , "B_BPVZ","B_Tr_T_Charge", "B_TRUEID", "B_Tr_T_absIP",
                        "B_Tr_T_PROBNN_PI", "B_Tr_T_PROBNN_P", "B_Tr_T_PROBNN_E", "B_Tr_T_PIDe", "B_Tr_T_PIDK", "B_Tr_T_PIDmu"]

# Definition of the variables that will be loaded from the NTuple
loading_variables = features + selection_variables
if "OS" in tagger:
        if optimized:
                features = features +["B_Tr_T_absIP"]
if "SS" in tagger:
    particle = tagger.removeprefix("SS")
    loading_variables += ["B_Tr_T_DeltaQ_" + particle]
    if particle in ("Proton", "Pion"):
        loading_variables += ["B_Tr_T_PIDP"]

# Make sure no feature is doubled 
loading_variables = np.unique(loading_variables).tolist() 
features = np.unique(features).tolist()
#features = features + ['label'] + ['entry']
train_dl, validation_dl, test_dl = pyTrain.prepare_data(inputPath, features, scalerPath)
print(f"Training set has {len(train_dl.dataset)} rows")
print(f"Validation set has {len(validation_dl.dataset)} rows")

model_name = 'Prova'
learning_rate = 0.001
print(len(features))
print(len(features))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork(modelName = model_name, n_features=len(features), optimizer_kwargs={"lr" : learning_rate}).to(device)
trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, repoPath, eventType, tagger, earlyStop = 2)
pyTrain.plot_losses(model, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, repoPath, eventType, tagger)

