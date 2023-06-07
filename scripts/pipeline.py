from dir_checker import check_directories
import sys
import numpy as np
import pyTorchTraining as pyTrain
# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

# Read from file path to the workspace folder and to the ROOT NTuple input
repoPath = np.genfromtxt(f"../config.txt", dtype = str, delimiter=",")[0]
rootPath = np.genfromtxt(f"../config.txt", dtype = str, delimiter=",")[1] # Path to be re-adjusted according to where it's your ROOT file

# Path to the ROOT file
inputPath = f"{rootPath}root/{eventType}.root:DecayTree"
# Path to where the scaler parameters will be saved
scalerPath = f"{repoPath}scaler/{eventType}/{tagger}/std_scaler.bin"

# Check whether the specified directories exist, otherwise create them
directory_list = ["calibrationPlots", "plots", "results", "root", "scaler"]
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

if "SS" in tagger:
    particle = tagger.removeprefix("SS")
    loading_variables += ["B_Tr_T_DeltaQ_" + particle]
    if particle in ("Proton", "Pion"):
        loading_variables += ["B_Tr_T_PIDP"]

# Make sure no feature is doubled 
loading_variables = np.unique(loading_variables).tolist() 

train_dl, validation_dl, test_dl = pyTrain.prepare_data(inputPath, loading_variables, scalerPath)
print(len(train_dl.dataset), len(test_dl.dataset))


## DA QUIIIIIIIIII
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))

finire modulo: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
plots: https://www.cs.toronto.edu/~lczhang/360/lec/w02/training.html