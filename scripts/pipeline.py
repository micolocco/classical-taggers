from dir_checker import check_directories
import sys
import numpy as np
import pyTorchTraining as pyTrain
from NNModel import NeuralNetwork
import torch
from configParameters import *
import preSelections as preSel
import time
import uproot
from saver import Saver
import pandas as pd

# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

# Path to the ROOT file
inputPath = f"{repoPath}root/{eventType}/{tagger}/selected.root:DecayTree"
# Path to where the scaler parameters will be saved
scalerPath = f"{repoPath}scaler/{eventType}/{tagger}/std_scaler.bin"

# Check whether the specified directories exist, otherwise create them
directory_list = ["calibrationPlots", "plots", "results", "root", "scaler", "csv", "cuts", "savedModels"]
check_directories(repoPath, directory_list, eventType, tagger)

#Definiiton of the object that will be used for saving
name_formatter = Saver(eventType, tagger, repoPath, KaonCombiner, grid_n, optimized)

# Definition of the features for the NN and the selection variables   
features = ["B_Tr_T_cos_diff_Phi",
        "B_Tr_T_PhiDistance",
        "B_Tr_T_PT",
        "B_Tr_T_CHI2DOF",
        "B_Tr_T_BPVIP",
        "B_Tr_T_GHOSTPROB",
        "B_Tr_T_IPSig",
        "diff_P",
        "B_Tr_T_EtaDistance",
        "P_proj",
        "EVIP"]

if "OS" in tagger:
    if optimized:
        features = features +["B_Tr_T_absIP"]

# Make sure no feature is doubled 
features = np.unique(features).tolist()

folder = "root"
prePath = name_formatter.assign_name(folder, "selected")
selected_rootPath = f"{prePath}.root"
start = time.time()

selection_variables = ["entry", "label" , "B_Tr_T_PROBNN_K", "B_Tr_T_P" , "B_Tr_T_ISMUON", "B_Tr_T_BPVX" , "B_BPVX" , 
                        "B_Tr_T_BPVY" ,"B_BPVY" ,"B_Tr_T_BPVZ" , "B_BPVZ","B_Tr_T_Charge", "B_TRUEID", "B_Tr_T_absIP",
                        "B_Tr_T_PROBNN_PI", "B_Tr_T_PROBNN_P", "B_Tr_T_PROBNN_E", "B_Tr_T_PIDe", "B_Tr_T_PIDK", "B_Tr_T_PIDmu"]

# Definition of the variables that will be loaded from the NTuple
loading_variables = features + selection_variables
loading_variables = np.unique(loading_variables).tolist() 
if "SS" in tagger:
    particle = tagger.removeprefix("SS")
    loading_variables += ["B_Tr_T_DeltaQ_" + particle]
    if particle in ("Proton", "Pion"):
        loading_variables += ["B_Tr_T_PIDP"]

# Reading dataset
if preSelected == False:
    print("Applying pre-selections")
    df = preSel.apply_preSelections(loading_variables + ['selected_track'], selected_rootPath, name_formatter, repoPath, eventType, tagger)
    
else:
	print(f"Reading input file")
	df = uproot.open(f"{selected_rootPath}:DecayTree").arrays(loading_variables + ['selected_track'], library = "pd" )
    
# Splitting dataset
df = df.sample(frac=1).reset_index(drop=True)
df.dropna(inplace = True)  
train_dl, validation_dl, test_dl = pyTrain.prepare_data(df[features + ['label', 'selected_track']], scalerPath)
print(f"Training set has {len(train_dl.dataset)} rows")
print(f"Validation set has {len(validation_dl.dataset)} rows")
print(f"Test set has {len(test_dl.dataset)} rows")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Train the model
if training: 
    model = NeuralNetwork(modelName = model_name, features=features, train_batch_size = 100, test_batch_size = 1024, optimizer_kwargs={"lr" : learning_rate}).to(device)
    trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, name_formatter, n_epochs=100, earlyStop = 5) # earlyStop must be < n_epochs
    pyTrain.plot_losses(model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, name_formatter)
    pyTrain.save_losses(model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, name_formatter)



# Load the best model (ie with the lowest training loss) 
bestModel = NeuralNetwork(modelName = model_name, features=features, optimizer_kwargs={"lr" : learning_rate}).to(device)
pyTrain.load_model(bestModel, name_formatter)
bestModel.eval()
yPredTest, yTrueTest = bestModel.evaluate_model(test_dl)
yPredTrain, yTrueTrain = bestModel.evaluate_model(train_dl)
# Plot the ROC curve for both, training and test sets
pyTrain.plot_ROC(model_name, yPredTest, yTrueTest, yPredTrain, yTrueTrain, name_formatter)
pyTrain.plot_NNoutput(model_name, yPredTest, yTrueTest, yPredTrain, yTrueTrain, name_formatter)
#pyTrain.mistag(model_name, bestModel, yPredTest, yTrueTest, yPredTrain, yTrueTrain, name_formatter)

# Calibrtaion of the tagger. 
test_indices = test_dl.dataset.indices
df_val = df.iloc[test_indices]
# Add equal amount of selected_track = 0 
#df_val = pd.concat([df_val,df[df.selected_track==0][:len(test_indices)]])
#print(df_val.shape[0])
df_val["eta"] = yPredTest.tolist()
df_val["TagDec"] = df_val[f"B_Tr_T_Charge"] *-1
df_vall.loc[df_vall.Eta > 0.5, "Eta"] *= -1   # classic
df_vall.loc[df_vall.Eta < 0, "Eta"] += 1   # classic