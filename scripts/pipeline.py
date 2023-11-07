from dir_checker import check_directories
import sys
import numpy as np
import pyTorchTraining as pyTrain
from NNModel import NeuralNetwork
import torch
import configParameters as config 
import preSelections as preSel
import time
import uproot
from saver import Saver
import pandas as pd
from sklearn.model_selection import train_test_split
from inputDataset import inputDataset
from torch.utils.data import DataLoader
import pickle
from matplotlib import pyplot as plt



# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]
tagger = sys.argv[2]

# Check whether the specified directories exist, otherwise create them
directory_list = ["calibrationPlots", "plots", "results", "root", "scaler", "csv", "cuts", "savedModels"]
check_directories(config.repoPath, directory_list, eventType, tagger)

#Definiton of the object that will be used for saving in the right name format
name_formatter = Saver(eventType, tagger, config.repoPath, config.KaonCombiner, config.grid_n, config.optimized)

# Definition of the features for the NN and the selection variables   
features = ["Bp_Tr_T_cos_diff_Phi",
        "Bp_Tr_T_PhiDistance",
        "Bp_Tr_T_PT",
        "Bp_Tr_T_CHI2DOF",
        "Bp_Tr_T_BPVIP",
        "Bp_Tr_T_GHOSTPROB",
        "Bp_Tr_T_IPSig",
        "diff_P",
        "Bp_Tr_T_EtaDistance",
        "P_proj",
        "EVIP"]

if "OS" in tagger:
    if config.optimized:
        features = features +["Bp_Tr_T_absIP"]

# Make sure no feature is doubled 
features = np.unique(features).tolist()

# Path to the ROOT input file
folder = "root"
rootPrePath = name_formatter.assign_name(folder,f"{config.sample_type}_selected")
selected_rootPath = f"{rootPrePath}.root"

# Path to where the scaler parameters will be saved
folder = "scaler"
scalerPrePath = name_formatter.assign_name(folder, f"{config.sample_type}_stdScaler")
scalerPath = f"{scalerPrePath}.pkl"

# Path to where the test set will be saved
folder = "csv"
testSetPrePath = name_formatter.assign_name(folder, f"{config.sample_type}_testSet")
testSetPath = f"{testSetPrePath}.csv"

selection_variables = ["entry", "Bp_Tr_T_PROBNN_K", "Bp_Tr_T_P" , "Bp_Tr_T_ISMUON", "Bp_Tr_T_BPVX" , "Bp_BPVX" , 
                        "Bp_Tr_T_BPVY" ,"Bp_BPVY" ,"Bp_Tr_T_BPVZ" , "Bp_BPVZ","Bp_Tr_T_Charge", "Bp_TRUEID", "Bp_Tr_T_absIP",
                        "Bp_Tr_T_PROBNN_PI", "Bp_Tr_T_PROBNN_P", "Bp_Tr_T_PROBNN_E", "Bp_Tr_T_PIDe", "Bp_Tr_T_PIDK", "Bp_Tr_T_PIDmu"]

start = time.time()

# Definition of the variables that will be loaded from the NTuple
loading_variables = features + selection_variables
loading_variables = np.unique(loading_variables).tolist() 
if "SS" in tagger:
    particle = tagger.removeprefix("SS")
    loading_variables += ["Bp_Tr_T_DeltaQ_" + particle]
    if particle in ("Proton", "Pion"):
        loading_variables += ["Bp_Tr_T_PIDP"]

# Reading dataset
if config.preSelected == False:
    print("Applying pre-selections")
    df = preSel.apply_preSelections(loading_variables, selected_rootPath, name_formatter, config.repoPath, eventType, tagger)[features + ['entry', 'Bp_TRUEID', 'Bp_Tr_T_Charge','selected_track']]
    
else:
	print(f"Reading input file")
	df = uproot.open(f"{selected_rootPath}:DecayTree").arrays(features + ['Bp_TRUEID','Bp_Tr_T_Charge','selected_track', 'entry'], library = "pd" )

df = df.sample(frac=1).reset_index(drop=True)
df.dropna(inplace = True)  
print(f"{df[df.selected_track==1].shape[0]} tracks among the {df.shape[0]} total tracks have been selected as tagging particles")

# Assignation of the tagging decision (d)
# d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
if ("Bd" or "Bs" in eventType) and (tagger == "SSKaon" or tagger == "SSPion" ): 
    df["TagDec"] = df[f"Bp_Tr_T_Charge"]
else:
    df["TagDec"] = df[f"Bp_Tr_T_Charge"] * (-1)

# Assignation of the label (it will be used as NN output)
# The label is given by the product of the tagging decision and the flavour charge of the B. 
# It indicates if the tagging decision is wrong or correct.
# -1 == wrong tag  1 == correct tag
df["label"] = df[f"TagDec"] * df[f"Bp_TRUEID"]/abs(df[f"Bp_TRUEID"])
df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0  

# Splitting dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
columns_to_drop = ['entry', 'selected_track', 'TagDec', 'Bp_TRUEID',]

# Train the model
if config.training: 
    #train_df, test_df = train_test_split(df[features + ['entry', 'selected_track', 'TagDec', 'Bp_TRUEID', 'label']], test_size=0.3)    
    train_df = df[features + ['entry', 'selected_track', 'TagDec', 'Bp_TRUEID', 'label']].sample(frac=config.test_split, random_state=200)
    test_df = df[features + ['entry', 'selected_track', 'TagDec', 'Bp_TRUEID', 'label']].drop(train_df.index)
    # Save test dataframe for calibration
    test_df.to_csv(f"{testSetPath}", index = False)

    train_dl, validation_dl = pyTrain.prepare_data(train_df[train_df['selected_track']==1].drop(columns = columns_to_drop), scalerPath)
    print(f"Training set has {len(train_dl.dataset)} rows")
    print(f"Validation set has {len(validation_dl.dataset)} rows")

    '''
    train_indices = train_dl.dataset.indices
    df1 = df.iloc[train_indices][['label','entry']]
    print(f' label 0 : {df1[df1.label==0].shape[0]}, label 1 : {df1[df1.label==1].shape[0]}')
    '''

    pyTrain.plot_features(df.iloc[train_dl.dataset.indices], name_formatter, name='input_features')
    model = NeuralNetwork(modelName = config.model_name, features=features, train_batch_size = 100, test_batch_size = 1024, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, name_formatter, n_epochs = config.n_epochs, earlyStop = config.earlyStop) # earlyStop must be < n_epochs
    pyTrain.plot_losses(config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, name_formatter)
    pyTrain.save_losses(config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, name_formatter)
    
    # Plot ROC curves for validation and train test
    bestModel.eval()
    yPredVal, yTrueVal = bestModel.evaluate_model(validation_dl)
    yPredTrain, yTrueTrain = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(model.modelName, yPredVal, yTrueVal, name_formatter, yPredTrain, yTrueTrain)
    # Fit with logistic regression and save it
    clf = pyTrain.logistic_regression(yPredTrain, yTrueTrain, name_formatter, model.modelName)
    #pyTrain.plot_NNoutput(config.model_name, yPredVal, yTrueVal, yPredTrain, yTrueTrain, name_formatter)
    pyTrain.plot_mistag(config.model_name, clf, yPredVal, yTrueVal, name_formatter, type = 'validation')

else:
    test_df = pd.read_csv(f"{testSetPath}")
    folder = "savedModels"
    prePath = name_formatter.assign_name(folder, config.model_name)
    clf = pickle.load(open(f"{prePath}_LogReg.pck", 'rb'))

# Load the best model (ie with the lowest training loss) and evaluate it on the test set
bestModel = NeuralNetwork(modelName = config.model_name, features=features, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
pyTrain.load_model(bestModel, name_formatter)
bestModel.eval()
# Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
test_dataset_sel1 = inputDataset(test_df[test_df['selected_track']==1].drop(columns = columns_to_drop), scalerPath, test = True)
test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
yPredTest, yTrueTest = bestModel.evaluate_model(test_dl_sel1)
pyTrain.plot_ROC(bestModel.modelName, yPredTest, yTrueTest, name_formatter)
pyTrain.plot_mistag(bestModel.modelName, clf, yPredTest, yTrueTest, name_formatter, type = 'Test')

#     
test_dataset = inputDataset(test_df.drop(columns = columns_to_drop), scalerPath, test = True)
test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]

test_df = test_df[['entry','selected_track', 'Eta', 'TagDec','Bp_TRUEID']]


print(test_df.loc[test_df.selected_track == 1].Eta)
plt.figure()
plt.hist(test_df.loc[test_df.selected_track == 1].Eta ,bins = 100 , density = True , histtype = "stepfilled" )
plt.savefig(f"debug/{config.sample_type}_etaNotnormalized.pdf")

test_df.loc[test_df.Eta > 0.5 ,"TagDec"] *= -1  
test_df.loc[test_df.Eta > 0.5, "Eta"] *= -1   
test_df.loc[test_df.Eta < 0, "Eta"] += 1   
test_df.loc[test_df.selected_track == 0, "TagDec"] = 0  # classic
test_df.loc[test_df.selected_track == 0, "Eta"] = 0.5  # classic

plt.figure()
plt.hist(test_df.loc[test_df.selected_track == 1].Eta ,bins = 100 , density = True , histtype = "stepfilled" )
plt.savefig(f"debug/{config.sample_type}_eta.pdf")
#print(test_df[ (test_df[ "Eta"] == 0.4930005622788447)])
#print(test_df.shape[0])
#print(test_df[test_df.selected_track == 1].shape[0])

df_TagParticles = test_df.sort_values(by = ["entry","selected_track","Eta"] , ascending = [True,False,True]).groupby("entry").first()
#print(df_TagParticles.shape[0])

pyTrain.plot_tagDec(df_TagParticles, config.model_name, name_formatter)
# Calibrating the tagger and saving parameters
pyTrain.calibration(config.model_name, tagger, df_TagParticles, eventType, name_formatter)