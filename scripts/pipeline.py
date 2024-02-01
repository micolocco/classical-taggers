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
from IPython import embed
from datetime import datetime   
import os
import dir_checker

# Decay and tagger type are given as inputs by the user 
decayType = sys.argv[1]
tagger = sys.argv[2]

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
    if config.optimized:
        features = features +["B_Tr_T_absIP"]

# Make sure no feature is doubled 
features = np.unique(features).tolist()

notSelected_rootPath = f"{config.repoPath}Data/{config.sample_type}/2_added_features/{decayType}/notSelected.root:DecayTree"
# Path to the ROOT input file
selected_rootPath = f"{config.repoPath}Data/{config.sample_type}/3_selected/{decayType}/selected.root"

# Check and eventually make output directory
dir_checker.make_dir(config.repoPath, config.sample_type, decayType)

# Directory that will be used for saving training/calibration or reading information for the calibration
if config.training:
    target_dir = datetime.now().strftime("%d_%m_%Y:%H%M%S")
    os.mkdir(f'{config.repoPath}savedModels/{config.sample_type}/{decayType}/{tagger}/{target_dir}')
else:
    target_dir = config.target_dir
    if target_dir is None:
        raise ValueError('target_dir must be set to an existing directory in configParameters.py')

target_path = f'{config.repoPath}savedModels/{config.sample_type}/{decayType}/{tagger}/{target_dir}'

# Path to where the scaler parameters will be saved
scalerPath = f"{target_path}/scaler.pkl"

# Path to where the test set will be saved
testSetPath = f"{target_path}/testSet.csv"

start = time.time()


# Reading dataset
if config.preSelected == False:
    # Definition of the variables that will be loaded from the NTuple
    cut_file = f'{config.repoPath}cuts/{config.sample_type}/{decayType}/{tagger}/{config.cut_file}'
    selection_variables = preSel.extract_selection_var(cut_file) + ['entry', 'B_TRUEID', 'B_Tr_T_Charge']
    loading_variables = features + selection_variables
    loading_variables = np.unique(loading_variables).tolist() 
    if "SS" in tagger:
        particle = tagger.removeprefix("SS")
        loading_variables += ["B_Tr_T_DeltaQ_" + particle]
        if particle in ("Proton", "Pion"):
            loading_variables += ["B_Tr_T_PIDP"]
    df = preSel.apply_preSelections(notSelected_rootPath, cut_file, loading_variables, selected_rootPath)[features + ['entry', 'B_TRUEID', 'B_Tr_T_Charge','selected_track']]
    
else:
	print(f"Reading input file: {selected_rootPath}")
	df = uproot.open(f"{selected_rootPath}:DecayTree").arrays(features + ['B_TRUEID','B_Tr_T_Charge','selected_track', 'entry'], library = "pd" )

df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
df.dropna(inplace = True)  
print(f"{df[df.selected_track==1].shape[0]} tracks among the {df.shape[0]} total tracks have been selected as tagging particles")

# Assignation of the tagging decision (d)
# d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
if ("Bd" or "Bs" in decayType) and (tagger == "SSKaon" or tagger == "SSPion" ): 
    df["TagDec"] = df[f"B_Tr_T_Charge"]
else:
    df["TagDec"] = df[f"B_Tr_T_Charge"] * (-1)

# Assignation of the label (it will be used as NN output)
# The label is given by the product of the tagging decision and the flavour charge of the B. 
# It indicates if the tagging decision is wrong or correct.
# -1 == wrong tag  1 == correct tag
df["label"] = df[f"TagDec"] * df[f"B_TRUEID"]/abs(df[f"B_TRUEID"])
df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0  
# Splitting dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
columns_to_drop = ['entry', 'selected_track', 'TagDec', 'B_TRUEID',]

# Train the model
if config.training: 
    #train_df = df[features + ['entry', 'selected_track', 'TagDec', 'B_TRUEID', 'label']].sample(frac=(config.train_split), random_state=20)
    #test_df = df[features + ['entry', 'selected_track', 'TagDec', 'B_TRUEID', 'label']].drop(train_df.index)
    train_df, test_df = pyTrain.splitByEvent(df[features + ['entry', 'selected_track', 'TagDec', 'B_TRUEID', 'label']])
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
    pyTrain.plot_features(df.iloc[train_dl.dataset.indices], target_path, name=f'{config.model_name}_inputFeatures')
    model = NeuralNetwork(modelName = config.model_name, features=features, train_batch_size = 100, test_batch_size = 1024, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, target_path, n_epochs = config.n_epochs) 
    pyTrain.plot_losses(config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)
    pyTrain.save_losses(config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)
    
    # Plot ROC curves for validation and train test
    bestModel.eval()
    yPredVal, yTrueVal = bestModel.evaluate_model(validation_dl)
    yPredTrain, yTrueTrain = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(model.modelName, yPredVal, yTrueVal, target_path, yPredTrain, yTrueTrain)
    # Fit with logistic regression and save it
    clf = pyTrain.logistic_regression(yPredTrain, yTrueTrain, target_path, model.modelName)
    #pyTrain.plot_NNoutput(config.model_name, yPredVal, yTrueVal, yPredTrain, yTrueTrain, target_path)
    pyTrain.plot_mistag(config.model_name, clf, yPredVal, yTrueVal, target_path, type = 'validation')
   

else:

    test_df = pd.read_csv(f"{testSetPath}")
    test_df = df[features + ['entry', 'selected_track', 'TagDec', 'B_TRUEID', 'label']]
    folder = "savedModels"
    #prePath = name_formatter.assign_name(folder, config.model_name)
    clf = pickle.load(open(f"{target_path}/LogReg.pck", 'rb'))

# Load the best model (ie with the lowest training loss) and evaluate it on the test set
bestModel = NeuralNetwork(modelName = config.model_name, features=features, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
pyTrain.load_model(bestModel, target_path)
bestModel.eval()
# Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
test_dataset_sel1 = inputDataset(test_df[test_df['selected_track']==1].drop(columns = columns_to_drop), scalerPath, test = True)
test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
yPredTest, yTrueTest = bestModel.evaluate_model(test_dl_sel1)
pyTrain.plot_ROC(bestModel.modelName, yPredTest, yTrueTest, target_path)
pyTrain.plot_mistag(bestModel.modelName, clf, yPredTest, yTrueTest, target_path, type = 'Test')

#     
test_dataset = inputDataset(test_df.drop(columns = columns_to_drop), scalerPath, test = True)
test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

#test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
test_df['Eta'] = 1- bestModel.evaluate_model(test_dl)[0]
test_df = test_df[['entry','selected_track', 'Eta', 'TagDec','B_TRUEID']]
#embed()

#print(test_df.loc[test_df.selected_track == 1].Eta)
plt.figure()
plt.hist(test_df.loc[test_df.selected_track == 1].Eta ,bins = 100 , density = True , histtype = "stepfilled" )
plt.savefig(f"{target_path}/etaNotnormalized.pdf")

test_df.loc[test_df.Eta > 0.5 ,"TagDec"] *= -1  
test_df.loc[test_df.Eta > 0.5, "Eta"] *= -1   
test_df.loc[test_df.Eta < 0, "Eta"] += 1   
test_df.loc[test_df.selected_track == 0, "TagDec"] = 0  # classic
test_df.loc[test_df.selected_track == 0, "Eta"] = 0.5  # classic

df_TagParticles = test_df.sort_values(by = ["entry","selected_track","Eta"] , ascending = [True,False,True]).groupby("entry").first()
#print(df_TagParticles.shape[0])

pyTrain.plot_tagDec(df_TagParticles, config.model_name, target_path)
# Calibrating the tagger and saving parameters
pyTrain.calibration(config.model_name, tagger, df_TagParticles, decayType, target_path)