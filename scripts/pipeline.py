import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import uproot
import pandas as pd
from inputDataset import inputDataset
import pickle
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
from pprint import pprint

# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)



def stats_printout(df, train_df, val_df, test_df):
    '''
    Function to print statistics about the dataset composition
    '''
    from rich.console import Console
    from rich.table import Table

    tot_evts = len(df['event_entry'].unique())
    sel_evts = len(df[df.selected==1]['event_entry'].unique())
    train_evts =  len(train_df['event_entry'].unique())
    val_evts =  len(val_df['event_entry'].unique())
    test_evts_sel =  len(test_df[test_df.selected==1]['event_entry'].unique())
    test_evts =  len(test_df['event_entry'].unique())

    print("\n Statistics used in the pipeline\n")

    console = Console()

    table = Table(show_header=True)
    table.add_column("Sets", justify="left", style='cyan')
    table.add_column("Events", justify="right", style="green")
    table.add_column("Tracks", justify="right", style="magenta")

    table.add_row("Before selection", f"{tot_evts}", f"{df.shape[0]}")
    table.add_row("After selection", f"{sel_evts}", f"{df[df.selected==1].shape[0]}")
    table.add_row("Train", f"{train_evts}", f"{train_df.shape[0]}")
    table.add_row("Validation", f"{val_evts}", f"{val_df.shape[0]}",)
    table.add_row("Calibration (only selected)", f"{test_evts_sel}", f"{test_df[test_df.selected==1].shape[0]}")
    table.add_row("Calibration (total)", f"{test_evts}", f"{test_df.shape[0]}")
    console.print(table)
    print("\nThe train and the validation sets are made of tracks passing the preselection.")
    print("The calibration set contains both selected and not selected events. \n")

def filter_rows(group):
    '''
    Function to avoid duplication of events due to multicandidates
    '''
    return group[group['entry']==group['entry'].unique()[0]]

# Decay and tagger type are given as inputs by the user
# Decay must be one among Bd2JpsiKst,  Bs2DsPi,  Bu2JpsiK 
# Taggers must be one among OSKaon, OSMuon, OSElectron, SSPion, SSProton, SSKaon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selected', help='File with preselection applied', nargs='+')
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon')) # add all the possible taggers
    parser.add_argument('--config', help='Config json', type=str) # add all the possible taggers
    parser.add_argument('--decayType', help='Config json', type=str) # add all the possible taggers

    cfg = parser.parse_args()
    pprint(cfg)

    if "OS" in cfg.tagger:
        if pyTrain.config.optimized:
            features = pyTrain.features +["B_Tr_T_absIP"]

    # Make sure no feature is doubled
    features = np.unique(pyTrain.features).tolist()
    # Path to the ROOT input file
    selected_files = cfg.selected

    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)
    # Path to where the scaler parameters will be saved
    scalerPath = f"{cfg.target_path}/scaler.pkl"
    # Path to where the test set will be saved
    testSetPath = f"{cfg.target_path}/testSet.csv"

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns_to_drop = ['event_entry', 'selected', 'TagDec', 'B_TRUEID',]

    # Reading datasets
    vars = features + ['B_TRUEID','B_Tr_T_Charge','selected', 'entry', 'RUNNUMBER', 'EVENTNUMBER']
    
    df = pd.DataFrame(columns=vars)
    for f in selected_files:
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(vars, library="pd")
        _df.dropna(inplace = True)
        df = (_df.copy() if df.empty else pd.concat([df, _df], ignore_index = True))
    df.sample(frac=1, random_state=pyTrain.config.seed).reset_index(drop=True)
    print("Removing multicandidates")
    df_grouped = df.groupby(['RUNNUMBER', 'EVENTNUMBER'])
    df = df_grouped.apply(filter_rows).drop(columns = ['RUNNUMBER', 'EVENTNUMBER']) #Drop multicandidates
    df.reset_index(inplace=True)
    df['event_entry'] = df.groupby(['RUNNUMBER', 'EVENTNUMBER']).ngroup() # in the concatenation the entries are the same among different files, needed to look at evt and run number to identify them
    df.drop(columns=['entry', 'level_2'], inplace=True)
    embed()
    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" or "Bs" in cfg.decayType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
        df["TagDec"] = df[f"B_Tr_T_Charge"]
    else:
        df["TagDec"] = df[f"B_Tr_T_Charge"] * (-1)

    # Assignation of the label (it will be used as NN output)
    # The label is given by the product of the tagging decision and the flavour charge of the B.
    # It indicates if the tagging decision is wrong or correct.
    # -1 == wrong tag  1 == correct tag
    df["label"] = df[f"TagDec"] * df[f"B_TRUEID"]/abs(df[f"B_TRUEID"])
    df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0

    plt.figure(figsize=(24,25))
    pos=0
    for i, col in enumerate(df.columns.to_list()):
        if col in features:
            plt.subplot(4, 3, pos + 1)
            plt.hist(df[col][df['label']==0][df['selected']==1], density = True, bins=100, label = "post select, label = 0", color='r', alpha=0.5, range=ranges[col])
            plt.hist(df[col][df['label']==1][df['selected']==1], density = True, bins=100, label = "post select, label = 1", color='r', alpha=0.2, range=ranges[col])
            plt.hist(df[col][df['label']==0], density = True, bins=100, label = "label = 0",color='b', alpha=0.5, range=ranges[col])
            plt.hist(df[col][df['label']==1], density = True, bins=100, label = "label = 1",color='b', alpha=0.2, range=ranges[col])
            #plt.hist(df[col], density = True, bins=100, color='r', alpha=0.5)
            plt.legend()
            plt.xlabel(nice_names[col])
            plt.tight_layout()
            pos+=1
    plt.savefig(f"{cfg.target_path}/preSelect_variables.pdf")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns_to_drop = ['event_entry', 'selected', 'TagDec', 'B_TRUEID',]
    df_selected = df.query('selected==1')[features + ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'label']]
    df_not_selected = df.query('selected==0')[features + ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'label']]
    # Split into train, validation, test sets
    train_df, val_df, test_df = pyTrain.splitByEvent(df_selected)
    # Add tracks that don't pass preselection to the test set (needed for calibration)
    test_df = pd.concat([test_df, df_not_selected], ignore_index =True)
    stats_printout(df, train_df, val_df, test_df)
    
    # Save test dataframe for calibration
    test_df.to_csv(f"{testSetPath}", index = False)
    train_df.drop(columns = columns_to_drop, inplace = True)
    val_df.drop(columns = columns_to_drop, inplace = True)
    train_dl, validation_dl = pyTrain.prepare_data(train_df=train_df, val_df=val_df, savePlot_path=cfg.target_path, scalerPath=scalerPath)


    
   
    model = NeuralNetwork(modelName = pyTrain.config.model_name, features=features, train_batch_size = 100, test_batch_size = 1024, optimizer_kwargs={"lr" : pyTrain.config.learning_rate}).to(device)
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, cfg.target_path, n_epochs = pyTrain.config.n_epochs)
    pyTrain.plot_losses(pyTrain.config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    pyTrain.save_losses(pyTrain.config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    
    # Plot ROC curves for validation and train test
    bestModel.eval()
    yPredVal, yTrueVal = bestModel.evaluate_model(validation_dl)
    yPredTrain, yTrueTrain = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(model.modelName, yPredVal, yTrueVal, cfg.target_path, yPredTrain, yTrueTrain)
    # Fit with logistic regression and save it
    clf = pyTrain.logistic_regression(yPredTrain, yTrueTrain, cfg.target_path, model.modelName)
    #pyTrain.plot_NNoutput(pyTrain.config.model_name, yPredVal, yTrueVal, yPredTrain, yTrueTrain, target_path)
    pyTrain.plot_mistag(pyTrain.config.model_name, clf, yPredVal, yTrueVal, cfg.target_path, type = 'validation')


    # else:
    '''
    test_df = pd.read_csv(f"{testSetPath}")

    clf = pickle.load(open(f"{cfg.target_path}/LogReg.pck", 'rb'))   
    #Load the best model (ie with the lowest training loss) and evaluate it on the test set
    bestModel = NeuralNetwork(modelName = pyTrain.config.model_name, features=features, optimizer_kwargs={"lr" : pyTrain.config.learning_rate}).to(device)
    pyTrain.load_model(bestModel, cfg.target_path)
    bestModel.eval()
    '''
    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    test_dataset_sel1 = inputDataset(test_df[test_df['selected']==1].drop(columns = columns_to_drop), scalerPath, test = True)
    test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
    print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
    yPredTest, yTrueTest = bestModel.evaluate_model(test_dl_sel1)
    pyTrain.plot_ROC(bestModel.modelName, yPredTest, yTrueTest, cfg.target_path)
    pyTrain.plot_mistag(bestModel.modelName, clf, yPredTest, yTrueTest, cfg.target_path, type = 'Test')

    #
    test_dataset = inputDataset(test_df.drop(columns = columns_to_drop), scalerPath, test = True)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['Eta'] = 1- bestModel.evaluate_model(test_dl)[0]
    test_df = test_df[['event_entry','selected', 'Eta', 'TagDec','B_TRUEID']]
    #embed()

    #print(test_df.loc[test_df.selected == 1].Eta)    
    plt.figure()
    plt.hist(test_df.loc[test_df.selected == 1].Eta ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.xlabel(r"$\eta$ Normalised")
    plt.savefig(f"{cfg.target_path}/etaNotnormalized.pdf")

    test_df.loc[test_df.Eta > 0.5 ,"TagDec"] *= -1
    test_df.loc[test_df.Eta > 0.5, "Eta"] *= -1
    test_df.loc[test_df.Eta < 0, "Eta"] += 1
    test_df.loc[test_df.selected == 0, "TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, "Eta"] = 0.5  # classic

    df_TagParticles = test_df.sort_values(by = ["event_entry","selected","Eta"] , ascending = [True,False,True]).groupby("event_entry").first()
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating")
    pyTrain.plot_tagDec(df_TagParticles, pyTrain.config.model_name, cfg.target_path)
    # Calibrating the tagger and saving parameters
    pyTrain.calibration(pyTrain.config.model_name, cfg.tagger, df_TagParticles, cfg.decayType, cfg.target_path)