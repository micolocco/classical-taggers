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
from scripts import ranges, nice_names
import mplhep as hep
hep.style.use("LHCb2")
# matplotlib_lhcb_style(plt)


def plot_mothers(df, mother, savepath, type, tagger):
    names = {
        'B_Tr_T_MC_MOTHER_ID': r'TRUEID Mother$(tag)$',
        'B_Tr_T_MC_GD_MOTHER_ID': r'TRUEID GDMother$(tag)$',
    }
    df_grouped = df.groupby(mother)
    ids = []
    counts = []
    for key, group in df_grouped:
        ids.append(int(key))
        counts.append(len(group))
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(ids))*3+3, counts)
    plt.xticks(np.arange(len(ids))*3+3, ids, rotation=45, fontsize=6)
    plt.xlim(0, len(ids)*3+3)
    plt.xlabel(names[mother])
    plt.minorticks_off()
    plt.legend(title=f'{tagger} {type} set protons')
    plt.savefig(f'{savepath}/{type}_tagProtons_{mother}.pdf')
    plt.close()
    plt.show()

def plot_kp_for_torch(df, type, savepath, tagger):
    true_kaons = df.query('B_Tr_T_TRUEID==321 or B_Tr_T_TRUEID==-321')
    true_protons = df.query('B_Tr_T_TRUEID==2212 or B_Tr_T_TRUEID==-2212')
    print(f'In the {type} samples there are {len(true_kaons)} true kaons and {len(true_protons)} true protons, out of a total {len(df)} tracks\n')
    plt.hist(true_kaons['B_Tr_T_P'], range=ranges['B_Tr_T_P'], bins=100, density=True, label='True kaons')
    plt.hist(true_protons['B_Tr_T_P'], range=ranges['B_Tr_T_P'], bins=100, density=True, label='True protons', alpha=0.8)
    plt.xlabel(r'$p(tag)~[\mathrm{MeV}/c^2]$')
    plt.legend(title=f'{tagger} {type} set')
    plt.savefig(f'{savepath}/{type}_pk.pdf')
    plt.close()

    plt.hist(true_protons.loc[df.label == 0].B_Tr_T_P, range=ranges['B_Tr_T_P'], bins=100, density=True, label='True protons wrong tag')
    plt.hist(true_protons.loc[df.label == 1].B_Tr_T_P, range=ranges['B_Tr_T_P'], bins=100, density=True, label='True protons correct tag', alpha=0.8)
    plt.xlabel(r'$p(tag)~[\mathrm{MeV}/c^2]$')
    plt.legend(title=f'{tagger} {type} set')
    plt.savefig(f'{savepath}/{type}_p_labels.pdf')
    plt.close()

    plt.hist(true_kaons.loc[df.label == 0].B_Tr_T_P, range=ranges['B_Tr_T_P'], bins=100, density=True, label='True kaons wrong tag')
    plt.hist(true_kaons.loc[df.label == 1].B_Tr_T_P, range=ranges['B_Tr_T_P'], bins=100, density=True, label='True kaons correct tag', alpha=0.8)
    plt.xlabel(r'$p(tag)~[\mathrm{MeV}/c^2]$')
    plt.legend(title=f'{tagger} {type} set')
    plt.savefig(f'{savepath}/{type}_k_labels.pdf')
    plt.close()
    plt.show()

    df_grouped = df.groupby("B_Tr_T_TRUEID")
    ids = []
    counts = []
    for key, group in df_grouped:
        ids.append(int(key))
        counts.append(len(group))
    plt.bar(np.arange(len(ids)) + 3.5, counts)
    plt.xticks(np.arange(len(ids)) + 3.5, ids, rotation=45, fontsize=15)
    plt.xlabel(r'TRUEID$(tag)$')
    plt.minorticks_off()
    plt.legend(title=f'{tagger} {type} set')
    plt.savefig(f'{savepath}/{type}_TRUEIDtag.pdf')
    plt.close()


    # plot_mothers(true_protons, 'B_Tr_T_MC_MOTHER_ID', savepath, type, tagger=cfg.tagger)
    # plot_mothers(true_protons, 'B_Tr_T_MC_GD_MOTHER_ID', savepath, type, tagger=cfg.tagger)


def plot_mistag_for_torch(df, savepath, tagger):
    df_mistag = df.query('Eta>0.5')
    df_grouped = df_mistag.groupby(['B_Tr_T_TRUEID'])
    ids = []
    counts = []
    for key, group in df_grouped:
        ids.append(int(key))
        counts.append(len(group))
    plt.bar(np.arange(len(ids)) + 2.5, counts, label=r'TRUEID for tracks with $\eta>0.5$')
    plt.xticks(np.arange(len(ids)) + 2.5, ids, rotation=45, fontsize=15)
    # plt.hist(df['B_Tr_T_TRUEID'], bins=20, label='Tag TRUE ID', range=(0,5000))
    plt.xlabel(r'TRUEID$(tag)$')
    plt.legend(title=f'{tagger}')
    plt.minorticks_off()
    plt.savefig(f'{savepath}/mistag_tagtrueid.pdf')
    plt.close()
    plt.show()

    true_kaons = df_mistag.query('B_Tr_T_TRUEID==321 or B_Tr_T_TRUEID==-321')
    true_protons = df_mistag.query('B_Tr_T_TRUEID==2212 or B_Tr_T_TRUEID==-2212')
    plt.hist(true_kaons['B_Tr_T_P'], range=ranges['B_Tr_T_P'], bins=50, density=True, label=r'True kaons with $\eta>0.5$')
    plt.hist(true_protons['B_Tr_T_P'], range=ranges['B_Tr_T_P'], bins=50, density=True, label=r'True protons with $\eta>0.5$', alpha=0.8)
    plt.xlabel(r'$p(tag)~[\mathrm{MeV}/c^2]$')
    plt.legend(title=f'{tagger}')
    plt.savefig(f'{savepath}/pk_mistag.pdf')
    plt.close()

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

    table = Table(show_header=True)
    table.add_column("", justify="left")
    table.add_column("correct tagging decision: 1", justify="left", style='cyan')
    table.add_column("wrong tagging decision: 0", justify="left", style='green')
    table.add_row("Training set", f"{train_df[train_df.label==1].shape[0]}", f"{train_df[train_df.label==0].shape[0]}")
    table.add_row("Test set", f"{test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]}", f"{test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]}")
    console.print(table)

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
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--seed', help='Random seed', default=2) 
    parser.add_argument('--config', help='Config json', type=str) 
    parser.add_argument('--decayType', help='Event decay', type=str) 

    cfg = parser.parse_args()
    pprint(cfg)
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file='scripts/tagger_features.yaml')
    # Path to the ROOT input file
    selected_files = cfg.selected
    print(f"The features used are: {features}")
    # Check and eventually make output directory where training info will be saved
    os.makedirs(cfg.target_path, exist_ok=True)
    # Path to where the scaler parameters will be saved
    scalerPath = f"{cfg.target_path}/scaler.pkl"
    # Path to where the test set will be saved
    testSetPath = f"{cfg.target_path}/testSet.csv"
    torchPath = f"{cfg.target_path}/torchPath"
    os.makedirs(torchPath, exist_ok=True)

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns_to_drop = ['event_entry', 'selected', 'TagDec', 'B_TRUEID',]

    # Reading datasets
    vars = features + ['B_TRUEID','B_Tr_T_Charge','selected', 'entry', 'RUNNUMBER', 'EVENTNUMBER', "B_Tr_T_TRUEID"]
    
    df = pd.DataFrame(columns=vars)
    for f in selected_files:
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(vars, library="pd")
        _df.dropna(inplace = True)
        df = (_df.copy() if df.empty else pd.concat([df, _df], ignore_index = True))
    if 'SS' in cfg.tagger:
        _features = []
        to_drop = []
        for f in features:
            if 'PID' in f or 'GHOST' in f or 'EtaDist' in f or 'nPV' in f:
                _features.append(f)
            else:
                _features.append('log('+f+')')
                if f != 'B_Tr_T_P': to_drop.append(f)
                df['log('+f+')'] = np.log(np.abs(df[f]))
        print(to_drop)
        df.drop(columns=to_drop, inplace=True)
        features = _features
    df.sample(frac=1, random_state=pyTrain.config.seed).reset_index(drop=True)
    print("Removing multicandidates")
    removal_time1 = time.time()
    df_grouped = df.groupby(['RUNNUMBER', 'EVENTNUMBER'])
    # df = df_grouped.apply(filter_rows).drop(columns = ['RUNNUMBER', 'EVENTNUMBER']) #Drop multicandidates
    df.reset_index(inplace=True)
    removal_time2 = round((time.time()- removal_time1) / 60 , 2) 
    print(f"Removing multicandidates required {removal_time2}s")
    df['event_entry'] = df.groupby(['RUNNUMBER', 'EVENTNUMBER']).ngroup() # in the concatenation the entries are the same among different files, needed to look at evt and run number to identify them
    # df.drop(columns=['entry', 'level_2'], inplace=True)
    
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
    '''
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
    plt.savefig(f"{cfg.target_path}/preSelect_variables.pdf")'''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    #df_selected = df.query('selected==1')[features + ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'label']]
    #df_not_selected = df.query('selected==0')[features + ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'label']]
    # Split data into training+validation set and test set
    train_df, val_df, test_df = pyTrain.splitByEvent(df[features + ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'B_Tr_T_TRUEID', 'label']])
    # For training: keep only tracks that pass the pre-selections. 
    # For calibration, events with 0 selected tracks must be kept. This is necessary to estimate the tagging efficiency correctly 
    # Training-validation sets splitting
    plot_kp_for_torch(train_df, type="train", savepath=torchPath, tagger=cfg.tagger)
    plot_kp_for_torch(val_df, type="validation", savepath=torchPath, tagger=cfg.tagger)
    plot_kp_for_torch(test_df, type="test", savepath=torchPath, tagger=cfg.tagger)
    stats_printout(df, train_df, val_df, test_df)


    # Save test dataframe for calibration
    # test_df.to_csv(f"{testSetPath}", index = False)
    columns_to_drop = ['event_entry', 'selected', 'TagDec', 'B_TRUEID', 'B_Tr_T_TRUEID']
    train_df.drop(columns = columns_to_drop, inplace = True)
    val_df.drop(columns = columns_to_drop, inplace = True)
    train_dl, validation_dl = pyTrain.prepare_data(train_df=train_df, features=features, val_df=val_df, savePlot_path=cfg.target_path, scalerPath=scalerPath)
    model = NeuralNetwork(modelName = pyTrain.config.model_name, features=features, seed=cfg.seed, train_batch_size = pyTrain.config.train_batch_size, test_batch_size = 1024, optimizer_kwargs={"lr" : pyTrain.config.learning_rate}).to(device)
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, cfg.target_path, n_epochs = pyTrain.config.n_epochs)
    pyTrain.plot_losses(pyTrain.config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    pyTrain.save_losses(pyTrain.config.model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    print(f"Training set has {train_df[train_df.label==0].shape[0]} wrong tagged tracks, {train_df[train_df.label==1].shape[0]} correctly tagged tracks")
    # Plot ROC curves for validation and train test
    bestModel.eval()
    yPredVal, yTrueVal = bestModel.evaluate_model(validation_dl)
    yPredTrain, yTrueTrain = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(model.modelName, yPredVal, yTrueVal, cfg.target_path, yPredTrain, yTrueTrain)
    # Fit with logistic regression and save it (non needed for the moment)
    clf = pyTrain.logistic_regression(yPredTrain, yTrueTrain, cfg.target_path, model.modelName)
    #pyTrain.plot_NNoutput_mistag(pyTrain.config.model_name, clf, yPredVal, yTrueVal, yPredTrain, yTrueTrain, cfg.target_path)
    #pyTrain.plot_mistag(pyTrain.config.model_name, clf, yPredVal, yTrueVal, cfg.target_path, type = 'validation')
    pyTrain.plot_mistag(name=bestModel.modelName, yPred=yPredTrain, yTrue=yTrueTrain, target_path=cfg.target_path, type = 'Training')
    plt.figure()
    plt.hist(1-yPredTrain ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 0, only selected")
    plt.savefig(f"{cfg.target_path}/trainingSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(yPredTrain ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 1, only selected")
    plt.savefig(f"{cfg.target_path}/trainingSet_prob1distrib.pdf")


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
    print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks")
    
    yPredTest, yTrueTest = bestModel.evaluate_model(test_dl_sel1)
    pyTrain.plot_ROC(bestModel.modelName, yPredTest, yTrueTest, cfg.target_path)
    pyTrain.plot_mistag(name=bestModel.modelName, yPred=yPredTest, yTrue=yTrueTest, target_path=cfg.target_path, type = 'Test')
    plt.figure()
    plt.hist(1-yPredTest,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 0, only selected")
    plt.savefig(f"{cfg.target_path}/testSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(yPredTest,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 1")
    plt.savefig(f"{cfg.target_path}/testSet_prob1distrib.pdf")
    #
    test_dataset = inputDataset(test_df.drop(columns = columns_to_drop), scalerPath, test = True)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true
    test_df['Eta'] = 1 - test_df['predictedProb']

    test_df = test_df[['event_entry','selected', 'Eta', 'TagDec','B_TRUEID', 'B_Tr_T_TRUEID', 'B_Tr_T_P']]

    #print(test_df.loc[test_df.selected == 1].Eta) 

    plot_mistag_for_torch(test_df, torchPath, cfg.tagger)

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