import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import uproot
import pandas as pd
from inputDataset import inputDataset
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from sklearn.model_selection import KFold

from scripts import ranges, nice_names, matplotlib_lhcb_style
# matplotlib_lhcb_style(plt)
import mplhep as hep
hep.style.use("LHCb2")
from rich.console import Console
from rich.table import Table


# Decay and tagger type are given as inputs by the user
# Decay must be one among Bd2JpsiKst,  Bs2DsPi,  Bu2JpsiK 
# Taggers must be one among OSKaon, OSMuon, OSElectron, SSPion, SSProton, SSKaon

def stats_printout(df, train_df, val_df, test_df):
    tot_evts = len(df['entry'].unique())
    sel_evts = len(df[df.selected==1]['entry'].unique())
    train_evts =  len(train_df[train_df.selected==1]['entry'].unique())
    val_evts =  len(val_df[val_df.selected==1]['entry'].unique())
    test_evts =  len(test_df[test_df.selected==1]['entry'].unique())
    test_evts_tot =  len(test_df['entry'].unique())

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
    table.add_row("Calibration only selected", f"{test_evts}", f"{test_df[test_df.selected==1].shape[0]}")
    table.add_row("Calibration tot", f"{test_evts_tot}", f"{test_df.shape[0]}")

    console.print(table)

    print("\nThe train and the validation sets are made of tracks passing the preselection.")
    print("The calibration set contains both selected and not selected events. \n")

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

    from pprint import pprint
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

    # Reading datasets
    vars = features + ['B_TRUEID','B_Tr_T_Charge','selected', 'entry', 'EVENTNUMBER', 'RUNNUMBER']
    df = pd.DataFrame(columns=vars)
    for f in selected_files:
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(vars, library="pd")
        df = pd.concat([df, _df], ignore_index = True)
    df['entry'] = df.groupby(['RUNNUMBER', 'EVENTNUMBER', 'entry']).ngroup() # in the concatenation the entries are the same among different files, needed to look at evt and run number to identify them
    df = df.sample(frac=1, random_state=pyTrain.config.seed).reset_index(drop=True)
    df.dropna(inplace = True)
    # mult_cand_grouped = len(df[['RUNNUMBER', 'EVENTNUMBER']].value_counts())
    # tot = len(df[['RUNNUMBER', 'EVENTNUMBER', 'entry']].value_counts())
    # mult_cand = ((tot-mult_cand_grouped)/tot)*100
    # print(f"Total number of multiple candidates tracks {mult_cand}%")
    # mult_cand_grouped_sel = len(df[df['selected']==1][['RUNNUMBER', 'EVENTNUMBER']].value_counts())
    # tot_sel = len(df[df['selected']==1][['RUNNUMBER', 'EVENTNUMBER', 'entry']].value_counts())
    # mult_cand_sel = ((tot_sel-mult_cand_grouped_sel)/tot_sel)*100
    # print(f"Multiple candidates tracks in the selected sample {mult_cand_sel}%")



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
    columns_to_drop = ['entry', 'selected', 'TagDec', 'B_TRUEID', 'EVENTNUMBER', 'RUNNUMBER']
    # Split into train, validation, test sets
    df_selected = df.query('selected==1')
    df_not_selected = df.query('selected==0').drop(columns=['B_Tr_T_Charge'])
    train_df, val_df, test_df, train_val_df = pyTrain.splitByEvent(df_selected[features + ['entry', 'selected', 'TagDec', 'B_TRUEID', 'label', 'EVENTNUMBER', 'RUNNUMBER']])
    test_df = pd.concat([test_df, df_not_selected], ignore_index =True)
    stats_printout(df, train_df, val_df, test_df)
    train_df = train_df.drop(columns=columns_to_drop)
    val_df = val_df.drop(columns=columns_to_drop)
    # Save test dataframe for calibration
    # test_df.to_csv(f"{testSetPath}", index = False)
    # train_dl, validation_dl, train_val_dl = pyTrain.prepare_data(train_df=train_df, val_df=val_df, train_val_df=train_val_df, savePlot_path=cfg.target_path, scalerPath=scalerPath)

    '''
    train_indices = train_dl.dataset.indices
    df1 = df.iloc[train_indices][['label','entry']]
    print(f' label 0 : {df1[df1.label==0].shape[0]}, label 1 : {df1[df1.label==1].shape[0]}')
    '''
    # k-folding
    n_splits=4
    k_folds = KFold(n_splits=n_splits)
    training_kFold_loss = []
    validation_kFold_loss = []
    best_kModels = []
    entry_values = train_val_df['entry'].values
    k=0
    for train_entries, val_entries in k_folds.split(entry_values):
        kfold_path = cfg.target_path + f"/{k}Fold"
        os.makedirs(kfold_path, exist_ok=True)
        model_name = pyTrain.config.model_name+f'_{k}Fold'
        model = NeuralNetwork(modelName = model_name, features=features, train_batch_size = 1024, test_batch_size = 1024, optimizer_kwargs={"lr" : pyTrain.config.learning_rate}).to(device)
        print(f'\n ------- Running the k-{k} fold ------- \n')
        print(f'With {train_entries} and {val_entries} \n')
        train_loader_subset, val_data_subset = pyTrain.prepare_kfolded_data(train_val_df, entry_values, train_entries, val_entries, columns_to_drop, scalerPath)
        bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_loader_subset, val_data_subset, kfold_path, n_epochs = pyTrain.config.n_epochs)
        pyTrain.plot_losses(model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, kfold_path)
        pyTrain.save_losses(model_name, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, kfold_path)
        # Plot ROC curves for validation and train test
        bestModel.eval()
        yPredVal, yTrueVal = bestModel.evaluate_model(val_data_subset, validation=True)
        yPredTrain, yTrueTrain = bestModel.evaluate_model(train_loader_subset)
        pyTrain.plot_ROC(model_name, yPredVal, yTrueVal, kfold_path, yPredTrain, yTrueTrain)
        # Fit with logistic regression and save it
        clf = pyTrain.logistic_regression(yPredTrain, yTrueTrain, kfold_path, model_name)
        #pyTrain.plot_NNoutput(pyTrain.config.model_name, yPredVal, yTrueVal, yPredTrain, yTrueTrain, target_path)
        pyTrain.plot_mistag(pyTrain.config.model_name, clf, yPredVal, yTrueVal, kfold_path, type = 'validation')
        training_kFold_loss.append(bestLosses[0])
        validation_kFold_loss.append(bestLosses[1])
        best_kModels.append(bestModel)
        k+=1

    # else:

    #     test_df = pd.read_csv(f"{testSetPath}")
    #     test_df = df[features + ['entry', 'selected', 'TagDec', 'B_TRUEID', 'label']]
    #     folder = "savedModels"
    #     #prePath = name_formatter.assign_name(folder, pyTrain.config.model_name)
    #     clf = pickle.load(open(f"{target_path}/LogReg.pck", 'rb'))

        # Load the best model (ie with the lowest training loss) and evaluate it on the test set
        # bestModel = NeuralNetwork(modelName = pyTrain.config.model_name, features=features, optimizer_kwargs={"lr" : pyTrain.config.learning_rate}).to(device)
        # pyTrain.load_model(bestModel, target_path)
        # bestModel.eval()
    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    test_dataset_sel1 = inputDataset(test_df[test_df['selected']==1].drop(columns = columns_to_drop), scalerPath, test = True)
    # test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
    test_dataset = inputDataset(test_df.drop(columns = columns_to_drop), scalerPath, test = True)
    # test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)
    print(f"Test set has {len(test_dataset_sel1)} tracks selected as tagging particles")
    yPredTest_kfold = []
    yTrueTest_kfold = []
    test_eta_kfold = []
    for bestModel in best_kModels:
        yPredTest, yTrueTest = bestModel.evaluate_model(test_dataset_sel1, validation=True)
        yPredTest_kfold.append(yPredTest)
        yTrueTest_kfold.append(yTrueTest)
        test_eta = 1- bestModel.evaluate_model(test_dataset, validation=True)[0]
        test_eta_kfold.append(test_eta)
    yPredTest_kmean = np.mean(np.array(yPredTest_kfold), axis=0)
    yTrueTest_kmean = yTrueTest_kfold[0] # The true are always the same for all the k-folds
    test_eta_kmean = np.mean(np.array(test_eta_kfold), axis=0)

    pyTrain.plot_ROC(bestModel.modelName, yPredTest_kmean, yTrueTest_kmean, cfg.target_path)
    # pyTrain.plot_mistag(bestModel.modelName, clf, yPredTest_kmean, yTrueTest_kmean, cfg.target_path, type = 'Test')

    #

    #test_df['Eta'] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['Eta'] = test_eta_kmean
    test_df = test_df[['entry','selected', 'Eta', 'TagDec','B_TRUEID']]
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

    df_TagParticles = test_df.sort_values(by = ["entry","selected","Eta"] , ascending = [True,False,True]).groupby("entry").first()
    #print(df_TagParticles.shape[0])

    pyTrain.plot_tagDec(df_TagParticles, pyTrain.config.model_name, cfg.target_path)
    # Calibrating the tagger and saving parameters
    pyTrain.calibration(pyTrain.config.model_name, cfg.tagger, df_TagParticles, cfg.decayType, cfg.target_path)