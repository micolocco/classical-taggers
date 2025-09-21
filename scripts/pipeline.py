import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import uproot
import pandas as pd
from scripts.inputDataset import inputDataset
import pickle
from matplotlib import pyplot as plt
from IPython import embed
import os
import argparse
from pprint import pprint
import datetime
import yaml
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import utils
'''
For testing purposes:
python scripts/pipeline.py --selected NTuple_test_<tagger>.root --tagger <tagger> --decayType <decayType> where NTuple_test_tagger.root is whatever NTuple with this name
example:
python scripts/pipeline.py --selected NTuple_test_OSKaon.root --tagger OSKaon --decayType Bu2JpsiK 
'''

def stats_printout(df, tagger, decayType, train_df, val_df, test_df):
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
    
    print(f"\n Statistics used in the {tagger} pipeline\n")

    console = Console()
    table = Table(show_header=True)
    table.add_column("", justify="left")
    table.add_column("Events", justify="left", style='cyan')
    table.add_column("Tracks", justify="left", style='green')
    table.add_row("Before selection", f"{tot_evts}", f"{df.shape[0]}")
    table.add_row("After selection", f"{sel_evts}", f"{df[df.selected==1].shape[0]}")
    table.add_row("Train", f"{train_evts}", f"{train_df.shape[0]}")
    table.add_row("Validation", f"{val_evts}", f"{val_df.shape[0]}",)
    table.add_row("Calibration (only selected)", f"{test_evts_sel}", f"{test_df[test_df.selected==1].shape[0]}")
    table.add_row("Calibration (total)", f"{test_evts}", f"{test_df.shape[0]}")
    console.print(table)
    print("\nThe train and the validation sets are made of tracks passing the preselection.")
    print("The calibration set contains both selected and not selected events. \n")

    print("Correct tagging decision l=1, wrong tagging decision l=0")
    if decayType[:2]=='Bu':
        ID=521
    if decayType[:2]=='Bd':
        ID=511
    if decayType[:2]=='Bs':
        ID=531
    B_correct_train = train_df[(train_df.label==1)&(train_df.B_TRUEID==-ID)].shape[0]
    antiB_correct_train = train_df[(train_df.label==1)&(train_df.B_TRUEID==ID)].shape[0]
    B_wrong_train  = train_df[(train_df.label==0)&(train_df.B_TRUEID==-ID)].shape[0]
    antiB_wrong_train  = train_df[(train_df.label==0)&(train_df.B_TRUEID==ID)].shape[0]
    B_correct_test = test_df[(test_df.selected==1)&(test_df.label==1)&(test_df.B_TRUEID==-ID)].shape[0]
    antiB_correct_test = test_df[(test_df.selected==1)&(test_df.label==1)&(test_df.B_TRUEID==ID)].shape[0]
    B_wrong_test = test_df[(test_df.selected==1)&(test_df.label==0)&(test_df.B_TRUEID==-ID)].shape[0]
    antiB_wrong_test = test_df[(test_df.selected==1)&(test_df.label==0)&(test_df.B_TRUEID==ID)].shape[0]
    table = Table(show_header=True)
    table.add_column("", justify="left")
    table.add_column("l=1, B", justify="left", style='cyan', overflow="fold")
    table.add_column("l=1, antiB", justify="left", style='cyan', overflow="fold")
    table.add_column("(N\[l=1,B]-N\[l=1,antiB])/N\[l=1]", justify="left", style='cyan', overflow="fold")
    table.add_column("l=0, B", justify="left", style='green', overflow="fold")
    table.add_column("l=0, antiB", justify="left", style='green', overflow="fold")
    table.add_column("(N\[l=0,B]-N\[l=0,antiB])/N\[l=0]", justify="left", style='green', overflow="fold")
    table.add_row("Training set", f"{B_correct_train}", f"{antiB_correct_train}",f"{(100*(B_correct_train-antiB_correct_train)/(B_correct_train+antiB_correct_train)):.2f}%", f"{B_wrong_train}", f"{antiB_wrong_train}",f"{(100*(B_wrong_train-antiB_wrong_train)/(B_wrong_train+antiB_wrong_train)):.2f}%")
    table.add_row("Test set", f"{B_correct_test}", f"{antiB_correct_test}",f"{(100*(B_correct_test-antiB_correct_test)/(B_correct_test+antiB_correct_test)):.2f}%", f"{B_wrong_test}", f"{antiB_wrong_test}",f"{(100*(B_wrong_test-antiB_wrong_test)/(B_wrong_test+antiB_wrong_test)):.2f}%")
    console.print(table)

# Decay and tagger type are given as inputs by the user
# Decay must be one among Bd2JpsiKst,  Bs2DsPi,  Bu2JpsiK 
# Taggers must be one among OSKaon, OSMuon, OSElectron, SSPion, SSProton, SSKaon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selected', help='File with preselection applied', nargs='+')
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--seed', help='Random seed', default=45) 
    parser.add_argument('--features', help='Input features for NN training', default='union') 
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/config_test') 
    parser.add_argument('--decayType', help='Event decay', type=str)
    parser.add_argument('--clean', help='Decide whatever cleaning the directories before running, w=False, a=True', action='store_true')
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--only_plot', help='Only plot input features and exit', action='store_true')
    
    print(f'Pipeline started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = parser.parse_args()
    pprint(cfg)
    # Load YAML configuration file
    with open(f'{cfg.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    # Path to the ROOT input file
    selected_files = cfg.selected
    print(f"The features used are: {features}")
    # Check and eventually make output directory where training info will be saved
    pyTrain.recreate_directory(cfg.target_path, clean=cfg.clean)
    # Path to where the scaler parameters will be saved
    scalerPath = f"{cfg.target_path}/st_scaler.pkl"
    transformerPath = f"{cfg.target_path}/powerTransformer.pkl"

    # Path to where the test set will be saved
    testSetPath = f"{cfg.target_path}/testSet.csv"

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reading datasets
    vars = features + ['B_TRUEID','B_Tr_T_Charge','selected',]
    
    df = pd.DataFrame(columns=vars)
    for i, f in enumerate(selected_files):
        print(f"Reading input file: {f}")
        with uproot.open("{}".format(f)) as _f:
            _df = _f[cfg.treename].arrays(vars+ ['RUNNUMBER', 'EVENTNUMBER'], library="pd")
        _df.dropna(inplace = True)
        _df["SAMPLENUMBER"] = i
        _df["event_entry"] = _df["SAMPLENUMBER"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'SAMPLENUMBER'], inplace=True)
        df = pd.concat([df, _df], ignore_index = True)
    
    df.sample(frac=1, random_state=45).reset_index(drop=True) # cfg.seed
    # Drop multicandidates
    #df = df.groupby("event_entry").first()


    # removal_time1 = time.time()
    # df = utils.remove_multicandidates(df)
    # #df = utils.remove_multicandidates(df)
    # removal_time2 = round((time.time()- removal_time1) / 60 , 2) 
    # print(f"Removing multicandidates required {removal_time2}s")
    
    # Assignation of the tagging decision (d)
    # d = (-1) * charge of the track --> neutral B: any OS taggers and SS proton tagger, charged B: any taggers
    if ("Bd" in cfg.decayType or "Bs" in cfg.decayType) and (cfg.tagger == "SSKaon" or cfg.tagger == "SSPion" ):
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"]
    else:
        df[f"{cfg.tagger}_TagDec"] = df[f"B_Tr_T_Charge"] * (-1)

    # Assignation of the label (it will be used as NN output)
    # The label is given by the product of the tagging decision and the flavour charge of the B.
    # It indicates if the tagging decision is wrong or correct.
    # -1 == wrong tag  1 == correct tag
    # When using data:
    #   - tagging decision: the B_TRUEID must be replaced with B_ID 
    #   - calibration: B_ID = reconstructed ID when moving to data!
    df["label"] = df[f"{cfg.tagger}_TagDec"] * df[f"B_TRUEID"]/abs(df[f"B_TRUEID"]) 
    df.loc[df.label == -1, "label"] = 0 # shifting the label from -1 to 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    #df_selected = df.query('selected==1')[features + ['event_entry', 'selected', f"{cfg.tagger}_TagDec", 'B_TRUEID', 'label']]
    #df_not_selected = df.query('selected==0')[features + ['event_entry', 'selected', f"{cfg.tagger}_TagDec", 'B_TRUEID', 'label']]
    # Split data into training+validation set and test set
    train_df, val_df, test_df = pyTrain.splitByEvent(df=df[features + ['event_entry', 'selected', f"{cfg.tagger}_TagDec", 'B_TRUEID', 'label']], seed=cfg.seed, train_val_split=config['train_val_split'])
    # For training: keep only tracks that pass the pre-selections. 
    # For calibration, events with 0 selected tracks must be kept. This is necessary to estimate the tagging efficiency correctly 
    # Training-validation sets splitting
    stats_printout(df=df, tagger=cfg.tagger, decayType=cfg.decayType,train_df=train_df, val_df=val_df, test_df=test_df)
    print(f"Training set has {train_df[train_df.label==1].shape[0]} correctly tagged tracks, {train_df[train_df.label==0].shape[0]} wrong tagged tracks")
    # Save test dataframe for calibration
    test_df.to_csv(f"{testSetPath}", index = False)
    columns_to_drop = ['event_entry', 'selected', f"{cfg.tagger}_TagDec", 'B_TRUEID',]
    #train_df.drop(columns = columns_to_drop, inplace = True)
    #val_df.drop(columns = columns_to_drop, inplace = True)
    train_dl, validation_dl = pyTrain.prepare_data(train_df=train_df.drop(columns = columns_to_drop), val_df=val_df.drop(columns = columns_to_drop), train_batch_size=config['train_batch_size'], seed=cfg.seed, scalerPath=scalerPath, transformerPath=transformerPath)
    if cfg.config!='configs/config_test':
        pyTrain.plot_features(data=train_df, features_list=features, target_path=cfg.target_path, flag='label', name=f'training_inputFeatures')
        if cfg.only_plot:
            sys.exit(0)

    model = NeuralNetwork(features=features, architecture=config['architecture'], seed=cfg.seed, optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=cfg.repo).to(device)
    print(f"\nThe NN architecture is: \n{model}\n")
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, cfg.target_path, config = config)
    pyTrain.plot_losses(cfg.tagger, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    pyTrain.save_losses(trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, cfg.target_path)
    # Plot ROC curves for validation and train test
    bestModel.eval()
    val_df['yPred'], val_df['yTrue'] = bestModel.evaluate_model(validation_dl)
    train_df['yPred'], train_df['yTrue'] = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(tagger=cfg.tagger, val_df=val_df, train_df=train_df, target_path =cfg.target_path)
    # Fit with logistic regression and save it (non needed for the moment)
    #clf = pyTrain.logistic_regression(df=train_df, target_path=cfg.target_path)
    #pyTrain.plot_NNoutput_mistag(config.model_name, clf, yPredVal, yTrueVal, train_df['yPred'], train_df['yTrue'], cfg.target_path)
    #pyTrain.plot_mistag(config.model_name, clf, yPredVal, yTrueVal, cfg.target_path, type = 'validation')
    pyTrain.plot_mistag(tagger=cfg.tagger, df=train_df, target_path=cfg.target_path, type = 'Training', show_trueB=False)
    plt.figure()
    plt.hist(1-train_df['yPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 0, only selected")
    plt.yscale("log")
    plt.savefig(f"{cfg.target_path}/trainingSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(train_df['yPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 1, only selected")
    plt.yscale("log")
    plt.savefig(f"{cfg.target_path}/trainingSet_prob1distrib.pdf")


    # else:
    '''
    test_df = pd.read_csv(f"{testSetPath}")

    clf = pickle.load(open(f"{cfg.target_path}/LogReg.pck", 'rb'))   
    #Load the best model (ie with the lowest training loss) and evaluate it on the test set
    bestModel = NeuralNetwork(features=features, optimizer_kwargs={"lr" : config.learning_rate}).to(device)
    pyTrain.load_model(bestModel, cfg.target_path)
    bestModel.eval()
    '''
    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    test_df_sel1 = test_df.query('selected==1').copy()
    test_dataset_sel1 = inputDataset(df=test_df_sel1.drop(columns = columns_to_drop))
    test_dataset_sel1.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl_sel1 = DataLoader(test_dataset_sel1, batch_size = 1024, shuffle=False)
    print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
    print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks")
    
    test_df_sel1['yPred'], test_df_sel1['yTrue'] = bestModel.evaluate_model(test_dl_sel1)
    pyTrain.plot_ROC(tagger=cfg.tagger, val_df=test_df_sel1, target_path =cfg.target_path)
    plt.figure()
    plt.hist(1-test_df_sel1['yPred'],bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 0, only selected")
    plt.savefig(f"{cfg.target_path}/testSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(test_df_sel1['yPred'],bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Test set: Probability of label 1")
    plt.savefig(f"{cfg.target_path}/testSet_prob1distrib.pdf")
    pyTrain.plot_mistag(tagger=cfg.tagger, df=test_df_sel1, target_path=cfg.target_path, type = 'Test')
    

    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)

    #test_df[f"{cfg.tagger}_Eta"] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values
    test_df[f"{cfg.tagger}_Eta"] = 1 - test_df['predictedProb']

    test_df = test_df[['event_entry','selected', f"{cfg.tagger}_Eta", f"{cfg.tagger}_TagDec", 'label','B_TRUEID']]

    #print(test_df.loc[test_df.selected == 1][f"{cfg.tagger}_Eta"]) 

    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{cfg.tagger}_Eta"] = 0.5  # classic
    pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=test_df.sort_values(by = ["selected",f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first(), plot_name=f'{cfg.target_path}/Not_Normalized_TagDec.pdf')
 
    # Eta Normalization [0, 0.5]
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] > 0.5 ,f"{cfg.tagger}_TagDec"] *= -1
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] > 0.5, f"{cfg.tagger}_Eta"] *= -1
    test_df.loc[test_df[f"{cfg.tagger}_Eta"] < 0, f"{cfg.tagger}_Eta"] += 1

    df_TagParticles = test_df.sort_values(by = ["selected",f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first()
    #df_TagParticles = test_df.sort_values(by = ["selected",f"{cfg.tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first() test this 
    
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating")
    pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=df_TagParticles,  plot_name=f'{cfg.target_path}/Normalized_TagDec.pdf')
    # Calibrating the tagger and saving parameters
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decayType, target_path=cfg.target_path)
    # Try both calibration functions
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decayType, target_path=cfg.target_path, calibration_option='logit')
    print(f'Pipeline finished on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
