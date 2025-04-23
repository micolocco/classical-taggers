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
import psutil
import json
from os.path import join

def read_files_reduce_unselected(files, vars, treename):
    df = pd.DataFrame(columns=vars)

    for i, f in enumerate(files):
        print(f"Reading input file {i}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used in Megabites: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}')

        id = os.path.basename(f)[:-5]
        if id[-7:-2] == '.data':
            id = id[:-7]
        else:
            id = id[:-3]

        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars+ ['RUNNUMBER', 'EVENTNUMBER'], library="pd")
        _df.dropna(inplace = True)


        #Drop 80% of unselected to reduce memory usage
        #TODO is sensible???
        mask = _df['selected'] == 0
        drop_indices = _df[mask].sample(frac=0.8, random_state=42).index
        _df = _df.drop(drop_indices)

        _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
        df = pd.concat([df, _df], ignore_index = True)
    
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test the tagger of a specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--testing_data', help='Files of training data', nargs='+')
    parser.add_argument('--target_path', help='Name of the output dir', type=str)
    parser.add_argument('--train_path', help='Name of the output dir', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--features', help='Input features for NN training', default='union') 
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/config_test') 
    parser.add_argument('--decay_type', help='Event decay', type=str)
    parser.add_argument('--seed', help='Random seed', default=45, type = int) 
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--data_type', help="Type of Data used, MC or Data",choices=('MC', 'Data'))
    parser.add_argument('--weight_type', help="Type of sample weight to be used for training on data", choices=('signal_weights', 'pdf_ratio', 'ones'))
    parser.add_argument('--model_path', help='Path to trained model', type=str)

    print(f'Training started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush = True)
    cfg = parser.parse_args()
    pprint(cfg)
    # Load YAML configuration file
    with open(f'{cfg.config}', 'r') as file:
        config = yaml.safe_load(file)
    
    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    # Path to the ROOT input file
    test_files = cfg.testing_data
    
    # Path to where the scaler parameters will be saved
    scalerPath = f"{cfg.train_path}/st_scaler.pkl"
    transformerPath = f"{cfg.train_path}/powerTransformer.pkl"


    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    

    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'


    if cfg.data_type == 'Data':
        print(f'features are {features}', flush = True)
        for i in range(len(features)):
            features[i] = features[i].replace("BPVIP", "OWNPVIP")
            features[i] = features[i].replace("B_TRUEID", "B_ID")

    vars = features + [BID,'selected', 'label',f"{cfg.tagger}_TagDec"] #'B_Tr_T_Charge',
    if cfg.data_type == 'Data':
        weight_label = cfg.weight_type
        if weight_label != 'ones':
            vars = vars + [weight_label]

    print(vars, flush = True)


    #Reading Data from files
    print(f'Reading of test files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(test_files)} files.", flush=True)
    test_df = read_files_reduce_unselected(test_files, vars = vars, treename=cfg.treename)
    print(f'Reading of test files ends {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)
    test_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    #Load model

    model_path = cfg.model_path

    bestModel = NeuralNetwork(features=features, architecture=config['architecture'], seed=cfg.seed, optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=cfg.repo)
    pyTrain.load_model(model=bestModel, target_path=model_path)

    columns_to_drop = ['event_entry', 'selected', f"{cfg.tagger}_TagDec", BID]#, 'B_DTF_PV_Jpsi_MASS']
    if cfg.data_type == 'Data' and weight_label != 'ones':
        columns_to_drop.append(weight_label)

    

    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    test_df_sel1 = test_df.query('selected==1').copy()
    test_dataset_sel1 = inputDataset(df=test_df_sel1.drop(columns = columns_to_drop))
    test_dataset_sel1.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl_sel1 = DataLoader(pyTrain.IndexedDataset(test_dataset_sel1), batch_size = 1024, shuffle=False)
    print(f"Test set has {len(test_dl_sel1.dataset)} tracks selected as tagging particles")
    print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks", flush = True)
    
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
    pyTrain.plot_mistag(tagger=cfg.tagger, df=test_df_sel1, target_path=cfg.target_path, type = 'Test', BID = BID)
    

    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(pyTrain.IndexedDataset(test_dataset), batch_size = 1024, shuffle=False)

    #test_df[f"{cfg.tagger}_Eta"] = clf.predict_proba(bestModel.evaluate_model(test_dl)[0])[:,0]
    test_df['predictedProb'] = bestModel.evaluate_model(test_dl)[0] # bestModel.evaluate_model returns predicted probabilities for label 1, true values
    test_df[f"{cfg.tagger}_Eta"] = 1 - test_df['predictedProb']

    test_df = test_df[['event_entry','selected', f"{cfg.tagger}_Eta", f"{cfg.tagger}_TagDec", 'label',BID]]

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
    
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating", flush = True)
    pyTrain.plot_tagDec(tagger =cfg.tagger, df_TagParticles=df_TagParticles,  plot_name=f'{cfg.target_path}/Normalized_TagDec.pdf')
    # Calibrating the tagger and saving parameters
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decay_type, target_path=cfg.target_path, BID = BID)
    # Try both calibration functions
    pyTrain.calibration(tagger=cfg.tagger, df_tag=df_TagParticles, eventType=cfg.decay_type, target_path=cfg.target_path, calibration_option='logit', BID = BID)


    print(f'testing ended on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
