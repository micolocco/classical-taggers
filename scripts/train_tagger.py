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


def stats_printout(tagger, decay_type, train_df, val_df, BID):
    '''
    Function to print statistics about the dataset composition
    '''
    from rich.console import Console
    from rich.table import Table
    
    train_evts =  len(train_df['event_entry'].unique())
    train_sel_evts = len(train_df[train_df.selected==1]['event_entry'].unique())
    val_evts =  len(val_df['event_entry'].unique())
    val_sel_evts = len(val_df[val_df.selected==1]['event_entry'].unique())


    tot_evts = train_evts + val_evts
    sel_evts = train_sel_evts + val_sel_evts
    
    print(f"\n Statistics used in the {tagger} pipeline\n")

    console = Console()
    table = Table(show_header=True)
    table.add_column("", justify="left")
    table.add_column("Events", justify="left", style='cyan')
    table.add_column("Tracks", justify="left", style='green')
    table.add_row("Before selection", f"{tot_evts}", f"{train_df.shape[0]+val_df.shape[0]}")
    table.add_row("After selection", f"{sel_evts}", f"{train_df[train_df.selected==1].shape[0]+val_df[val_df.selected==1].shape[0]}")
    table.add_row("Train", f"{train_evts}", f"{train_df.shape[0]}")
    table.add_row("Validation", f"{val_evts}", f"{val_df.shape[0]}",)
    console.print(table)
    print("\nThe train and the validation sets are made of tracks passing the preselection.")
    print("The calibration set contains both selected and not selected events. \n")

    print("Correct tagging decision l=1, wrong tagging decision l=0")
    if decay_type[:2]=='Bu':
        ID=521
    if decay_type[:2]=='Bd':
        ID=511
    if decay_type[:2]=='Bs':
        ID=531
    
    B_correct_train =     train_df[(train_df.label==1) &(train_df[BID]==-ID)].shape[0]
    antiB_correct_train = train_df[(train_df.label==1) &(train_df[BID]==ID) ].shape[0]
    B_wrong_train  =      train_df[(train_df.label==0) &(train_df[BID]==-ID)].shape[0]
    antiB_wrong_train  =  train_df[(train_df.label==0) &(train_df[BID]==ID) ].shape[0]
    table = Table(show_header=True)
    table.add_column("", justify="left")
    table.add_column("l=1, B", justify="left", style='cyan', overflow="fold")
    table.add_column("l=1, antiB", justify="left", style='cyan', overflow="fold")
    table.add_column("(N\[l=1,B]-N\[l=1,antiB])/N\[l=1]", justify="left", style='cyan', overflow="fold")
    table.add_column("l=0, B", justify="left", style='green', overflow="fold")
    table.add_column("l=0, antiB", justify="left", style='green', overflow="fold")
    table.add_column("(N\[l=0,B]-N\[l=0,antiB])/N\[l=0]", justify="left", style='green', overflow="fold")
    table.add_row("Training set", f"{B_correct_train}", f"{antiB_correct_train}",f"{(100*(B_correct_train-antiB_correct_train)/(B_correct_train+antiB_correct_train)):.2f}%", f"{B_wrong_train}", f"{antiB_wrong_train}",f"{(100*(B_wrong_train-antiB_wrong_train)/(B_wrong_train+antiB_wrong_train)):.2f}%")
    console.print(table)

def read_files(files, vars, treename):
    df = pd.DataFrame(columns=vars)

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used in Megabites: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}')

        id = os.path.basename(f)[:-5]
        if id[-7:-2] == '.data':
            id = id[:-7]
        else:
            id = id[:-3]

        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars+ ['RUNNUMBER', 'EVENTNUMBER'], library="pd")
        _df.dropna(inplace = True)
        _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
        df = pd.concat([df, _df], ignore_index = True)

    return df

def get_architecture(config):
    nL = config['numlayers']
    nN = config['numneurons']
    dp = config['dropout']
    return f'nL{nL}_nN{nN}_dp{dp}'

# Moving pipeline to a function, to allow for Hyperparameter tuning in different file
def training_pipeline(train_df, val_df, vars,  weight_label, BID, target_path, treename, 
                      tagger, seed, features, config, decay_type, 
                      repo, data_type, weight_type, num_threads = 1, clean = False, logfile = None):
    if logfile is not None:
        from scripts.batch_train_tagger import ThreadLocalStdout
        sys.stdout = ThreadLocalStdout()
        sys.stdout.set_log_file(logfile)

    print(f'Training started on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Load YAML configuration file
    with open(f'{config}', 'r') as file:
        config = yaml.safe_load(file)
    
    print(vars)

    
    print(f"The features used are: {features}", flush=True)
    # Check and eventually make output directory where training info will be saved
    pyTrain.recreate_directory(target_path, clean=clean)
    # Path to where the scaler parameters will be saved
    scalerPath = f"{target_path}/st_scaler.pkl"
    transformerPath = f"{target_path}/powerTransformer.pkl"

    if data_type == 'Data':
        weight_label = weight_type


    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    


    train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df.sample(frac=1, random_state=seed).reset_index(drop=True)


    
    train_batch_size = config['train_batch_size']
    val_batch_size = config['train_batch_size'] #Might want to change this to a seperate hyperparameter in the config file
    weights_train = None
    weights_val = None
    if data_type == 'Data' and weight_label != 'ones': 
        weights_train = train_df[weight_label].to_numpy()
        weights_val = val_df[weight_label].to_numpy()
    
    # For training: keep only tracks that pass the pre-selections. 
    # For calibration, events with 0 selected tracks must be kept. This is necessary to estimate the tagging efficiency correctly 
    # Training-validation sets splitting
    stats_printout(tagger=tagger, decay_type=decay_type,train_df=train_df, val_df=val_df, BID=BID)
    print(f"Training set has {train_df[train_df.label==1].shape[0]} correctly tagged tracks, {train_df[train_df.label==0].shape[0]} wrong tagged tracks")
    # Save test dataframe for calibration
    
    columns_to_drop = ['event_entry', 'selected', f"{tagger}_TagDec", BID]#, 'B_DTF_PV_Jpsi_MASS']
    if data_type == 'Data' and weight_label != 'ones':
        columns_to_drop.append(weight_label)
    
    print(train_df.drop(columns = columns_to_drop).columns)
    print(features)
    
    train_dl, validation_dl = pyTrain.prepare_data(train_df=train_df.drop(columns = columns_to_drop), val_df=val_df.drop(columns = columns_to_drop), train_batch_size=train_batch_size, seed=seed, scalerPath=scalerPath, transformerPath=transformerPath, test_batch_size = val_batch_size)
    if config!='configs/config_test':
        pyTrain.plot_features(data=train_df, features_list=features, target_path=target_path, flag='label', name=f'training_inputFeatures')
    model = NeuralNetwork(features=features, architecture=get_architecture(config), seed=seed, optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=repo).to(device)
    print(f"\nThe NN architecture is: \n{model}\n")

    for batch in validation_dl:
        if isinstance(batch, (list, tuple)):
            data = batch[0]  # inputs
            labels = batch[1]  # targets (optional, depending on dataset)
        else:
            data = batch  # e.g., for unsupervised data
        print(f"Data shape: {data.shape}")
        break

    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_EarlyStopping(model, train_dl, validation_dl, target_path, config = config, train_weights = weights_train, val_weights= weights_val, num_threads=num_threads)
    
    pyTrain.save_model(bestModel, target_path)
    
    pyTrain.plot_losses(tagger, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)
    pyTrain.save_losses(trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)
    # Plot ROC curves for validation and train test
    bestModel.eval()
    val_df['yPred'], val_df['yTrue'] = bestModel.evaluate_model(validation_dl)
    train_df['yPred'], train_df['yTrue'] = bestModel.evaluate_model(train_dl)
    pyTrain.plot_ROC(tagger=tagger, val_df=val_df, train_df=train_df, target_path =target_path)
    # Fit with logistic regression and save it (non needed for the moment)
    #clf = pyTrain.logistic_regression(df=train_df, target_path=target_path)
    #pyTrain.plot_NNoutput_mistag(config.model_name, clf, yPredVal, yTrueVal, train_df['yPred'], train_df['yTrue'], target_path)
    #pyTrain.plot_mistag(config.model_name, clf, yPredVal, yTrueVal, target_path, type = 'validation')
    pyTrain.plot_mistag(tagger=tagger, df=train_df, target_path=target_path, type = 'Training', show_trueB=False, BID = BID)
    plt.figure()
    plt.hist(1-train_df['yPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 0, only selected")
    plt.yscale("log")
    plt.savefig(f"{target_path}/trainingSet_prob0distrib.pdf")
    plt.figure()
    plt.hist(train_df['yPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
    plt.title(r"Training set: Probability of label 1, only selected")
    plt.yscale("log")
    plt.savefig(f"{target_path}/trainingSet_prob1distrib.pdf")

    print(f'Training ended on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the tagger on the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training_data', help='Files of training data', nargs='+')
    parser.add_argument('--validation_data', help='Files of validation data', nargs='+')
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--seed', help='Random seed', default=45, type = int) 
    parser.add_argument('--features', help='Input features for NN training', default='union') 
    parser.add_argument('--config', help='Config yaml', type=str, default='configs/config_test') 
    parser.add_argument('--decay_type', help='Event decay', type=str)
    parser.add_argument('--clean', help='Decide whatever cleaning the directories before running, w=False, a=True', action='store_true')
    parser.add_argument('--repo', help="Path to repository")
    parser.add_argument('--data_type', help="Type of Data used, MC or Data",choices=('MC', 'Data'))
    parser.add_argument('--weight_type', help="Type of sample weight to be used for training on data", choices=('signal_weights', 'pdf_ratio', 'ones'))
    parser.add_argument('--num_threads', help='Number of threads to use in training', type=int, default=1)
    
    cfg = parser.parse_args()
    pprint(cfg)

    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)


    if cfg.data_type == 'Data':
        print(f'features are {features}')
        for i in range(len(features)):
            features[i] = features[i].replace("BPVIP", "OWNPVIP")
            features[i] = features[i].replace("B_TRUEID", "B_ID")

    vars = features + [BID,'selected', 'label',f"{cfg.tagger}_TagDec"] #'B_Tr_T_Charge',
    if cfg.data_type == 'Data':
        weight_label = cfg.weight_type
        if weight_label != 'ones':
            vars = vars + [weight_label]



    #Reading Data from files
    print(f'Reading of training files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.training_data)} files.", flush=True)
    train_df = read_files(cfg.training_data, vars = vars, treename=cfg.treename)
    print(f'Reading of training files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    print(f'Reading of validation files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.validation_data)} files.", flush=True)
    val_df = read_files(cfg.validation_data, vars = vars, treename=cfg.treename)
    print(f'Reading of validation files ends {datetime.datetime.now().strftime("%H:%M:%S")}')



    training_pipeline(train_df=train_df, val_df=val_df, vars=vars, weight_label=weight_label, BID=BID, 
                      target_path=cfg.target_path, treename=cfg.treename, tagger=cfg.tagger, seed=cfg.seed, features=features, 
                      config=cfg.config, decay_type=cfg.decay_type, clean=cfg.clean, repo=cfg.repo, data_type=cfg.data_type, 
                      weight_type=cfg.weight_type, num_threads=cfg.num_threads)