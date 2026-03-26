import numpy as np
import torch
import uproot
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse
from pprint import pprint
import datetime
import yaml
# Local import
import scripts.pyTorchTraining as pyTrain
from scripts.NNModel import NeuralNetwork
from scripts.NNModel import NNDomainAdapted
from scripts import matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import psutil
from scripts.shareddataset import SharedDataset

import torch.distributed as dist
import torch.multiprocessing as mp
from rich.console import Console
from rich.table import Table
from io import StringIO
import matplotlib

from sklearn.metrics import accuracy_score
import random


def stats_printout(tagger, decay_type, train_df, val_df, BID):
    '''
    Function to print statistics about the dataset composition
    '''

    def count_events(df, selected=None):
        if selected is not None:
            df = df[df.selected == selected]
        return df['event_entry'].nunique()

    def count_tracks(df, selected=None):
        if selected is not None:
            df = df[df.selected == selected]
        return df.shape[0]

    # Compute all stats
    stats = {}

    stats[f"train_evts"] = count_events(train_df)
    stats[f"train_sel_evts"] = count_events(train_df, selected=1)
    stats[f"val_evts"] = count_events(val_df)
    stats[f"val_sel_evts"] = count_events(val_df, selected=1)
    stats[f"tot_evts"] = stats[f"train_evts"] + stats[f"val_evts"]
    stats[f"sel_evts"] = stats[f"train_sel_evts"] + stats[f"val_sel_evts"]
    stats[f"train_tracks"] = count_tracks(train_df)
    stats[f"train_sel_tracks"] = count_tracks(train_df, selected=1)
    stats[f"val_tracks"] = count_tracks(val_df)
    stats[f"val_sel_tracks"] = count_tracks(val_df, selected=1)
    stats[f"tot_tracks"] = stats[f"train_tracks"] + stats[f"val_tracks"]
    stats[f"sel_tracks"] = stats[f"train_sel_tracks"] + stats[f"val_sel_tracks"]

    print(f"\n Statistics used in the {tagger} pipeline\n")

    console = Console()
    table = Table(show_header=True)
    table.width = 120
    table.add_column("", justify="left")
    table.add_column("Events", justify="left", style='cyan')
    table.add_column("Tracks", justify="left", style='green')

    def fmt(val):
        return f"{int(val)}"

    table.add_row(
        "Before selection",
        fmt(stats["tot_evts"]),
        fmt(stats["tot_tracks"]),
    )
    table.add_row(
        "After selection",
        fmt(stats["sel_evts"]),
        fmt(stats["sel_tracks"]),
    )
    table.add_row(
        "Train",
        fmt(stats["train_evts"]),
        fmt(stats["train_tracks"]),
    )
    table.add_row(
        "Validation",
        fmt(stats["val_evts"]),
        fmt(stats["val_tracks"]),
    )
    # console.print(table)

    output = StringIO()
    console = Console(file=output, width=200)
    console.print(table)
    table_str = output.getvalue()
    print(table_str)
    output.close()

    print("\nThe train and the validation sets are made of tracks passing the preselection.")
    print("The calibration set contains both selected and not selected events. \n")

    print("Correct tagging decision l=1, wrong tagging decision l=0")
    if decay_type[:2]=='Bu':
        ID=521
    if decay_type[:2]=='Bd':
        ID=511
    if decay_type[:2]=='Bs':
        ID=531

    def count_label(df, label, bid):
        mask = (df.label == label) & (df[BID] == bid)
        return df.loc[mask].shape[0]

    label_stats = {}

    label_stats[f"B_correct_train"] = count_label(train_df, 1, -ID)
    label_stats[f"antiB_correct_train"] = count_label(train_df, 1, ID)
    label_stats[f"B_wrong_train"] = count_label(train_df, 0, -ID)
    label_stats[f"antiB_wrong_train"] = count_label(train_df, 0, ID)

    def asymm(a, b):
        return f"{100*(a-b)/(a + b):.2f}%"

    table = Table(show_header=True)
    table.width = 120
    table.add_column("", justify="left", width=12)
    table.add_column("l=1, B", justify="left", style='cyan', overflow="fold", width=8)
    table.add_column("l=1, antiB", justify="left", style='cyan', overflow="fold", width=8)
    table.add_column("l=1 \n (N(B)-N(antiB))/N", justify="left", style='cyan', overflow="fold", width=8)
    table.add_column("l=0, B", justify="left", style='green', overflow="fold", width=8)
    table.add_column("l=0, antiB", justify="left", style='green', overflow="fold", width=8)
    table.add_column("l=0 \n (N(B)-N(antiB))/N", justify="left", style='green', overflow="fold", width=8)

    row = [
        "Training set",
        fmt(label_stats["B_correct_train"]),
        fmt(label_stats["antiB_correct_train"]),
        asymm(label_stats["B_correct_train"], label_stats["antiB_correct_train"]),
        fmt(label_stats["B_wrong_train"]),
        fmt(label_stats["antiB_wrong_train"]),
        asymm(label_stats["B_wrong_train"], label_stats["antiB_wrong_train"])
    ]
    table.add_row(*row)
    # console.print(table)

    output = StringIO()
    console = Console(file=output, width=200)
    console.print(table)
    table_str = output.getvalue()
    print(table_str)
    output.close()

def read_files(files, vars, treename, event_type, data_type):
    df = pd.DataFrame(columns=vars)

    additional_vars = ['file_id', 'RUNNUMBER', 'EVENTNUMBER']

    if event_type[:2] == 'Bu' and data_type == 'data':
        additional_vars = additional_vars + ['B_TAU']


    loading_vars = vars + additional_vars

    if isinstance(files, str):
        files = [files]

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')



        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(loading_vars, library="pd")

        _df = _df[_df['selected'] == 1] #Keep only selected events, as only those are used for training and validation
        
        print(f"Number of tracks in file {i+1}: {_df.shape[0]}", flush=True)

        if 'domain' not in vars:
            _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        else:
            _df.loc[_df['domain'] == 0, 'event_entry'] = _df["file_id"].astype(str) + "_" + "data" + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.loc[_df['domain'] == 1, 'event_entry'] = _df["file_id"].astype(str) + "_" + "mc" + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)

        if event_type[:2] == 'Bu' and data_type == 'data': #remove data with a liftime greater then 2.2ps, to ensure low oscillation likelyhood
            _df = _df[_df['B_TAU'] < 2.2]


        df = pd.concat([df, _df], ignore_index = True)
    del _df
    df.drop(columns=additional_vars, inplace=True)
        
    return df

def get_architecture(config):
    if 'is_test' in config.keys(): #For config_test use architecture_test, a single architecture file used for testing strucural changes in architecture without the need of creating a whole set of architectures.yaml files. 
        return 'architecture_test'
    if 'architecture' in config.keys():
        return config['architecture']
    else:
        nL = config['numlayers']
        nN = config['numneurons']
        dp = config['dropout']
        bn = '_BN' if config['use_batch_norm'] else ''
        return f'nL{nL}_nN{nN}_dp{dp}{bn}'
    

def get_dataSets(train_df, val_df, config_name, target_path, seed, tagger, decay_type):
    # Path to where the scaler parameters will be saved
    print("Preparing datasets...", flush=True)

    #Increase reproducibility by setting the seed
    torch.manual_seed(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    scalerPath = f"{target_path}/st_scaler.pkl"
    transformerPath = f"{target_path}/powerTransformer.pkl"

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if 'config_test' not in config_name: #Skip this step when testing to accelerate testing process
        if 'domain' in train_df.columns:
            stats_printout(tagger=tagger, decay_type=decay_type, train_df=train_df.loc[train_df.domain == 1], val_df=val_df.loc[val_df.domain == 1], BID=BID)
        else:
            stats_printout(tagger=tagger, decay_type=decay_type, train_df=train_df, val_df=val_df, BID=BID)
    print(f"Training set has {train_df[train_df.label==1].shape[0]} correctly tagged tracks, {train_df[train_df.label==0].shape[0]} wrong tagged tracks")
    
    columns_to_drop = ['event_entry', 'selected', f"{tagger}_TagDec", BID]
    
    print(train_df.drop(columns = columns_to_drop).columns)
    print(features)
    
    train_ds, validation_ds = pyTrain.prepare_data(train_df=train_df.drop(columns = columns_to_drop), val_df=val_df.drop(columns = columns_to_drop), scalerPath=scalerPath, transformerPath=transformerPath)
    
    if 'config_test' not in config_name:
        pyTrain.plot_features(data=train_df, features_list=features, target_path=target_path, flag='label', name=f'training_inputFeatures')
    print(f"Datasets prepared", flush=True)
    return train_ds, validation_ds

def training(train_ds, validation_ds, vars, target_path, tagger, seed, features, 
             config, data_type, repo, num_threads = 1, clean = False):
    torch.jit.enable_onednn_fusion(True)
    start = datetime.datetime.now()
    print(f'Training started on {start.strftime("%Y-%m-%d %H:%M:%S")}')

    # Load YAML configuration file
    with open(f'{config}', 'r') as file:
        config = yaml.safe_load(file)
    print(vars)

    
    # Check and eventually make output directory where training info will be saved
    pyTrain.recreate_directory(target_path, clean=clean)
    device ="cpu"

    if data_type == 'domain_adapted':
        model = NNDomainAdapted(features=features, architecture=get_architecture(config), seed=seed, optimizer_kwargs={"lr" : config['learning_rate']},
                            repo_path=repo, alpha=config['alpha']).to(device)
    else:
        model = NeuralNetwork(features=features, architecture=get_architecture(config), seed=seed, 
                            optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=repo).to(device)
    print(f"\nThe NN architecture is: \n{model}\n")

    print(f'{num_threads} threads will be used for training', flush=True)

    if num_threads>1:
        if os.path.exists(f"{target_path}/port.temp"):#Remove port file if it remained after previous failed execution
            os.remove(f"{target_path}/port.temp")

        return_dict = mp.Manager().dict()
        return_dict['early_stopping'] = False #Flag to signal early stopping to all processes
        train_ds_name = f'train_set{id(train_ds)}'
        validation_ds_name = f'validation_set{id(validation_ds)}'
        if not isinstance(train_ds, SharedDataset):
            train_ds = SharedDataset(train_ds, train_ds_name)
            validation_ds = SharedDataset(validation_ds, validation_ds_name)

        mp.spawn(pyTrain.train_worker, args=(model, train_ds, 
                                validation_ds, target_path, 
                                config, seed, return_dict,
                                num_threads), nprocs=num_threads)

        train_ds.unlink(train_ds_name)
        validation_ds.unlink(validation_ds_name)
        os.remove(f"{target_path}/port.temp")
    else:
        return_dict = {}
        pyTrain.train_model_EarlyStopping(0, model, train_ds, 
                                validation_ds, target_path, 
                                config = config,
                                seed = seed,
                                return_dict = return_dict,
                                num_threads=num_threads)
    
    print(return_dict)

    bestModel = return_dict['bestModel']
    pyTrain.save_model(bestModel, target_path)

    pyTrain.plot_losses(tagger, return_dict['trainingEpoch_loss'], return_dict['validationEpoch_loss'], return_dict['bestEpoch'], return_dict['bestLosses'], target_path)
    pyTrain.save_losses(return_dict['trainingEpoch_loss'], return_dict['validationEpoch_loss'], return_dict['bestEpoch'], return_dict['bestLosses'], target_path)

    end = datetime.datetime.now()
    print(f'Training ended on {end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Training time: {end - start}')
    return bestModel



def gen_training_plots(model, num_threads, train_df, val_df, train_ds, validation_ds, target_path, tagger):
    # Plot ROC curves for validation and train test
    model.eval()
    print("Generating Plots")

    start = datetime.datetime.now()
    print(f"Evaluating model on validation and test set {start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if 'domain' in train_df.columns:
        #TODO Change to new parallelized inference function

        pred, true = model.evaluate_model(validation_ds)


        val_df['yPred'] = pred[:,0]
        val_df['dPred'] = pred[:,1] #Domain Pred
        val_df['yTrue'] = true[:,0]
        val_df['dTrue'] = true[:,1] #Domain true


        pred, true = model.evaluate_model(train_ds)
        train_df['yPred'] = list(pred[:,0])
        train_df['dPred'] = list(pred[:,1]) #Domain Pred
        train_df['yTrue'] = list(true[:,0])
        train_df['dTrue'] = list(true[:,1]) #Domain true
        del pred, true
    else:
        val_df['yPred'], val_df['yTrue'] = pyTrain.infere_model(model, validation_ds, target_path, num_threads)
        print('Validation set done', flush=True)
        train_df['yPred'], train_df['yTrue'] = pyTrain.infere_model(model, train_ds, target_path, num_threads)
        print('Training set done', flush=True)

    end = datetime.datetime.now()
    print(f"Evaluation done {end.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


    if 'domain' in train_df.columns:
        pyTrain.plot_ROC(tagger=tagger, val_df=val_df, train_df=train_df, target_path =target_path, 
                         trueLabel= 'dTrue', predLabel='dPred', fileLabel='domain_')
        pyTrain.plot_mistag(tagger=tagger, df=train_df, target_path=target_path, type = 'Training', show_trueB=False, 
                            trueLabel= 'dTrue', predLabel='dPred', fileLabel='domain_', correct_legend= "Domain 0", wrong_legend= "Domain 1")
        plt.figure()
        plt.hist(1-train_df['dPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
        plt.title(r"Training set: Probability of label 0, only selected")
        plt.yscale("log")
        plt.savefig(f"{target_path}/domain_trainingSet_prob0distrib.pdf")
        plt.figure()
        plt.hist(train_df['dPred'] ,bins = 100 , density = True , histtype = "stepfilled" )
        plt.title(r"Training set: Probability of label 1, only selected")
        plt.yscale("log")
        plt.savefig(f"{target_path}/domain_trainingSet_prob1distrib.pdf")

        # Calculate and print accuracies, both for domain and class
        # make sure the accuracies are calculated with each domain / class being equally often represented
        domain_weight = val_df['dTrue'].value_counts(normalize=True).to_dict()
        domain_weight = {k: 1/v for k, v in domain_weight.items()}
        class_weight = val_df['yTrue'].value_counts(normalize=True).to_dict()
        class_weight = {k: 1/v for k, v in class_weight.items()}

        print(f'Domain accuracy: {accuracy_score(val_df["dTrue"], val_df["dPred"]>0.5, sample_weight=val_df["dTrue"].map(domain_weight))}')
        val_df.dropna(subset=['yTrue'], inplace=True)
        train_df.dropna(subset=['yTrue'], inplace=True)    
        print(f'Class accuracy: {accuracy_score(val_df["yTrue"], val_df["yPred"]>0.5, sample_weight=val_df["yTrue"].map(class_weight))}')

    pyTrain.plot_ROC(tagger=tagger, val_df=val_df, train_df=train_df, target_path =target_path)


    pyTrain.plot_mistag(tagger=tagger, df=train_df, target_path=target_path, type = 'Training', show_trueB=False,)
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
    parser.add_argument('--data_type', help="Type of Data used, MC, Data or domain_adapted when using domain adaptation",choices=('MC', 'Data', 'domain_adapted'))
    parser.add_argument('--num_threads', help='Number of threads to use in training', type=int, default=1)

    cfg = parser.parse_args()
    pprint(cfg)
    matplotlib.rcParams.update({'axes.unicode_minus':False,})

    print(f"training with {cfg.num_threads} threads")

    BID = 'B_ID'

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)

    vars = features + [BID,'selected', 'label',f"{cfg.tagger}_TagDec"] 
    if cfg.data_type == 'domain_adapted':
        vars = vars + ['domain']

    print(vars)


    #Reading Data from files
    print(f'Reading of training files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.training_data)} files.", flush=True)
    train_df = read_files(cfg.training_data, vars = vars, treename=cfg.treename, event_type=cfg.decay_type, data_type=cfg.data_type)
    print(f'Reading of training files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    print(f'Reading of validation files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.validation_data)} files.", flush=True)
    val_df = read_files(cfg.validation_data, vars = vars, treename=cfg.treename, event_type=cfg.decay_type, data_type=cfg.data_type)
    print(f'Reading of validation files ends {datetime.datetime.now().strftime("%H:%M:%S")}')
    

    print(train_df.shape, flush=True)
    print(val_df.shape, flush=True)

    if 'domain' in train_df.columns:
        print(f'number of label 0 in training set: {train_df[train_df.label==0].shape[0]}')
        print(f'number of label 1 in training set: {train_df[train_df.label==1].shape[0]}')
        print(f'number of BID 521 in training set: {train_df[train_df[BID]==521].shape[0]}')
        print(f'number of BID -521 in training set: {train_df[train_df[BID]==-521].shape[0]}')
        print(f'number of domain 0 in training set: {train_df[train_df.domain==0].shape[0]}')
        print(f'number of domain 1 in training set: {train_df[train_df.domain==1].shape[0]}')

    train_ds, validation_ds = get_dataSets(train_df=train_df, val_df=val_df, config_name=cfg.config, 
                                           target_path=cfg.target_path, seed=cfg.seed, tagger=cfg.tagger, 
                                           decay_type=cfg.decay_type)

    model = training(train_ds=train_ds, validation_ds=validation_ds, vars=vars, data_type=cfg.data_type, 
                     target_path=cfg.target_path, tagger=cfg.tagger, seed=cfg.seed, features=features, 
                     config=cfg.config, repo=cfg.repo, num_threads=cfg.num_threads, clean=cfg.clean)

    gen_training_plots(model, cfg.num_threads, train_df, val_df, train_ds, validation_ds, cfg.target_path, cfg.tagger)

    print(f"Training finished, model saved in {cfg.target_path}")