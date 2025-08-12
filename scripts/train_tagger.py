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
from scripts.NNModel import NNDomainAdapted
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import utils
import psutil
from scripts.shareddataset import SharedDataset

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from rich.console import Console
from rich.table import Table
from io import StringIO
import itertools
import matplotlib

from sklearn.metrics import accuracy_score


def stats_printout(tagger, decay_type, train_df, val_df, BID):
    '''
    Function to print statistics about the dataset composition
    '''

    use_weights = 'signal_weights' in train_df.columns
    print(use_weights)
    def count_events(df, selected=None, weighted=False):
        if selected is not None:
            df = df[df.selected == selected]
        if weighted:
            return df.groupby('event_entry')['signal_weights'].first().sum()
        else:
            return df['event_entry'].nunique()

    def count_tracks(df, selected=None, weighted=False):
        if selected is not None:
            df = df[df.selected == selected]
        if weighted:
            return df['signal_weights'].sum()
        else:
            return df.shape[0]

    # Compute all stats, both weighted and unweighted if signal_weights present
    stats = {}
    for weighted in ([False, True] if use_weights else [False]):
        suffix = "_sweighted" if weighted else ""
        stats[f"train_evts{suffix}"] = count_events(train_df, weighted=weighted)
        stats[f"train_sel_evts{suffix}"] = count_events(train_df, selected=1, weighted=weighted)
        stats[f"val_evts{suffix}"] = count_events(val_df, weighted=weighted)
        stats[f"val_sel_evts{suffix}"] = count_events(val_df, selected=1, weighted=weighted)
        stats[f"tot_evts{suffix}"] = stats[f"train_evts{suffix}"] + stats[f"val_evts{suffix}"]
        stats[f"sel_evts{suffix}"] = stats[f"train_sel_evts{suffix}"] + stats[f"val_sel_evts{suffix}"]
        stats[f"train_tracks{suffix}"] = count_tracks(train_df, weighted=weighted)
        stats[f"train_sel_tracks{suffix}"] = count_tracks(train_df, selected=1, weighted=weighted)
        stats[f"val_tracks{suffix}"] = count_tracks(val_df, weighted=weighted)
        stats[f"val_sel_tracks{suffix}"] = count_tracks(val_df, selected=1, weighted=weighted)
        stats[f"tot_tracks{suffix}"] = stats[f"train_tracks{suffix}"] + stats[f"val_tracks{suffix}"]
        stats[f"sel_tracks{suffix}"] = stats[f"train_sel_tracks{suffix}"] + stats[f"val_sel_tracks{suffix}"]

    print(f"\n Statistics used in the {tagger} pipeline\n")

    console = Console()
    table = Table(show_header=True)
    table.width = 120
    table.add_column("", justify="left")
    table.add_column("Events", justify="left", style='cyan')
    table.add_column("Tracks", justify="left", style='green')
    if use_weights:
        table.add_column("Events (sweighted)", justify="left", style='cyan')
        table.add_column("Tracks (sweighted)", justify="left", style='green')

    def fmt(val, weighted):
        return f"{val:.2f}" if weighted else f"{int(val)}"

    # Unweighted row
    table.add_row(
        "Before selection",
        fmt(stats["tot_evts"], False),
        fmt(stats["tot_tracks"], False),
        fmt(stats["tot_evts_sweighted"], True) if use_weights else "",
        fmt(stats["tot_tracks_sweighted"], True) if use_weights else ""
    )
    table.add_row(
        "After selection",
        fmt(stats["sel_evts"], False),
        fmt(stats["sel_tracks"], False),
        fmt(stats["sel_evts_sweighted"], True) if use_weights else "",
        fmt(stats["sel_tracks_sweighted"], True) if use_weights else ""
    )
    table.add_row(
        "Train",
        fmt(stats["train_evts"], False),
        fmt(stats["train_tracks"], False),
        fmt(stats["train_evts_sweighted"], True) if use_weights else "",
        fmt(stats["train_tracks_sweighted"], True) if use_weights else ""
    )
    table.add_row(
        "Validation",
        fmt(stats["val_evts"], False),
        fmt(stats["val_tracks"], False),
        fmt(stats["val_evts_sweighted"], True) if use_weights else "",
        fmt(stats["val_tracks_sweighted"], True) if use_weights else ""
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

    def count_label(df, label, bid, weighted=False):
        mask = (df.label == label) & (df[BID] == bid)
        if weighted:
            return df.loc[mask, 'signal_weights'].sum()
        else:
            return df.loc[mask].shape[0]

    label_stats = {}
    for weighted in ([False, True] if use_weights else [False]):
        suffix = "_sweighted" if weighted else ""
        label_stats[f"B_correct_train{suffix}"] = count_label(train_df, 1, -ID, weighted=weighted)
        label_stats[f"antiB_correct_train{suffix}"] = count_label(train_df, 1, ID, weighted=weighted)
        label_stats[f"B_wrong_train{suffix}"] = count_label(train_df, 0, -ID, weighted=weighted)
        label_stats[f"antiB_wrong_train{suffix}"] = count_label(train_df, 0, ID, weighted=weighted)

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
    if use_weights:
        table.add_column("l=1, B (sweighted)", justify="left", style='cyan', overflow="fold", width=8)
        table.add_column("l=1, antiB (sweighted)", justify="left", style='cyan', overflow="fold", width=8)
        table.add_column("asymm (sweighted)", justify="left", style='cyan', overflow="fold", width=8)
        table.add_column("l=0, B (sweighted)", justify="left", style='green', overflow="fold", width=8)
        table.add_column("l=0, antiB (sweighted)", justify="left", style='green', overflow="fold", width=8)
        table.add_column("asymm (sweighted)", justify="left", style='green', overflow="fold", width=8)

    # Unweighted row
    row = [
        "Training set",
        fmt(label_stats["B_correct_train"], False),
        fmt(label_stats["antiB_correct_train"], False),
        asymm(label_stats["B_correct_train"], label_stats["antiB_correct_train"]),
        fmt(label_stats["B_wrong_train"], False),
        fmt(label_stats["antiB_wrong_train"], False),
        asymm(label_stats["B_wrong_train"], label_stats["antiB_wrong_train"])
    ]
    if use_weights:
        row += [
            fmt(label_stats["B_correct_train_sweighted"], True),
            fmt(label_stats["antiB_correct_train_sweighted"], True),
            asymm(label_stats["B_correct_train_sweighted"], label_stats["antiB_correct_train_sweighted"]),
            fmt(label_stats["B_wrong_train_sweighted"], True),
            fmt(label_stats["antiB_wrong_train_sweighted"], True),
            asymm(label_stats["B_wrong_train_sweighted"], label_stats["antiB_wrong_train_sweighted"])
        ]
    table.add_row(*row)
    # console.print(table)

    output = StringIO()
    console = Console(file=output, width=200)
    console.print(table)
    table_str = output.getvalue()
    print(table_str)
    output.close()



def read_files(files, vars, treename, reduce = False, weight_label = None,balance_data= False):
    df = pd.DataFrame(columns=vars)

    additional_vars = ['RUNNUMBER', 'EVENTNUMBER']

    if balance_data:
        additional_vars = additional_vars + ['B_Tr_T_Charge']
        if weight_label is not None:
            additional_vars = additional_vars + ['BID_signal_weights']

    if 'domain' in vars:
        additional_vars = additional_vars +['file_id']

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')

        if 'domain' not in vars:
            id = os.path.basename(f)[:-5]
            if id[-7:-2] == '.data':
                id = id[:-7]
            else:
                id = id[:-3]


        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars + additional_vars, library="pd")
        
        print(f"Number of tracks in file {i+1}: {_df.shape[0]}", flush=True)

        if 'domain' not in vars:
            _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        else:
            _df.loc[_df['domain'] == 0, 'event_entry'] = _df["file_id"].astype(str) + "_" + "data" + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
            _df.loc[_df['domain'] == 1, 'event_entry'] = _df["file_id"].astype(str) + "_" + "mc" + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)

        if reduce: #Probably take this out after bkg rejection
            df_event = _df.groupby("event_entry").first().reset_index()
            n_keep = int(len(df_event) * 0.4)

            # Get the indices of the rows with the highest weights
            top_indices = df_event[weight_label].nlargest(n_keep).index

            # Filter the DataFrame to keep only those rows, preserving original order
            df_event = df_event.loc[df_event.index.isin(top_indices)]

            _df = _df[_df['event_entry'].isin(df_event['event_entry'])].reset_index(drop=True)
            del df_event


        df = pd.concat([df, _df], ignore_index = True)
    del _df
    
    if balance_data:
        BIDs = df[BID].unique()
        trackCharges = df['B_Tr_T_Charge'].unique()

        df.groupby('event_entry')

        def num_events(df):
            if 'BID_signal_weights' in df.columns:
                return np.sum(df['BID_signal_weights'])
            else:
                return len(df)

        df_events = df.groupby('event_entry').first()
        yield_by_ID = [num_events(df_events[df_events[BID] == ID]) for ID in BIDs]
        min_yield = min(yield_by_ID)

        print(f"Before Event dropping:")
        print(f"IDs: {BIDs}")
        print(f"Yield by ID: {yield_by_ID}")
        print(f"Number of events with {BIDs[0]}: {len(df_events[df_events[BID] == BIDs[0]])}")
        print(f"Number of events with {BIDs[1]}: {len(df_events[df_events[BID] == BIDs[1]])}", flush=True)

        print(df.head(), flush=True)

        #Drop random events until the yield of both BIDs is roughly equal
        for ID, ID_yield in zip(BIDs, yield_by_ID):
            while ID_yield > min_yield:
                # Randomly select an event to drop

                n = int(ID_yield - min_yield)
                if n == 0:
                    n = 1
                event_to_drop = df_events[df_events[BID] == ID].sample(n=n, random_state=42).index
                #Drop all events_entries in event_to_drop#
                df = df[~df['event_entry'].isin(event_to_drop)]

                df_events = df.groupby('event_entry').first()
                ID_yield = num_events(df_events[df_events[BID] == ID])

        df_events = df.groupby('event_entry').first()
        yield_by_ID = [num_events(df_events[df_events[BID] == ID]) for ID in BIDs]

        print(f"After Event dropping:")
        print(f"IDs: {BIDs}")
        print(f"Yield by ID: {yield_by_ID}")
        print(f"Number of events with {BIDs[0]}: {len(df_events[df_events[BID] == BIDs[0]])}")
        print(f"Number of events with {BIDs[1]}: {len(df_events[df_events[BID] == BIDs[1]])}", flush=True)
        del df_events
                
        numTracks = [
                len(df[(df[BID] == BIDs[0]) & (df["B_Tr_T_Charge"] == trackCharges[0])]),
                len(df[(df[BID] == BIDs[0]) & (df["B_Tr_T_Charge"] == trackCharges[1])]),
                len(df[(df[BID] == BIDs[1]) & (df["B_Tr_T_Charge"] == trackCharges[0])]),
                len(df[(df[BID] == BIDs[1]) & (df["B_Tr_T_Charge"] == trackCharges[1])]),
        ]
        minTracks = np.min(numTracks)
        print(numTracks, flush=True)

        #Drop random tracks until all BID and Trackcharge combinations have same number of tracks
        for b, c in itertools.product(BIDs, trackCharges):
            comb = df[(df[BID] == b) & (df["B_Tr_T_Charge"] == c)]
            print(f"Before dropping tracks for {b}, {c}: {len(comb)} tracks", flush=True)
            if len(comb) > minTracks:
                idxs = comb.sample(n=len(comb) - minTracks, random_state=42).index
                df.drop(idxs, inplace=True)

        numTracks = [
            len(df[(df[BID] == BIDs[0]) & (df["B_Tr_T_Charge"] == trackCharges[0])]),
            len(df[(df[BID] == BIDs[0]) & (df["B_Tr_T_Charge"] == trackCharges[1])]),
            len(df[(df[BID] == BIDs[1]) & (df["B_Tr_T_Charge"] == trackCharges[0])]),
            len(df[(df[BID] == BIDs[1]) & (df["B_Tr_T_Charge"] == trackCharges[1])]),
        ]
        print(numTracks, flush=True)


    df.drop(columns=additional_vars, inplace=True)
        
    return df

def get_architecture(config):
    if 'architecture' in config.keys():
        return config['architecture']
    else:
        nL = config['numlayers']
        nN = config['numneurons']
        dp = config['dropout']
        return f'nL{nL}_nN{nN}_dp{dp}'

def get_dataSets(train_df, val_df, config_name, target_path, data_type, weight_type, seed, tagger, decay_type, indexed = True):
    # Path to where the scaler parameters will be saved
    torch.manual_seed(seed=seed)
    scalerPath = f"{target_path}/st_scaler.pkl"
    transformerPath = f"{target_path}/powerTransformer.pkl"

    if data_type == 'Data':
        weight_label = weight_type

    train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df.sample(frac=1, random_state=seed).reset_index(drop=True)


    
    weights_train = None
    weights_val = None
    if data_type == 'Data' and weight_label != 'ones': 
        weights_train = train_df[weight_label].to_numpy()
        weights_val = val_df[weight_label].to_numpy()

    # For training: keep only tracks that pass the pre-selections. 
    # For calibration, events with 0 selected tracks must be kept. This is necessary to estimate the tagging efficiency correctly 
    # Training-validation sets splitting
    if 'domain' in train_df.columns:
        stats_printout(tagger=tagger, decay_type=decay_type, train_df=train_df.loc[train_df.domain == 1], val_df=val_df.loc[val_df.domain == 1], BID=BID)
    else:
        stats_printout(tagger=tagger, decay_type=decay_type, train_df=train_df, val_df=val_df, BID=BID)
    print(f"Training set has {train_df[train_df.label==1].shape[0]} correctly tagged tracks, {train_df[train_df.label==0].shape[0]} wrong tagged tracks")
    # Save test dataframe for calibration
    
    columns_to_drop = ['event_entry', 'selected', f"{tagger}_TagDec", BID]#, 'B_DTF_PV_Jpsi_MASS']
    if data_type == 'Data' and weight_label != 'ones':
        columns_to_drop.append(weight_label)
        if weight_label != 'signal_weights':
            columns_to_drop.append('signal_weights')

    
    print(train_df.drop(columns = columns_to_drop).columns)
    print(features)
    
    train_ds, validation_ds = pyTrain.prepare_data(train_df=train_df.drop(columns = columns_to_drop), val_df=val_df.drop(columns = columns_to_drop), seed=seed, scalerPath=scalerPath, transformerPath=transformerPath, indexed=indexed)
    
    if config_name!='configs/config_test':
        pyTrain.plot_features(data=train_df, features_list=features, target_path=target_path, flag='label', name=f'training_inputFeatures')
    
    return train_ds, validation_ds, weights_train, weights_val

def training(train_ds, validation_ds, vars,  weights_train, weights_val, 
             target_path, tagger, seed, features, config, data_type,
             repo, num_threads = 1, clean = False, logfile = None):
    if logfile is not None:
        from scripts.batch_train_tagger import ThreadLocalStdout
        sys.stdout = ThreadLocalStdout()
        sys.stdout.set_log_file(logfile)
        print(f'Trained in Batch mode')

    torch.jit.enable_onednn_fusion(True)
    

    start = datetime.datetime.now()
    print(f'Training started on {start.strftime("%Y-%m-%d %H:%M:%S")}')

    # Load YAML configuration file
    with open(f'{config}', 'r') as file:
        config = yaml.safe_load(file)
    print(vars)

    
    # Check and eventually make output directory where training info will be saved
    pyTrain.recreate_directory(target_path, clean=clean)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")

    if data_type == 'domain_adapted':
        model = NNDomainAdapted(features=features, architecture=get_architecture(config), seed=seed, optimizer_kwargs={"lr" : config['learning_rate']},
                               repo_path=repo, alpha=1).to(device)
    else:
        model = NeuralNetwork(features=features, architecture=get_architecture(config), seed=seed, 
                              optimizer_kwargs={"lr" : config['learning_rate']}, repo_path=repo).to(device)
    print(f"\nThe NN architecture is: \n{model}\n")

    print(f'{num_threads} threads will be used for training', flush=True)
    if num_threads>1:
        return_dict = mp.Manager().dict()
        train_ds_name = f'train_set{id(train_ds)}'
        validation_ds_name = f'validation_set{id(validation_ds)}'
        if not isinstance(train_ds, SharedDataset):
            train_ds = SharedDataset(train_ds, train_ds_name)
            validation_ds = SharedDataset(validation_ds, validation_ds_name)

        mp.spawn(pyTrain.train_model_EarlyStopping, args=(model, train_ds, 
                                          validation_ds, target_path, 
                                          config, return_dict,
                                          weights_train, weights_val, 
                                          num_threads), nprocs=num_threads)
        train_ds.unlink(train_ds_name)
        validation_ds.unlink(validation_ds_name)
        os.remove(f"{target_path}/port.temp")
    else:
        return_dict = {}
        pyTrain.train_model_EarlyStopping(0, model, train_ds, 
                                validation_ds, target_path, 
                                config = config,
                                return_dict = return_dict,
                                train_weights = weights_train, 
                                val_weights= weights_val, 
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

def gen_training_plots(model, train_df, val_df, train_ds, validation_ds, target_path, tagger):
    # Plot ROC curves for validation and train test
    model.eval()
    print("Generating Plots")

    start = datetime.datetime.now()
    print(f"Evaluating model on validation set {start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if 'domain' in train_df.columns:

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
        val_df['yPred'], val_df['yTrue'] = model.evaluate_model(validation_ds)
        train_df['yPred'], train_df['yTrue'] = model.evaluate_model(train_ds)
    
    end = datetime.datetime.now()
    print(f"Validation set evaluation done {end.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


    if 'domain' in train_df.columns:
        pyTrain.plot_ROC(tagger=tagger, val_df=val_df, train_df=train_df, target_path =target_path, 
                         trueLabel= 'dTrue', predLabel='dPred', fileLabel='domain_')
        pyTrain.plot_mistag(tagger=tagger, df=train_df, target_path=target_path, type = 'Training', show_trueB=False, BID = BID, 
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
    parser.add_argument('--weight_type', help="Type of sample weight to be used for training on data", choices=('signal_weights', 'pdf_ratio', 'ones'))
    parser.add_argument('--num_threads', help='Number of threads to use in training', type=int, default=1)
    parser.add_argument('--reduce', help='Whether to drop data samples with low weights', action='store_true', default=False)
    parser.add_argument('--balance_dataset', help='Whether to balance number of B_id and track charge tracks', action='store_true', default=False)

    cfg = parser.parse_args()
    pprint(cfg)
    matplotlib.rcParams.update({'axes.unicode_minus':False,})

    print(f"training with {cfg.num_threads} threads")

    BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)


    if cfg.data_type == 'Data':
        print(f'features are {features}')
        for i in range(len(features)):
            features[i] = features[i].replace("BPVIP", "OWNPVIP")
            features[i] = features[i].replace("B_TRUEID", "B_ID")

    vars = features + [BID,'selected', 'label',f"{cfg.tagger}_TagDec"] #'B_Tr_T_Charge',
    weight_label = None
    if cfg.data_type == 'Data':
        weight_label = cfg.weight_type
        if weight_label != 'ones':
            vars = vars + [weight_label]
        if weight_label != 'signal_weights':
            vars = vars + ['signal_weights']

    if cfg.data_type == 'domain_adapted':
        vars = vars + ['domain']

    print(vars)



    #Reading Data from files
    print(f'Reading of training files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.training_data)} files.", flush=True)
    # train_df = read_files(cfg.training_data, vars = vars, treename=cfg.treename, augmentation=False, reduce = False, weight_label=None)
    train_df = read_files(cfg.training_data, vars = vars, treename=cfg.treename, reduce = cfg.reduce, weight_label=weight_label, balance_data = cfg.balance_dataset)
    print(f'Reading of training files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    print(f'Reading of validation files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.validation_data)} files.", flush=True)
    # val_df = read_files(cfg.validation_data, vars = vars, treename=cfg.treename, augmentation=False, reduce = False, weight_label=None)
    val_df = read_files(cfg.validation_data, vars = vars, treename=cfg.treename, reduce = cfg.reduce, weight_label=weight_label, balance_data = cfg.balance_dataset)
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

    train_ds, validation_ds, weights_train, weights_val = get_dataSets(train_df=train_df, val_df=val_df, config_name=cfg.config, 
                                                                          target_path=cfg.target_path, data_type=cfg.data_type, 
                                                                          weight_type=cfg.weight_type, seed=cfg.seed, tagger=cfg.tagger, 
                                                                          decay_type=cfg.decay_type, indexed= cfg.num_threads == 1)

    model = training(train_ds=train_ds, validation_ds=validation_ds, vars=vars, weights_train=weights_train, data_type=cfg.data_type,
                                      weights_val=weights_val, target_path=cfg.target_path, tagger=cfg.tagger, seed=cfg.seed, 
                                      features=features, config=cfg.config, repo=cfg.repo, num_threads=cfg.num_threads, clean=cfg.clean)
    #No shared memory needed for plot generation, as such the inputDataset must be indexed
    train_ds.indexed = True
    validation_ds.indexed = True

    gen_training_plots(model, train_df, val_df, train_ds, validation_ds, cfg.target_path, cfg.tagger)

    print(f"Training finished, model saved in {cfg.target_path}")