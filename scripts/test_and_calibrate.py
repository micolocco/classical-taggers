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
from scripts.train_tagger import get_architecture
from scripts.NNModel import NeuralNetwork
from scripts.NNModel import NNDomainAdapted
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
# import utils
import psutil
import json
from os.path import join

def read_files_reduce_unselected(files, vars, treename, seed):
    df = pd.DataFrame(columns=vars)

    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB at {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)


        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars+ ["file_id", "RUNNUMBER", "EVENTNUMBER"], library="pd")
        _df.dropna(inplace = True)


        _df["event_entry"] = _df["file_id"].astype(str) + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER', 'file_id'], inplace=True)
        
        #Drop rows which are not selected. Keep one per unique event_entry to ensure correct efficiency calculation
        print(f'Number of events with selected tracks before reduction: {_df[_df["selected"] == 1].event_entry.nunique()} out of {_df.event_entry.nunique()} total events', flush = True)

        _df = pd.concat([
            _df[_df['selected'] == 1],
            _df[_df['selected'] == 0].drop_duplicates('event_entry')
        ]).reset_index(drop=True)
        _df = _df.sample(frac=1, random_state=seed).reset_index(drop=True)



        df = pd.concat([df, _df], ignore_index = True)

    return df

def study_eta_omega_dist(df, split_by, prefix, target_path, tagger, data_type):
    bins = pyTrain.bins_by_yield(etas=df[f"{tagger}_Eta"].to_numpy(), weights=None, nbins= 25)
    
    #Remove any duplicate bin edges. May happen with very strong bunching around eta = 0.5
    bins = np.unique(bins)

    translation_dict = {'B_TRUEID': {-521: r'$B^-$', 521: r'$B^+$',-511: r'$\bar{B}^0$', 511: r'$B^0$'},
                        'B_ID': {-521: r'$B^-$', 521: r'$B^+$',-511: r'$\bar{B}^0$', 511: r'$B^0$'},
                        'OSKaon_TagDec': {1: 'Positive', -1: 'Negative'},
                        'OSMuon_TagDec': {1: 'Positive', -1: 'Negative'},
                        'OSElectron_TagDec': {1: 'Positive', -1: 'Negative'},
                        }


    plt.figure(figsize=(10, 6))

    if len(bins) > 2: #If there is only one or no bin the tagger is especially bad and does not need to be considered
        bin_contents = {}
        for split in np.unique(df[split_by]):
            
            if split == 0:
                continue #Skip untagged events/tracks

            bin_eta_means = []
            bin_eta_devs = []
            bin_omega_means = []
            bin_omega_devs = []
            for lower ,upper  in zip(bins[:-1], bins[1:]):
                mistags = 1-df.loc[(df[f"{tagger}_Eta"] > lower) & (df[f"{tagger}_Eta"] <= upper) & (df[split_by] == split), 'label'         ].to_numpy()
                etas =    df.loc[(df[f"{tagger}_Eta"] > lower) & (df[f"{tagger}_Eta"] <= upper) & (df[split_by] == split), f"{tagger}_Eta" ].to_numpy()
                if data_type == 'Data':
                    weights = df.loc[(df[f"{tagger}_Eta"] > lower) & (df[f"{tagger}_Eta"] <= upper) & (df[split_by] == split), "signal_weights"].to_numpy()
                else:
                    weights = np.ones_like(etas)

                    


                if np.sum(weights) != 0:
                    omega_mean = np.average(mistags, weights=weights)
                    omega_std_dev = np.sqrt(omega_mean*(1-omega_mean)/np.sum(weights))

                    eta_mean = np.average(etas, weights=weights)
                    eta_std_dev = np.sqrt(np.average((etas - eta_mean)**2, weights=weights))
                else:
                    omega_mean = np.nan
                    omega_std_dev = np.nan
                    eta_mean = np.nan
                    eta_std_dev = np.nan

                bin_omega_means.append(omega_mean)
                bin_omega_devs.append(omega_std_dev)
                bin_eta_means.append(eta_mean)
                bin_eta_devs.append(eta_std_dev)

            # prefix = 'B_' if split > 0 else 'Bbar_'
            bin_contents[f'{split}_eta_means'] =       bin_eta_means
            bin_contents[f'{split}_eta_means_unc'] =   bin_eta_devs
            bin_contents[f'{split}_omega_means'] =     bin_omega_means
            bin_contents[f'{split}_omega_means_unc'] = bin_omega_devs



            plt.errorbar(x=bin_eta_means, y=bin_omega_means, xerr=bin_eta_devs, yerr=bin_omega_devs, fmt='o', label=f'{translation_dict[split_by][split]}')

            # bin_omega_means = np.array(bin_omega_means)
            # bin_omega_devs = np.array(bin_omega_devs)

            # #Calculate increase per bin and uncertainty of it
            # increases = bin_omega_means[1:] - bin_omega_means[:-1]
            # increases_unc = np.sqrt(bin_omega_devs[1:]**2 + bin_omega_devs[:-1]**2)

            # bin_increases_by_id[str(split)] = increases
            # bin_increases_by_id[str(split) + '_unc'] = increases_unc
        
            # print(f'increases of {split_by}: {increases} +/- {increases_unc}')

        plt.legend(loc='upper left')
        plt.xlabel(r'Predicted mistag $\eta$')
        plt.ylabel(r'Measured mistag $\omega$')
        plt.tight_layout()
        plt.savefig(f"{target_path}/{prefix}_eta_omega_bins.pdf")
        plt.clf()
    else: #If there are only one or no bins give out nan as the increases
        labels = ['B_eta_means',    'B_eta_means_unc',    'B_omega_means',    'B_omega_means_unc',
                  'Bbar_eta_means', 'Bbar_eta_means_unc', 'Bbar_omega_means', 'Bbar_omega_means_unc']
        bin_contents =  {str(split): np.nan for split in labels}

    #dump bin_increases_by_id in yaml file. Later used to reject taggers where omega does not strictly rise with eta
    with open(f"{target_path}/{prefix}_eta_omega_bins.pkl", "wb") as f:
        pickle.dump(bin_contents, f)


def testing_pipeline(test_df, BID, target_path, train_path, tagger, features, config,
                     decay_type, seed, repo, data_type, model_path, domain_adapted=False, num_threads=1):
    start = datetime.datetime.now()
    print(f'Testing started on {start.strftime("%Y-%m-%d %H:%M:%S")}', flush = True)
    # Load YAML configuration file
    with open(f'{config}', 'r') as file:
        config_dict = yaml.safe_load(file)


    # Path to where the scaler parameters will be saved
    scalerPath = f"{train_path}/st_scaler.pkl"
    transformerPath = f"{train_path}/powerTransformer.pkl"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used: {device}")
    

    test_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    #Load model
    bestModel = NeuralNetwork(features=features, architecture=get_architecture(config_dict), seed=seed, optimizer_kwargs={"lr" : config_dict['learning_rate']}, repo_path=repo)
    
    
    if not domain_adapted:
        pyTrain.load_model(model=bestModel, target_path=model_path)
    else:
        pyTrain.load_model_without_domain_classifier(model=bestModel, target_path=model_path)

    bestModel.eval()

    columns_to_drop = ['event_entry', 'selected', f"{tagger}_TagDec", BID,]#, 'label']#, 'B_DTF_PV_Jpsi_MASS']
    if data_type == 'Data' :
        columns_to_drop.append('signal_weights')
        sweights = test_df['signal_weights']

        columns_to_drop.append('B_TAU')

    # Adjust test dataframe as input for the NN. Note: only selected track=1 are needed
    print(f'Columns:{test_df.columns}')

    test_dataset = inputDataset(df=test_df.drop(columns = columns_to_drop))
    test_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    test_dl = DataLoader(test_dataset, batch_size = 1024, shuffle=False)


    start_time = time.time()
    # yPred, yTrue = bestModel.evaluate_model(test_dl)
    test_df['yPred'], test_df['yTrue'] = bestModel.evaluate_model(test_dl)
    
    print(f"Time taken for inference: {time.time() - start_time:.2f} seconds", flush=True)
    
    # start_time = time.time()
    # test_df['yPred'], test_df['yTrue'] = pyTrain.infere_model(model=bestModel, ds=test_dataset, target_path=target_path, num_threads=num_threads)
    # print(f"Time taken for parallel inference: {time.time() - start_time:.2f} seconds", flush=True)

    # print("trues")
    # print(list(yTrue))
    # print(test_df['yTrue'].tolist())

    test_df[f"{tagger}_Eta"] = 1 - test_df['yPred']
    cols_to_keep = ['event_entry','selected', f"{tagger}_Eta", f"{tagger}_TagDec", 'label',BID, 'yTrue', 'yPred', ]
    cols_to_keep.append('B_TAU')
    test_df.drop(columns=test_df.columns.difference(cols_to_keep), inplace=True)

    print(f'Columns after prediction: {test_df.columns}', flush = True)


    print(f"Test set has {np.sum(test_df['selected']==1)} tracks selected as tagging particles")
    print(f"Test set has {test_df[(test_df['selected']==1)&(test_df['label']==0)].shape[0]} wrong tagged tracks, {test_df[(test_df['selected']==1)&(test_df['label']==1)].shape[0]} correctly tagged tracks", flush = True)
    
    pyTrain.plot_ROC(tagger=tagger, val_df=test_df[test_df['selected'] == 1], target_path =target_path)
    pyTrain.plot_mistag(tagger=tagger, df=test_df[test_df['selected'] == 1], target_path=target_path, type = 'Test')


    


    test_df.loc[test_df.selected == 0, f"{tagger}_TagDec"] = 0  # classic
    test_df.loc[test_df.selected == 0, f"{tagger}_Eta"] = 0.5  # classic

    pyTrain.plot_tagDec(tagger =tagger, df_TagParticles=test_df.sort_values(by = ["selected",f"{tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first(), plot_name=f'{target_path}/Not_Normalized_TagDec.pdf')
    if data_type == 'Data':
        test_df['signal_weights'] = sweights


    # Check if measured mistag is stricly rising depending on predicted mistag -> only then a good tagger
    study_eta_omega_dist(test_df.sort_values(by = ["selected",f"{tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first(), BID               , f'{BID}_bestTracks_'  , target_path, tagger, data_type)


    df_TagParticles = test_df.sort_values(by = ["selected",f"{tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first()
    del test_df

    print(f"Number of events with selected tracks: {df_TagParticles[df_TagParticles['selected'] == 1].shape[0]} out of {df_TagParticles.shape[0]} total events", flush = True)
    
    if data_type == 'Data':
        sweights_TagParticles = df_TagParticles['signal_weights'].to_numpy().astype(np.float64)
    else:
        sweights_TagParticles = None
    #df_TagParticles = test_df.sort_values(by = ["selected",f"{tagger}_Eta"] , ascending = [False,True]).groupby("event_entry").first() test this 
    
    print(f"{df_TagParticles.shape[0]} tracks used for calibrating", flush = True)
    pyTrain.plot_tagDec(tagger =tagger, df_TagParticles=df_TagParticles,  plot_name=f'{target_path}/Normalized_TagDec.pdf')

    mode = decay_type[:2]
    if data_type == 'MC':
        mode = 'Bu' #When truth information is availiable Bd or Bs mode is not needed
    
    
    

    # Calibrating the tagger and saving parameters
    mistag_info = pyTrain.calibration(tagger=tagger, df_tag=df_TagParticles, eventType=decay_type, target_path=target_path, weights=sweights_TagParticles, mode=mode)
    
    # Try both calibration functions
    logit_info  = pyTrain.calibration(tagger=tagger, df_tag=df_TagParticles, eventType=decay_type, target_path=target_path, weights=sweights_TagParticles, mode=mode, calibration_option='logit', )


    end = datetime.datetime.now()
    print(f'testing ended on {end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Time taken for testing and calibration: {end - start}', flush = True)

    return mistag_info['TaggingPower'], logit_info['TaggingPower']
     

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
    parser.add_argument('--model_path', help='Path to trained model', type=str)
    parser.add_argument('--domain_adapted', action='store_true', help='Model is domain adapted', )
    parser.add_argument('--num_threads', help='Number of threads to use for inference', default=1, type=int)

    cfg = parser.parse_args()
    pprint(cfg)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)
    BID = 'B_ID'

    features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)
    print(f"Features used: {features}", flush = True)

    vars = features + [BID, 'selected', 'label', f"{cfg.tagger}_TagDec"]
    if cfg.data_type == 'Data':
        vars = vars + ['signal_weights']
        if 'Bu' not in cfg.decay_type:
            vars.append('B_TAU')
    print(vars, flush = True)




    #Reading Data from files
    print(f'Reading of test files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(cfg.testing_data)} files.", flush=True)
    test_df = read_files_reduce_unselected(cfg.testing_data, vars = vars, treename=cfg.treename, seed=cfg.seed)
    print(f'Reading of test files ends {datetime.datetime.now().strftime("%H:%M:%S")}', flush = True)




    print(f'Number of Tracks in test set: {test_df.shape[0]}', flush = True)
    print(f'Number of selected tracks in test set: {test_df.selected.sum()}', flush = True)
    print(f'Number of events in test set: {test_df.event_entry.nunique()}', flush = True)
    print(f'Number of events with selected tracks in test set: {test_df[test_df.selected == 1].event_entry.nunique()}', flush = True)
    print(f'Average number of tracks per event: {test_df.shape[0]/test_df.event_entry.nunique()}', flush = True)
    print(f'Average number of selected tracks per event: {test_df.selected.sum()/test_df[test_df.selected == 1].event_entry.nunique()}', flush = True)

    testing_pipeline(test_df=test_df, BID=BID, target_path=cfg.target_path, train_path=cfg.train_path, 
                     tagger=cfg.tagger, features=features, config=cfg.config, decay_type=cfg.decay_type, 
                     seed=cfg.seed, repo=cfg.repo, data_type=cfg.data_type, model_path = cfg.model_path,
                     domain_adapted=cfg.domain_adapted, num_threads=cfg.num_threads)