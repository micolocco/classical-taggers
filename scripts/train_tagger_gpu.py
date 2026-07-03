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

from scripts.train_tagger import read_files, stats_printout, get_architecture, get_dataSets


def gen_training_plots(model, train_df, val_df, train_ds, validation_ds, target_path, tagger):
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
        val_df['yPred'], val_df['yTrue'] = model.evaluate_model(validation_ds, 'cuda') 
        print('Validation set done', flush=True)
        train_df['yPred'], train_df['yTrue'] = model.evaluate_model(train_ds, 'cuda')
        print('Training set done', flush=True)




    end = datetime.datetime.now()
    print(f"Evaluation done {end.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


    if 'domain' in train_df.columns:
        pyTrain.plot_ROC(tagger=tagger, val_df=val_df, train_df=train_df, target_path =target_path, 
                         trueLabel= 'dTrue', predLabel='dPred', fileLabel='domain_')
        pyTrain.plot_mistag(tagger=tagger, df=train_df, target_path=target_path, type = 'Training', show_trueB=False, 
                            trueLabel= 'dTrue', predLabel='dPred', fileLabel='domain_', correct_legend= "Domain 0", 
                            wrong_legend= "Domain 1", BID = BID)
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

def training(train_ds, validation_ds, vars, target_path, tagger, seed, features, 
             config, data_type, repo, num_threads = 1, clean = False):
    torch.jit.enable_onednn_fusion(True)
    start = datetime.datetime.now()
    print(f'Training started on {start.strftime("%Y-%m-%d %H:%M:%S")}')
    torch.use_deterministic_algorithms(False) # Maybe instead CUBLAS_WORKSPACE_CONFIG=:4096:8


    # Load YAML configuration file
    with open(f'{config}', 'r') as file:
        config = yaml.safe_load(file)
    print(vars)

    
    # Check and eventually make output directory where training info will be saved
    pyTrain.recreate_directory(target_path, clean=clean)
    if torch.cuda.is_available():
        device ="cuda"
    else:
        raise RuntimeError("GPU is not available.")

    if data_type == 'domain_adapted':
        model = NNDomainAdapted(features=features, architecture=get_architecture(config), seed=seed, optimizer_kwargs={"lr" : config['learning_rate']},
                            arch_location=os.path.join(repo, "NNarchitectures"), alpha=config['alpha']).to(device)
    else:
        model = NeuralNetwork(features=features, architecture=get_architecture(config), seed=seed, 
                            optimizer_kwargs={"lr" : config['learning_rate']}, arch_location=os.path.join(repo, "NNarchitectures")).to(device)
    print(f"\nThe NN architecture is: \n{model}\n")

    print(f'{num_threads} threads will be used for training', flush=True)

    # if num_threads>1:
    #     if os.path.exists(f"{target_path}/port.temp"):#Remove port file if it remained after previous failed execution
    #         os.remove(f"{target_path}/port.temp")

    #     return_dict = mp.Manager().dict()
    #     return_dict['early_stopping'] = False #Flag to signal early stopping to all processes
    #     train_ds_name = f'train_set{id(train_ds)}'
    #     validation_ds_name = f'validation_set{id(validation_ds)}'
    #     if not isinstance(train_ds, SharedDataset):
    #         train_ds = SharedDataset(train_ds, train_ds_name)
    #         validation_ds = SharedDataset(validation_ds, validation_ds_name)

    #     mp.spawn(pyTrain.train_worker, args=(model, train_ds, 
    #                             validation_ds, target_path, 
    #                             config, seed, return_dict,
    #                             num_threads), nprocs=num_threads)

    #     train_ds.unlink(train_ds_name)
    #     validation_ds.unlink(validation_ds_name)
    #     os.remove(f"{target_path}/port.temp")
    # else:
    #     return_dict = {}
    bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses = pyTrain.train_model_earlyStopping_GPU(model, train_ds, 
                                                                                                                       validation_ds, target_path, 
                                                                                                                       config = config,
                                                                                                                       seed = seed)


    pyTrain.save_model(bestModel, target_path)

    pyTrain.plot_losses(tagger, trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)
    pyTrain.save_losses(trainingEpoch_loss, validationEpoch_loss, bestEpoch, bestLosses, target_path)

    end = datetime.datetime.now()
    print(f'Training ended on {end.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Training time: {end - start}')
    return bestModel



parser = argparse.ArgumentParser(
    description='Train the tagger on the specified decay',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
cfg = parser.parse_args()
cfg.data_type = 'Data'
cfg.tagger = 'SSPion'
cfg.training_data= ['/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_0.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_1.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_2.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_3.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_4.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_5.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_6.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_7.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_8.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_9.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_10.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_11.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_12.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_13.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_14.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_15.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_16.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_17.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_18.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_19.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_20.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_21.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_22.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_23.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_24.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_25.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_26.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_27.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_28.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_29.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_30.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_31.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_32.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_33.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_34.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_35.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_36.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_37.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_38.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_39.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_40.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_41.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_42.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_43.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_44.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_45.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_46.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_47.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_48.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_49.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_50.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_51.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_52.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_53.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_54.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_55.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_56.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_57.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_58.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_59.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_60.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_61.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_62.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_63.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_64.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_65.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_66.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_67.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_68.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_69.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_70.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_71.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_72.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_73.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/train/samples_74.root']
cfg.validation_data=['/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_0.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_1.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_2.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_3.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_4.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_5.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_6.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_7.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_8.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_9.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_10.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_11.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_12.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_13.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_14.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_15.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_16.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_17.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_18.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_19.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_20.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_21.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_22.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_23.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_24.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_25.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_26.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_27.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_28.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_29.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_30.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_31.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_32.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_33.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_34.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_35.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_36.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_37.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_38.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_39.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_40.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_41.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_42.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_43.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_44.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_45.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_46.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_47.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_48.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_49.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_50.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_51.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_52.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_53.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_54.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_55.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_56.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_57.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_58.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_59.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_60.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_61.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_62.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_63.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_64.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_65.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_66.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_67.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_68.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_69.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_70.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_71.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_72.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_73.root', '/ceph-kernel/users/togasa/FlavourTagging/Data/NTuples/5_weighted/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/validation/samples_74.root']
cfg.target_path='/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/config_test/training'
cfg.treename='DecayTree;1'
cfg.tagger='SSPion'
cfg.seed=12
cfg.features='union_PROBNN'
cfg.config='/ceph/users/togasa/classical-taggers/model_configs/config_test.yaml'
cfg.decay_type='Bd2JpsiKst'
cfg.clean=False
cfg.repo='/ceph/users/togasa/classical-taggers'
cfg.data_type='Data'

# cfg.training_data   = cfg.training_data[:2]
# cfg.validation_data = cfg.validation_data[:2]



matplotlib.rcParams.update({'axes.unicode_minus':False,})


BID = 'B_ID' if cfg.data_type == 'Data' else 'B_TRUEID'

features = pyTrain.get_features(tagger=cfg.tagger, yaml_file=cfg.features, repo_path=cfg.repo)

vars = features + ['B_ID','selected', 'label',f"{cfg.tagger}_TagDec"] 
if cfg.data_type == 'domain_adapted':
    vars += ['domain']
elif cfg.data_type == 'MC':
    vars += ['B_TRUEID']

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

train_ds, validation_ds = get_dataSets(train_df=train_df, val_df=val_df, config_name=cfg.config, 
                                       target_path=cfg.target_path, seed=cfg.seed, tagger=cfg.tagger, 
                                       decay_type=cfg.decay_type, features=features, BID=BID)

model = training(train_ds=train_ds, validation_ds=validation_ds, vars=vars, data_type=cfg.data_type, 
                 target_path=cfg.target_path, tagger=cfg.tagger, seed=cfg.seed, features=features, 
                 config=cfg.config, repo=cfg.repo, clean=cfg.clean)

gen_training_plots(model, train_df, val_df, train_ds, validation_ds, cfg.target_path, cfg.tagger)

print(f"Training finished, model saved in {cfg.target_path}")