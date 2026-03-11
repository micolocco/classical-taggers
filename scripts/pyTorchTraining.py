import numpy as np 
import os
import time
import torch
from torch.utils.data import Dataset
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.special import expit
import json
from scripts.shareddataset import SharedDataset
import traceback
# Local imports
from scripts.NNModel import EarlyStopper
from scripts.inputDataset import inputDataset
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import yaml
import sys
import psutil

from torch.distributed import all_gather_object
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import socket
import random
import fcntl
import lhcb_ftcalib as ft



def recreate_directory(target_path, clean=False):
    '''Function to make sure that the ouptut directory exists and it's empty to 
    avoid possible issues with previous versions. The directory presistency can be regulated 
    through the clean parameter that by default is False
    '''
    import shutil
    import os
    if clean:
        # Check if the directory exists
        if os.path.exists(target_path):
            # Remove the entire directory and its contents
            try:
                shutil.rmtree(target_path)
            except Exception as e:
                print(f'Failed to delete {target_path}. Reason: {e}')
    # Create the directory
    try:
        os.makedirs(target_path, exist_ok=True)
        print(f'Output will be saved at {target_path}')
    except Exception as e:
        print(f'Failed to create {target_path}. Reason: {e}')

def get_features(tagger, yaml_file, repo_path):
    '''Function for assigning the input features corresponding to each tagger.
    The input features will be used for the training of the NN
    yaml_file: Configuration file for getting the input features'''
    with open(f'{repo_path}/tagger_inputFeatures/{yaml_file}.yaml', 'r') as file:
        config = yaml.safe_load(file)
        if tagger in config:
            return config[tagger]['features']
        else:
            print(f"Error: Tagger {tagger} not found in configuration.\n Please check {yaml_file} file ")
            return []


def splitByEvent (df, seed, train_val_split):
    '''Function to random split by events (not by index) the dataset into training and test set
    Set seed for reproducibility''' 
    import random
    events_list = np.unique(df.event_entry)
    random.Random(seed).shuffle(events_list)
    n_train_val = int(train_val_split*len(events_list)) # Divide
    n_train = int(0.8 * n_train_val)
    train_df = df[df.event_entry.isin(events_list[:n_train])].copy()
    val_df = df[df.event_entry.isin(events_list[n_train:n_train_val])].copy()
    test_df = df[df.event_entry.isin(events_list[n_train_val:])].copy()
    return train_df, val_df, test_df
    

def prepare_data(train_df, val_df, scalerPath, transformerPath):
    # Load the dataset
    train_dataset = inputDataset(df=train_df) #scaler=PowerTransformer() 
    train_dataset.scale(test=False, scalerPath=scalerPath, transformerPath=transformerPath)
    val_dataset = inputDataset(df=val_df)
    val_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)

    return train_dataset, val_dataset


def plot_features(data, features_list, target_path, name, flag, nbins=100):
    # Plot input features 
    plt.figure(figsize=(24,25))
    num_plots_per_axis = int(np.ceil(np.sqrt(len(features_list))))
    pos=0
    for i, col in enumerate(data.columns.to_list()):
        if col in features_list:
            plt.subplot(num_plots_per_axis, num_plots_per_axis, pos + 1) 
            if col in nice_names.keys():
                range_x=ranges[col]
                xlabel = nice_names[col]
            else:
                range_x=None
                xlabel = col
            
            bins = np.linspace(np.min(data[col]), np.max(data[col]), nbins+1)
            plt.hist(data[col][(data[flag]==0) & (data['B_ID']<0)].to_numpy().astype(float), density = True, bins=bins, label = f"{flag} = 0, B_ID < 0",color='b', alpha=0.5, range=range_x)
            plt.hist(data[col][(data[flag]==0) & (data['B_ID']>0)].to_numpy().astype(float), density = True, bins=bins, label = f"{flag} = 0, B_ID > 0",color='b', histtype='step', alpha=1, range=range_x)
            plt.hist(data[col][(data[flag]==1) & (data['B_ID']<0)].to_numpy().astype(float), density = True, bins=bins, label = f"{flag} = 1, B_ID < 0",color='r', alpha=0.5, range=range_x)
            plt.hist(data[col][(data[flag]==1) & (data['B_ID']>0)].to_numpy().astype(float), density = True, bins=bins, label = f"{flag} = 1, B_ID > 0",color='r', histtype='step', alpha=1, range=range_x)
            plt.xlabel(xlabel)

            plt.legend()
            plt.tight_layout()
            pos+=1
    plt.savefig(f"{target_path}/{name}.pdf")



def find_free_port(path):
    used_ports_path = f'/scratch/{path.split("/")[3]}/used_ports.txt' #Gets the path to the used ports file

    
    for _ in range(10):
        port = random.randint(1024, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Trying port {port}...", flush=True)
            try:
                #Check if port is already in use by own process
                if os.path.exists(used_ports_path):
                    with open(used_ports_path, 'r') as f:
                        fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
                        try:
                            used_ports = f.read().splitlines()
                            if str(port) in used_ports:
                                continue
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)

                # Try to bind the socket to the port
                s.bind(('localhost', port))
            
                # write the port to the file, making sure only one process writes to it at a time
                with open(used_ports_path, 'a') as f:
                    #Lock file
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        #Write the port to the file
                        f.write(f"\n{port}")
                    finally:
                        #Release file
                        fcntl.flock(f, fcntl.LOCK_UN)
                        return port
            except OSError:
                print(traceback.format_exc(), flush=True)
                continue
    raise RuntimeError("Could not find a free port")

def release_port(path, port):
    used_ports_path = f'/scratch/{path.split("/")[3]}/used_ports.txt' #Gets the path to the used ports file
    with open(used_ports_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
        try:
            used_ports = f.read().splitlines()
            if str(port) in used_ports:
                used_ports.remove(str(port))
                f.seek(0)
                f.truncate()  # Clear the file
                f.write('\n'.join(used_ports))  # Write back the remaining ports
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock

def ddp_setup(rank, world_size, target_path): #Create a way for the processes to communicate with each other
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    print('Setting up DDP for rank:', rank, 'of', world_size, flush=True)

    if rank == 0:
        #Get free port and write it to disk for the other processes to read
        socket_port = find_free_port(target_path)
        with open(f"{target_path}/port.temp", "w") as f:
            f.write(str(socket_port))
    else:
        #Read the port from disk
        while not os.path.exists(f"{target_path}/port.temp"):
            time.sleep(0.1)
        time.sleep(0.5)  # Ensure the file is fully written before reading
        with open(f"{target_path}/port.temp", "r") as f:
            socket_port = int(f.read().strip())


    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{socket_port}"    
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized with port {socket_port}", flush=True)
    #above needs to be run before distributed sampler or DistributedDataParallel is created. 'gloo' is needed for CPU training. Rank is a unique 
    #identifier for each process, and world_size is the total number of processes. 
    return socket_port

def report_memory_distributed():
    local_mem = torch.tensor([psutil.Process(os.getpid()).memory_info().rss / 1024**2], dtype=torch.float32)
    dist.all_reduce(local_mem, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        print(f"Total RAM used by all Processes: {local_mem.item():.2f} MiB")

def format_loss(loss): #such that a float as well as a array (case of domain adaptation) can be handled
    if isinstance(loss, np.float64):
        formatted_loss = f"{loss:.6f}"
    else:
        formatted_loss = f'{np.round(loss[0], 6)}, {np.round(loss[1], 6)}'

    return formatted_loss

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model_EarlyStopping(rank, model, train_ds, validation_ds, target_path, config, seed, return_dict, num_threads=1):
        lossValBest = 10000
        stopped = False
        bestEpoch = 0

               

        training_start = time.time()
        early_stopper = EarlyStopper(patience=config['patience'], min_delta=config['min_delta'])
        
        i = 1

        if num_threads>1:
            port = ddp_setup(rank, num_threads, target_path) 
            ddpmodel = DDP(model)

            module = ddpmodel.module
            train_sampler = DistributedSampler(train_ds, num_replicas=num_threads, rank=rank, shuffle=True, drop_last=True, seed=seed)
            validation_sampler = DistributedSampler(validation_ds, num_replicas=num_threads, rank=rank, shuffle=False, drop_last=True, seed=seed)
            shuffle=False
            shuffle=False #Shuffling the dataloader and setting a sampler is mutually exclusive
        else:
            module = model
            train_sampler = None
            validation_sampler = None
            shuffle=True
            shuffle=True

        
        train_batch_size = config['train_batch_size']//num_threads #Ensures same effective batch size regardless of number of threads
        val_batch_size = config['train_batch_size']//num_threads #Might want to change this to a seperate hyperparameter in the config file
        
        g = torch.Generator()
        g.manual_seed(seed)
        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=shuffle, sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
        validation_dl = DataLoader(validation_ds, batch_size=val_batch_size, shuffle=shuffle, sampler=validation_sampler, worker_init_fn=seed_worker, generator=g)
        
        trainingEpoch_loss = []
        validationEpoch_loss = []
        temp = module.validate_model(validation_dl)
        initialValidation_loss = np.array(temp).mean(axis=0)
        if rank == 0: #Only print on rank 0, to avoid duplicate printing in multi-threading
            print(f"The initial Validation Loss: {format_loss(initialValidation_loss)}")
        
        epochtimes = []
        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            if rank == 0:
                print(f"--------------Epoch:{epoch+1}/{config['n_epochs']}--------------")
            # Train over mini-batches
            stepLoss = np.array(module.train_model(train_dl)).mean(axis=0)
            trainingEpoch_loss.append(stepLoss)
            # Compute validation loss
            validationStep_loss = np.array(module.validate_model(validation_dl)).mean(axis=0)
            validationEpoch_loss.append(validationStep_loss)
            if rank == 0:
                print(f"Train:{format_loss(stepLoss)}, Validation:{format_loss(validationStep_loss)}, Time:{round((time.time()-epoch_start) ,2)}s, Early stopping counter: {early_stopper.counter}/{config['patience']}", flush=True)
                print(f'Estimated total RAM used: {psutil.Process(os.getpid()).memory_info().rss*num_threads / 1024 ** 2} MiB', flush=True)

            epochtimes.append((time.time()-epoch_start))

            if early_stopper.early_stop(validationEpoch_loss[-1]): #Ensure that early stopping is not triggered too early
                stopped = True 
                break
            if early_stopper.counter == 0:
                lossValBest = validationEpoch_loss[-1]
                lossTrainBest = trainingEpoch_loss[-1]
                bestEpoch = epoch
                bestModel = copy.deepcopy(module)
            i +=1
        if num_threads>1:
            destroy_process_group()
            release_port(target_path, port)
    

        #Getting around not being able to return from the DDP process
        if rank == 0:
            training_time = round((time.time()- training_start) / 60 , 2)
            #calculate standard deviation of epoch times
            epochtimes_mean = np.mean(epochtimes)
            epochtimes_std = np.std(epochtimes)

            print(f'Average time per epoch: {epochtimes_mean} +/- {epochtimes_std} seconds')
            print(f"Training finished in {training_time} min, {i-1} epochs, early stopping: {stopped}")

            return_dict['bestModel'] = bestModel
            return_dict['trainingEpoch_loss'] = trainingEpoch_loss
            return_dict['validationEpoch_loss'] = validationEpoch_loss
            return_dict['bestEpoch'] = bestEpoch
            return_dict['bestLosses'] = np.array([lossTrainBest, lossValBest], dtype=float)


def eval_model_multiprocessed(rank, model, ds, target_path, return_dict, num_threads=4):
    if num_threads == 1:
        raise Exception("This function is meant to be run in a multi-threaded environment. Use evaluate_model of model directly instead.")
    port = ddp_setup(rank, num_threads, target_path) 
    ddpmodel = DDP(model)
    module = ddpmodel.module

    sampler = DistributedSampler(ds, num_replicas=num_threads, rank=rank, drop_last=False)


    dl = DataLoader(ds, batch_size=1024, shuffle=False, sampler=sampler, drop_last=False)

    predictions, actuals = module.evaluate_model(dl)

    
    # Gather lists from all ranks
    all_preds, all_actuals = [None for _ in range(num_threads)], [None for _ in range(num_threads)]
    all_gather_object(all_preds, predictions)
    all_gather_object(all_actuals, actuals)

    # Only rank 0 merges them
    if rank == 0:
        predictions = np.concatenate(all_preds)[:len(ds)] #Remove padding
        actuals = np.concatenate(all_actuals)[:len(ds)]
        return_dict['predictions'] = predictions
        return_dict['actuals'] = actuals

    torch.distributed.barrier()
    destroy_process_group()
    release_port(target_path, port)
    


        
def save_model(model, target_path, filename = 'model.pth'):
    # target_path = name_formatter.assign_name(folder, target_path)
    torch.save(copy.deepcopy(model.state_dict()), f"{target_path}/{filename}")



def load_model(model, target_path):
   # target_path = name_formatter.assign_name(folder, target_path)    
    # saveName = name_formatter.assign_name(target_path, model.modelName)
    model_name = f"{target_path}/model.pth"
    model.load_state_dict(torch.load(model_name))
    print(f"The model used is {model_name}")

def load_model_without_domain_classifier(model, target_path):
    model_name = f"{target_path}/model.pth"

    full_dict = torch.load(model_name)

    class_dict = {key: value for key, value in full_dict.items() if 'domain_classifier' not in key}

    i = 0

    num_feature_extractor_layers = len([k for k in class_dict if 'feature_extract' in k])//2 #includes weights and bias
    num_class_classifier_layers = len([k for k in class_dict if 'class_classifier' in k])//2 #includes weights and bias

    for j in range(num_feature_extractor_layers):
        for p in ['weight', 'bias']:
            k_old = f"feature_extract.{j*3}.{p}"
            k_new = f"NN.{i}.{p}"
            class_dict[k_new] = class_dict.pop(k_old)

        i += 3

    for j in range(num_class_classifier_layers):
        for p in ['weight', 'bias']:
            k_old = f"class_classifier.{j*3}.{p}"
            k_new = f"NN.{i}.{p}"
            class_dict[k_new] = class_dict.pop(k_old)

        i += 3


    model.load_state_dict(class_dict)
    
    print(f"The model used is {model_name}")




def save_losses(trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    folder = f'{target_path}/losses'
    os.makedirs(f'{folder}', exist_ok=True)
    np.savetxt(f"{folder}/test.csv", valLoss, delimiter=",")
    np.savetxt(f"{folder}/train.csv", trainLoss, delimiter=",")
    if isinstance(bestLosses[0], np.float64):
        np.savetxt(f"{folder}/best.csv", [bestEpoch ,bestLosses[0],bestLosses[1]], delimiter=",")
    else:
        np.savetxt(f"{folder}/best.csv", [[bestEpoch, 0],bestLosses[0],bestLosses[1]], delimiter=",") # 0 only for formatting purposes



def plot_losses(tagger, trainLoss, valLoss, bestEpoch, bestLosses, target_path, filename = 'Loss'):
    trainLoss = np.array(trainLoss)
    valLoss = np.array(valLoss)


    if not trainLoss.ndim == 2:
        # Loss Plot
        plt.figure(figsize=(8,8))
        plt.plot(trainLoss, label='Training', c='orange')
        plt.plot(valLoss, label='Validation', c='blue')
        plt.axvline(bestEpoch, linestyle='--', color='tab:gray', label="Best epoch")
        plt.legend(loc="best")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title(f"{tagger}", fontsize=24)
        plt.savefig(f"{target_path}/{filename}.pdf")
        plt.close()
    else:
        # Subplots of class and domain loss
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

        loss_labels = ['Class Loss', 'Domain Loss']
        colors = ['orange', 'blue']

        for i in range(2):
            axs[i].plot(trainLoss[:,i], label='Training', c=colors[0])
            axs[i].plot(valLoss[:,i], label='Validation', c=colors[1])
            axs[i].axvline(bestEpoch, linestyle='--', color='tab:gray', label="Best epoch")
            axs[i].set_ylabel(loss_labels[i])
            axs[i].legend(loc="best")
            axs[i].set_title(f"{loss_labels[i]} over Epochs")

        axs[1].set_xlabel("Epoch")
        fig.suptitle(f"{tagger}", fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{target_path}/{filename}.pdf")
        plt.close()
   
def plot_ROC(tagger, val_df, target_path, train_df= None, trueLabel= 'yTrue', predLabel='yPred', fileLabel=''):
    
    plt.figure()
    lw  = 2
    fpr_test, tpr_test,_ = roc_curve(val_df[trueLabel], val_df[predLabel])
    roc_auc_test = round(auc(fpr_test, tpr_test),5)

    if train_df is not None: 
        label = f'Validation (area = {roc_auc_test})'
    else:
        label = f'Test (area = {roc_auc_test})'
    
    plt.plot(fpr_test, tpr_test, color='darkblue',lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f"{tagger}", fontsize=24)
    if train_df is not None:
        fpr, tpr,_ = roc_curve(train_df[trueLabel], train_df[predLabel])
        roc_auc = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'Train (area = {roc_auc})' )
        plt.legend(loc="lower right")
        plt.savefig(f"{target_path}/{fileLabel}ROC_TRAIN_VAL.pdf")
    else:
        plt.savefig(f"{target_path}/{fileLabel}ROC_TEST.pdf")


def plot_mistag(tagger, df, target_path, type, show_trueB=False, clf = None, nbins=100, trueLabel= 'yTrue', 
                predLabel='yPred', fileLabel='', correct_legend= "wrong tagging decision", wrong_legend= "correct tagging decision"):
    plt.figure()
    # plt.title("Mistag rate")
    plt.yscale("log")
    min_pred = np.min(1-df[predLabel])
    max_pred = np.max(1-df[predLabel])
    bins = np.linspace(min_pred, max_pred, nbins+1)

    if show_trueB:
        plt.hist(1-df[(df[trueLabel]==0)&(df['B_ID']<0)][predLabel], bins = bins, density = True, histtype="stepfilled", color = "skyblue", alpha = 0.5, label = f"true l=0, B")
        plt.hist(1-df[(df[trueLabel]==0)&(df['B_ID']>0)][predLabel], bins = bins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"true l=0, antiB")
        plt.hist(1-df[(df[trueLabel]==1)&(df['B_ID']<0)][predLabel], bins = bins, density = True, histtype="stepfilled", color = "salmon", alpha = 0.5, label = f"true l=1, B")
        plt.hist(1-df[(df[trueLabel]==1)&(df['B_ID']>0)][predLabel], bins = bins, density = True, histtype="stepfilled", color = "red", alpha = 0.5, label = f"true l=1, antiB")
    else:
        plt.hist(1-df[df[trueLabel] == 0][predLabel],bins = bins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = correct_legend)
        plt.hist(1-df[df[trueLabel] == 1][predLabel],bins = bins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = wrong_legend)
    
    data1=1-df[(df[trueLabel]==0)&(df['B_ID']<0)][predLabel]  
    data2=1-df[(df[trueLabel]==0)&(df['B_ID']>0)][predLabel]   
    data3=1-df[(df[trueLabel]==1)&(df['B_ID']<0)][predLabel]  
    data4=1-df[(df[trueLabel]==1)&(df['B_ID']>0)][predLabel]
    plt.title(f"{tagger}", fontsize=24)
    plt.xlabel(r"1 - NN output", fontsize=24)
    #plt.annotate(f'{len(df.yPred)} tracks', xy=(0, 1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.grid()
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title=f'{type}:{len(df[predLabel])} total tracks')
    plt.savefig(f"{target_path}/{fileLabel}NNoutput_{type}.pdf")

    plt.clf()
    plt.title(f'{type}: {tagger}', fontsize=24)
    plt.hist(data3, bins=bins, color='salmon',  alpha=0.4, density = True, histtype="bar",  label = f'true l=1, B')  
    plt.hist(data4, bins=bins, color='red',     alpha=1,   density = True, histtype="step", label = f'true l=1, antiB',)
    plt.hist(data1, bins=bins, color='skyblue', alpha=0.4, density = True, histtype="bar",  label = f'true l=0, B',)
    plt.hist(data2, bins=bins, color='blue',    alpha=1,   density = True, histtype="step", label = f'true l=0, antiB')
    plt.legend()
    plt.yscale('log')
    plt.xlabel(f'1 - NN output', fontsize=22)
    plt.ylabel('Normalized number of tracks', fontsize=22)

    plt.tight_layout()
    plt.savefig(f"{target_path}/{fileLabel}NNoutput_{type}_byTRUEID.pdf")

    
def plot_tagDec(tagger, df_TagParticles, plot_name='Normalized_TagDec.pdf',nbins=100):
    # Get the particle with the lowest mistag for each event
    plt.figure()
    plt.yscale("log")
    plt.hist(df_TagParticles.loc[(df_TagParticles[f"{tagger}_TagDec"] == -1)][f"{tagger}_Eta"] ,bins = 100 , density = True , histtype = "stepfilled" ,range=(df_TagParticles[f"{tagger}_Eta"].min(),df_TagParticles[f"{tagger}_Eta"].max()), color = "green" , alpha=0.5, label = f"Tag. dec: b")
    plt.hist(df_TagParticles.loc[(df_TagParticles[f"{tagger}_TagDec"] == 1)][f"{tagger}_Eta"] ,bins = 100 , density = True , histtype = "stepfilled" ,range=(df_TagParticles[f"{tagger}_Eta"].min(),df_TagParticles[f"{tagger}_Eta"].max()), color = "orange" , alpha=0.5, label = f"Tag. dec: anti-b")
    plt.grid()
    plt.xlabel(r"$\eta$",fontsize=24)
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title = f'{len(df_TagParticles[(df_TagParticles[f"{tagger}_TagDec"] == -1)|(df_TagParticles[f"{tagger}_TagDec"] == 1)])} tagged events')
    plt.title(f"{tagger} mistag", fontsize=24)
    print(f'Tagging decision plot saved at {plot_name}')
    plt.savefig(f"{plot_name}")
    plt.close()

def bins_by_yield(etas, weights, nbins):
    if weights is None:
        weights = np.ones_like(etas)

    # Sort etas and weights by eta
    sorted_indices = np.argsort(etas)
    sorted_etas = etas[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative sum of weights
    cum_weights = np.cumsum(sorted_weights)

    # Total yield and target yield per bin
    total_weight = cum_weights[-1]
    target_yields = np.linspace(0, total_weight, nbins + 1)

    # Interpolate to find the bin edges in eta space
    bin_edges = np.interp(target_yields, cum_weights, sorted_etas)

    return bin_edges

def calibration(tagger, df_tag, eventType, target_path, calibration_option='mistag',nbins = 7, weights = None, mode='Bu'):

    #Calibration of the taggers and parameters saving

    taggers = ft.TaggerCollection()
       
    if weights is None:
        weights = np.ones(len(df_tag))
    
    
    if mode != 'Bu':
        tau_ps = df_tag['B_DTF_PV_Jpsi_TAU'].to_numpy()
    else:
        tau_ps = None
    
    taggers.create_tagger(name = tagger, 
                          eta_data = df_tag[f"{tagger}_Eta"].to_numpy(), 
                          dec_data = df_tag[f"{tagger}_TagDec"].to_numpy(), 
                          weight = weights, 
                          B_ID = df_tag['B_ID'].to_numpy(),
                          mode = mode , 
                          tau_ps = tau_ps)

    npar = 2
    if calibration_option=='logit':
        taggers.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.logit))
    elif calibration_option=='mistag':
        taggers.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.mistag)) 
    elif calibration_option=='rlogit':
        taggers.set_calibration(ft.PolynomialCalibration(npar=npar, link=ft.link.rlogit))

        print('Not a valid calibration function')
    taggers.retry_on_error(use_link_alternative=ft.link.logit) # use logit link function if minimization did not converge the first time

    target_path = f'{target_path}/{calibration_option}'
    if os.path.isdir(f'{target_path}') == False:
        os.system(f"mkdir {target_path}")

    taggers.calibrate()


    scale = "linear"
    if weights is not None:
        #distribute the bins such that each bin has the same yield, aka the same sum of weights
        bins = bins_by_yield(df_tag[f"{tagger}_Eta"].values, weights, nbins)
        print(bins)

        if any(bins[:-1] == bins[1:]):
            print("Warning: Bins are not unique, using linspace instead.")
            bins = np.linspace(df_tag[f"{tagger}_Eta"].min(), df_tag[f"{tagger}_Eta"].max(), nbins+1)

        class_indices = df_tag['B_ID'].values
        class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: r'$\bar{B}^0$', }



        taggers.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                                file_name = 'split_calibration_curves.pdf', savepath = f'{target_path}', omega_range="minimal", 
                                                nbins = nbins, x_scale = scale, y_scale = scale)#, share_y= True, share_x = True)

        taggers.plot_calibration_curves(savepath = f'{target_path}', omega_range="minimal", nbins = nbins, x_scale = scale, y_scale = scale)


    info_dict = {"TaggingEfficiency"     : taggers[tagger].stats.tagging_efficiency(calibrated = False),
                "TaggingPower"           : taggers[tagger].stats.tagging_power(     calibrated = False),
                "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True ), 
                "TaggingPower_Cali"      : taggers[tagger].stats.tagging_power(     calibrated = True ),
                "EffectiveMistag_Cali"   : taggers[tagger].stats.effective_mistag(  calibrated = True ), 
                "EffectiveMistag"        : taggers[tagger].stats.effective_mistag(  calibrated = False),}
    for i in range(npar):
        info_dict[f"Fitpar_p{i}"]      = [taggers[tagger].stats.params.params_delta[i],      taggers[tagger].stats.params.errors_delta[i]]
        info_dict[f"Fitpar_deltap{i}"] = [taggers[tagger].stats.params.params_delta[i+npar], taggers[tagger].stats.params.errors_delta[i+npar]]


    with open(f"{target_path}/taggingInfo_{calibration_option}.json", "w") as f:
        json.dump(info_dict, f)
    print(f"Tagger parameters saved at {target_path}\n")
    print(f"Tagging information in a presentation-friendly format:\n")
    # Process the data
    processed_data = {key: propagate_and_round(value, 'Fitpar' not in key) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
    formatted_data = {key: f"{values[0]} +- {values[1]}" if len(values) > 1 else values[0] for key, values in processed_data.items()}
    # Print the formatted data
    for key, value in formatted_data.items():
        print(f"{key}: {value}")

    return info_dict

# Function to propagate and round the errors and values
def propagate_and_round(values, is_percentage=False):
    values = np.array(values) * 100 if is_percentage else np.array(values)

    try:
        if len(values) > 2:  # For TaggingPower_Cali and EffectiveMistag_Cali
            combined_error = np.sqrt(np.sum(np.square(values[1:])))
            rounded_error = round(combined_error, -int(np.floor(np.log10(combined_error))))
            
            significant_digit = int(np.floor(np.log10(rounded_error)))
            rounded_value = round(values[0], -significant_digit)
            return [rounded_value, rounded_error]
        else:  # For other data
            max_error = max(values[1:])
            rounded_errors = [round(err, -int(np.floor(np.log10(max_error)))) for err in values[1:]]
            significant_digit = int(np.floor(np.log10(max_error)))
            rounded_value = round(values[0], -significant_digit)
            
            return [rounded_value] + rounded_errors
    except Exception as e:
        print(f"Error in processing values: {values}. Error: {e}")
        values = [str(v) for v in values]
        values[-1] = f'{values[-1]} Error in Rounding'
        return values  # Fallback for unexpected cases

def print_taggingInfo(tag_file='taggingInfo.json'):
    # Read the data from the JSON file
    import json
    with open(tag_file, 'r') as file:
        info_dict = json.load(file)
    processed_data = {key: propagate_and_round(value) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" for key, values in processed_data.items()}
    # Print the formatted data
    for key, value in formatted_data.items():
        print(f"{key}: {value}")


