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
import scripts.pipeline
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

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import socket
import random
import fcntl


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

# def apply_log(df, features):
#     log_features = ['Column1']
#     for feature in features:
#         if 'log' in feature:
#             df.eval(f'log({feature}) = log({feature})', inplace = True)
#     return features


def splitByEvent (df, seed, train_val_split):
    '''Function to random split by events (not by index) the dataset into training and test set
    Use random.Random(2) to reproduce same shuffling''' 
    import random
    events_list = np.unique(df.event_entry)
    random.Random(seed).shuffle(events_list)
    n_train_val = int(train_val_split*len(events_list)) # Divide
    n_train = int(0.8 * n_train_val)
    train_df = df[df.event_entry.isin(events_list[:n_train])].copy()
    val_df = df[df.event_entry.isin(events_list[n_train:n_train_val])].copy()
    test_df = df[df.event_entry.isin(events_list[n_train_val:])].copy()
    return train_df.query('selected==1'), val_df.query('selected==1'), test_df
    

def prepare_data(train_df, val_df, seed, scalerPath, transformerPath, indexed = True): #, train_batch_size, test_batch_size = 1024, distributed = False
    # Load the dataset
    train_dataset = inputDataset(df=train_df, indexed=indexed) #scaler=PowerTransformer() 
    train_dataset.scale(test=False, scalerPath=scalerPath, transformerPath=transformerPath)
    val_dataset = inputDataset(df=val_df, indexed=indexed)
    val_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    # Prepare data loaders
    #torch.manual_seed(seed) # to ensure reproducibility

    # if distributed:
    #     train_sampler=DistributedSampler(train_dataset)
    #     val_sampler=DistributedSampler(val_dataset)
    # else:
    #     train_sampler=None
    #     val_sampler=None

    # train_dl = DataLoader(train_dataset, batch_size = train_batch_size, shuffle=False, sampler=train_sampler)
    # validation_dl = DataLoader(val_dataset, batch_size = test_batch_size, shuffle=False, sampler=val_sampler)
    # return train_dl, validation_dl 
    return train_dataset, val_dataset


def plot_features(data, features_list, target_path, name, flag, nbins=100):
    # Plot input features 
    plt.figure(figsize=(24,25))
    pos=0
    for i, col in enumerate(data.columns.to_list()):
        if col in features_list:
            plt.subplot(6, 6 , pos + 1) # hardcoded according to the number of features
            if col in nice_names.keys():
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', alpha=0.5, range=ranges[col])
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', alpha=0.5, range=ranges[col])
                plt.xlabel(nice_names[col])
            else:
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', alpha=0.5, )
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', alpha=0.5, )
                plt.xlabel(col)

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
                        f.write(f"{port}\n")
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
    # print(type(loss))
    if isinstance(loss, np.float64):
        formatted_loss = f"{loss:.6f}"
    else:
        formatted_loss = f'{np.round(loss[0], 6)}, {np.round(loss[1], 6)}'

    return formatted_loss

def train_model_EarlyStopping(rank, model, train_ds, validation_ds, target_path, config, return_dict, train_weights = None, 
                              val_weights= None, num_threads=1):
        lossValBest = 10000
        stopped = False
        bestEpoch = 0

               

        training_start = time.time()
        early_stopper = EarlyStopper(patience=config['patience'], min_delta=config['min_delta'])
        
        i = 1
        if train_weights is not None:
            train_weights = torch.from_numpy(train_weights)
        if val_weights is not None:
            val_weights = torch.from_numpy(val_weights)

        if num_threads>1:
            port = ddp_setup(rank, num_threads, target_path) 
            ddpmodel = DDP(model)

            module = ddpmodel.module
            train_sampler = DistributedSampler(train_ds, num_replicas=num_threads, rank=rank)
            validation_sampler = DistributedSampler(validation_ds, num_replicas=num_threads, rank=rank)
        else:
            module = model
            train_sampler = None
            validation_sampler = None
        
        train_batch_size = config['train_batch_size']//num_threads #Ensures same effective batch size regardless of number of threads
        val_batch_size = config['train_batch_size']//num_threads #Might want to change this to a seperate hyperparameter in the config file
        
        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=False, sampler=train_sampler)
        validation_dl = DataLoader(validation_ds, batch_size=val_batch_size, shuffle=False, sampler=validation_sampler)
        
        trainingEpoch_loss = []
        validationEpoch_loss = []
        initialValidation_loss = np.array(module.validate_model(validation_dl, sample_weights=val_weights)).mean(axis=0)
        if rank == 0: #Only print on rank 0, to avoid duplicate printing in multi-threading
            print(f"The initial Validation Loss: {format_loss(initialValidation_loss)}")
        
            epochtimes = []
            for epoch in range(config['n_epochs']):
                epoch_start = time.time()
                if num_threads>1:
                    train_dl.sampler.set_epoch(epoch)
                    validation_dl.sampler.set_epoch(epoch)
                if rank == 0:
                    print(f"--------------Epoch:{epoch+1}/{config['n_epochs']}--------------")
                # Train over mini-batches
                stepLoss = np.array(module.train_model(train_dl, epoch, config['n_epochs'], sample_weights=train_weights)).mean(axis=0)
                trainingEpoch_loss.append(stepLoss)
                # Compute validation loss
                validationStep_loss = np.array(module.validate_model(validation_dl, sample_weights=val_weights)).mean(axis=0)
                validationEpoch_loss.append(validationStep_loss)
                if rank == 0:
                    print(f"Train:{format_loss(stepLoss)}, Validation:{format_loss(validationStep_loss)}, Time:{round((time.time()-epoch_start) ,2)}s, Early stopping counter: {early_stopper.counter}/{config['patience']}", flush=True)
                if num_threads==1:
                    print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB', flush=True)
                else:
                    try:
                       report_memory_distributed() #To print the memory usage of all processes
                    except Exception as e:
                        print(f'Memory report failed in rank {rank}')
                epochtimes.append((time.time()-epoch_start))

                if early_stopper.early_stop(validationEpoch_loss[-1]): #Ensure that early stopping is not triggered too early
                    stopped = True 
                    break
                if early_stopper.counter == 0:
                    lossValBest = validationEpoch_loss[-1]
                    lossTrainBest = trainingEpoch_loss[-1]
                    bestEpoch = epoch
                    # save_model(model, target_path)
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

        
        
def save_model(model, target_path, filename = 'model.pth'):
    # target_path = name_formatter.assign_name(folder, target_path)
    torch.save(copy.deepcopy(model.state_dict()), f"{target_path}/{filename}")
    #save_hyperparameters(model, target_path)

def save_hyperparameters(model, target_path):
    info_dict = {
                'ModelName:' : model.modelName, 
                'learning_rate' : config.learning_rate,
                'patience' : config.patience,
                'min_delta' : config.min_delta,
                'activation_function' : config.activation_function,
                'n_epochs' : config.n_epochs,
                'train_val_split' : config.train_val_split,
                'train_batch_size': config.train_batch_size,
                }
    with open(f"{target_path}/hyperparameters.json", "w") as f:
        json.dump(info_dict, f)


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



def plot_losses(tagger, trainLoss, valLoss, bestEpoch, bestLosses, target_path):
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
        plt.savefig(f"{target_path}/Loss.pdf")
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
        plt.savefig(f"{target_path}/Loss.pdf")
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

def logistic_regression(df, target_path):
    
    clf = LogisticRegression().fit(df['yPred'], df['yTrue'].ravel())  
    pickle.dump(clf , open(f"{target_path}/LogReg.pck" , "wb"))
    return clf
'''
def plot_NNoutput_mistag (name, clf, yPredTest, yTrueTest, df['yPred'], df['yTrue'], target_path, nbins=100):
    
    plt.figure()
    LR_test = np.linspace(0, 1, 300)
    loss = expit(LR_test * clf.coef_ + clf.intercept_)
    plt.title("Logistic Regression")
    plt.grid()
    plt.plot(yPredTest[yTrueTest == 0][0:500], np.zeros(500) , "b.",alpha = 0.5, label = "Label = 0")
    plt.plot(yPredTest[yTrueTest == 1][0:500], np.ones(500) ,  "r.",alpha = 0.5, label = "Label = 1")
    plt.plot(LR_test, loss ,color = "k")
    plt.legend(loc = "best")
   # saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/LogReg.pdf")
    
    prob_train_0_height , prob_train_0_bin_edges= np.histogram(df['yPred'][df['yTrue'] == 0] , bins = nbins, density = True)
    prob_train_0_bin_edges = prob_train_0_bin_edges[:len(prob_train_0_bin_edges)-1]+ (prob_train_0_bin_edges[1]-prob_train_0_bin_edges[0])/2
    prob_train_1_height , prob_train_1_bin_edges= np.histogram(df['yPred'][df['yTrue'] == 1] ,bins = nbins, density = True)
    prob_train_1_bin_edges = prob_train_1_bin_edges[:len(prob_train_1_bin_edges)-1]+ (prob_train_1_bin_edges[1]-prob_train_1_bin_edges[0])/2

    y_test_predict_LR = clf.predict_proba(yPredTest)[:,0]
    y_train_predict_LR = clf.predict_proba(df['yPred'])[:,0]

    prob_train_0_height_LR , prob_train_0_bin_edges_LR= np.histogram(y_train_predict_LR[df['yTrue'] == 0], bins = nbins, density = True)
    prob_train_0_bin_edges_LR = prob_train_0_bin_edges_LR[:len(prob_train_0_bin_edges_LR)-1]+ (prob_train_0_bin_edges_LR[1]-prob_train_0_bin_edges_LR[0])/2
    prob_train_1_height_LR , prob_train_1_bin_edges_LR= np.histogram(y_train_predict_LR[df['yTrue'] == 1], bins = nbins, density = True)
    prob_train_1_bin_edges_LR = prob_train_1_bin_edges_LR[:len(prob_train_1_bin_edges_LR)-1]+ (prob_train_1_bin_edges_LR[1]-prob_train_1_bin_edges_LR[0])/2
    
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].set_title("NN Output")
    axs[0].set_yscale("log")
    axs[0].hist(yPredTest[yTrueTest == 0],bins = nbins, density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[0].hist(yPredTest[yTrueTest == 1],bins = nbins, density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[0].plot(prob_train_0_bin_edges, prob_train_0_height, "b.", label = "Train (Label = 0)")
    axs[0].plot(prob_train_1_bin_edges, prob_train_1_height, "r.", label = "Train (Label = 1)")
    axs[0].grid()
    axs[0].set_xlabel(r"NN output")
    axs[0].set_ylabel("Normalized number of tracks")
    axs[0].legend(loc = "best")
    axs[1].set_title("LogReg Output")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Normalized number of tracks")
    axs[1].set_xlabel(r"Logistic(NN ouput)")
    axs[1].hist(y_test_predict_LR[yTrueTest == 0],bins = nbins,density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[1].hist(y_test_predict_LR[yTrueTest == 1],bins = nbins,density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[1].plot(prob_train_0_bin_edges_LR ,prob_train_0_height_LR, "b.", label = "Train (Label = 0)")
    axs[1].plot(prob_train_1_bin_edges_LR ,prob_train_1_height_LR, "r.", label = "Train (Label = 1)")
    axs[1].grid()
    axs[1].legend(loc = "best")

    #folder = 'plots'
    #saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/NNoutput_sigmoid.pdf")
    plt.close()
'''

def plot_mistag(tagger, df, target_path, type, show_trueB=False, clf = None, nbins=100, BID = 'B_TrueID', trueLabel= 'yTrue', 
                predLabel='yPred', fileLabel='', correct_legend= "wrong tagging decision", wrong_legend= "correct tagging decision"):
    plt.figure()
    # plt.title("Mistag rate")
    plt.yscale("log")

    if clf:
        y_predict_LR = clf.predict_proba(df[predLabel])[:,0]
        plt.hist(y_predict_LR[df[trueLabel] == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = correct_legend)
        plt.hist(y_predict_LR[df[trueLabel] == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = wrong_legend)
        plt.title(f'{tagger} mistag after Logistic Regression', fontsize=24)
    else:
        if show_trueB:
            plt.hist(1-df[(df[trueLabel]==0)&(df[BID]==-521)][predLabel], bins = nbins, density = True, histtype="stepfilled", color = "skyblue", alpha = 0.5, label = f"true l=0, B")
            plt.hist(1-df[(df[trueLabel]==0)&(df[BID]==521)][predLabel], bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"true l=0, antiB")
            plt.hist(1-df[(df[trueLabel]==1)&(df[BID]==-521)][predLabel], bins = nbins, density = True, histtype="stepfilled", color = "salmon", alpha = 0.5, label = f"true l=1, B")
            plt.hist(1-df[(df[trueLabel]==1)&(df[BID]==521)][predLabel], bins = nbins, density = True, histtype="stepfilled", color = "red", alpha = 0.5, label = f"true l=1, antiB")
   
        else:
            plt.hist(1-df[df[trueLabel] == 0][predLabel],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = correct_legend)
            plt.hist(1-df[df[trueLabel] == 1][predLabel],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = wrong_legend)
    data1=1-df[(df[trueLabel]==0)&(df[BID]==-521)][predLabel]  
    data2=1-df[(df[trueLabel]==0)&(df[BID]==521)][predLabel]   
    data3=1-df[(df[trueLabel]==1)&(df[BID]==-521)][predLabel]  
    data4=1-df[(df[trueLabel]==1)&(df[BID]==521)][predLabel]
    plt.title(f"{tagger}", fontsize=24)
    plt.xlabel(r"1 - NN output", fontsize=24)
    #plt.annotate(f'{len(df.yPred)} tracks', xy=(0, 1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.grid()
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title=f'{type}:{len(df[predLabel])} total tracks')
    plt.savefig(f"{target_path}/{fileLabel}NNoutput_{type}.pdf")
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle(f'{type}: {tagger}', fontsize=24)
    axs[0].hist(data1, bins=nbins, color='skyblue', density = True, histtype="stepfilled", label = f'true l=0, B',)
    axs[0].hist(data2, bins=nbins, color='blue', density = True, histtype="step",label=f'true l=0, antiB')
    axs[0].legend()
    axs[1].hist(data3, bins=nbins, color='salmon', density = True, histtype="stepfilled",label=f'true l=1, B')  
    axs[1].hist(data4, bins=nbins, color='red', density = True, histtype="step",label=f'true l=1, antiB',)
    axs[1].legend()
    for ax in axs.flat:
        ax.set_yscale('log')
        ax.set_xlabel(f'1 - NN output', fontsize=22)
        ax.set_ylabel('Normalized number of tracks', fontsize=22)

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

def calibration(tagger, df_tag, eventType, target_path, calibration_option='mistag', BID = 'B_TrueID',nbins = 7, weights = None):

    #Calibration of the taggers and parameters saving
    import lhcb_ftcalib as ft

    taggers = ft.TaggerCollection()
       
    if weights is None:
        weights = np.ones(len(df_tag))
    
    
    taggers.create_tagger(name = tagger, eta_data = df_tag[f"{tagger}_Eta"].tolist(), dec_data = df_tag[f"{tagger}_TagDec"].tolist(), weight = weights, B_ID = df_tag[BID].tolist(),mode = eventType[:2] )

    if calibration_option=='logit':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    elif calibration_option=='mistag':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.mistag)) 
    else:
        print('Not a valid calibration function')
    taggers.retry_on_error(use_link_alternative=ft.link.logit) # use logit link function if minimization did not converge the first time

    target_path = f'{target_path}/{calibration_option}'
    if os.path.isdir(f'{target_path}') == False:
        os.system(f"mkdir {target_path}")
    try:
        taggers.calibrate()


        # Plotting of calibration curves
        scale = (lambda x: x**4, lambda x: x**1/4)
        scale = "linear"
        scale = "linear"
        if weights is not None:
            #distribute the bins such that each bin has the same yield, aka the same sum of weights 
            #distribute the bins such that each bin has the same yield, aka the same sum of weights 
            bins = bins_by_yield(df_tag[f"{tagger}_Eta"].values, weights, nbins)
            taggers.plot_calibration_curves(savepath = f'{target_path}', omega_range="minimal", bins = bins, x_scale = scale, y_scale = scale)
        else:
            taggers.plot_calibration_curves(savepath = f'{target_path}', omega_range="minimal", nbins = nbins, x_scale = scale, y_scale = scale)


        info_dict = {"TaggingEfficiency"     : taggers[tagger].stats.tagging_efficiency(calibrated = False),
                    "TaggingPower"           : taggers[tagger].stats.tagging_power(     calibrated = False),
                    "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True ), 
                    "TaggingPower_Cali"      : taggers[tagger].stats.tagging_power(     calibrated = True ),
                    "EffectiveMistag_Cali"   : taggers[tagger].stats.effective_mistag(  calibrated = True ), 
                    "EffectiveMistag"        : taggers[tagger].stats.effective_mistag(  calibrated = False),
                    "Fitpar_p0"              : [taggers[tagger].stats.params.params_delta[0], taggers[tagger].stats.params.errors_delta[0]],
                    "Fitpar_p1"              : [taggers[tagger].stats.params.params_delta[1], taggers[tagger].stats.params.errors_delta[1]],
                    "Fitpar_deltap0"         : [taggers[tagger].stats.params.params_delta[2], taggers[tagger].stats.params.errors_delta[2]],
                    "Fitpar_deltap1"         : [taggers[tagger].stats.params.params_delta[3], taggers[tagger].stats.params.errors_delta[3]],}
    except Exception as e: # Catch exceptions. Often caused by convergence issues in the training of the tagger
        print(f"An unexpected error occurred during calibration: {e}")
        print(traceback.format_exc())
        info_dict = {"TaggingEfficiency"     : [np.nan, np.nan],
                    "TaggingPower"           : [np.nan, np.nan],
                    "TaggingEfficiency_Cali" : [np.nan, np.nan], 
                    "TaggingPower_Cali"      : [np.nan, np.nan],
                    "EffectiveMistag_Cali"   : [np.nan, np.nan], 
                    "EffectiveMistag"        : [np.nan, np.nan],
                    "Fitpar_p0"              : [np.nan, np.nan],
                    "Fitpar_p1"              : [np.nan, np.nan],
                    "Fitpar_deltap0"         : [np.nan, np.nan],
                    "Fitpar_deltap1"         : [np.nan, np.nan],}
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


