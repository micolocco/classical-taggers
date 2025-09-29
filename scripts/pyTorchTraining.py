import numpy as np 
import os
import time
import torch
from torch.utils.data import DataLoader
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.special import expit
import json
import pandas as pd
import scripts.pipeline

# Local imports
from scripts.NNModel import EarlyStopper
from scripts.inputDataset import inputDataset
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import yaml


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

def plot_features_byID(data, features_list, ID, target_path, name, nbins=100):
    # Plot input features 
    plt.figure(figsize=(24,25))
    pos=0
    for i, col in enumerate(data.columns.to_list()):
        if col in features_list:
            plt.subplot(5, 4 , pos + 1) # hardcoded according to the number of features
            if col in nice_names.keys():
                plt.hist(data[col][(data['label']==0)&(data['B_TRUEID']==ID)], density = True, bins=nbins, label = f"label = 0, {ID}",color='b', histtype='step',  lw=2, range=ranges[col])
                plt.hist(data[col][(data['label']==0)&(data['B_TRUEID']==-ID)], density = True, bins=nbins, label = f"label = 0, -{ID}",color='r', histtype='step',  lw=2, range=ranges[col])
                plt.hist(data[col][(data['label']==1)&(data['B_TRUEID']==ID)], density = True, bins=nbins, label = f"label = 1, {ID}",color='orange', histtype='step',  lw=2, range=ranges[col])
                plt.hist(data[col][(data['label']==1)&(data['B_TRUEID']==-ID)], density = True, bins=nbins, label = f"label = 1, -{ID}",color='cyan', histtype='step',  lw=2, range=ranges[col])

                plt.xlabel(nice_names[col])
                plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            pos+=1
    plt.savefig(f"{target_path}/{name}.pdf")
    
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


#def splitByEvent (df, seed, train_val_split):
#    '''Function to random split by events (not by index) the dataset into training and test set
#    Use random.Random(2) to reproduce same shuffling''' 
#    import random
#    events_list = np.unique(df.event_entry)
#    random.Random(3).shuffle(events_list) # cfg.seed
#    n_train_val = int(train_val_split*len(events_list)) # Divide
#    n_train = int(0.8 * n_train_val)
#    train_df = df[df.event_entry.isin(events_list[:n_train])].copy()
#    val_df = df[df.event_entry.isin(events_list[n_train:n_train_val])].copy()
#    test_df = df[df.event_entry.isin(events_list[n_train_val:])].copy()
#    return train_df.query('selected==1'), val_df.query('selected==1'), test_df


import numpy as np
import pandas as pd

def _balance_four_bins(df, trueid_col="B_TRUEID", label_col="label", random_state=42, strict=True,):
    """
    Downsample df so that the four categories have equal counts:
      (TRUEID<0 & label=0), (TRUEID>0 & label=0),
      (TRUEID<0 & label=1), (TRUEID>0 & label=1)
    Returns a shuffled, balanced dataframe.
    """
    if df.empty:
        return df

    sign = np.where(df[trueid_col] > 0, "pos", "neg")
    g = pd.Series(sign, index=df.index) + "_" + df[label_col].astype(int).astype(str)

    needed = {"neg_0", "pos_0", "neg_1", "pos_1"}
    present = set(g.unique())
    missing = needed - present
    if missing:
        if strict:
            raise ValueError(f"Cannot balance: missing categories: {sorted(missing)}")
        # Soft mode: drop missing bins from the target set
        needed = needed - missing
        if not needed:
            return df  # nothing to balance

    sizes = g.value_counts()
    k = int(sizes.loc[list(needed)].min())

    rng = np.random.RandomState(random_state)
    parts = []
    for key in ["neg_0", "pos_0", "neg_1", "pos_1"]:
        if key not in needed:
            continue
        idx = g[g == key].index
        pick = rng.choice(idx, size=k, replace=False)
        parts.append(df.loc[pick])

    balanced = pd.concat(parts).sample(frac=1.0, random_state=random_state)
    return balanced

def balance_by_label(df, label_col="label", random_state=3):
    if df.empty:
        return df
    counts = df[label_col].value_counts()
    k = counts.min()
    rng = np.random.RandomState(random_state)
    parts = []
    for y in [0, 1]:
        idx = df[df[label_col] == y].index
        pick = rng.choice(idx, size=k, replace=False)
        parts.append(df.loc[pick])
    return pd.concat(parts).sample(frac=1.0, random_state=random_state)

def splitByEvent(df, seed=3, asym_level='asym_level1', train_val_split=0.8, trueid_col="B_TRUEID", label_col="label", strict_balance=True):
    """
    Split a dataset into training, validation, and test sets **by event ID**, 
    with optional balancing of classes.

    The split is performed at the event level (using `event_entry`) so that
    all tracks from the same event end up in the same subset. Each subset is
    then further processed according to the chosen `asym_level`:

    - `asym_level='asym_level0'`:
        Training, validation, and test subsets are balanced across the four
        categories defined by (`sign(B_TRUEID)`, `label`):
          (TRUEID > 0, label=0), (TRUEID > 0, label=1),
          (TRUEID < 0, label=0), (TRUEID < 0, label=1).
        Only rows with `selected==1` are balanced. Test rows with `selected==0`
        are kept unchanged and appended back afterwards.

    - `asym_level='asym_level1'` (default):
        Training and validation subsets are balanced across the same four
        categories as above (only `selected==1` rows).  
        The test subset is left unbalanced: all `selected==1` rows are kept
        without modification, and `selected==0` rows are appended.

    - `asym_level='asym_level2'`:
        No four-bin balancing is applied. Instead, training and validation
        subsets are balanced **only by label (0 vs 1)**, ignoring
        the B/anti-B distinction.  
        The test subset is left unbalanced (`selected==1` kept as-is,
        `selected==0` appended).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least `event_entry`, `selected`,
        `B_TRUEID`, and `label`.
    seed : int, optional
        Random seed for reproducibility (default=3).
    asym_level : {'asym_level0', 'asym_level1', 'asym_level2'}, optional
        Balancing mode (see description above).
    train_val_split : float, optional
        Fraction of events assigned to the train+val split (default=0.8).
        The remaining fraction is used for test (calibration sample).
    trueid_col : str, optional
        Name of the column holding the true B hadron ID (default="B_TRUEID").
    label_col : str, optional
        Name of the column holding the binary classification label (default="label").
    strict_balance : bool, optional
        If True, raise an error if one of the required bins is missing when balancing.

    Returns
    -------
    train_df : pandas.DataFrame
        Balanced training set (only `selected==1` rows).
    val_df : pandas.DataFrame
        Balanced validation set (only `selected==1` rows).
    test_df : pandas.DataFrame
        Test set containing all `selected==0` rows plus either balanced or
        unbalanced `selected==1` rows depending on `asym_level`.

    Notes
    -----
    - Balancing is performed by **downsampling** to the smallest bin size,
      so some tracks may be discarded.
    - The split is done at the event level to avoid leakage of tracks
      from the same event across train/val/test.
    """
    import random
    seed=3
    # Split by event
    events_list = np.unique(df.event_entry)
    random.Random(seed).shuffle(events_list)

    n_train_val = int(train_val_split * len(events_list))
    n_train = int(0.8 * n_train_val)

    train_df = df[df.event_entry.isin(events_list[:n_train])].copy()
    val_df   = df[df.event_entry.isin(events_list[n_train:n_train_val])].copy()
    test_df  = df[df.event_entry.isin(events_list[n_train_val:])].copy()

    # --- Train ---
    train_sel1 = train_df.query('selected==1').copy()
        # --- Val ---
    val_sel1 = val_df.query('selected==1').copy()
    # --- Test ---
    test_sel1 = test_df.query('selected==1').copy()
    test_sel0 = test_df.query('selected==0').copy()  # keep as-is
    if asym_level == 'asym_level0':
        train_bal  = _balance_four_bins(train_sel1, trueid_col=trueid_col, label_col=label_col,
                                    random_state=seed, strict=strict_balance)
        val_bal  = _balance_four_bins(val_sel1, trueid_col=trueid_col, label_col=label_col,
                                  random_state=seed, strict=strict_balance)
        test_sel1_bal = _balance_four_bins(test_sel1, trueid_col=trueid_col, label_col=label_col,
                                       random_state=seed, strict=strict_balance)
        test_out = pd.concat([test_sel1_bal, test_sel0], ignore_index=False).sort_index()
        return train_bal, val_bal, test_out
    elif asym_level == 'asym_level1':
        train_bal  = _balance_four_bins(train_sel1, trueid_col=trueid_col, label_col=label_col,
                                    random_state=seed, strict=strict_balance)
        val_bal  = _balance_four_bins(val_sel1, trueid_col=trueid_col, label_col=label_col,
                                  random_state=seed, strict=strict_balance)
        test_out = pd.concat([test_sel1, test_sel0], ignore_index=False).sort_index()
        return train_bal, val_bal, test_out
    elif asym_level == 'asym_level2':
        # Balance by label 0 and label 1 only, no matter B_TRUEID
        train_sel1_bal = balance_by_label(train_sel1)
        val_sel1_bal   = balance_by_label(val_sel1)
        test_out = pd.concat([test_sel1, test_sel0], ignore_index=False).sort_index()
        return train_sel1_bal, val_sel1_bal, test_out

from pathlib import Path

def get_anchor_dir(model_path: str, anchor: str = "union_PROBNN") -> Path:
    p = Path(model_path)
    parts = p.parts
    if anchor in parts:
        i = parts.index(anchor)
        return Path(*parts[:i+1])  # keep up to and including anchor
    # Fallback: if anchor not found, use the provided path itself
    return p

def _optimize_df_for_parquet(df: pd.DataFrame, cat_thresh: float = 0.4) -> pd.DataFrame:
    """Lightweight dtype optimization for smaller/faster Parquet."""
    df = df.copy()
    # low-cardinality objects -> category
    for c in df.select_dtypes(include=["object"]).columns:
        nunique = df[c].nunique(dropna=True)
        if nunique <= cat_thresh * len(df):
            df[c] = df[c].astype("category")
    return df

def prepare_data(train_df, val_df, scalerPath, transformerPath, train_batch_size, seed, test_batch_size = 1024):
    # Load the dataset
    train_dataset = inputDataset(df=train_df) #scaler=PowerTransformer() 
    # train_dataset.scale(test=False, scalerPath=scalerPath, transformerPath=transformerPath)
    val_dataset = inputDataset(df=val_df)
    # val_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    # Prepare data loaders
    torch.manual_seed(seed) # to ensure reproducibility
    train_dl = DataLoader(train_dataset, batch_size = train_batch_size, shuffle=False)
    validation_dl = DataLoader(val_dataset, batch_size = test_batch_size, shuffle=False)
    return train_dl, validation_dl 


def plot_features(data, features_list, target_path, name, flag, nbins=100):
    # Plot input features 
    plt.figure(figsize=(24,25))
    pos=0
    for i, col in enumerate(data.columns.to_list()):
        if col in features_list:
            plt.subplot(5, 4 , pos + 1) # hardcoded according to the number of features
            if col in nice_names.keys():
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', histtype='step',  lw=2, range=ranges[col])
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', histtype='step',  lw=2, range=ranges[col])
                plt.xlabel(nice_names[col])
            else:
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b',histtype='step',  lw=2, )
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', histtype='step',  lw=2,)
                plt.xlabel(col)

            plt.legend()
            plt.tight_layout()
            pos+=1
    plt.savefig(f"{target_path}/{name}.pdf")
    

def get_architecture(config):
    if 'architecture' in config.keys():
        return config['architecture']
    else:
        nL = config['numlayers']
        nN = config['numneurons']
        dp = config['dropout']
        return f'nL{nL}_nN{nN}_dp{dp}'

def train_model_EarlyStopping(model, train_dl, validation_dl, target_path, config):
        
        trainingEpoch_loss = []
        validationEpoch_loss = []
        lossValBest = 10000
        rollingAverageNew = 0
        rollingAverageOld = 10000 # just to be sure that the first rolling average value is lower than this
        stopped = False
        bestEpoch = 0

        training_start = time.time()
        early_stopper = EarlyStopper(patience=config['patience'], min_delta=config['min_delta'])
        
        i = 1
        initial_validation_loss = model.validate_model(validation_dl)
        print(f"The initial Validation Loss: {np.array(np.array(initial_validation_loss).mean()).mean():.6f}")

        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            print(f"--------------Epoch:{epoch+1}/{config['n_epochs']}-------------")
            stepLoss = model.train_model(train_dl, epoch, config['n_epochs'])
            # Train over mini-batches
            trainingEpoch_loss.append(np.array(stepLoss).mean())
            # Compute validation loss
            validationStep_loss = model.validate_model(validation_dl)
            validationEpoch_loss.append(np.array(validationStep_loss).mean())
            print(f"Train:{np.array(stepLoss).mean():.6f}, Validation:{np.array(validationStep_loss).mean():.6f}, Time:{round((time.time()-epoch_start) ,2)}s") 
            if early_stopper.early_stop(validationEpoch_loss[-1]): 
                stopped = True 
                break
            if early_stopper.counter == 0:
                lossValBest = validationEpoch_loss[-1]
                lossTrainBest = trainingEpoch_loss[-1]
                bestEpoch = epoch
                save_model(model, target_path)
                bestModel = copy.deepcopy(model)
            i +=1
        training_time = round((time.time()- training_start) / 60 , 2)
        print(f"Training finished in {training_time} min, {i-1} epochs, early stopping: {stopped}")
        return bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, np.array([lossTrainBest, lossValBest], dtype=float)
            #if epoch > earlyStop:  # check the termination condition
            #    rollingAverageNew = np.mean(validationEpoch_loss[-earlyStop:])
            #    if validationEpoch_loss[-1] < lossValBest:
            #        lossValBest = validationEpoch_loss[-1]
            #        lossTrainBest = trainingEpoch_loss[-1]
            #        bestEpoch = epoch
            #        save_model(model, name_formatter)
            #        bestModel = copy.deepcopy(model)
            #        
            #if epoch > (earlyStop +1):
            #    if rollingAverageNew > rollingAverageOld:
            #        stopped = True 
            #        break
            #    rollingAverageOld = rollingAverageNew 
            
        #return bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, dtype=float)

def save_model(model, target_path):
    # target_path = name_formatter.assign_name(folder, target_path)
    # torch.save(copy.deepcopy(model.state_dict()), f"{target_path}/model.pth")
    torch.save(copy.deepcopy(model), f"{target_path}/model.pth")
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


def save_losses(trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    folder = f'{target_path}/losses'
    os.makedirs(f'{folder}', exist_ok=True)
    np.savetxt(f"{folder}/test.csv", valLoss, delimiter=",")
    np.savetxt(f"{folder}/train.csv", trainLoss, delimiter=",")
    np.savetxt(f"{folder}/best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(tagger, trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    plt.figure()
    plt.plot(trainLoss, label='Training', c = 'orange')
    plt.plot(valLoss,label='Validation', c='blue')
    plt.axvline(bestEpoch, linestyle='--', color='tab:gray', label="Best epoch")
    plt.legend(loc = "best")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(f"{tagger}", fontsize=24)
    plt.savefig(f"{target_path}/Loss.pdf")
   
def plot_ROC(tagger, val_df, target_path, train_df= None):
    
    plt.figure()
    lw  = 2
    fpr_test, tpr_test,_ = roc_curve(val_df.yTrue, val_df.yPred)
    roc_auc_test = round(auc(fpr_test, tpr_test),5)
    plt.plot(fpr_test, tpr_test, color='darkblue',lw=lw, label=f'Test (area = {roc_auc_test})' )
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f"{tagger}", fontsize=24)
    if train_df is not None:
        fpr, tpr,_ = roc_curve(train_df.yTrue, train_df.yPred)
        roc_auc = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'Train (area = {roc_auc})' )
        plt.legend(loc="lower right")
        plt.savefig(f"{target_path}/ROC_TRAIN_VAL.pdf")
    else:
        plt.savefig(f"{target_path}/ROC_TEST.pdf")

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

def plot_mistag(tagger, decayType, df, target_path, type, show_trueB=False, clf = None, nbins=100):
    plt.figure()
    # plt.title("Mistag rate")
    plt.yscale("log")
    if decayType[:2]=='Bu':
        ID=521
    if decayType[:2]=='Bd':
        ID=511
    if decayType[:2]=='Bs':
        ID=531

    if clf:
        y_predict_LR = clf.predict_proba(df.yPred)[:,0]
        plt.hist(y_predict_LR[df.yTrue == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"wrong tagging decision")
        plt.hist(y_predict_LR[df.yTrue == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = f"correct tagging decision")
        plt.title(f'{tagger} mistag after Logistic Regression', fontsize=24)
    else:
        if show_trueB:
            plt.hist(1-df.yPred[(df.yTrue==0)&(df.B_TRUEID==-ID)],bins = nbins, density = True, histtype="stepfilled", color = "skyblue", alpha = 0.5, label = f"l=0, antiB")
            plt.hist(1-df.yPred[(df.yTrue==0)&(df.B_TRUEID==ID)],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"l=0, B")
            plt.hist(1-df.yPred[(df.yTrue==1)&(df.B_TRUEID==-ID)],bins = nbins, density = True, histtype="stepfilled", color = "salmon", alpha = 0.5, label = f"true l=1, antiB")
            plt.hist(1-df.yPred[(df.yTrue==1)&(df.B_TRUEID==ID)],bins = nbins, density = True, histtype="stepfilled", color = "red", alpha = 0.5, label = f"true l=1, B")
   
        else:
            plt.hist(1-df.yPred[df.yTrue == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"wrong tagging decision")
            plt.hist(1-df.yPred[df.yTrue == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = f"correct tagging decision")
    data1=1-df.yPred[(df.yTrue==0)&(df.B_TRUEID==-ID)]  
    data2=1-df.yPred[(df.yTrue==0)&(df.B_TRUEID==ID)]   
    data3=1-df.yPred[(df.yTrue==1)&(df.B_TRUEID==-ID)]  
    data4=1-df.yPred[(df.yTrue==1)&(df.B_TRUEID==ID)]
    plt.title(f"{tagger}", fontsize=24)
    plt.xlabel(r"1 - NN output", fontsize=24)
    #plt.annotate(f'{len(df.yPred)} tracks', xy=(0, 1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.grid()
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title=f'{type}:{len(df.yPred)} total tracks')
    plt.savefig(f"{target_path}/NNoutput_{type}.pdf")
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle(f'{type}: {tagger}', fontsize=24)
    axs[0].hist(data1, bins=nbins, color='skyblue', density = True, histtype="stepfilled", label = f'wrong tag. dec, TRUE ID=-521',)
    axs[0].hist(data2, bins=nbins, color='blue', density = True, histtype="step",label=f'wrong tag. dec, TRUE ID=521')
    axs[0].legend()
    axs[1].hist(data3, bins=nbins, color='salmon', density = True, histtype="stepfilled",label=f'correct tag. dec, TRUE ID=-521')  
    axs[1].hist(data4, bins=nbins, color='red', density = True, histtype="step",label=f'correct tag. dec, TRUE ID=521',)
    axs[1].legend()
    for ax in axs.flat:
        ax.set_yscale('log')
        ax.set_xlabel(f'1 - NN output', fontsize=22)
        ax.set_ylabel('Normalized number of tracks', fontsize=22)

    plt.tight_layout()
    plt.savefig(f"{target_path}/NNoutput_{type}_byTRUEID.pdf")

    
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


def calibration(tagger, df_tag, eventType, target_path, B_ID= 'B_TRUEID',calibration_option='mistag', nbins=10, weights=None):

    #Calibration of the taggers and parameters saving
    import lhcb_ftcalib as ft

    taggers = ft.TaggerCollection()
    # Define array of 1 is there are no weights (sweights on data)
    if weights is None:
        weights = np.ones(len(df_tag))

    
    taggers.create_tagger(name = tagger, eta_data = df_tag[f"{tagger}_Eta"].tolist(), dec_data = df_tag[f"{tagger}_TagDec"].tolist(), B_ID = df_tag[B_ID].tolist(),mode = 'Bu', weight=weights ) # to be changed in mode = eventType[:2], B_ID = reconstructed ID when moving to data!
    
    if calibration_option=='logit':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    elif calibration_option=='mistag':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.mistag)) 
    else:
        print('Not a valid calibration function')
    taggers.retry_on_error(use_link_alternative=ft.link.logit) # use logit link function if minimization did not converge the first time
    taggers.calibrate()
    # Plotting of calibration curves
    target_path = f'{target_path}/{calibration_option}'
    if os.path.isdir(f'{target_path}') == False:
        os.system(f"mkdir {target_path}")

    
    taggers.calibrate()
    # Plotting of calibration curves
    scale = (lambda x: x**4, lambda x: x**1/4)
    scale = "linear"
    #distribute the bins such that each bin has the same yield, aka the same sum of weights
    # bins = bins_by_yield(df_tag[f"{tagger}_Eta"].values, weights, nbins)
    # print(bins)
    # if any(bins[:-1] == bins[1:]):
    #     print("Warning: Bins are not unique, using linspace instead.")
    #     bins = np.linspace(df_tag[f"{tagger}_Eta"].min(), df_tag[f"{tagger}_Eta"].max(), nbins+1)

    class_indices = df_tag[B_ID].values
    class_label_dict = {521: '$B^+$', -521: '$B^-$', 511: '$B^0$', -511: '$\overline{B}^0$', 531: '$B_s^0$', -531: '$\overline{B}_s^0$'}

    taggers.draw_split_calibration_curve(nrows = 1, ncols = 2, class_indices = class_indices, class_label_dict = class_label_dict,
                                            file_name = 'split_calibration_curves.pdf', savepath = f'{target_path}', omega_range="minimal", 
                                            nbins = nbins, x_scale = scale, y_scale = scale)#, share_y= True, share_x = True)

    taggers.plot_calibration_curves(savepath = f'{target_path}', omega_range="minimal", nbins = nbins, x_scale = scale, y_scale = scale)


    info_dict = {"TaggingEfficiency" : taggers[tagger].stats.tagging_efficiency(calibrated = False),
    "TaggingPower" : taggers[tagger].stats.tagging_power(calibrated = False) ,
    "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True), "TaggingPower_Cali" : taggers[tagger].stats.tagging_power(calibrated = True),
    "EffectiveMistag_Cali" : taggers[tagger].stats.effective_mistag(calibrated = True) , "EffectiveMistag" : taggers[tagger].stats.effective_mistag(calibrated = False) }
    with open(f"{target_path}/taggingInfo_{calibration_option}.json", "w") as f:
        json.dump(info_dict, f)
    print(f"Tagger parameters saved at {target_path}\n")
    print(f"Tagging information in a presentation-friendly format:\n")
    # Process the data
    processed_data = {key: propagate_and_round(value) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" for key, values in processed_data.items()}
    # Print the formatted data
    for key, value in formatted_data.items():
        print(f"{key}: {value}")

# Function to propagate and round the errors and values
def propagate_and_round(values):
    values = np.array(values) * 100  # Multiply all values by 100

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





