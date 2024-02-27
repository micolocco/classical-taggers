import numpy as np 
import sys 
import pandas as pd 
import os
import time
from inputDataset import inputDataset
from torch.utils.data import DataLoader
import copy
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.special import expit
from NNModel import EarlyStopper
import configParameters as config
import json 



def splitByEvent (df):
    import random
    '''Function to random split by events (not by index) the dataset into training and test set
    Use random.Random(2) to reproduce same shuffling''' 
    events_list = np.unique(df.entry)
    random.Random(2).shuffle(events_list) 
    n_train = int(config.train_split*len(events_list))
    train_df = df[df.entry.isin(events_list[:n_train])]
    test_df = df[df.entry.isin(events_list[n_train:])]
    return train_df, test_df


def prepare_data(train_df, scalerPath, train_batch_size = 32, test_batch_size = 1024):
    # Load the dataset
    train_dataset = inputDataset(train_df, scalerPath, test = False)
    # Splitting in train and validation datasets
    train, validation = train_dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size = train_batch_size, shuffle=False)
    validation_dl = DataLoader(validation, batch_size = test_batch_size, shuffle=False)
    return train_dl, validation_dl 


def plot_features(data, target_path, name, flag, nbins=100):

    # Plot input features 
    plt.figure(figsize=(24,50))
    try:
        for i, col in enumerate(data.columns.to_list()):
            plt.subplot(10, 3, i + 1)
            
            if col == 'B_Tr_T_BPVIP':
                plotting_data = data[data[col]<2.5]
                plt.hist(plotting_data[col][plotting_data[flag]==0], density = True, bins=nbins, label = "Label = 0",color='b', alpha=0.5)
                plt.hist(plotting_data[col][plotting_data[flag]==1], density = True, bins=nbins, label = "Label = 1",color='r', alpha=0.5)
            else:
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', alpha=0.5)
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', alpha=0.5)
           
            plt.legend()
            plt.title(col)
            plt.tight_layout()
        #saveName = name_formatter.assign_name(folder, name)
        plt.savefig(f"{target_path}/{name}.pdf")
    except Exception as e:
        print(col,e)

def train_model_EarlyStopping(model, train_dl, validation_dl, target_path, n_epochs = 500):
        
        trainingEpoch_loss = []
        validationEpoch_loss = []
        lossValBest = 10000
        rollingAverageNew = 0
        rollingAverageOld = 10000 # just to be sure that the first rolling average value is lower than this
        stopped = False
        bestEpoch = 0

        training_start = time.time()
        early_stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

        i = 1
        for epoch in range(n_epochs):
            epoch_start = time.time()
            print(f"--------------Epoch:{epoch+1}/{n_epochs}-------------")
            stepLoss = model.train_model(train_dl, epoch, n_epochs)
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
        print(f"Training finished in {training_time} min, {i} epochs, early stopping: {stopped}")
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
    torch.save(copy.deepcopy(model.state_dict()), f"{target_path}/{model.modelName}_model.pth")
    save_hyperparameters(model, target_path)

def save_hyperparameters(model, target_path):
    info_dict = {
                'ModelName:' : model.modelName, 
                'learning_rate' : config.learning_rate,
                'patience' : config.patience,
                'min_delta' : config.min_delta,
                'activation_function' : config.activation_function,
                'n_epochs' : config.n_epochs,
                'train_split' : config.train_split,
                'randomSeed' : config.seed
                }
    with open(f"{target_path}/hyperparameters.json", "w") as f:
        json.dump(info_dict, f)


def load_model(model, target_path):
   # target_path = name_formatter.assign_name(folder, target_path)    
    # saveName = name_formatter.assign_name(target_path, model.modelName)
    model.load_state_dict(torch.load(f"{target_path}/{model.modelName}_model.pth"))
    

def save_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    folder = f'{target_path}/losses'
    os.mkdir(f'{folder}')
    #saveName = name_formatter.assign_name(folder, name)
    np.savetxt(f"{folder}/test.csv", valLoss, delimiter=",")
    np.savetxt(f"{folder}/train.csv", trainLoss, delimiter=",")
    np.savetxt(f"{folder}/best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    plt.figure()
    plt.plot(trainLoss, label='Training loss', c = 'orange')
    plt.plot(valLoss,label='Validation loss', c='blue')
    plt.plot((bestEpoch, bestEpoch), (bestLosses[0], bestLosses[1]) ,"k--", label = "Best epoch")
    plt.legend(loc = "best")
    plt.title(name)
    plt.savefig(f"{target_path}/Loss.pdf")
   
def plot_ROC(name, yPred, yTrue, target_path, yPredTrain = None, yTrueTrain = None):
    
    plt.figure()
    lw  = 2
    fpr_test, tpr_test,_ = roc_curve(yTrue, yPred)
    roc_auc_test = round(auc(fpr_test, tpr_test),5)
    plt.plot(fpr_test, tpr_test, color='darkblue',lw=lw, label=f'Test (area = {roc_auc_test})' )
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve ')
    folder = 'plots'
    plt.legend(loc="lower right")
    #.assign_name(folder, name)
    if yPredTrain is not None:
        fpr, tpr,_ = roc_curve(yTrueTrain, yPredTrain)
        roc_auc = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'Train (area = {roc_auc})' )
        plt.legend(loc="lower right")
        plt.savefig(f"{target_path}/ROC_TRAIN_VAL.pdf")
    else:
        plt.savefig(f"{target_path}/ROC_TEST.pdf")

def logistic_regression(yPredTrain, yTrueTrain, target_path, name):
    
    clf = LogisticRegression().fit(yPredTrain, yTrueTrain.ravel())  
    #prePath = target_path.assign_name(folder, name)
    pickle.dump(clf , open(f"{target_path}/LogReg.pck" , "wb"))
    return clf

def plot_NNoutput_mistag (name, clf, yPredTest, yTrueTest, yPredTrain, yTrueTrain, target_path, nbins=100):
    
    plt.figure()
    LR_test = np.linspace(0, 1, 300)
    loss = expit(LR_test * clf.coef_ + clf.intercept_).ravel()
    plt.title("Logistic Regression")
    plt.grid()
    plt.plot(yPredTest[yTrueTest == 0][0:500], np.zeros(500) , "b.",alpha = 0.5, label = "Label = 0")
    plt.plot(yPredTest[yTrueTest == 1][0:500], np.ones(500) ,  "r.",alpha = 0.5, label = "Label = 1")
    plt.plot(LR_test, loss ,color = "k")
    plt.legend(loc = "best")
    folder = 'plots'
   # saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/LogReg.pdf")
    
    prob_train_0_height , prob_train_0_bin_edges= np.histogram(yPredTrain[yTrueTrain == 0] , bins = nbins, density = True)
    prob_train_0_bin_edges = prob_train_0_bin_edges[:len(prob_train_0_bin_edges)-1]+ (prob_train_0_bin_edges[1]-prob_train_0_bin_edges[0])/2
    prob_train_1_height , prob_train_1_bin_edges= np.histogram(yPredTrain[yTrueTrain == 1] ,bins = nbins, density = True)
    prob_train_1_bin_edges = prob_train_1_bin_edges[:len(prob_train_1_bin_edges)-1]+ (prob_train_1_bin_edges[1]-prob_train_1_bin_edges[0])/2

    y_test_predict_LR = clf.predict_proba(yPredTest)[:,0]
    y_train_predict_LR = clf.predict_proba(yPredTrain)[:,0]

    prob_train_0_height_LR , prob_train_0_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 0], bins = nbins, density = True)
    prob_train_0_bin_edges_LR = prob_train_0_bin_edges_LR[:len(prob_train_0_bin_edges_LR)-1]+ (prob_train_0_bin_edges_LR[1]-prob_train_0_bin_edges_LR[0])/2
    prob_train_1_height_LR , prob_train_1_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 1], bins = nbins, density = True)
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
    axs[1].set_xlabel(r"Mistag rate $\eta$")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 0],bins = nbins,density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 1],bins = nbins,density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[1].plot(prob_train_0_bin_edges_LR ,prob_train_0_height_LR, "b.", label = "Train (Label = 0)")
    axs[1].plot(prob_train_1_bin_edges_LR ,prob_train_1_height_LR, "r.", label = "Train (Label = 1)")
    axs[1].grid()
    axs[1].legend(loc = "best")

    #folder = 'plots'
    #saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/NNoutput_mistag.pdf")
    plt.close()

def plot_mistag(name, clf, yPred, yTrue, target_path, type, nbins=100):

    plt.figure()
    plt.title("Mistag rate")
    plt.yscale("log")
    y_predict_LR = clf.predict_proba(yPred)[:,0]
    plt.hist(y_predict_LR[yTrue.ravel() == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"{type} (Label = 0)")
    plt.hist(y_predict_LR[yTrue.ravel() == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = f"{type} (Label = 1)")
    plt.xlabel(r"$\eta$")
    plt.grid()
    plt.ylabel("Normalized number of tracks")
    plt.legend(loc = "best")
   # folder = 'plots'
   # saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/mistag_{type}.pdf")

def plot_tagDec(df_TagParticles, name, target_path, nbins=100):
    # Get the particle with the lowest mistag for each event
    plt.figure()
    plt.title(r"$\eta$ TagParticle")
    plt.yscale("log")
    plt.hist(df_TagParticles.loc[df_TagParticles.TagDec == -1].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "blue" , alpha = 0.5, label = f"(TagDec = -1)")
    plt.hist(df_TagParticles.loc[df_TagParticles.TagDec == 1].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "red" ,  alpha = 0.5,label = f"(TagDec = 1)")
    plt.grid()
    plt.xlabel(r"$\eta$")
    plt.ylabel("Normalized number of tracks")
    plt.legend(loc = "best")
    #folder = 'plots'
    #aveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/mistag_TagDec.pdf")
    plt.close()

def calibration(modelName, tagger, df_tag, eventType, target_path):

    #Calibration of the taggers and parameters saving
    import lhcb_ftcalib as ft

    taggers = ft.TaggerCollection()
    taggers.create_tagger(name = tagger, eta_data = df_tag.Eta.tolist(), dec_data = df_tag.TagDec.tolist(), B_ID = df_tag.B_TRUEID.tolist(),mode = eventType[:2])
    taggers.set_calibration(ft.PolynomialCalibration(npar = 2,link =  ft.link.mistag))
    taggers.calibrate()

    # Plotting of calibration curves
   # folder = 'calibrationPlots'
   # saveName = name_formatter.assign_name(folder, modelName)
  #  if os.path.isdir(saveName) == False:
       #     os.system(f"mkdir {saveName}")
    taggers.plot_calibration_curves(savepath = target_path, omega_range="minimal", nbins=10)

    #folder = 'results'
   # saveName = name_formatter.assign_name(folder, modelName)
    info_dict = {"TaggingEfficiency" : taggers[tagger].stats.tagging_efficiency(calibrated = False),
    "TaggingPower" : taggers[tagger].stats.tagging_power(calibrated = False) ,
    "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True), "TaggingPower_Cali" : taggers[tagger].stats.tagging_power(calibrated = True),
    "EffectiveMistag_Cali" : taggers[tagger].stats.effective_mistag(calibrated = True) , "EffectiveMistag" : taggers[tagger].stats.effective_mistag(calibrated = False) }
    with open(f"{target_path}/taggingInfo.json", "w") as f:
        json.dump(info_dict, f)
    print(f"Tagger parameters saved at {target_path}")

