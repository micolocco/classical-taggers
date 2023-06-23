import numpy as np 
import sys 
import pandas as pd 
import os
import time
import uproot
from inputDataset import inputDataset
from torch.utils.data import DataLoader
import copy
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve


# Prepare the dataset
def prepare_data(inputPath, features, scalerPath, train_batch_size = 32, test_batch_size = 1024):
    
    # Load the dataset
    dataset = inputDataset(inputPath, features, scalerPath)
    # Splitting
    train, validation, test = dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size = train_batch_size, shuffle=False)
    validation_dl = DataLoader(validation, batch_size = test_batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size = test_batch_size, shuffle=False)
    return train_dl, validation_dl, test_dl

def train_model_EarlyStopping(model, train_dl, validation_dl, repoPath, eventType, tagger, n_epochs = 500, earlyStop = 75):
       
        trainingEpoch_loss = []
        validationEpoch_loss = []
        lossValBest = 10000
        rollingAverageNew = 0
        rollingAverageOld = 10000 # just to be sure that the first rolling average value is lower than this
        stopped = False
        bestEpoch = 0

        training_start = time.time()

        i = 1
        for epoch in range(n_epochs):
            epoch_start = time.time()
            print(f"--------------Epoch:{epoch}/{n_epochs-1}-------------")
            stepLoss = model.train_model(train_dl, epoch, n_epochs)
            # Train over mini-batches
            trainingEpoch_loss.append(np.array(stepLoss).mean())
            # Compute validation loss
            validationStep_loss = model.validate_model(validation_dl)
            validationEpoch_loss.append(np.array(validationStep_loss).mean())
            print(f"Train:{np.array(stepLoss).mean():.6f}, Validation:{np.array(validationStep_loss).mean():.6f}, Time:{round((time.time()-epoch_start) ,2)}s") 
            if epoch > earlyStop:  # check the termination condition
                rollingAverageNew = np.mean(validationEpoch_loss[-earlyStop:])
                if validationEpoch_loss[-1] < lossValBest:
                    lossValBest = validationEpoch_loss[-1]
                    lossTrainBest = trainingEpoch_loss[-1]
                    bestEpoch = epoch
                    save_model(model, repoPath, eventType, tagger)
                    
            if epoch > (earlyStop +1):
                if rollingAverageNew > rollingAverageOld:
                    stopped = True 
                    break
                rollingAverageOld = rollingAverageNew 

            i +=1
        training_time = round((time.time()- training_start) / 60 , 2)
        print(f"Training finished in {training_time} min, {i} epochs, early stopping: {stopped}")
        return trainingEpoch_loss, validationEpoch_loss, bestEpoch, np.array([lossTrainBest, lossValBest], dtype=float)
        
def save_model(model, repoPath, eventType, tagger, KaonCombiner = False, grid_n = None):
    
    prePath = f'{repoPath}savedModels/{eventType}/{tagger}/'
    if grid_n != None:
        if KaonCombiner:
            pthPath = f"{prePath}combiner_{grid_n}_{model.modelName}.pth"
        else:
            pthPath = f"{prePath}{grid_n}_{model.modelName}.pth"
        torch.save(copy.deepcopy(model.state_dict()), pthPath) #use deepcopy to prevent overfitting otherwise your best best_model_state will keep getting updated by the subsequent training iterations

    else:
        torch.save(copy.deepcopy(model.state_dict()), f"{prePath}{model.modelName}.pth")

def load_model(model, repoPath, eventType, tagger, KaonCombiner = False, grid_n = None):

    prePath = f'{repoPath}savedModels/{eventType}/{tagger}/'
    if grid_n != None:
        if KaonCombiner:
            pthPath = f"{prePath}combiner_{grid_n}_{model.modelName}.pth"
        else:
            pthPath = f"{prePath}{grid_n}_{model.modelName}.pth"
        model.load_state_dict(torch.load(f"{pthPath}"))

    else:
        model.load_state_dict(torch.load(f"{pthPath}{model.modelName}.pth"))
    

def save_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, repoPath, eventType, tagger, KaonCombiner = False, grid_n= None):
    
    prePath = f'{repoPath}csv/{eventType}/{tagger}/'
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = name + f"combiner_{grid_n}"
        else:
            name = name + str(grid_n)
        np.savetxt(f"{prePath}{name}_Test.csv", valLoss, delimiter=",")
        np.savetxt(f"{prePath}{name}_Train.csv", trainLoss, delimiter=",")
        np.savetxt(f"{prePath}{name}_Best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, repoPath, eventType, tagger, KaonCombiner = False, grid_n= None):
    
    plt.plot(trainLoss, label='Training loss', c = 'orange')
    plt.plot(valLoss,label='Validation loss', c='blue')
    plt.plot((bestEpoch, bestEpoch), (bestLosses[0], bestLosses[1]) ,"k--", label = "Best epoch")
    plt.legend(loc = "best")
    plt.title(name)
    prePath = f'{repoPath}plots/{eventType}/{tagger}/'
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = name + f"combiner_{grid_n}"
        else:
            name = name + str(grid_n)
    plt.savefig(f"{prePath}{name}_Loss.pdf")
   
def plot_ROC(name, yPred, yTrue, yPredTrain, yTrueTrain, eventType, tagger, repoPath, KaonCombiner = False, grid_n= None):
    
    fpr, tpr,_ = roc_curve(yTrueTrain, yPredTrain)
    fpr_test, tpr_test,_ = roc_curve(yTrue, yPred)
    roc_auc = round(auc(fpr, tpr),2)
    roc_auc_test = round(auc(fpr_test, tpr_test),2)
    lw  = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'Train (area = {roc_auc})' )
    plt.plot(fpr_test, tpr_test, color='darkblue',lw=lw, label=f'Test (area = {roc_auc_test})' )
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve ')
    plt.legend(loc="lower right")
    prePath = f'{repoPath}plots/{eventType}/{tagger}/'
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = name + f"combiner_{grid_n}"
        else:
            name = name + str(grid_n)
    plt.savefig(f"{prePath}{name}_ROC.pdf")

def plot_mistag(name, yPred, yTrue, nbins, eventType, tagger, repoPath, KaonCombiner = False, grid_n= None):
   
    plt.title("Mistag rate")
    plt.yscale("log")
    plt.hist(yPred[yTrue == 0], bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = "Test (Label = 0)")
    plt.hist(yPred[yTrue == 1], bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = "Test (Label = 1)")
    plt.plot(Prob_train_0_bin_edges ,Prob_train_0_height, "b.", label = "Train (Label = 0)")
    plt.plot(Prob_train_1_bin_edges ,Prob_train_1_height, "r.", label = "Train (Label = 1)")
    plt.xlabel(r"Mistag rate $\eta$")
    plt.grid()
    plt.ylabel("Normalized number of tracks")
    plt.legend(loc = "best")

    prePath = f'{repoPath}plots/{eventType}/{tagger}/'
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = name + f"combiner_{grid_n}"
        else:
            name = name + str(grid_n)
    plt.savefig(f"{prePath}{name}_mistag.pdf")
       