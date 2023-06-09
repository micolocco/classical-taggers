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


# Prepare the dataset
def prepare_data(inputPath, features, scalerPath, train_batch_size = 32, test_batch_size = 1024):
    
    # Load the dataset
    dataset = inputDataset(inputPath, features, scalerPath)
    # Splitting
    train, validation, test = dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size = train_batch_size, shuffle=True)
    validation_dl = DataLoader(validation, batch_size = test_batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return train_dl, validation_dl, test_dl

def train_model_EarlyStopping(model, train_dl, validation_dl, repoPath, eventType, tagger, n_epochs = 500, earlyStop = 75):
       
        trainingEpoch_loss = []
        validationEpoch_loss = []
        lossValBest = 10000
        rollingAverageNew = 0
        rollingAverageOld = 10000 # just to be sure that the first rolling average value is lower than this
        stopped = False
        
        training_start = time.time()

        i = 0
        for epoch in range(n_epochs):
            model.train()
            stepLoss, training_loss = model.train_model(train_dl, epoch, n_epochs)
            # Train over mini-batches
            trainingEpoch_loss.append(np.array(stepLoss).mean())
            # Compute validation loss
            validationStep_loss, validation_loss = model.validate_model(validation_dl)
            validationEpoch_loss.append(np.array(validationStep_loss).mean())

            if epoch > earlyStop:  # check the termination condition
                rollingAverageNew = np.mean(validationEpoch_loss[-earlyStop:])
                if validation_loss < lossValBest:
                    lossValBest = validation_loss
                    lossTrainBest = training_loss
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
        return trainingEpoch_loss, validationEpoch_loss, bestEpoch, [lossValBest, lossTrainBest]

def save_model(model, repoPath, eventType, tagger, KaonCombiner = False, grid_n = None):
    
    prePath = f'{repoPath}modelSave/{eventType}/{tagger}/'
    if grid_n != None:
        if KaonCombiner:
            pthPath = f"{prePath}combiner_{grid_n}_{model.modelName}.pth"
        else:
            pthPath = f"{prePath}{grid_n}_{model.modelName}.pth"
        torch.save(copy.deepcopy(model.state_dict()), pthPath) #use deepcopy to prevent overfitting otherwise your best best_model_state will keep getting updated by the subsequent training iterations

    else:
        torch.save(copy.deepcopy(model.state_dict()), f"{prePath}{model.modelName}.pth")

def save_losses(model, trainLoss, valLoss, bestEpoch, bestLosses, repoPath, eventType, tagger, KaonCombiner = False, grid_n= None):
    
    prePath = f'{repoPath}csv/{eventType}/{tagger}/'
    name = model.name
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = model.modelName + f"combiner_{grid_n}"
        else:
            name = model.modelName + str(grid_n)
        np.savetxt(f"{prePath}{name}_Test.csv", valLoss, delimiter=",")
        np.savetxt(f"{prePath}{name}_Train.csv", trainLoss, delimiter=",")
        np.savetxt(f"{prePath}{name}_Best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(model, trainLoss, valLoss, bestEpoch, bestLosses, repoPath, eventType, tagger, KaonCombiner = False, grid_n= None):
    
    plt.figure() 
    plt.plot(trainLoss, label='Training loss')
    plt.plot(valLoss,label='Validation loss')
    plt.plot((bestEpoch, bestEpoch), (bestLosses[0], bestLosses[1]) ,"k--", label = "Best epoch")
    plt.legend(loc = "best")
    plt.title(model.modelName)
    prePath = f'{repoPath}plots/{eventType}/{tagger}/'
    name = model.name
    if grid_n != None: # save the loss values 
        if KaonCombiner:
            name = name + f"combiner_{grid_n}"
        else:
            name = name + str(grid_n)
    plt.savefig(f"{prePath}{name}_Loss.pdf")