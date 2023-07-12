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
from sklearn.linear_model import LogisticRegression
import pickle



# Prepare the dataset
def prepare_data(df, scalerPath, train_batch_size = 32, test_batch_size = 1024):
    
    # Load the dataset
    dataset = inputDataset(df, scalerPath)
    # Splitting
    train, validation, test = dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size = train_batch_size, shuffle=False)
    validation_dl = DataLoader(validation, batch_size = test_batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size = test_batch_size, shuffle=False)
    return train_dl, validation_dl, test_dl

def train_model_EarlyStopping(model, train_dl, validation_dl, name_formatter, n_epochs = 500, earlyStop = 75):
       
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
                    save_model(model, name_formatter)
                    
            if epoch > (earlyStop +1):
                if rollingAverageNew > rollingAverageOld:
                    stopped = True 
                    break
                rollingAverageOld = rollingAverageNew 

            i +=1
        training_time = round((time.time()- training_start) / 60 , 2)
        print(f"Training finished in {training_time} min, {i} epochs, early stopping: {stopped}")
        return trainingEpoch_loss, validationEpoch_loss, bestEpoch, np.array([lossTrainBest, lossValBest], dtype=float)
        
def save_model(model, name_formatter):
    
    folder = 'savedModels'
    saveName = name_formatter.assign_name(folder, model.modelName)
    torch.save(copy.deepcopy(model.state_dict()), f"{saveName}.pth")

def load_model(model, name_formatter):

    folder = 'savedModels'
    saveName = name_formatter.assign_name(folder, model.modelName)
    model.load_state_dict(torch.load(f"{saveName}.pth"))
    

def save_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, name_formatter):
    
    folder = 'csv'
    saveName = name_formatter.assign_name(folder, name)
    np.savetxt(f"{saveName}_Test.csv", valLoss, delimiter=",")
    np.savetxt(f"{saveName}_Train.csv", trainLoss, delimiter=",")
    np.savetxt(f"{saveName}_Best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(name, trainLoss, valLoss, bestEpoch, bestLosses, name_formatter):
    
    plt.figure()
    plt.plot(trainLoss, label='Training loss', c = 'orange')
    plt.plot(valLoss,label='Validation loss', c='blue')
    plt.plot((bestEpoch, bestEpoch), (bestLosses[0], bestLosses[1]) ,"k--", label = "Best epoch")
    plt.legend(loc = "best")
    plt.title(name)
    folder = 'plots'
    saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{saveName}_Loss.pdf")
   
def plot_ROC(name, yPred, yTrue, yPredTrain, yTrueTrain, name_formatter):
    
    plt.figure()
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
    folder = 'plots'
    saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{saveName}_ROC.pdf")

def plot_NNoutput(name, yPred, yTrue, yPredTrain, yTrueTrain, name_formatter, nbins=100):

    plt.figure()
    plt.title("NN Output")
    plt.yscale("log")
    plt.hist(yPred[yTrue == 0], bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = "Test (Label = 0)")
    plt.hist(yPred[yTrue == 1], bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = "Test (Label = 1)")
    prob_train_0_height , prob_train_0_bin_edges= np.histogram(yPredTrain[yTrueTrain == 0] , bins = nbins, density = True)
    prob_train_0_bin_edges = prob_train_0_bin_edges[:len(prob_train_0_bin_edges)-1]+ (prob_train_0_bin_edges[1]-prob_train_0_bin_edges[0])/2
    prob_train_1_height , prob_train_1_bin_edges= np.histogram(yPredTrain[yTrueTrain == 1] ,bins = nbins, density = True)
    prob_train_1_bin_edges = prob_train_1_bin_edges[:len(prob_train_1_bin_edges)-1]+ (prob_train_1_bin_edges[1]-prob_train_1_bin_edges[0])/2
    plt.plot(prob_train_0_bin_edges ,prob_train_0_height, "b.", label = "Train (Label = 0)")
    plt.plot(prob_train_1_bin_edges ,prob_train_1_height, "r.", label = "Train (Label = 1)")
    #plt.xlabel(r"Mistag rate $\eta$")
    plt.xlabel(r"NN Output")
    plt.grid()
    plt.ylabel("Normalized number of tracks")
    plt.legend(loc = "best")
    folder = 'plots'
    saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{saveName}_NNOutput.pdf")

def mistag (name, model, yPredTest, yTrueTest, yPredTrain, yTrueTrain, name_formatter, nbins=100):
    clf = LogisticRegression().fit(yPredTrain, yTrueTrain.ravel())  
    folder = "savedModels"
    prePath = name_formatter.assign_name(folder, "LogReg")
    pickle.dump(clf , open(f"{prePath}.pck" , "wb"))
    eta = clf.predict_proba(yPredTest)[:,1]

    
    plt.figure()
    '''
    LR_test = np.linspace(0, 1, 300)
    loss = expit(LR_test * clf.coef_ + clf.intercept_).ravel()
    plt.title("Logistic Regression")
    plt.grid()
    plt.plot(yPredTest[yTrueTest == 0][0:1000], np.zeros(1000) , "b.",alpha = 0.5, label = "Label = 0")
    plt.plot(yPredTest[yTrueTest == 1][0:1000], np.ones(1000) ,  "r.",alpha = 0.5, label = "Label = 1")
    plt.plot(LR_test, loss ,color = "k")
    plt.legend(loc = "best")
    folder = 'plots'
    saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{saveName}_LogReg.pdf")'''
    
    prob_train_0_height , prob_train_0_bin_edges= np.histogram(yPredTrain[yTrueTrain == 0] , bins = nbins, density = True)
    prob_train_0_bin_edges = prob_train_0_bin_edges[:len(prob_train_0_bin_edges)-1]+ (prob_train_0_bin_edges[1]-prob_train_0_bin_edges[0])/2
    prob_train_1_height , prob_train_1_bin_edges= np.histogram(yPredTrain[yTrueTrain == 1] ,bins = nbins, density = True)
    prob_train_1_bin_edges = prob_train_1_bin_edges[:len(prob_train_1_bin_edges)-1]+ (prob_train_1_bin_edges[1]-prob_train_1_bin_edges[0])/2

    y_test_predict_LR = clf.predict_proba(yPredTest)[:,1]
    y_train_predict_LR = clf.predict_proba(yPredTrain)[:,1]

    prob_train_0_height_LR , prob_train_0_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 0], bins = nbins, density = True)
    prob_train_0_bin_edges_LR = prob_train_0_bin_edges_LR[:len(prob_train_0_bin_edges_LR)-1]+ (prob_train_0_bin_edges_LR[1]-prob_train_0_bin_edges_LR[0])/2
    prob_train_1_height_LR , prob_train_1_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 1], bins = nbins, density = True)
    prob_train_1_bin_edges_LR = prob_train_1_bin_edges_LR[:len(prob_train_1_bin_edges_LR)-1]+ (prob_train_1_bin_edges_LR[1]-prob_train_1_bin_edges_LR[0])/2
    
    plt.figure()
    fig, axs = plt.subplots(1,3, figsize = (10,5))
    axs[0].set_title("NN Output")
    axs[0].set_yscale("log")
    axs[0].hist(yPredTest[yTrueTest == 0],bins = nbins, density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[0].hist(yPredTest[yTrueTest == 1],bins = nbins, density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[0].plot(prob_train_0_bin_edges, prob_train_0_height, "b.", label = "Train (Label = 0)")
    axs[0].plot(prob_train_1_bin_edges, prob_train_1_height, "r.", label = "Train (Label = 1)")
    axs[0].grid()
    axs[0].set_ylabel("Normalized number of tracks")
    axs[0].legend(loc = "best")
    axs[1].set_title("LogReg Output")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Normalized number of tracks")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 0],bins = nbins,density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 1],bins = nbins,density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[1].plot(prob_train_0_bin_edges_LR ,prob_train_0_height_LR, "b.", label = "Train (Label = 0)")
    axs[1].plot(prob_train_1_bin_edges_LR ,prob_train_1_height_LR, "r.", label = "Train (Label = 1)")
    axs[1].grid()
    axs[1].legend(loc = "best")

    folder = 'plots'
    saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{saveName}_output.pdf")
    plt.close()

def eta_determination (yPredTest):
    if 