import numpy as np 
import sys 
import pandas as pd 
import os
import time
import uproot
import inputDataset
from torch.utils.data import DataLoader


# Prepare the dataset
def prepare_data(inputPath, loading_variables, scalerPath, train_batch_size = 32, test_batch_size = 1024):
    # Load the dataset
    dataset = inputDataset(inputPath, loading_variables, scalerPath)
    # Splitting
    train, validation, test = dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, train_batch_size = train_batch_size, shuffle=True)
    validation_dl = DataLoader(validation, batch_size = test_batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size=test_batch_size, shuffle=False)
    return train_dl, validation_dl, test_dl

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat