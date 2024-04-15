import torch
from torch import nn
import numpy as np 
import pandas as pd
import time
from numpy import vstack
from sklearn.metrics import accuracy_score
import configParameters as config

class NeuralNetwork(nn.Module):

    torch.manual_seed(config.seed) # needed to be sure the result is reproducible

    def __init__(self, modelName, features, optimizer=torch.optim.Adam, optimizer_kwargs={}, loss=nn.BCELoss(), train_batch_size = 32, test_batch_size = 1024 ): #originally 75, for testing purpose changed to 200
        super().__init__()
        self.modelName = modelName
        self.features = features
        self.criterion = loss
        self.NN = nn.Sequential(
            nn.Linear(len(self.features), 3),
           # nn.Dropout(0.5),
            nn.ELU(), 
            nn.Linear(3, 3),
            nn.ELU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
           # nn.Linear(len(self.features), 32),
           # nn.Dropout(0.5),
           # nn.ELU(), #ELU, ReLU # activation_function=nn.ELU()
           # nn.Linear(32, 64),
           # nn.Dropout(0.5),
           # nn.ELU(),
           # nn.Linear(64, 32),
           # nn.Dropout(0.5),
           # nn.ELU(),
           # nn.Linear(32, 1),
           # nn.Sigmoid()
        )
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def forward(self, x): # from the input tensor x it gives the output tensor of the NN
        return self.NN(x)

    # train the model
    def train_model(self, train_dl, epoch,n_epochs):
        n_total_steps = len(train_dl)
        stepLoss = []
        self.train()
        # enumerate mini batches
        for i, (inputsTrain, targetsTrain) in enumerate(train_dl):
            # Clear the gradients
            self.optimizer.zero_grad()
            # compute the model output
            yPredTrain = self(inputsTrain)
            training_loss = self.criterion(yPredTrain, targetsTrain)
            training_loss.backward()
            # update model weights
            self.optimizer.step()
            # Calculate per batch loss
            stepLoss.append(training_loss.item())
            #if (i+1) % 1000 == 0:
                #print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {training_loss.item():.4f}')
        return stepLoss

    def validate_model(self, validation_dl):
        self.eval()
        validationStep_loss = []
        for i, (inputsVal, targetsVal) in enumerate(validation_dl):
            # Forward pass
            yPredVal = self(inputsVal)
            validation_loss = self.criterion(yPredVal, targetsVal)
            validationStep_loss.append(validation_loss.item())
        return validationStep_loss


    # Evaluate the model
    def evaluate_model(self, test_dl):
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yPred = self(inputs)
            # retrieve numpy array
            yPred = yPred.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            #yPred = yPred.round()
            # store
            predictions.append(yPred)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        return predictions, actuals
  
class EarlyStopper:

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

