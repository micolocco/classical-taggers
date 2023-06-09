import torch
from torch import nn
import numpy as np 
import pandas as pd
import time
from numpy import vstack
from sklearn.metrics import accuracy_score

class NeuralNetwork(nn.Module):
    torch.manual_seed(42) # needed to be sure the result is reproducable

    def __init__(self, modelName, n_features, optimizer=torch.optim.Adam, optimizer_kwargs={}, loss=nn.BCELoss(), train_batch_size = 32, test_batch_size = 1024 ): #originally 75, for testing purpose changed to 200
        super().__init__()
        self.modelName = modelName
        self.n_input = n_features # can be change in PyTorchTraining.py or use the default which can be seen above
        self.criterion = loss
        self.NN = nn.Sequential(
            nn.Linear(self.n_input, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
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
            if (i+1) % 1 == 0:
                print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {training_loss.item():.4f}')
            return stepLoss, training_loss

    def validate_model(self, validation_dl):
        self.eval()
        for i, (inputsVal, targetsVal) in enumerate(validation_dl):
            validationStep_loss = []
            # Forward pass
            yPredVal = self(inputsVal)
            validation_loss = self.criterion(yPredVal, targetsVal)
            validationStep_loss.append(validation_loss.item())
        return validationStep_loss, validation_loss

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
            yPred = yPred.round()
            # store
            predictions.append(yPred)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
    
  
'''
    def fit(self,x,y,x_test,y_test): # training with back propagation
        if self.isScaled == False:
            self.scaler.fit(x)
            self.isScaled = True
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        x_test = torch.tensor(x_test).float()
        y_test = torch.tensor(y_test).float()
        self.train()
        y_pred = self(x)
        loss = self.loss(y_pred, y.reshape(y_pred.shape))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        y_pred_test = self(x_test)
        loss_train = self.loss(y_pred,y.reshape(y_pred.shape)).item()
        loss_test = self.loss(y_pred_test,y_test.reshape(y_pred_test.shape)).item()
        return loss_train, loss_test
    
    def get_scaledx(self,x):
        if self.isScaled:
            return(self.scaler.transform(x))# takes a pandas dataframe
'''
