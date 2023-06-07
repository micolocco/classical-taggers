import torch
from torch import nn
import numpy as np 
import pandas as pd

class NeuralNetwork(nn.Module):
    torch.manual_seed(42) # needed to be sure the result is reproducable

    def __init__(self, n_features, optimizer=torch.optim.Adam, optimizer_kwargs={}, loss=nn.BCELoss(), train_batch_size = 32, test_batch_size = 1024, n_epochs = 100):
        super().__init__()
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
        self.n_epochs = n_epochs

    def forward(self, x): # from the input tensor x it gives the output tensor of the NN
        return self.NN(x)

    # train the model
    def train_model(self, train_dl):
        # enumerate epochs
        for epoch in range(self.n_epochs):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                yhat = self(inputs)
                # calculate loss
                loss = self.criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                self.optimizer.step()
  
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
