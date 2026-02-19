import torch
from torch import nn
import numpy as np 
import pandas as pd
import time
from numpy import vstack
from sklearn.metrics import accuracy_score
import yaml



from scripts.gradient_reversal.module import GradientReversal
import copy

class NeuralNetwork(nn.Module):

    def __init__(self, features, architecture, optimizer=torch.optim.Adam, optimizer_kwargs={}, 
                 seed=6, loss=nn.BCELoss(reduction='none'), repo_path="", create_network=True): 
        '''
        The loss function's reduce flag must be set to none to enable individual sample weighting.
        '''
        super().__init__()
        torch.manual_seed(seed) # needed to be sure the result is reproducible
        self.features = features
        self.criterion = loss
        if create_network:
            self.NN = self.create_network(architecture, repo_path=repo_path)
            self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)


    def create_network(self, architecture, repo_path=""):
        with open(f'{repo_path}/NNarchitectures/{architecture}.yaml', 'r') as file:
            architecture_config = yaml.safe_load(file)['architecture']
        # Set in_features for the first layer dynamically
        architecture_config[0]['params']['in_features'] = len(self.features)
        layers = []
        for layer in architecture_config:
            layer_type = layer['type']
            layer_params = layer['params']
            layer_class = getattr(nn, layer_type)
            layers.append(layer_class(**layer_params))
        return nn.Sequential(*layers)

    def forward(self, x): # from the input tensor x it gives the output tensor of the NN
        return self.NN(x).view(-1)

    def __str__(self):
        '''
        Print NN structure
        '''
        return str(self.NN)

    # train the model
    def train_model(self, train_dl, epoch,n_epochs, sample_weights=None):
        n_total_steps = len(train_dl)
        stepLoss = []
        self.train() #Sets model to training mode
        # enumerate mini batches

        for i, ((inputsTrain, targetsTrain), indices) in enumerate(train_dl):
            # Clear the gradients
            self.optimizer.zero_grad(set_to_none=True)
            # compute the model output
            yPredTrain = self(inputsTrain)
            if sample_weights is not None:
                batch_weights = sample_weights[indices]
            else:
                batch_weights = None
            training_loss = self.calc_loss(yPredTrain, targetsTrain, sample_weights=batch_weights)
            stepLoss.append(training_loss.tolist())
            if isinstance(self, NNDomainAdapted):
                training_loss = training_loss.sum() #Sum the losses for class and domain classifier
            training_loss.backward()
            # update model weights
            self.optimizer.step()
            # Calculate per batch loss
            #if (i+1) % 1000 == 0:
                #print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {training_loss.item():.4f}')
        return stepLoss
    
    def validate_model(self, validation_dl, sample_weights=None):
        self.eval() #Sets model to evaluation mode
        validationStep_loss = []
        for i, ((inputsVal, targetsVal), indices) in enumerate(validation_dl):
    
            # Forward pass
            yPredVal = self(inputsVal)
            if sample_weights is not None:
                batch_weights = sample_weights[indices]
            else:
                batch_weights = None
            with torch.no_grad():
                validation_loss = self.calc_loss(yPredVal, targetsVal, sample_weights=batch_weights)
            validationStep_loss.append(validation_loss.tolist())
        return validationStep_loss
    
    def BCELoss(self, yPred, target): #Testing purpose
        x = torch.log(yPred)
        y = torch.log(1 - yPred)

        return -torch.mul(x, target.float()) - torch.mul(y, (1 - target).float())

    def calc_loss(self, yPred, target, sample_weights = None):
        # if sample_weights is None:
        #     sample_weights = torch.ones(target.shape)


        loss = self.criterion(yPred.view(-1, 1), target.view(-1, 1)).view(-1)

        if sample_weights is not None:
            loss = torch.matmul(loss,sample_weights.float()) / torch.sum(sample_weights)


        return loss.mean()

    # Evaluate the model
    def evaluate_model(self, test_dl):
        self.eval()
        predictions, actuals = list(), list()
        for i, ((inputs, targets), _) in enumerate(test_dl):
            # evaluate the model on the test set
            with torch.no_grad():
                yPred = self(inputs)
            # retrieve numpy array
            yPred = yPred.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((-1, 1))

            # store
            predictions.append(yPred)
            actuals.append(actual)


        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        if isinstance(self, NNDomainAdapted):
            shape = (-1,2)
        else:
            shape = (-1,1)
        predictions, actuals = np.array(predictions).reshape(shape), np.array(actuals).reshape(shape)
        return np.array([predictions, actuals])

class NNDomainAdapted(NeuralNetwork):

    def __init__(self, features, architecture, optimizer=torch.optim.Adam, optimizer_kwargs={}, seed=6, 
                 loss=nn.BCELoss(reduction='none'), repo_path="", alpha= 1.0): 
        '''
        The loss function's reduce flag must be set to none to enable individual sample weighting.
        '''
        super().__init__(features, architecture, optimizer=optimizer, optimizer_kwargs={}, seed=seed, 
                         loss=loss, repo_path=repo_path, create_network = False)

        self.feature_extract, self.class_classifier, self.domain_classifier = self.create_network(architecture, repo_path=repo_path, alpha=alpha)
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)

    def create_network(self, architecture, repo_path="", alpha=1.0):
        with open(f'{repo_path}/NNarchitectures/{architecture}.yaml', 'r') as file:
            architecture_config = yaml.safe_load(file)['architecture']
        # Set in_features for the firs()t layer dynamically
        architecture_config[0]['params']['in_features'] = len(self.features)
        layers = []
        for layer in architecture_config:
            layer_type = layer['type']
            layer_params = layer['params']
            layer_class = getattr(nn, layer_type)
            layers.append(layer_class(**layer_params))

        num_layers = (len(layers)-2)/3 #-2 to remove final layer from the count /3 because each layer is composed of linear, activation and dropout layers
        
        feat_ex = nn.Sequential(*layers[:int(num_layers/2*3)])
        class_clf = nn.Sequential(*layers[int(num_layers/2*3):])



        domain_clf_layers = [GradientReversal(alpha = alpha)] +[copy.deepcopy(i) for i in  layers[int(num_layers/2*3):]] #TODO Alpha maybe hyperparameter?
        domain_clf = nn.Sequential(*domain_clf_layers)
        return feat_ex, class_clf, domain_clf

    def forward(self, x): # from the input tensor x it gives the output tensor of the NN
        features = self.feature_extract(x)
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(features)
        
        return torch.stack([class_pred, domain_pred], axis=1).view(-1,2)
    
    def calc_loss(self, yPred, target, sample_weights = None):

        class_loss = super().calc_loss(yPred[:,0][target[:,1] == 1], target[:,0][target[:,1] == 1], None) #Only train label classifier on domain 1 (MC) samples
        dom_loss = super().calc_loss(yPred[:,1], target[:,1], sample_weights)

        return torch.stack([class_loss, dom_loss])

    def __str__(self):
        '''
        Print NN structure
        '''

        return "Feature extractor:" +  str(self.feature_extract) + "\n Class classifier:" + str(self.class_classifier) + "\n Domain classifier:" + str(self.domain_classifier)

class EarlyStopper:

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if isinstance(validation_loss, np.ndarray):
            v_loss = validation_loss[0] # For domain adaptation use only class loss for early stopping
        else:
            v_loss = validation_loss
        if v_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = v_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

