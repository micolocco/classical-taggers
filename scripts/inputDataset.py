from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from pickle import dump
from pickle import load
import numpy as np
from torch import tensor
import torch

# Dataset definition
class inputDataset(Dataset):
    # load the dataset
    def __init__(self, df):
        if 'domain' in df.columns:
            self.y = df[['label', 'domain']].astype(np.float32).to_numpy().reshape(-1, 2)
            self.X = df.drop(columns=['label', 'domain']).astype(np.float32).to_numpy()
        else:
            self.y = df[['label']].astype(np.float32).to_numpy().reshape(-1, 1)
            self.X = df.drop(columns=['label']).astype(np.float32).to_numpy()


    # Length of the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, index):
        return [self.X[index], self.y[index]]

    def __getitems__(self, indices):
        X = self.X[indices]
        y = self.y[indices]

        return list(zip(X, y))  


    # Apply scaling
    def scale (self, test, scalerPath, transformerPath):
        if test:
            scaler = load(open(scalerPath, 'rb'))
            transformer =load(open(transformerPath, 'rb'))
            self.X = transformer.transform(scaler.transform(self.X))
            #self.X = scaler.transform(self.X)
        else:
            scaler = StandardScaler()
            transformer = PowerTransformer()
            self.X = transformer.fit_transform(scaler.fit_transform(self.X))

            dump(scaler, open(scalerPath, 'wb'))
            dump(transformer, open(transformerPath, 'wb'))
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
