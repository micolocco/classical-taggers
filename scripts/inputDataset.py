from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from pickle import dump
from pickle import load
import numpy as np

# Dataset definition
class inputDataset(Dataset):
    # load the dataset
    def __init__(self, df):
        
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # ensure output data is floats
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Length of the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, index):
        return [self.X[index], self.y[index]]
    
    # Apply scaling
    def scale (self, test, scalerPath, transformerPath):
        if test:
            scaler = load(open(scalerPath, 'rb'))
            transformer =load(open(transformerPath, 'rb'))
            self.X = transformer.transform(scaler.transform(self.X))
        
        else:
            scaler = StandardScaler()
            transformer = PowerTransformer()
            self.X = transformer.fit_transform(scaler.fit_transform(self.X))
            dump(scaler, open(scalerPath, 'wb'))
            dump(transformer, open(transformerPath, 'wb'))
        
