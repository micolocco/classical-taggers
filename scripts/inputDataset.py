from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pickle import load
import numpy as np

# Dataset definition
class inputDataset(Dataset):
    # load the dataset
    def __init__(self, df, scalerPath, test = True):
        
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.scaler = StandardScaler()
        if test:
            self.scaler = load(open(scalerPath, 'rb'))
            self.X = self.scaler.transform(self.X)

        else:
            self.X = self.scaler.fit_transform(self.X)
            dump(self.scaler, open(scalerPath, 'wb'))
        # ensure output data is floats
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Length of the dataset
    def __len__(self):
        return len(self.X)

    # Get a row at an index
    def __getitem__(self, index):
        return [self.X[index], self.y[index]]
    
    # Get indexes for train and test rows
    def get_splits(self, n_train = 0.8):
        # determine sizes
        train_size = int(n_train * len(self.X))
        val_size = len(self.X) - train_size
        if val_size < 1:
            print("Pre-selection cuts are too tight")
            exit()
        # Split dataset into train, validation and test
        return random_split(self, np.array([train_size, val_size]))