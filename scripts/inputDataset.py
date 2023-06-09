from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import time
import uproot

# Dataset definition
class inputDataset(Dataset):
    # load the dataset
    def __init__(self, inputPath, features, scalerPath):
        start = time.time()
        print(f"Reading {inputPath} file")
        df = uproot.open(inputPath).arrays(features + ['label'], library = "pd" )
        print(f"Finished reading in {round(-start+ time.time() , 2)}s")
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        dump(self.scaler, scalerPath, compress=True)
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
    def get_splits(self, n_train = 0.8, n_validation = 0.1):
        # determine sizes
        train_size = round(n_train * len(self.X))
        validation_size = round(n_validation * len(self.X))
        test_size = len(self.X) - (train_size + validation_size)
        # Split dataset into train, validation and test
        return random_split(self, [train_size, validation_size, test_size])