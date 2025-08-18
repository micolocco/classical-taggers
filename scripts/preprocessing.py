import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from torch.utils.data import DataLoader

class StandardScalerLayer(nn.Module):
    """
    A PyTorch layer that mimics sklearn's StandardScaler with frozen mean & scale.
    """
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.scale


class PowerTransformerLayer(nn.Module):
    """
    A PyTorch layer that mimics sklearn's PowerTransformer with frozen lambdas.
    Only implements the Yeo-Johnson method.
    """

    def __init__(self, lambdas):
        super().__init__()
        lambdas = torch.tensor(lambdas, dtype=torch.float32)
        self.register_buffer("lambdas", lambdas)
    def forward(self, x):
        out = torch.empty_like(x)
        n_features = x.shape[1]

        for j in range(n_features):
            lam = self.lambdas[j]
            col = x[:, j]
            pos = col >= 0

            if lam != 0:
                out[pos, j] = ((col[pos] + 1).pow(lam) - 1) / lam
            else:
                out[pos, j] = torch.log(col[pos] + 1)

            if lam != 2:
                out[~pos, j] = -(((-col[~pos] + 1).pow(2 - lam) - 1) / (2 - lam))
            else:
                out[~pos, j] = -torch.log(-col[~pos] + 1)

        return out

def fit_freeze_preprocessing(train_dl):
    """
    Fits StandardScaler and PowerTransformer on training data,
    returns a nn.Sequential module with frozen parameters.
    """
    # Collect full training data into a numpy array
    X_list = []
    for batch_x, _ in train_dl:
        X_list.append(batch_x.numpy())
    X_train = np.vstack(X_list)

    # Fit sklearn StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit sklearn PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    pt.fit(X_scaled)

    # Convert fitted sklearn transformers to frozen PyTorch layers
    scaler_layer = StandardScalerLayer(mean=scaler.mean_, scale=scaler.scale_)
    pt_layer = PowerTransformerLayer(lambdas=pt.lambdas_)

    # Return as a single preprocessing module
    preprocess_module = nn.Sequential(scaler_layer, pt_layer)
    return preprocess_module