import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

class StandardScalerLayer(nn.Module):
    """
    A PyTorch layer that mimics sklearn's StandardScaler with frozen mean & scale.
    Works with any input shape, scaling along the last dimension.
    """
    def __init__(self, mean, scale):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, x):
        # ensure broadcasting only along the last dimension
        return (x - self.mean.view((1,) * (x.ndim - 1) + (-1,))) / \
               self.scale.view((1,) * (x.ndim - 1) + (-1,))


class PowerTransformerLayer(nn.Module):
    """
    A PyTorch layer that mimics sklearn's PowerTransformer with frozen lambdas.
    Yeo-Johnson only. Batch-safe for any input shape (..., features).
    """
    def __init__(self, lambdas):
        super().__init__()
        lambdas = torch.as_tensor(lambdas, dtype=torch.float32)
        self.register_buffer("lambdas", lambdas)

    def forward(self, x):
        # reshape lambdas for broadcasting to last dim
        lam_shape = (1,) * (x.ndim - 1) + (-1,)
        lambdas = self.lambdas.view(lam_shape)

        col_is_pos = x >= 0

        # positive branch
        pos_out = torch.where(
            lambdas != 0,
            ((x + 1).pow(lambdas) - 1) / lambdas,
            torch.log(x + 1)
        )

        # negative branch
        neg_out = torch.where(
            lambdas != 2,
            -(((-x + 1).pow(2 - lambdas) - 1) / (2 - lambdas)),
            -torch.log(-x + 1)
        )

        # combine
        return torch.where(col_is_pos, pos_out, neg_out)

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