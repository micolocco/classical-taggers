import numpy as np
import numba as nb


# @nb.njit(nb.float64(nb.int32[:], nb.float64[:], nb.int32, nb.int32))
def gini(y:       np.ndarray[np.int32],
                weights: np.ndarray[np.float64],
                n_classes: int,
                node_prediction: int):
    n = y.shape[0]

    if n == 0:
        return np.inf
    
    one_hot_weighted = np.eye(n_classes)[y] * weights[:, np.newaxis]
    class_ratios = np.sum(one_hot_weighted, axis=0) / np.sum(weights)

    return np.sum(class_ratios * (1 - class_ratios))