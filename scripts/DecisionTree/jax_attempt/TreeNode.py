import numpy as np
import numba as nb
import pytreeclass as ptc
from .Criterion import gini
import jax
# @nb.experimental.jitclass([
#     ('is_leaf', nb.types.boolean),
#     ('left', nb.optional(nb.types.pyobject)),
#     ('right', nb.optional(nb.types.pyobject)),
#     ('column', nb.int32),
#     ('cut_value', nb.float64),
#     ('weight_trained_on', nb.float64),

#     ('depth', nb.int32),
#     ('node_index', nb.int32),
#     ('criterion', nb.types.pyobject),

#     ('n_classes', nb.int32),
#     ('num_samples_trained_on', nb.int32),
#     ('num_samples_per_class', nb.int32[:]),
#     ('prediction', nb.int32),
#     ('impurity_if_leaf', nb.float64)
# ])

class TreeNode(ptc.TreeClass):
    def __init__(self, 
                 depth: int,
                 node_index: int,
                 criterion: int,
                 y: np.ndarray[np.int32],
                 weights: np.ndarray[np.float64],
                 n_classes: int):
        self.is_leaf = False
        self.left = None
        self.right = None
        self.column = -1
        self.cut_value = np.inf
        self.weight_trained_on = np.sum(weights)

        self.depth = depth
        self.node_index = node_index
        self.criterion = criterion

        self.n_classes = n_classes
        self.num_samples_trained_on = y.shape[0]


        if y.shape[0] > 0:
            counts = np.bincount(y, minlength=n_classes)
            self.num_samples_per_class = counts
            self.prediction = int(np.argmax(counts))
        else:
            self.num_samples_per_class = np.zeros(n_classes, dtype=np.int32)
            self.prediction = -1

        self.impurity_if_leaf = -np.inf


    def predict(self, X: np.ndarray[np.float64]):

        if self.is_leaf:
            return np.full(X.shape[0], self.prediction, dtype=np.float64)

        indices_left = np.where(X[:, self.column] <= self.cut_value)[0]
        indices_right = np.where(X[:, self.column] > self.cut_value)[0]

        y_pred = np.empty(X.shape[0], dtype=np.int32)

        y_pred[indices_left] = self.left.predict(X[indices_left])
        y_pred[indices_right] = self.right.predict(X[indices_right])

        return y_pred


    def calc_impurity(self,
                      X: np.ndarray[np.float64],
                      y: np.ndarray[np.int32],
                      weights: np.ndarray[np.float64]):

        if self.left is None and self.right is None:
            if self.criterion == 0:
                self.impurity_if_leaf = gini(y, weights, self.n_classes, self.prediction)
            else:
                raise NotImplementedError(f"Criterion {self.criterion} not implemented")

            # self.impurity_if_leaf = self.criterion(y, weights, self.n_classes, self.prediction)
            return self.impurity_if_leaf

        indices_left  = np.where(X[:, self.column] <= self.cut_value)[0]
        indices_right = np.where(X[:, self.column] > self.cut_value)[0]

        Nt_l = np.sum(weights[indices_left])
        Nt_r = np.sum(weights[indices_right])
        Nt = Nt_l + Nt_r

        imp_l = self.left.calc_impurity(
            X[indices_left], y[indices_left], weights[indices_left]
        )
        imp_r = self.right.calc_impurity(
            X[indices_right], y[indices_right], weights[indices_right]
        )

        return (Nt_l / Nt) * imp_l + (Nt_r / Nt) * imp_r


    def impurity_decrease(self,
                          X: np.ndarray[np.float64],
                          y: np.ndarray[np.int32],
                          N: int,
                          weights: np.ndarray[np.float64]):
        return np.sum(weights)/N * (self.impurity_if_leaf - self.calc_impurity(X, y, weights))



    def cut_valid(self,
                    X: np.ndarray[np.float64],
                    weights: np.ndarray[np.float64],
                    max_depth: int,
                    min_samples_split: int,
                    min_samples_leaf: int,
                    min_weight_fraction_leaf: float):
        if self.depth >= max_depth:
            return False

        if X.shape[0] <= min_samples_split:
            return False


        col = X[:, self.column]

        if np.sum(col <= self.cut_value) <= min_samples_leaf:
            return False
        if np.sum(col > self.cut_value) <= min_samples_leaf:
            return False
        
        indices_left  = np.where(col <= self.cut_value)[0]
        indices_right = np.where(col > self.cut_value)[0]

        if np.sum(weights[indices_left]) <= min_weight_fraction_leaf:
            return False
        if np.sum(weights[indices_right]) <= min_weight_fraction_leaf:
            return False


        return True


    def count_leaf_nodes(self):
        if self.is_leaf:
            return 1
        return self.left.count_leaf_nodes() + self.right.count_leaf_nodes()

    def max_depth(self):
        if self.is_leaf:
            return self.depth
        return max(self.left.max_depth(), self.right.max_depth())
    