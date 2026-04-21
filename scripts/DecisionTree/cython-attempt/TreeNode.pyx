import numpy as np
cimport numpy as np


cdef class TreeNode:
    def __init__(self, int depth, int node_index,
                 object criterion,
                 np.ndarray[np.int32_t, ndim=1] y,
                 np.ndarray[np.double_t, ndim=1] weights,
                 int n_classes):
        cdef np.ndarray counts
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


    cpdef predict(self, np.ndarray X):
        cdef Py_ssize_t n = X.shape[0]

        if self.is_leaf:
            return np.full(n, self.prediction, dtype=np.float64)

        cdef np.ndarray indices_left = np.where(X[:, self.column] <= self.cut_value)[0]
        cdef np.ndarray indices_right = np.where(X[:, self.column] > self.cut_value)[0]

        cdef np.ndarray y_pred = np.empty(n, dtype=np.float64)

        y_pred[indices_left] = self.left.predict(X[indices_left])
        y_pred[indices_right] = self.right.predict(X[indices_right])

        return y_pred


    cpdef double calc_impurity(self,
                      np.ndarray X,
                      np.ndarray y,
                      np.ndarray weights):

        if self.left is None and self.right is None:
            self.impurity_if_leaf = self.criterion(y, weights, self.n_classes, self.prediction)
            return self.impurity_if_leaf

        cdef np.ndarray indices_left  = np.where(X[:, self.column] <= self.cut_value)[0]
        cdef np.ndarray indices_right = np.where(X[:, self.column] > self.cut_value)[0]

        cdef double Nt_l = np.sum(weights[indices_left])
        cdef double Nt_r = np.sum(weights[indices_right])
        cdef double Nt = Nt_l + Nt_r

        cdef double imp_l = self.left.calc_impurity(
            X[indices_left], y[indices_left], weights[indices_left]
        )
        cdef double imp_r = self.right.calc_impurity(
            X[indices_right], y[indices_right], weights[indices_right]
        )

        return (Nt_l / Nt) * imp_l + (Nt_r / Nt) * imp_r


    cpdef double impurity_decrease(self,
                          np.ndarray X,
                          np.ndarray y,
                          double N,
                          np.ndarray weights):
        cdef double Nt = np.sum(weights)
        return Nt/N * (self.impurity_if_leaf - self.calc_impurity(X, y, weights))



    cpdef bint cut_valid(self,
                  np.ndarray X,
                  np.ndarray weights,
                  int max_depth,
                  int min_samples_split,
                  int min_samples_leaf,
                  double min_weight_fraction_leaf):

        if self.depth >= max_depth:
            return False

        if X.shape[0] <= min_samples_split:
            return False


        cdef np.ndarray col = X[:, self.column]

        if np.sum(col <= self.cut_value) <= min_samples_leaf:
            return False
        if np.sum(col > self.cut_value) <= min_samples_leaf:
            return False

        if np.sum(weights[col <= self.cut_value]) <= min_weight_fraction_leaf:
            return False
        if np.sum(weights[col > self.cut_value]) <= min_weight_fraction_leaf:
            return False


        return True


    cpdef int count_leaf_nodes(self):
        if self.is_leaf:
            return 1
        return self.left.count_leaf_nodes() + self.right.count_leaf_nodes()

    cpdef int max_depth(self):
        if self.is_leaf:
            return self.depth
        return max(self.left.max_depth(), self.right.max_depth())
    