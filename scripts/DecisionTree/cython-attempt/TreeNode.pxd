cimport numpy as np

cdef class TreeNode:
    cdef public bint is_leaf
    cdef public object left, right
    cdef public int column, depth, node_index, n_classes
    cdef public double cut_value, impurity_if_leaf
    cdef public int prediction
    cdef public int num_samples_trained_on
    cdef public double weight_trained_on

    cdef public np.ndarray num_samples_per_class
    cdef public object criterion

    # methods (declare ALL you want fast access to)
    cpdef predict(self, np.ndarray X)

    cpdef double calc_impurity(self,
                              np.ndarray X,
                              np.ndarray y,
                              np.ndarray weights)

    cpdef double impurity_decrease(self,
                                  np.ndarray X,
                                  np.ndarray y,
                                  double N,
                                  np.ndarray weights)

    cpdef bint cut_valid(self,
                         np.ndarray X,
                         np.ndarray weights,
                         int max_depth,
                         int min_samples_split,
                         int min_samples_leaf,
                         double min_weight_fraction_leaf)

    cpdef int count_leaf_nodes(self)

    cpdef int max_depth(self)