import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from multiprocessing import Process, Manager, Pool

cimport numpy as np

from DecisionTree cimport TreeNode

np.import_array()



cdef class DecisionTree:
    cpdef gini(self,
             np.ndarray[np.int32_t, ndim=1]     y,
             np.ndarray[np.float64_t, ndim=1] weights,
             int                              n_classes,
             int                              node_prediction)
    
