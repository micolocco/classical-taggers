import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from multiprocessing import Process, Manager, Pool

cimport numpy as np

from DecisionTree.TreeNode import TreeNode
cimport DecisionTree.TreeNode as TN

np.import_array()


'''
Decision Tree Implementation

Termination criteria:
- Maximum depth of the tree is reached
- Minimum number of samples required to split a node is not met
- Minimum number of samples required to be at a leaf node is not met
- Minimum weight fraction of the sum total of weights required to be at a leaf node is not met
- Minimum imimpurity decrease is not met

'''


class DecisionTree:
    

    def __init__(self, criterion, max_depth = None, min_samples_split = None, min_samples_leaf = None, min_weight_fraction_leaf = None, min_impurity_decrease = None, n_threads=1):
        self.max_depth = max_depth
        if max_depth is None:
            self.max_depth = np.inf
        self.min_samples_split = min_samples_split
        if min_samples_split is None:
            self.min_samples_split = 2
        self.min_samples_leaf = min_samples_leaf
        if min_samples_leaf is None:
            self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        if min_weight_fraction_leaf is None:
            self.min_weight_fraction_leaf = 0.0
        self.min_impurity_decrease = min_impurity_decrease
        if min_impurity_decrease is None:
            self.min_impurity_decrease = 0.0
        self.criterion = criterion
        self.root = None
        self.n_classes = None
        self.n_threads = n_threads

    def _get_optimal_split_of_column(self, X, y, N, weights, column,depth, node_index, return_dict):
        unique_values = np.unique(X[:, column])
        unique_values = np.sort(unique_values)


        return_dict[column] = (None, 0, np.inf)
        
        cuts = unique_values[:-1]+(unique_values[1:]-unique_values[:-1])/2
        for cut_value in cuts:
            node = TN.TreeNode(depth, node_index, self.criterion, y, weights, self.n_classes)
            node.cut_value = cut_value
            node.column = column

            if node.cut_valid(X, weights, self.max_depth, self.min_samples_split, self.min_samples_leaf, self.min_weight_fraction_leaf):
                node.calc_impurity(X, y, weights)
                node.left = TN.TreeNode(node.depth + 1, node.node_index * 2 + 1, self.criterion, y, weights, self.n_classes)
                node.right = TN.TreeNode(node.depth + 1, node.node_index * 2 + 2, self.criterion, y, weights, self.n_classes)

                impurity_decrease = node.impurity_decrease(X, y, N, weights)
                if impurity_decrease > return_dict[column][1] and impurity_decrease >= self.min_impurity_decrease:
                    return_dict[column] = (cut_value, impurity_decrease, node.impurity_if_leaf)
        
    
    def _get_optimal_split_parallel(self, X, y, N, weights, depth, node_index):
        manager = Manager()
        return_dict = manager.dict()

        pool = Pool(processes=self.n_threads)


        errors = []
        def error_callback(e):
            errors.append(e)
            traceback.print_exc()

        for column in range(X.shape[1]):
            pool.apply_async(self._get_optimal_split_of_column, args=(X, y, N, weights, column, depth, node_index, return_dict), error_callback=error_callback)
        pool.close()
        pool.join()

        if errors:
            raise Exception(f"Errors occurred in parallel processing: {errors}")

        best_column = None
        best_cut_value = None
        best_impurity = -np.inf
        best_impurity_decrease = 0


        for column, (cut_value, impurity_decrease, impurity_if_leaf) in return_dict.items():
            if impurity_decrease > best_impurity_decrease and impurity_decrease >= self.min_impurity_decrease:
                best_impurity_decrease = impurity_decrease
                best_column = column
                best_cut_value = cut_value
                best_impurity = impurity_if_leaf
        
        return best_column, best_cut_value, best_impurity


    def _build_tree(self, X, y, N, weights, depth, index):
        best_column, best_cut_value, best_impurity = self._get_optimal_split_parallel(X, y, N, weights, depth, index)


        node = TN.TreeNode(depth, index, self.criterion, y, weights, self.n_classes)

        if best_column is None:
            node.is_leaf = True
            node.calc_impurity(X, y, weights)
            return node
        


        node.column = best_column
        node.cut_value = best_cut_value
        node.impurity_if_leaf = best_impurity


        indices_left  = np.where(X[:, node.column] <= node.cut_value)[0]
        indices_right = np.where(X[:, node.column] > node.cut_value)[0]


        X_left = X[indices_left]
        y_left = y[indices_left]
        X_right = X[indices_right]
        y_right = y[indices_right]

        node.left = self._build_tree(X_left, y_left, N, weights[indices_left], depth + 1, index * 2 + 1)
        node.right = self._build_tree(X_right, y_right, N, weights[indices_right], depth + 1, index * 2 + 2)


        return node



    def _prune_tree(self, node, X, y, weights):
        if node.is_leaf:
            return

        # Notation is terrible but based on the formula for alpha effective in cost complexity pruning from sklearn documentation / The book
        # "Classification and Regression Trees" by Breiman, Friedman, Olshen, and Stone

        T = node.count_leaf_nodes()
        Rt = node.impurity_if_leaf
        RT_t = node.calc_impurity(X, y, weights)
        alpha_eff = (Rt - RT_t) / (T - 1)

        if alpha_eff < self.min_impurity_decrease:
            node.is_leaf = True
            node.left = None
            node.right = None
        else:
            indices_left  = np.where(X[:, node.column] <= node.cut_value)[0]
            indices_right = np.where(X[:, node.column] > node.cut_value)[0]
            X_left = X[indices_left]
            y_left = y[indices_left]
            X_right = X[indices_right]
            y_right = y[indices_right]
            self._prune_tree(node.left,  X_left,  y_left,  weights[ indices_left])
            self._prune_tree(node.right, X_right, y_right, weights[ indices_right])


    def fit(self, X, y, weights=None):
        if weights is None:
            norm_weights = np.ones(y.shape[0], dtype=np.float64) / y.shape[0]
        else:
            sum_weights = np.sum(weights)
            norm_weights = weights / sum_weights 
        self.n_classes = len(np.unique(y))
        N = np.sum(norm_weights)



        self.col_index_map = {i: col for i, col in enumerate(X.columns)}
        self.reverse_col_index_map = {col: i for i, col in enumerate(X.columns)}

        self.root = self._build_tree(X.values, y, N, norm_weights, depth=0, index=0)


        self._prune_tree(self.root, X.values, y, weights=norm_weights)

    def predict(self, X):
        return self.root.predict(X.values)

    def gini(self,
             np.ndarray[np.int32_t, ndim=1]     y,
             np.ndarray[np.float64_t, ndim=1] weights,
             int                              n_classes,
             int                              node_prediction):

        cdef Py_ssize_t i, n
        n = y.shape[0]

        if n == 0:
            return 0.0

        cdef double total_weight = 0.0
        cdef double[:] class_sums = np.zeros(n_classes, dtype=np.float64)

        for i in range(n):
            class_sums[y[i]] += weights[i]
            total_weight += weights[i]

        cdef double g = 0.0
        cdef double p

        for i in range(n_classes):
            if total_weight > 0:
                p = class_sums[i] / total_weight
                g += p - p * p

        return g

    

    def _node_info(self, node, dict_class_names=None):
        info = ''
        if not node.is_leaf:
            info +=  f"{self.col_index_map[node.column]} <= {node.cut_value}"

        info += f"\nimpurity={node.impurity_if_leaf:.5f}"
        info += f"\nsamples={node.num_samples_trained_on}"
        info += f"\nweight={node.weight_trained_on:.5f}"
        info += f"\nclass_counts={node.num_samples_per_class}"


        if dict_class_names is not None:
            prediction = dict_class_names.get(node.prediction, node.prediction)
        else:
            prediction = node.prediction
        info += f"\npred={prediction}"

        return info

    def _assign_tree_positions(self):
        """Assign x/y positions to each node so the whole tree can be drawn cleanly."""
        if self.root is None:
            return {}, 0

        positions = {}
        leaf_counter = [0]

        def _visit(node):
            if node is None:
                return None

            if node.is_leaf:
                x = leaf_counter[0]
                leaf_counter[0] += 1
            else:
                left_x = _visit(node.left)
                right_x = _visit(node.right)
                if left_x is None and right_x is None:
                    x = leaf_counter[0]
                    leaf_counter[0] += 1
                elif left_x is None:
                    x = right_x
                elif right_x is None:
                    x = left_x
                else:
                    x = 0.5 * (left_x + right_x)

            positions[node] = (x, -node.depth)
            return x

        _visit(self.root)
        return positions, max(1, leaf_counter[0])

    def plot_tree(self, figsize=None, node_fontsize=8, dict_class_names=None):
        """Plot the entire decision tree in a matplotlib figure.

        Parameters
        ----------
        figsize : tuple or None
            Optional `(width, height)` for the figure. If None, a size is chosen
            automatically from tree depth and number of leaves.
        node_fontsize : int
            Font size used in node labels.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Figure and axes containing the tree plot.
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() before plot_tree().")

        positions, n_leaves = self._assign_tree_positions()
        tree_depth = self.root.max_depth()

        if figsize is None:
            width = max(8, 1.6 * n_leaves)
            height = max(4, 1.6 * (tree_depth + 1))
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)

        def _draw(node, dict_class_names=None):
            if node is None:
                return

            x, y = positions[node]

            ax.text(
                x,
                y,
                self._node_info(node, dict_class_names),
                ha="center",
                va="center",
                fontsize=node_fontsize,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#f4f6f8" if node.is_leaf else "#dbeafe",
                    "edgecolor": "#334155",
                },
            )

            for child, edge_label in ((node.left, "True"), (node.right, "False")):
                if child is None:
                    continue

                cx, cy = positions[child]
                ax.plot([x, cx], [y, cy], color="#475569", linewidth=1.2)
                offset_direction = -1 if edge_label == "True" else 1
                ax.text(
                    0.5 * (x + cx) + 0.1 * offset_direction,
                    0.5 * (y + cy) + 0.1,
                    edge_label,
                    fontsize=max(6, node_fontsize - 1),
                    color="#334155",
                    ha="center",
                    va="center",
                )
                _draw(child, dict_class_names)

        _draw(self.root, dict_class_names)

        ax.set_xlim(-0.8, n_leaves - 0.2)
        ax.set_ylim(-(tree_depth + 0.8), 0.8)
        ax.axis("off")
        fig.tight_layout()
        return fig, ax
    

        