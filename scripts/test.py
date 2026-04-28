from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm 
from time import time
import uproot

import sys
from pathlib import Path

# from scripts.DecisionTree.DecisionTree import DecisionTree
# from DecisionTree import DecisionTree
# from DecisionTree import gini


def test_decision_tree(df, kargs, use_weights=False, debug_mode=False, num_threads=4):
    X = df.drop(columns=['particle'])
    y = pd.Series(df['particle'])

    if use_weights:
        X['weights'] = np.random.rand(len(X))  # Add random weights for testing



    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X, y

    if use_weights:
        weights_train = X_train['weights'].values
        # weights_test = X_test['weights'].values
        X_train = X_train.drop(columns=['weights'])
        # X_test = X_test.drop(columns=['weights'])
    else:
        weights_train = None
        weights_test = None

    kargs_with_threads = kargs.copy()
    kargs_with_threads['n_threads'] = num_threads
    # print(f"Custom Decision Tree training with {num_threads} threads started", flush=True)

    # start_time = time()
    # DT = DecisionTree(criterion=0, **kargs_with_threads)

    # # DT.fit(X_train, np.array(y_train, dtype=np.int32), weights=weights_train)
    # end_time = time()
    # print(f"Custom Decision Tree training time ({num_threads} threads): {end_time - start_time:.4f} seconds", flush=True)

    # if debug_mode:
    #     fig, _ = DT.plot_tree(dict_class_names={i: name for i, name in enumerate(df.target_names)})
    #     fig.savefig("/ceph/users/togasa/FlavourTagging/MC/DT_outputs/testing/balanced/custom_tree.pdf")

    # y_pred = DT.predict(X_test)

    # print("Accuracy:", accuracy_score(y_test, y_pred, sample_weight=weights_test), flush=True)

    # #Compare with sklearn

    sklearn_DT = DecisionTreeClassifier(criterion = "gini", **kargs)

    start_time = time()
    sklearn_DT.fit(X_train, y_train)
    end_time = time()
    print(f"Sklearn Decision Tree training time: {end_time - start_time:.4f} seconds", flush=True)

    # y_pred_sklearn = sklearn_DT.predict(X_test)

    #Make figure of sklearn tree

    # print(f"Sklearn Accuracy: {accuracy_score(y_test, y_pred_sklearn, sample_weight=weights_test)}\n", flush=True)
    if debug_mode:
        plt.figure(figsize=(12,8))
        plot_tree(sklearn_DT, feature_names=df.drop(columns=['particle']).columns, class_names = ["OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon", "otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP", "Others", "notSamePV"], filled=True, precision=5)
        plt.savefig("/ceph/users/togasa/FlavourTagging/MC/DT_outputs/testing/balanced/sklearn_tree.pdf")

        # print(f'Custom Decision Tree Predictions: {y_pred}', flush=True)
        # print(f'Sklearn Decision Tree Predictions: {y_pred_sklearn}', flush=True)

        # print(y_pred == y_pred_sklearn, flush=True)

        pd.set_option('display.max_columns', 50)
        # print(f"All predictions same: {all(X_test[y_pred != y_pred_sklearn])}", flush=True)

    # return all(y_pred == y_pred_sklearn)

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # df_iris = load_iris()


    # df_breast = load_breast_cancer()

    with uproot.open("/ceph/users/togasa/FlavourTagging/MC/DT_outputs/testing/balanced/DT_trainingset_10k.root") as _f:
        df = _f['DecayTree;1'].arrays(["B_Tr_T_cos_PhiDistance", "B_Tr_T_PhiDistance", "B_Tr_T_diff_z", "B_Tr_T_DeltaR", "diff_P", "P_proj", "t", "EVIP", "B_Tr_T_absOWNPV_IP", "B_Tr_T_EtaDistance", "B_Tr_T_DeltaQ_Pion", "B_Tr_T_DeltaQ_Muon", "B_Tr_T_DeltaQ_Electron", "B_Tr_T_DeltaQ_Proton", "B_Tr_T_DeltaQ_Kaon", "B_Tr_T_Signal_TagPart_PT", "B_Tr_T_OWNPVIPSig", "logEVIP", "logP_proj", "B_Tr_T_atanPT_PZ", "B_OWNPV_X", "B_OWNPV_Y", "B_OWNPV_Z", "B_ENDV_X", "B_ENDV_Y", "B_ENDV_Z", "B_ENERGY", "B_ETA", "B_M", "B_P", "B_PHI", "B_PT", "B_PX", "B_PY", "B_PZ", "B_nPVs", "B_nTracks", "B_Tr_T_TRACKISLONG", "B_Tr_T_OWNPVIP", "B_Tr_T_OWNPVIPCHI2", "B_Tr_T_Charge", "B_Tr_T_ISMUON", "B_Tr_T_ENERGY", "B_Tr_T_Eta", "B_Tr_T_MINIP", "B_Tr_T_MINIPChi2", "B_Tr_T_P", "B_Tr_T_PT", "B_Tr_T_PIDK", "B_Tr_T_PIDe", "B_Tr_T_PIDmu", "B_Tr_T_PIDP", "B_Tr_T_PROBNN_GHOST", "B_Tr_T_PROBNN_E", "B_Tr_T_PROBNN_K", "B_Tr_T_PROBNN_P", "B_Tr_T_PROBNN_MU", "B_Tr_T_PROBNN_PI", "B_Tr_T_CHI2DOF", "B_Tr_T_GHOSTPROB", "B_Tr_T_PX", "B_Tr_T_PY", "B_Tr_T_PZ", "B_Tr_T_X", "B_Tr_T_Y", "B_Tr_T_Z", "B_Tr_T_IPChi2BVTX", "B_Tr_T_IPBVTX", 'particle'], library="pd")


    # df = df[:1_000_000]

    # depths = [2, 3, 4, 5, 6]
    # min_samples_splits = [2, 5, 80]
    # min_samples_leafs = [1, 10, 50]
    # min_weight_fraction_leafs = [0.0, 0.01, 0.5]
    # min_impurity_decreases = [0.0, 0.01, 0.2]

    # combinations = list(itertools.product(depths, min_samples_splits, min_samples_leafs, min_weight_fraction_leafs, min_impurity_decreases))
    # for dep, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease in tqdm(combinations):
    #     print(f"Testing with max_depth={dep}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, min_weight_fraction_leaf={min_weight_fraction_leaf}, min_impurity_decrease={min_impurity_decrease}", flush=True)
    #     kargs = {'max_depth':               dep,
    #             'min_samples_split':        min_samples_split,
    #             'min_samples_leaf':         min_samples_leaf,
    #             'min_weight_fraction_leaf': min_weight_fraction_leaf,
    #             #  'max_leaf_nodes':           None,
    #             'min_impurity_decrease':    min_impurity_decrease,
    #             #  'criterion':                gini,
    #     }
    #     is_same_iris = test_decision_tree(df_iris, kargs, False)
    #     is_same_breast = test_decision_tree(df_breast, kargs, False)
    #     if not is_same_iris:
    #         print(f"Difference found in Iris dataset with parameters: {kargs}", flush=True)
    #     if not is_same_breast:
    #         print(f"Difference found in Breast Cancer dataset with parameters: {kargs}", flush=True)


    # kargs = {'max_depth':               2,
    #         'min_samples_split':        2,
    #         'min_samples_leaf':         50,
    #         'min_weight_fraction_leaf': 0.01,
    #         #  'max_leaf_nodes':           None,
    #         'min_impurity_decrease':    0.2,
    #         #  'criterion':                gini,
    # }


    #expand df_breast by adding random noise to every feature and concatenating it to the original dataset, repeat this process 100 times
    # for _ in range(2):
    #     noise = np.random.normal(0, 0.1, df_breast.data.shape)
    #     noisy_data = df_breast.data + noise
    #     df_breast.data = np.concatenate((df_breast.data, noisy_data), axis=0)
    #     df_breast.target = np.concatenate((df_breast.target, df_breast.target), axis=0)
    #     print(df_breast.data.shape, flush=True)

    kargs =  {'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 100, 'min_weight_fraction_leaf': 0.4, 'min_impurity_decrease': 0.0}
    kargs =  {'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'min_impurity_decrease': 0.009, 'class_weight':'balanced'}
    # kargs =  {'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'min_impurity_decrease': 0.009,}


    # test_decision_tree(df_breast, kargs, True, True)
    for num_threads in [8,]:

        test_decision_tree(df, kargs, True, True, num_threads=num_threads)


