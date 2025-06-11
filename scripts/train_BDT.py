import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

import uproot
import os
import psutil
import argparse
import datetime    
import pickle

from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


#Bu2JpsiK classifier from sin2beta ananote
# B_Vtx_Chi2NDOF                     -> B_CHI2VXNDOF
# B_MINIPCHI2                        -> B_MIN_OWNPV_IPCHI2
# B_LOKI_ETA                         -> B_ETA
# B_FitJpsiConst_chi2_flat           -> B_DTF_PV_Jpsi_CHI2
# B_FitJpsiConst_J_psi_1S_IP_flat    -> Jpsi_OWNPV_IP
# B_FitJpsiConst_J_psi_1S_P0_IP_flat -> muplus_OWNPV_IP
# B_FitJpsiConst_J_psi_1S_P1_IP_flat -> muminus_OWNPV_IP
# K_LOKI_ETA                         -> hplus_ETA
# K_MINIP                            -> hplus_MINIP
# K_IP_OWNPV                         -> hplus_OWNPV_IP

vars_by_decay = {'Bu2JpsiK': ['B_CHI2VXNDOF',
                              'B_MIN_OWNPV_IPCHI2',
                              'B_ETA',
                              'B_DTF_PV_Jpsi_CHI2',
                              'Jpsi_OWNPV_IP',
                              'muplus_OWNPV_IP',
                              'muminus_OWNPV_IP',
                              'hplus_ETA',
                              'hplus_MINIP',
                              'hplus_OWNPV_IP',],
}

def read_files(files, vars, treename, only_upper, massname):
    df = pd.DataFrame(columns=vars)


    for i, f in enumerate(files):
        print(f"Reading input file {i+1}/{len(files)}: {f}", flush=True)
        print(f'Total RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MiB')

        id = os.path.basename(f)[:-5]
        if id[-7:-2] == '.data':
            id = id[:-7]
        else:
            id = id[:-3]

        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(vars+ ['RUNNUMBER', 'EVENTNUMBER'], library="pd")
        
        _df.dropna(inplace = True)
        _df["event_entry"] = id + "_" + _df["RUNNUMBER"].astype(str) + "_" + _df["EVENTNUMBER"].astype(str)
        _df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
        if only_upper: #Only take upper mass sideband from data as example of combinatorial background
            _df = _df[_df[massname] > 5350]


        df = pd.concat([df, _df], ignore_index = True)

    return df


def plot_by_label(df, massname, outpath, filename, xlim=(5200, 5600), bins = 40):
    bins = np.linspace(xlim[0], xlim[1], bins+1)

    plt.figure(figsize=(8, 6))
    for label, group in df.groupby('label'):
        plt.hist(group[massname], bins=bins, alpha=0.5, label=f'Label {label}')
    plt.xlabel(f'{massname} in MeV')
    plt.ylabel('counts per $10$MeV')
    plt.legend()
    plt.title(f'Histogram of {massname} split by label')
    plt.savefig(f'{outpath}/{filename}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the BDT for the event selection of the specified decay',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--real_data', help='Files of real data for the combinatorial background', nargs='+')
    parser.add_argument('--mc_data', help='Files of simulated data as signal target', nargs='+')
    parser.add_argument('--target_path', help='Name of the output dir', type=str, default='../test')
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree;1')
    parser.add_argument('--decay_type', help='Event decay', type=str)
    parser.add_argument('--massname', help='Name of the invariant mass variable', type=str)
    parser.add_argument('--num_threads', help='Number of threads used for training', default=45, type = int) 

    

    cfg = parser.parse_args()
    pprint(cfg)
    seed = 45
    
    data_files = cfg.real_data
    mc_files = cfg.mc_data

    training_vars = vars_by_decay[cfg.decay_type]
    vars_to_load = training_vars + [cfg.massname]

    print(f'Reading of data files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(data_files)} files.", flush=True)
    data = read_files(data_files, vars_to_load, cfg.treename, True, cfg.massname)
    data['label'] = 0
    print(f'Reading of data files ends {datetime.datetime.now().strftime("%H:%M:%S")}')

    print(f'Reading of MC files begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    print(f"Reading a total of {len(mc_files)} files.", flush=True)
    MC = read_files(mc_files, vars_to_load, cfg.treename, False, cfg.massname)
    MC['label'] = 1
    print(f'Reading of MC files ends {datetime.datetime.now().strftime("%H:%M:%S")}')



    df = pd.concat([data, MC])
    df= df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(df['label'].to_numpy())
    print(df['label'].unique())

    print(f'num_rows: {len(df["event_entry"])} unique events: {df["event_entry"].nunique()}')

    plot_by_label(df, cfg.massname, cfg.target_path, 'before_classifier')


    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=[cfg.massname, 'label', 'event_entry']).to_numpy(), df['label'].to_numpy(), test_size=0.2, random_state=seed, stratify= df['label'].to_numpy()
    )
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        learning_rate=0.01,
        max_depth=6,
        seed=seed,
        booster='gbtree',
        n_estimators=500,
        early_stopping_rounds=20,
        n_jobs=cfg.num_threads,  # Use cfg.num_threads for parallelism
        use_label_encoder=False,
        verbosity=1,
    )

    print(f'Training begins {datetime.datetime.now().strftime("%H:%M:%S")}')
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )
    print(f'Training ends {datetime.datetime.now().strftime("%H:%M:%S")}')
    del X_train, y_train

    # params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'tree_method': 'hist',
    #     'learning_rate': 0.01,
    #     'max_depth': 6,
    #     'seed': seed,
    #     'booster': 'gbtree',
    # }

    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dval = xgb.DMatrix(X_val, label=y_val)
    # bst = xgb.train(
    #     params,
    #     dtrain,
    #     num_boost_round=500,
    #     evals=[(dtrain, 'train'), (dval, 'val')],
    #     early_stopping_rounds=20,
    #     verbose_eval=10
    # )



    
    y_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"Val ROC AUC: {auc:.4f}")


    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='best')
    plt.savefig(os.path.join(cfg.target_path, 'roc_curve.pdf'))
    plt.close()


    # Plot loss
    results = model.evals_result()
    plt.figure()
    plt.plot(results['validation_0']['logloss'], label='Train Loss')
    plt.plot(results['validation_1']['logloss'], label='Validation Loss')
    plt.xlabel('Boosting Round')
    plt.ylabel('Log Loss')
    plt.title('Training and Val Loss')
    plt.legend()
    plt.savefig(os.path.join(cfg.target_path, 'loss_curve.pdf'))
    plt.close()

    # dmatrix_all = xgb.DMatrix(df.drop(columns=[cfg.massname, 'label', 'event_entry']))
    # Find BDT cut value that reduces label 1 count by 5%

    
    tot_signal = y_val.sum()

    cut = 1.0
    while cut >= 0:
        signal_after_cut = (y_val[y_pred>cut]).sum()
        if signal_after_cut >= 0.95*tot_signal:
            break
        cut -= 0.001
        
    cut = round(cut, 3)
    print(f"BDT cut value for 5% reduction in label 1: {cut}")

    df['bdt_output'] = model.predict(df.drop(columns=[cfg.massname, 'label', 'event_entry']).to_numpy())
    df = df[df['bdt_output'] > cut]

    plot_by_label(df, cfg.massname, cfg.target_path, 'after_classifier')



    save_data = {
        'model': model,
        'cut': cut
    }

    with open(os.path.join(cfg.target_path, 'bdt_model.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Model saved {datetime.datetime.now().strftime("%H:%M:%S")}')
