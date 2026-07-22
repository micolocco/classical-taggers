import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from tqdm import tqdm
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from time import time
import datetime
import os


def tagging_hist(df, feature, n_bins=50, figsize=(10,6), yscale = 'linear', xscale = 'linear', axa=None):
    sort_feat = df[feature].sort_values()

    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'cyan', 12: 'black'}
    label = {0: "OSKaon", 1: "OSMuon", 2: "OSElectron", 3: "SSPion", 4: "SSProton", 5: "SSKaon", 12: "notSamePV"}

    min_val = sort_feat.iloc[int(len(sort_feat)*0.01)]
    max_val = sort_feat.iloc[int(len(sort_feat)*0.99)]

    bins = np.linspace(min_val, max_val, n_bins)

    if axa is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axa

    for particle in df['particle'].unique():
        frac =  1

        subset = df[df['particle'] == particle].sample(frac=frac, random_state=42)
        ax.hist(subset[feature], bins=bins, alpha=0.5, label=label[particle], color=colors[particle], density=True, histtype='step')
    
    ax.set_xlabel(feature)
    ax.set_ylabel(f'Density per {np.round((max_val - min_val)/n_bins, 2)}')
    ax.set_xlim(min_val, max_val)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)   
    if axa is None:
        ax.legend()
        plt.show()

def eval_feature(column, particle, num_cuts=100, use_tqdm=False):
    """
    Find the cut on `column` that minimizes the weighted Gini impurity.

    The split is evaluated using all particle classes on both sides of the cut:
    - `column > cut`
    - `column <= cut`

    Returns
    -------
    best_cut : float
        Cut value with the lowest impurity.
    best_impurity : float
        Weighted Gini impurity at that cut.
    """
    sorted_unique_values = np.sort(np.unique(column))
    low = sorted_unique_values[int(len(sorted_unique_values) * 0.01)]  # Exclude the lowest 1% of values
    high = sorted_unique_values[int(len(sorted_unique_values) * 0.99)]  # Exclude the highest 1% of values

    cuts = np.linspace(low, high, num=num_cuts)

    best_cut = None
    best_impurity = np.inf
    impurities = {}

    if use_tqdm:
        cuts = tqdm(cuts, desc="Evaluating cuts")

    for cut in cuts:
        passed = particle[column > cut]
        failed = particle[column <= cut]

        def gini(s):
            if len(s) == 0:
                return 0.0
            probs = s.value_counts(normalize=True)
            return 1.0 - np.sum(probs.values ** 2)

        gini_pass = gini(passed)
        gini_fail = gini(failed)

        total = len(particle)
        weighted_gini = (len(passed) / total) * gini_pass + (len(failed) / total) * gini_fail

        impurities[cut] = weighted_gini

        if weighted_gini < best_impurity:
            best_impurity = weighted_gini
            best_cut = cut

    

    return impurities, best_cut, best_impurity

def eval_feature_addition(used_features, available_features, X_train, y_train, X_test, num_cuts=100, use_tqdm=False):
    """
    Add one feature at a time, refit LDA, evaluate LD1, and return Gini impurity scores.
    """
    used_features = list(used_features)
    available_features = [f for f in available_features if f not in used_features]

    best_lda = None
    best_impurity = np.inf
    best_cut = None
    best_lda_obj = None

    impurity_scores = {}

    if use_tqdm:
        available_features = tqdm(available_features, desc="Evaluating feature additions")
    

    for feat in available_features:
        current_features = used_features + [feat]

        X_train_subset = X_train[current_features]
        X_test_subset = X_test[current_features]

        lda_local = LinearDiscriminantAnalysis()
        lda_local.fit(X_train_subset, y_train)

        X_test_lda = lda_local.transform(X_test_subset)
        ld1 = X_test_lda[:, 0] if X_test_lda.ndim > 1 else X_test_lda.ravel()

        _, cut, impurity = eval_feature(ld1, y_test, num_cuts=num_cuts)
        impurity_scores[feat] = impurity

        if impurity < best_impurity:
            best_impurity = impurity
            best_lda = X_test_lda
            best_cut = cut
            best_lda_obj = lda_local

    best_feature = min(impurity_scores, key=impurity_scores.get) if impurity_scores else None
    best_score = impurity_scores.get(best_feature, np.nan) if best_feature is not None else np.nan

    return impurity_scores, best_feature, best_score, best_cut, best_lda, best_lda_obj


def get_efficiencies(column, particle, cut):


    mapping = {0 : 'OSKaon_eff', 
               1 : 'OSMuon_eff', 
               2 : 'OSElectron_eff', 
               3 : 'SSPion_eff', 
               4 : 'SSProton_eff', 
               5 : 'SSKaon_eff', 
               12: 'notSamePV_eff'}

    efficiencies = {}
    for part in particle.unique():
        total = (particle == part).sum()
        selected = ((column > cut) & (particle == part)).sum()
        efficiency = selected / total if total > 0 else 0
        efficiencies[mapping[part]] = efficiency

    return efficiencies

features_added = ['B_Tr_T_cos_PhiDistance', 'B_Tr_T_PhiDistance', 'B_Tr_T_diff_z', 'B_Tr_T_DeltaR', 'diff_P', 'P_proj', 't', 'EVIP', 'B_Tr_T_absOWNPV_IP', 'B_Tr_T_EtaDistance', 'B_Tr_T_DeltaQ_Pion', 'B_Tr_T_DeltaQ_Muon', 'B_Tr_T_DeltaQ_Electron', 'B_Tr_T_DeltaQ_Proton', 'B_Tr_T_DeltaQ_Kaon', 'B_Tr_T_Signal_TagPart_PT', 'B_Tr_T_OWNPVIPSig', 'logEVIP', 'logP_proj', 'B_Tr_T_atanPT_PZ']
load_extra = ['EVENTNUMBER','RUNNUMBER']
features_noMC = [
    'B_OWNPV_X',
    'B_OWNPV_Y',
    'B_OWNPV_Z',
    'B_ENDV_X',
    'B_ENDV_Y',
    'B_ENDV_Z',
    'B_ENERGY',
    'B_ETA',
    'B_M',
    'B_P',
    'B_PHI',
    'B_PT',
    'B_PX',
    'B_PY',
    'B_PZ',
    'B_nPVs',
    'B_nTracks',
    'B_Tr_T_TRACKISLONG',
    'B_Tr_T_OWNPVIP',
    'B_Tr_T_OWNPVIPCHI2',
    'B_Tr_T_Charge',
    'B_Tr_T_ISMUON',
    'B_Tr_T_ENERGY',
    'B_Tr_T_Eta',
    'B_Tr_T_MINIP',
    'B_Tr_T_MINIPChi2',
    'B_Tr_T_P',
    'B_Tr_T_PT',
    'B_Tr_T_PIDK',
    'B_Tr_T_PIDe',
    'B_Tr_T_PIDmu',
    'B_Tr_T_PIDP',
    'B_Tr_T_PROBNN_GHOST',
    'B_Tr_T_PROBNN_E',
    'B_Tr_T_PROBNN_K',
    'B_Tr_T_PROBNN_P',
    'B_Tr_T_PROBNN_MU',
    'B_Tr_T_PROBNN_PI',
    #'B_Tr_T_firstX',
    #'B_Tr_T_firstY',
    #'B_Tr_T_firstZ',
    #'B_Tr_T_firstTX',
    #'B_Tr_T_firstTY',
    #'B_Tr_T_OWNPV_X',
    #'B_Tr_T_OWNPV_XERR',
    #'B_Tr_T_OWNPV_Y',
    #'B_Tr_T_OWNPV_YERR',
    #'B_Tr_T_OWNPV_Z',
    #'B_Tr_T_OWNPV_ZERR',
    #'B_Tr_T_Phi',
    #'B_Tr_T_M',
    'B_Tr_T_CHI2DOF',
    'B_Tr_T_GHOSTPROB',
    'B_Tr_T_PX',
    'B_Tr_T_PY',
    'B_Tr_T_PZ',
    'B_Tr_T_X',
    'B_Tr_T_Y',
    'B_Tr_T_Z',
    #'B_Tr_T_OBJECT_KEY',
    'B_Tr_T_IPChi2BVTX',
    'B_Tr_T_IPBVTX',]
features_added += ['B_Tr_T_endSV_Z']
features = features_noMC + features_added

start_time = time()

drop_features = {
    'OS_to_SS' : 
        [
            'B_Tr_T_PIDK',
            'B_Tr_T_PIDe',
            'B_Tr_T_PIDmu',
            'B_Tr_T_PIDP',
            'B_Tr_T_PROBNN_GHOST',
            'B_Tr_T_PROBNN_E',
            'B_Tr_T_PROBNN_K',
            'B_Tr_T_PROBNN_P',
            'B_Tr_T_PROBNN_MU',
            'B_Tr_T_PROBNN_PI',
            'B_Tr_T_GHOSTPROB',
            'B_Tr_T_ISMUON'
        ]

}


path = "/ceph/users/togasa/FlavourTagging/MC/DT_outputs/allBKGCAT_notSamePV_noOSP_SSK_balanced/DT_trainingset.root"

outpath = "/ceph/users/togasa/collected_pdfs/lda_investigation"
os.makedirs(outpath, exist_ok=True)


with uproot.open(path) as f:
    tree = f["DecayTree;1"]
    _df_raw = tree.arrays(library="pd")


LDA_configs = {
    # 'OSK_to_all' : [[0], [1,2,3,4,5,12]],
    # 'OSMu_to_all': [[1], [1,2,3,4,5,12]],
    # 'OSE_to_all' : [[2], [1,2,3,4,5,12]],
    # 'SSPi_to_all': [[3], [0,1,2,4,5,12]],
    # 'SSP_to_all' : [[4], [0,1,2,3,5,12]],
    # 'SSK_to_all' : [[5], [0,1,2,3,4,12]],

    # 'OSK_to_target' : [[0], [1,2,3,4,5]],
    # 'OSMu_to_target': [[1], [1,2,3,4,5]],
    # 'OSE_to_target' : [[2], [1,2,3,4,5]],
    # 'SSPi_to_target': [[3], [0,1,2,4,5]],
    # 'SSP_to_target' : [[4], [0,1,2,3,5]],
    # 'SSK_to_target' : [[5], [0,1,2,3,4]],

    # 'OSK_to_SSPi': [[0], [3]],
    # 'OSK_to_SSP' : [[0], [4]],
    # 'OSK_to_SSK' : [[0], [5]],
    # 'SSPi_to_SSP': [[3], [4]],
    # 'SSPi_to_SSK': [[3], [5]],
    # 'SSP_to_SSK' : [[4], [5]],
    
    'OS_to_SS' : [[0,1,2], [3,4,5]],

    # 'Not_samePV' : [[0,1,2,3,4,5], [12]],
}

for LDA_config in tqdm(LDA_configs.keys()):
    out_lda_path = f"{outpath}/{LDA_config}"

    os.makedirs(f"{outpath}/{LDA_config}", exist_ok=True)

    used_particles = LDA_configs[LDA_config]

    particles_to_keep = used_particles[0]+used_particles[1]

    _df = _df_raw.loc[_df_raw['particle'].isin(particles_to_keep)]

    min_count = _df["particle"].value_counts().min()

    df = (
        _df.groupby("particle", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    df["particle"].value_counts()

    df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)


    use_features = features.copy() + ['particle']

    if LDA_config in drop_features.keys():
        features = [feat for feat in features if feat not in drop_features[LDA_config]]


    df_DT = df[features + ['particle']].sample(frac=1, random_state=42).reset_index(drop=True)

    feature_funcs = ['','','','','',
                    '','_abslog','_abslog','','_abslog',
                    '','_abslog','_abslog','_abslog','_abslog',
                    '','_abssqrt','','_abslog','_abslog',
                    '','','_abslog','','_abslog',
                    '_abslog','_abslog','_abslog','','',
                    '','','_abslog','','',
                    '_abslog','','','_abslog','_abslog',
                    '','','_abslog','','_abslog',
                    '','_abslog','_abslog','','',
                    '_abslog','','_abslog','_abslog','_abslog',
                    '_abssqrt','_abslog','_abssqrt','_abslog','_abslog',
                    '_abslog','_abslog','_abslog','_abssqrt','_abslog',
                    '','','_abslog','_abslog']



    features_expanded = features.copy()
    for feature, func in zip(features, feature_funcs):
        if func == '_abslog':
            df_DT[f'{feature}_abslog'] = np.log(np.abs(df_DT[feature])+1e-6)
            features_expanded += [f'{feature}_abslog']
        elif func == '_abssqrt':
            df_DT[f'{feature}_abssqrt'] = np.sqrt(np.abs(df_DT[feature]))
            features_expanded += [f'{feature}_abssqrt']
        elif func == '':
            continue
        else:
            raise ValueError(f"Unknown function '{func}' for feature '{feature}'")
        
    mapping = {part: 0 for part in particles_to_keep}
    for part in used_particles[0]:
        mapping[part] = 1

    df_DT['target'] = df_DT['particle'].map(mapping)

    split_idx = int(len(df_DT) * 0.5)

    target_col = 'target'

    train_df = df_DT.iloc[:split_idx]
    test_df = df_DT.iloc[split_idx:]

    X_train = train_df[features_expanded]
    y_train = train_df[target_col]

    X_test = test_df[features_expanded]
    y_test = test_df[target_col]
    y_part = test_df["particle"]


    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    lda_cols = [f"LD{i+1}" for i in range(df_DT[target_col].nunique()-1)]

    df_test_transformed = pd.DataFrame(X_test_lda, columns=lda_cols, index=test_df.index)
    df_test_transformed["target"] = y_test.values
    df_test_transformed["particle"] = y_part.values

    pd.merge(df_test_transformed, X_test, left_index=True, right_index=True)
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'cyan', 12: 'black'}
    label = {0: "OSKaon", 1: "OSMuon", 2: "OSElectron", 3: "SSPion", 4: "SSProton", 5: "SSKaon", 12: "notSamePV"}



    df_test_transformed['color'] = df_test_transformed['particle'].map(colors)

    tagging_hist(df_test_transformed, 'LD1', n_bins=100, figsize=(10,6), yscale = 'linear')
    plt.savefig(f"{out_lda_path}/LD1_baseline_histogram.pdf")

    baseline_impurities, best_cut, baseline_gini = eval_feature(df_test_transformed['LD1'], df_test_transformed['target'], num_cuts=100)


    results = pd.DataFrame([{"added_feature": "/", "gini": baseline_gini, "gini_ratio": 1.0, **get_efficiencies(df_test_transformed["LD1"], df_test_transformed["particle"], best_cut)}])


    current_features = []
    available_features = list(X_train.columns)
    target_gini_ratio = 1.05 
    best_lda_obj = None

    gini_ratio = np.inf
    step = 0

    while gini_ratio > target_gini_ratio and available_features:
        addition_scores, best_feature, best_score, best_cut, best_lda, best_lda_obj = eval_feature_addition(
            used_features=current_features,
            available_features=available_features,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            num_cuts=100,
            use_tqdm=False
        )

        current_features.append(best_feature)
        available_features.remove(best_feature)
        current_gini = best_score
        gini_ratio = current_gini / baseline_gini
        step += 1

        results.loc[len(results)] = {
            "added_feature": best_feature,
            "gini": current_gini,
            "gini_ratio": gini_ratio,
            **get_efficiencies(best_lda[:, 0], y_part, best_cut)
        }  
    
    results.to_csv(f"{out_lda_path}/feature_addition_results.csv", index=False)

    it_add_no_baseline = results[results["added_feature"] != "/"].copy()

    plt.figure(figsize=(max(8, len(it_add_no_baseline)*0.25), 6))

    plt.bar(it_add_no_baseline["added_feature"], it_add_no_baseline["gini"], color="steelblue", alpha=0.7, label='Gini after addition')

    plt.hlines(y=baseline_gini,                   xmin=-0.5, xmax=len(it_add_no_baseline)-0.5, color='black', linestyle='--', label='Baseline Gini')
    plt.hlines(y=baseline_gini*target_gini_ratio, xmin=-0.5, xmax=len(it_add_no_baseline)-0.5, color='red', linestyle='--', label='Target Gini')

    plt.xticks(rotation=45)

    plt.xlabel("added feature")
    plt.xlim(-0.5, len(it_add_no_baseline)-0.5)
    plt.ylim(baseline_gini*0.75, max(it_add_no_baseline["gini"])*1.05)

    plt.ylabel("Gini")
    plt.tight_layout()
    plt.yscale('linear')
    plt.legend()
    plt.savefig(f"{out_lda_path}/gini_progression.pdf")

    lda_df = pd.DataFrame({
        "LDA" : best_lda.reshape(-1),
        "particle" : df_test_transformed["particle"].values,
        "target" : df_test_transformed["target"].values,
    })

    tagging_hist(lda_df, 'LDA', n_bins=100, figsize=(10,6), yscale = 'linear')
    plt.savefig(f"{out_lda_path}/LD1_minimal_histogram.pdf")

    # Create a heatmap of the LDA scalings matrix

    scalings_matrix = best_lda_obj.scalings_
    feature_labels = results["added_feature"].values[1:]  # Exclude the baseline row

    plt.figure(figsize=(len(lda_cols)/1.5+3, len(feature_labels)/3+3))
    sns.heatmap(
        scalings_matrix,
        xticklabels=lda_cols,
        yticklabels=feature_labels,
        cmap='RdBu_r',
        center=0,
        fmt='.2f',
        cbar_kws={'label': 'Scaling Coefficient'}
    )

    for i in range(scalings_matrix.shape[0]):
        for j in range(scalings_matrix.shape[1]):
            plt.text(
                j + 0.5,  # X position
                i + 0.5,  # Y position
                f"{scalings_matrix[i, j]:.2f}",  # Text content
                ha='center', 
                va='center', 
                color='black', 
                fontsize=10, 
        )


    plt.title('LDA Scalings Matrix Heatmap')
    plt.xlabel('Linear Discriminants')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{out_lda_path}/minimal_lda_scalings_heatmap.pdf', dpi=300)

    X_test[lda_cols] = best_lda

    corr = X_test.corr()[features][-len(lda_cols):]

    plt.figure(figsize=(corr.shape[1] * 0.6+2, corr.shape[0] * 0.6+2))
    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        cbar_kws={'label': 'Correlation'},
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation Matrix: LDA Variables vs Input Features')
    plt.xlabel('Input Features')
    plt.ylabel('LDA Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(
                j + 0.5,  # X position
                i + 0.5,  # Y position
                f"{corr.iloc[i, j]:.2f}",  # Text content
                ha='center', 
                va='center', 
                color='black', 
                fontsize=10, 
        )

    plt.savefig(f'{out_lda_path}/minimal_lda_correlation_heatmap.pdf', dpi=300)

    print(f'Particle set: {LDA_config} completed in {np.round(time()-start_time, 2)} seconds at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f"Script ended in {np.round(time()-start_time, 2)} seconds at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
