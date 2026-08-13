import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from tqdm import tqdm
import os

def tagging_hist(ypred, particles, n_bins=50, figsize=(10,6), yscale = 'linear', xscale = 'linear', axa=None, parts = [0, 1, 2, 3, 4, 5, 12], linestyle='solid'):
    df = pd.DataFrame({'feature': ypred, 'particle': particles})



    sort_feat = df['feature'].copy().sort_values()

    sort_feat.dropna(inplace=True)
    sort_feat = sort_feat[~np.isinf(sort_feat)]

    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'cyan', 12: 'black'}
    label = {0: "OSKaon", 1: "OSMuon", 2: "OSElectron", 3: "SSPion", 4: "SSProton", 5: "SSKaon", 12: "notSamePV"}

    min_val =  sort_feat.iloc[int(len(sort_feat)*0.01)]
    max_val =  sort_feat.iloc[int(len(sort_feat)*0.99)]

    bins = np.linspace(min_val, max_val, n_bins)

    if axa is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = axa

    for particle in df['particle'].unique():
        if particle in parts:
            frac =  1


            subset = df[df['particle'] == particle].sample(frac=frac, random_state=42)
            ax.hist(subset['feature'], bins=bins, alpha=0.5, label=label[particle], color=colors[particle], density=True, histtype='step', linestyle=linestyle)
    
    ax.set_xlabel('feature')
    ax.set_ylabel(f'Density per {np.round((max_val - min_val)/n_bins, 2)}')
    ax.set_xlim(min_val, max_val)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)   
    if axa is None:
        ax.legend()
        # plt.show()

def read_files(paths, features, treename='DecayTree;1'):
    additional_features = ['B_Tr_T_absID', 'B_Tr_T_Origin_Flag', 'B_Tr_T_MC_MOTHER_ID']

    df = pd.DataFrame(columns=features+additional_features)

    for f in paths:
        decay = os.path.basename(os.path.dirname(f))
        with uproot.open("{}".format(f)) as _f:
            _df = _f[treename].arrays(features + additional_features, library="pd")
            _df['decay'] = decay

            df = pd.concat([df, _df], ignore_index = True)

    condition_particle_pairs = [
        ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),                                        0),
        ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),                                        1),
        ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22), 2),
        ((df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay != "Bs2DsPi"),              3),
        ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay != "Bs2DsPi"),              4),
        ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1) & (df.decay == "Bs2DsPi"),              5),
        
        ((df.B_Tr_T_Origin_Flag == 100), 12),
    ]
    # condition_particle_pairs = [
    #     ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 2),                                        0),
    #     ((df.B_Tr_T_absID ==   13) & (df.B_Tr_T_Origin_Flag == 2),                                        1),
    #     ((df.B_Tr_T_absID ==   11) & (df.B_Tr_T_Origin_Flag == 2) & (abs(df.B_Tr_T_MC_MOTHER_ID) !=  22), 2),
    #     ((df.B_Tr_T_absID ==  211) & (df.B_Tr_T_Origin_Flag == 1),                                        3),
    #     ((df.B_Tr_T_absID == 2212) & (df.B_Tr_T_Origin_Flag == 1),                                        4),
    #     ((df.B_Tr_T_absID ==  321) & (df.B_Tr_T_Origin_Flag == 1),                                        5),
        
    #     ((df.B_Tr_T_Origin_Flag == 100), 12),
    # ]
    df.drop(columns=['decay'], inplace=True)

    conditions = [pair[0] for pair in condition_particle_pairs]
    particle_type = [pair[1] for pair in condition_particle_pairs]

    df['particle'] = np.select(conditions, particle_type, default=np.nan)

    df.drop(columns=additional_features, inplace=True)
    df.dropna(subset=['particle'], inplace=True)



    return df

paths = [
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bs2DsPi/00266991_00000001_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bu2JpsiK/00266989_00000001_1.mc.root",
    "/ceph/users/togasa/FlavourTagging/MC/NTuples/1_added_features/Bd2JpsiKst/00267659_00000001_1.mc.root"
]




processed_features = [
]
total_features = [
    'B_ENDV_X', 
    'B_ENDV_Y', 
    'B_ENDV_Z', 
    'B_OWNPV_X', 
    'B_OWNPV_Y', 
    'B_OWNPV_Z', 
    'B_CHI2VXNDOF', 
    'B_MIN_OWNPV_IPCHI2', 
    'B_ETA', 
    'B_PHI', 
    'B_M', 
    'B_P', 
    'B_PT', 
    'B_PX',
    'B_PY', 
    'B_PZ', 
    'B_ENERGY',
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
    'B_Tr_T_firstZ', 
    'B_Tr_T_OWNPV_X', 
    'B_Tr_T_OWNPV_Y', 
    'B_Tr_T_OWNPV_Z', 
    'B_Tr_T_Phi', 
    'B_Tr_T_M', 
    'B_Tr_T_CHI2DOF', 
    'B_Tr_T_GHOSTPROB', 
    'B_Tr_T_PX', 
    'B_Tr_T_PY', 
    'B_Tr_T_PZ', 
    'B_Tr_T_X', 
    'B_Tr_T_Y',  
    'B_Tr_T_Z', 
    'B_Tr_T_IPChi2BVTX', 
    'B_Tr_T_IPBVTX', 
    'B_Tr_T_PhiDistance', 
    'B_Tr_T_Signal_TagPart_PT', 
    'B_Tr_T_cos_PhiDistance', 
    'B_Tr_T_EtaDistance', 
    'B_Tr_T_DeltaR', 
    'B_Tr_T_DeltaQ_Pion', 
    'B_Tr_T_DeltaQ_Muon', 
    'B_Tr_T_DeltaQ_Electron', 
    'B_Tr_T_DeltaQ_Proton', 
    'B_Tr_T_DeltaQ_Kaon', 
    'B_Tr_T_OWNPVIPSig', 
    'B_Tr_T_absOWNPV_IP', 
    'B_TAU', 
    'B_TAUERR', 
    'B_Tr_T_endSV_Z', 
    'diff_P', 
    'P_proj', 
    't', 
    'EVIP', 
    'logEVIP', 
    'logP_proj', 
    'B_Tr_T_atanPT_PZ', 
    'B_Tr_T_DeltaP_X', 
    'B_Tr_T_DeltaP_Y', 
    'B_Tr_T_DeltaP_Z', 
    'B_Tr_T_DeltaP_X_abs', 
    'B_Tr_T_DeltaP_Y_abs', 
    'B_Tr_T_DeltaP_Z_abs', 
    'B_Tr_T_DeltaP', 
    'B_Tr_T_DeltaP_T', 
    'B_Tr_T_Theta', 
    'B_Theta', 
    'B_Tr_T_ThetaDistance', 
    'B_Tr_T_diff_x', 
    'B_Tr_T_diff_y', 
    'B_Tr_T_diff_z', 
    'B_Tr_T_diff_rho', 
    'B_Tr_T_diff_R', 
    'B_Tr_T_AngleSep', 
    'B_Tr_T_AngleSep_diff_z', 
    'B_Tr_T_AngleSep_diff_rho', 
    'B_Tr_T_AngleSep_diff_R', 
    'B_Tr_T_PhiDist_diff_z', 
    'B_Tr_T_PhiDist_diff_rho', 
    'B_Tr_T_PhiDist_diff_R', 
    'B_Tr_T_ThetaDist_diff_z', 
    'B_Tr_T_ThetaDist_diff_rho', 
    'B_Tr_T_ThetaDist_diff_R', 
    'B_Tr_T_ProbNN_EoverMu', 
    'B_Tr_T_ProbNN_EoverPi', 
    'B_Tr_T_ProbNN_EoverK', 
    'B_Tr_T_ProbNN_EoverP', 
    'B_Tr_T_ProbNN_MuoverE', 
    'B_Tr_T_ProbNN_MuoverPi', 
    'B_Tr_T_ProbNN_MuoverK', 
    'B_Tr_T_ProbNN_MuoverP', 
    'B_Tr_T_ProbNN_PioverE', 
    'B_Tr_T_ProbNN_PioverMu', 
    'B_Tr_T_ProbNN_PioverK', 
    'B_Tr_T_ProbNN_PioverP', 
    'B_Tr_T_ProbNN_KoverE', 
    'B_Tr_T_ProbNN_KoverMu', 
    'B_Tr_T_ProbNN_KoverPi', 
    'B_Tr_T_ProbNN_KoverP', 
    'B_Tr_T_ProbNN_PoverE', 
    'B_Tr_T_ProbNN_PoverMu', 
    'B_Tr_T_ProbNN_PoverPi', 
    'B_Tr_T_ProbNN_PoverK', 
    'B_Tr_T_ProbNN_PioverHadron', 
    'B_Tr_T_ProbNN_KoverHadron', 
    'B_Tr_T_ProbNN_PoverHadron', 
    'B_Tr_T_IPBVTX_IPChi2BVTX', 
    'B_Tr_T_MINIPChi2_IPChi2BVTX', 
    'B_Tr_T_EoverP', 
    'B_Tr_T_EoverPT', 
    'B_Tr_T_ET', 


    'B_Tr_T_Rho',
    'B_Tr_T_R',
    'B_Tr_T_firstRho',
    'B_Tr_T_firstR',
    'B_Tr_T_cos_AngleSep_diff_z',
    'B_Tr_T_cos_AngleSep_diff_rho',
    'B_Tr_T_cos_AngleSep_diff_R',
    'B_Tr_T_sin_AngleSep_diff_z',
    'B_Tr_T_sin_AngleSep_diff_rho',
    'B_Tr_T_sin_AngleSep_diff_R',
    'B_Tr_T_cosPhiDist_diff_z',
    'B_Tr_T_cosPhiDist_diff_rho',
    'B_Tr_T_cosPhiDist_diff_R',
]


feature_chunks = np.array_split(total_features, 10)


for feature_chunk in tqdm(feature_chunks):
    df = read_files(paths, features=feature_chunk.tolist())
    for feature in feature_chunk:
        df[feature] = df[feature].astype(float)

        os.makedirs(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/linear/", exist_ok=True)
        os.makedirs(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/ylog/", exist_ok=True)
        os.makedirs(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/abslog/", exist_ok=True)



        tagging_hist(df[feature], df['particle'], n_bins=50, figsize=(10,6), yscale = 'linear', xscale = 'linear', axa=None, parts = [0, 1, 2, 3, 4, 5, 12], linestyle='solid')
        plt.savefig(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/linear/{feature}.png")
        tagging_hist(df[feature], df['particle'], n_bins=50, figsize=(10,6), yscale = 'log',    xscale = 'linear', axa=None, parts = [0, 1, 2, 3, 4, 5, 12], linestyle='solid')
        plt.savefig(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/ylog/{feature}.png")
        tagging_hist(np.log(np.abs(df[feature].values)), df['particle'], n_bins=50, figsize=(10,6), yscale = 'linear', xscale = 'linear', axa=None, parts = [0, 1, 2, 3, 4, 5, 12], linestyle='solid')
        plt.savefig(f"/ceph/users/togasa/collected_pdfs/DT_feature_candidates/abslog/{feature}.png")