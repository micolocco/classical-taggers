import pandas as pd
import uproot
import numpy as np
import os
import argparse
import datetime
from time import time
from tqdm import tqdm
import awkward as ak
from pprint import pprint
import yaml
from collections import OrderedDict

data_vars_translation = {
    "B_Tr_T_zfirst": "B_Tr_T_firstZ",
}

def get_mass_label(decayType):

    if "Jpsi" in decayType:
        return f"B_DTF_PV_Jpsi_MASS"
    elif "Ds" in decayType:
        return f"B_DTF_PV_Ds_MASS"
    else:
        raise ValueError(f"Decay type {decayType} not recognized for mass label assignment")

def DeltaQ(df, Mass):
    E = np.sqrt(Mass**2 + df[f'B_Tr_T_PX']**2 + df[f'B_Tr_T_PY']**2 + df[f'B_Tr_T_PZ']**2)
    return np.sqrt(
        (E + df[f'B_ENERGY'])**2 - (
            (df[f'B_Tr_T_PX'] + df[f'B_PX'])**2 +
            (df[f'B_Tr_T_PY'] + df[f'B_PY'])**2 +
            (df[f'B_Tr_T_PZ'] + df[f'B_PZ'])**2
        )) - df[f'B_M'] - Mass

# Phi distance definition from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/src/Utils/TaggingHelpers.cpp?ref_type=heads#L43
def dPhi(df):
    cos_Tr_T_Phi = np.cos(df[f'B_Tr_T_Phi'])
    sin_Tr_T_Phi = np.sin(df[f'B_Tr_T_Phi'])
    cos_Phi = np.cos(df[f'B_PHI'])
    sin_Phi = np.sin(df[f'B_PHI'])

    x_arctan = cos_Tr_T_Phi * sin_Phi - cos_Phi * sin_Tr_T_Phi
    y_arctan = cos_Tr_T_Phi * cos_Phi + sin_Phi * sin_Tr_T_Phi
    df[f'B_Tr_T_PhiDistance'] = np.arctan2(x_arctan, y_arctan)
    return df

def E_T(df, Mass):
    return np.sqrt(Mass**2 + df[f'B_Tr_T_PT']**2)

def process_chunk(df, evtType):
    df = df.copy()


    df = dPhi(df)
    df.loc[:,f'B_Tr_T_Signal_TagPart_PT'] = np.sqrt((df[f'B_PX'] + df[f'B_Tr_T_PX'])**2 + (df[f'B_PY'] + df[f'B_Tr_T_PY'])**2)
    df.loc[:,f'B_Tr_T_cos_PhiDistance'] = np.cos(df[f'B_Tr_T_PhiDistance'])
    df.loc[:,f'B_Tr_T_EtaDistance'] = np.abs(df[f'B_ETA'] - df[f'B_Tr_T_Eta'])
    df.loc[:,f'B_Tr_T_DeltaR'] = (df[f'B_ETA'] - df[f'B_Tr_T_Eta'])**2 + df[f'B_Tr_T_PhiDistance']**2
    df.loc[:,f'B_Tr_T_DeltaQ_Pion'] = DeltaQ(df, 139.5706)
    df.loc[:,f'B_Tr_T_DeltaQ_Muon'] = DeltaQ(df, 105.65837)
    df.loc[:,f'B_Tr_T_DeltaQ_Electron'] = DeltaQ(df, 0.51100)
    df.loc[:,f'B_Tr_T_DeltaQ_Proton'] = DeltaQ(df, 938.27208)
    df.loc[:,f'B_Tr_T_DeltaQ_Kaon'] = DeltaQ(df, 493.677)
    df.loc[:,f'B_Tr_T_OWNPVIPSig'] = np.sqrt(df[f'B_Tr_T_OWNPVIPCHI2'])
    df.loc[:,f'B_Tr_T_absOWNPV_IP'] = np.abs(df[f'B_Tr_T_OWNPVIP'])
    df.loc[:,f'B_TAU'] = df[get_mass_label(evtType).replace("MASS", "CTAU")]/0.29979 #Convert from mm to ps using speed of light in mm/ps
    df.loc[:,f'B_TAUERR'] = df[get_mass_label(evtType).replace("MASS", "CTAUERR")]/0.29979 #Convert from mm to ps using speed of light in mm/ps
    


    
    df.loc[:,f'B_Tr_T_endSV_Z'] = np.abs(df[f'B_ENDV_Z'] - df[f'B_Tr_T_firstZ'])
     

    df.loc[:,'diff_P'] = np.abs(df[f'B_P'] - df[f'B_Tr_T_P'])
    df.loc[:,'P_proj'] = df[f'B_ENERGY'] * df[f'B_Tr_T_ENERGY'] - (
        df[f'B_Tr_T_PX'] * df[f'B_PX'] +
        df[f'B_Tr_T_PY'] * df[f'B_PY'] +
        df[f'B_Tr_T_PZ'] * df[f'B_PZ']
    )
    num = (
        df[f'B_ENDV_X']**2 + df[f'B_ENDV_Y']**2 + df[f'B_ENDV_Z']**2 -
        df[f'B_ENDV_X'] * df[f'B_Tr_T_X'] -
        df[f'B_ENDV_Y'] * df[f'B_Tr_T_Y'] -
        df[f'B_ENDV_Z'] * df[f'B_Tr_T_Z']
    )
    den = (
        df[f'B_ENDV_X'] * df[f'B_Tr_T_PX'] +
        df[f'B_ENDV_Y'] * df[f'B_Tr_T_PY'] +
        df[f'B_ENDV_Z'] * df[f'B_Tr_T_PZ']
    )
    df.loc[:,'t'] = num / den
    df.loc[:,'EVIP'] = np.sqrt(
        df[f'B_Tr_T_X']**2 + df[f'B_Tr_T_Y']**2 + df[f'B_Tr_T_Z']**2 +
        df['t']**2 * (df[f'B_Tr_T_PX']**2 + df[f'B_Tr_T_PY']**2 + df[f'B_Tr_T_PZ']**2) +
        2 * df['t'] * (df[f'B_Tr_T_X'] * df[f'B_Tr_T_PX'] + df[f'B_Tr_T_Y'] * df[f'B_Tr_T_PY'] + df[f'B_Tr_T_Z'] * df[f'B_Tr_T_PZ'])
    )
    df.loc[:,'logEVIP'] = np.log(df['EVIP'])
    df.loc[:,'logP_proj'] = np.log(df['P_proj'])
    df.loc[:,f'B_Tr_T_atanPT_PZ'] = np.arctan2(df[f'B_Tr_T_PT'], df[f'B_Tr_T_PZ'])


    # Momenta differences
    df.loc[:,f'B_Tr_T_DeltaP_X'] = df[f'B_PX'] - df[f'B_Tr_T_PX']
    df.loc[:,f'B_Tr_T_DeltaP_Y'] = df[f'B_PY'] - df[f'B_Tr_T_PY']
    df.loc[:,f'B_Tr_T_DeltaP_Z'] = df[f'B_PZ'] - df[f'B_Tr_T_PZ']
    df.loc[:,f'B_Tr_T_DeltaP_X_abs'] = np.abs(df[f'B_PX'] - df[f'B_Tr_T_PX'])
    df.loc[:,f'B_Tr_T_DeltaP_Y_abs'] = np.abs(df[f'B_PY'] - df[f'B_Tr_T_PY'])
    df.loc[:,f'B_Tr_T_DeltaP_Z_abs'] = np.abs(df[f'B_PZ'] - df[f'B_Tr_T_PZ'])
    df.loc[:,f'B_Tr_T_DeltaP'] = np.sqrt(df[f'B_Tr_T_DeltaP_X']**2 + df[f'B_Tr_T_DeltaP_Y']**2 + df[f'B_Tr_T_DeltaP_Z']**2)
    df.loc[:,f'B_Tr_T_DeltaP_T'] = np.sqrt(df[f'B_Tr_T_DeltaP_X']**2 + df[f'B_Tr_T_DeltaP_Y']**2)

    #Eta as angle
    df.loc[:,f'B_Tr_T_Theta']         = np.arctan2(df[f'B_Tr_T_PT'], df[f'B_Tr_T_PZ'])
    df.loc[:,f'B_Theta']              = np.arctan2(df[f'B_PT'], df[f'B_PZ'])
    df.loc[:,f'B_Tr_T_ThetaDistance'] = np.abs(df[f'B_Theta'] - df[f'B_Tr_T_Theta'])

    #Positional differences
    df.loc[:,f'B_Tr_T_diff_x'] = np.abs(df[f'B_OWNPV_X'] - df[f'B_Tr_T_OWNPV_X'])
    df.loc[:,f'B_Tr_T_diff_y'] = np.abs(df[f'B_OWNPV_Y'] - df[f'B_Tr_T_OWNPV_Y'])
    df.loc[:,f'B_Tr_T_diff_z'] = np.abs(df[f'B_OWNPV_Z'] - df[f'B_Tr_T_OWNPV_Z'])
    df.loc[:,f'B_Tr_T_diff_rho'] = np.sqrt(df[f'B_Tr_T_diff_x']**2 + df[f'B_Tr_T_diff_y']**2) #Distance in the transverse plane
    df.loc[:,f'B_Tr_T_diff_R'] = np.sqrt(df[f'B_Tr_T_diff_x']**2 + df[f'B_Tr_T_diff_y']**2 + df[f'B_Tr_T_diff_z']**2) #Distance in 3D space

    #Angular seperation (combination of theta and phi differences) 
    df.loc[:,f'B_Tr_T_AngleSep'] = np.acos(np.cos(df[f'B_Tr_T_ThetaDistance']) * np.cos(df[f'B_Tr_T_PhiDistance']))
    df.loc[:,f'B_Tr_T_cos_AngleSep'] = np.cos(df[f'B_Tr_T_AngleSep'])
    df.loc[:,f'B_Tr_T_sin_AngleSep'] = np.sin(df[f'B_Tr_T_AngleSep'])


    #Products of angle differences and distances
    df.loc[:,f'B_Tr_T_AngleSep_diff_z']   = df[f'B_Tr_T_AngleSep'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_AngleSep_diff_rho'] = df[f'B_Tr_T_AngleSep'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_AngleSep_diff_R']   = df[f'B_Tr_T_AngleSep'] * df[f'B_Tr_T_diff_R']
    df.loc[:,f'B_Tr_T_cos_AngleSep_diff_z']   = df[f'B_Tr_T_cos_AngleSep'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_cos_AngleSep_diff_rho'] = df[f'B_Tr_T_cos_AngleSep'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_cos_AngleSep_diff_R']   = df[f'B_Tr_T_cos_AngleSep'] * df[f'B_Tr_T_diff_R']
    df.loc[:,f'B_Tr_T_sin_AngleSep_diff_z']   = df[f'B_Tr_T_sin_AngleSep'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_sin_AngleSep_diff_rho'] = df[f'B_Tr_T_sin_AngleSep'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_sin_AngleSep_diff_R']   = df[f'B_Tr_T_sin_AngleSep'] * df[f'B_Tr_T_diff_R']
        


    df.loc[:,f'B_Tr_T_PhiDist_diff_z']   = df[f'B_Tr_T_PhiDistance'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_PhiDist_diff_rho'] = df[f'B_Tr_T_PhiDistance'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_PhiDist_diff_R']   = df[f'B_Tr_T_PhiDistance'] * df[f'B_Tr_T_diff_R']
    df.loc[:,f'B_Tr_T_cosPhiDist_diff_z']   = df[f'B_Tr_T_cos_PhiDistance'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_cosPhiDist_diff_rho'] = df[f'B_Tr_T_cos_PhiDistance'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_cosPhiDist_diff_R']   = df[f'B_Tr_T_cos_PhiDistance'] * df[f'B_Tr_T_diff_R']

    df.loc[:,f'B_Tr_T_ThetaDist_diff_z']   = df[f'B_Tr_T_ThetaDistance'] * df[f'B_Tr_T_diff_z']
    df.loc[:,f'B_Tr_T_ThetaDist_diff_rho'] = df[f'B_Tr_T_ThetaDistance'] * df[f'B_Tr_T_diff_rho']
    df.loc[:,f'B_Tr_T_ThetaDist_diff_R']   = df[f'B_Tr_T_ThetaDistance'] * df[f'B_Tr_T_diff_R']


    #ProbbNN ratios
    df.loc[:,f'B_Tr_T_ProbNN_EoverMu']  = df[f'B_Tr_T_PROBNN_E']  / (df[f'B_Tr_T_PROBNN_E']  + df[f'B_Tr_T_PROBNN_MU'])
    df.loc[:,f'B_Tr_T_ProbNN_EoverPi']  = df[f'B_Tr_T_PROBNN_E']  / (df[f'B_Tr_T_PROBNN_E']  + df[f'B_Tr_T_PROBNN_PI'])
    df.loc[:,f'B_Tr_T_ProbNN_EoverK']   = df[f'B_Tr_T_PROBNN_E']  / (df[f'B_Tr_T_PROBNN_E']  + df[f'B_Tr_T_PROBNN_K'])
    df.loc[:,f'B_Tr_T_ProbNN_EoverP']   = df[f'B_Tr_T_PROBNN_E']  / (df[f'B_Tr_T_PROBNN_E']  + df[f'B_Tr_T_PROBNN_P'])

    df.loc[:,f'B_Tr_T_ProbNN_MuoverE']  = df[f'B_Tr_T_PROBNN_MU'] / (df[f'B_Tr_T_PROBNN_MU'] + df[f'B_Tr_T_PROBNN_E'])
    df.loc[:,f'B_Tr_T_ProbNN_MuoverPi'] = df[f'B_Tr_T_PROBNN_MU'] / (df[f'B_Tr_T_PROBNN_MU'] + df[f'B_Tr_T_PROBNN_PI'])
    df.loc[:,f'B_Tr_T_ProbNN_MuoverK']  = df[f'B_Tr_T_PROBNN_MU'] / (df[f'B_Tr_T_PROBNN_MU'] + df[f'B_Tr_T_PROBNN_K'])
    df.loc[:,f'B_Tr_T_ProbNN_MuoverP']  = df[f'B_Tr_T_PROBNN_MU'] / (df[f'B_Tr_T_PROBNN_MU'] + df[f'B_Tr_T_PROBNN_P'])

    df.loc[:,f'B_Tr_T_ProbNN_PioverE']  = df[f'B_Tr_T_PROBNN_PI'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_E'])
    df.loc[:,f'B_Tr_T_ProbNN_PioverMu'] = df[f'B_Tr_T_PROBNN_PI'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_MU'])
    df.loc[:,f'B_Tr_T_ProbNN_PioverK']  = df[f'B_Tr_T_PROBNN_PI'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_K'])
    df.loc[:,f'B_Tr_T_ProbNN_PioverP']  = df[f'B_Tr_T_PROBNN_PI'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_P'])

    df.loc[:,f'B_Tr_T_ProbNN_KoverE']   = df[f'B_Tr_T_PROBNN_K']  / (df[f'B_Tr_T_PROBNN_K']  + df[f'B_Tr_T_PROBNN_E'])
    df.loc[:,f'B_Tr_T_ProbNN_KoverMu']  = df[f'B_Tr_T_PROBNN_K']  / (df[f'B_Tr_T_PROBNN_K']  + df[f'B_Tr_T_PROBNN_MU'])
    df.loc[:,f'B_Tr_T_ProbNN_KoverPi']  = df[f'B_Tr_T_PROBNN_K']  / (df[f'B_Tr_T_PROBNN_K']  + df[f'B_Tr_T_PROBNN_PI'])
    df.loc[:,f'B_Tr_T_ProbNN_KoverP']   = df[f'B_Tr_T_PROBNN_K']  / (df[f'B_Tr_T_PROBNN_K']  + df[f'B_Tr_T_PROBNN_P'])

    df.loc[:,f'B_Tr_T_ProbNN_PoverE']   = df[f'B_Tr_T_PROBNN_P']  / (df[f'B_Tr_T_PROBNN_P']  + df[f'B_Tr_T_PROBNN_E'])
    df.loc[:,f'B_Tr_T_ProbNN_PoverMu']  = df[f'B_Tr_T_PROBNN_P']  / (df[f'B_Tr_T_PROBNN_P']  + df[f'B_Tr_T_PROBNN_MU'])
    df.loc[:,f'B_Tr_T_ProbNN_PoverPi']  = df[f'B_Tr_T_PROBNN_P']  / (df[f'B_Tr_T_PROBNN_P']  + df[f'B_Tr_T_PROBNN_PI'])
    df.loc[:,f'B_Tr_T_ProbNN_PoverK']   = df[f'B_Tr_T_PROBNN_P']  / (df[f'B_Tr_T_PROBNN_P']  + df[f'B_Tr_T_PROBNN_K'])

    df.loc[:,f'B_Tr_T_ProbNN_PioverHadron']  = df[f'B_Tr_T_PROBNN_PI'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_K'] + df[f'B_Tr_T_PROBNN_P'])
    df.loc[:,f'B_Tr_T_ProbNN_KoverHadron']   = df[f'B_Tr_T_PROBNN_K'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_K'] + df[f'B_Tr_T_PROBNN_P'])
    df.loc[:,f'B_Tr_T_ProbNN_PoverHadron']   = df[f'B_Tr_T_PROBNN_P'] / (df[f'B_Tr_T_PROBNN_PI'] + df[f'B_Tr_T_PROBNN_K'] + df[f'B_Tr_T_PROBNN_P'])
        
    #Combination of IP and IPChi2
    df.loc[:,f'B_Tr_T_IPBVTX_IPChi2BVTX'] = df[f'B_Tr_T_IPBVTX'] * df[f'B_Tr_T_IPChi2BVTX']
    df.loc[:,f'B_Tr_T_MINIPChi2_IPChi2BVTX'] = df[f'B_Tr_T_MINIPChi2'] * df[f'B_Tr_T_IPChi2BVTX']

    # Energy momentum ratios
    df.loc[:,f'B_Tr_T_EoverP']  = df[f'B_Tr_T_ENERGY'] / df[f'B_Tr_T_P']
    df.loc[:,f'B_Tr_T_EoverPT'] = df[f'B_Tr_T_ENERGY'] / df[f'B_Tr_T_PT']
    df.loc[:,f'B_Tr_T_ET']      = E_T(df, df[f'B_Tr_T_M'])

    # Positional distances
    df.loc[:,f'B_Tr_T_Rho'] = np.sqrt(df[f'B_Tr_T_X']**2 + df[f'B_Tr_T_Y']**2)
    df.loc[:,f'B_Tr_T_R'] = np.sqrt(df[f'B_Tr_T_X']**2 + df[f'B_Tr_T_Y']**2 + df[f'B_Tr_T_Z']**2)
    df.loc[:,f'B_Tr_T_firstRho'] = np.sqrt(df[f'B_Tr_T_firstX']**2 + df[f'B_Tr_T_firstY']**2)
    df.loc[:,f'B_Tr_T_firstR'] = np.sqrt(df[f'B_Tr_T_firstX']**2 + df[f'B_Tr_T_firstY']**2 + df[f'B_Tr_T_firstZ']**2)


    df.loc[:,f'signal_is_Bs'] = float('Bs' in evtType)




    with_na=df.shape[0]
    print(f"\nDropping NaN values and converting data types for chunk with shape {df.shape}", flush=True)
    # Drop NaN values and convert data types
    df.dropna(inplace=True)
    print("Dropped tracks=", with_na - df.shape[0], flush=True)
    df = df.astype({col: 'float32' for col in df.select_dtypes(include='float64').columns})
    #df.drop(columns=df.select_dtypes(include=['object']).columns, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def mc_vars_to_data_vars(variables):
        variables = [v for v in variables if ("TRUE" not in v and "BKGCAT" not in v and "Origin_Flag" not in v and "MC" not in v)]  
        variables.append("B_Tr_T_IsInTree")
        variables.append("FillNumber")
        return variables


def get_loading_vars(evtType, data_type, loading_var_path = "configs/loading_variables.txt", signal_class_feat_path = "configs/signal_classifier_features.yaml", classical_selection_features_path = "configs/classic_selection_features.yaml"):
    with open(loading_var_path, 'r') as f:
        loading_variables = f.read().splitlines()
    with open(signal_class_feat_path, 'r') as f:
        signal_class_features = yaml.safe_load(f)
        loading_variables += signal_class_features[evtType]
    with open(classical_selection_features_path, 'r') as f:
        classical_selection_features = yaml.safe_load(f)
        loading_variables += classical_selection_features[evtType]

    #Add CTAU
    loading_variables.append(get_mass_label(evtType).replace("MASS", "CTAU"))
    loading_variables.append(get_mass_label(evtType).replace("MASS", "CTAUERR"))

    # BPV -> OWNPV will need to be changed for everything in the future productions!!!!
    if data_type == 'Data':
        loading_variables = mc_vars_to_data_vars(loading_variables)
    loading_variables.append("B_ID")
    loading_variables.append(get_mass_label(evtType))


    return loading_variables

if __name__ == '__main__':
    start_script_time = time()

    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=2500) 
    parser.add_argument('--data_type', help='Type of data (MC or Data)', type=str, choices=('MC', 'Data'))
    parser.add_argument('--loading_features', help='Path to file containing all features to load', type=str)
    parser.add_argument('--signal_class_features', help='Path to yaml file containing the features used by the signal classifier', type=str)
    parser.add_argument('--classic_selection_features', help='Path to yaml file containing the features used by the classical selection', type=str)
    cfg = parser.parse_args()
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)



    prefix = cfg.evtType[:2] + "_" if cfg.data_type == 'MC' else "B_"
    abs_id_map = {'Bs2DsPi': 531, 'Bd2JpsiKst': 511, 'Bu2JpsiK': 521, 'Bd2DmPi': 511, 'Bs2JpsiPhi': 531}
    abs_id = abs_id_map.get(cfg.evtType)

    loading_variables = get_loading_vars(cfg.evtType, cfg.data_type, cfg.loading_features, cfg.signal_class_features, cfg.classic_selection_features)

    # Drop duplicates while preserving order
    loading_variables = list(OrderedDict.fromkeys(loading_variables))


    print(f'Loading variables: {loading_variables}')
    print('Started processing')

    file_id = os.path.basename(cfg.raw)[:-5]
    if file_id[-7:-2] == '.data':
        file_id = file_id[:-7]
    else:
        file_id = file_id[:-3]
    file_id = int(file_id)
    #Optimize memory usage by input files batch by batch instead of loading the whole file at once.  
    candidate_index = 0
    with uproot.recreate(cfg.output) as fout:
        chunk_iter = uproot.iterate({cfg.raw: cfg.treename}, filter_name=loading_variables, library="ak", step_size=cfg.batch_size)

        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            chunk['candidate_index'] = np.arange(len(chunk)) + candidate_index
            candidate_index += len(chunk)
            chunk = ak.to_dataframe(chunk)

            missing_cols = [col for col in loading_variables if col not in chunk.columns]
            if len(missing_cols) > 0:
                raise ValueError(f"Missing columns in chunk {i}: {missing_cols}")

            chunk.reset_index(drop=True, inplace=True)
            chunk.loc[:,'file_id'] = file_id
            start = time()
            chunk = chunk.copy()

            if cfg.data_type == 'Data':
                chunk = chunk[chunk[f'B_Tr_T_IsInTree'] != 1]
            else:
                chunk = chunk[np.abs(chunk[f'B_TRUEID']) == abs_id]
                chunk.reset_index(drop=True, inplace=True)
                chunk[f'B_Tr_T_absID'] = np.abs(chunk[f'B_Tr_T_TRUE_PARTICLE_ID'])
            

            chunk = process_chunk(chunk, cfg.evtType)
            if i == 0:
                fout["DecayTree"] = chunk
            else:
                fout["DecayTree"].extend(chunk)
    print(list(chunk.columns))
    


    print(f"All chunks processed and saved to {cfg.output}")
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"Time required: {time() -  start_script_time:.2f} seconds")