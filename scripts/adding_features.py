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

data_vars_translation = {
    "B_Tr_T_zfirst": "B_Tr_T_firstZ",
}

def DeltaQ(df, Mass, prefix):
    E = np.sqrt(Mass**2 + df[f'{prefix}Tr_T_PX']**2 + df[f'{prefix}Tr_T_PY']**2 + df[f'{prefix}Tr_T_PZ']**2)
    return np.sqrt(
        (E + df[f'{prefix}ENERGY'])**2 - (
            (df[f'{prefix}Tr_T_PX'] + df[f'{prefix}PX'])**2 +
            (df[f'{prefix}Tr_T_PY'] + df[f'{prefix}PY'])**2 +
            (df[f'{prefix}Tr_T_PZ'] + df[f'{prefix}PZ'])**2
        )) - df[f'{prefix}M'] - Mass

# Phi distance definition from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/src/Utils/TaggingHelpers.cpp?ref_type=heads#L43
def dPhi(df, prefix):
    cos_Tr_T_Phi = np.cos(df[f'{prefix}Tr_T_Phi'])
    sin_Tr_T_Phi = np.sin(df[f'{prefix}Tr_T_Phi'])
    cos_Phi = np.cos(df[f'{prefix}PHI'])
    sin_Phi = np.sin(df[f'{prefix}PHI'])

    x_arctan = cos_Tr_T_Phi * sin_Phi - cos_Phi * sin_Tr_T_Phi
    y_arctan = cos_Tr_T_Phi * cos_Phi + sin_Phi * sin_Tr_T_Phi
    df[f'{prefix}Tr_T_PhiDistance'] = np.arctan2(x_arctan, y_arctan)
    # df[f'{prefix}Tr_T_minPhiDistance'] = df.groupby('entry')[f'{prefix}Tr_T_PhiDistance'].transform(lambda x: np.abs(x).min())
    return df

def process_chunk(df, prefix, is_data):
    df = df.copy()

    prx = "OWNPV_" if cfg.data_calib else "BPV"
    prxip = "OWNPVIP" if cfg.data_calib else "BPVIP" 
    endx = "ENDV_" if cfg.data_calib else "END_V"


    df = dPhi(df, prefix)
    df.loc[:,f'{prefix}Tr_T_diff_z'] = np.abs(df[f'{prefix}{prx}Z'] - df[f'{prefix}Tr_T_{prx}Z'])
    df.loc[:,f'{prefix}Tr_T_Signal_TagPart_PT'] = np.sqrt((df[f'{prefix}PX'] + df[f'{prefix}Tr_T_PX'])**2 + (df[f'{prefix}PY'] + df[f'{prefix}Tr_T_PY'])**2)
    df.loc[:,f'{prefix}Tr_T_cos_PhiDistance'] = np.cos(df[f'{prefix}Tr_T_PhiDistance'])
    df.loc[:,f'{prefix}Tr_T_EtaDistance'] = np.abs(df[f'{prefix}ETA'] - df[f'{prefix}Tr_T_Eta'])
    df.loc[:,f'{prefix}Tr_T_DeltaR'] = (df[f'{prefix}ETA'] - df[f'{prefix}Tr_T_Eta'])**2 + df[f'{prefix}Tr_T_PhiDistance']**2
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Pion'] = DeltaQ(df, 139.5706, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Muon'] = DeltaQ(df, 105.65837, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Electron'] = DeltaQ(df, 0.51100, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Proton'] = DeltaQ(df, 938.27208, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Kaon'] = DeltaQ(df, 493.677, prefix)
    df.loc[:,f'{prefix}Tr_T_{prxip}Sig'] = np.sqrt(df[f'{prefix}Tr_T_{prxip}CHI2'])
    df.loc[:,f'{prefix}Tr_T_abs{prx}IP'] = np.abs(df[f'{prefix}Tr_T_{prxip}'])

     

    if not is_data:
        df.loc[:,'diff_P'] = np.abs(df[f'{prefix}P'] - df[f'{prefix}Tr_T_P'])
        df.loc[:,'P_proj'] = df[f'{prefix}ENERGY'] * df[f'{prefix}Tr_T_ENERGY'] - (
            df[f'{prefix}Tr_T_PX'] * df[f'{prefix}PX'] +
            df[f'{prefix}Tr_T_PY'] * df[f'{prefix}PY'] +
            df[f'{prefix}Tr_T_PZ'] * df[f'{prefix}PZ']
        )
        num = (
            df[f'{prefix}{endx}X']**2 + df[f'{prefix}{endx}Y']**2 + df[f'{prefix}{endx}Z']**2 -
            df[f'{prefix}{endx}X'] * df[f'{prefix}Tr_T_X'] -
            df[f'{prefix}{endx}Y'] * df[f'{prefix}Tr_T_Y'] -
            df[f'{prefix}{endx}Z'] * df[f'{prefix}Tr_T_Z']
        )
        den = (
            df[f'{prefix}{endx}X'] * df[f'{prefix}Tr_T_PX'] +
            df[f'{prefix}{endx}Y'] * df[f'{prefix}Tr_T_PY'] +
            df[f'{prefix}{endx}Z'] * df[f'{prefix}Tr_T_PZ']
        )
        df.loc[:,'t'] = num / den
        df.loc[:,'EVIP'] = np.sqrt(
            df[f'{prefix}Tr_T_X']**2 + df[f'{prefix}Tr_T_Y']**2 + df[f'{prefix}Tr_T_Z']**2 +
            df['t']**2 * (df[f'{prefix}Tr_T_PX']**2 + df[f'{prefix}Tr_T_PY']**2 + df[f'{prefix}Tr_T_PZ']**2) +
            2 * df['t'] * (df[f'{prefix}Tr_T_X'] * df[f'{prefix}Tr_T_PX'] + df[f'{prefix}Tr_T_Y'] * df[f'{prefix}Tr_T_PY'] + df[f'{prefix}Tr_T_Z'] * df[f'{prefix}Tr_T_PZ'])
        )
        df.loc[:,f'{prefix}Tr_T_EoverP'] = df[f'{prefix}Tr_T_ENERGY'] / df[f'{prefix}Tr_T_P']
        df.loc[:,'logEVIP'] = np.log(df['EVIP'])
        df.loc[:,'logP_proj'] = np.log(df['P_proj'])
        df.loc[:,f'{prefix}Tr_T_atanPT_PZ'] = np.arctan2(df[f'{prefix}Tr_T_PT'], df[f'{prefix}Tr_T_PZ'])

    df.columns = df.columns.str.replace(f'{prefix}', 'B_', regex=False)
    with_na=df.shape[0]
    print(f"Dropping NaN values and converting data types for chunk with shape {df.shape}")
    # Drop NaN values and convert data types
    df.dropna(inplace=True)
    print("Dropped tracks=", with_na - df.shape[0])
    df = df.astype({col: 'float32' for col in df.select_dtypes(include='float64').columns})
    #df.drop(columns=df.select_dtypes(include=['object']).columns, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_loading_vars(evtType, data_calib, loading_var_path = "configs/loading_variables.txt"):
    prefix = evtType[:2] + "_" if not data_calib else "B_"

    with open(loading_var_path, 'r') as f:
        loading_variables = f.read().splitlines()

    loading_variables_withPrefix = []
    

    # BPV -> OWNPV will need to be changed for everything in the future productions!!!!
    if data_calib:

        loading_variables_withPrefix.append("B_Tr_T_IsInTree")
        loading_variables_withPrefix.append("B_ID")
        loading_variables_withPrefix.append("B_DTF_PV_Jpsi_MASS")
        loading_variables_withPrefix.append("B_DTF_PV_MASS")
        loading_variables_withPrefix.append("FillNumber")
        for v in loading_variables:
            if "TRUE" in v or "BKGCAT" in v or "Origin_Flag" in v or "MC" in v: continue
            if "BPV" in v: v=v.replace("BPV", "OWNPV_").replace("OWNPV_IP", "OWNPVIP")
            if "END_V" in v: v=v.replace("END_V", "ENDV_")
            loading_variables_withPrefix.append(v) if v not in data_vars_translation.keys() else loading_variables_withPrefix.append(data_vars_translation[v])
        loading_variables_withPrefix.remove('B_Tr_T_OWNPVIP')
        loading_variables_withPrefix.remove('B_Tr_T_OWNPVIPCHI2')

        # loading_variables_withPrefix.append('B_Tr_T_BPVIP')
        loading_variables_withPrefix += ['signal_weights', 'background_weights', 'pdf_ratio', 'entry', 'subentry', 'BID_signal_weights', 'BID_background_weights']
    else:
        loading_variables_withPrefix = [var.replace("B_", prefix) for var in loading_variables]

        loading_variables_withPrefix.append(f"{prefix}DTF_PV_Jpsi_MASS")
        loading_variables_withPrefix.append(f"{prefix}DTF_PV_MASS")
        print(loading_variables_withPrefix)
    
    return loading_variables_withPrefix

if __name__ == '__main__':
    start_script_time = time()

    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=1000) 
    parser.add_argument('--data_calib', action="store_true", default=False)
    parser.add_argument('--loading_features', help='Path to file containing all features to load', type=str)
    
    cfg = parser.parse_args()
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)



    prefix = cfg.evtType[:2] + "_" if not cfg.data_calib else "B_"
    abs_id_map = {'Bs2DsPi': 531, 'Bd2JpsiKst': 511, 'Bu2JpsiK': 521, 'Bd2DmPi': 511, 'Bs2JpsiPhi': 531}
    abs_id = abs_id_map.get(cfg.evtType)

    loading_variables = get_loading_vars(cfg.evtType, cfg.data_calib, cfg.loading_features)
    print(f'Loading variables: {loading_variables}')
    print('Started processing')

    #Optimize memory usage by input files batch by batch instead of loading the whole file at once.  
    with uproot.recreate(cfg.output) as fout:
        chunk_iter = uproot.iterate({cfg.raw: cfg.treename}, filter_name=loading_variables, library="ak", step_size=cfg.batch_size)

        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            chunk = ak.to_dataframe(chunk)
            if "entry" not in chunk.columns: #TODO remove if weighting is moved to later stage in pipeline
                chunk.reset_index(inplace=True)

            start = time()
            chunk = chunk.copy()
            if cfg.data_calib:
                chunk = chunk[chunk[f'{prefix}Tr_T_IsInTree'] != 1]
            else:
                chunk = chunk[np.abs(chunk[f'{prefix}TRUEID']) == abs_id]
                chunk.reset_index(drop=True, inplace=True)
                chunk[f'{prefix}Tr_T_absID'] = np.abs(chunk[f'{prefix}Tr_T_TRUEID'])

            chunk = process_chunk(chunk, prefix, cfg.data_calib)

            if i == 0:
                fout["Tuple/DecayTree"] = chunk
            else:
                fout["Tuple/DecayTree"].extend(chunk)
    print(list(chunk.columns))
    


    print(f"All chunks processed and saved to {cfg.output}")
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"Time required: {time() -  start_script_time:.2f} seconds")