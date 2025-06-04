import pandas as pd
import uproot
import numpy as np
import os
import argparse
import datetime
from time import time
from tqdm import tqdm
from IPython import embed

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
    #df[f'{prefix}Tr_T_minPhiDistance'] = df.groupby('entry')[f'{prefix}Tr_T_PhiDistance'].transform(lambda x: np.abs(x).min())
    return df

def process_chunk(df, prefix, is_data):
    df = df.copy()
    df = dPhi(df, prefix)
    df.loc[:,f'{prefix}Tr_T_diff_z'] = np.abs(df[f'{prefix}OWNPV_Z'] - df[f'{prefix}Tr_T_OWNPV_Z'])
    df.loc[:,f'{prefix}Tr_T_Signal_TagPart_PT'] = np.sqrt((df[f'{prefix}PX'] + df[f'{prefix}Tr_T_PX'])**2 + (df[f'{prefix}PY'] + df[f'{prefix}Tr_T_PY'])**2)
    df.loc[:,f'{prefix}Tr_T_cos_PhiDistance'] = np.cos(df[f'{prefix}Tr_T_PhiDistance'])
    df.loc[:,f'{prefix}Tr_T_EtaDistance'] = np.abs(df[f'{prefix}ETA'] - df[f'{prefix}Tr_T_Eta'])
    df.loc[:,f'{prefix}Tr_T_DeltaR'] = (df[f'{prefix}ETA'] - df[f'{prefix}Tr_T_Eta'])**2 + df[f'{prefix}Tr_T_PhiDistance']**2
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Pion'] = DeltaQ(df, 139.5706, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Muon'] = DeltaQ(df, 105.65837, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Electron'] = DeltaQ(df, 0.51100, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Proton'] = DeltaQ(df, 938.27208, prefix)
    df.loc[:,f'{prefix}Tr_T_DeltaQ_Kaon'] = DeltaQ(df, 493.677, prefix)
    df.loc[:,f'{prefix}Tr_T_OWNPVIPSig'] = np.sqrt(df[f'{prefix}Tr_T_OWNPVIPCHI2'])
    df.loc[:,f'{prefix}Tr_T_absOWNPV_IP'] = np.abs(df[f'{prefix}Tr_T_OWNPVIP'])

    if not is_data:
        df.loc[:,'diff_P'] = np.abs(df[f'{prefix}P'] - df[f'{prefix}Tr_T_P'])
        df.loc[:,'P_proj'] = df[f'{prefix}ENERGY'] * df[f'{prefix}Tr_T_ENERGY'] - (
            df[f'{prefix}Tr_T_PX'] * df[f'{prefix}PX'] +
            df[f'{prefix}Tr_T_PY'] * df[f'{prefix}PY'] +
            df[f'{prefix}Tr_T_PZ'] * df[f'{prefix}PZ']
        )
        num = (
            df[f'{prefix}ENDV_X']**2 + df[f'{prefix}ENDV_Y']**2 + df[f'{prefix}ENDV_Z']**2 -
            df[f'{prefix}ENDV_X'] * df[f'{prefix}Tr_T_X'] -
            df[f'{prefix}ENDV_Y'] * df[f'{prefix}Tr_T_Y'] -
            df[f'{prefix}ENDV_Z'] * df[f'{prefix}Tr_T_Z']
        )
        den = (
            df[f'{prefix}ENDV_X'] * df[f'{prefix}Tr_T_PX'] +
            df[f'{prefix}ENDV_Y'] * df[f'{prefix}Tr_T_PY'] +
            df[f'{prefix}ENDV_Z'] * df[f'{prefix}Tr_T_PZ']
        )
        df.loc[:,'t'] = num / den
        df.loc[:,'EVIP'] = np.sqrt(
            df[f'{prefix}Tr_T_X']**2 + df[f'{prefix}Tr_T_Y']**2 + df[f'{prefix}Tr_T_Z']**2 +
            df['t']**2 * (df[f'{prefix}Tr_T_PX']**2 + df[f'{prefix}Tr_T_PY']**2 + df[f'{prefix}Tr_T_PZ']**2) +
            2 * df['t'] * (df[f'{prefix}Tr_T_X'] * df[f'{prefix}Tr_T_PX'] + df[f'{prefix}Tr_T_Y'] * df[f'{prefix}Tr_T_PY'] + df[f'{prefix}Tr_T_Z'] * df[f'{prefix}Tr_T_PZ'])
        )
        df.loc[:,f'{prefix}Tr_T_eoverP'] = df[f'{prefix}Tr_T_Charge'] / df[f'{prefix}Tr_T_P']
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


# Variables not used for pre-selections and training are included in extra_var array to speed up NTuples processing
# List of variables (includes MC variables)
loading_variables = [
        'EVENTNUMBER',
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_OWNPVIP',
        'B_Tr_T_Charge',
        "B_Tr_T_ISMUON",
        'B_Tr_T_PROBNN_GHOST',
        'B_Tr_T_IPBVTX',
        "B_ID",
        "B_DTF_PV_MASS",
        "B_DTF_PV_MASSERR",
        'B_DTF_PV_CTAU',
        'B_DTF_PV_CTAUERR',
        'B_ENERGY',
        'B_ETA',
        'B_M',
        'B_P',
        'B_PHI',
        'B_PT',
        'B_PX',
        'B_PY',
        'B_PZ',
        'B_TRUEID',
        'B_nPVs',
        'B_nTracks',
        'RUNNUMBER',
        'B_Tr_T_OWNPVIPCHI2',
        'B_Tr_T_Eta',
        'B_Tr_T_P',
        'B_Tr_T_PT',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_Phi',
        'B_Tr_T_GHOSTPROB',
        'B_Tr_T_PX',
        'B_Tr_T_PY',
        'B_Tr_T_PZ',
        'B_Tr_T_IPChi2BVTX',
        'B_Tr_T_Origin_Flag',
        'B_Tr_T_IsInTree',
        'B_Tr_T_CHI2DOF',
        'B_ID',
        'B_BKGCAT',
        'B_Tr_T_TRUE_PARTICLE_ID',
        'B_Tr_T_OWNPV_Z',
        'B_OWNPV_Z',
        #'B_Tr_T_TRUEPRIMARYVERTEX_X',
        #'B_Tr_T_TRUEPRIMARYVERTEX_Y',
        #'B_Tr_T_TRUEPRIMARYVERTEX_Z',
        #'B_Tr_T_TRUEORIGINVERTEX_X',
        #'B_Tr_T_TRUEORIGINVERTEX_Y',
        #'B_Tr_T_TRUEORIGINVERTEX_Z',
        'B_Tr_T_MC_MOTHER_ID',
        #'B_Tr_T_MC_MOTHER_KEY',
        'B_Tr_T_MC_GD_MOTHER_ID',
        #'B_Tr_T_MC_GD_MOTHER_KEY',
        'B_Tr_T_MC_GD_GD_MOTHER_ID',
        #'B_Tr_T_MC_GD_GD_MOTHER_KEY',
        ]
run2_taggers_variables = [
        'B_Run2_SSPion_Dec',
        'B_Run2_SSPion_Omega',
        #'B_Run2_SSPion_MVA',
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        #'B_Run2_SSKaon_MVA',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        #'B_Run2_SSProton_MVA',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        #'B_Run2_OSKaon_MVA',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        #'B_Run2_OSElectron_MVA',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
        #'B_Run2_OSMuon_MVA',
        'B_Run2_OSVertexCharge_Dec',
        'B_Run2_OSVertexCharge_Omega',
        #'B_Probability_Medium_0_Run2OSVertexCharge_Dec',
        #'B_Probability_Medium_0_Run2OSVertexCharge_Omega'
    ]

extra_vars = [
        'B_Tr_T_firstX',
        'B_Tr_T_firstY',
        'B_Tr_T_firstZ',
        'B_Tr_T_firstTX',
        'B_Tr_T_firstTY',
        'B_OWNPV_X',
        'B_OWNPV_Y',
        'B_OWNPV_Z',
        'B_ENDV_X',
        'B_ENDV_Y',
        'B_ENDV_Z',
        'B_Tr_T_OWNPV_X',
        'B_Tr_T_OWNPV_XERR',
        'B_Tr_T_OWNPV_Y',
        'B_Tr_T_OWNPV_YERR',
        'B_Tr_T_OWNPV_ZERR',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PIDP',
        'B_Tr_T_M',
        'B_Tr_T_X',
        'B_Tr_T_Y',
        'B_Tr_T_Z',
        'B_Tr_T_MINIP',
        'B_Tr_T_MINIPChi2',
        'B_Tr_T_ENERGY',
    ]

if __name__ == '__main__':
    start_script_time = time()

    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=1000) #1000
    parser.add_argument('--data_calib', action="store_true", default=False)
    parser.add_argument('--signal_weights', action='store_true', help='store signal_weights if they are already in the NTuples') # action='store_true' means args.signal_weights will be set to True if the --signal_weights argument is provided on the command line.
    
    cfg = parser.parse_args()
    from pprint import pprint
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)

    prefix = 'B_'
    abs_id_map = {'Bs2DsPi': 531, 'Bd2JpsiKst': 511, 'Bu2JpsiK': 521, 'Bd2DmPi': 511, 'Bs2JpsiPhi': 531}
    abs_id = abs_id_map.get(cfg.evtType)
    '''
    if cfg.evtType=='Bs2DsPi':
        prefix = 'Bd' + "_"
    else:
        # Modify the prefix based on evtType
        prefix = cfg.evtType[:2] + "_"
    '''
    #prefix = cfg.evtType[:2] + "_"

    if cfg.evtType == 'Bu2JpsiK' or cfg.evtType == 'Bd2JpsiKst':
        weights =  ['sWeights',
                    'signal_weights',
                    'background_weights',
                    'fraction_weights',
                    'reweighter_weights',
                    'reweighter_weights_raw']
    #elif cfg.evtType == 'Bs2DsPi':
    #    weights = ['nSig_uo_kkpi_2022_Evts_sw']
    loading_variables = loading_variables + run2_taggers_variables
    if cfg.data_calib:
        loading_variables = [v for v in loading_variables if "TRUE" not in v and "Flag" not in v and "MC" not in v and "BKGCAT" not in v]
        loading_variables += ['FillNumber']
        if cfg.signal_weights:
            loading_variables += weights
        loading_variables = np.unique(loading_variables).tolist()
        if "Jpsi" in cfg.evtType:
            loading_variables.append("B_DTF_PV_Jpsi_MASS")
            #loading_variables.append("B_DTF_PV_Jpsi_MASSERR")
        elif "Ds" in cfg.evtType:
            loading_variables.append("B_DTF_PV_Ds_MASS") 
            #loading_variables.append("B_DTF_PV_Ds_MASSERR")
        print("Loading variables are: ", loading_variables)
         # Equivalent for data of Origin_Flag != 0 (included later on in the pre-selections)
        #local_file = stage_remote(cfg.raw)
        with uproot.open(cfg.raw) as f:
            df = f[cfg.treename].arrays(loading_variables, library="pd")
        df = df[df[f'{prefix}Tr_T_IsInTree'] != 1]
    # Replace B_ in the loading variables if there is a prefix
    #loading_variables_withPrefix = [var.replace("B_", prefix) for var in loading_variables]
    #print(f'{loading_variables_withPrefix}')
    
    print('Started processing')
    #process_file_in_batches(cfg.raw, loading_variables_withPrefix, cfg.treename, prefix, abs_id, cfg.evtType, cfg.batch_size, cfg.output)
    #print('Started reading')

    #Optimize memory usage by reading only the needed variable
    #Read full list of variables if input file ends with 1_1.mc.root as these samples will be used for DT training 
    # else read a selection of variables

    if not cfg.data_calib:
        loading_variables = loading_variables + extra_vars

    with uproot.recreate(cfg.output) as fout:
        chunk_iter = uproot.iterate({cfg.raw: cfg.treename}, filter_name=loading_variables, library="pd", step_size=cfg.batch_size)
        for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
            start = time()
            chunk = chunk.copy()
            if cfg.data_calib:
                chunk = chunk[chunk[f'{prefix}Tr_T_IsInTree'] != 1]
            else:
                chunk = chunk[np.abs(chunk[f'{prefix}TRUEID']) == abs_id]
                chunk.reset_index(drop=True, inplace=True)
                chunk[f'{prefix}Tr_T_absID'] = np.abs(chunk[f'{prefix}Tr_T_TRUE_PARTICLE_ID'])

            chunk = process_chunk(chunk, prefix, cfg.data_calib)

            if i == 0:
                fout["Tuple/DecayTree"] = chunk
            else:
                fout["Tuple/DecayTree"].extend(chunk)


    print(f"All chunks processed and saved to {cfg.output}")

    print(f'Modified NTuple processed and saved to {cfg.output}')
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"Time required: {time() -  start_script_time:.2f} seconds")
    print(f"Memory usage (MB):", chunk.memory_usage(deep=True).sum() / 1e6)
