import pandas as pd
import uproot
import numpy as np
import os
import argparse
import awkward as ak
import datetime

import os
#from memory_profiler import profile

'''
def stage_remote(remote_url, max_retries=3):
    """Copy remote_url into /tmp (or $TMPDIR) and return the local path."""
    fn = os.path.basename(remote_url)
    local = os.path.join(tempfile.gettempdir(), fn)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        for attempt in range(1, max_retries+1):
            print(f"Staging {remote_url} → {local} (attempt {attempt})")
            ret = subprocess.call(["xrdcp", "-f", remote_url, local])
            if ret == 0:
                break
            elif attempt == max_retries:
                raise RuntimeError(f"xrdcp failed after {max_retries} tries")
    return local
'''


def DeltaQ(df,Mass, prefix):
        E =np.sqrt( Mass**2 + df[f'{prefix}Tr_T_PX']**2 + df[f'{prefix}Tr_T_PY']**2 + df[f'{prefix}Tr_T_PZ']**2)
        DeltaQ = np.sqrt( (E + df[f'{prefix}ENERGY'])**2  - ((df[f'{prefix}Tr_T_PX'] + df[f'{prefix}PX'])**2 + (df[f'{prefix}Tr_T_PY'] + df[f'{prefix}PY'])**2 + (df[f'{prefix}Tr_T_PZ'] + df[f'{prefix}PZ'])**2 )   ) -df[f'{prefix}M']  - Mass
        return(DeltaQ)

# Phi distance definition from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/src/Utils/TaggingHelpers.cpp?ref_type=heads#L43
def min_dPhi(df, prefix):
    df.eval(f'{prefix}Tr_T_cos_Phi=cos({prefix}Tr_T_Phi)', inplace=True)
    df.eval(f'{prefix}Tr_T_sin_Phi=sin({prefix}Tr_T_Phi)', inplace=True)
    df.eval(f'{prefix}cos_Phi=cos({prefix}PHI)', inplace=True)
    df.eval(f'{prefix}sin_Phi=sin({prefix}PHI)', inplace=True)
    df.eval(f'x_arctan=({prefix}Tr_T_cos_Phi*{prefix}sin_Phi) - ({prefix}cos_Phi*{prefix}Tr_T_sin_Phi)', inplace=True)
    df.eval(f'y_arctan=({prefix}Tr_T_cos_Phi*{prefix}cos_Phi) + ({prefix}sin_Phi*{prefix}Tr_T_sin_Phi)', inplace=True)
    df.eval(f'{prefix}Tr_T_PhiDistance = arctan2(x_arctan, y_arctan)', inplace=True, engine='python')
    # A bit of a hack to add the minimum distance
    _df = df.groupby('entry').apply(lambda group: np.min(np.abs(group[f'{prefix}Tr_T_PhiDistance']))).reset_index(name=f'{prefix}Tr_T_minPhiDistance')
    df = pd.merge(df, _df, on='entry', how='left')
    df.drop([f'{prefix}Tr_T_cos_Phi', f'{prefix}Tr_T_sin_Phi', f'{prefix}cos_Phi', f'{prefix}sin_Phi', 'x_arctan', 'y_arctan'], axis=1, inplace=True)
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
        'B_Probability_Medium_0_Run2OSVertexCharge_Dec',
        'B_Probability_Medium_0_Run2OSVertexCharge_Omega'
    ]

extra_vars = [
        'B_Tr_T_firstX',
        'B_Tr_T_firstY',
        'B_Tr_T_firstZ',
        'B_Tr_T_firstTX',
        'B_Tr_T_firstTY',
        'B_OWNPV_X',
        'B_OWNPV_Y',
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
#@profile
def main():
    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=250) #1000
    parser.add_argument('--data_calib', action="store_true", default=False)
    parser.add_argument('--signal_weights', action='store_true', help='store signal_weights if they are already in the NTuples') # action='store_true' means args.signal_weights will be set to True if the --signal_weights argument is provided on the command line.
    
    cfg = parser.parse_args()
    from pprint import pprint
    pprint(cfg)
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)

    B_abs_id_dic = {
    'Bs2DsPi': 531,
    'Bd2JpsiKst': 511,
    'Bu2JpsiK': 521,
    'Bd2DmPi': 511,
    'Bs2JpsiPhi': 531,
    }
    '''
    if cfg.evtType=='Bs2DsPi':
        prefix = 'Bd' + "_"
    else:
        # Modify the prefix based on evtType
        prefix = cfg.evtType[:2] + "_"
    '''
    #prefix = cfg.evtType[:2] + "_"
    prefix = 'B_'
    if cfg.evtType == 'Bu2JpsiK' or cfg.evtType == 'Bd2JpsiKst':
        weights =  ['sWeights',
                    'signal_weights',
                    'background_weights',
                    'fraction_weights',
                    'reweighter_weights',
                    'reweighter_weights_raw']
    #elif cfg.evtType == 'Bs2DsPi':
    #    weights = ['nSig_uo_kkpi_2022_Evts_sw']
    local_loading_variables = np.unique(loading_variables + run2_taggers_variables).tolist()
    if cfg.data_calib:
        local_loading_variables = [v for v in local_loading_variables if "TRUE" not in v and "Flag" not in v and "MC" not in v and "BKGCAT" not in v]
        local_loading_variables += ['FillNumber']
        if cfg.signal_weights:
            local_loading_variables += weights
        local_loading_variables = np.unique(local_loading_variables).tolist()
        if "Jpsi" in cfg.evtType:
            local_loading_variables.append("B_DTF_PV_Jpsi_MASS")
            #local_loading_variables.append("B_DTF_PV_Jpsi_MASSERR")
        elif "Ds" in cfg.evtType:
            local_loading_variables.append("B_DTF_PV_Ds_MASS") 
            #local_loading_variables.append("B_DTF_PV_Ds_MASSERR")
        local_loading_variables = np.unique(local_loading_variables).tolist()
        print("Loading variables are: ", local_loading_variables)
         # Equivalent for data of Origin_Flag != 0 (included later on in the pre-selections)
        #local_file = stage_remote(cfg.raw)
        with uproot.open(cfg.raw) as f:
            df = f[cfg.treename].arrays(local_loading_variables, library="pd")
        df = df[df[f'{prefix}Tr_T_IsInTree'] != 1]
    # Replace B_ in the loading variables if there is a prefix
    #local_loading_variables_withPrefix = [var.replace("B_", prefix) for var in local_loading_variables]
    #print(f'{local_loading_variables_withPrefix}')
    
    print('Started processing')
    #process_file_in_batches(cfg.raw, local_loading_variables_withPrefix, cfg.treename, prefix, abs_id, cfg.evtType, cfg.batch_size, cfg.output)
    #print('Started reading')

    #Optimize memory usage by reading only the needed variable
    #Read full list of variables if input file ends with 1_1.mc.root as these samples will be used for DT training 
    # else read a selection of variables

    # drop the B mesons or other particles that are not of interest and perform operations on MC variables
    if not cfg.data_calib:
        local_loading_variables = np.unique(local_loading_variables + extra_vars).tolist()
        with uproot.open("{}".format(cfg.raw)) as f:
            df = f[cfg.treename].arrays(local_loading_variables, library="pd")
        abs_id = B_abs_id_dic[cfg.evtType]
        df.drop(df[abs(df[f'{prefix}TRUEID']) != abs_id ].index , inplace = True)
        df.reset_index(inplace=True, drop = False)
        df.eval(f'{prefix}Tr_T_absID =abs({prefix}Tr_T_TRUE_PARTICLE_ID)', inplace = True)
        df[f'{prefix}Tr_T_Origin_Flag'].astype(int)
   
   # Add this diagnostic line to see the data types
    print("DataFrame dtypes after loading:")
    print(df.dtypes)
    
    # Add some needed features
    # A bit of a hack to add the minimum distance
    df = min_dPhi(df, prefix)
    df.eval(f'{prefix}Tr_T_diff_z = abs({prefix}OWNPV_Z - {prefix}Tr_T_OWNPV_Z)' , inplace = True)
    df.eval(f'{prefix}Tr_T_Signal_TagPart_PT = sqrt(({prefix}PX + {prefix}Tr_T_PX) **2 + ({prefix}PY + {prefix}Tr_T_PY)**2)', inplace = True)
    df.eval(f'{prefix}Tr_T_cos_PhiDistance=cos({prefix}Tr_T_PhiDistance)', inplace=True)
    df.eval(f'{prefix}Tr_T_EtaDistance = abs({prefix}ETA - {prefix}Tr_T_Eta)', inplace = True)
    df.eval(f'{prefix}Tr_T_DeltaR= ({prefix}ETA - {prefix}Tr_T_Eta)**2 + {prefix}Tr_T_PhiDistance**2', inplace = True)
    df[f'{prefix}Tr_T_DeltaQ_Pion'] = DeltaQ(df,139.5706, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Muon'] = DeltaQ(df,105.65837, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Electron'] = DeltaQ(df,0.51100, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Proton'] = DeltaQ(df,938.27208, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Kaon'] = DeltaQ(df,493.677, prefix)
    df.eval(f'{prefix}Tr_T_OWNPVIPSig = sqrt({prefix}Tr_T_OWNPVIPCHI2)' , inplace = True) # IPSig == IPErr
    df.eval(f'{prefix}Tr_T_absOWNPV_IP = abs({prefix}Tr_T_OWNPVIP)', inplace = True)


  



    if not cfg.data_calib: # Avoid to read not used branches for data calibration, still needed for MC for the DT training to select features
        df.eval(f'diff_P = abs({prefix}P - {prefix}Tr_T_P)', inplace = True)
        df.eval(f'P_proj = {prefix}ENERGY*{prefix}Tr_T_ENERGY - ({prefix}Tr_T_PX*{prefix}PX + {prefix}Tr_T_PY*{prefix}PY +{prefix}Tr_T_PZ*{prefix}PZ ) ', inplace = True)
        df.eval(f't = ({prefix}ENDV_X**2 + {prefix}ENDV_Y**2 + {prefix}ENDV_Z**2 - {prefix}ENDV_X*{prefix}Tr_T_X - {prefix}ENDV_Y*{prefix}Tr_T_Y - {prefix}ENDV_Z*{prefix}Tr_T_Z) / ({prefix}ENDV_X * {prefix}Tr_T_PX + {prefix}ENDV_Y * {prefix}Tr_T_PY + {prefix}ENDV_Z * {prefix}Tr_T_PZ)' , inplace = True)
        df.eval(f'EVIP = sqrt(({prefix}Tr_T_X**2 + {prefix}Tr_T_Y**2 + {prefix}Tr_T_Z**2) + t**2 * ({prefix}Tr_T_PX**2 + {prefix}Tr_T_PY**2 + {prefix}Tr_T_PZ**2) + 2*t*({prefix}Tr_T_X * {prefix}Tr_T_PX + {prefix}Tr_T_Y * {prefix}Tr_T_PY + {prefix}Tr_T_Z * {prefix}Tr_T_PZ))', inplace = True)
        df.eval(f'{prefix}Tr_T_eoverP = {prefix}Tr_T_Charge/{prefix}Tr_T_P', inplace = True)
        df.eval('logEVIP = log(EVIP)', inplace = True)
        df.eval('logP_proj = log(P_proj)', inplace = True)
        df.eval(f'{prefix}Tr_T_atanPT_PZ = arctan2({prefix}Tr_T_PT, {prefix}Tr_T_PZ)', engine='python', inplace=True)


    df.columns = df.columns.str.replace(f'{prefix}', 'B_', regex=False)
    print(f'Total shape should be {df.shape[0]}')

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(cfg.output) as f:
        f['Tuple/DecayTree'] = df

    print(f'Modified NTuple processed and saved to {cfg.output}')
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
    #print(f'Finished adding features on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')