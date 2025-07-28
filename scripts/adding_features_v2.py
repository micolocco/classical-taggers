import pandas as pd
import uproot
import numpy as np
import os
import argparse
import awkward as ak
import datetime

from scripts.train_BDT import vars_by_decay

data_vars_translation = {
     "B_Tr_T_zfirst": "B_Tr_T_firstZ",
}

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
    df.drop([f'{prefix}Tr_T_cos_Phi', f'{prefix}Tr_T_sin_Phi', f'{prefix}cos_Phi', f'{prefix}sin_Phi', 'x_arctan', 'y_arctan'], axis=1)
    return df

def process_chunk(df, prefix, abs_id):
    # Perform your data processing here
    df.drop(df[abs(df[f'{prefix}TRUEID']) != abs_id ].index , inplace = True)
    df.reset_index(inplace=True, drop = False)
    # Add some needed features
    # A bit of a hack to add the minimum distance
    df = min_dPhi(df, prefix)
    df.eval(f'{prefix}Tr_T_cos_PhiDistance=cos({prefix}Tr_T_PhiDistance)', inplace=True)
    df.eval(f'{prefix}Tr_T_diff_z = abs({prefix}BPVZ - {prefix}Tr_T_BPVZ)' , inplace = True)
    df.eval(f'{prefix}Tr_T_DeltaR= ({prefix}ETA - {prefix}Tr_T_Eta)**2 + {prefix}Tr_T_PhiDistance**2', inplace = True)
    df.eval(f'diff_P = abs({prefix}P - {prefix}Tr_T_P)', inplace = True)
    df.eval(f'P_proj = {prefix}ENERGY*{prefix}Tr_T_ENERGY - ({prefix}Tr_T_PX*{prefix}PX + {prefix}Tr_T_PY*{prefix}PY +{prefix}Tr_T_PZ*{prefix}PZ ) ', inplace = True)
    df.eval(f't = ({prefix}END_VX**2 + {prefix}END_VY**2 + {prefix}END_VZ**2 - {prefix}END_VX*{prefix}Tr_T_X - {prefix}END_VY*{prefix}Tr_T_Y - {prefix}END_VZ*{prefix}Tr_T_Z) / ({prefix}END_VX * {prefix}Tr_T_PX + {prefix}END_VY * {prefix}Tr_T_PY + {prefix}END_VZ * {prefix}Tr_T_PZ)' , inplace = True)
    df.eval(f'EVIP = sqrt(({prefix}Tr_T_X**2 + {prefix}Tr_T_Y**2 + {prefix}Tr_T_Z**2) + t**2 * ({prefix}Tr_T_PX**2 + {prefix}Tr_T_PY**2 + {prefix}Tr_T_PZ**2) + 2*t*({prefix}Tr_T_X * {prefix}Tr_T_PX + {prefix}Tr_T_Y * {prefix}Tr_T_PY + {prefix}Tr_T_Z * {prefix}Tr_T_PZ))', inplace = True)
    df.eval(f'{prefix}Tr_T_absBPVIP = abs({prefix}Tr_T_BPVIP)', inplace = True)
    df[f'{prefix}Tr_T_Origin_Flag'].astype(int)
    df.eval(f'{prefix}Tr_T_EtaDistance = abs({prefix}ETA - {prefix}Tr_T_Eta)', inplace = True)
    df[f'{prefix}Tr_T_DeltaQ_Pion'] = DeltaQ(df,139.5706, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Muon'] = DeltaQ(df,105.65837, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Electron'] = DeltaQ(df,0.51100, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Proton'] = DeltaQ(df,938.27208, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Kaon'] = DeltaQ(df,493.677, prefix)
    df.eval(f'{prefix}Tr_T_Signal_TagPart_PT = sqrt(({prefix}PX + {prefix}Tr_T_PX) **2 + ({prefix}PY + {prefix}Tr_T_PY)**2)', inplace = True)
    df.eval(f'{prefix}Tr_T_eoverP = {prefix}Tr_T_Charge/{prefix}Tr_T_P', inplace = True)
    df.eval(f'{prefix}Tr_T_absID =abs({prefix}Tr_T_TRUEID)', inplace = True)
    df.eval('logEVIP = log(EVIP)', inplace = True)
    df.eval(f'{prefix}Tr_T_BPVIPSig = sqrt({prefix}Tr_T_BPVIPCHI2)' , inplace = True) # IPSig == IPErr
    df.eval('logP_proj = log(P_proj)', inplace = True)
    df.eval(f'{prefix}Tr_T_atanPT_PZ = arctan2({prefix}Tr_T_PT, {prefix}Tr_T_PZ)', engine='python', inplace=True)
    return df

'''
def dataframe_to_awkward(df):
    """Convert a pandas DataFrame to an awkward array."""
    return ak.Array(df.to_dict(orient="list"))
    
def process_file_in_batches(input_path, loading_variables, treename, prefix, abs_id, decay, batch_size, output_file):
    
    file_exists = os.path.exists(output_file) 
    print(file_exists)
    total_shape=0
    with uproot.open(input_path) as f:
        tree = f[treename]
        batch_counter = 0
        for batch in tree.iterate(loading_variables, library="pd", step_size=batch_size):
            batch_counter += 1
            print(f"Processing batch {batch_counter}...")
            # Read a batch of data
            if len(batch) == 0:
                break
            # Monitor memory usage
            #mem_usage = memory_profiler.memory_usage()[0]
            #print(f"Memory usage after processing batch {batch_counter}: {mem_usage:.2f} MB")
            
            #Adjust batch size if necessary (for example, if memory usage exceeds a certain threshold)
            #if mem_usage > 2000:  # Example: If memory usage exceeds 2 GB
            #    batch_size = int(batch_size / 2)
            #    print(f"Reducing batch size to {batch_size}")
            
            # Process the batch
            processed_batch = process_chunk(batch, prefix, abs_id)
            processed_batch.columns = processed_batch.columns.str.replace(f'{prefix}', 'B_', regex=False)
            total_shape = total_shape + processed_batch.shape[0]
            # Convert the DataFrame to an awkward array
            awkward_array = dataframe_to_awkward(processed_batch)
            # Write or append to the ROOT file
            #if file_exists and batch_counter!=1:
            if file_exists:
                with uproot.update(output_file) as out_file:
                    out_file['Tuple/DecayTree'] = awkward_array
                    print('Appending batch..')
            else:
                with uproot.recreate(output_file) as out_file:
                    out_file['Tuple/DecayTree'] = awkward_array
                    file_exists = True
                    print("Creating new NTuple...")


        print(f"All {batch_counter} batches processed.") 
        print(f"Total shape should be {total_shape}")   

'''
        
loading_variables = [
        'B_BPVX',
        'B_BPVY',
        'B_BPVZ',
        'B_END_VX',
        'B_END_VY',
        'B_END_VZ',
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
        'B_BKGCAT',
        'B_nPVs',
        'B_nTracks',
        'EVENTNUMBER',
        'RUNNUMBER',
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_OWNPVIP',
        'B_Tr_T_OWNPVIPCHI2',
        'B_Tr_T_BPVIP',
        'B_Tr_T_BPVIPCHI2',
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
        'B_Tr_T_zfirst',
        'B_Tr_T_BPVX',
        'B_Tr_T_BPVY',
        'B_Tr_T_BPVZ',
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
        'B_Tr_T_OBJECT_KEY',
        'B_Tr_T_Origin_Flag',
        'B_Tr_T_TRUEID',
        'B_Tr_T_TRUEPRIMARYVERTEX_X',
        'B_Tr_T_TRUEPRIMARYVERTEX_Y',
        'B_Tr_T_TRUEPRIMARYVERTEX_Z',
        'B_Tr_T_TRUEORIGINVERTEX_X',
        'B_Tr_T_TRUEORIGINVERTEX_Y',
        'B_Tr_T_TRUEORIGINVERTEX_Z',
        'B_Tr_T_MC_MOTHER_ID',
        'B_Tr_T_MC_MOTHER_KEY',
        'B_Tr_T_MC_GD_MOTHER_ID',
        'B_Tr_T_MC_GD_MOTHER_KEY',
        'B_Tr_T_MC_GD_GD_MOTHER_ID',
        'B_Tr_T_MC_GD_GD_MOTHER_KEY',
        'B_Run2_SSPion_Dec',
        'B_Run2_SSPion_Omega',
        'B_Run2_SSPion_MVA',
        'B_Run2_SSKaon_Dec',
        'B_Run2_SSKaon_Omega',
        'B_Run2_SSKaon_MVA',
        'B_Run2_SSProton_Dec',
        'B_Run2_SSProton_Omega',
        'B_Run2_SSProton_MVA',
        'B_Run2_OSKaon_Dec',
        'B_Run2_OSKaon_Omega',
        'B_Run2_OSKaon_MVA',
        'B_Run2_OSElectron_Dec',
        'B_Run2_OSElectron_Omega',
        'B_Run2_OSElectron_MVA',
        'B_Run2_OSMuon_Dec',
        'B_Run2_OSMuon_Omega',
        'B_Run2_OSMuon_MVA',
        ]

def get_loading_vars(evtType, data_calib):
    prefix = evtType[:2] + "_" if not data_calib else "B_"

    loading_variables_withPrefix = []
    prx = "OWNPV_" if data_calib else "BPV"
    prxip = "OWNPVIP" if data_calib else "BPVIP" 
    endx = "ENDV_" if data_calib else "END_V"

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
        
    else:
        loading_variables_withPrefix = [var.replace("B_", prefix) for var in loading_variables]

        loading_variables_withPrefix.append(f"{prefix}DTF_PV_Jpsi_MASS")
        loading_variables_withPrefix.append(f"{prefix}DTF_PV_MASS")
        print(loading_variables_withPrefix)
    
    return loading_variables_withPrefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=250) #1000
    parser.add_argument('--data_calib', action="store_true", default=False)
    
    cfg = parser.parse_args()
    
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

    prefix = cfg.evtType[:2] + "_" if not cfg.data_calib else "B_"

    loading_variables_withPrefix = []
    prx = "OWNPV_" if cfg.data_calib else "BPV"
    prxip = "OWNPVIP" if cfg.data_calib else "BPVIP" 
    endx = "ENDV_" if cfg.data_calib else "END_V"

    loading_variables_withPrefix = get_loading_vars(cfg.evtType, cfg.data_calib)

    if cfg.data_calib:
        loading_variables_withPrefix += ['signal_weights', 'background_weights', 'pdf_ratio', 'entry', 'subentry', 'BID_signal_weights', 'BID_background_weights']
        loading_variables_withPrefix = list(dict.fromkeys(loading_variables_withPrefix)) #removes duplicates

    print(f'{loading_variables_withPrefix}')
    print('Started processing')
    #process_file_in_batches(cfg.raw, loading_variables_withPrefix, cfg.treename, prefix, abs_id, cfg.evtType, cfg.batch_size, cfg.output)
    #print('Started reading')
    print(f'Reading {cfg.raw}')
    with uproot.open("{}".format(cfg.raw)) as f:
        df = f[cfg.treename].arrays(loading_variables_withPrefix, library="pd")
    if not cfg.data_calib:
        # drop the B mesons or other particles that are not of interest
        abs_id = B_abs_id_dic[cfg.evtType]
        df.drop(df[abs(df[f'{prefix}TRUEID']) != abs_id ].index , inplace = True)
        df.reset_index(inplace=True, drop = False)
        df.eval(f'{prefix}Tr_T_absID =abs({prefix}Tr_T_TRUEID)', inplace = True) # Is this to change with the reconstructed ID ? 
    # Add some needed features
    # A bit of a hack to add the minimum distance
    print(df.head(10))
    df = min_dPhi(df, prefix)
    df.eval(f'{prefix}Tr_T_cos_PhiDistance=cos({prefix}Tr_T_PhiDistance)', inplace=True)
    df.eval(f'{prefix}Tr_T_diff_z = abs({prefix}{prx}Z - {prefix}Tr_T_{prx}Z)' , inplace = True)
    df.eval(f'{prefix}Tr_T_DeltaR= ({prefix}ETA - {prefix}Tr_T_Eta)**2 + {prefix}Tr_T_PhiDistance**2', inplace = True)
    df.eval(f'diff_P = abs({prefix}P - {prefix}Tr_T_P)', inplace = True)
    df.eval(f'P_proj = {prefix}ENERGY*{prefix}Tr_T_ENERGY - ({prefix}Tr_T_PX*{prefix}PX + {prefix}Tr_T_PY*{prefix}PY +{prefix}Tr_T_PZ*{prefix}PZ ) ', inplace = True)
    df.eval(f't = ({prefix}{endx}X**2 + {prefix}{endx}Y**2 + {prefix}{endx}Z**2 - {prefix}{endx}X*{prefix}Tr_T_X - {prefix}{endx}Y*{prefix}Tr_T_Y - {prefix}{endx}Z*{prefix}Tr_T_Z) / ({prefix}{endx}X * {prefix}Tr_T_PX + {prefix}{endx}Y * {prefix}Tr_T_PY + {prefix}{endx}Z * {prefix}Tr_T_PZ)' , inplace = True)
    df.eval(f'EVIP = sqrt(({prefix}Tr_T_X**2 + {prefix}Tr_T_Y**2 + {prefix}Tr_T_Z**2) + t**2 * ({prefix}Tr_T_PX**2 + {prefix}Tr_T_PY**2 + {prefix}Tr_T_PZ**2) + 2*t*({prefix}Tr_T_X * {prefix}Tr_T_PX + {prefix}Tr_T_Y * {prefix}Tr_T_PY + {prefix}Tr_T_Z * {prefix}Tr_T_PZ))', inplace = True)
    df.eval(f'{prefix}Tr_T_abs{prxip} = abs({prefix}Tr_T_{prxip})', inplace = True)
    df.eval(f'{prefix}Tr_T_EtaDistance = abs({prefix}ETA - {prefix}Tr_T_Eta)', inplace = True)
    df[f'{prefix}Tr_T_DeltaQ_Pion'] = DeltaQ(df,139.5706, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Muon'] = DeltaQ(df,105.65837, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Electron'] = DeltaQ(df,0.51100, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Proton'] = DeltaQ(df,938.27208, prefix)
    df[f'{prefix}Tr_T_DeltaQ_Kaon'] = DeltaQ(df,493.677, prefix)
    df.eval(f'{prefix}Tr_T_Signal_TagPart_PT = sqrt(({prefix}PX + {prefix}Tr_T_PX) **2 + ({prefix}PY + {prefix}Tr_T_PY)**2)', inplace = True)
    df.eval(f'{prefix}Tr_T_eoverP = {prefix}Tr_T_Charge/{prefix}Tr_T_P', inplace = True)
    df.eval('logEVIP = log(EVIP)', inplace = True)
    df.eval(f'{prefix}Tr_T_{prxip}Sig = sqrt({prefix}Tr_T_{prxip}CHI2)' , inplace = True) # IPSig == IPErr
    df.eval('logP_proj = log(P_proj)', inplace = True)
    df.eval(f'{prefix}Tr_T_atanPT_PZ = arctan2({prefix}Tr_T_PT, {prefix}Tr_T_PZ)', engine='python', inplace=True)

    if not cfg.data_calib: df[f'{prefix}Tr_T_Origin_Flag'].astype(int)

    df.columns = df.columns.str.replace(f'{prefix}', 'B_', regex=False)
    if cfg.data_calib:
        # Equivalent for data of Origin_Flag != 0
        df = df[df['B_Tr_T_IsInTree'] != 1]
    print(f'Total shape should be {df.shape[0]}')

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(cfg.output) as f:
        f['Tuple/DecayTree'] = df

    print(f'Modified NTuple processed and saved to {cfg.output}')
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

