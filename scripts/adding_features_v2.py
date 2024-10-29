import pandas as pd
import uproot
import numpy as np
import os
import argparse
import awkward as ak
import datetime

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
    df.eval(f'{prefix}Tr_T_absIP = abs({prefix}Tr_T_BPVIP)', inplace = True)
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
    df.eval('EVIP = log(EVIP)', inplace = True)
    df.eval(f'{prefix}Tr_T_BVIPSig = sqrt({prefix}Tr_T_BPVIPCHI2)' , inplace = True) # IPSig == IPErr
    df.eval('P_proj = log(P_proj)', inplace = True)
    df.eval(f'{prefix}Tr_T_atanPT_PZ = arctan2({prefix}Tr_T_PT, {prefix}Tr_T_PZ)', engine='python', inplace=True)
    return df

def dataframe_to_awkward(df):
    """Convert a pandas DataFrame to an awkward array."""
    return ak.Array(df.to_dict(orient="list"))

'''
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
#Method 3: single DecayTree without holding in memory. Use extend() to append to the DecayTree without creating a new one
import uproot
import awkward as ak
import os

def process_file_in_batches(input_path, loading_variables, treename, prefix, abs_id, decay, batch_size, output_file):
    file_exists = os.path.exists(output_file)
    
    with uproot.open(input_path) as f:
        tree = f[treename]
        batch_counter = 0
        all_data = []  # Accumulate all data in this list
        total_shape = 0
        for batch in tree.iterate(loading_variables, library="pd", step_size=batch_size):
            batch_counter += 1
            print(f"Processing batch {batch_counter}...")

            # Process the batch
            processed_batch = process_chunk(batch, prefix, abs_id)
            processed_batch.columns = processed_batch.columns.str.replace(f'{prefix}', 'B_', regex=False)
            total_shape += processed_batch.shape[0]

            # Convert to awkward array
            awkward_array = dataframe_to_awkward(processed_batch)
            all_data.append(awkward_array)  # Accumulate the awkward arrays

        # Once all batches are processed, concatenate and write
        final_data = ak.concatenate(all_data, axis=0)  # Combine all batches

        # Write to a new NTuple or update the existing one
        with uproot.recreate(output_file) as out_file:
            out_file['Tuple/DecayTree'] = final_data
            print("Final NTuple written with all data.")

    print(f"All {batch_counter} batches processed.") 
    print(f"Total shape should be {total_shape}")     





'''
Method 2: single DecayTree that holds in memory teh batches
def process_file_in_batches(input_path, loading_variables, treename, prefix, abs_id, decay, batch_size, output_file):
    
    file_exists = os.path.exists(output_file)
    total_shape = 0
    combined_batches = []  # To accumulate processed batches in memory
    with uproot.open(input_path) as f:
        tree = f[treename]
        batch_counter = 0
        for batch in tree.iterate(loading_variables, library="pd", step_size=batch_size):
            batch_counter += 1
            print(f"Processing batch {batch_counter}...")
            
            if len(batch) == 0:
                break
            
            # Process the batch
            processed_batch = process_chunk(batch, prefix, abs_id)
            processed_batch.columns = processed_batch.columns.str.replace(f'{prefix}', 'B_', regex=False)
            total_shape += processed_batch.shape[0]
            
            # Append processed batch to the combined list
            combined_batches.append(processed_batch)
        
        # Combine all processed batches into a single DataFrame
        final_data = pd.concat(combined_batches)
        
        # Convert the combined DataFrame to an awkward array
        awkward_array = dataframe_to_awkward(final_data)
        
        # Write the final dataset to the ROOT file (overwriting the existing tree)
        with uproot.recreate(output_file) as out_file:
            out_file['Tuple/DecayTree'] = awkward_array
            print("Created single DecayTree with all batches.")

    print(f"All {batch_counter} batches processed.") 
    print(f"Total shape should be {total_shape}")     
'''
        
loading_variables = [
        #'B_BPVX',
        #'B_BPVY',
        'B_BPVZ',
        'B_END_VX',
        'B_END_VY',
        'B_END_VZ',
        'B_ENERGY',
        'B_ETA',
        'B_M',
        #'B_OLD_SSPionBDT_Mistag',
        'B_P',
        'B_PHI',
        'B_PT',
        'B_PX',
        'B_PY',
        'B_PZ',
        'B_TRUEID',
        'B_Tr_T_BPVIP',
        'B_Tr_T_BPVIPCHI2',
        'B_Tr_T_BPVX',
        'B_Tr_T_BPVY',
        'B_Tr_T_BPVZ',
        'B_Tr_T_CHI2DOF',
        'B_Tr_T_Charge',
        'B_Tr_T_ENERGY',
        'B_Tr_T_Eta',
        'B_Tr_T_GHOSTPROB',
        'B_Tr_T_ISMUON',
        'B_Tr_T_M',
        'B_Tr_T_MC_GD_GD_MOTHER_ID',
        #'B_Tr_T_MC_GD_GD_MOTHER_KEY',
        'B_Tr_T_MC_GD_MOTHER_ID',
        #'B_Tr_T_MC_GD_MOTHER_KEY',
        'B_Tr_T_MC_MOTHER_ID',
        #'B_Tr_T_MC_MOTHER_KEY',
        'B_Tr_T_MINIP',
        'B_Tr_T_MINIPChi2',
        #'B_Tr_T_OBJECT_KEY',
        'B_Tr_T_Origin_Flag', # tag codes in https://gitlab.cern.ch/lhcb/Rec/-/blob/7a77f1c3ba0e2384c2becbd7f4acf22e79b898f0/Phys/DaVinciMCKernel/include/Kernel/MCTaggingHelper.h#L17
        'B_Tr_T_P',
        'B_Tr_T_PIDK',
        'B_Tr_T_PIDP',
        'B_Tr_T_PIDe',
        'B_Tr_T_PIDmu',
        'B_Tr_T_PROBNN_E',
        'B_Tr_T_PROBNN_GHOST',
        'B_Tr_T_PROBNN_K',
        'B_Tr_T_PROBNN_MU',
        'B_Tr_T_PROBNN_P',
        'B_Tr_T_PROBNN_PI',
        'B_Tr_T_PT',
        'B_Tr_T_PX',
        'B_Tr_T_PY',
        'B_Tr_T_PZ',
        'B_Tr_T_Phi',
        'B_Tr_T_TRACKISLONG',
        'B_Tr_T_TRUEID',
        #'B_Tr_T_TRUEORIGINVERTEX_X',
        #'B_Tr_T_TRUEORIGINVERTEX_Y',
        #'B_Tr_T_TRUEORIGINVERTEX_Z',
        #'B_Tr_T_TRUEPRIMARYVERTEX_X',
        #'B_Tr_T_TRUEPRIMARYVERTEX_Y',
        #'B_Tr_T_TRUEPRIMARYVERTEX_Z',
        'B_Tr_T_X',
        'B_Tr_T_Y',
        'B_Tr_T_Z',
        'B_nPVs',
        'B_nTracks',
        'EVENTNUMBER',
        'RUNNUMBER']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add features used to select tracks and to train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being used', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')
    parser.add_argument('--batch_size', help='Size of the data batch to process at a time', type=int, default=250) #1000
    
    cfg = parser.parse_args()
    
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)

    B_abs_id_dic = {
    'Bs2DsPi': 531,
    'Bd2JpsiKst': 511,
    'Bu2JpsiK': 521,
    'Bd2DmPi': 511,
    'Bs2JpsiPhi': 531,
    }
    # drop the B mesons or other particles that are not of interest
    abs_id = B_abs_id_dic[cfg.evtType]
    '''
    if cfg.evtType=='Bs2DsPi':
        prefix = 'Bd' + "_"
    else:
        # Modify the prefix based on evtType
        prefix = cfg.evtType[:2] + "_"
    '''
    prefix = cfg.evtType[:2] + "_"
    # Replace B_ in the loading variables
    loading_variables_withPrefix = [var.replace("B_", prefix) for var in loading_variables]

    print(f'{loading_variables_withPrefix}')
    print('Started processing')
    process_file_in_batches(cfg.raw, loading_variables_withPrefix, cfg.treename, prefix, abs_id, cfg.evtType, cfg.batch_size, cfg.output)
    print(f'Modified NTuple processed and saved to {cfg.output}')
    print(f'Creation time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

