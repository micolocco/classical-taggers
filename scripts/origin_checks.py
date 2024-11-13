import numpy as np 
import pandas as pd 
from IPython import embed
import uproot
import glob
import matplotlib.pyplot as plt

from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})

'''
python scripts/origin_checks.py >log_check_Bd_20.log
'''
def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    #if decay == 'Bs2DsPi':
    #    return 'Hlt2B2OC_BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    else:
        return 'Tuple/DecayTree'

if __name__ == '__main__':

    base_pattern = '/ceph/users/molocco/Data/withUT_MC_2024/2_added_features/'
    ##folders = ['Bs2DsPi']
    #folders = ['Bd2JpsiKst']
    #folders = ['Bu2JpsiK', 'Bd2JpsiKst', 'Bs2DsPi']
    folders = ['Bu2JpsiK', 'Bd2JpsiKst',]

   # loading_variables = ['B_Tr_T_absID','B_Tr_T_Origin_Flag', 'B_TRUEID', 'B_Tr_T_Charge', 'B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID']
    loading_variables = ['B_Tr_T_TRUEID','B_Tr_T_Origin_Flag', 'B_TRUEID', 'B_Tr_T_Charge', 'B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID']

    df = pd.DataFrame(columns=loading_variables)

    B_abs_id_dic = {
            'Bs2DsPi': 531,
            'Bd2JpsiKst': 511,
            'Bu2JpsiK': 521,
            'Bd2DmPi': 511,
            'Bs2JpsiPhi': 531,
            }

    for decay in folders:
        pattern = f'{base_pattern}/{decay}/*.root'
        root_files = []
        root_files.extend(glob.glob(pattern))
        treename = find_tree_name(decay)
        for f in root_files:
            print(f"Reading input file: {f}")
            with uproot.open("{}".format(f)) as _f:
                _df = _f[treename].arrays(loading_variables, library="pd")
                df = pd.concat([df, _df], ignore_index = True)
        

        # drop the B mesons or other particles that are not of interest
        abs_id = B_abs_id_dic[decay]
        df.drop(df[abs(df[f'B_TRUEID']) != abs_id ].index , inplace = True)
        df.reset_index(inplace=True, drop = False)
        print(f'{decay} candidates: {df.shape[0]}')      
    
        df.B_Tr_T_Origin_Flag.astype(int)
        df.eval(f'B_Tr_T_absID =abs(B_Tr_T_TRUEID)', inplace = True)

        conditions = [
        (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==2), # OSKaon
        (df.B_Tr_T_absID==13) & (df.B_Tr_T_Origin_Flag==2), # OSMuon
        (df.B_Tr_T_absID==11) & (df.B_Tr_T_Origin_Flag==2), # OSElectron
        (df.B_Tr_T_absID==211) & (df.B_Tr_T_Origin_Flag==1), # SSPion
    # ((df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1)) | ((df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1)), # SSProton and SSKaon
        (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==1), # SSProton
        (df.B_Tr_T_absID==321) & (df.B_Tr_T_Origin_Flag==1), # SSKaon
        (df.B_Tr_T_absID==2212) & (df.B_Tr_T_Origin_Flag==2), # OSProton
        ]
        
        particle_type = {"OSKaon":1,
                        "OSMuon":2,
                        "OSElectron":3,
                        "SSPion":4,
                        #"SSProton+SSKaon": 5,
                        "SSProton":5,
                        "SSKaon":6,
                        "OSProton":7}
        '''
        particle_type = {
                    "SSPion":1,
                    "SSProton":2,
                    "SSKaon":3,
                }
        '''
        df['ID_type'] = np.select(conditions, particle_type.values())
        df.loc[~df['ID_type'].isin(particle_type.values()), 'ID_type'] = 0
        # Assign the corresponding particle type
        df['particle'] = df['ID_type'].map({v: k for k, v in particle_type.items()})
        # If ID_type is not in particle_type values, set 'particle' to None
        df.loc[~df['ID_type'].isin(particle_type.values()), 'particle'] = 'not_taggingPart'
        df = df[df.ID_type!=0]
        
        #print(df[['ID_type','particle', 'B_TRUEID', 'B_Tr_T_Charge', 'B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID']].value_counts())
        
        df['sign_tag'] = (df['B_TRUEID']/abs(df['B_TRUEID'])) * df['B_Tr_T_Charge']
        df[['B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID']].astype(int)
        for key in particle_type:
            print(f'{key}')
            
            # Get the total count of tracks classified as the current particle type
            total_tracks = df[df.particle == key].shape[0]
            print(f"Total tracks classified as {key}: {total_tracks}")
            
            # Calculate the variety of MOTHER_IDs and their counts
            #variety = abs(df[(df.particle == key) & (df['sign_tag']==+1)].B_Tr_T_MC_MOTHER_ID.astype(int)).value_counts()
            variety = df[df.particle == key].copy()
            #variety[['B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID']] = abs(variety[['B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID']].astype(int))
            #variety = abs(df[(df.particle == key)][['B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', ]]).value_counts()
            variety_counts = variety[['B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_ID', 'sign_tag', 'B_TRUEID', 'B_Tr_T_Charge']].value_counts()
            #print(f"Variety of MOTHER_IDs: {len(variety)}")
            # Calculate the percentage for each MOTHER_ID
            variety_percentage = (variety_counts / total_tracks) * 100
            
            # Group by MOTHER_ID and calculate the mode of sign_tag
            #sign_tag_mode = df[df.particle == key].groupby(df.B_Tr_T_MC_MOTHER_ID.astype(int))['sign_tag'].agg(pd.Series.mode)
            
            # Combine the counts and percentages into a single DataFrame for better display
            combined = pd.DataFrame({'Count': variety_counts, 'Percentage': variety_percentage})
            # Name the first column as 'MOTHER_ID'
            combined.index.name = 'MOTHER_ID'
            # Print the combined DataFrame
            print(combined[:100].to_string(index=True, float_format="%.1f"))
        print('-----------------------------------------------------------------------------------------')
        print('\n')
            #print('Charge-B Flavour relation for MOTHER_ID=5:')
            #print(df[(df.particle == key) & (abs(df['B_Tr_T_MC_MOTHER_ID']).astype(int)==5)][['B_TRUEID', 'B_Tr_T_Charge']].value_counts())
            
    '''
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

    # Plot each histogram in its own subplot
    axs[0].plot(df[df.particle=='SSKaon'].B_Tr_T_MC_MOTHER_ID, marker='o', label="SSKaon",  color='m', lw=2)
    axs[0].legend()
    axs[0].set_title('SSKaon')

    axs[1].plot(df[df.particle=='SSPion'].B_Tr_T_MC_MOTHER_ID, marker='o', label="SSPion",  color='m', lw=2)
    axs[1].legend()
    axs[1].set_title('SSPion')

    axs[2].plot(df[df.particle=='SSProton'].B_Tr_T_MC_MOTHER_ID, marker='o', label="SSProton",  color='m', lw=2)
    axs[2].legend()
    axs[2].set_title('SSProton')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    #plt.show()
    '''