import pandas as pd
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from saver import Saver
import configParameters as config
import os

repoPath = '/ceph/users/molocco/classical-taggers/'
# Decay and tagger type are given as inputs by the user 
eventType = sys.argv[1]

# Get the raw root file 
# Charge and absID are needed for the label 
# List of daughters of the signal B
path_to_tuple = f'../Data/{config.sample_type}/1_raw/{eventType}/merged.root:Tuple/DecayTree;1'

if eventType == 'Bs2DsPi': 
    abs_id = 531
    daughters = ['Ds_P','piMinus_P','KPlus_P','KMinus_P']

elif eventType == 'Bd2JpsiKst':
    abs_id = 511
    daughters = ['muPlus_P','muMinus_P','Kstar_P','Jpsi_P']

elif eventType == 'Bu2JpsiK':
    B = 'B'
    abs_id = 521
    daughters = ['KPlus_P','muMinus_P','muPlus_P','Jpsi_P']

stepsize = 100000    #the Root file will gel load in chunks

def DeltaQ(df,Mass):
    E =np.sqrt( Mass**2 + df['B_Tr_T_PX']**2 + df['B_Tr_T_PY']**2 + df['B_Tr_T_PZ']**2)
    DeltaQ = np.sqrt( (E + df['B_ENERGY'])**2  - ((df['B_Tr_T_PX'] + df['B_PX'])**2 + (df['B_Tr_T_PY'] + df['B_PY'])**2 + (df['B_Tr_T_PZ'] + df['B_PZ'])**2 )   ) -df.B_M  - Mass
    return(DeltaQ)

start_time = time.time()
run_time = time.time()

loading_variables = ['B_nPVs','B_nTracks','B_TRUEID', 'B_Tr_T_Charge','B_Tr_T_P','B_PHI','B_Tr_T_Phi','B_BPVZ','B_BPVY','B_BPVX','B_Tr_T_BPVZ', \
                'B_Tr_T_BPVY','B_Tr_T_BPVX','B_ETA','B_Tr_T_Eta','B_Tr_T_BPVIP','B_Tr_T_PX','B_Tr_T_PY','B_Tr_T_PZ','B_P', \
                'B_ENERGY','B_PX','B_PY','B_PZ','B_Tr_T_TRUEID', 'B_PT', 'B_Tr_T_ENERGY', 'B_Tr_T_PT'  ,'B_M', 'B_Tr_T_ISMUON', \
                'B_END_VX','B_END_VY','B_END_VZ','B_Tr_T_X','B_Tr_T_Y','B_Tr_T_Z', 'B_Tr_T_M', 'B_Tr_T_PROBNN_MU', 'B_Tr_T_PROBNN_E', 'B_Tr_T_PIDK', 'B_Tr_T_PIDe', \
                'B_Tr_T_PIDmu', 'B_Tr_T_PIDP','B_Tr_T_CHI2DOF','B_Tr_T_PROBNN_P','B_Tr_T_PROBNN_K','B_Tr_T_PROBNN_PI','B_Tr_T_GHOSTPROB',\
                'B_Tr_T_BPVIPCHI2', 'B_Tr_T_TRACKISLONG', 'B_Tr_T_MINIP', 'B_Tr_T_MINIPChi2', 'B_Tr_T_PROBNN_GHOST', 'B_Tr_T_TRUEPRIMARYVERTEX_X', 'B_Tr_T_TRUEPRIMARYVERTEX_Y', 'B_Tr_T_TRUEPRIMARYVERTEX_Z',\
                'B_Tr_T_TRUEORIGINVERTEX_X', 'B_Tr_T_TRUEORIGINVERTEX_Y', 'B_Tr_T_TRUEORIGINVERTEX_Z', 'B_Tr_T_MC_MOTHER_ID', 'B_Tr_T_MC_MOTHER_KEY',\
                'B_Tr_T_MC_GD_MOTHER_ID', 'B_Tr_T_MC_GD_MOTHER_KEY', 'B_Tr_T_MC_GD_GD_MOTHER_ID', 'B_Tr_T_MC_GD_GD_MOTHER_KEY', 'B_Tr_T_OBJECT_KEY',\
                'B_OLD_SSPionBDT_Mistag', 'B_OLD_SSPionBDT_Decision']+daughters

print('Started reading')
for df in uproot.iterate(path_to_tuple, loading_variables, step_size=stepsize, library = 'pd'):
    
    print('-------------------Start Block-----------------------------')
    df.drop(df[abs(df['B_TRUEID']) != abs_id ].index , inplace = True) # drop the B mesons or other particles that are not of interest
    for daughter in daughters: #remove daughter
        df.drop(df.loc[df['B_Tr_T_P'] == df[daughter]].index, inplace = True)
    df.reset_index(inplace=True, drop = False) 
    #Add some needed features
    df.eval('B_Tr_T_cos_diff_Phi=cos(B_PHI-B_Tr_T_Phi)', inplace=True)
    df.eval('B_Tr_T_diff_z = abs(B_BPVZ - B_Tr_T_BPVZ)' , inplace = True)
    df.eval('B_Tr_T_PhiDistance =abs(B_PHI - B_Tr_T_Phi)' , inplace = True)  

    df.eval('B_Tr_T_DeltaR= (B_ETA - B_Tr_T_Eta)**2 + B_Tr_T_PhiDistance**2', inplace = True)
    df.eval('diff_P = abs(B_P - B_Tr_T_P)', inplace = True)
    df.eval('P_proj = B_ENERGY*B_Tr_T_ENERGY - (B_Tr_T_PX*B_PX + B_Tr_T_PY*B_PY +B_Tr_T_PZ*B_PZ ) ', inplace = True)
    df.eval('t = (B_END_VX**2 + B_END_VY**2 + B_END_VZ**2 - B_END_VX*B_Tr_T_X - B_END_VY*B_Tr_T_Y - B_END_VZ*B_Tr_T_Z) / (B_END_VX * B_Tr_T_PX + B_END_VY * B_Tr_T_PY + B_END_VZ * B_Tr_T_PZ)' , inplace = True)
    df.eval('EVIP = sqrt((B_Tr_T_X**2 + B_Tr_T_Y**2 + B_Tr_T_Z**2) + t**2 * (B_Tr_T_PX**2 + B_Tr_T_PY**2 + B_Tr_T_PZ**2) + 2*t*(B_Tr_T_X * B_Tr_T_PX + B_Tr_T_Y * B_Tr_T_PY + B_Tr_T_Z * B_Tr_T_PZ))', inplace = True)
    df.eval('B_Tr_T_absIP = abs(B_Tr_T_BPVIP)', inplace = True)
    
    df.eval('B_Tr_T_EtaDistance = abs(B_ETA - B_Tr_T_Eta)', inplace = True)
    df['B_Tr_T_DeltaQ_Pion'] = DeltaQ(df,139.5706)
    df['B_Tr_T_DeltaQ_Mu'] = DeltaQ(df,105.65837)
    df['B_Tr_T_DeltaQ_Electron'] = DeltaQ(df,0.51100)
    df['B_Tr_T_DeltaQ_Proton'] = DeltaQ(df,938.27208)
    df['B_Tr_T_DeltaQ_Kaon'] = DeltaQ(df,493.677)
    
    df.eval('B_Tr_T_Signal_TagPart_PT = sqrt((B_PX + B_Tr_T_PX) **2 + (B_PY + B_Tr_T_PY)**2)', inplace = True)
    df.eval('B_Tr_T_eoverP = B_Tr_T_Charge/B_Tr_T_P', inplace = True)
    df.eval('B_Tr_T_absID =abs(B_Tr_T_TRUEID)', inplace = True)

    df.eval('EVIP = log(EVIP)', inplace = True)
    df.eval('B_Tr_T_IPSig = sqrt(B_Tr_T_BPVIPCHI2)' , inplace = True)
    df.eval('P_proj = log(P_proj)', inplace = True)
    
    print(f'Block finished in {round(time.time() - run_time,2)}s')
    print('--------------------End Block------------------------------')
    print()

    run_time = time.time()

path = f'../Data/{config.sample_type}/2_added_features/{eventType}/notSelected.root'
with uproot.recreate(f'{path}') as f:
    f['DecayTree'] = df

print(f'Modified NTuple saved at {path}')
