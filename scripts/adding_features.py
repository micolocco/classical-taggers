import pandas as pd
import uproot
import numpy as np
import os
import argparse
from tqdm import tqdm

loading_variables =[
    'B_BPVX',
    'B_BPVY',
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
    #'B_Tr_T_BPVX',
    #'B_Tr_T_BPVY',
    'B_Tr_T_BPVZ',
    'B_Tr_T_CHI2DOF',
    'B_Tr_T_Charge',
    'B_Tr_T_ENERGY',
    'B_Tr_T_Eta',
    'B_Tr_T_GHOSTPROB',
    'B_Tr_T_ISMUON',
    'B_Tr_T_M',
    #'B_Tr_T_MC_GD_GD_MOTHER_ID',
    #'B_Tr_T_MC_GD_GD_MOTHER_KEY',
    #'B_Tr_T_MC_GD_MOTHER_ID',
    #'B_Tr_T_MC_GD_MOTHER_KEY',
    #'B_Tr_T_MC_MOTHER_ID',
    #'B_Tr_T_MC_MOTHER_KEY',
    'B_Tr_T_MINIP',
    'B_Tr_T_MINIPChi2',
    'B_Tr_T_OBJECT_KEY',
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
    # 'B_Tr_T_TRUEORIGINVERTEX_X',
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

def read_data(file, treename, loading_variables, max_events=None, batch_size=100):
    df = pd.DataFrame()
    with uproot.open("{}".format(cfg.raw)) as f:
        tree = f[cfg.treename]
        num_events = tree.num_entries
        max_events = None
        if max_events and max_events < num_events:
            num_events = max_events
        print(f'Reading {num_events} entries...')
        batch_size = 100
        num_batches = (num_events + batch_size) // batch_size
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            stop = min((i + 1) * batch_size, num_events)
            batch_data = tree.arrays(
                expressions=loading_variables, library="pd", entry_start=start, entry_stop=stop)
            df = pd.concat([df, batch_data])
    return df

df_save = pd.DataFrame(columns=loading_variables)

def DeltaQ(df,Mass):
    E =np.sqrt( Mass**2 + df['B_Tr_T_PX']**2 + df['B_Tr_T_PY']**2 + df['B_Tr_T_PZ']**2)
    DeltaQ = np.sqrt( (E + df['B_ENERGY'])**2  - ((df['B_Tr_T_PX'] + df['B_PX'])**2 + (df['B_Tr_T_PY'] + df['B_PY'])**2 + (df['B_Tr_T_PZ'] + df['B_PZ'])**2 )   ) -df.B_M  - Mass
    return(DeltaQ)

# Phi distance definition from https://gitlab.cern.ch/lhcb/Phys/-/blob/run2-patches/Phys/FlavourTagging/src/Utils/TaggingHelpers.cpp?ref_type=heads#L43
def min_dPhi(df):
    df.eval('B_Tr_T_cos_Phi=cos(B_Tr_T_Phi)', inplace=True)
    df.eval('B_Tr_T_sin_Phi=sin(B_Tr_T_Phi)', inplace=True)
    df.eval('B_cos_Phi=cos(B_PHI)', inplace=True)
    df.eval('B_sin_Phi=sin(B_PHI)', inplace=True)
    df.eval('x_arctan=(B_Tr_T_cos_Phi*B_sin_Phi) - (B_cos_Phi*B_Tr_T_sin_Phi)', inplace=True)
    df.eval('y_arctan=(B_Tr_T_cos_Phi*B_cos_Phi) + (B_sin_Phi*B_Tr_T_sin_Phi)', inplace=True)
    df.eval('B_Tr_T_PhiDistance = arctan2(x_arctan, y_arctan)', inplace=True, engine='python')
    # A bit of a hack to add the minimum distance
    _df = df.groupby('entry').apply(lambda group: np.min(np.abs(group['B_Tr_T_PhiDistance']))).reset_index(name='B_Tr_T_minPhiDistance')
    df = pd.merge(df, _df, on='entry', how='left')
    df.drop(['B_Tr_T_cos_Phi', 'B_Tr_T_sin_Phi', 'B_cos_Phi', 'B_sin_Phi', 'x_arctan', 'y_arctan'], axis=1)
    return df

B_abs_id_dic = {
    'Bs2DsPi': 531,
    'Bd2JpsiKst': 511,
    'Bu2JpsiK': 521,
    'Bd2DmPi': 511,
    'Bs2JpsiPhi': 531,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add features used to select tracks and to train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--raw', help='Raw file', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--evtType', help='Decay which is being useed', type=str, choices=('Bs2DsPi', 'Bd2JpsiKst', 'Bu2JpsiK', 'Bd2DmPi', 'Bs2JpsiPhi'))
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='Tuple/DecayTree')

    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    print('Started reading')
    df = read_data(cfg.raw, cfg.treename, loading_variables)

    # drop the B mesons or other particles that are not of interest
    abs_id = B_abs_id_dic[cfg.evtType]
    df.drop(df[abs(df['B_TRUEID']) != abs_id ].index , inplace = True)
    df.reset_index(inplace=True, drop = False)
    # Add some needed features
    # A bit of a hack to add the minimum distance
    df = min_dPhi(df)
    df.eval('B_Tr_T_cos_PhiDistance=cos(B_Tr_T_PhiDistance)', inplace=True)
    df.eval('B_Tr_T_diff_z = abs(B_BPVZ - B_Tr_T_BPVZ)' , inplace = True)
    df.eval('B_Tr_T_DeltaR= (B_ETA - B_Tr_T_Eta)**2 + B_Tr_T_PhiDistance**2', inplace = True)
    df.eval('diff_P = abs(B_P - B_Tr_T_P)', inplace = True)
    df.eval('P_proj = B_ENERGY*B_Tr_T_ENERGY - (B_Tr_T_PX*B_PX + B_Tr_T_PY*B_PY +B_Tr_T_PZ*B_PZ ) ', inplace = True)
    df.eval('t = (B_END_VX**2 + B_END_VY**2 + B_END_VZ**2 - B_END_VX*B_Tr_T_X - B_END_VY*B_Tr_T_Y - B_END_VZ*B_Tr_T_Z) / (B_END_VX * B_Tr_T_PX + B_END_VY * B_Tr_T_PY + B_END_VZ * B_Tr_T_PZ)' , inplace = True)
    df.eval('EVIP = sqrt((B_Tr_T_X**2 + B_Tr_T_Y**2 + B_Tr_T_Z**2) + t**2 * (B_Tr_T_PX**2 + B_Tr_T_PY**2 + B_Tr_T_PZ**2) + 2*t*(B_Tr_T_X * B_Tr_T_PX + B_Tr_T_Y * B_Tr_T_PY + B_Tr_T_Z * B_Tr_T_PZ))', inplace = True)
    df.eval('B_Tr_T_absIP = abs(B_Tr_T_BPVIP)', inplace = True)
    df.B_Tr_T_Origin_Flag.astype(int)
    df.eval('B_Tr_T_EtaDistance = abs(B_ETA - B_Tr_T_Eta)', inplace = True)
    df['B_Tr_T_DeltaQ_Pion'] = DeltaQ(df,139.5706)
    df['B_Tr_T_DeltaQ_Muon'] = DeltaQ(df,105.65837)
    df['B_Tr_T_DeltaQ_Electron'] = DeltaQ(df,0.51100)
    df['B_Tr_T_DeltaQ_Proton'] = DeltaQ(df,938.27208)
    df['B_Tr_T_DeltaQ_Kaon'] = DeltaQ(df,493.677)
    df.eval('B_Tr_T_Signal_TagPart_PT = sqrt((B_PX + B_Tr_T_PX) **2 + (B_PY + B_Tr_T_PY)**2)', inplace = True)
    df.eval('B_Tr_T_eoverP = B_Tr_T_Charge/B_Tr_T_P', inplace = True)
    df.eval('B_Tr_T_absID =abs(B_Tr_T_TRUEID)', inplace = True)
    df.eval('EVIP = log(EVIP)', inplace = True)
    df.eval('B_Tr_T_BVIPSig = sqrt(B_Tr_T_BPVIPCHI2)' , inplace = True) # IPSig == IPErr
    df.eval('P_proj = log(P_proj)', inplace = True)
    df.eval('B_Tr_T_atanPT_PZ = arctan2(B_Tr_T_PT, B_Tr_T_PZ)', engine='python', inplace=True)

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with uproot.recreate(cfg.output) as f:
        f[cfg.treename] = df

    print(f'Modified NTuple saved at {cfg.output}')
