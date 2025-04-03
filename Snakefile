from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import os
from copy import deepcopy

import json

try:
    data = config['DATA']
    MC = config['MC']
    out = config['OUT']
    lowerMass = config['Mass_range_lower']
    upperMass = config['Mass_range_upper']

    
    repo = config['REPO']
except:
    raise RuntimeError("Make sure to specify snakemake config")

def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]

taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2DsPi': ['SSKaon'],
    'Bd2DmPi': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon',],
    'Bs2JpsiPhi': ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSProton', 'SSKaon']

    
}
# TO DO create a rules that copy the files from eos to the cluster



ntuples_eos_withUT = {
    'Bu2JpsiK': in_data(MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237568/0000/00237568_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237568/0000/00237568_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237568/0000/00237568_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237568/0000/00237568_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237568/0000/00237568_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000003_1.mc.root
'''.split('\n')),

    'Bd2JpsiKst': in_data(MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237614/0000/00237614_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237569/0000/00237569_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237569/0000/00237569_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237569/0000/00237569_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237569/0000/00237569_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237614/0000/00237614_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237614/0000/00237614_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237614/0000/00237614_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237614/0000/00237614_00000005_1.mc.root
'''.split('\n')),
    'Bd2DmPi': in_data(MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000003_1.mc.root
'''.split('\n')),
    'Bs2DsPi': in_data(MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000007_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237578/0000/00237578_00000008_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237585/0000/00237585_00000007_1.mc.root
'''.split('\n')),
    'Bs2JpsiPhi': in_data(MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226271/0000/00226271_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhMC/anaprod/lhcb/MC/Dev/MC.ROOT/00226273/0000/00226273_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000003_1.mc.root
'''.split('\n'))
}

#Data files
with open("data_calibration/block12_list.txt", "r") as f:
    files_s24c2 = f.readlines()
files_s24c2 = [line.strip() for line in files_s24c2][:80] ## TODO CHANGE TO ALL DATA FILES !!!!

raw_path = os.path.dirname(files_s24c2[0])

mc_ids = {}
for decay in taggers_conf.keys():
    path = join(MC, decay + '/')
    dec_ids = []
    for f in os.listdir(join(path)):
        if '.root' in f:
            dec_ids.append(f)
    mc_ids[decay] = dec_ids


raw_data = {
    'Bu2JpsiK': [join(out, 'Data/withUT_MC_2024/1_raw/Bu2JpsiK', os.path.basename(f)) for f in files_s24c2],
    'Bd2JpsiKst': [join(out, 'Data/withUT_MC_2024/1_raw/Bd2JpsiKst', os.path.basename(f)) for f in files_s24c2]
}

added_features_data = {}
for decay, path_list in raw_data.items():
    added_features_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        added_features_data[decay].update({tagger: [f.replace('1_raw', f'2_added_features').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

selected_data = {}
for decay, path_list in raw_data.items():
    selected_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        selected_data[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

tagged_data = {}
for decay, path_list in raw_data.items():
    tagged_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        tagged_data[decay].update({tagger: [f.replace('1_raw', f'5_weighted').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})




# TO DO add the other configurations e.g. noUT, openVELO etc
ntuples_raw_withUT_mc = deepcopy(ntuples_eos_withUT)
for k,v in ntuples_raw_withUT_mc.items():
    ntuples_raw_withUT_mc[k] = [f.replace(os.path.dirname(f), f'{out}/MC/withUT_MC_2024/1_raw/{k}') for f in v]

ntuples_added_features_withUT_mc = {}
for k,v in ntuples_raw_withUT_mc.items():
    ntuples_added_features_withUT_mc.update({k: [f.replace('1_raw', '2_added_features') for f in v]})

ntuples_selected_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_selected_withUT_mc.update({decay: {}})
    #print(decay, path_list)
    #print('-------')
    for tagger in taggers_conf[decay] :
        ntuples_selected_withUT_mc[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

ntuples_tagged_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_tagged_withUT_mc.update({decay: {}})
    for tagger in taggers_conf[decay]:
        # Filter the paths to only include those ending with '4_1.mc.root'
        filtered_paths = [
            f.replace('1_raw', '5_weighted')
             .replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN')
            for f in path_list if f.endswith('4_1.mc.root')
        ]
        ntuples_tagged_withUT_mc[decay].update({tagger: filtered_paths})

    #ntuples_selected_withUT_mc.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN_IPSig') for f in v]})
    #ntuples_selected_withUT_mc.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{taggers_conf[decay][i]}/{k}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in v]})
    #ntuples_selected_withUT_mc.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/only_OSKaon') for f in v]})

# Function to read paths from the generated file
def read_generated_paths(data, file):
    with open(file, 'r') as f:
        paths = [join(data,line.strip()) for line in f]
    return paths

def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    if decay == 'Bs2DsPi':
        return 'Hlt2B2OC_BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    if decay == 'Bu2JpsiK':
        return 'BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree'
    if decay == 'Bd2JpsiKst':
        return 'BdToJpsiKstar_JpsiToMuMu_Detached/DecayTree'
    else:
        return 'Tuple/DecayTree'

# Read the generated paths
generated_paths_SSPion = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSPion.txt'))
generated_paths_SSKaon = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSKaon.txt'))
generated_paths_SSProton = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSProton.txt'))
generated_paths_OSKaon = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSKaon.txt'))
generated_paths_OSElectron = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSElectron.txt'))
generated_paths_OSMuon = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSMuon.txt'))

rule all:
    input:
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs1024_simple_dm0.0001/signal_weights/ROC_TRAIN_VAL.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs1024_simple_dm0.0001/pdf_ratio/ROC_TRAIN_VAL.pdf',


        ## ntuples_tagged_withUT_mc['Bd2JpsiKst']['OSKaon'],
        ## ntuples_tagged_withUT_mc['Bd2JpsiKst']['OSElectron'],
        ## ntuples_tagged_withUT_mc['Bd2JpsiKst']['OSMuon'],
        ## ntuples_tagged_withUT_mc['Bd2JpsiKst']['SSProton'],
        ## ntuples_tagged_withUT_mc['Bd2JpsiKst']['SSPion'],



        #generated_paths_OSKaon,
        #generated_paths_SSPion,
        #generated_paths_SSProton,
        #generated_paths_OSElectron,
        #generated_paths_OSMuon,     

        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        #'/ceph/users/qfuehring/Data/savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs2048_simple_dm0.0/ROC_TRAIN_VAL.pdf',
        #'/ceph/users/qfuehring/Data/savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf',
        #ntuples_selected_withUT_mc['Bd2JpsiKst']['OSKaon'],
        #ntuples_selected_withUT_mc['Bd2JpsiKst']['OSElectron'],
        #ntuples_selected_withUT_mc['Bd2JpsiKst']['OSMuon'],
        #ntuples_selected_withUT_mc['Bd2JpsiKst']['SSPion'],
        #ntuples_selected_withUT_mc['Bd2JpsiKst']['SSProton'],
        #join(data, 'savedModels/withUT_MC_2024/combinations/Bd2JpsiKst/all_Calibration.pdf'),


        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),


        #ntuples_tagged_withUT_mc['Bd2JpsiKst']['OSKaon'],

        #'/ceph/users/qfuehring/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/balanced/treeSchema.pdf',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/unbalanced/treeSchema.pdf',

       # ntuples_added_features_withUT_mc['Bu2JpsiK'],
       # ntuples_added_features_withUT_mc['Bd2JpsiKst'],

        #ntuples_tagged_withUT_mc['Bs2JpsiPhi']['OSMuon'],
        #ntuples_tagged_withUT_mc['Bs2JpsiPhi']['OSKaon'],
        #ntuples_tagged_withUT_mc['Bs2JpsiPhi']['OSElectron'],
        ##ntuples_tagged_withUT_mc['Bs2JpsiPhi']['SSKaon'],
        #ntuples_tagged_withUT_mc['Bd2DmPi']['OSKaon'],
        #ntuples_tagged_withUT_mc['Bd2DmPi']['OSElectron'],
        #ntuples_tagged_withUT_mc['Bd2DmPi']['SSPion'],
        #ntuples_tagged_withUT_mc['Bd2DmPi']['OSMuon'],
        #ntuples_tagged_withUT_mc['Bd2DmPi']['SSProton'],




       # ntuples_selected_withUT_mc['Bu2JpsiK']['OSKaon'],
       # ntuples_selected_withUT_mc['Bu2JpsiK']['OSElectron'],


        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #         ntuples_selected_withUT_mc['Bd2JpsiKst']['SSProton'],

        #ntuples_added_features_withUT_mc['Bs2DsPi'],
        #ntuples_added_features_withUT_mc['Bu2JpsiK'],
        #ntuples_added_features_withUT_mc['Bd2JpsiKst']


        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bd2JpsiKst/00237614_00000002_1.mc.root',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00237567_00000001_1.mc.root',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00237568_00000001_1.mc.root'

        # generated_paths_OSKaon,
        # generated_paths_SSPion
        #generated_paths
        #ntuples_selected_withUT_mc['Bs2DsPi']['SSKaon'],
        #ntuples_selected_withUT_mc['Bd2JpsiKst'],
        #ntuples_selected_withUT_mc['Bu2JpsiK']['OSMuon'],
        #ntuples_selected_withUT_mc['Bu2JpsiK']['OSKaon'],
        #ntuples_selected_withUT_mc['Bu2JpsiK']['OSElectron'],
        #join(data, 'withUT_MC_2024/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf')
        #generated_paths_SSKaon,


decays_to_tag = ['Bu2JpsiK', 'Bd2JpsiKst']
all_configs = [f[:-5] for f in os.listdir(join(repo, "configs/")) if f.startswith("lr")]


from scripts.replace_path import seeds

rule get_optimized:
    input:
        script = join(repo, 'scripts/getOptimized.py'),
        modelPath = join(out, "{data_type}/savedModels/withUT_MC_2024"),
        

        tagging_infos = lambda wildcards : [out + "/{data_type}/" + f"savedModels/withUT_MC_2024/{decay}/{tagger}/{wildcards.cut_name}/union_PROBNN/{seed}/{config}/logit/taggingInfo_logit.json"
            for decay in decays_to_tag
            for tagger in taggers_conf[decay] 
            for seed in seeds
            for config in all_configs]
    log: 
        join(repo, "best_tagger_candidates/{cut_name}/{data_type, (MC|Data)}/candidatedTaggers_logit.log")
    output:
        join(repo, "best_tagger_candidates/{cut_name}/{data_type, (MC|Data)}/candidatedTaggers_logit.json")
    params:
    #     outpath = join(repo, "best_tagger_candidates")
    run:       
        cmd = [
            'python', input.script,
            f'--model_prePath {input.modelPath}',
            f'--cut {wildcards.cut_name}',
            f'--output {params.out}',
            '--features union_PROBNN',
            f'&> {log}',
            #'--features',
        ]
        shell(' '.join(cmd))

def get_raw_paths(decay, id, data_type):
    if data_type == 'MC':
        return join(MC, '{decay}/v1_taggers/{id}.root')
    elif data_type == 'Data':
        return join(data, f'{id[:8]}/{id[9:13]}' + '/{id}.root')
    else:
        print(f"data type is {data_type} instead of MC or Data. Somethings broken")
        raise RuntimeError

rule add_features:
# For some NTuples it's necessary to run locally (snakemake only, not on condor)
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        #script = join(repo, 'scripts/adding_features.py'), # Needed for Bs2JpsiPhi Bd2DmPi
        raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.id, wildcards.data_type)
    log: 
        join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output: 
        root =join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'), 
    resources:
        mem_mb = 20_000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue
        #request_disk = 50000
    run:
        tree = find_tree_name(wildcards.decay)
        dataCalib = '--data_calib' if wildcards.data_type == 'Data' else ''

        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
            f'{dataCalib}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

'''
rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        #data = glob.glob(f'/ceph/users/qfuehring/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root')
    output:
        pdf=join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf'),
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.log')
    params:
        target_path = lambda wildcards: join(data, f'{wildcards.sample_type}/DT_outputs/')
    resources:
        mem_mb = 40000, # Specify memory requirement in megabytes 
    run:
        cmd = [
            'python', input.script,
            '--target_path {params.target_path}',
            '&> {log}',
        ]
        shell(' '.join(cmd))
'''

rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        data = join(out, 'MC/{sample_type}/2_added_features'),
    output:
        pdf = join(out, 'MC/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.pdf'),
    log:
        join(out, 'MC/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.log')
    params:
        target_path = lambda wildcards: join(out, f'{wildcards.sample_type}/DT_outputs/notSamePV_noOSP')
    resources:
        mem_mb = 20_000, # Specify memory requirement in megabytes
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 8, # medium queue


    run:
        cmd = (
            f'python {input.script} '
            f'--base_pattern {input.data} '  # Pass input root files
            f'--target_path {params.target_path} '  # Pass the target path
            f'--balanced {wildcards.balanced} '  # Specify if classes are balance dor not
           # f'--unify_SS '  # Specify if SSKaon and SSProton should be unified in single class
            f'--BKG0 '
            f'&> {log}'  # Redirect stdout and stderr to log file
        )
        shell(cmd)

rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        added_features = join(out, '{data_type}/{sample_type}/2_added_features/{decay}/{id}.root'),
    output: join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id,.*}.root'),
    # output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root'),
    log: join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    resources:
        mem_mb = 2_000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue

    run:
        # tree = find_tree_name(wildcards.decay)
        data_calib = '--data_calib' if wildcards.data_type == 'Data' else ''
        BKG0 = '--BKG0' if wildcards.data_type == 'MC' else ''
       
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            #'--treename', tree,
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            data_calib, BKG0,
            '--repo', repo,
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule MC_Mass_Fit:
    input:
        script = join(repo, 'scripts/sweights.py'),
        selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
    output:
        mc_res = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/mc_fit/mc_res.json'),
    log:
        join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/mc_fit/mc_res.log'),
    resources:
        mem_mb = 10_000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue

    run:
        tree = find_tree_name(wildcards.decay)
        out_path = os.path.dirname(os.path.dirname(output.mc_res))


        cmd = [
            'python {input.script}',
            '--sim_files {input.selected}',
            '--range {lowerMass} {upperMass}', 
            '--treename "DecayTree;1"',# tree,
            '--simulation',
            '--output ', out_path,
            '--decayType {wildcards.decay}',
            '&> {log}'
        ]
        shell(' '.join(cmd))


rule add_weights:
    input:
        script = join(repo, 'scripts/sweights.py'),
        mc_res = join(out, 'Data/{sample_type}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/mc_fit/mc_res.json'),
        mc_selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
        data_selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in selected_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
    output:
        data_res = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/data_res.json'),
        sweights = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/sweights.root'),
    log:
        join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/data_res.log'),
    resources:
        mem_mb = 40_000, 
        MaxRunHours = 6, 
    params:
        tagged_prePath = join(out, 'Data/withUT_MC_2024/5_weighted/'),
        taggers = lambda wildcards: ' '.join(taggers_conf[wildcards.decay]),
        # tree = lambda wildcards: find_tree_name(wildcards.decay),
    run:
        out_path = os.path.dirname(os.path.dirname(output.data_res))
        cmd = [
            'python {input.script}',
            '--data_files {input.data_selected}',
            '--sim_files {input.mc_selected}',
            '--range {lowerMass} {upperMass}', 
            '--treename "DecayTree;1"',
            '--output', out_path,
            '--decayType {wildcards.decay}',
            '--features union_PROBNN',
            '--sim_fit {input.mc_res}',
            '--cut notSamePV_noOSP',
            '&> {log}'
        ]
        shell(' '.join(cmd))

# Define the function to extract the decay based on the tagger
def extract_decay(tagger):
    if tagger in ['OSKaon', 'OSMuon', 'OSElectron']:
        return 'Bu2JpsiK'
    elif tagger == 'SSKaon':
        return 'Bs2DsPi'
    elif tagger in ['SSPion', 'SSProton']:
        return 'Bd2JpsiKst'
    else:
        raise ValueError(f"Unknown tagger: {tagger}")

def extract_best(tagger, cut, data_type,link='logit'):
    #Read the best tagger candidate config from json file with the best hyperparameter combination

    with open(join(repo, f'best_tagger_candidates/{cut}/{data_type}/candidatedTaggers_{link}.json'), 'r') as f: # trainOn
    # with checkpoints.get_optimized.get(tagger = tagger, cut_name = cut).output[0].open() as f:
        data = json.load(f)

    seed = int(data[tagger]['seed'])
    lr = float(data[tagger]['learning_rate'])
    bs = int(data[tagger]['batch_size'])
    arch = data[tagger]['architecture']
    dm = float(data[tagger]['min_delta'])
    config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'
    return {'config':config, 'seed':seed, 'lr':lr, 'bs':bs, 'arch':arch, 'dm':dm, 'tagger':tagger, 'cut':cut, 'link':link}

rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        # best_tagger = join(repo, 'best_tagger_candidates/{cut_name}/{data_type, (MC|Data)}/candidatedTaggers_logit.json'),
        selected = join(out, '{data_type}/{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{features}/{id}.root'),
        model=lambda wildcards: join(out, '{data_type}' + f'/savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("config")}/model.pth'), 
        transformer=lambda wildcards: join(out,'{data_type}' + f'/savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("config")}/powerTransformer.pkl'), 
        scaler=lambda wildcards: join(out, '{data_type}' + f'/savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("config")}/st_scaler.pkl'), 
        config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("config")}.yaml'),
    output:
        root = join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/5_weighted/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id}.root'),
    log:
        join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/5_weighted/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id}.log'),
    resources:
        mem_mb = 40_000, 
        MaxRunHours = 3, 
    params:
        model_prePath =lambda wildcards: join(out, f'{wildcards.data_type}/savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/')
    run:
        print(f"Processing file: {wildcards.id}")
        print(f"Selected input file: {input.selected}")
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type)
        cmd = [
            'python', input.script,
            '--selected {input.selected}', 
            '--taggedData {output.root}',  
            '--model {input.model}',
            '--modelPrePath {params.model_prePath}',
            '--scaler {input.scaler}',
            '--transformer {input.transformer}',
            '--config {input.config}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            f'--arch {config.get("arch")}',
            f'--lr {config.get("lr")}',
            f'--seed {config.get("seed")}',
            '--repo', repo,
            '--cut {wildcards.cut_name}',
            '&>{log}'
        ]
        shell(' '.join(cmd))



rule combine_tagger: #TODO fix input for data
    input:
        tagged = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_tagged_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        script = join(repo, 'scripts/combineTagger.py'),
    output:
        pdf=join(out, '{data_type, (MC|Data)}/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_Calibration.pdf'),
    log: join(out, '{data_type, (MC|Data)}/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_log.log')
    resources:
        #mem_mb = 20_000, # Specify memory requirement in megabytes 
        ##gpus = 1,
        #OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        #MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(out, f'{wildcards.data_type}/savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
    run:
        cmd = [
            'python', input.script,
            '--tagger '
            '--tagged {input.tagged}',
            '--target_path {params.target_path}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule gen_configs:
    input:
        script = join(repo, "scripts/generate_configFiles.py"),

    output:
        join(repo, 'configs/configs/lr{lr,(0.001|0.01|0.1)}_bs{bs,(32|128|1024|2048)}_{a, (simple|complex)}_dm{dm, (0.0|0.0001|0.001|0.01)}.yaml'),
    log: 
        join(repo, 'configs/generate_configs{lr,(0.001|0.01|0.1)}_bs{bs,(32|128|1024|2048)}_{a, (simple|complex)}_dm{dm, (0.0|0.0001|0.001|0.01)}.log'),
    run:
        cmd = [
            'python', input.script,
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule train_tagger_MC:
    input:
        #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT_mc[f'{wildcards.decay}']],
        #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT_mc[f'{wildcards.decay}']],
        selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
        script = join(repo, 'scripts/pipeline.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        pdf=join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/ROC_TRAIN_VAL.pdf'),
        model=join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/model.pth'),
        scaler=join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/st_scaler.pkl'),
        transformer=join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/powerTransformer.pkl'),
        taggingInfo=join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/logit/taggingInfo_logit.json'),
    log: 
        join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training_log.log')
    resources:
        mem_mb = 20_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        target_path = lambda wildcards: join(out, f'MC/savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
    run:
        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--target_path {params.target_path}',
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--config {input.config}',
            '--decayType {wildcards.decay}',
            '--repo', repo,
            #'--clean',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule train_tagger_data:
    input:
        # selected = lambda wildcards: [
        #     f.replace('cutName', f'{wildcards.cut_name}')
        #     for f in selected_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        # ],
        weighted = join(out, 'Data/{sample_type}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/sweights.root'),

        script = join(repo, 'scripts/pipeline.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        pdf=join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/ROC_TRAIN_VAL.pdf'),
        model=join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/model.pth'),
        scaler=join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/st_scaler.pkl'),
        transformer=join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/powerTransformer.pkl'),
        taggingInfo=join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/logit/taggingInfo_logit.json'),
    log: 
        join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio)}/training_log.log')
    resources:
        mem_mb = 40_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        target_path = lambda wildcards: join(out, f'Data/savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/{wildcards.weight_type}')
    run:
        cmd = [
            'python', input.script,
            # '--selected {input.selected}',
            '--selected {input.weighted}',
            '--target_path', params.target_path,
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decayType {wildcards.decay}',
            '--weight_type {wildcards.weight_type}',
            '--repo', repo,
            '--train_on_data',
            #'--clean',
            '&> {log}',
        ]
        shell(' '.join(cmd))

# rule calibrate_tagger:
#     input:
#         join(repo,'best_tagger_candidates/{decay}/{tagger}/{cut_name}/{config}/candidatedTaggers_logit.json'), 
    
#         selected = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']],
#         script = join(repo, 'scripts/calibration.py'),
#         #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT_mc[f'{wildcards.decay}']],
#         #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT_mc[f'{wildcards.decay}']],
#     output:
#         pdf=join(out, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{config}/Normalized_TagDec.pdf'),
        
#     log: 
#         join(out, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{config}/calibration_log.log')
#     # resources:
#     #     mem_mb = 128000, # Specify memory requirement in megabytes 
#     #     #gpus = 1,
#     #     MaxRunHours = 4,
#     #     #request_disk = 1024000
#     params:
#         config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,decay=wildcards.decay,data_type=wildcards.data_type).get("config")}.yaml'),
#         target_path = lambda wildcards: join(out, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/')
#     run:
#         cmd = [
#             'python', input.script,
#             '--target_path {params.target_path}',
#             '--tagger {wildcards.tagger}',
#             #'--seed {wildcards.seed}',
#             '--config {params.config}',
#             '--decayType {wildcards.decay}',
#             '--repo', repo,
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))
