from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import os
import re
from copy import deepcopy
import yaml
import json

try:
    data = config['DATA']
    MC = config['MC']
    out = config['OUT']
    lowerMass = config['Mass_range_lower']
    upperMass = config['Mass_range_upper']
    repo = config['REPO']

    batched = config['use_batched'] 
except:
    raise RuntimeError("Make sure to specify snakemake config")

def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]

taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2DsPi': ['SSKaon'],
    # 'Bd2DmPi': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon',],
    # 'Bs2JpsiPhi': ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSProton', 'SSKaon']

    
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
#     'Bd2DmPi': in_data(MC, '''
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000002_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000003_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000002_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000003_1.mc.root
# '''.split('\n')),
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
#     'Bs2JpsiPhi': in_data(MC, '''
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000002_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000003_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226271/0000/00226271_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhMC/anaprod/lhcb/MC/Dev/MC.ROOT/00226273/0000/00226273_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000001_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000002_1.mc.root
#         root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000003_1.mc.root
# '''.split('\n'))
}

#Data files
with open("data_calibration/block12_list.txt", "r") as f:
    files_s24c2 = f.readlines()
files_s24c2 = [line.strip() for line in files_s24c2]#[:100]
raw_path = os.path.dirname(files_s24c2[0])
data_ids = [os.path.basename(i)[:-5] for i in files_s24c2]

mc_ids = {}
for decay in taggers_conf.keys():
    path = join(MC, 'withUT_MC_2024/1_raw/' + decay + '/')

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
    added_features_data.update({decay: [f.replace('1_raw', '2_added_features') for f in path_list]})

selected_data = {}
for decay, path_list in raw_data.items():
    selected_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        selected_data[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

weighted_data = {}
for decay, path_list in raw_data.items():
    weighted_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        weighted_data[decay].update({tagger: [f.replace('1_raw', f'1_weighted').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

train_split_data = {}
for decay, path_list in raw_data.items():
    train_split_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        train_split_data[decay].update({tagger: [f.replace('1_raw', f'5_split').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN/train') for f in path_list]})

tagged_data = {}
for decay, path_list in raw_data.items():
    tagged_data.update({decay: {}})
    for tagger in taggers_conf[decay] :
        tagged_data[decay].update({tagger: [f.replace('1_raw', f'6_tagged').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})



# TO DO add the other configurations e.g. noUT, openVELO etc
ntuples_raw_withUT_mc = deepcopy(ntuples_eos_withUT)
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_raw_withUT_mc[decay] = [f.replace(os.path.dirname(f), f'{out}/MC/withUT_MC_2024/1_raw/{decay}') for f in path_list]

ntuples_added_features_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_added_features_withUT_mc.update({decay: [f.replace('1_raw', '2_added_features') for f in path_list]})

ntuples_selected_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_selected_withUT_mc.update({decay: {}})
    #print(decay, path_list)
    #print('-------')
    for tagger in taggers_conf[decay] :
        ntuples_selected_withUT_mc[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

ntuples_train_split_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_train_split_withUT_mc.update({decay: {}})
    #print(decay, path_list)
    #print('-------')
    for tagger in taggers_conf[decay] :
        ntuples_train_split_withUT_mc[decay].update({tagger: [f.replace('1_raw', f'5_split').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN/train') for f in path_list]})

ntuples_tagged_withUT_mc = {}
for decay, path_list in ntuples_raw_withUT_mc.items():
    ntuples_tagged_withUT_mc.update({decay: {}})
    for tagger in taggers_conf[decay]:
        # Filter the paths to only include those ending with '4_1.mc.root'
        filtered_paths = [
            f.replace('1_raw', '6_tagged')
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
generated_paths_SSPion     = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSPion.txt'    ))
generated_paths_SSKaon     = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSKaon.txt'    ))
generated_paths_SSProton   = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_SSProton.txt'  ))
generated_paths_OSKaon     = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSKaon.txt'    ))
generated_paths_OSElectron = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSElectron.txt'))
generated_paths_OSMuon     = read_generated_paths(out,join(repo,'paths_for_snakemake/generated_paths_OSMuon.txt'    ))

#Define a hyperparameter chunk, used for parallelization of the training during hyperparameter optimization

with open(join(repo,'configs/hyperpar_intervals.yaml'), 'r') as file:
    intervals = yaml.safe_load(file)


intervals = {key[1:]: value for key, value in intervals.items()}


hyper_par_chunk = expand('nL{nL}_nN{nN}', nL=intervals['numlayers'], nN=intervals['numneurons'])
weights = [
    'ones',
    'signal_weights',
    'pdf_ratio'
]

wildcard_constraints:
    sample_type = '(withUT_MC_2024|noUT_MC_2024)',
    data_type   = '(MC|Data)',
    decay       = '(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)',
    tagger      = '(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)',
    weight      = '|'.join(weights),
    data_type_or_adapted = '(Data|MC|domain_adapted)',
    cut_name = "[^/]+", #don't allow slashes in wildcards to avoid problems with paths
    features = "[^/]+",
    seed = '[^/]+',
    config = '[^/]+',

#In how many splits the combined DataFrame should be split when using domain adaptation
combined_df_n_splits = 20

rule all:
    input:
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels_micol/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels_micol/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',

        

        # expand('/ceph/users/togasa/FlavourTagging/NTuples/domain_adapted/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs4096_nL6_nN256_alpha{alpha}/testing/Data/logit/taggingInfo_logit.json',
        #        alpha = [0, 0.01, 0.1, 0.5, 1]),
        # expand('/ceph/users/togasa/FlavourTagging/NTuples/domain_adapted/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs4096_nL6_nN256_alpha{alpha}/testing/MC/logit/taggingInfo_logit.json',
        #        alpha = [0, 0.01, 0.1, 0.5, 1]),
        # expand("/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_{alpha}/{id}.root",
        #     id = data_ids, alpha = [0, 0.01, 0.1, 0.5, 1]),
        # expand("/ceph/users/togasa/FlavourTagging/NTuples/MC/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_{alpha}/{id}",
        #     id = [basename(f) for f in ntuples_train_split_withUT_mc['Bu2JpsiK']['OSKaon']],alpha = [0, 0.01, 0.1, 0.5, 1]),
        


        # "/ceph/users/togasa/FlavourTagging/NTuples/MC/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_DA_0.5/00237567_00000001_1.mc.root"

        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels_micol/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/testing/Data/logit/taggingInfo_logit.json',

        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels_micol/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels_micol/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels_micol/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/model.pth'
        # # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/model.pth'
        # # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/model.pth'
    



        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron_Run3_Calibration.pdf',
        
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.001_bs2048_simple_dm0.001/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs1024_simple_dm0.0001/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs1024_simple_dm0.001/testing/Data/logit/taggingInfo_logit.json',

        # expand("/ceph/users/togasa/FlavourTagging/NTuples/Data/withUT_MC_2024/6_tagged/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/trained_{trained_on}/{id}.root",
        #     trained_on = ['Data', 'MC'],
        #     tagger = ['OSKaon', 'OSElectron', 'OSMuon'],
        #     id = data_ids,),
        
        # expand("/ceph/users/togasa/FlavourTagging/NTuples/MC/withUT_MC_2024/6_tagged/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/trained_{trained_on}/{id}",
        #     trained_on = ['Data', 'MC'],
        #     id = [basename(f) for f in ntuples_train_split_withUT_mc['Bu2JpsiK']['OSKaon']],),
        # expand("/ceph/users/togasa/FlavourTagging/NTuples/MC/withUT_MC_2024/6_tagged/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/trained_{trained_on}/{id}",
        #     trained_on = ['Data', 'MC'],
        #     id = [basename(f) for f in ntuples_train_split_withUT_mc['Bu2JpsiK']['OSElectron']],),
        # expand("/ceph/users/togasa/FlavourTagging/NTuples/MC/withUT_MC_2024/6_tagged/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/trained_{trained_on}/{id}",
        #     trained_on = ['Data', 'MC'],
        #     id = [basename(f) for f in ntuples_train_split_withUT_mc['Bu2JpsiK']['OSMuon']],),
        
        

        
        
                
        

def kernel_available():
    """
    Check if the ceph-kernel is mounted.
    """
    return os.path.exists('/ceph-kernel/users')

def path_to_kernel(path):
    if isinstance(path, str):
        return path.replace("ceph", "ceph-kernel")
    elif isinstance(path, list):
        return [path.replace("ceph", "ceph-kernel") for path in path]
    else:
        raise TypeError("Input must be a string or a list of strings.")

def copy_to_scratch(paths):
    scratch_paths = []


    # check whether ceph-kernel is mounted. If yes, copy from there
    to_replace = 'ceph/users'
    if kernel_available():
        paths = path_to_kernel(paths)
        to_replace = 'ceph-kernel/users'
    
    if 'users' in paths[0]:
        replace_with = 'scratch'
    else:
        to_replace = to_replace[:-6]
        replace_with = 'scratch/togasa'

    
    
    for path in paths:
        path_scratch = path.replace(to_replace, replace_with)
        #make sure path on scratch exists or is created
        shell(f'mkdir -p {os.path.dirname(path_scratch)}')
        #Copy data from ceph to scratch
        shell(f'cp -u {path} {path_scratch}')
        scratch_paths.append(path_scratch)
    return scratch_paths




decays_to_tag = ['Bu2JpsiK', 'Bd2JpsiKst']
all_configs = [f[:-5] for f in os.listdir(join(repo, "configs/")) if f.startswith("lr")]


rule train_signal_classifier:
    input:
        script = join(repo, 'scripts/train_BDT.py'),
        data = lambda wildcards: [get_raw_paths(wildcards.decay, id, 'Data') for id in data_ids], 
        mc = lambda wildcards: [get_raw_paths(wildcards.decay, id[:-5], 'MC') for id in mc_ids[wildcards.decay]], 
    output:
        BDT = join(out, 'Data/{sample_type}/1_weighted/{decay}/BDT/bdt_model.pkl')
    log:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/BDT/BDT_train.log')
    resources:
        max_retries=0,
        mem_mb = 20_000,
        MaxRunHours = 4,
    threads:
        8,
    run:
        out_path = os.path.dirname(output.BDT)
        tree = find_tree_name(wildcards.decay)

        if kernel_available():
            data = path_to_kernel(input.data)
            mc   = path_to_kernel(input.mc)
        else:
            data = input.data
            mc   = input.mc

        cmd = [
            'python', input.script,
            '--real_data', ' '.join(data),
            '--mc_data', ' '.join(mc),
            '--target_path', out_path,
            '--treename', tree,
            '--decay_type {wildcards.decay}',
            '--massname B_DTF_PV_Jpsi_MASS',
            '--num_threads {threads}',
            '&> {log}',
        ]


        shell(' '.join(cmd))

def get_raw_paths(decay, id, data_type):
    if data_type == 'MC':
        return join(MC, f'withUT_MC_2024/1_raw/{decay}/{id}.root')
    elif data_type == 'Data':
        return join(data, f'{id[:8]}/{id[9:13]}' + f'/{id}.root')
    else:
        print(f"data type is {data_type} instead of MC or Data. Somethings broken")
        raise RuntimeError

rule add_features:
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        #script = join(repo, 'scripts/adding_features.py'), # Needed for Bs2JpsiPhi Bd2DmPi
        # raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.id, wildcards.data_type)
        weighted = lambda wildcards: join(out, 'Data/{sample_type}/1_weighted/{decay}/weighted_files/{id}.root') 
                                    #  if wildcards.data_type == 'Data' else get_raw_paths(wildcards.decay, wildcards.id, 'MC'),
                                     if wildcards.data_type == 'Data' else  join(MC, '{decay}/v1_taggers/{id}.root')
    log:                            
        join(out, '{data_type}/{sample_type}/2_added_features/{decay}/.{id,.*}.log')
    output: 
        root =join(out, '{data_type}/{sample_type}/2_added_features/{decay}/{id,.*}.root'), 
    resources:
        max_retries=0,
        mem_mb = lambda wildcards: 150_000 if wildcards.data_type == 'MC' else 20_000, # MC needs unreasonable amounts of memory TODO FIX??
        MaxRunHours = 2, # short queue
    run:
        tree = find_tree_name(wildcards.decay) if wildcards.data_type == 'MC' else '"DecayTree;1"'
        dataCalib = '--data_calib' if wildcards.data_type == 'Data' else ''

        cmd = [
            'python', input.script,
            # '--raw {input.raw}',
            '--raw {input.weighted}',
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
        pdf=join(data, '{sample_type}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf'),
    log: join(data, '{sample_type}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.log')
    params:
        target_path = lambda wildcards: join(data, f'{wildcards.sample_type}/DT_outputs/')
    resources:
        max_retries=0,
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
        pdf = join(out, 'MC/{sample_type}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.pdf'),
    log:
        join(out, 'MC/{sample_type}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.log')
    params:
        target_path = lambda wildcards: join(out, f'{wildcards.sample_type}/DT_outputs/notSamePV_noOSP')
    resources:
        max_retries=0,
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
        # added_features = join(out, '{data_type}/{sample_type}/2_added_features/{decay}/{id}.root'),
        added_features = join(out, '{data_type}/{sample_type}/2_added_features/{decay}/{id}.root'),

    output: join(out, '{data_type}/{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{features}/{id,.*}.root'),
    # output: join(data, '{sample_type}/3_selected/{decay}/{cut_name}/{id,.*}.root'),
    log: join(out, '{data_type}/{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{features}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    resources:
        max_retries=0,
        mem_mb = 20_000, # Specify memory requirement in megabytes
        MaxRunHours = 2, # short queue
    run:
        # tree = find_tree_name(wildcards.decay)
        data_calib = '--data_calib' if wildcards.data_type == 'Data' else ''
        BKG0 = '--BKG0' if wildcards.data_type == 'MC' else ''
       
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            # '--treename', tree,
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--evtType {wildcards.decay}',
            data_calib, BKG0,
            '--repo', repo,
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule MC_Mass_Fit:
    input:
        script = join(repo, 'scripts/mass_fits.py'),
        # selected = lambda wildcards: [
        #     f.replace('cutName', f'{wildcards.cut_name}')
        #     for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcard'] 
        #     if not f.endswith('4_1.mc.root')
        # ],
        raw = lambda wildcards: [get_raw_paths(wildcards.decay, id[:-5], 'MC') for id in mc_ids[wildcards.decay]],#[:1],
        BDT = join(out, 'Data/{sample_type}/1_weighted/{decay}/BDT/bdt_model.pkl')
    output:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/mc_fit/mc_res_before_cut.json'),
        mc_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/mc_fit/mc_res_after_cut.json'),
    log:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/mc_fit/mc_res.log'),
    resources:
        max_retries=0,
        mem_mb = 20_000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue
    threads:
        8,
    run:
        tree = find_tree_name(wildcards.decay)
        out_path = os.path.dirname(os.path.dirname(output.mc_res))

        # raw = [f.replace(' /ceph/FlavourTagging/NTuples/Run3/MC/withUT_MC_2024/1_raw', '/scratch/togasa/MC/') for f in input.raw]
        if kernel_available():
            raw = path_to_kernel(input.raw)
        else:
            raw = input.raw


        cmd = [
            'python {input.script}',
            # '--sim_files {input.selected}',
            '--sim_files', ' '.join(raw),
            '--range {lowerMass} {upperMass}', 
            '--treename', tree,
            '--simulation',
            '--BDT {input.BDT}',
            '--output', out_path,
            '--decayType {wildcards.decay}',
            '--num_threads {threads}',
            '&> {log}'
        ]
        shell(' '.join(cmd))
        

rule thesis_mass_fit: #Plots without pulls for thesis 
    input:
        script = join(repo, 'scripts/no_pull_mass_plots.py'),
        mc_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/mc_fit/mc_res_before_cut.json'),
        data_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/data_fit/data_res_after_cut.json'),
        
        data_raw = lambda wildcards: [get_raw_paths(wildcards.decay, id, 'Data') for id in data_ids],#[:1],

        BDT = join(out, 'Data/{sample_type}/1_weighted/{decay}/BDT/bdt_model.pkl')
    output:
        png = join(out, 'Data/{sample_type}/1_weighted/{decay}/data_no_pull_plot/fit_after_cut.pdf'),
    log:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/data_no_pull_plot/no_pulls.log'),
    resources:
        max_retries=0,
        mem_mb = 20_000, 
        MaxRunHours = 6,
    threads:
        8,
    run:
        tree = find_tree_name(wildcards.decay)

        out_path = os.path.dirname(os.path.dirname(output.png))

        #Use ceph-kernel if available to increase file reading performance
        if kernel_available():
            raw = path_to_kernel(input.data_raw)
        else:
            raw = input.data_raw



        cmd = [
            'python {input.script}',
            '--data_files ', ' '.join(raw),
            '--range {lowerMass} {upperMass}', 
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--treename', tree,
            '--output', out_path,
            '--decayType {wildcards.decay}',
            '--sim_fit {input.mc_res}',
            '--data_fit {input.data_res}',
            '--cut notSamePV_noOSP',
            '--BDT {input.BDT}',
            '--num_threads {threads}',
            '&> {log}'
        ]
        shell(' '.join(cmd))

rule data_Mass_Fit: 
    input:
        script = join(repo, 'scripts/mass_fits.py'),
        mc_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/mc_fit/mc_res_before_cut.json'),


        data_raw = lambda wildcards: [get_raw_paths(wildcards.decay, id, 'Data') for id in data_ids],#[:1],

        BDT = join(out, 'Data/{sample_type}/1_weighted/{decay}/BDT/bdt_model.pkl')
    output:
        # join(out, 'Data/{sample_type}/1_weighted/{decay}/{cut_name}/{features}/data_fit/model.dll'),
        data_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/data_fit/data_res_after_cut.json'),
        weights = join(out, 'Data/{sample_type}/1_weighted/{decay}/data_fit/weights.root'),
    log:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/data_fit/data_res.log'),
    resources:
        max_retries=0,
        mem_mb = 20_000, 
        MaxRunHours = 6,
    threads:
        8,
    run:
        tree = find_tree_name(wildcards.decay)

        out_path = os.path.dirname(os.path.dirname(output.data_res))

        #Use ceph-kernel if available to increase file reading performance
        if kernel_available():
            raw = path_to_kernel(input.data_raw)
        else:
            raw = input.data_raw



        cmd = [
            'python {input.script}',
            '--data_files ', ' '.join(raw),
            '--range {lowerMass} {upperMass}', 
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--treename', tree,
            '--output', out_path,
            '--decayType {wildcards.decay}',
            '--sim_fit {input.mc_res}',
            '--cut notSamePV_noOSP',
            '--BDT {input.BDT}',
            '--num_threads {threads}',
            '&> {log}'
        ]
        shell(' '.join(cmd))

def get_memory_usage(ID):
    ID_numb = int(ID.split('_')[-2])
    if ID_numb > 800:
        return 90_000  # IDs over 800 happen to be larger files in this case. No need to increase memory usage for smaller files
    else:
        return 30_000  # For smaller IDs, use less memory

rule add_weights:
    input:
        script = join(repo, 'scripts/add_weights.py'),
        # selected = join(out, 'Data/{sample_type}/3_selected/{decay}/{cut_name}/{features}/{id}.data24.root'),
        # selected = join(out, 'Data/{sample_type}/3_selected/{decay}/{cut_name}/{features}/{id}.data24.root'),
        data_raw = lambda wildcards: get_raw_paths(wildcards.decay, f'{wildcards.id}.data24', 'Data'),

        # mc_res = join(out, 'Data/{sample_type}/1_weighted/{decay}/{cut_name}/{features}/mc_fit/mc_res.json'),
        # model = join(out,  'Data/{sample_type}/1_weighted/{decay}/{cut_name}/{features}/data_fit/data_res.json'),
        weights = join(out, 'Data/{sample_type}/1_weighted/{decay}/data_fit/weights.root'),

    output:
        # join(out, 'Data/{sample_type}/1_weighted/{decay}/plots/validate_sweights_{id}.png'),
        weighted = join(out, 'Data/{sample_type}/1_weighted/{decay}/weighted_files/{id}.data24.root'),
    log:
        join(out, 'Data/{sample_type}/1_weighted/{decay}/weighted_files/{id}.data24.log'),
    resources:
        max_retries=0,
        mem_mb = lambda wildcards: get_memory_usage(wildcards.id), 
        MaxRunHours = 2, 
    run:
        out_path = os.path.dirname(output.weighted)
        tree = find_tree_name(wildcards.decay)

        cmd = [
            'python {input.script}',
            # '--data_file {input.selected}',
            '--data_file {input.data_raw}',
            # '--data_fit_model {input.model}',
            # '--treename "DecayTree;1"',
            '--treename', tree,
            # '--sim_fit_model {input.mc_res}' ,
            '--out_path', out_path,
            '--decayType {wildcards.decay}',
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--range {lowerMass} {upperMass}',
            '--weight_file {input.weights}',
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
    if 'architecture' in data[tagger].keys():
        arch = data[tagger]['architecture']
        dm = float(data[tagger]['min_delta'])
        config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'
        return {'config':config, 'seed':seed, 'lr':lr, 'bs':bs, 'arch':arch, 'dm':dm, 'tagger':tagger, 'cut':cut, 'link':link}
    else:
        nl = int(data[tagger]['numlayers'])
        nn = int(data[tagger]['numneurons'])
        config = f'lr{lr}_bs{bs}_nL{nl}_nN{nn}'
        return {'config':config, 'seed':seed, 'lr':lr, 'bs':bs, 'numlayers':nl, 'numneurons':nn, 'tagger':tagger, 'cut':cut, 'link':link}

def get_model_path(wildcards):
    data_type = wildcards.data_type_or_adapted
    sample_type = wildcards.sample_type
    tagger = wildcards.tagger
    decay = extract_decay(tagger)
    cut_name = wildcards.cut_name
    features = wildcards.features
    best = extract_best(tagger=tagger, cut=cut_name,data_type=data_type)


    model = join(out, f'{data_type}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{best["seed"]}/{best["config"]}')

    if data_type == 'Data':
        model = join(model, 'pdf_ratio')
    
    return join(model, 'training')


rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        split = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/test/{id}.root'),

        model= lambda wildcards: join(get_model_path(wildcards), 'model.pth'),
        transformer=lambda wildcards: join(get_model_path(wildcards),'powerTransformer.pkl'), 
        scaler=lambda wildcards: join(get_model_path(wildcards),'st_scaler.pkl'), 

        config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type_or_adapted).get("config")}.yaml'),
    output:
        root = join(out, '{data_type}/{sample_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{id}.root'),
    log:
        join(out, '{data_type}/{sample_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{id}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type)
        
        domain = '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else ''

        cmd = [
            'python', input.script,
            '--selected {input.split}', 
            '--taggedData {output.root}',  
            '--model {input.model}',
            '--scaler {input.scaler}',
            '--transformer {input.transformer}',
            '--config {input.config}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--data_type {wildcards.data_type}', 
            f'--seed {config.get("seed")}',
            '--repo', repo,
            domain,
            '--cut {wildcards.cut_name}',
            '&>{log}'
        ]
        shell(' '.join(cmd))


def get_tagged_paths(wildcards):
    taggers = wildcards.combinationName.split('_')
    data_type = wildcards.data_type
    sample_type = wildcards.sample_type
    decay = wildcards.decay
    cut_name = wildcards.cut_name
    features = wildcards.features
    data_type_or_adapted = wildcards.data_type_or_adapted

    pre_path = join(out, f'{data_type}/{sample_type}/6_tagged/{decay}')

    all_paths = []
    for tagger in taggers:
        pre_path_tagger = join(pre_path, f'{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/')
        if data_type == 'Data':
            ids = data_ids
        else:
            ids = [basename(f)[:-5] for f in ntuples_train_split_withUT_mc[decay][tagger]]

        all_paths.extend(join(pre_path_tagger, f'{id}.root') for id in ids)

    return all_paths

rule combine_tagger: 
    input:
        script = join(repo, 'scripts/combineTagger.py'),

        tagged = get_tagged_paths,
    output:
        pdf=join(out, '{data_type}/savedModels/{sample_type}/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_Calibration.pdf'),
    log:    join(out, '{data_type}/savedModels/{sample_type}/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_log.log')
    resources:
        max_retries=0,

        mem_mb = 40_000, 
        MaxRunHours = 4,
    run:
        tagged_prePath = join(input.tagged[0].split('6_tagged')[0], '6_tagged/')
        out_path = os.path.dirname(output.pdf)
        out_path = join(out, '{wildcards.data_type}/savedModels/{wildcards.sample_type}')

        cmd = [
            'python', input.script,
            '--tagger OSKaon OSMuon OSElectron',
            f'--tagged_prePath {tagged_prePath}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            f'--outputPath {out_path}',
            '--features {wildcards.features}',
            '--cut {wildcards.cut_name}',
            '--trained_on {wildcards.data_type_or_adapted}',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule combine_MC_Data: #Combines data and MC for domain adaptation
    input:
        script = join(repo, 'scripts/combine_MC_Data.py'),
        # data = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f in ntuples_tagged_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        data = lambda wildcards: [join(out, f'Data/{wildcards.sample_type}/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.data_portion}/{id}.root') for id in data_ids],
        MC = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}').replace('train', f'{wildcards.data_portion}')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
    output:
        root = [join(out, 'domain_adapted/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}', f'samples_{i}.root') for i in range(combined_df_n_splits)],
    log:
        join(out, 'domain_adapted/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}/log.log'),
    wildcard_constraints:
        data_portion = 'train|validation',
    resources:
        max_retries=0,
        mem_mb = 25_000, 
        MaxRunHours = 2, # short queue
    run:
        out_path = os.path.dirname(output.root[0])

        #Use ceph-kernel if available to increase file reading performance
        if kernel_available():
            data = path_to_kernel(input.data)
            MC   = path_to_kernel(input.MC)
        else:
            data = input.data
            MC   = input.MC

        cmd = [
            'python {input.script}',
            '--data_files', ' '.join(data),
            '--mc_files', ' '.join(MC),
            '--target_path', out_path,
            '--splits ', str(combined_df_n_splits),
            # '--decayType {wildcards.decay}',
            # '--tagger {wildcards.tagger}',
            # '--cut_name {wildcards.cut_name}',
            # '--features {wildcards.features}',
            '&> {log}'
        ]
        shell(' '.join(cmd))





# rule split_sample:
#     input:
#         script = join(repo, 'scripts/split_train_val_test.py'),
#         to_split = lambda wildcards:  join(out, f'{wildcards.data_type}/{wildcards.sample_type}/3_selected/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.id}.root'),

#         hyper_int = join(repo, 'configs/hyperpar_intervals.yaml'), # For the train-val proportions
#     output:
#         train      = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/train/{id}.root'),
#         validation = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/validation/{id}.root'),
#         test       = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/test/{id}.root'),
#     log:
#         join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/log/.{id}.log'),
#     resources:
#         max_retries=0,
#         mem_mb = 15_000,
#         MaxRunHours = 3,
#     run:
#         out_path = os.path.dirname(os.path.dirname(output.train))

#         cmd = [
#             'python', input.script,
#             '--weighted {input.to_split}',
#             '--target_path', out_path,
#             '--config {input.hyper_int}',
#             '--decayType {wildcards.decay}',
#             '--treename "DecayTree;1"',
#             '--tagger {wildcards.tagger}',
#             '--data_type {wildcards.data_type}',
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))


def get_chunk(middle_path, filename):
    #If Batched mode is on, training job trains all configuration in a hyperparameter chunk 
    #(currently all batch sizes, layers, etc. except learning rate and weight type)
    
    if batched:
        return [join(out, middle_path+ 'lr{learning_rate}_bs{batch_size}_' + f+ filename) for f in hyper_par_chunk]
    else:
        return join(out, middle_path+ '{config}' + filename)

def get_log(data_type):
    weight = "{weight_type}/" if data_type != 'MC' else ""
    logs = {}
    # logs['training_logs'] = list(get_chunk(data_type + '/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', f'/{weight}training/training_log.log'))
    logs = get_chunk(data_type + '/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', f'/{weight}training/training_log.log')
    if batched:
        logs = list(logs)
        if data_type == 'MC':
            weight_name = 'MC'
        else:
            weight_name = '{weight_type}'
        # logs['chunk_log'] =[join(out, data_type + '/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/.' + weight + '_logs/lr{learning_rate}_training_chunk.log')]
        logs = logs + [join(out, data_type + '/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/chunk_logs/' + weight_name + '_lr{learning_rate}_bs{batch_size}_training_chunk.log')]
    return logs

rule train_tagger_MC: #TODO Remove alle the "if batched" stuff. not used anymore
    #If the batched flag from the config file is set to True, this rule trains a chunk of hyperparameters, if False it trains only one hyperparameter configuration 
    input:
        script = join(repo, 'scripts/batch_train_tagger.py') if batched else join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            # if not f.endswith('4_1.mc.root')
        ],
        val = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'validation')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
            # if not f.endswith('4_1.mc.root')
        ],
        
        config = [join(repo, 'configs/lr{learning_rate}_bs{batch_size}_' + f'{remaining_conf}.yaml') for remaining_conf in hyper_par_chunk] if batched
                 else join(repo, 'configs/{config}.yaml'),
    output:
        # ROC=         get_chunk('MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/ROC_TRAIN_VAL.pdf'),
        model=       get_chunk('MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/model.pth'),
        scaler=      get_chunk('MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/st_scaler.pkl'),
        transformer= get_chunk('MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/powerTransformer.pkl'),
    params:
        pre_path = join(out, 'MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}')
    log:
        get_log('MC'),
        # get_chunk('MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/training_log.log'),
        # chunk_log = join(out, 'MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/training_chunk.log')
        #      if batched else None,
    resources:
        max_retries=0,
        mem_mb = 30_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
    threads:
        len(hyper_par_chunk)+1 if batched else 4,
    run:
        if kernel_available():
            train_scratch = path_to_kernel(input.train)
            val_scratch = path_to_kernel(input.val)
        else:
            train_scratch = input.train
            val_scratch = input.val

        # train_scratch = copy_to_scratch(input.train)
        # val_scratch = copy_to_scratch(input.val)

        if not batched:
            outpath = os.path.dirname(output.model)

        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train_scratch),
            '--validation_data', ' '.join(val_scratch),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--repo', repo,
            '--data_type MC',
            # '--balance_dataset',
            #'--clean',
        ]

        if batched:
            logs = list(log)
            training_logs = logs[:-1]
            chunk_log = logs[len(logs)-1]

            conditional_cmd =[
                # '--training_logs', ' '.join(log.training_logs),
                '--training_logs', ' '.join(training_logs),
                '--pre_path', params.pre_path,
                '--configs', ' '.join(input.config),
                # '&>',  log.chunk_log,
                '&>',  chunk_log,
            ]
        else:
            conditional_cmd = [
                '--target_path', outpath,
                '--config {input.config}',
                '--num_threads {threads}',
                '&> {log}',
            ]
        cmd = cmd + conditional_cmd
        shell(' '.join(cmd))

rule train_tagger_data:
    input:
        script = join(repo, 'scripts/batch_train_tagger.py') if batched else join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
        val = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'validation')
            for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],

        config = [join(repo, 'configs/lr{learning_rate}_bs{batch_size}_' + f'{remaining_conf}.yaml') for remaining_conf in hyper_par_chunk] if batched
                 else join(repo, 'configs/{config}.yaml'),
    output:
        # ROC=         get_chunk('Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/{weight_type}/training/ROC_TRAIN_VAL.pdf'),
        model=       get_chunk('Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/{weight_type}/training/model.pth'),
        scaler=      get_chunk('Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/{weight_type}/training/st_scaler.pkl'),
        transformer= get_chunk('Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/{weight_type}/training/powerTransformer.pkl'),
    params:
        pre_path = join(out, 'Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}')
    log:
        get_log('Data'),
        # get_chunk('Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/', '/training/training_log.log'),
        # chunk_log = join(out, 'Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/training_chunk.log')
        #     if batched else None,
    resources:
        max_retries=0,
        mem_mb = 60_000 if batched else 40_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        #OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 8, # long queue
    threads:
        len(hyper_par_chunk)+1     if batched else 8,
    run:
        train_scratch = copy_to_scratch(input.train)
        val_scratch = copy_to_scratch(input.val)

        if not batched:
            outpath = os.path.dirname(output.model)

        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train_scratch),
            '--validation_data', ' '.join(val_scratch),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--weight_type {wildcards.weight_type}',
            '--repo', repo,
            '--data_type Data',
            '--balance_dataset',
            # '--domain B_ID',
            #'--clean',
        ]

        if batched:
            logs = list(log)
            chunk_log = logs[len(logs)-1]
            training_logs = logs[:-1]

            conditional_cmd =[
                # '--training_logs', ' '.join(log.training_logs),
                '--training_logs', ' '.join(training_logs),
                '--pre_path', params.pre_path,
                '--configs', ' '.join(input.config),
                # '&>',  log.chunk_log,
                '&>',  chunk_log,
            ]

        else:
            conditional_cmd = [
                '--target_path', outpath,
                '--config {input.config}',
                '--num_threads {threads}',
                '&> {log}',
            ]

        cmd = cmd + conditional_cmd
        shell(' '.join(cmd))

rule train_tagger_domain_adapted:
    input:
        script = join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: 
            [join(out, f"domain_adapted/{wildcards.sample_type}/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/train/samples_{i}.root") 
             for i in range(combined_df_n_splits)], 
        val =   lambda wildcards: 
            [join(out, f"domain_adapted/{wildcards.sample_type}/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/validation/samples_{i}.root")
             for i in range(combined_df_n_splits)],

        config = join(repo, 'configs/{config}.yaml'),
    output:
        model=       join(out, 'domain_adapted/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
        scaler=      join(out, 'domain_adapted/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
        transformer= join(out, 'domain_adapted/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),

    log:
        join(out, 'domain_adapted/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/training_log.log'),
    resources:
        max_retries=0,
        mem_mb = 30_000, 
        MaxRunHours = 16, # long queue
        threads = 8, #
    threads:
        8,
    run:
        train_scratch = copy_to_scratch(input.train)
        val_scratch = copy_to_scratch(input.val)

        outpath = os.path.dirname(output.model)

        # shell('sleep $(($RANDOM%200))')  # Sleep for a random time to make race conditions less likely, up to 200 seconds

        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train_scratch),
            '--validation_data', ' '.join(val_scratch),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--repo', repo,
            '--data_type domain_adapted',
            '--balance_dataset',
            '--target_path', outpath,
            '--config {input.config}',
            '--num_threads {threads}',
            '&> {log}',
        ]

        shell(' '.join(cmd))


rule calibrate_on_MC:
    input:
        testing = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'test')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
            if not f.endswith('4_1.mc.root')
        ],
        model = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        logit = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/testing_log.log')
    wildcard_constraints:
        weight_or_empty = '(' + '|'.join([i + '/' for i in weights] + ['']) + ')', #For Data trained taggers needs to represent the weight, for MC it is empty
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 2,
        # request_disk = 256_000
    run:
        outpath = os.path.dirname(os.path.dirname(output.logit))
        model_path = os.path.dirname(input.model)

        if kernel_available():
            test_kernel = path_to_kernel(input.testing)
        else:
            test_kernel = input.testing


        cmd = [
            'python', input.script,
            '--testing_data', ' '.join(test_kernel),
            '--target_path', outpath,
            '--train_path', model_path,
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed {wildcards.seed}',
            '--repo', repo,
            '--data_type MC',
            '--model_path', model_path,
            '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else '',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule calibrate_on_data:
    input:
        testing = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'test')
            for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        model = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        logit = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/testing_log.log')
    wildcard_constraints:
        weight_or_empty = '(' + '|'.join([i + '/' for i in weights] + ['']) + ')', #For Data needs to represent the weight, for MC it is empty
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 2,
        # request_disk = 256_000
    run:
        outpath = os.path.dirname(os.path.dirname(output.logit))
        model_path = os.path.dirname(input.model)

        if kernel_available():
            test_kernel = path_to_kernel(input.testing)
        else:
            test_kernel = input.testing


        cmd = [
            'python', input.script,
            '--testing_data', ' '.join(test_kernel),
            '--target_path', outpath,
            '--train_path', model_path,
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed {wildcards.seed}',
            '--repo', repo,
            '--data_type Data',
            '--model_path', model_path,
            '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else '',
            '&> {log}',
        ]
        shell(' '.join(cmd))



#Everything following this are Temporary TODO remove
def get_micols_model(wildcards):
        if wildcards.tagger == 'OSKaon':
            model = '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/model.pth'
        elif wildcards.tagger == 'OSElectron':
            model = '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/model.pth'
        elif wildcards.tagger == 'OSMuon':
            model = '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/model.pth'
        else:
            raise ValueError(f"Unknown tagger: {wildcards.tagger}")
        return model
    
rule calibrate_micols_tagger_on_data:
    input:
        testing = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'test')
            for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        model = get_micols_model,
        

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        logit = join(out, '{data_type_or_adapted}/savedModels_micol/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels_micol/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels_micol/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/testing_log.log')
    wildcard_constraints:
        weight_or_empty = '(' + '|'.join([i + '/' for i in weights] + ['']) + ')', #For Data needs to represent the weight, for MC it is empty
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 2,
        # request_disk = 256_000
    run:
        outpath = os.path.dirname(os.path.dirname(output.logit))
        model_path = os.path.dirname(input.model)

        if kernel_available():
            test_kernel = path_to_kernel(input.testing)
        else:
            test_kernel = input.testing


        cmd = [
            'python', input.script,
            '--testing_data', ' '.join(test_kernel),
            '--target_path', outpath,
            '--train_path', model_path,
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed {wildcards.seed}',
            '--repo', repo,
            '--data_type Data',
            '--model_path', model_path,
            '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else '',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule add_tagDec_micols_tagger:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        split = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/test/{id}.root'),

        model= get_micols_model,
        transformer= lambda wildcards: get_micols_model(wildcards).replace('model.pth', 'powerTransformer.pkl'),
        scaler= lambda wildcards: get_micols_model(wildcards).replace('model.pth', 'st_scaler.pkl'),

        config = lambda wildcards: join(repo, f'configs/{basename(dirname(get_micols_model(wildcards)))}.yaml'),
    output:
        root = join(out, '{data_type}/{sample_type}_micols/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{id}.root'),
    log:
        join(out, '{data_type}/{sample_type}_micols/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{id}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type)
        
        domain = '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else ''

        cmd = [
            'python', input.script,
            '--selected {input.split}', 
            '--taggedData {output.root}',  
            '--model {input.model}',
            '--scaler {input.scaler}',
            '--transformer {input.transformer}',
            '--config {input.config}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--data_type {wildcards.data_type}', 
            f'--seed {config.get("seed")}',
            '--repo', repo,
            domain,
            '--cut {wildcards.cut_name}',
            '&>{log}'
        ]
        shell(' '.join(cmd))


def get_tagged_paths(wildcards):
    taggers = wildcards.combinationName.split('_')
    data_type = wildcards.data_type
    sample_type = wildcards.sample_type
    decay = wildcards.decay
    cut_name = wildcards.cut_name
    features = wildcards.features
    data_type_or_adapted = wildcards.data_type_or_adapted

    pre_path = join(out, f'{data_type}/{sample_type}_micols/6_tagged/{decay}')

    all_paths = []
    for tagger in taggers:
        pre_path_tagger = join(pre_path, f'{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/')
        if data_type == 'Data':
            ids = data_ids
        else:
            ids = [basename(f)[:-5] for f in ntuples_train_split_withUT_mc[decay][tagger]]

        all_paths.extend(join(pre_path_tagger, f'{id}.root') for id in ids)

    return all_paths

rule combine_micols_tagger: 
    input:
        script = join(repo, 'scripts/combineTagger.py'),

        tagged = get_tagged_paths,
    output:
        pdf=join(out, '{data_type}/savedModels_micol/{sample_type}/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_Calibration.pdf'),
    log:    join(out, '{data_type}/savedModels_micol/{sample_type}/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_log.log')
    resources:
        max_retries=0,

        mem_mb = 40_000, 
        MaxRunHours = 4,
    run:
        tagged_prePath = join(input.tagged[0].split('6_tagged')[0], '6_tagged/')
        # out_path = os.path.dirname(output.pdf)
        # out_path = join(out, '{wildcards.data_type}/savedModels/{wildcards.sample_type}')
        out_path = join(output.pdf.split(wildcards.sample_type)[0], wildcards.sample_type)

        cmd = [
            'python', input.script,
            '--tagger OSKaon OSMuon OSElectron',
            f'--tagged_prePath {tagged_prePath}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            f'--outputPath {out_path}',
            '--features {wildcards.features}',
            '--cut {wildcards.cut_name}',
            '--trained_on {wildcards.data_type_or_adapted}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

def get_da_models(wildcards):
    return f'/ceph/users/togasa/FlavourTagging/NTuples/domain_adapted/savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs4096_nL6_nN256_alpha{wildcards.alpha}/training/model.pth'

rule add_tagDec_da_tagger:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        split = join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/test/{id}.root'),

        model= ancient(get_da_models),
        transformer= ancient(lambda wildcards: get_da_models(wildcards).replace('model.pth', 'powerTransformer.pkl')),
        scaler= ancient(lambda wildcards: get_da_models(wildcards).replace('model.pth', 'st_scaler.pkl')),

        config = lambda wildcards: join(repo, f'configs/{basename(dirname(dirname(get_da_models(wildcards))))}.yaml'),
    output:
        root = join(out, '{data_type}/{sample_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_DA_{alpha}/{id}.root'),
    log:
        join(out, '{data_type}/{sample_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_DA_{alpha}/{id}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type)
        
        domain = '--domain_adapted'

        cmd = [
            'python', input.script,
            '--selected {input.split}', 
            '--taggedData {output.root}',  
            '--model {input.model}',
            '--scaler {input.scaler}',
            '--transformer {input.transformer}',
            '--config {input.config}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--data_type {wildcards.data_type}', 
            f'--seed {config.get("seed")}',
            '--repo', repo,
            domain,
            '--cut {wildcards.cut_name}',
            '&>{log}'
        ]
        shell(' '.join(cmd))
