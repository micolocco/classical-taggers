from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import pandas as pd
import os
import re
from copy import deepcopy
import yaml
import json
import glob

configfile: 'configs/config_taggers.yaml' #Default config file, can be overwritten when calling snakemake with --configfile <file>

try:
    data = config['DATA']
    MC = config['MC']
    out = config['OUT']
    lowerMass = config['Mass_range_lower']
    upperMass = config['Mass_range_upper']
    repo = config['REPO']

    #In how many sections combined dataframes are splint into when usind domain adaptation or combining small data files
    combined_df_n_splits = config['combined_df_n_splits'] 
except:
    raise RuntimeError("No valid snakemake config found")

def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]

taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2DsPi': ['SSKaon'],
}

all_taggers = set()
for tagger_list in taggers_conf.values():
    all_taggers.update(tagger_list)
all_taggers = list(all_taggers)

def get_raw_paths(decay, ID, data_type):
    end_path = f'withUT_MC_2024/1_raw/{decay}/{ID}.root'
    if data_type == 'MC':
        # return join(MC, f'{decay}/v1_taggers/{ID}.root')
        return join(MC, end_path)
    elif data_type == 'Data':
        return join(data, f'{ID[:8]}/{ID[9:13]}' + f'/{ID}.root')
        # return join(data, end_path)
    else:
        raise RuntimeError(f"data type is {data_type} instead of MC or Data. Somethings broken")

#Raw Data Files
raw_data = {}
for decay in taggers_conf.keys():
    base_path = join(dirname(get_raw_paths(decay, '00266999_00000001_1.data24', 'Data')) , '*.root')
    files = glob.glob(base_path)
    raw_data[decay] = files
combine_indices = {decay: list(range(len(raw_data[decay]))) for decay in taggers_conf.keys()}
np.random.seed(42)
for decay in taggers_conf.keys():
    np.random.shuffle(combine_indices[decay])
    combine_indices[decay] = np.array_split(combine_indices[decay], combined_df_n_splits)

#Raw MC files
raw_mc = {}
for decay in taggers_conf.keys():
    base_path = join(dirname(get_raw_paths(decay, 'ID', 'MC')) , '*.root')
    files = glob.glob(base_path)
    raw_mc[decay] = files


#Create lists of all file names after each step
#Data
feat_added_data = {key: [join(out, 'Data/NTuples/1_added_features', key, os.path.basename(f)) for f in raw_data[key]] for key in taggers_conf.keys()}

combined_data = {}
for decay, path_list in feat_added_data.items():
    path = os.path.dirname(path_list[0])
    combined_data.update({decay: [join(path, f'combined/samples_{i}.root') for i in range(combined_df_n_splits)]})


train_split_data = {}
for decay, path_list in combined_data.items():
    train_split_data.update({decay: [f.replace(f'1_added_features/{decay}/combined', f'2_split/{decay}').replace(decay, f'{decay}/train') for f in path_list]})

event_selected_data = {}
for decay, path_list in train_split_data.items():
    event_selected_data.update({decay: [f.replace(f'2_split', f'3_event_selected') for f in path_list]})

selected_data = {}
for decay, path_list in event_selected_data.items():
    selected_data.update({decay: {}})
    for tagger in taggers_conf[decay] : 
        selected_data[decay].update({tagger: [f.replace('3_event_selected', f'4_track_selected').replace('train', f'{tagger}/cut_name/features/train') for f in path_list]})

weighted_data = {}
for decay, path_list in selected_data.items():
    weighted_data.update({decay: {}})
    for tagger, path_list in path_list.items():
        weighted_data[decay].update({tagger: [f.replace('4_track_selected', f'5_weighted') for f in path_list]})

tagged_data = {}
for decay, path_list in selected_data.items():
    tagged_data.update({decay: {}})
    for tagger, path_list in path_list.items():
        tagged_data[decay].update({tagger: [f.replace('4_track_selected', f'6_tagged').replace('train', 'trained_on') for f in path_list]})


#MC
feat_added_mc = deepcopy(raw_mc)
for decay, path_list in feat_added_mc.items():
    feat_added_mc[decay] = [f.replace(os.path.dirname(f), f'{out}MC/NTuples/1_added_features/{decay}') for f in path_list]

train_split_mc = {}
for decay, path_list in feat_added_mc.items():
    train_split_mc.update({decay: [f.replace('1_added_features', f'2_split').replace(decay, f'{decay}/train') for f in path_list]})

event_selected_mc = {}
for decay, path_list in train_split_mc.items():
    event_selected_mc.update({decay: [f.replace('2_split', '3_event_selected') for f in path_list]})

selected_mc = {}
for decay, path_list in event_selected_mc.items():
    selected_mc.update({decay: {}})
    for tagger in taggers_conf[decay] :
        selected_mc[decay].update({tagger: [f.replace('3_event_selected', f'4_track_selected').replace('train', f'{tagger}/cut_name/features/train') for f in path_list]})

tagged_mc = {}
for decay, path_list in selected_mc.items():
    tagged_mc.update({decay: {}})
    for tagger, path_list in path_list.items():
        tagged_mc[decay].update({tagger: [f.replace('4_track_selected', f'6_tagged').replace('train', 'trained_on') for f in path_list]})


# Function to read paths from the generated file
def read_generated_paths(data, file):
    with open(file, 'r') as f:
        paths = [join(data,line.strip()) for line in f]
    return paths

def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    if decay == 'Bs2DsPi':
        return 'BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    if decay == 'Bu2JpsiK':
        return 'BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree'
    if decay == 'Bd2JpsiKst':
        return 'BdToJpsiKstar_JpsiToMuMu_Detached/DecayTree'
    else:
        return 'Tuple/DecayTree'


#Define a hyperparameter chunk, used for parallelization of the training during hyperparameter optimization

with open(join(repo,'configs/hyperpar_intervals.yaml'), 'r') as file:
    intervals = yaml.safe_load(file)


intervals = {key[1:]: value for key, value in intervals.items()}




wildcard_constraints:
    data_type    = '(MC|Data)',
    decay        = '(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)',
    binning      = '(|Tau1of4|Tau2of4|Tau3of4|Tau4of4)', #Empty string for no binning, Tau1of4 for first tau bin, etc.
    tagger       = '(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)',
    data_type_or_adapted = '(Data|MC|domain_adapted)',
    partition = '(train|validation|test)',
    is_selected = '(selected|non_selected)',
    cut_name = "[^/]+", #don't allow slashes in wildcards to avoid problems with paths
    features = "[^/]+",
    seed = '[^/]+',
    config = '[^/]+',
    ID = '[^/]+',


rule all:
    input:
        #No BN Data
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 

        #BN Data
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        #BN MC
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 

        #No BN MC
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
               tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 


        #OS bins Bu2JpsiK
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau1of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau2of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau3of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau4of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau1of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau2of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau3of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau4of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau1of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau2of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau3of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiKTau4of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',

        #OS bins Bd2JpsiKst
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        
        #SS bins Bd2JpsiKst
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',


        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',





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


rule add_features:
    input:
        script = join(repo, 'scripts/adding_features.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
        raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.ID, wildcards.data_type),
    output: 
        root =join(out, '{data_type}/NTuples/1_added_features/{decay}/{ID}.root'), 
    log:                            
        join(out, '{data_type}/NTuples/1_added_features/{decay}/.{ID}.log')
    resources:
        max_retries=0,
        mem_mb = 10_000,
        MaxRunHours = 1, # short queue
    run:
        tree = find_tree_name(wildcards.decay)
       
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
            '--loading_features {input.loading_vars}',
            '--signal_class_features {input.signal_class_features}',
            '--data_type {wildcards.data_type}',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule combine_small_files:
    input:
        script = join(repo, 'scripts/combine_dataframes.py'),
        data = lambda wildcards: np.array(feat_added_data[wildcards.decay])[combine_indices[wildcards.decay][int(wildcards.ID)]],
    output:
        join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.root'),
    log: 
        join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 90_000,
        MaxRunHours = 1, # short queue
    run:
        path = os.path.dirname(output[0])

        if kernel_available():
            data = path_to_kernel(input.data)
        else:
            data = input.data

        cmd = [
            f'python {input.script} ',
            f'--data_files ', ' '.join(data),  
            f'--target_path {path} ',  
            f'--treename "DecayTree;1" ', 
            f'--splits 1',  
            f'--evtType {wildcards.decay} ', 
            f'--index {wildcards.ID} ',
            f'&> {log}', 
        ]

        shell(' '.join(cmd))

rule split_sample:
    input:
        script = join(repo, 'scripts/split_train_val_test.py'),
        to_split = lambda wildcards: join(out, f'MC/NTuples/1_added_features/{wildcards.decay}/{wildcards.ID}.root') if wildcards.data_type == 'MC' 
                                else join(out, f'Data/NTuples/1_added_features/{wildcards.decay}/combined/{wildcards.ID}.root'),

        hyper_int = join(repo, 'configs/hyperpar_intervals.yaml'), # For the train-val proportions
    output:
        train      = join(out, '{data_type}/NTuples/2_split/{decay}/train/{ID}.root'),
        validation = join(out, '{data_type}/NTuples/2_split/{decay}/validation/{ID}.root'),
        test       = join(out, '{data_type}/NTuples/2_split/{decay}/test/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/2_split/{decay}/log/.{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 65_000,
        MaxRunHours = 1,
    run:
        out_path = os.path.dirname(os.path.dirname(output.train))
        treename = '"DecayTree;1"' 
        
        cmd = [
            'python', input.script,
            '--to_split {input.to_split}',
            '--target_path', out_path,
            '--config {input.hyper_int}',
            '--treename', treename,
            '--data_type {wildcards.data_type}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule train_signal_classifier:
    input:
        script = join(repo, 'scripts/train_BDT.py'),
        data = lambda wildcards: combined_data[wildcards.decay],
        mc = lambda wildcards: feat_added_mc[wildcards.decay], 
        signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
    output:
        BDT = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl')
    log:
        join(out, 'Data/signal_classifier/{decay}/BDT_train.log')
    resources:
        max_retries=0,
        mem_mb = 20_000,
        MaxRunHours = 4,
    threads:
        8,
    run:
        out_path = os.path.dirname(output.BDT)

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
            '--treename "DecayTree;1"',
            '--decay_type {wildcards.decay}',
            '--massname B_DTF_PV_Jpsi_MASS',
            '--num_threads {threads}',
            '--signal_class_features', input.signal_class_features,
            '&> {log}',
        ]


        shell(' '.join(cmd))

rule event_selection: #Applies BDT signal selection and in case a bin is supplied in addition to the decay, also cuts away events outside the bin
    input:
        script = join(repo, 'scripts/apply_event_selection.py'),
        classifier = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl'),
        data = join(out, '{data_type}/NTuples/2_split/{decay}/{partition}/{ID}.root'),
        signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
        bin_file = lambda wildcards: join(repo, 'configs/binnings.yaml') if wildcards.binning else [],
    output:
        join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/.{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 30_000,
        MaxRunHours = 1,
    run:
        if kernel_available():
            data = path_to_kernel(input.data)
        else:
            data = input.data
        binning = '--binning {wildcards.binning} --bin_file {input.bin_file}' if wildcards.binning else ''

        cmd = [
            'python', input.script,
            '--data {data}',
            '--target {output}',
            '--treename "DecayTree;1"',
            '--decay_type {wildcards.decay}',
            binning,
            '--BDT {input.classifier}',
            '--signal_class_features {input.signal_class_features}',
            '&> {log}',
        ]

        shell(' '.join(cmd))

def get_DT_input_paths(wildcards):
    all_mc_files = []
    for decay in feat_added_mc.keys():
        all_mc_files.extend(feat_added_mc[decay])

    return [file for file in all_mc_files if file.endswith('01_1.mc.root')]

rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        data = get_DT_input_paths
    output:
        cuts = expand(join(out, "MC/DT_outputs/{{cut_name}}/{{balanced}}/cuts/{tagger}_preselections.txt"), tagger=all_taggers)
    log:              join(out, "MC/DT_outputs/{cut_name}/{balanced}/tree_schema.log")
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 4,
    run:
        target_path = os.path.dirname(os.path.dirname(output.cuts[0]))

        if kernel_available():
            data = path_to_kernel(input.data)
        else:
            data = input.data

        cmd = [
            f'python {input.script}',
            f'--input_files {data}',
            f'--target_path {target_path}',
            f'--balanced {wildcards.balanced}',
           # f'--unify_SS',
            f'--BKG0',
            f'&> {log}',
        ]
        shell(' '.join(cmd))



rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        to_select = lambda wildcards : join(out, '{data_type}/NTuples/2_split/{decay}{binning}/{partition}/{ID}.root') if wildcards.data_type == 'MC' and wildcards.binning == ''
                                  else join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/{ID}.root'),
    output: join(out, '{data_type}/NTuples/4_track_selected/{decay}{binning}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:    join(out, '{data_type}/NTuples/4_track_selected/{decay}{binning}/{tagger}/{cut_name}/{features}/{partition}/.{ID}.log')
    resources:
        max_retries=0,
        mem_mb = 20_000, # Specify memory requirement in megabytes
        MaxRunHours = 1, # short queue
    run:
        BKG0 = '--BKG0' if wildcards.data_type == 'MC' else ''
        cut_file = join(repo, 'cuts/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt')

        if kernel_available():
            to_select = path_to_kernel(input.to_select)
        else:
            to_select = input.to_select
       
        cmd = [
            'python', input.script,
            '--to_select', to_select,
            '--output {output}',
            # '--treename', tree,
            '--cut_file', cut_file,
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--evtType {wildcards.decay}',
            BKG0,
            '--data_type {wildcards.data_type}',
            '--repo', repo,
            '&> {log}',
        ]
        shell(' '.join(cmd))


def get_fit_input_paths(wildcards):
    if wildcards.is_selected == 'selected':
        if wildcards.data_type == 'MC':
            files = event_selected_mc  [wildcards.decay]
        else:
            files = event_selected_data[wildcards.decay]
    else:
        if wildcards.data_type == 'MC':
            files = train_split_mc     [wildcards.decay]
        else:
            files = train_split_data   [wildcards.decay]


    return [file.replace('train', wildcards.partition).replace(wildcards.decay, f'{wildcards.decay}{wildcards.binning}') for file in files]

rule mass_fit:
    input:
        script = join(repo, 'scripts/mass_fits.py'),
        data = get_fit_input_paths,

        mc_res = lambda wildcards: join(out, 'MC/mass_fit/{decay}{binning}/{partition}/event_{is_selected}_fit.json') if wildcards.data_type == 'Data' else [],
    output: 
        join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/event_{is_selected}_fit.json'),
        join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/weights_{is_selected}.root'), #In case of MC weights is a dummy file containing just the information used in the fitting without weights
    log:    join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/event_{is_selected}_fit.log'),
    resources:
        max_retries=0,
        mem_mb = 32_000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue
    threads:
        4,
    run:
        out_path = os.path.dirname(output[0])

        if kernel_available():
            data = path_to_kernel(input.data)
        else:
            data = input.data

        cmd = [
            'python {input.script}',
            '--input_files', ' '.join(data),
            '--range {lowerMass} {upperMass}', 
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--treename "DecayTree;1"',
            '--simulation'             if wildcards.data_type   == 'MC'       else '',
            '--sim_fit {input.mc_res}' if wildcards.data_type   == 'Data'     else '',
            '--selected'               if wildcards.is_selected == 'selected' else '',
            '--output', out_path,
            '--decay_type {wildcards.decay}',
            '--num_threads {threads}',
            '&> {log}'
        ]
        print(' '.join(cmd))
        shell(' '.join(cmd))

rule add_weights:
    input:
        script = join(repo, 'scripts/add_weights.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        selected = join(out, 'Data/NTuples/4_track_selected/{decay}{binning}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
        weights = join(out, 'Data/mass_fit/{decay}{binning}/{partition}/weights_selected.root'),
    output:
        weighted = join(out, 'Data/NTuples/5_weighted/{decay}{binning}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:
        join(out, 'Data/NTuples/5_weighted/{decay}{binning}/{tagger}/{cut_name}/{features}/{partition}/{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        out_path = os.path.dirname(output.weighted)

        if kernel_available():
            selected = path_to_kernel(input.selected)
            weights = path_to_kernel(input.weights)
        else:
            selected = input.selected
            weights = input.weights

        cmd = [
            'python {input.script}',
            '--data_file', selected,
            '--out_path', out_path,
            '--decayType {wildcards.decay}',
            '--range {lowerMass} {upperMass}',
            '--weight_file', weights,
            '--loading_features {input.loading_vars}',
            '&> {log}'
        ]

        shell(' '.join(cmd))

rule train_tagger_MC: 
    input:
        script =join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}')
            for f in selected_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
        val = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'validation')
            for f in selected_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        model=       join(out, 'MC/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
        scaler=      join(out, 'MC/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
        transformer= join(out, 'MC/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),
    log:
        join(out, 'MC/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/training_log.log'),
    priority: -1, # Lower priority for tagger training so all prior steps are executed first
    resources:
        max_retries=0,
        mem_mb = 30_000, # Specify memory requirement in megabytes 
        MaxRunHours = 15, 
    threads:
        4,
    run:
        outpath = os.path.dirname(output.model)
        if kernel_available():
            outpath = path_to_kernel(outpath)
            train = path_to_kernel(input.train)
            val = path_to_kernel(input.val)
        else:
            train = input.train
            val = input.val



        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train),
            '--validation_data', ' '.join(val),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--repo', repo,
            '--data_type MC',
            '--target_path', outpath,
            '--config {input.config}',
            '--num_threads {threads}',
            '&> {log}',
        ]
        print(' '.join(cmd))
        shell(' '.join(cmd))


rule train_tagger_data:
    input:
        script = join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}')
            for f in weighted_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
        val = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'validation')
            for f in weighted_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],

        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        model=       join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
        scaler=      join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
        transformer= join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),
    log:
        join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/training_log.log'),
    priority: -1, # Lower priority for tagger training so all prior steps are executed first
    resources:
        max_retries=0,
        mem_mb = 40_000, # Specify memory requirement in megabytes 
        MaxRunHours = 12, # long queue
    threads:
        8,
    run:
        outpath = os.path.dirname(output.model)
        if kernel_available():
            outpath = path_to_kernel(outpath)
            train = path_to_kernel(input.train)
            val = path_to_kernel(input.val)
        else:
            train = input.train
            val = input.val



        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train),
            '--validation_data', ' '.join(val),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--repo', repo,
            '--data_type Data',
            '--target_path', outpath,
            '--config {input.config}',
            '--num_threads {threads}',
            '&> {log}',
        ]
        print(' '.join(cmd))
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

def get_testing_inputs(wildcards):
    if wildcards.data_type == 'Data':
        files_dict = weighted_data
    else:
        files_dict = selected_mc

    return [f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'test').replace(wildcards.decay, f'{wildcards.decay}{wildcards.binning}') 
            for f in files_dict[f'{wildcards.decay}'][f'{wildcards.tagger}']]

rule test_and_calibrate:
    input:
        testing = get_testing_inputs,
        model = lambda wildcards: 
                join(out, f'{wildcards.data_type_or_adapted}/savedModels/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/training/model.pth'),


        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        logit =  join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/mistag/taggingInfo_mistag.json'),
        calibration_logit =  join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/logit/calibration.json'),
        calibration_mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/mistag/calibration.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/testing_log.log')
    priority: -2, # Lower priority for efficient use of requested cores
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 4
    # threads:
    #     4,
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
            '--data_type {wildcards.data_type}',
            '--model_path', model_path,
            '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else '',
            '--num_threads {threads}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

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

def get_model_path(wildcards, all_taggers=False):
    data_type = wildcards.data_type_or_adapted
    if not all_taggers:
        tagger = [wildcards.tagger]
    else:
        tagger = get_taggers_from_combination(wildcards.combinationName)
    
    decay = [extract_decay(tag) for tag in tagger]
    cut_name = wildcards.cut_name
    features = wildcards.features

    best = [extract_best(tagger=tag, cut=cut_name,data_type=data_type) for tag in tagger]

    
    model = [join(out, f'{data_type}/savedModels/{dec}/{tag}/{cut_name}/{features}/{bes["seed"]}/{bes["config"]}') for tag, dec, bes in zip(tagger, decay, best)]

    if not all_taggers:
        model = model[0]
    return model


rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        to_tag = lambda wildcards: join(out, '{data_type}/NTuples/5_weighted/{decay}/{tagger}/{cut_name}/{features}/test/{ID}.root') if wildcards.data_type == 'Data' 
                             else join(out, '{data_type}/NTuples/4_track_selected/{decay}/{tagger}/{cut_name}/{features}/test/{ID}.root'),
        

        model=       lambda wildcards: join(get_model_path(wildcards), 'training/model.pth'),
        transformer= lambda wildcards: join(get_model_path(wildcards), 'training/powerTransformer.pkl'), 
        scaler=      lambda wildcards: join(get_model_path(wildcards), 'training/st_scaler.pkl'), 
        calibration =lambda wildcards: join(get_model_path(wildcards), f'testing/{wildcards.data_type}/logit/calibration.json'),

        config = lambda wildcards: join(repo, f'model_configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type_or_adapted).get("config")}.yaml'),
    output:
        root = join(out, '{data_type}/NTuples/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type)
        
        domain = '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else ''

        cmd = [
            'python', input.script,
            '--to_tag {input.to_tag}', 
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
            '--calibration {input.calibration}',
            '--repo', repo,
            domain,
            '&>{log}'
        ]
        shell(' '.join(cmd))


def get_tagged_paths(wildcards):
    taggers = wildcards.combinationName.split('_')
    data_type = wildcards.data_type
    decay = wildcards.decay
    cut_name = wildcards.cut_name
    features = wildcards.features
    data_type_or_adapted = wildcards.data_type_or_adapted

    pre_path = join(out, f'{data_type}/Ntuples/6_tagged/{decay}')

    all_paths = []
    for tagger in taggers:
        paths = tagged_mc[decay][tagger] if data_type == 'MC' else tagged_data[decay][tagger]
        paths = np.array(paths)
        paths = np.char.replace(paths, 'cut_name/features', f'{cut_name}/{features}')
        paths = np.char.replace(paths, 'trained_on', f'trained_{data_type_or_adapted}')
        all_paths = np.concatenate((all_paths, paths))

    return all_paths

def get_taggers_from_combination(combinationName):
    return combinationName.split('_')

rule combine_tagger: 
    input:
        script = join(repo, 'scripts/combineTagger.py'),

        tagged = get_tagged_paths,

        
        calibration = lambda wildcards: [join(path.replace('Bu2JpsiK', wildcards.decay).replace('Bd2JpsiKst', wildcards.decay), f'testing/{wildcards.data_type}/logit/calibration.json') 
                                         for path in get_model_path(wildcards, all_taggers=True)],
    output:
        pdf=              join(out, '{data_type}/savedModels/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}/{combinationName}_Run3_Calibration.pdf'),
        all_tagged_data = join(out, '{data_type}/savedModels/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}/combined_tagged.root'),
    log:    
        join(out, '{data_type}/savedModels/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}/{combinationName}_Run3_log.log')
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 4,
    run:
        tagged_prePath = join(input.tagged[0].split('6_tagged')[0], '6_tagged/')
        out_path = os.path.dirname(output.pdf)
        out_path = join(out, '{wildcards.data_type}/savedModels/')



        cmd = [
            'python', input.script,
            '--tagger', ' '.join(get_taggers_from_combination(wildcards.combinationName)),
            f'--tagged_prePath {tagged_prePath}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            f'--outputPath {out_path}',
            '--features {wildcards.features}',
            '--cut {wildcards.cut_name}',
            '--data_type {wildcards.data_type}',
            '--trained_on {wildcards.data_type_or_adapted}',
            '--calibrations', ' '.join(input.calibration),
            '&> {log}',
        ]
        print(' '.join(cmd))
        shell(' '.join(cmd))








###TILL HERE THE PIPELINE IS REWORKED AND FUNCTIONAL, RULES BELOW MAY NEED TO BE ADJUSTED TO NEW FOLDER STRUCTURE AND SCRIPT-CHANGES










rule combine_MC_Data: #Combines data and MC for domain adaptation
    input:
        script = join(repo, 'scripts/combine_dataframes.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        # data = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f in ntuples_tagged_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        data = lambda wildcards: [join(out, f'Data/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.data_portion}/{ID}.root') for ID in data_ids],
        MC = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}').replace('train', f'{wildcards.data_portion}')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
    output:
        root = [join(out, 'domain_adapted/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}', f'samples_{i}.root') for i in range(combined_df_n_splits)],
    log:
        join(out, 'domain_adapted/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}/log.log'),
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
            '--loading_features {input.loading_vars}',
            '&> {log}'
        ]
        shell(' '.join(cmd))


rule train_tagger_domain_adapted:
    input:
        script = join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: 
            [join(out, f"domain_adapted/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/train/samples_{i}.root") 
             for i in range(combined_df_n_splits)], 
        val =   lambda wildcards: 
            [join(out, f"domain_adapted/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/validation/samples_{i}.root")
             for i in range(combined_df_n_splits)],

        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        model=       join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
        scaler=      join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
        transformer= join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),

    log:
        join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/training_log.log'),
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
            # '--balance_dataset',
            '--target_path', outpath,
            '--config {input.config}',
            '--num_threads {threads}',
            '&> {log}',
        ]

        shell(' '.join(cmd))


