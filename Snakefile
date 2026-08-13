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
    repo = config['REPO']

    #In how many sections combined dataframes are splint into when usind domain adaptation or combining small data files
    combined_df_n_splits = config['combined_df_n_splits'] 
except:
    raise RuntimeError("No valid snakemake config found")


massranges = {'Bu2JpsiK': (5200, 5350), 
              'Bd2JpsiKst': (5200, 5400), 
              'Bs2JpsiKst': (5200, 5400), 
              'Bs2DsPi': (5200, 6000),}


def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]

taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst': ['SSPion', 'SSProton', 'SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2DsPi': ['SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2JpsiKst': ['SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
}

all_taggers = set()
for tagger_list in taggers_conf.values():
    all_taggers.update(tagger_list)
all_taggers = list(all_taggers)

def get_raw_paths(decay, ID, data_type):
    if data_type == 'MC':
        end_path = f'withUT_MC_2024/1_raw/{decay}/{ID}.root'
        return join(MC, end_path)
    elif data_type == 'Data':
        return join(data, f'{ID[:8]}/{ID[9:13]}' + f'/{ID}.root')
    else:
        raise RuntimeError(f"data type is {data_type} instead of MC or Data. Somethings broken")

#Raw Data Files
raw_data = {}
for decay in taggers_conf.keys():
    first_file = '00266999_00000001_1.data24' if decay != 'Bs2DsPi' else '00267010_00000001_1.data24'

    base_path = join(dirname(get_raw_paths(decay, first_file, 'Data')) , '*.root')
    files = glob.glob(base_path)
    raw_data[decay] = files
combine_indices = {decay: list(range(len(raw_data[decay]))) for decay in taggers_conf.keys()}
np.random.seed(42)
for decay in taggers_conf.keys():
    np.random.shuffle(combine_indices[decay])
    combine_indices[decay] = np.array_split(combine_indices[decay], combined_df_n_splits[decay])

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
    if decay == 'Bs2DsPi':
        combined_data.update({decay: path_list})
    else:
        path = os.path.dirname(path_list[0])
        combined_data.update({decay: [join(path, f'combined/samples_{i}.root') for i in range(combined_df_n_splits[decay])]})

train_split_data = {}
for decay, path_list in combined_data.items():
    replace_string = f'1_added_features/{decay}'
    if decay != 'Bs2DsPi':
        replace_string += '/combined'

    train_split_data.update({decay: [f.replace(replace_string, f'2_split/{decay}').replace(decay, f'{decay}/train') for f in path_list]})

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
    decay        = '(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi|Bs2JpsiKst)',
    binning      = '(|Tau1of4|Tau2of4|Tau3of4|Tau4of4)', #Empty string for no binning, Tau1of4 for first tau bin, etc.
    tagger       = '(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)',
    data_type_or_adapted = '(Data|MC|domain_adapted)',
    partition = '(train|validation|test)',
    is_selected = '(selected|non_selected)',
    model_types = '(trained_Data|trained_MC|trained_domain_adapted|trained_Data_BN|trained_MC_BN|trained_domain_adapted_BN|Run3v1|Run3v0)',
    benchmark_version = "[^/]+", #don't allow slashes in wildcards to avoid problems with paths
    cut_name = "[^/]+",
    features = "[^/]+",
    seed = '[^/]+',
    config = '[^/]+',
    ID = '[^/]+',
    combinationName = '[^/]+',
    BN = '(_BN|)', #Empty string for no BN, _BN for with BN
    selection = '(/non_selected|)', # Added to some rules to allow for testing on non-selected data. Is empty for selected

rule all:
    input:
        # no BN Data combinatopns
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # BN Data combinatopns
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # no BN Data combinatopns
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # BN Data combinatopns
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/combinations/Run3/trained_MC_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',






        # #No BN Data
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 

        # #BN Data
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        # #BN MC
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 

        # #No BN MC
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 


        # '/ceph/users/togasa/FlavourTagging/Data/signal_classifier/Bu2JpsiK/bdt_model.pkl',
        # '/ceph/users/togasa/FlavourTagging/Data/signal_classifier/Bd2JpsiKst/bdt_model.pkl',
        # '/ceph/users/togasa/FlavourTagging/Data/signal_classifier/Bs2DsPi/bdt_model.pkl',

        # [file.replace('train', 'test') for file in event_selected_data['Bs2DsPi']],


        # 'ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/config_test/training/model.pth',


        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_only_hadron_lda/decision_tree_model.txt', 
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_lda/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_only_tagger_lda/decision_tree_model.txt',


        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_1_2_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_1_1_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_1_2_1/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_1_1_1/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_1_1_1_2_1/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_1_1_1_1_05/decision_tree_model.txt',

        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_1_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_05_2_1_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_2_6_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_4_4_1_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_05_05_1_025_2/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_05_2_6_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_025_4_1_4_05/decision_tree_model.txt',
        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/v0_settings_confWeighted_2_05_1_025_2/decision_tree_model.txt',
        



        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bs2JpsiKst/test/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bs2JpsiKst/validation/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/MC/savedModels/Bs2DsPi/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSKaon_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json',


        # # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSKaon_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bs2JpsiKst/combinations/Run3/Run3v0/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSKaon_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/test/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/train/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/validation/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/test/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/train/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/validation/weights_selected.root',



        # '/ceph/users/togasa/FlavourTagging/Data/NTuples/4_track_selected/Bs2DsPi/non_selected/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/test/samples_0.root',

        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bs2DsPi/non_selected/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',



        # # Benchmark bins Bd2JpsiKst
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau1of4/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau2of4/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau3of4/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau4of4/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau1of4/OSMuon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau2of4/OSMuon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau3of4/OSMuon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau4of4/OSMuon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau1of4/OSElectron/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau2of4/OSElectron/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau3of4/OSElectron/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau4of4/OSElectron/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau1of4/SSPion/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau2of4/SSPion/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau3of4/SSPion/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau4of4/SSPion/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau1of4/SSProton/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau2of4/SSProton/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau3of4/SSProton/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKstTau4of4/SSProton/Run3v1/testing/Data/logit/taggingInfo_logit.json',


        # # #OS bins Bd2JpsiKst
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSKaon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSMuon/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        
        # # #SS bins Bd2JpsiKst
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSPion/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN64/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSProton/notSamePV_noOSP/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',


        # # #OS bins Bd2JpsiKst New Tree
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        
        # # #SS bins Bd2JpsiKst New Tree
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN128/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau1of4/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau2of4/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau3of4/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKstTau4of4/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/taggingInfo_logit.json',


        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.0001_bs8192_nL8_nN32/testing/Data/logit/calibration.json',

        # '/ceph/users/togasa/FlavourTagging/MC/DT_outputs/allBKGCAT_notSamePV_noOSP_SSK_balanced/decision_tree_model.pkl',



        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN') for l in selected_mc['Bs2DsPi']['SSKaon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'test') for l in selected_mc['Bs2DsPi']['SSKaon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'validation') for l in selected_mc['Bs2DsPi']['SSKaon']],
        
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN') for l in selected_mc['Bs2DsPi']['OSKaon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'test') for l in selected_mc['Bs2DsPi']['OSKaon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'validation') for l in selected_mc['Bs2DsPi']['OSKaon']],
        
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN') for l in selected_mc['Bs2DsPi']['OSMuon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'test') for l in selected_mc['Bs2DsPi']['OSMuon']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'validation') for l in selected_mc['Bs2DsPi']['OSMuon']],
        
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN') for l in selected_mc['Bs2DsPi']['OSElectron']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'test') for l in selected_mc['Bs2DsPi']['OSElectron']],
        # [l.replace('cut_name', 'allBKGCAT_notSamePV_noOSP_SSK_balanced').replace('features', 'union_PROBNN').replace('train', 'validation') for l in selected_mc['Bs2DsPi']['OSElectron']],


        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/test/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/train/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/validation/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/test/weights_non_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/train/weights_non_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bd2JpsiKst/validation/weights_non_selected.root',

        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/test/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/train/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/validation/weights_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/test/weights_non_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/train/weights_non_selected.root',
        # '/ceph/users/togasa/FlavourTagging/Data/mass_fit/Bu2JpsiK/validation/weights_non_selected.root',




        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/non_selected/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/non_selected/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/non_selected/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/Run3v1/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN_edited_for_benchmark/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        
        

        # combined_data['Bu2JpsiK'],
        # combined_data['Bd2JpsiKst'],
        # feat_added_mc['Bu2JpsiK'],
        # feat_added_mc['Bd2JpsiKst'],
        # feat_added_mc['Bs2DsPi'],



        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',





        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSMuon/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/OSElectron/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSPion/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSProton/Run3v1/testing/Data/logit/taggingInfo_logit.json',
        # '/ceph/users/togasa/FlavourTagging/MC/benchmarkModels/Bd2JpsiKst/SSKaon/Run3v1/testing/Data/logit/taggingInfo_logit.json',


        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data_BN/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/notSamePV_noOSP/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data_BN/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_Data/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',


        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/notSamePV_noOSP/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',

        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/OSKaon_OSMuon_OSElectron/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton/combined_tagged.root',
        # '/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/combinations/Run3/trained_MC/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/SSPion_SSProton_OSKaon_OSMuon_OSElectron/combined_tagged.root',


        # Old Tree stuff
        #No BN Data
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        #BN Data
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/Data/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        #BN MC
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}_BN/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        
        #No BN MC
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/Data/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]),
        # expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bd2JpsiKst/{tagger}/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs8192_nL{nl}_nN{nn}/testing/MC/logit/taggingInfo_logit.json', 
        #        tagger=['SSPion', 'SSProton', 'OSKaon', 'OSMuon', 'OSElectron'], lr=[0.0001, 0.001], nl=[6, 8], nn=[32, 64, 128]), 
        




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

def add_features_hours(wildcards):
    if wildcards.decay == 'Bs2DsPi':
        return 2
    else:
        return 1

rule add_features:
    input:
        script = join(repo, 'scripts/adding_features.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
        classical_selection_features = join(repo, 'configs/classic_selection_features.yaml'),
        raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.ID, wildcards.data_type),
    output: 
        root =join(out, '{data_type}/NTuples/1_added_features/{decay}/{ID}.root'), 
    log:                            
        join(out, '{data_type}/NTuples/1_added_features/{decay}/.{ID}.log')
    resources:
        max_retries=0,
        request_memory = 10_000,
        mem = 10_000,
        MaxRunHours = add_features_hours, # Bs2DsPi is a bit bigger so give it more time
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
            '--classic_selection_features {input.classical_selection_features}',
            '--data_type {wildcards.data_type}',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule combine_small_files:
    input:
        script = join(repo, 'scripts/combine_dataframes.py'),
        data = lambda wildcards: np.array(feat_added_data[wildcards.decay if wildcards.decay != 'Bs2JpsiKst' else 'Bd2JpsiKst'])[combine_indices[wildcards.decay][int(wildcards.ID)]],
    output:
        join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.root'),
    log: 
        join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.log'),
    resources:
        max_retries=0,
        request_memory = 70_000,
        mem = 70_000,
        MaxRunHours = 4, # medium queue to avoid memory issues
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

        print(' '.join(cmd))
        shell(' '.join(cmd))

def get_to_split(wildcards):
    decay = wildcards.decay
    if decay == 'Bs2JpsiKst':
        decay = 'Bd2JpsiKst' # For Bs2JpsiKst use the Bs fraction of Bd2JpsiKst tuples


    if wildcards.data_type == 'MC' or wildcards.decay == 'Bs2DsPi': 
        #Skipp combining for MC as its not needed and Bs2DsPi as the files are too big
        return join(out, f'{wildcards.data_type}/NTuples/1_added_features/{wildcards.decay}/{wildcards.ID}.root')
    else:
        return join(out, f'Data/NTuples/1_added_features/{wildcards.decay}/combined/{wildcards.ID}.root'),
        

def get_memory_split_and_eventSelect(wildcards):
    if wildcards.decay == 'Bs2DsPi': 
        return 128_000
    else:
        return 32_000

rule split_sample:
    input:
        script = join(repo, 'scripts/split_train_val_test.py'),
        to_split = get_to_split,
                                

        hyper_int = join(repo, 'configs/hyperpar_intervals.yaml'), # For the train-val proportions
    output:
        train      = join(out, '{data_type}/NTuples/2_split/{decay}/train/{ID}.root'),
        validation = join(out, '{data_type}/NTuples/2_split/{decay}/validation/{ID}.root'),
        test       = join(out, '{data_type}/NTuples/2_split/{decay}/test/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/2_split/{decay}/log/.{ID}.log'),
    resources:
        max_retries=0,
        request_memory = get_memory_split_and_eventSelect,
        mem = get_memory_split_and_eventSelect,
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
            '--decay {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

def get_obs_name(decay):
    if decay == 'Bu2JpsiK':
        return 'B_DTF_PV_Jpsi_MASS'
    elif decay == 'Bd2JpsiKst':
        return 'B_DTF_PV_Jpsi_MASS'
    elif decay == 'Bs2JpsiKst':
        return 'B_DTF_PV_Jpsi_MASS'
    elif decay == 'Bs2DsPi':
        return 'B_DTF_PV_Ds_MASS'
    else:
        raise ValueError(f'Unknown decay {decay}')

rule train_signal_classifier:
    input:
        script = join(repo, 'scripts/train_BDT.py'),
        data = lambda wildcards: combined_data[wildcards.decay] if wildcards.decay != 'Bs2DsPi' else feat_added_data[wildcards.decay],
        mc = lambda wildcards: feat_added_mc[wildcards.decay], 
        signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
    output:
        BDT = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl')
    log:
        join(out, 'Data/signal_classifier/{decay}/BDT_train.log')
    resources:
        max_retries=0,
        request_memory = 20_000,
        mem = 20_000,
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
            f'--massname {get_obs_name(wildcards.decay)}',
            '--num_threads {threads}',
            '--signal_class_features', input.signal_class_features,
            f'--USB_start {massranges[wildcards.decay][1]}',
            '&> {log}',
        ]


        shell(' '.join(cmd))

rule event_selection: #Applies BDT signal selection and in case a bin is supplied in addition to the decay, also cuts away events outside the bin
    input:
        script = join(repo, 'scripts/apply_event_selection.py'),
        data   = join(out,  '{data_type}/NTuples/2_split/{decay}/{partition}/{ID}.root'),

        signal_class_features        = join(repo, 'configs/signal_classifier_features.yaml'),
        classifier = lambda wildcards: join(out, f'Data/signal_classifier/{wildcards.decay if wildcards.decay != "Bs2JpsiKst" else "Bd2JpsiKst"}/bdt_model.pkl'), 
        classical_selection_features = join(repo, 'configs/classic_selection_features.yaml'),

        bin_file = lambda wildcards:   join(repo, 'configs/binnings.yaml') if wildcards.binning else [],
    output:
        join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/.{ID}.log'),
    resources:
        max_retries=0,
        request_memory = get_memory_split_and_eventSelect,
        mem = get_memory_split_and_eventSelect,
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
            '--classical_selection_features {input.classical_selection_features}',
            '&> {log}',
        ]

        shell(' '.join(cmd))

def get_DT_input_paths():
    all_mc_files = []
    for decay in feat_added_mc.keys():
        all_mc_files.extend(feat_added_mc[decay])

    return [file for file in all_mc_files if file.endswith('01_1.mc.root')]

rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        data = get_DT_input_paths,
        settings = join(repo, 'DT_arguments/{cut_name}.txt'),
    output:
        model =        join(out, "MC/DT_outputs/{cut_name}/decision_tree_model.txt")
        # cuts  = expand(join(out, "MC/DT_outputs/{{cut_name}}/cuts/{tagger}.txt"), tagger=all_taggers), TODO recomment in
    log:               join(out, "MC/DT_outputs/{cut_name}/tree_schema.log")
    resources:
        request_memory = 150_000, # Specify memory requirement in megabytes
        mem = 150_000,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 4,
        requirements='(Machine == "rhino.e5.physik.tu-dortmund.de")',
    run:
        target_path = os.path.dirname(output.model)

        if kernel_available():
            data = path_to_kernel(input.data)
        else:
            data = input.data


        cmd = [
            f'python {input.script}',
            f'--input_files', ' '.join(data),
            f'--target_path {target_path}',
        ]

        #Read settings file and paste all arguments into the command
        with open(input.settings, 'r') as f:
            settings = f.read()
            #remove all line breaks
            settings = settings.replace('\n', ' ')
            split_settings = [f'--{arg} ' for arg in settings.split('--') if arg]

            cmd.extend(split_settings)
        


        print(' '.join(cmd))
        cmd += [f'&> {log}']
        shell(' '.join(cmd))



rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        to_select = lambda wildcards : join(out, '{data_type}/NTuples/2_split/{decay}{binning}/{partition}/{ID}.root') if (wildcards.data_type == 'MC' and wildcards.binning == '') or wildcards.selection != ''
                                  else join(out, '{data_type}/NTuples/3_event_selected/{decay}{binning}/{partition}/{ID}.root'),
    output: join(out, '{data_type}/NTuples/4_track_selected/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:    join(out, '{data_type}/NTuples/4_track_selected/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{partition}/.{ID}.log')
    resources:
        max_retries=0,
        request_memory = 20_000, # Specify memory requirement in megabytes
        mem = 20_000,
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

    paths = [file.replace('train', wildcards.partition).replace(wildcards.decay, f'{wildcards.decay}{wildcards.binning}') for file in files]
    return paths

def get_fit_mc_res(wildcards):
    if wildcards.data_type == 'MC':
        return []
    mc_decay = wildcards.decay if wildcards.decay != "Bs2JpsiKst" else "Bd2JpsiKst"
    return join(out, 'MC/mass_fit/' + f'{mc_decay}' + '{binning}/{partition}/event_{is_selected}_fit.json') 

rule mass_fit:
    input:
        script = join(repo, 'scripts/mass_fit/mass_fits.py'),
        data = get_fit_input_paths,

        mc_res = get_fit_mc_res,
    output: 
        join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/event_{is_selected}_fit.json'),
        join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/weights_{is_selected}.root'), #In case of MC weights is a dummy file containing just the information used in the fitting without weights
    log:    join(out, '{data_type}/mass_fit/{decay}{binning}/{partition}/event_{is_selected}_fit.log'),
    resources:
        max_retries=0,
        request_memory = 32_000, # Specify memory requirement in megabytes
        mem = 32_000,
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
            f'--range {massranges[wildcards.decay][0]} {massranges[wildcards.decay][1]}', 
            f'--obs_name {get_obs_name(wildcards.decay)}',
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
        script       = join(repo, 'scripts/add_weights.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        selected     = join(out, 'Data/NTuples/4_track_selected/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
        weights      = lambda wildcards: join(out, 'Data/mass_fit/{decay}{binning}/{partition}', f'weights{"" if wildcards.selection == "" else "_non"}_selected.root'),
    output: weighted = join(out, 'Data/NTuples/5_weighted/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:               join(out, 'Data/NTuples/5_weighted/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{partition}/{ID}.log'),
    resources:
        max_retries=0,
        request_memory = 40_000, 
        mem = 40_000,
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
           f'--range {massranges[wildcards.decay][0]} {massranges[wildcards.decay][1]}', 
            '--weight_file', weights,
            '--loading_features {input.loading_vars}',
            '&> {log}'
        ]

        shell(' '.join(cmd))

# rule train_tagger_MC: 
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
request_memory = 30_000, # Specify memory requirement in megabytes 
mem = 30_000,
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
request_memory = 40_000, # Specify memory requirement in megabytes 
mem = 40_000,
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

    if 'benchmark_version' in wildcards.keys():
        if wildcards.benchmark_version == 'Run3v1':
            cut_name = 'allBKGCAT_notSamePV_noOSP_SSK_balanced'
            features = 'union_PROBNN_edited_for_benchmark'
        else:
            raise ValueError(f"Unknown benchmark version: {wildcards.benchmark_version}")
    else:
        cut_name = wildcards.cut_name
        features = wildcards.features

    return [f.replace('cut_name', f'{cut_name}').replace('train', 'test').replace(wildcards.decay, f'{wildcards.decay}{wildcards.binning}{wildcards.selection}').replace('features', f'{features}')
            for f in files_dict[f'{wildcards.decay}'][f'{wildcards.tagger}']]

rule test_and_calibrate:
    input:
        testing = get_testing_inputs,
        model = lambda wildcards: 
                join(out, f'{wildcards.data_type_or_adapted}/savedModels/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/training/model.pth'),


        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        logit =  join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/mistag/taggingInfo_mistag.json'),
        calibration_logit =  join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/logit/calibration.json'),
        calibration_mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/mistag/calibration.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{decay}{binning}{selection}/{tagger}/{cut_name}/{features}/{seed}/{config}/testing/{data_type}/testing_log.log')
    priority: -2, # Lower priority for efficient use of requested cores
    resources:
        max_retries=0,
        request_memory = 35_000, # Specify memory requirement in megabytes 
        mem = 35_000,
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

rule test_and_calibrate_benchmark:
    input:
        testing = get_testing_inputs,
        model = join(repo, 'benchmark_tagger/{benchmark_version}/{tagger}/model.pth'),
        calibration_config = join(repo, 'benchmark_tagger/{benchmark_version}/{tagger}/best_calib.yaml'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'benchmark_tagger/{benchmark_version}/{tagger}/model_config.yaml'),
    output:
        logit              = join(out, 'MC/benchmarkModels/{decay}{binning}{selection}/{tagger}/{benchmark_version}/testing/{data_type}/logit/taggingInfo_logit.json'),
        mistag             = join(out, 'MC/benchmarkModels/{decay}{binning}{selection}/{tagger}/{benchmark_version}/testing/{data_type}/mistag/taggingInfo_mistag.json'),
        calibration_logit  = join(out, 'MC/benchmarkModels/{decay}{binning}{selection}/{tagger}/{benchmark_version}/testing/{data_type}/logit/calibration.json'),
        calibration_mistag = join(out, 'MC/benchmarkModels/{decay}{binning}{selection}/{tagger}/{benchmark_version}/testing/{data_type}/mistag/calibration.json'),
    log:                     join(out, 'MC/benchmarkModels/{decay}{binning}{selection}/{tagger}/{benchmark_version}/testing/{data_type}/testing_log.log'),
    priority: -2, # Lower priority for efficient use of requested cores
    resources:
        max_retries=0,
        request_memory = 35_000, # Specify memory requirement in megabytes 
        mem = 35_000,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 4
    run:
        outpath = os.path.dirname(os.path.dirname(output[0]))
        model_path = os.path.dirname(input.model)

        if kernel_available():
            test_kernel = path_to_kernel(input.testing)
        else:
            test_kernel = input.testing


        cmd = [
            'python', input.script,
            '--testing_data', ' '.join(test_kernel),
            '--target_path', outpath,
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features union_PROBNN_edited_for_benchmark',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed 45',
            '--repo', repo,
            '--data_type {wildcards.data_type}',
            '--model_path', model_path,
            '--calibration_config {input.calibration_config}',
            '--benchmark_version {wildcards.benchmark_version}',
            '&> {log}',
        ]

        shell(' '.join(cmd))

# def extract_best(tagger, cut, data_type,link='logit', BN = ''):
def extract_best(tagger, cut, model_type ,link='logit'):
    #Read the best tagger candidate config from json file with the best hyperparameter combination

    

    with open(join(repo, f'best_tagger_candidates/{cut}/{model_type.removeprefix("trained_")}/candidatedTaggers_{link}.json'), 'r') as f: # trainOn
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
        BN = '_BN' if 'BN' in model_type else ''

        nl = int(data[tagger]['numlayers'])
        nn = int(data[tagger]['numneurons'])
        config = f'lr{lr}_bs{bs}_nL{nl}_nN{nn}{BN}'
        return {'config':config, 'seed':seed, 'lr':lr, 'bs':bs, 'numlayers':nl, 'numneurons':nn, 'tagger':tagger, 'cut':cut, 'link':link}

def get_model_path(wildcards, all_taggers=False):
    # if 'Run3v' not in wildcards.model_types:
    #     data_type = wildcards.model_types.removeprefix('trained_')
    # else:
    #     data_type = wildcards.data_type

    if not all_taggers:
        tagger = [wildcards.tagger]
    else:
        tagger = get_taggers_from_combination(wildcards.combinationName)
        
    if 'Run3v' not in wildcards.model_types:
        
        if all_taggers:
            decay = [wildcards.decay for _ in tagger]
        else:
            decay = [extract_decay(tag) for tag in tagger]

        # BN = 'BN' if 'BN' in wildcards.model_types else ''
        # best = [extract_best(tagger=tag, cut=cut_name, data_type=data_type, BN=BN) for tag in tagger]
        best = [extract_best(tagger=tag, cut=wildcards.cut_name, model_type=wildcards.model_types) for tag in tagger]
        model = [join(out, f'{wildcards.data_type}/savedModels/{dec}/{tag}/{wildcards.cut_name}/{wildcards.features}/{bes["seed"]}/{bes["config"]}/training/model.pth') for tag, dec, bes in zip(tagger, decay, best)]
    else:
        model = [join(repo, f'benchmark_tagger/{wildcards.model_types}/{tag}/model.pth') for tag in tagger]

    if not all_taggers:
        model = model[0]
    return model

def get_best_link(wildcards, tagger=None):
    if tagger is None:
        tagger = wildcards.tagger


    if 'Run3v' in wildcards.model_types:
        with open(join(repo, f'benchmark_tagger/{wildcards.model_types}/{tagger}/best_calib.yaml'), 'r') as f:
            best = yaml.safe_load(f)
        return best.get('link')
    else:
        return 'logit'


rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        to_tag = lambda wildcards: join(out, '{data_type}/NTuples/5_weighted/{decay}{selection}/{tagger}/{cut_name}/{features}/test/{ID}.root') if wildcards.data_type == 'Data' 
                              else join(out, '{data_type}/NTuples/4_track_selected/{decay}{selection}/{tagger}/{cut_name}/{features}/test/{ID}.root'),
        

        model       = lambda wildcards: get_model_path(wildcards),
        scaler      = lambda wildcards: get_model_path(wildcards).replace('training/model.pth', 'training/st_scaler.pkl'), 
        transformer = lambda wildcards: get_model_path(wildcards).replace('training/model.pth', 'training/powerTransformer.pkl'), 
        calibration = lambda wildcards: get_model_path(wildcards).replace('training/model.pth', f'testing/{wildcards.data_type}/logit/calibration.json')
                        if 'Run3v' not in wildcards.model_types 
                        else join(out, f'MC/benchmarkModels/{wildcards.decay}{wildcards.selection}/{wildcards.tagger}/{wildcards.model_types}/testing/{wildcards.data_type}/{get_best_link(wildcards)}/calibration.json'),


        # config = lambda wildcards: join(repo, f'model_configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name, data_type=wildcards.data_type_or_adapted, BN=wildcards.BN).get("config")}.yaml'),
        config = lambda wildcards: join(repo, f'model_configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name, model_type=wildcards.model_types).get("config")}.yaml') 
                              if 'Run3v' not in wildcards.model_types 
                              else join(repo, f'benchmark_tagger/{wildcards.model_types}/{wildcards.tagger}/model_config.yaml'),
    output:
        root = join(out, '{data_type}/NTuples/6_tagged/{decay}{selection}/{tagger}/{cut_name}/{features}/{model_types}/{ID}.root'),
    log:
        join(out, '{data_type}/NTuples/6_tagged/{decay}{selection}/{tagger}/{cut_name}/{features}/{model_types}/{ID}.log'),
    resources:
        max_retries=0,
        request_memory = 40_000, 
        mem = 40_000,
        MaxRunHours = 1, 
    run:
        # config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name, data_type=wildcards.data_type_or_adapted, BN=wildcards.BN)

        if 'Run3v' in wildcards.model_types:
            with open(join(repo, f'benchmark_tagger/{wildcards.model_types}/{wildcards.tagger}/best_calib.yaml'), 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name, model_type=wildcards.model_types)



        # with open(input.config, 'r') as f:
        #     config = yaml.safe_load(f)

        # domain = '--domain_adapted' if wildcards.data_type_or_adapted == 'domain_adapted' else ''
        domain = '--domain_adapted' if 'domain_adapted' in wildcards.model_types else ''
        benchmark_version = f'--benchmark_version {wildcards.model_types}' if 'Run3v' in wildcards.model_types else ''

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
            benchmark_version,
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
    model_types = wildcards.model_types

    if 'Run3v0' == model_types:
        # Run3v0 are the Run2 models. The tagging information for these is stored in every tuple. We can use the Run3v1 tuples for them
        model_types = 'Run3v1'

    selection = wildcards.selection

    pre_path = join(out, f'{data_type}{selection}/Ntuples/6_tagged/{decay}')

    all_paths = []
    for tagger in taggers:
        paths = tagged_mc[decay][tagger] if data_type == 'MC' else tagged_data[decay][tagger]
        paths = np.array(paths)
        paths = np.char.replace(paths, 'cut_name/features', f'{cut_name}/{features}')
        paths = np.char.replace(paths, 'trained_on', model_types)
        all_paths = np.concatenate((all_paths, paths))

    return all_paths

def get_taggers_from_combination(combinationName):
    return combinationName.split('_')

def get_calibrations_for_combination(wildcards):
    
    if "Run3v" in wildcards.model_types:
        if wildcards.model_types == 'Run3v0':
            return []

        return [join(out, f'MC/benchmarkModels/{wildcards.decay}{wildcards.selection}/{tagger}/{wildcards.model_types}/testing/{wildcards.data_type}/{get_best_link(wildcards, tagger)}/calibration.json') 
                for tagger in get_taggers_from_combination(wildcards.combinationName)]
    else:
        return [path.replace(f'{wildcards.decay}', f'{wildcards.decay}{wildcards.selection}').replace('training/model.pth', f'testing/{wildcards.data_type}/logit/calibration.json') 
                for path in get_model_path(wildcards, all_taggers=True)]

rule combine_tagger: 
    input:
        script = join(repo, 'scripts/combineTagger.py'),
        tagged = get_tagged_paths,

        calibration = get_calibrations_for_combination,
    output:
        pdf=              join(out, '{data_type}/savedModels/{decay}{selection}/combinations/Run3/{model_types}/{cut_name}/{features}/{combinationName}/{combinationName}_Calibration.pdf'),
        all_tagged_data = join(out, '{data_type}/savedModels/{decay}{selection}/combinations/Run3/{model_types}/{cut_name}/{features}/{combinationName}/combined_tagged.root'),
    log:                  join(out, '{data_type}/savedModels/{decay}{selection}/combinations/Run3/{model_types}/{cut_name}/{features}/{combinationName}/{combinationName}_log.log')
    resources:
        max_retries=0,
        request_memory = 40_000, 
        mem = 40_000,
        MaxRunHours = 4,
    run:
        tagged_prePath = join(input.tagged[0].split('6_tagged')[0], '6_tagged/')
        out_path = os.path.dirname(output.pdf)

        # Get the first tagged file and replace the tagger part with a placeholder to construct the path to the calibration files
        first_tagged = input.tagged[0]

        for tagger in get_taggers_from_combination(wildcards.combinationName):
            if tagger in first_tagged:
                tagger_placeholder_path = os.path.dirname(first_tagged.replace(tagger, 'tagger_placeholder'))
                break

        if wildcards.model_types == 'Run3v0':
            run = 'Run2'
        else:
            run = 'Run3'


        cmd = [
            'python', input.script,
            '--tagger', ' '.join(get_taggers_from_combination(wildcards.combinationName)),
            # f'--tagged_prePath {tagged_prePath}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            f'--input_files {tagger_placeholder_path}',
            f'--outputPath {out_path}',
            '--features {wildcards.features}',
            '--cut {wildcards.cut_name}',
            '--data_type {wildcards.data_type}',
            # '--trained_on {wildcards.data_type_or_adapted}',
            '--calibrations', ' '.join(input.calibration),
            '--run', run,
            '&> {log}',
        ]
        print(' '.join(cmd))
        shell(' '.join(cmd))








###TILL HERE THE PIPELINE IS REWORKED AND FUNCTIONAL, RULES BELOW MAY NEED TO BE ADJUSTED TO NEW FOLDER STRUCTURE AND SCRIPT-CHANGES










# rule combine_MC_Data: #Combines data and MC for domain adaptation
#     input:
#         script = join(repo, 'scripts/combine_dataframes.py'),
#         loading_vars = join(repo, 'configs/loading_variables.txt'),
#         # data = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f in ntuples_tagged_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']],
#         data = lambda wildcards: [join(out, f'Data/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.data_portion}/{ID}.root') for ID in data_ids],
#         MC = lambda wildcards: [
#             f.replace('cutName', f'{wildcards.cut_name}').replace('train', f'{wildcards.data_portion}')
#             for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
#         ],
#     output:
#         root = [join(out, 'domain_adapted/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}', f'samples_{i}.root') for i in range(combined_df_n_splits[wildcards.decay])],
#     log:
#         join(out, 'domain_adapted/5_split/{decay}/{tagger}/{cut_name}/{features}/{data_portion}/log.log'),
#     wildcard_constraints:
#         data_portion = 'train|validation',
#     resources:
#         max_retries=0,
#         request_memory = 25_000, 
#         mem = 25_000,
#         MaxRunHours = 2, # short queue
#     run:
#         out_path = os.path.dirname(output.root[0])

#         #Use ceph-kernel if available to increase file reading performance
#         if kernel_available():
#             data = path_to_kernel(input.data)
#             MC   = path_to_kernel(input.MC)
#         else:
#             data = input.data
#             MC   = input.MC

#         cmd = [
#             'python {input.script}',
#             '--data_files', ' '.join(data),
#             '--mc_files', ' '.join(MC),
#             '--target_path', out_path,
#             '--splits ', str(combined_df_n_splits[wildcards.decay]),
#             '--loading_features {input.loading_vars}',
#             '&> {log}'
#         ]
#         shell(' '.join(cmd))


# rule train_tagger_domain_adapted:
#     input:
#         script = join(repo, 'scripts/train_tagger.py'),
#         train = lambda wildcards: 
#             [join(out, f"domain_adapted/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/train/samples_{i}.root") 
#              for i in range(combined_df_n_splits[wildcards.decay])], 
#         val =   lambda wildcards: 
#             [join(out, f"domain_adapted/5_split/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/validation/samples_{i}.root")
#              for i in range(combined_df_n_splits[wildcards.decay])],

#         config = join(repo, 'model_configs/{config}.yaml'),
#     output:
#         model=       join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
#         scaler=      join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
#         transformer= join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),

#     log:
#         join(out, 'domain_adapted/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/training_log.log'),
#     resources:
#         max_retries=0,
#         request_memory = 30_000, 
#         mem = 30_000,
#         MaxRunHours = 16, # long queue
#         threads = 8, #
#     threads:
#         8,
#     run:
#         train_scratch = copy_to_scratch(input.train)
#         val_scratch = copy_to_scratch(input.val)

#         outpath = os.path.dirname(output.model)

#         # shell('sleep $(($RANDOM%200))')  # Sleep for a random time to make race conditions less likely, up to 200 seconds

#         cmd = [
#             'python', input.script,
#             '--training_data', ' '.join(train_scratch),
#             '--validation_data', ' '.join(val_scratch),
#             '--tagger {wildcards.tagger}',
#             '--seed {wildcards.seed}',
#             '--features {wildcards.features}',
#             '--decay_type {wildcards.decay}',
#             '--repo', repo,
#             '--data_type domain_adapted',
#             # '--balance_dataset',
#             '--target_path', outpath,
#             '--config {input.config}',
#             '--num_threads {threads}',
#             '&> {log}',
#         ]

#         shell(' '.join(cmd))


