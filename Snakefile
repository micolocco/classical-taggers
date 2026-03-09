from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import pandas as pd
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

    #In how many sections combined dataframes are splint into when usind domain adaptation or combining small data files
    combined_df_n_splits = config['combined_df_n_splits'] 
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
files_s24c2 = [line.strip() for line in files_s24c2]
raw_path = os.path.dirname(files_s24c2[0])
data_ids = [os.path.basename(i)[:-5] for i in files_s24c2]


combine_indices = [i for i in range(len(files_s24c2))]
np.random.seed(42)
np.random.shuffle(combine_indices)
combine_indices = np.array_split(combine_indices, combined_df_n_splits)

mc_ids = {}
for decay in taggers_conf.keys():
    path = join(MC, 'withUT_MC_2024/1_raw/' + decay + '/')

    dec_ids = []
    for f in os.listdir(join(path)):
        if '.root' in f:
            dec_ids.append(f)
    mc_ids[decay] = dec_ids


#Create lists of all file names after each step
#Data
feat_added_data = {
    'Bu2JpsiK': [join(out, 'Data/NTuples/1_added_features/Bu2JpsiK', os.path.basename(f)) for f in files_s24c2],
    'Bd2JpsiKst': [join(out, 'Data/NTuples/1_added_features/Bd2JpsiKst', os.path.basename(f)) for f in files_s24c2]
}

combined_data = {}
for decay, path_list in feat_added_data.items():
    path = os.path.dirname(path_list[0])
    combined_data.update({decay: [join(path, f'combined/samples_{i}.root') for i in range(combined_df_n_splits)]})

train_split_data = {}
for decay, path_list in combined_data.items():
    train_split_data.update({decay: [f.replace(f'1_added_features/{decay}/combined', f'2_split/{decay}').replace(decay, f'{decay}/train') for f in path_list]})


selected_data = {}
for decay, path_list in train_split_data.items():
    selected_data.update({decay: {}})
    for tagger in taggers_conf[decay] : 
        selected_data[decay].update({tagger: [f.replace('2_split', f'3_selected').replace('train', f'{tagger}/cut_name/features/train') for f in path_list]})

weighted_data = {}
for decay, path_list in selected_data.items():
    weighted_data.update({decay: {}})
    for tagger, path_list in path_list.items():
        weighted_data[decay].update({tagger: [f.replace('3_selected', f'4_weighted') for f in path_list]})



#MC
feat_added_mc = deepcopy(ntuples_eos_withUT)
for decay, path_list in feat_added_mc.items():
    feat_added_mc[decay] = [f.replace(os.path.dirname(f), f'{out}MC/NTuples/1_added_features/{decay}') for f in path_list]

train_split_mc = {}
for decay, path_list in feat_added_mc.items():
    train_split_mc.update({decay: [f.replace('1_added_features', f'2_split').replace(decay, f'{decay}/train') for f in path_list]})

selected_mc = {}
for decay, path_list in train_split_mc.items():
    selected_mc.update({decay: {}})
    for tagger in taggers_conf[decay] :
        selected_mc[decay].update({tagger: [f.replace('2_split', f'3_selected').replace('train', f'{tagger}/cut_name/features/train') for f in path_list]})


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


weights = [
    'ones',
    'signal_weights',
    'pdf_ratio'
]

wildcard_constraints:
    data_type   = '(MC|Data)',
    decay       = '(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)',
    tagger      = '(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)',
    weight      = '|'.join(weights),
    data_type_or_adapted = '(Data|MC|domain_adapted)',
    partition = '(train|validation|test)',
    cut_name = "[^/]+", #don't allow slashes in wildcards to avoid problems with paths
    features = "[^/]+",
    seed = '[^/]+',
    config = '[^/]+',
    ID = '[^/]+',


rule all:
    input:
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/{seed}/lr0.001_bs8192_nL6_nN64/testing/Data/logit/taggingInfo_logit.json',
            tagger = ['OSKaon', 'OSMuon', 'OSElectron'], seed = [1, 3, 12, 18, 22, 28, 32, 42, 55, 65, 71, 81, 101, 111, 121, 123]),
        expand('/ceph/users/togasa/FlavourTagging/MC/savedModels/Bu2JpsiK/{tagger}/notSamePV_noOSP/union_PROBNN/{seed}/lr0.001_bs8192_nL6_nN64_BN/testing/Data/logit/taggingInfo_logit.json',
            tagger = ['OSKaon', 'OSMuon', 'OSElectron'], seed = [1, 3, 12, 18, 22, 28, 32, 42, 55, 65, 71, 81, 101, 111, 121, 123]),



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


def get_raw_paths(decay, ID, data_type):
    if data_type == 'MC':
        return join(MC, f'{decay}/v1_taggers/{ID}.root')
    elif data_type == 'Data':
        return join(data, f'{ID[:8]}/{ID[9:13]}' + f'/{ID}.root')
    else:
        print(f"data type is {data_type} instead of MC or Data. Somethings broken")
        raise RuntimeError

# rule add_features:
#     input:
#         script = join(repo, 'scripts/adding_features.py'),
#         loading_vars = join(repo, 'configs/loading_variables.txt'),
#         signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
#         raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.ID, wildcards.data_type),
#     output: 
#         root =join(out, '{data_type}/NTuples/1_added_features/{decay}/{ID}.root'), 
#     log:                            
#         join(out, '{data_type}/NTuples/1_added_features/{decay}/.{ID}.log')
#     resources:
#         max_retries=0,
#         mem_mb = 10_000,
#         MaxRunHours = 1, # short queue
#     run:
#         tree = find_tree_name(wildcards.decay)
#         dataCalib = '--data_calib' if wildcards.data_type == 'Data' else ''

#         cmd = [
#             'python', input.script,
#             '--raw {input.raw}',
#             '--output {output}',
#             '--evtType {wildcards.decay}',
#             '--treename', tree,
#             '--loading_features {input.loading_vars}',
#             '--signal_class_features {input.signal_class_features}',
#             f'{dataCalib}',
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))


# rule combine_small_files:
#     input:
#         script = join(repo, 'scripts/combine_dataframes.py'),
#         data = lambda wildcards: np.array(feat_added_data[wildcards.decay])[combine_indices[int(wildcards.ID)]],
#     output:
#         join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.root'),
#     log: 
#         join(out, 'Data/NTuples/1_added_features/{decay}/combined/samples_{ID}.log'),
#     resources:
#         max_retries=0,
#         mem_mb = 90_000,
#         MaxRunHours = 1, # short queue
#     run:
#         path = os.path.dirname(output[0])

#         if kernel_available():
#             data = path_to_kernel(input.data)
#         else:
#             data = input.data

#         cmd = [
#             f'python {input.script} ',
#             f'--data_files ', ' '.join(data),  
#             f'--target_path {path} ',  
#             f'--treename "DecayTree;1" ', 
#             f'--splits 1',  
#             f'--evtType {wildcards.decay} ', 
#             f'--index {wildcards.ID} ',
#             f'&> {log}', 
#         ]

#         shell(' '.join(cmd))

# rule split_sample:
#     input:
#         script = join(repo, 'scripts/split_train_val_test.py'),
#         to_split = lambda wildcards: join(out, f'MC/NTuples/1_added_features/{wildcards.decay}/{wildcards.ID}.root') if wildcards.data_type == 'MC' 
#                                 else join(out, f'Data/NTuples/1_added_features/{wildcards.decay}/combined/{wildcards.ID}.root'),

#         hyper_int = join(repo, 'configs/hyperpar_intervals.yaml'), # For the train-val proportions
#     output:
#         train      = join(out, '{data_type}/NTuples/2_split/{decay}/train/{ID}.root'),
#         validation = join(out, '{data_type}/NTuples/2_split/{decay}/validation/{ID}.root'),
#         test       = join(out, '{data_type}/NTuples/2_split/{decay}/test/{ID}.root'),
#     log:
#         join(out, '{data_type}/NTuples/2_split/{decay}/log/.{ID}.log'),
#     resources:
#         max_retries=0,
#         mem_mb = 65_000,
#         MaxRunHours = 1,
#     run:
#         out_path = os.path.dirname(os.path.dirname(output.train))
#         treename = '"DecayTree;1"' 
        
#         cmd = [
#             'python', input.script,
#             '--to_split {input.to_split}',
#             '--target_path', out_path,
#             '--config {input.hyper_int}',
#             '--treename', treename,
#             '--data_type {wildcards.data_type}',
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))

# rule train_signal_classifier:
#     input:
#         script = join(repo, 'scripts/train_BDT.py'),
#         data = lambda wildcards: combined_data[wildcards.decay],
#         mc = lambda wildcards: feat_added_mc[wildcards.decay], 
#         signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
#     output:
#         BDT = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl')
#     log:
#         join(out, 'Data/signal_classifier/{decay}/BDT_train.log')
#     resources:
#         max_retries=0,
#         mem_mb = 20_000,
#         MaxRunHours = 4,
#     threads:
#         8,
#     run:
#         out_path = os.path.dirname(output.BDT)

#         if kernel_available():
#             data = path_to_kernel(input.data)
#             mc   = path_to_kernel(input.mc)
#         else:
#             data = input.data
#             mc   = input.mc

#         cmd = [
#             'python', input.script,
#             '--real_data', ' '.join(data),
#             '--mc_data', ' '.join(mc),
#             '--target_path', out_path,
#             '--treename "DecayTree;1"',
#             '--decay_type {wildcards.decay}',
#             '--massname B_DTF_PV_Jpsi_MASS',
#             '--num_threads {threads}',
#             '--signal_class_features', input.signal_class_features,
#             '&> {log}',
#         ]


#         shell(' '.join(cmd))

# rule MC_Mass_Fit:
#     input:
#         script = join(repo, 'scripts/mass_fits.py'),
#         data = lambda wildcards: [i.replace('train', wildcards.partition) for i in train_split_mc[wildcards.decay]],
#         BDT = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl'),
#         signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
#     output:
#         join(out, 'MC/mass_fit/{decay}/{partition}/fit_before_cut.json'),
#         mc_res = join(out, 'MC/mass_fit/{decay}/{partition}/fit_after_cut.json'),
#     log:
#         join(out, 'MC/mass_fit/{decay}/{partition}/fit.log'),
#     resources:
#         max_retries=0,
#         mem_mb = 32_000, # Specify memory requirement in megabytes
#         MaxRunHours = 4, # medium queue
#     threads:
#         4,
#     run:
#         out_path = os.path.dirname(output.mc_res)

#         if kernel_available():
#             data = path_to_kernel(input.data)
#         else:
#             data = input.data


#         cmd = [
#             'python {input.script}',
#             '--input_files', ' '.join(data),
#             '--range {lowerMass} {upperMass}', 
#             '--treename "DecayTree;1"',
#             '--simulation',
#             '--BDT {input.BDT}',
#             '--output', out_path,
#             '--decay_type {wildcards.decay}',
#             '--num_threads {threads}',
#             '--signal_class_features {input.signal_class_features}',
#             '&> {log}'
#         ]
#         shell(' '.join(cmd))

# rule data_Mass_Fit: 
#     input:
#         script = join(repo, 'scripts/mass_fits.py'),
#         mc_res = join(out, 'MC/mass_fit/{decay}/{partition}/fit_before_cut.json'),
#         data = lambda wildcards: [i.replace('train', wildcards.partition) for i in train_split_data[wildcards.decay]],
#         signal_class_features = join(repo, 'configs/signal_classifier_features.yaml'),
#         BDT = join(out, 'Data/signal_classifier/{decay}/bdt_model.pkl'),
#     output:
#         data_res = join(out, 'Data/mass_fit/{decay}/{partition}/fit_after_cut.json'),
#         weights = join(out, 'Data/mass_fit/{decay}/{partition}/weights.root'),
#     log:
#         join(out, 'Data/mass_fit/{decay}/{partition}/fit.log'),
#     resources:
#         max_retries=0,
#         mem_mb = 20_000, 
#         MaxRunHours = 4,
#     threads:
#         8,
#     run:
#         out_path = os.path.dirname(output.data_res)

#         #Use ceph-kernel if available to increase file reading performance
#         if kernel_available():
#             data = path_to_kernel(input.data)
#         else:
#             data = input.data



#         cmd = [
#             'python {input.script}',
#             '--input_files', ' '.join(data),
#             '--range {lowerMass} {upperMass}', 
#             '--obs_name B_DTF_PV_Jpsi_MASS',
#             '--treename "DecayTree;1"',
#             '--output', out_path,
#             '--decay_type {wildcards.decay}',
#             '--sim_fit {input.mc_res}',
#             '--cut notSamePV_noOSP',
#             '--BDT {input.BDT}',
#             '--num_threads {threads}',
#             '--signal_class_features {input.signal_class_features}',

#             '&> {log}'
#         ]
#         shell(' '.join(cmd))

# rule train_DT:
#     input:
#         script = join(repo, 'scripts/origin_DT_cut.py'),
#         data = join(out, 'MC/2_added_features'),
#     output:
#         pdf = join(out, 'MC/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.pdf'),
#     log:
#         join(out, 'MC/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.log')
#     resources:
#         max_retries=0,
#         mem_mb = 20_000, # Specify memory requirement in megabytes
#         OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
#         MaxRunHours = 8, # medium queue
#     run:
#         target_path = os.path.dirname(os.path.dirname(output.pdf))

#         cmd = (
#             f'python {input.script} '
#             f'--base_pattern {input.data} '  # Pass input root files
#             f'--target_path {target_path} '  # Pass the target path
#             f'--balanced {wildcards.balanced} '  # Specify if classes are balance dor not
#            # f'--unify_SS '  # Specify if SSKaon and SSProton should be unified in single class
#             f'--BKG0 '
#             f'&> {log}'  # Redirect stdout and stderr to log file
#         )
#         shell(cmd)


rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        to_select = join(out, '{data_type}/NTuples/2_split/{decay}/{partition}/{ID}.root'),
    output: join(out, '{data_type}/NTuples/3_selected/{decay}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:    join(out, '{data_type}/NTuples/3_selected/{decay}/{tagger}/{cut_name}/{features}/{partition}/.{ID}.log')
    resources:
        max_retries=0,
        mem_mb = 20_000, # Specify memory requirement in megabytes
        MaxRunHours = 1, # short queue
    run:
        BKG0 = '--BKG0' if wildcards.data_type == 'MC' else ''
        cut_file = join(repo, 'cuts/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt')
       
        cmd = [
            'python', input.script,
            '--to_select {input.to_select}',
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
        MaxRunHours = 4, 
    threads:
        4,
    run:
        if kernel_available():
            train = path_to_kernel(input.train)
            val = path_to_kernel(input.val)
        else:
            train = input.train
            val = input.val


        outpath = os.path.dirname(output.model)

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

rule add_weights:
    input:
        script = join(repo, 'scripts/add_weights.py'),
        loading_vars = join(repo, 'configs/loading_variables.txt'),
        selected = join(out, 'Data/NTuples/3_selected/{decay}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
        weights = join(out, 'Data/mass_fit/{decay}/{partition}/weights.root'),
    output:
        weighted = join(out, 'Data/NTuples/4_weighted/{decay}/{tagger}/{cut_name}/{features}/{partition}/{ID}.root'),
    log:
        join(out, 'Data/NTuples/4_weighted/{decay}/{tagger}/{cut_name}/{features}/{partition}/{ID}.log'),
    resources:
        max_retries=0,
        mem_mb = 40_000, 
        MaxRunHours = 1, 
    run:
        out_path = os.path.dirname(output.weighted)

        cmd = [
            'python {input.script}',
            '--data_file {input.selected}',
            '--out_path', out_path,
            '--decayType {wildcards.decay}',
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--range {lowerMass} {upperMass}',
            '--weight_file {input.weights}',
            '--loading_features {input.loading_vars}',
            '&> {log}'
        ]

        shell(' '.join(cmd))

rule train_tagger_data:
    input:
        script = join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}')
            for f in selected_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
        ],
        val = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'validation')
            for f in selected_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],

        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        model=       join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_type}/training/model.pth'),
        scaler=      join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_type}/training/st_scaler.pkl'),
        transformer= join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_type}/training/powerTransformer.pkl'),
    log:
        join(out,'Data/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_type}/training/training_log.log'),
    priority: -1, # Lower priority for tagger training so all prior steps are executed first
    resources:
        max_retries=0,
        mem_mb = 40_000, # Specify memory requirement in megabytes 
        MaxRunHours = 8, # long queue
    threads:
        8,
    run:
        if kernel_available():
            train = path_to_kernel(input.train)
            val = path_to_kernel(input.val)
        else:
            train = input.train
            val = input.val


        outpath = os.path.dirname(output.model)

        cmd = [
            'python', input.script,
            '--training_data', ' '.join(train),
            '--validation_data', ' '.join(val),
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--decay_type {wildcards.decay}',
            '--weight_type {wildcards.weight_type}',
            '--repo', repo,
            '--data_type Data',
            '--target_path', outpath,
            '--config {input.config}',
            '--num_threads {threads}',
            '&> {log}',
        ]

        shell(' '.join(cmd))

rule calibrate_on_MC:
    input:
        testing = lambda wildcards: [
            f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'test')
            for f in selected_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        model = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        logit = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/MC/testing_log.log')
    wildcard_constraints:
        weight_or_empty = '(' + '|'.join([i + '/' for i in weights] + ['']) + ')', #For Data trained taggers needs to represent the weight, for MC it is empty
    priority: -2, # Lower priority for efficient use of requested cores
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 2,
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
        testing = lambda wildcards: [f.replace('cut_name', f'{wildcards.cut_name}').replace('features', f'{wildcards.features}').replace('train', 'test')
            for f in weighted_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        model = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'model_configs/{config}.yaml'),
    output:
        logit  = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/logit/taggingInfo_logit.json'),
        mistag = join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, '{data_type_or_adapted}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_or_empty}testing/Data/testing_log.log')
    wildcard_constraints:
        weight_or_empty = '(' + '|'.join([i + '/' for i in weights] + ['']) + ')', #For Data needs to represent the weight, for MC it is empty
    priority: -2, # Lower priority for efficient use of requested cores
    resources:
        max_retries=0,
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 2,
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







###TILL HERE THE PIPELINE IS REWORKED AND FUNCTIONAL, RULES BELOW MAY NEED TO BE ADJUSTED TO NEW FOLDER STRUCTURE AND SCRIPT-CHANGES






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
    tagger = wildcards.tagger
    decay = extract_decay(tagger)
    cut_name = wildcards.cut_name
    features = wildcards.features
    best = extract_best(tagger=tagger, cut=cut_name,data_type=data_type)


    model = join(out, f'{data_type}/savedModels/{decay}/{tagger}/{cut_name}/{features}/{best["seed"]}/{best["config"]}')

    if data_type == 'Data':
        model = join(model, 'pdf_ratio')
    
    return join(model, 'training')


rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        split = join(out, '{data_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/test/{ID}.root'),

        model= lambda wildcards: join(get_model_path(wildcards), 'model.pth'),
        transformer=lambda wildcards: join(get_model_path(wildcards),'powerTransformer.pkl'), 
        scaler=lambda wildcards: join(get_model_path(wildcards),'st_scaler.pkl'), 

        config = lambda wildcards: join(repo, f'model_configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type_or_adapted).get("config")}.yaml'),
    output:
        root = join(out, '{data_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{ID}.root'),
    log:
        join(out, '{data_type}/6_tagged/{decay}/{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/{ID}.log'),
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
    decay = wildcards.decay
    cut_name = wildcards.cut_name
    features = wildcards.features
    data_type_or_adapted = wildcards.data_type_or_adapted

    pre_path = join(out, f'{data_type}/6_tagged/{decay}')

    all_paths = []
    for tagger in taggers:
        pre_path_tagger = join(pre_path, f'{tagger}/{cut_name}/{features}/trained_{data_type_or_adapted}/')
        if data_type == 'Data':
            ids = data_ids
        else:
            ids = [basename(f)[:-5] for f in ntuples_train_split_withUT_mc[decay][tagger]]

        all_paths.extend(join(pre_path_tagger, f'{ID}.root') for ID in ids)

    return all_paths

rule combine_tagger: 
    input:
        script = join(repo, 'scripts/combineTagger.py'),

        tagged = get_tagged_paths,
    output:
        pdf=join(out, '{data_type}/savedModels/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_Calibration.pdf'),
    log:    join(out, '{data_type}/savedModels/{decay}/combinations/Run3/trained_{data_type_or_adapted}/{cut_name}/{features}/{combinationName}_Run3_log.log')
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


