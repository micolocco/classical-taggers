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
    # if batched:
    #     ruleorder: batched_train_tagger_data > train_tagger_data 
    #     ruleorder: batched_train_tagger_MC > train_tagger_MC 
    # else:
    #     ruleorder: train_tagger_data > batched_train_tagger_data
    #     ruleorder: train_tagger_MC > batched_train_tagger_MC
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
files_s24c2 = [line.strip() for line in files_s24c2]
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
        weighted_data[decay].update({tagger: [f.replace('1_raw', f'4_weighted').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

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
rule all:
    input:
        '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/config_test/pdf_ratio/training/model.pth',
        

        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.001_bs4096_nL2_nN4/pdf_ratio/training/model.pth',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.001_bs32768_nL2_nN4/pdf_ratio/training/model.pth',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs4096_nL2_nN4/pdf_ratio/training/model.pth',
        # '/ceph/users/togasa/FlavourTagging/NTuples/Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr0.01_bs32768_nL2_nN4/pdf_ratio/training/model.pth',
        

        # expand(join(out, 'Data/savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/12/lr{lr}_bs{bs}_nL{nL}_nN{nN}/{weight}/training/model.pth'),
        #        lr=intervals['learning_rate'], bs=intervals['train_batch_size'], nL=intervals['numlayers'], nN=intervals['numneurons'], weight=weights),




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
    output:
        join(repo, "best_tagger_candidates/{cut_name}/{data_type, (MC|Data)}/candidatedTaggers_logit.json")
    log: 
        join(repo, "best_tagger_candidates/{cut_name}/{data_type, (MC|Data)}/candidatedTaggers_logit.log")
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
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        #script = join(repo, 'scripts/adding_features.py'), # Needed for Bs2JpsiPhi Bd2DmPi
        raw = lambda wildcards: get_raw_paths(wildcards.decay, wildcards.id, wildcards.data_type)
    log: 
        join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output: 
        root =join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'), 
    resources:
        mem_mb = 30_000, 
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
        mem_mb = 30_000, # Specify memory requirement in megabytes
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
        script = join(repo, 'scripts/mass_fits.py'),
        selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
    output:
        mc_res = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/mc_fit/mc_res.json'),
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

def copy_to_scratch(paths):
    scratch_paths = []

    # check whether ceph-kernel is mounted. If yes, copy from there
    to_replace = 'ceph/users'
    if os.path.exists(paths[0].replace("ceph", "ceph-kernel")):
        paths = [path.replace("ceph", "ceph-kernel") for path in paths]
        to_replace = 'ceph-kernel/users'


    for path in paths:
        path_scratch = path.replace(to_replace, "scratch")
        #make sure path on scratch exists or is created
        shell(f'mkdir -p {os.path.dirname(path_scratch)}')
        #Copy data from ceph to scratch
        shell(f'cp {path} {path_scratch}')
        scratch_paths.append(path_scratch)
    return scratch_paths

rule data_Mass_Fit: 
    input:
        script = join(repo, 'scripts/mass_fits.py'),
        mc_res = join(out, 'Data/{sample_type}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/mc_fit/mc_res.json'),
        mc_selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_selected_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] if not f.endswith('4_1.mc.root')
        ],
        data_selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in selected_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
    output:
        # join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/model.dll'),
        data_res = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/data_res.json'),
        # sweights = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/sweights.root'),
    log:
        join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/data_res.log'),
    resources:
        mem_mb = 10_000, 
        MaxRunHours = 6,
        request_disk = 256_000,
    params:
        tagged_prePath = join(out, 'Data/withUT_MC_2024/5_weighted/'),
        taggers = lambda wildcards: ' '.join(taggers_conf[wildcards.decay]),
        # tree = lambda wildcards: find_tree_name(wildcards.decay),
    run:
        out_path = os.path.dirname(os.path.dirname(output.data_res))


        selected_scratch = copy_to_scratch(input.data_selected)


        cmd = [
            'python {input.script}',
            # '--data_files {input.data_selected}',
            '--data_files ', ' '.join(selected_scratch),
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

rule add_weights:
    input:
        script = join(repo, 'scripts/add_weights.py'),
        selected = join(out, 'Data/{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{features}/{id}.root'),
        mc_res = join(out, 'Data/{sample_type}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/mc_fit/mc_res.json'),
        model = join(out,  'Data/{sample_type}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/data_fit/data_res.json'),
    output:
        join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/plots/validate_sweights_{id}.png'),
        weighted = join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/{id}.root'),
    log:
        join(out, 'Data/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_weighted/{decay}/{tagger}/{cut_name}/{features}/{id}.log'),
    resources:
        mem_mb = 30_000, 
        MaxRunHours = 2, 
    run:
        out_path = os.path.dirname(output.weighted)

        cmd = [
            'python {input.script}',
            '--selected {input.selected}',
            '--data_fit_model {input.model}',
            '--treename "DecayTree;1"',
            '--sim_fit_model {input.mc_res}' ,
            '--out_path', out_path,
            '--decayType {wildcards.decay}',
            '--obs_name B_DTF_PV_Jpsi_MASS',
            '--range {lowerMass} {upperMass}',
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



rule combine_tagger: 
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

rule split_sample:
    input:
        script = join(repo, 'scripts/split_train_val_test.py'),
        to_split = lambda wildcards:  join(out, f'{wildcards.data_type}/{wildcards.sample_type}/{"3_selected" if wildcards.data_type == "MC" else "4_weighted"}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.id}.root'),
        config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,data_type=wildcards.data_type).get("config")}.yaml'),
    output:
        train      = join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/5_split/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/train/{id}.root'),
        validation = join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/5_split/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/validation/{id}.root'),
        test       = join(out, '{data_type, (MC|Data)}/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/5_split/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/test/{id}.root'),
    log:
        join(out, '{data_type}/{sample_type}/5_split/{decay}/{tagger}/{cut_name}/{features}/log/.{id}.log'),
    resources:
        mem_mb = 15_000,
        MaxRunHours = 3,
    run:
        out_path = os.path.dirname(os.path.dirname(output.train))

        cmd = [
            'python', input.script,
            '--weighted {input.to_split}',
            '--target_path', out_path,
            '--config {input.config}',
            '--decayType {wildcards.decay}',
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--data_type {wildcards.data_type}',
            '&> {log}',
        ]
        shell(' '.join(cmd))








# rule train_tagger_MC:
#     input:
#         train = lambda wildcards: [
#             f.replace('cutName', f'{wildcards.cut_name}')
#             for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
#             if not f.endswith('4_1.mc.root')
#         ],
#         val = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'validation')
#             for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
#             if not f.endswith('4_1.mc.root')
#         ],

#         script = join(repo, 'scripts/train_tagger.py'),
#         config = join(repo, 'configs/{config}.yaml'),
#     output:
#         ROC=         join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training/ROC_TRAIN_VAL.pdf'),
#         model=       join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),
#         scaler=      join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training/st_scaler.pkl'),
#         transformer= join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training/powerTransformer.pkl'),
#         # taggingInfo= join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/logit/taggingInfo_logit.json'),
#         #   |-> created in pytrain.calibration
#     log: 
#         join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training/training_log.log')
#     resources:
#         mem_mb = 20_000, # Specify memory requirement in megabytes 
#         #gpus = 1,
#         OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
#         MaxRunHours = 24, # long queue
#         request_disk = 256_000
#     run:
#         outpath = os.path.dirname(output.model)

#         train_scratch = copy_to_scratch(input.train)
#         val_scratch = copy_to_scratch(input.val)




#         cmd = [
#             'python', input.script,
#             # '--training_data {input.train}',
#             # '--validation_data {input.val}',
#             '--training_data', ' '.join(train_scratch),
#             ' --validation_data', ' '.join(val_scratch),
#             ' --target_path', outpath,
#             '--tagger {wildcards.tagger}',
#             '--seed {wildcards.seed}',
#             '--features {wildcards.features}',
#             '--config {input.config}',
#             '--decay_type {wildcards.decay}',
#             '--data_type MC',
#             '--repo', repo,
#             #'--clean',
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))

# rule train_tagger_data:
#     input:
#         join(repo, 'scripts/NNModel.py'),
#         join(repo, 'scripts/pyTorchTraining.py'),

#         train = lambda wildcards: [
#             f.replace('cutName', f'{wildcards.cut_name}')
#             for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
#         ],
#         val = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'validation')
#             for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
#         ],

#         script = join(repo, 'scripts/train_tagger.py'),
#         config = join(repo, 'configs/{config}.yaml'),
#     output:
#         ROC=         join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/training/ROC_TRAIN_VAL.pdf'),
#         model=       join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/training/model.pth'),
#         scaler=      join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/training/st_scaler.pkl'),
#         transformer= join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/training/powerTransformer.pkl'),
#         # taggingInfo= join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/logit/taggingInfo_logit.json'),
#         #   |-> created in pytrain.calibration
#     log: 
#         join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/training/training_log.log')
#     resources:
#         mem_mb = 30_000, # Specify memory requirement in megabytes 
#         #gpus = 1,
#         OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
#         MaxRunHours = 8, # long queue
#         machine="heemskerck.e5.physik.tu-dortmund.de",
#     run:
#         outpath = os.path.dirname(output.model)
        
#         train_scratch = copy_to_scratch(input.train)
#         val_scratch = copy_to_scratch(input.val)



#         cmd = [
#             'python', input.script,
#             # '--training_data {input.train}',
#             # '--validation_data {input.val}',
#             '--training_data', ' '.join(train_scratch),
#             '--validation_data', ' '.join(val_scratch),
#             '--target_path', outpath,
#             '--tagger {wildcards.tagger}',
#             '--seed {wildcards.seed}',
#             '--features {wildcards.features}',
#             '--config', input.config,
#             '--decay_type {wildcards.decay}',
#             '--weight_type {wildcards.weight_type}',
#             '--repo', repo,
#             '--data_type Data',
#             '--num_threads 1',
#             #'--clean',
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
    # logs['training_logs'] = list(get_chunk(data_type + '/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', f'/{weight}training/training_log.log'))
    logs = get_chunk(data_type + '/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', f'/{weight}training/training_log_test.log')
    if batched:
        logs = list(logs)
        if data_type == 'MC':
            weight_name = 'MC'
        else:
            weight_name = '{weight_type}'
        # logs['chunk_log'] =[join(out, data_type + '/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/.' + weight + '_logs/lr{learning_rate}_training_chunk.log')]
        logs = logs + [join(out, data_type + '/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/chunk_logs/' + weight_name + '_lr{learning_rate}_bs{batch_size}_training_chunk.log')]
    return logs

rule train_tagger_MC:
    #If the batched flag from the config file is set to True, this rule trains a chunk of hyperparameters, if False it trains only one hyperparameter configuration 
    input:
        script = join(repo, 'scripts/batch_train_tagger.py') if batched else join(repo, 'scripts/train_tagger.py'),
        train = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
        val = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'validation')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
            if not f.endswith('4_1.mc.root')
        ],
        
        config = [join(repo, 'configs/lr{learning_rate}_bs{batch_size}_' + f'{remaining_conf}.yaml') for remaining_conf in hyper_par_chunk] if batched
                 else join(repo, 'configs/{config}.yaml'),
    output:
        # ROC=         get_chunk('MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/ROC_TRAIN_VAL.pdf'),
        model=       get_chunk('MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/model.pth'),
        scaler=      get_chunk('MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/st_scaler.pkl'),
        transformer= get_chunk('MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/powerTransformer.pkl'),
    params:
        pre_path = join(out, 'MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}')
    log:
        get_log('MC'),
        # get_chunk('MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/training_log.log'),
        # chunk_log = join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/training_chunk.log')
        #      if batched else None,
    resources:
        mem_mb = 30_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
    threads:
        len(hyper_par_chunk)+1 if batched else 1,
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
            '--repo', repo,
            '--data_type MC',
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
                '&> {log}',
            ]
        print(cmd)
        cmd = cmd + conditional_cmd
        print(cmd)
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
        # ROC=         get_chunk('Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/{weight_type,(signal_weights|pdf_ratio|ones)}/training/ROC_TRAIN_VAL.pdf'),
        model=       get_chunk('Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/{weight_type,(signal_weights|pdf_ratio|ones)}/training/model.pth'),
        scaler=      get_chunk('Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/{weight_type,(signal_weights|pdf_ratio|ones)}/training/st_scaler.pkl'),
        transformer= get_chunk('Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/{weight_type,(signal_weights|pdf_ratio|ones)}/training/powerTransformer.pkl'),
    params:
        pre_path = join(out, 'Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}')
    log:
        get_log('Data'),
        # get_chunk('Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/', '/training/training_log.log'),
        # chunk_log = join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/training_chunk.log')
        #     if batched else None,
    resources:
        mem_mb = 65_000 if batched else 15_000, # Specify memory requirement in megabytes 
        #gpus = 1,
        #OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 100, # long queue
    threads:
        len(hyper_par_chunk)+2     if batched else 1,
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
                '&> {log}',
            ]

        print('cmd:', cmd)
        cmd = cmd + conditional_cmd
        shell(' '.join(cmd))

rule test_and_calibrate_tagger_MC:
    input:
        testing = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'test')
            for f in ntuples_train_split_withUT_mc[f'{wildcards.decay}'][f'{wildcards.tagger}']
            if not f.endswith('4_1.mc.root')
        ],
        model = join(out, 'MC/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        ROC = join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/testing/ROC_TEST.pdf'),
        logit = join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/testing/logit/taggingInfo_logit.json'),
        mistag = join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/testing/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, 'MC/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/testing/testing_log.log')

    resources:
        mem_mb = 20_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 6,
        request_disk = 256_000
    run:
        outpath = os.path.dirname(output.ROC)
        model_path = os.path.dirname(input.model)

        test_scratch = copy_to_scratch(input.testing)

        cmd = [
            'python', input.script,
            # '--testing_data {input.testing}',
            '--testing_data', ' '.join(test_scratch),
            '--target_path', outpath,
            '--train_path', outpath.replace('testing', 'training'),
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed {wildcards.seed}',
            '--repo', repo,
            '--data_type MC',
            '--model_path', model_path,
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule test_and_calibrate_tagger_data:
    input:
        testing = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}').replace('train', 'test')
            for f in train_split_data[f'{wildcards.decay}'][f'{wildcards.tagger}']
        ],
        model = join(out, 'Data/savedModels/{sample_type}/{decay}/{tagger}/{cut_name}/{features}/{seed}/{config}/{weight_type}/training/model.pth'),

        script = join(repo, 'scripts/test_and_calibrate.py'),
        config = join(repo, 'configs/{config}.yaml'),
    output:
        ROC = join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/testing/ROC_TEST.pdf'),
        logit = join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/testing/logit/taggingInfo_logit.json'),
        mistag = join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/testing/mistag/taggingInfo_mistag.json'),
    log: 
        join(out, 'Data/savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/{weight_type,(signal_weights|pdf_ratio|ones)}/testing/testing_log.log')
    resources:
        mem_mb = 35_000, # Specify memory requirement in megabytes 
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 4,
        # request_disk = 256_000
    run:
        outpath = os.path.dirname(output.ROC)
        model_path = os.path.dirname(input.model)

        test_scratch = copy_to_scratch(input.testing)

        cmd = [
            'python', input.script,
            # '--testing_data {input.testing}',
            '--testing_data', ' '.join(test_scratch),
            '--target_path', outpath,
            '--train_path', outpath.replace('testing', 'training'),
            '--treename "DecayTree;1"',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--config', input.config,
            '--decay_type {wildcards.decay}',
            '--seed {wildcards.seed}',
            '--repo', repo,
            '--data_type Data',
            '--weight_type {wildcards.weight_type}',
            '--model_path', model_path,
            '&> {log}',
        ]
        shell(' '.join(cmd))
