from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import os
from copy import deepcopy

import json

try:
    dataIn = config['DATAIN']
    dataOut = config['DATAOUT']

    
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
    'Bu2JpsiK': in_data(dataIn, '''
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

    'Bd2JpsiKst': in_data(dataIn, '''
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
    'Bd2DmPi': in_data(dataIn, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000003_1.mc.root
'''.split('\n')),
    'Bs2DsPi': in_data(dataIn, '''
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
    'Bs2JpsiPhi': in_data(dataIn, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226269/0000/00226269_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226271/0000/00226271_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226273/0000/00226273_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/Dev/MC.ROOT/00226275/0000/00226275_00000003_1.mc.root
'''.split('\n'))
}

# TO DO add the other configurations e.g. noUT, openVELO etc
ntuples_raw_withUT = deepcopy(ntuples_eos_withUT)
for k,v in ntuples_raw_withUT.items():
    ntuples_raw_withUT[k] = [f.replace(os.path.dirname(f), f'{dataOut}/withUT_MC_2024/1_raw/{k}') for f in v]

ntuples_added_features_withUT = {}
for k,v in ntuples_raw_withUT.items():
    ntuples_added_features_withUT.update({k: [f.replace('1_raw', '2_added_features') for f in v]})

ntuples_selected_withUT = {}
for decay, path_list in ntuples_raw_withUT.items():
    ntuples_selected_withUT.update({decay: {}})
    #print(decay, path_list)
    #print('-------')
    for tagger in taggers_conf[decay] :
        ntuples_selected_withUT[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN') for f in path_list]})

ntuples_tagged_withUT = {}
for decay, path_list in ntuples_raw_withUT.items():
    ntuples_tagged_withUT.update({decay: {}})
    for tagger in taggers_conf[decay]:
        # Filter the paths to only include those ending with '4_1.mc.root'
        filtered_paths = [
            f.replace('1_raw', '4_tagged')
             .replace(decay, f'{decay}/{tagger}/notSamePV_noOSP/union_PROBNN')
            for f in path_list if f.endswith('4_1.mc.root')
        ]
        ntuples_tagged_withUT[decay].update({tagger: filtered_paths})

    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN_IPSig') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{taggers_conf[decay][i]}/{k}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/only_OSKaon') for f in v]})

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
generated_paths_SSPion = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_SSPion.txt'))
generated_paths_SSKaon = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_SSKaon.txt'))
generated_paths_SSProton = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_SSProton.txt'))
generated_paths_OSKaon = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_OSKaon.txt'))
generated_paths_OSElectron = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_OSElectron.txt'))
generated_paths_OSMuon = read_generated_paths(dataOut,join(repo,'paths_for_snakemake/generated_paths_OSMuon.txt'))


rule all:
    input:
        ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'],
        ntuples_tagged_withUT['Bd2JpsiKst']['OSElectron'],
        ntuples_tagged_withUT['Bd2JpsiKst']['OSMuon'],
        ntuples_tagged_withUT['Bd2JpsiKst']['SSProton'],
        ntuples_tagged_withUT['Bd2JpsiKst']['SSPion'],

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
        #ntuples_selected_withUT['Bd2JpsiKst']['OSKaon'],
        #ntuples_selected_withUT['Bd2JpsiKst']['OSElectron'],
        #ntuples_selected_withUT['Bd2JpsiKst']['OSMuon'],
        #ntuples_selected_withUT['Bd2JpsiKst']['SSPion'],
        #ntuples_selected_withUT['Bd2JpsiKst']['SSProton'],
        #join(data, 'savedModels/withUT_MC_2024/combinations/Bd2JpsiKst/all_Calibration.pdf'),


        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),


        #ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'],

        #'/ceph/users/qfuehring/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/balanced/treeSchema.pdf',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/unbalanced/treeSchema.pdf',

       # ntuples_added_features_withUT['Bu2JpsiK'],
       # ntuples_added_features_withUT['Bd2JpsiKst'],

        #ntuples_tagged_withUT['Bs2JpsiPhi']['OSMuon'],
        #ntuples_tagged_withUT['Bs2JpsiPhi']['OSKaon'],
        #ntuples_tagged_withUT['Bs2JpsiPhi']['OSElectron'],
        ##ntuples_tagged_withUT['Bs2JpsiPhi']['SSKaon'],
        #ntuples_tagged_withUT['Bd2DmPi']['OSKaon'],
        #ntuples_tagged_withUT['Bd2DmPi']['OSElectron'],
        #ntuples_tagged_withUT['Bd2DmPi']['SSPion'],
        #ntuples_tagged_withUT['Bd2DmPi']['OSMuon'],
        #ntuples_tagged_withUT['Bd2DmPi']['SSProton'],




       # ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],
       # ntuples_selected_withUT['Bu2JpsiK']['OSElectron'],


        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        #         ntuples_selected_withUT['Bd2JpsiKst']['SSProton'],

        #ntuples_added_features_withUT['Bs2DsPi'],
        #ntuples_added_features_withUT['Bu2JpsiK'],
        #ntuples_added_features_withUT['Bd2JpsiKst']


        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bd2JpsiKst/00237614_00000002_1.mc.root',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00237567_00000001_1.mc.root',
        #'/ceph/users/qfuehring/Data/withUT_MC_2024/2_added_features/Bu2JpsiK/00237568_00000001_1.mc.root'

        # generated_paths_OSKaon,
        # generated_paths_SSPion
        #generated_paths
        #ntuples_selected_withUT['Bs2DsPi']['SSKaon'],
        #ntuples_selected_withUT['Bd2JpsiKst'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSMuon'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSElectron'],
        #join(data, 'withUT_MC_2024/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf')
        #generated_paths_SSKaon,


decays_to_tag = ['Bu2JpsiK', 'Bd2JpsiKst']
all_configs = [f for f in os.listdir(join(repo, "configs/")) if f.startswith("lr")]


from scripts.replace_path import seeds

rule get_optimized:
    input:
        script = join(repo, 'scripts/getOptimized.py'),
        modelPath = join(dataOut, "savedModels/withUT_MC_2024"),
        

        tagging_infos = lambda wildcards : [
            join(dataOut, f"/savedModels/withUT_MC_2024/{decay}/{tagger}/{wildcards.cut_name}/union_PROBNN/{seed}/{config}/logit/taggingInfo_logit.json")
            for decay in decays_to_tag
            for tagger in taggers_conf[decay]  # Dynamically select taggers for each decay
            for seed in seeds
            for config in all_configs # configs definieren und so
]
    log: 
        join(repo, "best_tagger_candidates/{cut_name}/candidatedTaggers_logit.log")
    output:
        join(repo, "best_tagger_candidates/{cut_name}/candidatedTaggers_logit.json")
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

rule add_features:
# For some NTuples it's necessary to run locally (snakemake only, not on condor)
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        #script = join(repo, 'scripts/adding_features.py'), # Needed for Bs2JpsiPhi Bd2DmPi
        raw = join(dataIn, '{decay}/{id}.root')
    log: join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output: 
        root =join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'), 
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue
        #request_disk = 50000
    run:
        tree = find_tree_name(wildcards.decay)
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
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
        data = join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features'),
    output:
        pdf = join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.pdf'),
    log:
        join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/notSamePV_noOSP/{balanced}/treeSchema.log')
    params:
        target_path = lambda wildcards: join(dataOut, f'{wildcards.sample_type}/DT_outputs/notSamePV_noOSP')
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes
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
        added_features = join(dataOut, '{sample_type}/2_added_features/{decay}/{id}.root'),
    output: join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id,.*}.root'),
    # output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root'),
    log : join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes
        MaxRunHours = 4, # medium queue

    run:
       # tree = find_tree_name(wildcards.decay)
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            #'--treename', tree,
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--BKG0',
            '&> {log}',
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

def extract_best(tagger, cut,link='logit'):
    #Read the best tagger candidate config from json file with the best hyperparameter combination

    with open(join(repo, f'best_tagger_candidates/{cut}/candidatedTaggers_{link}.json'), 'r') as f:
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
        selected = join(dataOut, '{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{features}/{id}'),
        model=lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("config")}/model.pth'), 
        transformer=lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("config")}/powerTransformer.pkl'), 
        scaler=lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("seed")}/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("config")}/st_scaler.pkl'), 
        best_tagger = join(repo, "best_tagger_candidates/{cut_name}/candidatedTaggers_logit.json"),
        config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name).get("config")}.yaml'),
    output:
        root = join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id}'),
    log:
        join(dataOut, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{id}.log'),
    run:
        print(f"Processing file: {wildcards.id}")
        print(f"Selected input file: {input.selected}")
        config = extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name)
        cmd = [
            'python', input.script,
            '--selected {input.selected}', 
            '--taggedData {output.root}',  
            '--model {input.model}',
            '--scaler {input.scaler}',
            '--transformer {input.transformer}',
            '--config {input.config}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            f'--arch {config.get("arch")}',
            f'--lr {config.get("lr")}',
            f'--seed {config.get("seed")}',
            '&> {log}'
        ]
        shell(' '.join(cmd))



rule combine_tagger:
    input:
        tagged = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_tagged_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        script = join(repo, 'scripts/combineTagger.py'),
    output:
        pdf=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_Calibration.pdf'),
    log: join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_log.log')
    resources:
        #mem_mb = 20000, # Specify memory requirement in megabytes 
        ##gpus = 1,
        #OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        #MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
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
        join(repo, 'configs/configs/lr{lr,(0.001|0.01|0.1)}_bs{bs,(32|128|1024|2048)}_{a, {simple|complex}}_dm{dm, (0.0|0.0001|0.001|0.01)}.yaml'),
    log: 
        join(repo, 'configs/generate_configs{lr,(0.001|0.01|0.1)}_bs{bs,(32|128|1024|2048)}_{a, {simple|complex}}_dm{dm, (0.0|0.0001|0.001|0.01)}.log'),
    run:
        cmd = [
            'python', input.script,
            '&> {log}',
        ]
        shell(' '.join(cmd))
    

rule train_tagger:
    input:
        #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        selected = lambda wildcards: [
            f.replace('cutName', f'{wildcards.cut_name}') 
            for f in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}'] 
            if not f.endswith('4_1.mc.root')
        ],
        script = join(repo, 'scripts/pipeline.py'),
        config = lambda wildcards: join(repo, 'configs/{config}.yaml'),
    output:
        pdf=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/ROC_TRAIN_VAL.pdf'),
        model=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/model.pth'),
        scaler=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/st_scaler.pkl'),
        transformer=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/powerTransformer.pkl'),
        taggingInfo=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/logit/taggingInfo_logit.json'),
    log: 
        join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training_log.log')
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        target_path = lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
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
            #'--clean',
            '&> {log}',
        ]
        shell(' '.join(cmd))

# rule calibrate_tagger:
#     input:
#         join(repo,'best_tagger_candidates/{decay}/{tagger}/{cut_name}/{config}/candidatedTaggers_logit.json'), 
    
#         selected = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
#         script = join(repo, 'scripts/calibration.py'),
#         #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
#         #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
#     output:
#         pdf=join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{config}/Normalized_TagDec.pdf'),
        
#     log: 
#         join(dataOut, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{config}/calibration_log.log')
#     # resources:
#     #     mem_mb = 128000, # Specify memory requirement in megabytes 
#     #     #gpus = 1,
#     #     MaxRunHours = 4,
#     #     #request_disk = 1024000
#     params:
#         config = lambda wildcards: join(repo, f'configs/{extract_best(tagger=wildcards.tagger, cut=wildcards.cut_name,decay=wildcards.decay).get("config")}.yaml'),
#         target_path = lambda wildcards: join(dataOut, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/')
#     run:
#         cmd = [
#             'python', input.script,
#             '--target_path {params.target_path}',
#             '--tagger {wildcards.tagger}',
#             #'--seed {wildcards.seed}',
#             '--config {params.config}',
#             '--decayType {wildcards.decay}',
#             '&> {log}',
#         ]
#         shell(' '.join(cmd))
