from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import os
from copy import deepcopy
from snakemake.io import dynamic
from snakemake.io import ancient

try:
    # RAW_MC isn't used as the folder structure is different
    raw_MC = config['RAW_MC']
    modified_MC = config['MODIFIED_MC']
    repo = config['REPO']
except:
    raise RuntimeError("Make sure to specify snakemake config")


def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]


taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSKaon', 'SSProton'],
    'Bd2JpsiKst': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon', 'SSKaon'],
    'Bs2DsPi': ['SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2DmPi': ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon',],
    'Bs2JpsiPhi': ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSProton', 'SSKaon']


}
# TO DO create a rules that copy the files from eos to the cluster


# Mag Up only as in Run 3 mag up was mostly used
ntuples_eos_withUT = {
    'Bu2JpsiK': in_data(raw_MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000007_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000008_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000012_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000009_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000013_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000010_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000011_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000014_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266989/0000/00266989_00000015_1.mc.root
'''.split('\n')),

    'Bd2JpsiKst': in_data(raw_MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00267659/0000/00267659_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00267659/0000/00267659_00000002_1.mc.root
'''.split('\n')),
    'Bd2DmPi': in_data(raw_MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266995/0000/00266995_00000001_1.mc.root
r       oot://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266995/0000/00266995_00000002_1.mc.root
'''.split('\n')),
    'Bs2DsPi': in_data(raw_MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266991/0000/00266991_00000006_1.mc.root
'''.split('\n')),
    'Bs2JpsiPhi': in_data(raw_MC, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266987/0000/00266987_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266987/0000/00266987_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266987/0000/00266987_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266987/0000/00266987_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00266987/0000/00266987_00000005_1.mc.root
'''.split('\n'))
}

# TO DO add the other configurations e.g. noUT, openVELO etc
ntuples_raw_withUT = deepcopy(ntuples_eos_withUT)
for k,v in ntuples_raw_withUT.items():
    ntuples_raw_withUT[k] = [f.replace(os.path.dirname(f), f'{raw_MC}/withUT_MC_2024/1_raw/{k}') for f in v]

ntuples_added_features_withUT = {}
for k, v in ntuples_raw_withUT.items():
    ntuples_added_features_withUT.update({k: [f.replace(
        f'{raw_MC}', f'{modified_MC}').replace('1_raw', '2_added_features') for f in v]})
    # ntuples_added_features_withUT.update({k: [f.replace(f'{raw_MC}', f'{modified_MC}/withUT_MC_2024/2_added_features') for f in v]})

ntuples_selected_withUT = {}
for decay, path_list in ntuples_raw_withUT.items():
    ntuples_selected_withUT.update({decay: {}})
    # print(decay, path_list)
    # print('-------')
    for tagger in taggers_conf[decay]:
        # ntuples_selected_withUT[decay].update({tagger: [f.replace(f'{raw_MC}', f'{modified_MC}/withUT_MC_2024/3_selected').replace(decay, f'{decay}/{tagger}/{{cut_name}}/{{balanced}}/{{features}}') for f in path_list]})
        ntuples_selected_withUT[decay].update({tagger: [f.replace(f'{raw_MC}', f'{modified_MC}').replace(
            f'1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/{{cut_name}}/{{balanced}}/{{features}}') for f in path_list]})

ntuples_tagged_withUT = {}
for decay, path_list in ntuples_raw_withUT.items():
    ntuples_tagged_withUT.update({decay: {}})
    for tagger in taggers_conf[decay]:
        # Filter the paths to only include those ending with '4_1.mc.root'
        filtered_paths = [
            f.replace(f'{raw_MC}', f'{modified_MC}').replace(
                '1_raw', '4_tagged')
            .replace(decay, f'{decay}/{tagger}/{{cut_name}}/{{balanced}}/{{features}}/{{asym_level}}')
            for f in path_list
            #for f in path_list if f.endswith('01_1.mc.root') #4_1.mc.root hold out sample, actually it depends on the number of files
        ]
        ntuples_tagged_withUT[decay].update({tagger: filtered_paths})

    # ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN_IPSig') for f in v]})
    # ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{taggers_conf[decay][i]}/{k}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in v]})
    # ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/only_OSKaon') for f in v]})

# Function to read paths from the generated file


def read_generated_paths(file):
    with open(file, 'r') as f:
        paths = [line.strip() for line in f]
    return paths


def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    if decay == 'Bs2DsPi':
        # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
        return 'BdToDsmPi_DsmToKpKmPim/DecayTree'
    if decay == 'Bu2JpsiK':
        return 'BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree'
    if decay == 'Bd2JpsiKst':
        return 'BdToJpsiKstar_JpsiToMuMu_Detached/DecayTree'
    else:
        return 'Tuple/DecayTree'


# Read the generated paths
generated_paths_SSPion = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_SSPion.txt'))
generated_paths_SSKaon = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_SSKaon.txt'))
generated_paths_SSProton = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_SSProton.txt'))
generated_paths_OSKaon = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_OSKaon.txt'))
generated_paths_OSElectron = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_OSElectron.txt'))
generated_paths_OSMuon = read_generated_paths(
    join(repo, 'paths_for_snakemake/generated_paths_OSMuon.txt'))

# print(expand(ntuples_selected_withUT['Bd2JpsiKst']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK_balanced'], balanced=['balanced'],features=['union_PROBNN']),)

# print(expand(join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf'))
rule all:
    input:
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.01_bs128_nL2_nN8/asym_level1/ROC_TRAIN_VAL.pdf'),
        generated_paths_OSKaon,
        generated_paths_SSPion,
        generated_paths_SSProton,
        generated_paths_OSElectron,
        generated_paths_OSMuon,
        generated_paths_SSKaon

        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.01_bs4096_nL2_nN64/asym_level1/ROC_TRAIN_VAL.pdf'), #$ Using seed like Thomas
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.001_bs8192_nL2_nN64/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.01_bs2048_nL2_nN128/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.01_bs4096_nL2_nN64/asym_level1/ROC_TRAIN_VAL.pdf'), # like OSKaon
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.01_bs4096_nL2_nN64/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/12/lr0.01_bs4096_nL2_nN64/asym_level1/ROC_TRAIN_VAL.pdf'),

        #expand(ntuples_tagged_withUT['Bd2JpsiKst']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bd2JpsiKst']['SSProton'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bs2DsPi']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bs2DsPi']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bs2DsPi']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),
        #expand(ntuples_tagged_withUT['Bs2DsPi']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'], features=['union_PROBNN'], asym_level=['asym_level1']),

        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/input_plot/training_inputFeatures.pdf'),

        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),
        #join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level1/ROC_TRAIN_VAL.pdf'),

        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level0/ROC_TRAIN_VAL.pdf'),
#
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/asym_level2/ROC_TRAIN_VAL.pdf'),

        # ntuples_added_features_withUT['Bu2JpsiK'],
        # ntuples_added_features_withUT['Bd2JpsiKst'],
        # ntuples_added_features_withUT['Bs2DsPi']

        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSProton'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['SSProton'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bd2JpsiKst']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),

        # ntuples_added_features_withUT['Bu2JpsiK'],
        # ntuples_added_features_withUT['Bd2JpsiKst'],
        # ntuples_added_features_withUT['Bs2DsPi']

        # '/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/DT_outputs/final_cut/unbalanced/treeSchema.pdf',
       # '/ceph/users/molocco/FlavourTagging/MC/withUT_MC_2024/DT_outputs/final_cut/balanced/treeSchema.pdf',

        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bu2JpsiK']['SSProton'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),

        # ntuples_added_features_withUT['Bs2DsPi'],
        # expand(ntuples_tagged_withUT['Bs2DsPi']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bs2DsPi']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bs2DsPi']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bs2DsPi']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_tagged_withUT['Bs2DsPi']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),

        # ntuples_tagged_withUT['Bs2DsPi']['OSKaon'],
        # ntuples_tagged_withUT['Bs2DsPi']['OSElectron'],
        # ntuples_tagged_withUT['Bs2DsPi']['OSMuon'],

        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(modified_MC, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf'),

        # expand(ntuples_selected_withUT['Bd2JpsiKst']['SSPion'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_selected_withUT['Bd2JpsiKst']['SSProton'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_selected_withUT['Bu2JpsiK']['OSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_selected_withUT['Bu2JpsiK']['OSMuon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_selected_withUT['Bu2JpsiK']['OSElectron'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(ntuples_selected_withUT['Bs2DsPi']['SSKaon'], cut_name=['allBKGCAT_notSamePV_noOSP_SSK'], balanced=['balanced'],features=['union_PROBNN']),
        # expand(join(modified_MC, 'withUT_MC_2024/DT_outputs/allBKGCAT_notSamePV_noOSP_SSK_balanced/balanced/cuts/OSKaon_preselections.txt'))
        # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024//Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs2048_simple_dm0.0/ROC_TRAIN_VAL.pdf',
        # ntuples_added_features_withUT['Bu2JpsiK'],
       # ntuples_added_features_withUT['Bd2JpsiKst'],
        # join(modified_MC, 'withUT_MC_2024/DT_outputs/allBKGCAT_notSamePV_noOSP_SSK_balanced/balanced/tree_schema.pdf')
        # ntuples_added_features_withUT['Bs2DsPi'],
        # ntuples_added_features_withUT['Bu2JpsiK'],
        # ntuples_added_features_withUT['Bd2JpsiKst'],
        # ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'],
        # ntuples_tagged_withUT['Bd2JpsiKst']['OSElectron'],
        # ntuples_tagged_withUT['Bd2JpsiKst']['OSMuon'],
        # ntuples_tagged_withUT['Bd2JpsiKst']['SSProton'],
        # ntuples_tagged_withUT['Bd2JpsiKst']['SSPion'],


        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf'),
        # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024//Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs2048_simple_dm0.0/ROC_TRAIN_VAL.pdf',
        # '/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024//Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple_dm0.0/ROC_TRAIN_VAL.pdf',
        # ntuples_selected_withUT['Bd2JpsiKst']['OSKaon'],
        # ntuples_selected_withUT['Bd2JpsiKst']['OSElectron'],
        # ntuples_selected_withUT['Bd2JpsiKst']['OSMuon'],
        # ntuples_selected_withUT['Bd2JpsiKst']['SSPion'],
        # ntuples_selected_withUT['Bd2JpsiKst']['SSProton'],
        # join(data, 'savedModels/withUT_MC_2024/combinations/Bd2JpsiKst/all_Calibration.pdf'),

        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/notSamePV_noOSP/union_PROBNN/14/lr0.001_bs1024_simple/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/notSamePV_noOSP/union_PROBNN/45/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple/ROC_TRAIN_VAL.pdf'),

        # ntuples_tagged_withUT['Bd2JpsiKst']['OSKaon'],

        # '/ceph/users/molocco/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/balanced/treeSchema.pdf',
        # '/ceph/users/molocco/Data/withUT_MC_2024/DT_outputs/notSamePV_noOSP/unbalanced/treeSchema.pdf',

       # ntuples_added_features_withUT['Bu2JpsiK'],
       # ntuples_added_features_withUT['Bd2JpsiKst'],

        # ntuples_tagged_withUT['Bs2JpsiPhi']['OSMuon'],
        # ntuples_tagged_withUT['Bs2JpsiPhi']['OSKaon'],
        # ntuples_tagged_withUT['Bs2JpsiPhi']['OSElectron'],
        # ntuples_tagged_withUT['Bs2JpsiPhi']['SSKaon'],
        # ntuples_tagged_withUT['Bd2DmPi']['OSKaon'],
        # ntuples_tagged_withUT['Bd2DmPi']['OSElectron'],
        # ntuples_tagged_withUT['Bd2DmPi']['SSPion'],
        # ntuples_tagged_withUT['Bd2DmPi']['OSMuon'],
        # ntuples_tagged_withUT['Bd2DmPi']['SSProton'],

       # ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],

        # generated_paths_OSKaon,
        # generated_paths_SSPion
        # generated_paths
        # ntuples_selected_withUT['Bs2DsPi']['SSKaon'],
        # ntuples_selected_withUT['Bd2JpsiKst'],
        # ntuples_selected_withUT['Bu2JpsiK']['OSMuon'],
        # ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],
        # ntuples_selected_withUT['Bu2JpsiK']['OSElectron'],
        # join(data, 'withUT_MC_2024/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf')
        # generated_paths_SSKaon,

'''
rule add_features:
    # For some NTuples it's necessary to run locally (snakemake only, not on condor)
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        # script = join(repo, 'scripts/adding_features.py'), # Needed for Bs2JpsiPhi Bd2DmPi
        # raw = join(data, '{sample_type}/1_raw/{decay}/{id}.root')
        raw = join(raw_MC, '{sample_type}/1_raw/{decay}/{id}.root')
    log: join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output:
        root = join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'),
        # request_disk = 50000
    resources:
        mem_mb = 20000,  # Specify memory requirement in megabytes
        MaxRunHours = 4,  # medium queue

    run:
        tree = find_tree_name(wildcards.decay)
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output.root}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
            '&> {log}',
        ]
        shell(' '.join(cmd))
'''

all_taggers = sorted({t for taggers in taggers_conf.values() for t in taggers})

'''
rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        data = join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features'),
    output:
        #pdf = join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/{spec}/{balanced}/tree_schema.pdf'),
        #cuts = join(modified_MC, "{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/{cut_name}/balanced/cuts/{tagger}_preselections.txt")
        cuts = expand(join(modified_MC, "{{sample_type,(withUT_MC_2024|noUT_MC_2024)}}/DT_outputs/{{cut_name}}/{{balanced}}/cuts/{tagger}_preselections.txt"), tagger=all_taggers)
    log:
        join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/{cut_name}/{balanced, (balanced|unbalanced)}/tree_schema.log')
    params:
        target_path = lambda wildcards: join(modified_MC, f'{wildcards.sample_type}/DT_outputs/{wildcards.cut_name}/{wildcards.balanced}/')
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
       # MaxRunHours = 8, # medium queue
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
'''
'''
rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        added_features = join(modified_MC, '{sample_type}/2_added_features/{decay}/{id}.root'),
        # Cut file is in the MC path
        cut_file = join(modified_MC, 'withUT_MC_2024/DT_outputs/{cut_name}/balanced/cuts/{tagger}_preselections.txt')

    output: join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{balanced}/{features}/{id,.*}.root'),
    # output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root'),
    log: join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{balanced}/{features}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    resources:
        mem_mb = 20000,  # Specify memory requirement in megabytes
        MaxRunHours = 4,  # medium queue

    run:
       # tree = find_tree_name(wildcards.decay)
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            # '--treename', tree,
            '--cut_file', input.cut_file,
            # join(modified_MC, '{wildcards.sample_type}/DT_outputs/{wildcards.cut_name}/balanced/cuts/{wildcards.decay}/{tagger}_preselections.txt')
            # '--cut_file', join(modified_MC, '{wildcards.sample_type}/DT_outputs/{wildcards.cut_name}/balanced/cuts/{wildcards.decay}/{tagger}_preselections.txt'),
            '--evtType {wildcards.decay}',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--BKG0',
            '--repo {repo}',
            '&> {log}',
        ]
        shell(' '.join(cmd))
'''

rule train_tagger:
    input:
        # selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        # selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        selected = lambda wildcards: [
            f.replace('{cut_name}', wildcards.cut_name)
             .replace('{balanced}', wildcards.balanced)
             .replace('{features}', wildcards.features)
            for f in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']
            # if not f.endswith('4_1.mc.root')
        ],
        script = ancient(join(repo, 'scripts/pipeline.py')),
    output:
        pdf = join(modified_MC, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}_{balanced}/{features}/{seed}/{config}/{asymmetry_level}/model.pth'),
        #pdf=join(modified_MC, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}_{balanced}/{features}/{seed}/{config}/{asymmetry_level}/training_inputFeatures.pdf'),
        # Replaced with this to profuce only input features plot
    log: join(modified_MC, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}_{balanced}/{features}/{seed}/{config}/{asymmetry_level}/training_log.log')
    resources:
        mem_mb = 80000,  # Specify memory requirement in megabytes
        # gpus = 1,
        # Allow exit code 1 for debugging
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",
        MaxRunHours = 8,  # medium queue
        request_disk = 1024000
    params:
        #script = join(repo, 'scripts/pipeline.py'),
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(modified_MC, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}_{wildcards.balanced}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/{wildcards.asymmetry_level}')
    run:
        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--target_path {params.target_path}',
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--config {params.config}',
            '--decayType {wildcards.decay}',
            '--asymmetry_level {wildcards.asymmetry_level}',
            # '--clean',
            '--repo {repo}',
            #'--only_plot',
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


# TO BE CHECKED
rule add_tagDec:
    input:
        script = join(repo, 'scripts/adding_tagDec.py'),
        selected = join(modified_MC, '{sample_type}/3_selected/{decay}/{tagger}/{cut_name}/{balanced}/{features}/{id}.root')
    output:
        root = join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{balanced}/{features}/{asym_level}/{id}.root'),
    log:
        join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{balanced}/{features}/{asym_level}/{id}.log'),
    params:
        modelPrePath = lambda wildcards: join(modified_MC, f'savedModels/{wildcards.sample_type}/{extract_decay(wildcards.tagger)}/{wildcards.tagger}/{wildcards.cut_name}_{wildcards.balanced}/{wildcards.features}/'),
        taggedDataPath = join(modified_MC, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{balanced}/{features}/{asym_level}/'),
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes

    run:
        print(f"Processing file: {wildcards.id}")
        print(f"Selected input file: {input.selected}")

        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--cut {wildcards.cut_name}_{wildcards.balanced}',  # Cut name 
            '--taggedData {output.root}',  
            '--modelPrePath {params.modelPrePath}',
            '--decayType {wildcards.decay}', # Decay used for evaluating the tagger
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--asymmetry_level {wildcards.asym_level}',
            '--repo', repo,
            '&> {log}'
        ]
        shell(' '.join(cmd))


rule combine_tagger:
    input:
        tagged = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_tagged_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        script = join(repo, 'scripts/combineTagger.py'),
    output:
        pdf = join(modified_MC, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_Calibration.pdf'),
    log: join(modified_MC, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/combinations/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{combinationName}_log.log')
    resources:
        # mem_mb = 20000, # Specify memory requirement in megabytes
        # gpus = 1,
        # OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        # MaxRunHours = 24, # long queue
        # request_disk = 1024000
    params:
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(modified_MC, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
    run:
        cmd = [
            'python', input.script,
            '--tagger '
            '--tagged {input.tagged}',
            '--target_path {params.target_path}',
            '--combinationName {wildcards.combinationName}',
            '--decayType {wildcards.decay}',
            # '--repo', repo,
            '&> {log}',
        ]
        shell(' '.join(cmd))


#  rule calibrate_tagger:
#     input:
#         #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
#         #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
#         selected = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
#         script = join(repo, 'scripts/calibration.py'),
#     output:
#         pdf=join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{seed}/{config}/Normalized_TagDec.pdf'),
#     log: join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{seed}/{config}/calibration_log.log')
#     # resources:
#     #     mem_mb = 128000, # Specify memory requirement in megabytes
#     #     #gpus = 1,
#     #     MaxRunHours = 4,
#     #     #request_disk = 1024000
#     params:
#         config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
#         target_path = lambda wildcards: join(data, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.seed}/{wildcards.config}/')
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
