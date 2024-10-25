from os.path import join, exists, dirname, basename, splitext, split
from os import makedirs
import numpy as np
import os
from copy import deepcopy

try:
    data = config['DATA']
    repo = config['REPO']
except:
    raise RuntimeError("Make sure to specify snakemake config")

def in_data(data_path, list_of_files):
    return [join(data_path, i) for i in list_of_files if '#' not in i and len(i) > 0]

taggers_conf = {
    'Bu2JpsiK': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bu2JpsiK_new': ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst': ['SSPion', 'SSProton'],
    'Bs2DsPi': ['SSKaon'],
    'Bd2DmPi': ['SSPion', 'SSProton'],
    'Bs2JpsiPhi': ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSProton', 'SSKaon']

    
}
# TO DO create a rules that copy the files from eos to the cluster
ntuples_eos_withUT = {
    'Bu2JpsiK': in_data(data, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214053/0000/00214053_00000007_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214051/0000/00214051_00000007_1.mc.root
'''.split('\n')),

    'Bu2JpsiK_new': in_data(data, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237567_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237568_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237568_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237568_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00237567/0000/00237568_00000004_1.mc.root
'''.split('\n')),

    'Bd2JpsiKst': in_data(data, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214049/0000/00214049_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214049/0000/00214049_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214049/0000/00214049_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214049/0000/00214049_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214049/0000/00214049_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214047/0000/00214047_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214047/0000/00214047_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214047/0000/00214047_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214047/0000/00214047_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214047/0000/00214047_00000005_1.mc.root
'''.split('\n')),
    'Bd2DmPi': in_data(data, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214045/0000/00214045_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214043/0000/00214043_00000003_1.mc.root
'''.split('\n')),
    'Bs2DsPi': in_data(data, '''
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000006_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000002_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000003_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000004_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000005_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229396/0000/00229396_00000006_1.mc.root
'''.split('\n')),
    'Bs2JpsiPhi': in_data(data, '''
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
    ntuples_raw_withUT[k] = [f.replace(os.path.dirname(f), f'{data}/withUT_MC_2024/1_raw/{k}') for f in v]

ntuples_added_features_withUT = {}
for k,v in ntuples_raw_withUT.items():
    ntuples_added_features_withUT.update({k: [f.replace('1_raw', '2_added_features') for f in v]})

ntuples_selected_withUT = {}
for decay, path_list in ntuples_raw_withUT.items():
    ntuples_selected_withUT.update({decay: {}})
    #print(decay, path_list)
    #print('-------')
    for tagger in taggers_conf[decay] :
       
        ntuples_selected_withUT[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in path_list]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN_IPSig') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{taggers_conf[decay][i]}/{k}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/only_OSKaon') for f in v]})

# Function to read paths from the generated file
def read_generated_paths(data, file):
    with open(file, 'r') as f:
        paths = [join(data,line.strip()) for line in f]
    return paths

from scripts.utils import find_tree_name

# Read the generated paths
generated_paths_SSPion = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_SSPion.txt'))
generated_paths_SSKaon = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_SSKaon.txt'))
generated_paths_SSProton = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_SSProton.txt'))
generated_paths_OSKaon = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_OSKaon.txt'))
generated_paths_OSElectron = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_OSElectron.txt'))
generated_paths_OSMuon = read_generated_paths(data,join(repo,'paths_for_snakemake/generated_paths_OSMuon.txt'))


rule all:
    input:
        #generated_paths
        #ntuples_selected_withUT['Bs2DsPi']['SSKaon'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['OSMuon'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['OSKaon'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['OSElectron'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['SSPion'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['SSKaon'],
        #ntuples_selected_withUT['Bs2JpsiPhi']['SSProton'],
        #ntuples_selected_withUT['Bd2JpsiKst'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSMuon'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],
        #ntuples_added_features_withUT['Bs2DsPi'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSElectron'],
        #ntuples_added_features_withUT['Bd2DmPi'],
        #ntuples_added_features_withUT['Bd2JpsiKst']
        #join(data, 'withUT_MC_2024/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf')
        #generated_paths_SSKaon,
        #generated_paths_SSPion,
        #generated_paths_SSProton,
        #generated_paths_OSKaon,
        #generated_paths_OSElectron,
        #generated_paths_OSMuon,
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/14/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/10/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/12/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/45/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),

        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin/ROC_TRAIN_VAL.pdf'),

        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/config_quentin_2/ROC_TRAIN_VAL.pdf'),

        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),

        join(data, 'savedModels/withUT_MC_2024/Bs2JpsiPhi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.001_bs32_simple/ROC_TRAIN_VAL.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bd2DmPi/SSPion/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/union/2/lr0.01_bs32_simple/ROC_TRAIN_VAL.pdf'),

rule add_features:
    input:
        script = join(repo, 'scripts/adding_features.py'),
        raw = join(data, '{sample_type}/1_raw/{decay}/{id}.root')
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output: 
        root =join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'),
    resources:
        mem_mb = 20000,
        MaxRunHours = 4,
    run:
        tree = find_tree_name(wildcards.decay)
        cmd = ' '.join([
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
            '| tee {log}',
        ])
        print(cmd)
        shell(cmd)
'''
rule train_DT:
    input:
        script = join(repo, 'scripts/origin_DT_cut.py'),
        #data = glob.glob(f'/ceph/users/molocco/classical-taggers/Data/{config.sample_type}/2_added_features/*/*.root')
    output:
        pdf=join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf'),
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.log')
    params:
        target_path = lambda wildcards: join(data, f'{wildcards.sample_type}/DT_outputs/')
    resources:
        mem_mb = 40000, # Specify memory requirement in megabytes 
    run:
        cmd = ' '.join([
            'python', input.script,
            '--target_path {params.target_path}',
            '| tee {log}',
        ])
        print(cmd)
        shell(cmd)
'''

rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        added_features = join(data, '{sample_type}/2_added_features/{decay}/{id}.root'),
    output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{id,.*}.root'),
    # output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root'),
    log : join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    resources:
        mem_mb = 20000,
        MaxRunHours = 4,
    run:
        tree = find_tree_name(wildcards.decay)
        cmd = ' '.join([
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            '--treename', tree,
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {wildcards.tagger}',
            #'--features {wildcards.features}',
            '| tee {log}',
        ])
        print(cmd)
        shell(cmd)

rule train_tagger:
    input:
        #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        selected = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        script = join(repo, 'scripts/pipeline.py'),
    output:
        pdf=join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/ROC_TRAIN_VAL.pdf'),
    log: join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bs2JpsiPhi|Bu2JpsiK|Bu2JpsiK_new|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{features}/{seed}/{config}/training_log.log')
    resources:
        mem_mb = 20000, # Specify memory requirement in megabytes 
        #gpus = 1,
        OnExitRemove = "ExitCode == 0 || ExitCode == 1",  # Allow exit code 1 for debugging
        MaxRunHours = 24, # long queue
        #request_disk = 1024000
    params:
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(data, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.features}/{wildcards.seed}/{wildcards.config}/')
    run:
        cmd = ' '.join([
            'python', input.script,
            '--selected {input.selected}',
            '--target_path {params.target_path}',
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--features {wildcards.features}',
            '--config {params.config}',
            '--decayType {wildcards.decay}',
            #'--clean',
            '| tee {log}',
        ])
        print(cmd)
        shell(cmd)

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
#         cmd = ' '.join([
#             'python', input.script,
#             '--target_path {params.target_path}',
#             '--tagger {wildcards.tagger}',
#             #'--seed {wildcards.seed}',
#             '--config {params.config}',
#             '--decayType {wildcards.decay}',
#             '| tee {log}',
#         ])
#         print(cmd)
#         shell(cmd)