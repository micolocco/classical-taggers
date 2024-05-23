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
    'Bd2JpsiKst': ['SSPion', 'SSProton'],
    'Bs2DsPi': ['SSKaon'],
    'Bd2DmPi': ['SSPion', 'SSProton']

    
}#'Bu2JpsiK': 'OSElectron'

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
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214057/0000/00214057_00000001_1.mc.root
        root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/MC/Dev/MC.ROOT/00214055/0000/00214055_00000001_1.mc.root
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
       
        ntuples_selected_withUT[decay].update({tagger: [f.replace('1_raw', f'3_selected').replace(decay, f'{decay}/{tagger}/cutName') for f in path_list]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN_IPSig') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{taggers_conf[decay][i]}/{k}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin') for f in v]})
    #ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/only_OSKaon') for f in v]})

rule all:
    input:
        #ntuples_selected_withUT['Bs2DsPi'],
        #ntuples_selected_withUT['Bd2DmPi'],
        #ntuples_selected_withUT['Bd2JpsiKst'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSMuon'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSKaon'],
        #ntuples_added_features_withUT['Bs2DsPi'],
        #ntuples_selected_withUT['Bu2JpsiK']['OSElectron'],
        #ntuples_added_features_withUT['Bd2DmPi'],
        #ntuples_added_features_withUT['Bd2JpsiKst']
        #join(data, 'withUT_MC_2024/DT_outputs/tree_schema_maxDepth_Balanced_SSKSSP_noOSP.pdf')
       #
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/only_TRUE/2/withTransformer/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/only_TRUE/2/withTransformer/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/only_TRUE/2/withTransformer/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/only_TRUE/2/withTransformer/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/only_TRUE/2/withTransformer/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/only_TRUE/2/withTransformer/mistag_Training.pdf'),
##
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/noTransformer_run2Feat/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSElectron/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/noTransformer_run2Feat/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSMuon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/noTransformer_run2Feat/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/noTransformer_run2Feat/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/config_test/mistag_Training.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bs2DsPi/SSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/config_test/mistag_Training.pdf'),
        
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/10/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/200/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/345/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/11/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/104/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/95/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/45/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/72/mistag_Training.pdf'),
        #join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/58/mistag_Training.pdf')

        

rule add_features:
    input:
        script = join(repo, 'scripts/adding_features.py'),
        raw = join(data, '{sample_type}/1_raw/{decay}/{id}.root')
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    output: 
        root =join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root'),
    run:
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

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
        cmd = [
            'python', input.script,
            '--target_path {params.target_path}',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        added_features = join(data, '{sample_type}/2_added_features/{decay}/{id}.root'),
    output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{id,.*}.root'),
    # output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root'),
    log : join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger, (OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/.{id,.*}.log')
    # params:
    #     tagger = lambda wildcards: taggers_conf[wildcards.decay]
    run:
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {wildcards.tagger}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule train_tagger:
    input:
        #selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN_IPSig', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        #selected = lambda wildcards: [f.replace('cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        selected = lambda wildcards: [f.replace('cutName', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}'][f'{wildcards.tagger}']],
        script = join(repo, 'scripts/pipeline.py'),
    output:
        pdf=join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{seed}/{config}/mistag_Training.pdf'),
    log: join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|OSMuon|OSElectron|SSPion|SSProton|SSKaon)}/{cut_name}/{seed}/{config}/log.log')
    resources:
        mem_mb = 128000, # Specify memory requirement in megabytes 
        gpus = 1,
        MaxRunHours = 4,
        #request_disk = 1024000
    params:
        config = lambda wildcards: join(repo, f'configs/{wildcards.config}'),
        target_path = lambda wildcards: join(data, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}/{wildcards.seed}/{wildcards.config}/')
    run:
        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--target_path {params.target_path}',
            '--tagger {wildcards.tagger}',
            '--seed {wildcards.seed}',
            '--config {params.config}',
            '--decayType {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))
 