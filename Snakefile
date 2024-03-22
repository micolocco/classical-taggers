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
    'Bu2JpsiK': 'OSKaon'
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
for k,v in ntuples_raw_withUT.items():
    ntuples_selected_withUT.update({k: [f.replace('1_raw', f'3_selected').replace(k, f'{k}/cut_Run2Summer2017Opt_v2_noProbNN') for f in v]})

rule all:
    input:
        #ntuples_selected_withUT['Bs2DsPi'],
        #ntuples_selected_withUT['Bd2DmPi'],
        #ntuples_selected_withUT['Bd2JpsiKst'],
        #ntuples_selected_withUT['Bu2JpsiK']
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig_noOriginFlag/mistag_validation.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN/mistag_validation.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_IPSig/mistag_validation.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_eta4.7/mistag_validation.pdf'),
        join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_eta4.8/mistag_validation.pdf'),
        # join(data, 'savedModels/withUT_MC_2024/Bu2JpsiK/OSKaon/cut_Run2Summer2017Opt_v2_noProbNN_noGhosts/mistag_validation.pdf')
        

rule add_features:
    input:
        script = join(repo, 'scripts/adding_features.py'),
        raw = join(data, '{sample_type}/1_raw/{decay}/{id}.root')
    output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{id,.*}.root')
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/2_added_features/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/.{id,.*}.log')
    run:
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output}',
            '--evtType {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))

rule add_selection:
    input:
        script = join(repo, 'scripts/preSelections.py'),
        added_features = join(data, '{sample_type}/2_added_features/{decay}/{id}.root'),
    output: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/{id,.*}.root')
    log: join(data, '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{cut_name}/.{id,.*}.log')
    params:
        tagger = lambda wildcards: taggers_conf[wildcards.decay]
    run:
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            '--cut_file', join(repo, 'cuts/{wildcards.sample_type}/{wildcards.decay}/{params.tagger}/{wildcards.cut_name}.txt'),
            '--tagger {params.tagger}',
            '&> {log}',
        ]
        shell(' '.join(cmd))


rule train_tagger:
    input:
        selected = lambda wildcards: [f.replace('cut_Run2Summer2017Opt_v2_noProbNN', f'{wildcards.cut_name}') for f  in ntuples_selected_withUT[f'{wildcards.decay}']],
        script = join(repo, 'scripts/pipeline.py'),
    output: join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|SSKaon)}/{cut_name}/mistag_validation.pdf')
    log: join(data, 'savedModels/{sample_type,(withUT_MC_2024|noUT_MC_2024)}/{decay,(Bu2JpsiK|Bd2JpsiKst|Bd2DmPi|Bs2DsPi)}/{tagger,(OSKaon|SSKaon)}/{cut_name}/mistag_validation.log')
    params:
        config = lambda wildcards: join(repo, f'configs/config_{wildcards.tagger}.json'),
        target_path = lambda wildcards: join(data, f'savedModels/{wildcards.sample_type}/{wildcards.decay}/{wildcards.tagger}/{wildcards.cut_name}')
    run:
        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--target_path {params.target_path}',
            '--tagger {wildcards.tagger}',
            '--config {params.config}',
            '--decayType {wildcards.decay}',
            '&> {log}',
        ]
        shell(' '.join(cmd))