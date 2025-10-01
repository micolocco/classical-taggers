import os
import argparse
from pprint import pprint

'''
This script allows to create for each tagger a txt file (`generated_paths_<tagger>.txt`) containing a path of type:
savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf
for each of the hyperparameter combination created by running the script `generate_configFiles.py`.
The txt file generated_paths_<tagger>.txt will be used in the snakemake file to get all the Neural Networks that must be trained.
'''
seeds = [12, 45]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--repoPath', help='Path to where the outputs must be stored', type=str, default='.') 


    cfg = parser.parse_args()
    # Define the directory containing the YAML files

    config_dir = f'{cfg.repoPath}/configs'

    taggers_dict = {
    'OSKaon': 'Bu2JpsiK',
    'OSElectron': 'Bu2JpsiK',
    'OSMuon': 'Bu2JpsiK',
    'SSPion': 'Bd2JpsiKst',
    'SSProton': 'Bd2JpsiKst',
    'SSKaon': 'Bs2DsPi'
    }
    modelDir = f'/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024'
    features = 'union_PROBNN'
    cut = 'allBKGCAT_notSamePV_noOSP_SSK_balanced'
    new_paths = []
    # Define the original path with the placeholder to be replaced
    #original_path = f'savedModels/withUT_MC_2024/{decayType}/{tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/{seed}/lr0.1_bs32_complex/ROC_TRAIN_VAL.pdf'
    os.makedirs(f'{cfg.repoPath}/paths_for_snakemake/', exist_ok=True)

    # List all YAML files in the config directory
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and f != 'config_test.yaml']
    # Iterate over each YAML file and replace the placeholder in the path
    for tagger in taggers_dict.keys():
        new_paths = []
        outputfile= f'{cfg.repoPath}/paths_for_snakemake/generated_paths_{tagger}.txt'
        for yaml_file in yaml_files:
            for seed in seeds:
                decayType = taggers_dict[tagger]
                original_path = f'{modelDir}/{decayType}/{tagger}/{cut}/{features}/{seed}/hyperparameter_combo/asym_level1/ROC_TRAIN_VAL.pdf'
                # Extract the base name without the .yaml extension
                base_name = os.path.splitext(yaml_file)[0]
                # Replace the placeholder in the original path with the base name
                new_path = original_path.replace('hyperparameter_combo', base_name)
                new_paths.append(new_path)
                print(new_path)
        print(f'Total paths created:{len(yaml_files)*len(seeds)}')
        #mode = 'a' if os.path.exists(outputfile) else 'w'
        mode = 'w'
        with open(outputfile, mode) as f:
            for path in new_paths:
                f.write(f"{path}\n")
