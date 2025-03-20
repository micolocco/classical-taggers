import os
import argparse
from pprint import pprint
'''
python replace_path.py --tagger <tagger> --decayType <decay> --seed <seed> --append <True, False>
This script allows to create for each tagger a txt file (`generated_paths_<tagger>.txt`) containing a path of type:
savedModels/withUT_MC_2024/Bd2JpsiKst/SSProton/notSamePV_noOSP/union_PROBNN/2/lr0.001_bs128_simple_dm0.0/ROC_TRAIN_VAL.pdf
for each of the hyperparameter combination created by running the script `generate_configFiles.py`.
The txt file generated_paths_<tagger>.txt will be used in the snakemake file to get all the Neural Networks that must be trained.
'''
seeds = [2, 10, 12, 14, 45, 90, 120]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--decayType', help='Event decay', type=str, choices=('Bu2JpsiK, Bd2JpsiKst, Bs2DsPi, Bd2DPi, Bs2JpsiPhi'))
    parser.add_argument('--append', help='Decide whatever appending generated file path or overwrite, w=False, a=True', action='store_true')
    parser.add_argument('--cut', help='Cut used', type=str)
    parser.add_argument('--features', help='Input features used for NN training',) 
    parser.add_argument('--modelDir', help='Path to the directory where the models will be stored', type=str) 
    parser.add_argument('--repoPath', help='Path to where the outputs must be stored', type=str) 


    cfg = parser.parse_args()
    # Define the directory containing the YAML files

    config_dir = f'{cfg.repoPath}/configs'
    os.makedirs(config_dir, exist_ok=True)

    new_paths = []
    # Define the original path with the placeholder to be replaced
    #original_path = f'savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/{cfg.seed}/lr0.1_bs32_complex/ROC_TRAIN_VAL.pdf'
    outputfile= f'{cfg.repoPath}/paths_for_snakemake/generated_paths_{cfg.tagger}.txt'
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

    # List all YAML files in the config directory
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and f != 'config_test.yaml']
    print(f"Added paths {outputfile}: \n")
    # Iterate over each YAML file and replace the placeholder in the path
    for yaml_file in yaml_files:
        for seed in seeds:
            original_path = f'{cfg.modelDir}/{cfg.decayType}/{cfg.tagger}/{cfg.cut}/{cfg.features}/{seed}/hyperparameter_combo/ROC_TRAIN_VAL.pdf'
            # Extract the base name without the .yaml extension
            base_name = os.path.splitext(yaml_file)[0]
            # Replace the placeholder in the original path with the base name
            new_path = original_path.replace('hyperparameter_combo', base_name)
            new_paths.append(new_path)
            print(new_path)
    print(f'Total paths created:{len(yaml_files)*len(seeds)}')
    mode = 'a' if cfg.append else 'w'
    with open(outputfile, mode) as f:
        for path in new_paths:
            f.write(f"{path}\n")
