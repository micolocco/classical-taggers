import os
import argparse
from pprint import pprint
'''
python replace_path.py --tagger <tagger> --decayType <decay> --seed <seed> --append <True, False>
'''
seeds = [2, 10, 12, 14, 45, 90, 120]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tagger', help='Tagger type', type=str, choices=('OSKaon', 'SSKaon', 'OSMuon', 'OSElectron', 'SSPion', 'SSProton')) # add all the possible taggers
    parser.add_argument('--decayType', help='Event decay', type=str, choices=('Bu2JpsiK, Bd2JpsiKst, Bs2DsPi, Bd2DPi, Bs2JpsiPhi'))
    parser.add_argument('--seed', help='Random seed', type=str, default=45)
    parser.add_argument('--append', help='Decide whatever appending generated file path or overwrite, w=False, a=True', action='store_true')
    parser.add_argument('--cut', help='Cut used', type=str)

    cfg = parser.parse_args()
    # Define the directory containing the YAML files
    config_dir = '/home/molocco/classical-taggers/configs'
    new_paths = []
    # Define the original path with the placeholder to be replaced
    #original_path = f'savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/{cfg.seed}/lr0.1_bs32_complex/ROC_TRAIN_VAL.pdf'
    outputfile = f'/home/molocco/classical-taggers/paths_for_snakemake/generated_paths_{cfg.tagger}.txt'
    # List all YAML files in the config directory
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and f != 'config_test.yaml']
    print(f"Added paths {outputfile}: \n")
    # Iterate over each YAML file and replace the placeholder in the path
    for yaml_file in yaml_files:
        for seed in seeds:
            original_path = f'savedModels/withUT_MC_2024/{cfg.decayType}/{cfg.tagger}/{cfg.cut}/{seed}/lr0.1_bs32_complex/ROC_TRAIN_VAL.pdf'
            # Extract the base name without the .yaml extension
            base_name = os.path.splitext(yaml_file)[0]
            # Replace the placeholder in the original path with the base name
            new_path = original_path.replace('lr0.1_bs32_complex', base_name)
            new_paths.append(new_path)
            print(new_path)
    print(f'Total paths created:{len(yaml_files)*len(seeds)}')
    mode = 'a' if cfg.append else 'w'
    with open(outputfile, mode) as f:
        for path in new_paths:
            f.write(f"{path}\n")
