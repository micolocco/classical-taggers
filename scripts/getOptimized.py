from IPython import embed
import numpy as np
import json, os
import utils
from itertools import product
from uncertainties import ufloat
import argparse
from scripts.generate_configFiles import learning_rates
from scripts.generate_configFiles import train_batch_sizes
from scripts.generate_configFiles import architectures
from scripts.generate_configFiles import min_delta
from scripts.replace_path import seeds


'''
python scripts/getOptimized.py --cut <cutName>
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_prePath', help='Name of the output dir', type=str, default='/ceph/users/togasa/FlavourTagging/NTuples/MC/savedModels/withUT_MC_2024')
    parser.add_argument('--cut', help='Cut type to be used', type=str)
    parser.add_argument('--output', help='Where the best tagger candidates configs will be saved', type=str, default='./best_tagger_candidates')
    parser.add_argument('--features', help='Input features for NN training', default='union_PROBNN') 
    
    cfg = parser.parse_args()

    from pprint import pprint
    pprint(cfg)

    #learning_rates = [0.1, 0.01, 0.001]
    #batch_sizes = [32, 128, 1024, 2048]
    #architectures = ['simple', 'complex']
    #seeds = [2, 10, 12, 14, 45]

    tagger_dict = {
        "OSKaon": "Bu2JpsiK",
        "OSElectron": "Bu2JpsiK",
        "OSMuon": "Bu2JpsiK",
        "SSPion": "Bd2JpsiKst",
        "SSProton": "Bd2JpsiKst",
        "SSKaon": "Bs2DsPi",
    }

    # Generate all possible combinations of hyperparameters
    #architectures = ['simple']

    combinations = list(product(seeds, learning_rates, train_batch_sizes, architectures, min_delta))

    max_ratios = {}
    link = 'logit' # 'mistag'
    for tagger, decay in tagger_dict.items():
        max_ratio = -np.inf
        best_hyperparams = None
        for seed, lr, bs, arch, dm in combinations:

            # Read tagging power values from JSON files
            results_folder = f"{cfg.model_prePath}/{decay}/{tagger}/{cfg.cut}/{cfg.features}/{seed}" #cfg.seed   
            folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_{arch}_dm{dm}")

            json_file = os.path.join(folder_path, f"{link}/taggingInfo_{link}.json")
            print(json_file)
            
            if os.path.exists(json_file):
                print("enters outermost if")
                data = utils.load_and_process_json(json_file)
                tagging_power = data['TaggingPower_Cali']
                if not np.isnan(tagging_power.nominal_value) and tagging_power.nominal_value != 0:
                    try:
                        ratio = tagging_power.nominal_value / tagging_power.std_dev
                        ratio_precision =  tagging_power.std_dev / tagging_power.nominal_value 
                    except ZeroDivisionError:
                        print("Check std deviation or nominal value. They might be 0")
                    #print(tagger, arch, lr, seed, bs)
                    #print(tagging_power[0], tagging_power[1])
                    #print(ratio
                    
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_hyperparams = {
                            "calibrated tagging power": tagging_power,
                            "seed": seed,
                            "learning_rate": lr,
                            "batch_size": bs,
                            "architecture": arch,
                            "min_delta": dm,
                            "max_ratio": max_ratio,
                            "precision": ratio_precision,
                        }
        
        if best_hyperparams:
            max_ratios[tagger] = best_hyperparams

    # Output the dictionary with the maximum ratios and corresponding hyperparameters
    print(json.dumps(max_ratios,  indent=4, default=str))
    # filename=f'{cfg.outputPath}/{cfg.cut}/candidatedTaggers_{link}.json'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)

    filename = cfg.output
    os.makedirs(os.path.dirname(os.path.dirname(filename)), exist_ok=True)


    with open(filename, 'w') as f:
        json.dump(max_ratios, f, indent=4, default=str)  # `default=str` to handle non-serializable objects
    print(f"Max ratios saved to {filename}")

