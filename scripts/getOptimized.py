from IPython import embed
import numpy as np
import json, os
import utils
from itertools import product
from uncertainties import ufloat


learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 128, 1024, 2048]
architectures = ['simple', 'complex']
seeds = [2, 10, 12, 14, 45]
#cut = 'cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin'
cut = 'cut_DT_PROBNN_unbalanced_round3'
features = 'union_PROBNN'

outputPath= '/home/molocco/classical-taggers/best_tagger_candidates'
tagger_dict = {
    "OSKaon": "Bu2JpsiK",
    "OSElectron": "Bu2JpsiK",
    "OSMuon": "Bu2JpsiK",
    "SSPion": "Bd2JpsiKst",
    "SSProton": "Bd2JpsiKst",
    "SSKaon": "Bs2DsPi",
}

# Generate all possible combinations of hyperparameters
combinations = list(product(seeds, learning_rates, batch_sizes, architectures))

max_ratios = {}
link = 'logit' # 'mistag'
for tagger, decay in tagger_dict.items():
    max_ratio = -np.inf
    best_hyperparams = None
    for seed, lr, bs, arch in combinations:
        # Read tagging power values from JSON files
        results_folder = f"/ceph/users/molocco/Data/savedModels/withUT_MC_2024/{decay}/{tagger}/{cut}/{features}/{seed}" #cfg.seed   
        folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_{arch}")
        json_file = os.path.join(folder_path, f"{link}/taggingInfo_{link}.json")

        if os.path.exists(json_file):
            data = utils.load_and_process_json(json_file)
            print(data)
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
                        "max_ratio": max_ratio,
                        "precision": ratio_precision,
                    }
    
    if best_hyperparams:
        max_ratios[tagger] = best_hyperparams

# Output the dictionary with the maximum ratios and corresponding hyperparameters
print(json.dumps(max_ratios,  indent=4, default=str))
filename=f'{outputPath}/candidatedTaggers_{link}_{cut}.json'
with open(filename, 'w') as f:
    json.dump(max_ratios, f, indent=4, default=str)  # `default=str` to handle non-serializable objects
print(f"Max ratios saved to {filename}")

