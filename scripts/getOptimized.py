from IPython import embed
import matplotlib
import numpy as np
import json, os
import utils
from itertools import product
from uncertainties import ufloat
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import yaml
import pickle
from pprint import pprint

import numpy as np

from scripts import matplotlib_lhcb_style
matplotlib_lhcb_style(plt)

'''
python scripts/getOptimized.py --cut <cutName> --data_type <MC/Data> 
'''

def plot_hyperparams_vs_tagging_power(df, target_path, num_features):
    """
    Plots hyperparameters against tagging_power in a 2x2 subplot grid with linear regression lines.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing hyperparameters and 'tagging_power'.
                           Expected columns: ['learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power']
    """
    print(f"Plotting hyperparameters vs tagging power")
    hyperparams = ['learning_rate', 'batch_size', 'num_layers', 'num_neurons']
    xscale = ['log', 'log', 'linear', 'log']  
    target = 'tagging_power'

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    translation_dict = {'learning_rate' : 'Learning rate',
                        'batch_size': 'Batch size',
                        'num_layers': 'Layer count',
                        'num_neurons': 'Number of Neurons per Layer',
                        'tagging_power': r'$\eta_{\text{eff}}$'}

    xticks = {
        'learning_rate': [0.0001, 0.001, 0.01,],
        'batch_size': [2048, 4096, 8192],
        'num_layers': [2, 4, 6, 8, 16],
        'num_neurons': [8, 16, 32, 64, 128, 256]
    }

    for i, hp in enumerate(hyperparams):
        print(f"Plotting {hp} vs {target}")
        ax = axs[i]
        sns.scatterplot(data=df, x=hp, y=target, ax=ax, alpha=0.6)

        # Fit linear regression
        X = df[[hp]].values
        y = df[target].values
        model = LinearRegression()
        # model = PolynomialFeatures(degree=2)
        model.fit(X, y)
        
        xfit = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
        y_pred = model.predict(xfit)
        ax.plot(xfit, y_pred, color='red', linewidth=2)


        # ax.set_title(f'{hp} vs {target}')
        ax.set_xscale(xscale[i])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        print(f'Setting xticks for {hp}: {xticks[hp]}')
        plt.sca(axs[i])
        plt.xticks(xticks[hp], [str(i) for i in xticks[hp]])



        # ax.set_xticks(xticks[hp])
        # ax.set_xticklabels([str(i) for i in xticks[hp]])
        
        ax.set_xlabel(translation_dict[hp])
        ax.set_ylabel(translation_dict[target])    

    plt.tight_layout()
    print(f"Saving plot to {target_path}")
    plt.savefig(target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_prePath', help='Name of the output dir', type=str, default='/ceph/users/togasa/FlavourTagging')
    parser.add_argument('--cut', help='Cut type to be used', type=str, default='notSamePV_noOSP')
    parser.add_argument('--outpath', help='Where the best tagger candidates configs will be saved', type=str, default='./best_tagger_candidates')
    parser.add_argument('--plot_path', help='Where the plots will be saved', type=str, default='/ceph/users/togasa/FlavourTagging/')
    parser.add_argument('--features', help='Input features for NN training', default='union_PROBNN') 
    parser.add_argument('--data_type', help='Type of data to be used', type=str, choices=['MC', 'Data'])  # 'MC' or 'Data'
    parser.add_argument('--tagger_input', help='File of the tagger inputs, only needed for num parameter plot', type=str, default='/home/togasa/classical-taggers/tagger_inputFeatures/union_PROBNN.yaml')
    parser.add_argument('--BN', help='Whether to check batch normalized models', action='store_true')
    
    cfg = parser.parse_args()

    pprint(cfg)

    data_type = cfg.data_type

    tagger_dict = {
        "OSKaon": "Bu2JpsiK",
        "OSElectron": "Bu2JpsiK",
        "OSMuon": "Bu2JpsiK",
        "SSPion": "Bd2JpsiKst",
        "SSProton": "Bd2JpsiKst",
        # "SSKaon": "Bs2DsPi",
    }

    #open tagger input features
    with open(cfg.tagger_input, 'r') as f:
        tagger_input_features = yaml.safe_load(f)

    print(tagger_input_features)

    with open(f'configs/hyperpar_intervals.yaml', 'r') as file:
        intervals = yaml.safe_load(file)
    
    
    learning_rates = intervals['-learning_rate']
    train_batch_sizes = intervals['-train_batch_size']
    num_layers = intervals['-numlayers']
    num_neurons = intervals['-numneurons']
    seeds  = [12]

    learning_rates = [0.0001, 0.001]
    num_layers = [8, 16]
    num_neurons = [32, 64, 128]
    train_batch_sizes = [8192]

    combinations = list(product(seeds, learning_rates, train_batch_sizes, num_layers, num_neurons))
    print(combinations)

    full_df =          pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc', 'link'])
    max_ratios = {}
    best_models = {}
    for link in  ['logit', 'mistag']:
        performances = pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc'])

        for tagger, decay in tagger_dict.items():
            max_ratio = -np.inf
            best_hyperparams = None
            best_model = None
            for seed, lr, bs, nl, nn in combinations:

                # Read tagging power values from JSON files
                results_folder = f"{cfg.model_prePath}/{data_type}/savedModels/{decay}/{tagger}/{cfg.cut}/{cfg.features}/{seed}" #cfg.seed
                config = f"lr{lr}_bs{bs}_nL{nl}_nN{nn}"
                if cfg.BN: config += f"_BN"
                folder_path = os.path.join(results_folder, config)

                json_file_data = os.path.join(folder_path, f"testing/Data/{link}/taggingInfo_{link}.json")
                json_file_mc = os.path.join(folder_path, f"testing/MC/{link}/taggingInfo_{link}.json")

                print(f'Checking files {json_file_data} and {json_file_mc}')
                if os.path.exists(json_file_data):
                    data = utils.load_and_process_json(json_file_data)
                    tagging_power = data['TaggingPower_Cali']

                    if os.path.exists(json_file_mc):
                        data_mc = utils.load_and_process_json(json_file_mc)
                        tagging_power_mc = data_mc['TaggingPower_Cali']
                    else:
                        tagging_power_mc = ufloat(np.nan, np.nan)


                    if not np.isnan(tagging_power.nominal_value) and tagging_power.nominal_value != 0:
                        if tagging_power.std_dev > 1e-4 and tagging_power.nominal_value > 1e-4: # Make sure tagging power and uncertainty are realistic
                            try:
                                ratio = tagging_power.nominal_value / tagging_power.std_dev
                                ratio_precision =  tagging_power.std_dev / tagging_power.nominal_value 
                            except ZeroDivisionError:
                                print("Check std deviation or nominal value. They might be 0")


                            if ratio > max_ratio:
                                max_ratio = ratio
                                best_hyperparams = {
                                    "calibrated tagging power": tagging_power,
                                    "calibrated tagging power MC": tagging_power_mc,
                                    "seed": seed,
                                    "learning_rate": lr,
                                    "batch_size": bs,
                                    "numlayers": nl,
                                    "numneurons": nn,
                                    "max_ratio": max_ratio,
                                    "precision": ratio_precision,
                                }
                                best_model = json_file_data
                            print(f"Tagger: {tagger}, Seed: {seed}, LR: {lr}, Bs: {bs}, NL: {nl}, NN: {nn}, Tagging Power: {tagging_power.nominal_value}, Ratio: {ratio}, Tagging Power MC: {tagging_power_mc.nominal_value}")
                            performances.loc[len(performances)] = [tagger, lr, bs, nl, nn, tagging_power.nominal_value, tagging_power.std_dev, tagging_power_mc.nominal_value, tagging_power_mc.std_dev]

            if best_hyperparams:
                max_ratios[tagger] = best_hyperparams
                best_models[tagger] = best_model
                print(best_model)
                best_models[tagger + '_cal_plot'] = best_model.replace(f'taggingInfo_{link}.json', f'{tagger}_Calibration.pdf')

        # Output the dictionary with the maximum ratios and corresponding hyperparameters
        print(json.dumps(max_ratios,  indent=4, default=str))
        # filename=f'{cfg.outputPath}/{cfg.cut}/candidatedTaggers_{link}.json'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        path_name = os.path.join(cfg.outpath, f'{cfg.cut}/{data_type}')
        if cfg.BN: path_name = path_name.replace(data_type, data_type+'_BN')
        plot_path = os.path.join(cfg.plot_path, f'{data_type}/hyperparameters_plots/{cfg.cut}/{link}')
        if cfg.BN: plot_path = plot_path.replace(data_type, data_type+'_BN')
        os.makedirs(path_name, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        #save best_models to a json file. Makes it easier to check the calibrations
        with open(os.path.join(plot_path, f'best_models_{link}.json'), 'w') as f:
            json.dump(best_models, f, indent=4, default=str)

        for tagger, decay in tagger_dict.items():
            plot_hyperparams_vs_tagging_power(performances[performances['tagger'] == tagger], os.path.join(plot_path, f'{tagger}_Hyperparams_vs_TaggingPower.png'), len(tagger_input_features[tagger]['features']))

        performances['link'] = link
        full_df = pd.concat([full_df, performances], ignore_index=True)


        filename = os.path.join(path_name, f'candidatedTaggers_{link}.json')
        os.makedirs(os.path.dirname(os.path.dirname(filename)), exist_ok=True)


        with open(filename, 'w') as f:
            json.dump(max_ratios, f, indent=4, default=str)  # `default=str` to handle non-serializable objects
        print(f"Max ratios with link {link} saved to {filename}")

    # Save the full DataFrame to a CSV file
    df_path = os.path.dirname(plot_path)
    full_df.to_csv(os.path.join(df_path, 'performances.csv'), index=False)   


    print(f"All results saved to {cfg.outpath}/{cfg.cut}/{data_type}/")

