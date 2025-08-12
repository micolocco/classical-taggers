from IPython import embed
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

import numpy as np

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

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, hp in enumerate(hyperparams):
        print(f"Plotting {hp} vs {target}")
        ax = axs[i]
        sns.scatterplot(data=df, x=hp, y=target, ax=ax, color='blue', alpha=0.6)

        # Fit linear regression
        X = df[[hp]].values
        y = df[target].values
        model = LinearRegression()
        # model = PolynomialFeatures(degree=2)
        model.fit(X, y)
        
        xfit = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
        y_pred = model.predict(xfit)
        # y_pred = model.transform(xfit)
        

        ax.plot(xfit, y_pred, color='red', linewidth=2)

        ax.set_title(f'{hp} vs {target}')

        ax.set_xscale(xscale[i])
        ax.set_xlabel(hp)
        ax.set_ylabel(target)

    print(f"Plotting num_params vs {target}")
    ax = axs[4]

    df['num_params'] = df['num_layers'] * df['num_neurons'] * (df['num_neurons'] + 1)  # Assuming a fully connected layer with bias
    df['num_params'] = df['num_params'] + num_features * df['num_layers'] + df['num_neurons']    # Adding the input and output layer parameters

    hp = 'num_params'

    sns.scatterplot(data=df, x=hp, y=target, ax=ax, color='blue', alpha=0.6)

    # Fit linear regression
    X = df[[hp]].values
    y = df[target].values
    model = LinearRegression()
    # model = PolynomialFeatures(degree=2)
    model.fit(X, y)
    
    xfit = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
    y_pred = model.predict(xfit)
    # y_pred = model.transform(xfit)
    

    ax.plot(xfit, y_pred, color='red', linewidth=2)

    ax.set_title(f'{hp} vs {target}')
    ax.set_xscale('log')
    ax.set_xlabel(hp)
    ax.set_ylabel(target)

    #Calculate the correlation coefficients and display them in subplot 6
    ax = axs[5]
    corr = df[hyperparams+ ['num_params', 'tagging_power']].corr()[target].drop(target)
    print(corr)
    sns.barplot(x=corr.index, y=corr.values, ax=ax, palette='viridis')
    ax.set_title(f'Correlation with {target}')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_xlabel('Hyperparameter')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    #highlight the zero line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')


    

    plt.tight_layout()
    print(f"Saving plot to {target_path}")
    plt.savefig(target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_prePath', help='Name of the output dir', type=str, default='/ceph-kernel/users/togasa/FlavourTagging/NTuples')
    parser.add_argument('--cut', help='Cut type to be used', type=str)
    parser.add_argument('--outpath', help='Where the best tagger candidates configs will be saved', type=str, default='./best_tagger_candidates')
    parser.add_argument('--plot_path', help='Where the plots will be saved', type=str, default='/ceph-kernel/users/togasa/FlavourTagging/NTuples')
    parser.add_argument('--features', help='Input features for NN training', default='union_PROBNN') 
    parser.add_argument('--data_type', help='Type of data to be used', type=str, choices=['MC', 'Data'])  # 'MC' or 'Data'
    parser.add_argument('--tagger_input', help='File of the tagger inputs, only needed for num parameter plot', type=str, default='/ceph/users/togasa/classical-taggers/tagger_inputFeatures/union_PROBNN.yaml')

    
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
        # "SSPion": "Bd2JpsiKst",
        # "SSProton": "Bd2JpsiKst",
        # "SSKaon": "Bs2DsPi",
    }

    #open tagger input features
    with open(cfg.tagger_input, 'r') as f:
        tagger_input_features = yaml.safe_load(f)

    print(tagger_input_features)

    # Generate all possible combinations of hyperparameters
    #architectures = ['simple']

    with open(f'configs/hyperpar_intervals.yaml', 'r') as file:
        intervals = yaml.safe_load(file)
    
    
    learning_rates = intervals['-learning_rate']
    train_batch_sizes = intervals['-train_batch_size']
    num_layers = intervals['-numlayers']
    num_neurons = intervals['-numneurons']
    seeds  = [12]

    # learning_rates=[0.01, 0.001,]
    # train_batch_sizes=[4096, 32768]
    # num_layers = [2,4]
    # num_neurons=[4,8]

    # combinations = list(product(seeds, learning_rates, train_batch_sizes, num_layers, num_neurons))
    combinations = list(product(seeds, learning_rates, train_batch_sizes, num_layers, num_neurons))


    full_df = pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc', 'link'])
    max_ratios = {}
    for link in  ['logit', 'mistag']:
        performances = pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc'])
        for tagger, decay in tagger_dict.items():
            max_ratio = -np.inf
            best_hyperparams = None
            for seed, lr, bs, nl, nn in combinations:

                # Read tagging power values from JSON files
                results_folder = f"{cfg.model_prePath}/{cfg.data_type}/savedModels/withUT_MC_2024/{decay}/{tagger}/{cfg.cut}/{cfg.features}/{seed}" #cfg.seed
                folder_path = os.path.join(results_folder, f"lr{lr}_bs{bs}_nL{nl}_nN{nn}")

                if cfg.data_type == 'Data':
                    folder_path = os.path.join(folder_path, 'pdf_ratio')

                json_file_data = os.path.join(folder_path, f"testing/Data/{link}/taggingInfo_{link}.json")
                json_file_mc = os.path.join(folder_path, f"testing/MC/{link}/taggingInfo_{link}.json")
                # print(f"{folder_path} Data exists: {os.path.exists(json_file_data)} MC exists: {os.path.exists(json_file_mc)}")
                # print(json_file_mc)
                if os.path.exists(json_file_data) and os.path.exists(json_file_mc):
                    data = utils.load_and_process_json(json_file_data)
                    tagging_power = data['TaggingPower_Cali']
                    deltap1 = data['Fitpar_deltap1']
                    deltap0 = data['Fitpar_deltap0']
                    print(f'deltap1: {deltap1},deltap0: {deltap0},tg: {tagging_power}')
                    if np.abs(deltap1.nominal_value) > 1 or np.abs(deltap0.nominal_value) > 1: #Ignore insensible result
                        print(f"Skipping {tagger} with lr={lr}, bs={bs}, nl={nl}, nn={nn} due to high deltap1 or deltap0")
                        continue

                    data_mc = utils.load_and_process_json(json_file_mc)
                    tagging_power_mc = data_mc['TaggingPower_Cali']
                


                    if not np.isnan(tagging_power.nominal_value) and tagging_power.nominal_value != 0:
                        if tagging_power.std_dev > 1e-4 and tagging_power.nominal_value > 1e-4: # Make sure tagging power and uncertainty are realistic
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
                                    "calibrated tagging power MC": tagging_power_mc,
                                    "seed": seed,
                                    "learning_rate": lr,
                                    "batch_size": bs,
                                    "numlayers": nl,
                                    "numneurons": nn,
                                    "max_ratio": max_ratio,
                                    "precision": ratio_precision,
                                }
                            print(f"Tagger: {tagger}, Seed: {seed}, LR: {lr}, Bs: {bs}, NL: {nl}, NN: {nn}, Tagging Power: {tagging_power.nominal_value}, Ratio: {ratio}")
                            performances.loc[len(performances)] = [tagger, lr, bs, nl, nn, tagging_power.nominal_value, tagging_power.std_dev, tagging_power_mc.nominal_value, tagging_power_mc.std_dev]

            if best_hyperparams:
                max_ratios[tagger] = best_hyperparams

        # Output the dictionary with the maximum ratios and corresponding hyperparameters
        print(json.dumps(max_ratios,  indent=4, default=str))
        # filename=f'{cfg.outputPath}/{cfg.cut}/candidatedTaggers_{link}.json'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        path_name = os.path.join(cfg.outpath, f'{cfg.cut}/{cfg.data_type}')
        plot_path = os.path.join(cfg.plot_path, f'{cfg.data_type}/hyperparameters_plots/{cfg.cut}/{link}')
        os.makedirs(plot_path, exist_ok=True)

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
    full_df.to_csv(os.path.join(os.path.dirname(plot_path), 'performances.csv'), index=False)
    
    print(f"All results saved to {cfg.outpath}/{cfg.cut}/{cfg.data_type}/")

