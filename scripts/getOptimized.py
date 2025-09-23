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
        # y_pred = model.transform(xfit)
        # ax.plot(xfit, y_pred, color='red', linewidth=2)


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

    # print(f"Plotting num_params vs {target}")
    # ax = axs[4]

    # df['num_params'] = df['num_layers'] * df['num_neurons'] * (df['num_neurons'] + 1)  # Assuming a fully connected layer with bias
    # df['num_params'] = df['num_params'] + num_features * df['num_layers'] + df['num_neurons']    # Adding the input and output layer parameters

    # hp = 'num_params'

    # sns.scatterplot(data=df, x=hp, y=target, ax=ax, color='blue', alpha=0.6)

    # # Fit linear regression
    # X = df[[hp]].values
    # y = df[target].values
    # model = LinearRegression()
    # # model = PolynomialFeatures(degree=2)
    # model.fit(X, y)
    
    # xfit = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
    # y_pred = model.predict(xfit)
    # # y_pred = model.transform(xfit)
    

    # ax.plot(xfit, y_pred, color='red', linewidth=2)

    # ax.set_title(f'{hp} vs {target}')
    # ax.set_xscale('log')
    # ax.set_xlabel(hp)
    # ax.set_ylabel(target)

    # #Calculate the correlation coefficients and display them in subplot 6
    # ax = axs[5]
    # corr = df[hyperparams+ ['num_params', 'tagging_power']].corr()[target].drop(target)
    # print(corr)
    # sns.barplot(x=corr.index, y=corr.values, ax=ax, palette='viridis')
    # ax.set_title(f'Correlation with {target}')
    # ax.set_ylabel('Correlation Coefficient')
    # ax.set_xlabel('Hyperparameter')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # #highlight the zero line
    # ax.axhline(0, color='black', linewidth=0.8, linestyle='-')


    

    plt.tight_layout()
    print(f"Saving plot to {target_path}")
    plt.savefig(target_path)

def mistag_is_monotonic(bin_data):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_prePath', help='Name of the output dir', type=str, default='/ceph/users/togasa/FlavourTagging/NTuples')
    parser.add_argument('--cut', help='Cut type to be used', type=str, default='notSamePV_noOSP')
    parser.add_argument('--outpath', help='Where the best tagger candidates configs will be saved', type=str, default='./best_tagger_candidates')
    parser.add_argument('--plot_path', help='Where the plots will be saved', type=str, default='/ceph/users/togasa/FlavourTagging/NTuples')
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

    data_type = cfg.data_type
    for data_type in ['MC', 'Data']:

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


        max_deltp =1

        full_df = pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc',  'Bbar_min_diff', 'Bbar_mean_diff', 'B_min_diff', 'B_mean_diff','bin_diffs_std', 'link'])
        max_ratios = {}
        best_models = {}
        for link in  ['logit']:#, 'mistag', 'rlogit']:
            performances = pd.DataFrame(columns=['tagger', 'learning_rate', 'batch_size', 'num_layers', 'num_neurons', 'tagging_power', 'tagging_power_unc', 'tagging_power_mc', 'tagging_power_mc_unc', 'Bbar_min_diff', 'Bbar_mean_diff', 'B_min_diff', 'B_mean_diff','bin_diffs_std'])

            delta0s = []
            delta1s = []
            delta2s = []


            for tagger, decay in tagger_dict.items():
                max_ratio = -np.inf
                best_hyperparams = None
                best_model = None
                for seed, lr, bs, nl, nn in combinations:

                    # Read tagging power values from JSON files
                    results_folder = f"{cfg.model_prePath}/{data_type}/savedModels/withUT_MC_2024/{decay}/{tagger}/{cfg.cut}/{cfg.features}/{seed}" #cfg.seed
                    config = f"lr{lr}_bs{bs}_nL{nl}_nN{nn}"
                    folder_path = os.path.join(results_folder, config)

                    if data_type == 'Data':
                        folder_path = os.path.join(folder_path, 'pdf_ratio')

                    json_file_data = os.path.join(folder_path, f"testing/Data/{link}/taggingInfo_{link}.json")
                    json_file_mc = os.path.join(folder_path, f"testing/MC/{link}/taggingInfo_{link}.json")


                    if os.path.exists(json_file_data) and os.path.exists(json_file_mc):
                        data = utils.load_and_process_json(json_file_data)
                        tagging_power = data['TaggingPower_Cali']
                        data_mc = utils.load_and_process_json(json_file_mc)
                        tagging_power_mc = data_mc['TaggingPower_Cali']

                        diff_file  = os.path.join(folder_path, f'testing/Data/eta_omega_bins.pkl')

                        try:
                            with open(diff_file, 'rb') as f:
                                bin_contents = pickle.load(f)


                            omega_means_bbar = np.array(bin_contents['Bbar_omega_means'])
                            omega_unc_bbar = np.array(bin_contents['Bbar_omega_means_unc'])
                            if not np.isnan(omega_means_bbar).any() and len(omega_means_bbar) > 1:
                                diffs = omega_means_bbar[1:]-omega_means_bbar[:-1]
                                sigs = np.sqrt(omega_unc_bbar[1:]**2 + omega_unc_bbar[:-1]**2)
                                bbar_min = np.min(diffs/sigs)
                                bbar_mean = np.mean(diffs/sigs)

                            else:
                                bbar_min = np.nan
                                bbar_mean = np.nan
                            


                            omega_means_b = np.array(bin_contents['B_omega_means'])
                            omega_unc_b = np.array(bin_contents['B_omega_means_unc'])
                            if not np.isnan(omega_means_b).any() and len(omega_means_b) > 1:
                                diffs = omega_means_b[1:]-omega_means_b[:-1]
                                sigs = np.sqrt(omega_unc_b[1:]**2 + omega_unc_b[:-1]**2)
                                b_min = np.min(diffs/sigs)
                                b_mean = np.mean(diffs/sigs)

                                bin_diffs = np.abs(omega_means_bbar-omega_means_b)
                                bin_diffs_mean = np.mean(bin_diffs)
                                bin_diffs_std = np.max(bin_diffs)/np.min(bin_diffs)

                                if not np.isfinite(bin_diffs_std):
                                    bin_diffs_std = np.nan
                            else:
                                b_min = np.nan
                                b_mean = np.nan
                                bin_diffs_std = np.nan

                            
                        except FileNotFoundError:
                            print(f'warning file {diff_file} not found')
                            bbar_min = np.nan
                            bbar_mean = np.nan
                            b_min = np.nan
                            b_mean = np.nan
                            bin_diffs_std = np.nan

                        deltap0 = data['Fitpar_deltap0']
                        deltap1 = data['Fitpar_deltap1']
                        deltap2 = data['Fitpar_deltap2']

                    
                        deltap0_mc = data['Fitpar_deltap0']
                        deltap1_mc = data['Fitpar_deltap1']
                        deltap2_mc = data['Fitpar_deltap2']

                        deltas = np.array([deltap0, deltap1, deltap2, deltap0_mc, deltap1_mc, deltap2_mc])
                        deltas = np.abs(deltas)

                        if any(deltas > max_deltp): #Ignore insensible result
                            print(f'deltap1: {deltap0},deltap0: {deltap1}, deltap2: {deltap2},tg: {tagging_power}')
                            print(f"Skipping {tagger} with lr={lr}, bs={bs}, nl={nl}, nn={nn} due to high deltap1 or deltap0")
                            continue
                            
                        delta0s.append(deltap0.nominal_value)
                        delta1s.append(deltap1.nominal_value)
                        delta2s.append(deltap2.nominal_value)
                    


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
                                    best_model = json_file_data
                                print(f"Tagger: {tagger}, Seed: {seed}, LR: {lr}, Bs: {bs}, NL: {nl}, NN: {nn}, Tagging Power: {tagging_power.nominal_value}, Ratio: {ratio}, Bbar_min_diff: {bbar_min}, bbar_mean_diff: {bbar_mean}, B_min_diff: {b_min}, b_mean_diff: {b_mean}, bin_diffs_std: {bin_diffs_std}")
                                performances.loc[len(performances)] = [tagger, lr, bs, nl, nn, tagging_power.nominal_value, tagging_power.std_dev, tagging_power_mc.nominal_value, tagging_power_mc.std_dev, bbar_min, bbar_mean, b_min, b_mean, bin_diffs_std]

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
            plot_path = os.path.join(cfg.plot_path, f'{data_type}/hyperparameters_plots/{cfg.cut}/{link}')
            os.makedirs(plot_path, exist_ok=True)

            #save best_models to a json file. Makes it easier to check the calibrations
            with open(os.path.join(plot_path, f'best_models_{link}.json'), 'w') as f:
                json.dump(best_models, f, indent=4, default=str)

            for tagger, decay in tagger_dict.items():
                plot_hyperparams_vs_tagging_power(performances[performances['tagger'] == tagger], os.path.join(plot_path, f'{tagger}_Hyperparams_vs_TaggingPower.png'), len(tagger_input_features[tagger]['features']))

            performances['link'] = link
            full_df = pd.concat([full_df, performances], ignore_index=True)

            # Plot FitPar_deltas as histogramm
            plt.figure(figsize=(12, 6))
            bins = np.linspace(-2, 2, 100)
            delta0s = np.array(delta0s)
            delta1s = np.array(delta1s)
            delta2s = np.array(delta2s)
            all_deltas = np.concatenate([delta0s, delta1s, delta2s])
            
            # plt.hist(all_deltas, bins=bins, alpha=0.5, density=True, label=r'All $\Delta s$')
            plt.hist(delta0s, bins=bins, alpha=0.5, density=True, label=r'$\Delta p_0$')
            plt.hist(delta1s, bins=bins, alpha=0.5, density=True, label=r'$\Delta p_1$')
            plt.hist(delta2s, bins=bins, alpha=0.5, density=True, label=r'$\Delta p_2$')


            for tagger in tagger_dict.keys():
                plt.clf()
                tagger_df = full_df[full_df['tagger'] == tagger]
                plt.hist(tagger_df['bin_diffs_std'], bins=50, alpha=0.5, label=r'',)# range=(0, 1))
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.legend()
                # plt.savefig(os.path.join(plot_path, f'FitPar_deltas.png'))
                plt.savefig(os.path.join(plot_path, f'{tagger}_bbar_diffs.png'))
                plt.close()
                plt.clf()

                for suffix in ['mean_diff', 'min_diff']:
                    tagger_df = full_df[full_df['tagger'] == tagger]
                    min = np.min([np.min(tagger_df['B_' + suffix]), np.min(tagger_df['Bbar_' + suffix])])
                    max = np.max([np.max(tagger_df['B_' + suffix]), np.max(tagger_df['Bbar_' + suffix])])

                    min = -2
                    max = 2

                    bins = np.linspace(min, max, 50)
                    


                    for prefix in ['B_', 'Bbar_']:  

                        plt.hist(tagger_df[prefix + suffix], bins=bins, alpha=0.5, label=f'{prefix}{suffix}',)


                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.legend()
                    # plt.savefig(os.path.join(plot_path, f'FitPar_deltas.png'))
                    plt.savefig(os.path.join(plot_path, f'{tagger}_{suffix}.png'))
                    plt.close()
                    plt.clf()

            filename = os.path.join(path_name, f'candidatedTaggers_{link}.json')
            os.makedirs(os.path.dirname(os.path.dirname(filename)), exist_ok=True)


            with open(filename, 'w') as f:
                json.dump(max_ratios, f, indent=4, default=str)  # `default=str` to handle non-serializable objects
            print(f"Max ratios with link {link} saved to {filename}")

        # Save the full DataFrame to a CSV file
        full_df.to_csv(os.path.join(os.path.dirname(plot_path), 'performances.csv'), index=False)

        
        
        print(f"All results saved to {cfg.outpath}/{cfg.cut}/{data_type}/")

