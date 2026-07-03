import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})


import matplotlib.pyplot as plt
import os
from IPython import embed

def plot_features_byOrigin(data, features, target_path, nbins=100):
    os.makedirs(target_path, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 14,          # base
        "axes.labelsize": 14,     # axis labels
        "xtick.labelsize": 14,    # tick label sizes
        "ytick.labelsize": 14,
        "legend.fontsize": 12,    # legend text
        "lines.linewidth": 1,   # slightly thicker for clarity
    })
    
    colors = {
        "notSamePV": "orange",
        "OSKaon": "violet",
        "OSMuon": "blue",
        "OSElectron": "cyan",
        "SSPion": "green",
        "SSProton": "brown",
        "SSKaon": "red"
    }

    features_DT_used = [
        "B_Tr_T_PROBNN_E",
        "B_Tr_T_PROBNN_MU",
        "B_Tr_T_diff_z",
        "B_Tr_T_PROBNN_PI",
        "B_Tr_T_PIDK",
        "B_Tr_T_IPChi2BVTX",
        "B_Tr_T_PROBNN_K",
        "B_Tr_T_OWNPVIPCHI2",
        "B_Tr_T_PROBNN_P"
        ]
    for col in features:
        fig, ax = plt.subplots(figsize=(5.6, 3.9))  # Small, proportional figure size
        
        xlabel = nice_names.get(col, col)
        rng = ranges.get(col, None)
        
        for particle, color in colors.items():
            ax.hist(data[col][data['particle'] == particle],
                    density=True,
                    bins=nbins,
                    label=particle,
                    histtype='step',
                    color=color,
                    lw=1.2,
                    range=rng)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalised number of tracks")
        ax.legend(frameon=False, loc='best')
        if col in features_DT_used:
            ax.set_yscale('log')  # Set y-axis to logarithmic scale for better visibility
        plt.tight_layout(pad=0.2)
        if not os.path.exists(f"{target_path}/feature_plots"):
            os.makedirs(f"{target_path}/feature_plots")
    
        plt.savefig(f"{target_path}/feature_plots/{col}_byOrigin.pdf", bbox_inches='tight', transparent=False)
        plt.close(fig)





def plot_used_features(data, features, target_path, nbins=100):
    plt.figure(figsize=(24,25))
    for i, col in enumerate(features):
        plt.subplot(int(np.ceil(np.sqrt(len(features)))), int(np.ceil(np.sqrt(len(features)))), i + 1)
        plt.xlabel(nice_names.get(col, col))
        range = ranges.get(col, (data[col].min(), data[col].max()))
        plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=range)
        plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=range)
        plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=range)
        plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=range)
        plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=range)
        plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=range)
        plt.hist(data[col][data['particle']=='notSamePV'], density = True, bins=nbins, label = f"notSamePV", histtype='step', color='orange', lw=2, range=range)
    
    plt.legend() 
    plt.tight_layout()
    plt.savefig(f"{target_path}/onlyUsed_DT_features_byOrigin.pdf")

def plot_features_byParticle(data, features, nbins=100):
   
    plt.figure(figsize=(100,100))
    for i, col in enumerate(features):
        plt.subplot(10, 8, i + 1)
        #plt.hist(data[col][data.B_Tr_T_absID==321], density = True, bins=nbins, label = f"Kaon", histtype='step', color='m', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==13], density = True, bins=nbins, label = f"Muon", histtype='step', color='b', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==11], density = True, bins=nbins, label = f"Electron", histtype='step', color='c', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==211], density = True, bins=nbins, label = f"Pion", histtype='step', color='g', lw=2, range=ranges[col])
        #plt.hist(data[col][data.B_Tr_T_absID==2212], density = True, bins=nbins, label = f"Proton", histtype='step', color='y', lw=2, range=ranges[col])
        plt.hist(data[col][data.B_Tr_T_absID==321], density = True, bins=nbins, label = f"Kaon", histtype='step', color='m', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==13], density = True, bins=nbins, label = f"Muon", histtype='step', color='b', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==11], density = True, bins=nbins, label = f"Electron", histtype='step', color='c', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==211], density = True, bins=nbins, label = f"Pion", histtype='step', color='g', lw=2, )
        plt.hist(data[col][data.B_Tr_T_absID==2212], density = True, bins=nbins, label = f"Proton", histtype='step', color='y', lw=2, )
        #plt.hist(data[col][data['particle']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys():
            plt.xlabel(nice_names[col])
        else:
            plt.xlabel(col)
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{cfg.target_path}/DT_features_byParticle.pdf")

def plot_features_bySign(data, features, nbins=100):
    keys_list = pd.unique(data['particle']).tolist()
    print(f'Particle: {keys_list}')
    # Plot input features 
    for particle in keys_list:
        plt.figure(figsize=(100,100))
        for i, col in enumerate(features):
            plt.subplot(10, 8, i + 1)
            plt.xlabel(col)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==1) & (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2)
            #plt.hist(data[col][(data['particle']==particle) & (abs(data['B_Tr_T_MC_MOTHER_ID'])==5) & (data['sign_tag']==-1) &  (data['B_Tr_T_Charge']==1)], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==1) ], density = True, bins=nbins, label = f"sign=1", histtype='step', color='m', lw=2,)
            plt.hist(data[col][(data['particle']==particle) & (data['sign_tag']==-1) ], density = True, bins=nbins, label = f"sign=-1", histtype='step', color='b', lw=2)  
            plt.legend() 
            #plt.yscale('log')
        plt.tight_layout()
        plt.title(f'{particle}')
        plt.savefig(f"{cfg.target_path}/DT_features_{particle}.pdf")
        print(f"Saved plot {cfg.target_path}/DT_features_{particle}.pdf")

def count_BKGCAT(df):
    # Assuming you have a specific particle type in mind, like "OSElectron"
    particle_list = pd.unique(df['particle']).tolist()
    for particle in particle_list:
        # Filter the DataFrame for the specified particle
        df_particle = df[df['particle'] == particle]
        # Calculate the percentage of each B_BKGCAT value
        bkgcat_counts = df_particle['B_BKGCAT'].value_counts(normalize=True) * 100
        # Print the results
        print(f"Percentage distribution of B_BKGCAT values for {particle}:")
        for bkgcat, percentage in bkgcat_counts.items():
            print(f"B_BKGCAT {bkgcat}: {percentage:.2f}%")
        print('\n')

def metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, savepath=None, unify_classes=None, create_heatmap=True):
    '''
    Function to get metrics (in form of a table) for the amount of true VS predicted particle types.
    Denominator can be the amount of predicted particles or of true particles for a specific type.
    The normalization parameter allows to choose if computing the %s with respect to the predicted (type B) or 
    true particles (type A).
    '''
    from rich.console import Console
    from rich.table import Table
    import numpy as np
    from rich import print as rprint

    console = Console(width = 400)  # Set console width for better table display, is saved to file anyway

    if unify_classes is None:
        unify_classes = []

    # Unify the specified classes into "UnifiedOthers"
    def unify_labels(labels, unify_classes, unified_label="UnifiedOthers"):
        return np.array([unified_label if label in unify_classes else label for label in labels])

    y_true_unified = unify_labels(y_true, unify_classes)
    y_predicted_unified = unify_labels(y_predicted, unify_classes)

    # Adjust the possible_particle list to include the unified class
    possible_particle_unified = [p for p in possible_particle if p not in unify_classes]
    if "UnifiedOthers" not in possible_particle_unified and unify_classes != []:
        possible_particle_unified.append("UnifiedOthers")

    # Initialize the table
    table = Table(show_header=True, title=title, expand=True)
    table.add_column("True \\ Predicted", justify="left", style='cyan')
    for key in possible_particle_unified:
        table.add_column(key, justify="right", style="green")

    df_confusion = pd.DataFrame(columns=["True", "Predicted", "Count"])

    # Compute percentages
    all_perc_Vectors = []
    for particle_A in possible_particle_unified:
        perc_Vector = []
        for particle_B in possible_particle_unified:
            if particle_B in sorted(np.unique(y_predicted_unified)):
                if normalization == 'predicted':
                    denom = len(y_predicted_unified[y_predicted_unified == particle_B])
                else:
                    denom = len(y_true_unified[y_true_unified == particle_A])

                unique_values, counts = np.unique(y_true_unified[y_predicted_unified == particle_B] == particle_A, return_counts=True)

                # Check if 'True' exists in unique_values before accessing counts
                if True in unique_values:
                    true_count_index = np.where(unique_values == True)[0][0]
                    true_count = counts[true_count_index]
                    percentage = (true_count / denom) * 100
                    perc_Vector.append("{:.2f}".format(percentage))
                else:
                    perc_Vector.append("0.00")  # No 'True' values, so 0% match

                df_confusion[len(df_confusion)] = [particle_A, particle_B, perc_Vector[-1]]
            else:
                perc_Vector.append("Not predicted")
        all_perc_Vectors.append(perc_Vector)
        table.add_row(particle_A, *perc_Vector)

    if create_heatmap:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a heatmap from the percentage vectors
        heatmap_data = np.array([[float(value) if value != "Not predicted" else 0 for value in perc_Vector] for perc_Vector in all_perc_Vectors])
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=possible_particle_unified, yticklabels=possible_particle_unified, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath.replace(".txt", "_heatmap.pdf"))

    



    # Print or save the table
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)

        df_confusion.to_csv(savepath.replace(".txt", ".csv"), index=False)
    else:
        console.print(table)


def get_decision_paths(clf, feature_names):
    """
    Recursively traverse the tree to extract decision paths.

    Returns:
        A list of tuples: (list of conditions, predicted_class)
    """
    tree_ = clf.tree_
    paths = []
    
    def recurse(node, current_conditions):
        # If the node is not a leaf
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            # Left child: condition is feature <= threshold
            left_conditions = current_conditions + [f"({feature} <= {threshold:.4f})"]
            recurse(tree_.children_left[node], left_conditions)
            # Right child: condition is feature > threshold
            right_conditions = current_conditions + [f"({feature} > {threshold:.4f})"]
            recurse(tree_.children_right[node], right_conditions)
        else:
            # Leaf node: get the class label for the node.
            value = tree_.value[node]
            class_index = value.argmax()
            class_label = clf.classes_[class_index]
            current_conditions.append('(B_Tr_T_Origin_Flag !=0)')
            paths.append((current_conditions, class_label))
    
    recurse(0, [])
    return paths

