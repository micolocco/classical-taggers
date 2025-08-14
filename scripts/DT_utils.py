import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
plt.rcParams['text.usetex'] = False # HD cluster has some problems with dvp not found
plt.rcParams.update({'axes.unicode_minus' : False})

def plot_features_byOrigin(data, features, target_path, nbins=100):
    # Plot input features 
    plt.figure(figsize=(100,100))
    for i, col in enumerate(features):
        plt.subplot(10, 7, i + 1)
        # Ranges and names must be adapted
        #plt.hist(data[col][data['particle']==particle_type['OSKaon']], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=)
        #plt.hist(data[col][data['particle']==particle_type['OSMuon']], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['OSElectron']], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSPion']], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSProton']], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
        #plt.hist(data[col][data['particle']==particle_type['SSKaon']], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
       

        #plt.hist(data[col][data['particle']==6], density = True, bins=nbins, label = f"{particle_type[6]}", histtype='step', color='r', )
        if col in nice_names.keys(): #uGly hack
            plt.xlabel(nice_names[col])
            plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
            plt.hist(data[col][data['particle']=='notSamePV'], density = True, bins=nbins, label = f"notSamePV", histtype='step', color='orange', lw=2, range=ranges[col])
        
        else:
            plt.xlabel(col)
            plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, )
            plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, )
            plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, )
            plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, )
            plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, )
            plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, )
            plt.hist(data[col][data['particle']=='notSamePV'], density = True, bins=nbins, label = f"notSamePV", histtype='step', color='orange', lw=2, range=ranges[col])
        plt.legend() 
        plt.tight_layout()
    plt.savefig(f"{target_path}/DT_features_byOrigin.pdf")

def plot_used_features(data, features, target_path, nbins=100):
    plt.figure(figsize=(100,100))
    for i, col in enumerate(features):
        plt.subplot(3, 3, i + 1)
        plt.xlabel(nice_names[col])
        plt.hist(data[col][data['particle']=='OSKaon'], density = True, bins=nbins, label = f"OSKaon", histtype='step', color='m', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='OSMuon'], density = True, bins=nbins, label = f"OSMuon", histtype='step', color='b', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='OSElectron'], density = True, bins=nbins, label = f"OSElectron", histtype='step', color='c', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='SSPion'], density = True, bins=nbins, label = f"SSPion", histtype='step', color='g', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='SSProton'], density = True, bins=nbins, label = f"SSProton", histtype='step', color='y', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='SSKaon'], density = True, bins=nbins, label = f"SSKaon", histtype='step', color='r', lw=2, range=ranges[col])
        plt.hist(data[col][data['particle']=='notSamePV'], density = True, bins=nbins, label = f"notSamePV", histtype='step', color='orange', lw=2, range=ranges[col])
    
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


def old_metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, savepath=None,):      
    '''
    Function to get metrics (in form of a table) for the amount of true VS predicted particle types
    Denominator can be the amount of predicted particles or of true partricle for a specific type
    Each table cell is filled with:
        n(true=possible_particle_A & pred=possible_particle_B) / n(true=possible_particle_A)
        or
        n(true=possible_particle_A & pred=possible_particle_B) / n(pred=possible_particle_B)
    with n=number of cases
    The normalization parameter allows to choose if computing the %s with respect to the predicted (type B) or 
    true particles (type A)
    '''
    from rich.console import Console
    from rich.table import Table
    import numpy as np
    from rich import print as rprint


    console = Console()

    table = Table(show_header=True, title=title)
    table.add_column("True \ Predicted", justify="left", style='cyan')
    #table.add_column("OSKaon", justify="right", style="green")
    #table.add_column("OSMuon", justify="right", style="green")
    #table.add_column("OSElectron", justify="right", style="green")
    #table.add_column("SSPion", justify="right", style="green")
    #table.add_column("SSProton+SSKaon", justify="right", style="green")
    #table.add_column("SSKaon", justify="right", style="green")
    #table.add_column("OSProton", justify="right", style="green")
    for key in possible_particle:
        table.add_column(key, justify="right", style="green")
 
    for particle_A in possible_particle: #for possible_particle in sorted(y_true.unique())
        percVector = []
        for particle_B in possible_particle: # for particle in possible_particle:
            if particle_B in sorted(np.unique(y_predicted)):
                if normalization == 'predicted':
                    denom = len(y_predicted[y_predicted == particle_B])
                else:
                    denom = len(y_true[y_true == particle_A])
                
                unique_values, counts = np.unique(y_true[y_predicted == particle_B] == particle_A, return_counts=True)

                # Check if 'True' exists in unique_values before accessing counts[1]
                if True in unique_values:
                    true_count_index = np.where(unique_values == True)[0][0]
                    true_count = counts[true_count_index]
                    percentage = (true_count / denom) * 100
                    percVector.append("{:.2f}".format(percentage))
                else:
                    percVector.append("0.00")  # No 'True' values, so 0% match
            else:
                percVector.append("Not predicted")

        table.add_row(particle_A, *percVector)
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)
    else:
        console.print(table)

def metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, savepath=None, unify_classes=None):
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

    console = Console()

    if unify_classes is None:
        unify_classes = []

    # Unify the specified classes into "UnifiedOthers"
    def unify_labels(labels, unify_classes, unified_label="UnifiedOthers"):
        return np.array([unified_label if label in unify_classes else label for label in labels])

    y_true_unified = unify_labels(y_true, unify_classes)
    y_predicted_unified = unify_labels(y_predicted, unify_classes)

    # Adjust the possible_particle list to include the unified class
    possible_particle_unified = [p for p in possible_particle if p not in unify_classes]
    if "UnifiedOthers" not in possible_particle_unified:
        possible_particle_unified.append("UnifiedOthers")

    # Initialize the table
    table = Table(show_header=True, title=title)
    table.add_column("True \\ Predicted", justify="left", style='cyan')
    for key in possible_particle_unified:
        table.add_column(key, justify="right", style="green")

    # Compute percentages
    for particle_A in possible_particle_unified:
        percVector = []
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
                    percVector.append("{:.2f}".format(percentage))
                else:
                    percVector.append("0.00")  # No 'True' values, so 0% match
            else:
                percVector.append("Not predicted")

        table.add_row(particle_A, *percVector)

    # Print or save the table
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)
    else:
        console.print(table)



def new_metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, balanced=False, savepath=None, uncertainty=False):
    from rich.console import Console
    from rich.table import Table
    import numpy as np
    from rich import print as rprint


    console = Console()
    table = Table(show_header=True, title=title)
    table.add_column("True \ Predicted", justify="left", style='cyan')

    # Initialize weights
    weight = np.ones_like(y_true, dtype=float)
    if balanced:
        for particle_truth in possible_particle:
            mask = (y_true == particle_truth)
            weight[mask] = np.nan_to_num(1 / np.sum(mask))

    # Add columns
    for key in possible_particle:
        table.add_column(key, justify="right", style="green")

    # Precompute counts
    true_counts = {particle: np.sum(weight * (y_true == particle)) for particle in possible_particle}
    predicted_counts = {particle: np.sum(weight * (y_predicted == particle)) for particle in possible_particle}

    # Compute metrics
    for particle_truth in possible_particle:
        perc_vector = []
        for particle_prediction in possible_particle:
            k = np.sum(weight * ((y_predicted == particle_prediction) & (y_true == particle_truth)))
            n = predicted_counts[particle_prediction] if normalization == 'predicted' else true_counts[particle_truth]
            if n > 0:
                if uncertainty:
                    perc_vector.append(f"{(k/n)*100:.4f} ± {(((k/n)*(1-k/n))/n)*100:.4f}")
                else:
                    perc_vector.append(f"{(k/n)*100:.2f}")
            else:
                perc_vector.append("Not predicted")
        table.add_row(particle_truth, *perc_vector)

    # Output table
    # console.print(table)
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)

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
'''
def metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, balanced=False, savepath=None, uncertainty=False):
    
    #Function to get metrics (in form of a table) for the amount of true VS predicted particle types
    #Denominator can be the amount of predicted particles or of true partricle for a specific type
    #Each table cell is filled with:
    #    n(true=possible_particle_A & pred=possible_particle_B) / n(true=possible_particle_A)
    #    or
    #    n(true=possible_particle_A & pred=possible_particle_B) / n(pred=possible_particle_B)
    #with n=number of cases
    #The normalization parameter allows to choose if computing the %s with respect to the predicted (type B) or 
    #true particles (type A)
    
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    import numpy as np

    console = Console()

    table = Table(show_header=True, title=title)
    table.add_column("True \ Predicted", justify="left", style='cyan')

    weight = 1
    if balanced:
        weight = np.sum([np.nan_to_num((y_true == particle_truth) / np.sum(y_true == particle_truth)) for particle_truth in possible_particle], axis=0)

    for key in possible_particle:
        table.add_column(key, justify="right", style="green")
    for particle_truth in possible_particle: #for possible_particle in sorted(y_true.unique())
        percVector = []
        for particle_prediction in possible_particle: # for particle in possible_particle:
            if particle_prediction in np.unique(y_predicted):
                k = np.sum(((y_predicted == particle_prediction) & (y_true == particle_truth)) * weight)
                if normalization=='predicted':
                    n = np.sum((y_predicted == particle_prediction) * weight)
                else:
                    n = np.sum((y_true == particle_truth) * weight)
                if uncertainty:
                    percVector.append("{:.4f} ± {:.4f}".format((k/n)*100, (((k/n)*(1-k/n))/n)*100)) # Binominal variance of the efficiency from https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
                    # percVector.append("{:.4f} ± {:.4f}".format((k/n)*100, ((((k+1)*(k+2))/((n+2)*(n+3)))-(((k+1)**2)/((n+2)**2)))*100)) # Bayesian variance of the efficiency from https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
                else:
                    percVector.append("{:.2f}".format((k/n)*100))
            else:
                percVector.append("Not predicted")
        table.add_row(particle_truth, *percVector)
    console.print(table)
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)
'''

