

def metric_table(y_true, y_predicted, particle_type_dict, true_type_dict=None, title='Versus True', normalization=None, balanced = False, savepath=None, uncertainty=False):      
    '''
    Function to get metrics (in form of a table) for the amount of true VS predicted particle types
    Denominator can be the amount of predicted particles or of true partricle for a specific type
    Each table cell is filled with:
        n( pred=particle_type_A & true=particle_type_B) / n(true=particle_type_A)
    with n=number of cases
    The normalization parameter allows to choose if computing the %s with respect to the predicted (type B) or 
    true particles (type A)
    '''
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    import numpy as np

    console = Console()

    table = Table(show_header=True, title=title)
    table.add_column("True \ Predicted", justify="left", style='cyan')
        
    if not true_type_dict:
        true_type_dict = particle_type_dict
        
    weight = 1
    if balanced:
        weight = np.sum([np.nan_to_num((y_true == value) / np.sum(y_true == value)) for value in true_type_dict.values()], axis=0)
    
    for key in particle_type_dict.keys():
        table.add_column(key, justify="right", style="green")
    for key, value in true_type_dict.items():
        percVector = []
        for prediction_ID in np.arange(list(particle_type_dict.values())[0], list(particle_type_dict.values())[-1]+1):
            if prediction_ID in np.unique(y_predicted):
                k = np.sum((y_predicted == prediction_ID) * (y_true == value) * weight)
                if normalization=='predicted':
                    n = np.sum((y_predicted == prediction_ID) * weight)
                else:
                    n = np.sum((y_true == value) * weight)
                if uncertainty:
                    percVector.append("{:.4f} ± {:.4f}".format((k/n)*100, (((k/n)*(1-k/n))/n)*100)) # Binominal variance of the efficiency from https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
                    # percVector.append("{:.4f} ± {:.4f}".format((k/n)*100, ((((k+1)*(k+2))/((n+2)*(n+3)))-(((k+1)**2)/((n+2)**2)))*100)) # Bayesian variance of the efficiency from https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
                else:
                    percVector.append("{:.2f}".format((k/n)*100))
            else:
                percVector.append("Not predicted")
        table.add_row(key, *percVector)
    console.print(table)
    if savepath:
        with open(savepath, "w") as f:
            rprint(table, file=f)

