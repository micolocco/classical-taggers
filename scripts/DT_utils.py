

def metric_table(y_true, y_predicted, particle_type_dict, title='Versus True', normalization=None):      
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
    import numpy as np

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
    for key in particle_type_dict.keys():
        table.add_column(key, justify="right", style="green")
    for key, value in particle_type_dict.items():
        percVector = []
        for prediction_ID in np.arange(list(particle_type_dict.values())[0], list(particle_type_dict.values())[-1]+1):
            if prediction_ID in np.unique(y_predicted):
                if normalization=='predicted':
                    denom=len(y_predicted[y_predicted==prediction_ID])
                else:
                    denom=len(y_true[y_true==value])
                percVector.append("{:.2f}".format(((np.unique(y_true[y_predicted==prediction_ID]==value,return_counts=True)[1][1])/denom)*100))
            else:
                percVector.append("Not predicted")
        table.add_row(key, *percVector)
    console.print(table)

