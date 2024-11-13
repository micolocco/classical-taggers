

def metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None):      
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
    console.print(table)

