
def metric_table(y_true, y_predicted, possible_particle, title='Versus True', normalization=None, balanced=False, savepath=None, uncertainty=False):
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

