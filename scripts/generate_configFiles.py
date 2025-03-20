import yaml
import os
import argparse
from pprint import pprint

'''
This script creates a yaml file for every possible combination of hyperparameters given to the function `generate_yaml_file`. 
The different hyperparameter values (like learning rates, batch size etc) are specified thorugh arrays. 
The yaml file will be located in a `configs` folder created at the path specified trhough the argument --repoPath.
'''
def generate_yaml_file(repo, learning_rate, train_batch_size, architecture, min_delta):
    config = {
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'architecture': architecture,
        # Hardcoded values. If grid search is needed, move them belove and set them as function arguments
        'train_val_split': 0.6,
        'n_epochs': 500,
        'patience': 25,
        'min_delta': min_delta,
        'activation_function': 'ELU',
    }

    # Create directory if it doesn't exist
    if not os.path.exists(f'{repo}/configs'):
        os.makedirs(f'{repo}/configs')

    file_name = f'configs/lr{learning_rate}_bs{train_batch_size}_{architecture}_dm{dm}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(config, file)
    print(f"Generated {file_name}")

# Define array of hyperparameters to iterate over
learning_rates = [0.001, 0.01, 0.1]
train_batch_sizes = [32, 128, 1024, 2048]
architectures = ['simple', 'complex']
min_delta = [0.0, 0.0001, 0.001, 0.01]
#train_val_split = 0.6
#n_epochs = 500
#patience = 25 #75 #100
#min_delta = 0.00
#activation_function = 'ELU' #ReLU, ELU
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script creates a yaml file for every possible combination of hyperparameters given to the function `generate_yaml_file`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--repoPath', help='Path to where the outputs must be stored', type=str) 
    cfg = parser.parse_args()
    # Generate YAML files for all combinations of learning rates and batch sizes
    i = 0
    for lr in learning_rates:
        for bs in train_batch_sizes:
            for ar in architectures:
                for dm in min_delta:
                    generate_yaml_file(repoPath=cfg.repoPath, learning_rate=lr, train_batch_size=bs, architecture=a, min_delta=dm)
                    i+=1
    print(f'Generated config {i} files')
