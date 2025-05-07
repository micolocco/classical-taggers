import yaml
import os
from itertools import product
import numpy as np

def generate_yaml_file(config):
        # Create directory if it doesn't exist
    # if not os.path.exists('../configs'):
    #     os.makedirs('../configs')

    file_name = f'configs/lr{config["learning_rate"]}_bs{config["train_batch_size"]}_nL{config["numlayers"]}_nN{config["numneurons"]}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(config, file)
    print(f"Generated {file_name}")

def generate_architecture_yaml_file(config):
    # Create directory if it doesn't exist
    # if not os.path.exists('../configs'):
    #     os.makedirs('../configs')

    file_name = f'NNarchitectures/nL{config["numlayers"]}_nN{config["numneurons"]}_dp{config["dropout"]}.yaml'
    if not os.path.isfile('configs'):
        nnShape = np.ones(config['numlayers'], dtype=int)*config['numneurons']
        nnShape = np.append(nnShape, 1)

        with open(file_name, 'w') as file:
            file.write('architecture:\n')
            file.write('  - type: Linear\n')
            file.write(f'    params: {{in_features: len(self.features), out_features: {config["numneurons"]}}}\n')
            for i in range(0, len(nnShape)-1):
                file.write(f'  - type: Dropout\n')
                file.write(f'    params: {{p: {config["dropout"]}}}\n')
                file.write(f'  - type: ELU\n')
                file.write(f'    params: {{}}\n')
                
                file.write(f'  - type: Linear\n')
                file.write(f'    params: {{in_features: {nnShape[i]}, out_features: {nnShape[i+1]}}}\n')
            file.write(f'  - type: Sigmoid\n')
            file.write(f'    params: {{}}\n')


        print(f"Generated {file_name}")

if __name__ == '__main__':
    with open(f'configs/hyperpar_intervals.yaml', 'r') as file:
        intervals = yaml.safe_load(file)

    lists = []
    for value in intervals.values():
        if isinstance(value, list):
            lists.append(value)
        else:
            lists.append([value])


    combinations = product(*lists)

    result = list(combinations)

    # Print result
    for idx, comb in enumerate(result):        
        config = {}
        for i, key in enumerate(intervals.keys()):
            config[key[1:]] = comb[i]

        #Generate config file
        generate_yaml_file(config)
        #Generate architecture file if not already present
        generate_architecture_yaml_file(config)
