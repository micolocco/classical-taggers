import yaml
import os
from itertools import product
import numpy as np

def generate_yaml_file(config, use_alpha=False, use_BN=False):
        # Create directory if it doesn't exist
    # if not os.path.exists('../configs'):
    #     os.makedirs('../configs')

    file_name = f'model_configs/lr{config["learning_rate"]}_bs{config["train_batch_size"]}_nL{config["numlayers"]}_nN{config["numneurons"]}.yaml'

    if use_alpha:
        file_name = file_name.replace('.yaml', f'_alpha{config["alpha"]}.yaml')
    if use_BN:
        file_name = file_name.replace('.yaml', f'_BN.yaml')

    with open(file_name, 'w') as file:
        yaml.dump(config, file)
    print(f"Generated {file_name}")

def generate_architecture_yaml_file(config, use_BN=False):
    # Create directory if it doesn't exist
    # if not os.path.exists('../configs'):
    #     os.makedirs('../configs')

    file_name = f'NNarchitectures/nL{config["numlayers"]}_nN{config["numneurons"]}_dp{config["dropout"]}'
    if use_BN:
        file_name += f'_BN'
    file_name += '.yaml'

    if not os.path.isfile(file_name):
        nnShape = np.ones(config['numlayers'], dtype=int)*config['numneurons']
        nnShape = np.append(nnShape, 1)

        with open(file_name, 'w') as file:
            file.write('architecture:\n')
            file.write('  - type: Linear\n')
            file.write(f'    params: {{in_features: len(self.features), out_features: {config["numneurons"]}}}\n')
            for i in range(0, len(nnShape)-1):
                if use_BN:
                    file.write(f'  - type: BatchNorm1d\n')
                    file.write(f'    params: {{num_features: {nnShape[i]}}}\n')
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

    for use_BN in [False, True]:
        for use_alpha in [False, True]:
            lists = []
            for key, value in intervals.items():
                if key != '-alpha' or use_alpha:
                    if isinstance(value, list):
                        lists.append(value)
                    else:
                        lists.append([value])


            combinations = product(*lists)

            result = list(combinations)

            # Print result
            for idx, comb in enumerate(result):        
                config = {}
                keys = intervals.keys()
                if not use_alpha:
                    keys = [key for key in keys if key != '-alpha']
                for i, key in enumerate(keys):
                    config[key[1:]] = comb[i]
                config['use_batch_norm'] = use_BN

                # print(config)

                #Generate config file
                generate_yaml_file(config, use_alpha=use_alpha, use_BN=use_BN)

                #Generate architecture file if not already present
                if not use_alpha: #Agnostic to alpha -> Only needs to be generated once
                    generate_architecture_yaml_file(config, use_BN=use_BN)
