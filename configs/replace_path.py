import os
'''
Script for generating all the output lines to be inserted in the Snakefile when all the configurations (in yaml format) in the configs folder are wanted
python replace_path.py
'''

# Define the directory containing the YAML files
config_dir = '/home/molocco/classical-taggers/configs'
new_paths = []
# Define the original path with the placeholder to be replaced
original_path = 'savedModels/withUT_MC_2024/Bu2JpsiKst/OSKaon/cut_DT_unbalanced_minGain_maxDepth_SSKSSP_withOrigin/2/lr0.1_bs32_complex/mistag_Training.pdf'
outputfile = '/home/molocco/classical-taggers/generated_paths.txt'
# List all YAML files in the config directory
yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and f != 'config_test.yaml']
print(f"Added paths {outputfile}: \n")
# Iterate over each YAML file and replace the placeholder in the path
for yaml_file in yaml_files:
    # Extract the base name without the .yaml extension
    base_name = os.path.splitext(yaml_file)[0]
    # Replace the placeholder in the original path with the base name
    new_path = original_path.replace('lr0.1_bs32_complex', base_name)
    new_paths.append(new_path)
    print(new_path)

with open(outputfile, 'a') as f:
    for path in new_paths:
        f.write(f"{path}\n")
