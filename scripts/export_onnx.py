import torch
from scripts.NNModel import NeuralNetwork
import scripts.pyTorchTraining as pyTrain
from os.path import join
import json
from IPython import embed
import os 
import onnx

pre_path = "/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/"
cut = 'allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN'
repo = '/home/molocco/classical-taggers'

os.makedirs(f'{pre_path}/onnx_models', exist_ok=True)
taggers_dict = {
    'OSKaon': 'Bu2JpsiK',
    'OSElectron': 'Bu2JpsiK',
    'OSMuon': 'Bu2JpsiK',
    'SSPion': 'Bd2JpsiKst',
    'SSProton': 'Bd2JpsiKst',
    'SSKaon': 'Bs2DsPi'
}
#Load the best model (ie with the lowest training loss) and evaluate it on the test set
json_file=f'candidatedTaggers_logit.json'
#Read the best tagger candidate config from json file with the best hyperparameter combination
with open(f'{repo}/best_tagger_candidates/allBKGCAT_notSamePV_noOSP_SSK_balanced/{json_file}', 'r') as f:
    data = json.load(f)

for tagger in taggers_dict.keys():
    print(f"Exporting {tagger} to ONNX format")
    seed = int(data[tagger]['seed'])
    lr = float(data[tagger]['learning_rate'])
    bs = int(data[tagger]['batch_size'])
    arch = data[tagger]['architecture']
    dm = float(data[tagger]['min_delta'])
    config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'
    #with open(f'{repo}/configs/{config}.yaml', 'r') as file:
    #    config = yaml.safe_load(file)
    decay = taggers_dict[tagger]
    pre_path_full = join(pre_path, f"{decay}/{tagger}/{cut}")

    model_path = join(pre_path_full, f"{seed}/{config}/asym_level2")
    model= f"{model_path}/model.pth"
    features = pyTrain.get_features(tagger=tagger, yaml_file='union_PROBNN', repo_path=repo)
    bestModel = torch.load(model, weights_only=False)
    bestModel.eval() 
    # Create a dummy input matching the input shape (len(features))
    dummy_input = torch.randn(1, len(features))
    
    # Export to ONNX
    torch.onnx.export(
        bestModel.NN,                            # Export the inner nn.Sequential
        dummy_input,                         # Dummy input
        f"{pre_path}/onnx_models/{tagger}_model.onnx",                        # Output file name
        input_names=["float_input"],         # Input name
        output_names=["output_probability"], # Output name
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "output_probability": {0: "batch_size"},
        },
        # opset_version=11                     # Choose your ONNX opset version
    )
     # Check
    

    # Load the ONNX model
    model_onnx = onnx.load(f"{pre_path}/onnx_models/{tagger}_model.onnx")
    # Inspect input and output
    print("Inputs:")
    for input in model_onnx.graph.input:
        print(f"  name: {input.name}")
        print(f"  type: {input.type}")

    print("\nOutputs:")
    for output in model_onnx.graph.output:
        print(f"  name: {output.name}")
        print(f"  type: {output.type}")
