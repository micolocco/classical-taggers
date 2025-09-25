import torch
from scripts.NNModel import NeuralNetwork
import scripts.pyTorchTraining as pyTrain
from os.path import join
import yaml
import json
from IPython import embed

pre_path = "/ceph/users/molocco/FlavourTagging/MC/savedModels/withUT_MC_2024/"
cut = 'allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN'
repo = '/home/molocco/classical-taggers'

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
    seed = int(data[tagger]['seed'])
    lr = float(data[tagger]['learning_rate'])
    bs = int(data[tagger]['batch_size'])
    arch = data[tagger]['architecture']
    dm = float(data[tagger]['min_delta'])
    config = f'lr{lr}_bs{bs}_{arch}_dm{dm}'
    with open(f'{repo}/configs/{config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    decay = taggers_dict[tagger]
    pre_path_full = join(pre_path, f"{decay}/{tagger}/{cut}")

    model_path = join(pre_path_full, f"{seed}/{config}")
    model= f"{model_path}/model.pth"
    features = pyTrain.get_features(tagger=tagger, yaml_file='union_PROBNN', repo_path=repo)
    embed()
    bestModel = torch.load(model)
    model.eval() 
    # Create a dummy input matching the input shape (len(features))
    dummy_input = torch.randn(1, len(features))

    # Export to ONNX
    torch.onnx.export(
        model.NN,                            # Export the inner nn.Sequential
        dummy_input,                         # Dummy input
        f"{tagger}_model.onnx",                        # Output file name
        input_names=["float_input"],         # Input name
        output_names=["output_probability"], # Output name
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "output_probability": {0: "batch_size"},
        },
        # opset_version=11                     # Choose your ONNX opset version
    )
    embed()