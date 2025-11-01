from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["WANDB_SILENT"] = "true"

datasets = [
    {'german': ['age', 'sex']}, 
    {'compas': ['sex', 'race']}, 
    {'adult': ['race', 'sex']}
]

for dataset in tqdm(datasets):
    name = list(dataset.keys())[0]
    features = list(dataset.values())[0]
    for f in features:
        tqdm.write(f"{name}, {f}")
        os.system(f"/usr/bin/python3 /workspaces/ml-fairness-thesis/src/mlp.py {name} {f}")