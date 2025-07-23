import json
from itertools import product, chain
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["WANDB_SILENT"] = "true"

datasets = [
    {'german': ['age', 'sex']}, 
    {'compas': ['sex', 'race']}, 
    {'adult': ['race', 'sex']}
]

a = [
    ["axiom_demographic_parity"],
    ["axiom_demographic_parity_pos", "axiom_demographic_parity_neg"],
    ["axiom_demographic_parity_sim"],
]

ws = [1, 2, 3]

only_classification = [([], 0)]

with open('./src/KnowledgeBaseAxioms.json', 'rt') as f:
    config = json.load(f)

with open('./src/elapsed.txt', 'rt') as f_elapsed:
    elapsed = [e.strip() for e in f_elapsed.readlines()]

for ax, w in tqdm(chain(only_classification, product(a, ws))):

    for axiom in config:
        if axiom['name'] in ["axiom_positive_class", "axiom_negative_class"]:
            # Always train with class axiom weighted 1
            axiom['infos']['training'] = True
            axiom['infos']['weight'] = 1
        elif axiom['name'] in ax:
            axiom['infos']['training'] = True
            axiom['infos']['weight'] = w
        else:
            axiom['infos']['training'] = False
            axiom['infos']['weight'] = 0

    with open('./src/KnowledgeBaseAxioms.json', 'wt') as f:
        json.dump(config, f)

    for dataset in datasets:
        name = list(dataset.keys())[0]
        features = list(dataset.values())[0]
        for f in features:
            current = f"{ax}, {w}, {name}, {f}"
            tqdm.write(current)
            if not current in elapsed:
                os.system(f"/usr/bin/python3 /workspaces/ml-fairness-thesis/src/wan.py {name} {f}")
                with open('./src/elapsed.txt', 'at') as f_elapsed:
                    f_elapsed.write(f"{current}\n")