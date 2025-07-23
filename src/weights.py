import json
import numpy as np
from tqdm import tqdm
import os
from itertools import product
import datetime
import pytz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["WANDB_SILENT"] = "true"

dataset = ['adult']
feature = ['sex']
axs = ["pos_neg"]
implies = ['Godel']
p_mean = [1]
aggregator_deviation = [2]

ws = np.linspace(1, 3, num=21)
# ws = [0]

with open('./src/KnowledgeBaseAxioms.json', 'rt') as json_file:
    config = json.load(json_file)

with open('./src/elapsed.txt', 'rt') as f_elapsed:
    elapsed = [e.strip() for e in f_elapsed.readlines()]

for ds, f, i, p, w, a, d in tqdm(list(product(dataset, feature, implies, p_mean, ws, axs, aggregator_deviation))):

    for axiom in config:
        if axiom['name'] in ["axiom_positive_class", "axiom_negative_class"]:
            # Always train with class axiom weighted 1
            axiom['infos']['training'] = True
            axiom['infos']['weight'] = 1
        elif axiom['name'] in ["axiom_demographic_parity_pos", 'axiom_demographic_parity_neg']:
            axiom['infos']['training'] = not w == 0
            axiom['infos']['weight'] = w
        else:
            axiom['infos']['training'] = False
            axiom['infos']['weight'] = 0

    with open('./src/KnowledgeBaseAxioms.json', 'wt') as json_file:
        json.dump(config, json_file)

    current = f"{ds}, {f}, {a} => implies: {i}, p_mean: {p}, weight: {w}, aggregator:{d}" 
    now_utc = datetime.datetime.now(pytz.utc)
    timezone = pytz.timezone('Europe/Rome')
    now_local = now_utc.astimezone(timezone)

    tqdm.write(f"{now_local.strftime('%Y-%m-%d %H:%M:%S')} - {current}")
    if not current in elapsed:
        os.system(f"/usr/bin/python3 /workspaces/ml-fairness-thesis/src/wan.py {ds} {f} {i} {p} {d}")
        with open('./src/elapsed.txt', 'at') as f_elapsed:
            f_elapsed.write(f"{current}\n")