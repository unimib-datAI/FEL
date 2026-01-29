import sys

sys.path.append("../")
from utils import StandardScaleData_ExcludingFeature_simple, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import numpy as np
import pandas as pd
import yaml
import argparse
from metrics_plots import plot_group_metrics

set_seed(42)


def prepare_dataset(df, target_variable,  protected_attribute):
    Y = df[target_variable].to_numpy()
    X = df.drop(columns=[target_variable]).to_numpy()
    X, _ = StandardScaleData_ExcludingFeature_simple(X, protected_attribute['index'])
    return df, X, Y


def load_model(kb, filepath: str):
    data = np.load(filepath, allow_pickle=True)
    vars = kb.trainable_variables
    for i, v in enumerate(vars):
        key = f"var_{i}"
        if key not in data:
            raise KeyError(f"Missing weight '{key}' in {filepath}")
        v.assign(data[key])
    return kb


def inference(kb, X):
    preds = kb._oracle.predict(X)
    preds = np.asarray(preds).reshape(-1)
    print("Sample predictions:", preds)
    return preds


def main(args):
    with open(args.config, 'r') as f:
        conf =  yaml.safe_load(f)

    df = pd.read_csv(args.dataset)

    protected_attribute = {
        'name': conf["data"]["sensitive_feature"],
        'index': df.columns.get_loc(conf["data"]["sensitive_feature"]),
        'privileged': conf["data"]["protected_values"]["privileged"],
        'unprivileged': conf["data"]["protected_values"]["unprivileged"]
    }
    label_map = {
        "positive": conf["data"]["labels"]["favourable"],
        "negative": conf["data"]["labels"]["unfavourable"]
    }
        
    df, X, Y  = prepare_dataset(df, conf["data"]["target_variable"],  protected_attribute)
    impliesOperator = get_implies_operator(conf["model"]["implies"])
    
    ltnOps = LTNOps(impliesOperator, conf["model"]["p_mean"], conf["model"]["aggregator_deviation"])
    kb = KnowledgeBase.KnowledgeBase(
            X, X,
            Y, Y,
            label_map,
            protected_attribute['privileged'],
            protected_attribute['unprivileged'],
            hidden_layer_sizes= conf["model"]["hidden_layer_sizes"],
            fuzzy_ops=ltnOps,
            sensitive_feature_index=protected_attribute['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
    )

    model_path = args.model or "models/kb.npz"
    kb = load_model(kb, model_path)
    preds = inference(kb, X)
    plot_group_metrics(
        y_true=Y,
        y_pred=preds,
        sensitive=X[:, protected_attribute['index']],
        pos_label=label_map["positive"],
        neg_label=label_map["negative"],
        privileged_value=protected_attribute["privileged"],
        unprivileged_value=protected_attribute["unprivileged"],
        output_path="models/inference_metrics.png"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--dataset', help='Optional override for dataset CSV path')
    parser.add_argument('--model', help='Path to trained model')
    args = parser.parse_args()
    main(args)
