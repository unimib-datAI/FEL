import sys

sys.path.append("../")
from utils import StandardScaleData_ExcludingFeature_simple, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import numpy as np
import pandas as pd
import yaml
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
    scores = kb._oracle.predict(X, logits=True)
    scores = np.asarray(scores).reshape(-1)
    print("Sample predictions:", preds)
    return preds, scores


def plot_inference_metrics(y_true, y_pred, y_scores, sensitive, pos_label, neg_label, privileged_value, unprivileged_value, output_path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_scores = np.asarray(y_scores).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)
    def group_mask(value):
        return sensitive == value

    def compute_rates(y_t, y_p):
        cm = confusion_matrix(y_t, y_p, labels=[neg_label, pos_label])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return cm, tpr, fpr

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()

    groups = [
        ("privileged", privileged_value),
        ("unprivileged", unprivileged_value),
    ]

    for i, (name, value) in enumerate(groups):
        mask = group_mask(value)
        cm, tpr, fpr = compute_rates(y_true[mask], y_pred[mask])
        axs[i].imshow(cm, cmap="Blues")
        axs[i].set_title(f"Confusion Matrix ({name})")
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("True")
        axs[i].set_xticks([0, 1])
        axs[i].set_yticks([0, 1])
        axs[i].set_xticklabels([neg_label, pos_label])
        axs[i].set_yticklabels([neg_label, pos_label])
        for r in range(2):
            for c in range(2):
                axs[i].text(c, r, cm[r, c], ha="center", va="center", color="black")

    # Selection rate
    sel_rates = []
    tprs = []
    fprs = []
    for name, value in groups:
        mask = group_mask(value)
        sel_rates.append(np.mean(y_pred[mask] == pos_label) if np.any(mask) else np.nan)
        cm, tpr, fpr = compute_rates(y_true[mask], y_pred[mask])
        tprs.append(tpr)
        fprs.append(fpr)

    axs[2].bar(["privileged", "unprivileged"], sel_rates, color=["#1f77b4", "#ff7f0e"])
    axs[2].set_title("Selection Rate")
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Rate")

    # TPR/FPR per group
    x = np.arange(len(groups))
    width = 0.35
    axs[3].bar(x - width/2, tprs, width, label="TPR")
    axs[3].bar(x + width/2, fprs, width, label="FPR")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(["privileged", "unprivileged"])
    axs[3].set_title("TPR / FPR by Group")
    axs[3].set_ylim(0, 1)
    axs[3].legend()

    fig.suptitle("Inference Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    preds, scores = inference(kb, X)
    plot_inference_metrics(
        y_true=Y,
        y_pred=preds,
        y_scores=scores,
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
