import sys

sys.path.append("../")
import json
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import pandas as pd
from tqdm import trange
import yaml
import argparse
import os
import numpy as np
import tensorflow as tf
from metrics_plots import plot_group_metrics, plot_training_metrics

set_seed(42)


def save_kb_weights(kb, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    weights = {f"var_{i}": v.numpy() for i, v in enumerate(kb.trainable_variables)}
    np.savez(filepath, **weights)


def resolve_fairness_config(conf):
    fairness_conf = conf.get("fairness")
    if fairness_conf is None:
        raise ValueError("Missing required config section: fairness")
    if "enabled" not in fairness_conf:
        raise ValueError("Missing required fairness field: fairness.enabled")
    if "weight" not in fairness_conf:
        raise ValueError("Missing required fairness field: fairness.weight")

    enabled = bool(fairness_conf["enabled"])
    weight = float(fairness_conf["weight"])
    return enabled, weight


def apply_fairness_axiom_config(conf, axioms_path="./src/KnowledgeBaseAxioms.json"):
    fairness_enabled, fairness_weight = resolve_fairness_config(conf)
    if fairness_enabled and fairness_weight <= 0:
        raise ValueError(
            "Invalid fairness config: when fairness.enabled is true, fairness.weight must be > 0."
        )

    with open(axioms_path, "r") as f:
        axioms = json.load(f)

    target_axiom = "axiom_demographic_parity"
    found_target = False

    for axiom in axioms:
        if axiom.get("name") == target_axiom:
            axiom["infos"]["training"] = fairness_enabled
            axiom["infos"]["weight"] = fairness_weight if fairness_enabled else 0
            found_target = True

    if not found_target:
        raise ValueError(
            f"Cannot find {target_axiom} in {axioms_path}. Unable to apply fairness YAML config."
        )

    with open(axioms_path, "w") as f:
        json.dump(axioms, f, indent=4)


def prepare_dataset(df, target_variable,  protected_attribute):
    Y = df[target_variable].to_numpy()
    X = df.drop(columns=[target_variable]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        np.squeeze(Y),
        test_size=0.33,
        stratify=np.squeeze(Y))

    X_train, X_test, _ = StandardScaleData_ExcludingFeature(
            X_train, X_test, protected_attribute['index'])
    
    return df, X_train, X_test, y_train, y_test


def build_metadata(conf, df):
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
    return protected_attribute, label_map


def build_kb(conf, X_train, X_test, y_train, y_test, protected_attribute, label_map):
    implies_operator = get_implies_operator(conf["model"]["implies"])
    ltn_ops = LTNOps(implies_operator, conf["model"]["p_mean"], conf["model"]["aggregator_deviation"])
    return KnowledgeBase.KnowledgeBase(
        X_train, X_test,
        y_train, y_test,
        label_map,
        protected_attribute['privileged'],
        protected_attribute['unprivileged'],
        hidden_layer_sizes=conf["model"]["hidden_layer_sizes"],
        fuzzy_ops=ltn_ops,
        sensitive_feature_index=protected_attribute['index'],
        config_file='./src/KnowledgeBaseAxioms.json'
    )


def train_loop(kb, epochs, log_interval):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs_hist = []
    test_acc_hist = []
    test_dpr_hist = []
    test_dpd_hist = []
    test_eor_hist = []
    test_eod_hist = []
    pbar = trange(epochs, desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        with tf.GradientTape() as tape:
            loss = 1. - kb.train_step()  # type: ignore
        grads = tape.gradient(loss, kb.trainable_variables)
        optimizer.apply_gradients(zip(grads, kb.trainable_variables))

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logs = kb.get_logs()
            train_acc = logs.get('train_classification_metrics', {}).get('accuracy')
            test_acc = logs.get('test_classification_metrics', {}).get('accuracy')

            fm = logs.get("fairness_metrics", {})
            train_dpr = fm.get("train_demographic_parity_ratio")
            test_dpr = fm.get("test_demographic_parity_ratio")
            train_dpd = fm.get("train_demographic_parity_difference")
            test_dpd = fm.get("test_demographic_parity_difference")
            test_eor = fm.get("test_equalized_odds_ratio")
            test_eod = fm.get("test_equalized_odds_difference")

            epochs_hist.append(epoch + 1)
            test_acc_hist.append(test_acc)
            test_dpr_hist.append(test_dpr)
            test_dpd_hist.append(test_dpd)
            test_eor_hist.append(test_eor)
            test_eod_hist.append(test_eod)

            pbar.set_postfix({
                "train_acc": f"{train_acc:.3f}",
                "test_acc": f"{test_acc:.3f}",
                "train_dpr": f"{train_dpr:.3f}",
                "test_dpr": f"{test_dpr:.3f}",
                "train_dpd": f"{train_dpd:.3f}",
                "test_dpd": f"{test_dpd:.3f}",
            })

    return {
        "epochs": epochs_hist,
        "test_acc": test_acc_hist,
        "test_dpr": test_dpr_hist,
        "test_dpd": test_dpd_hist,
        "test_eor": test_eor_hist,
        "test_eod": test_eod_hist,
    }


def main(args):
    with open(args.config, 'r') as f:
        conf =  yaml.safe_load(f)

    apply_fairness_axiom_config(conf)
    df = pd.read_csv(args.dataset)

    protected_attribute, label_map = build_metadata(conf, df)
    df, X_train, X_test, y_train, y_test = prepare_dataset(df, conf["data"]["target_variable"], protected_attribute)
    kb = build_kb(conf, X_train, X_test, y_train, y_test, protected_attribute, label_map)

    epochs = conf["training"]["epochs"]
    log_interval = max(1, epochs // 100)
    histories = train_loop(kb, epochs, log_interval)

    plot_training_metrics(histories, "models/test_metrics.png")
    plot_group_metrics(
        y_true=kb._Ytest,
        y_pred=kb._oracle.predict(kb._Xtest),
        sensitive=kb._Xtest[:, protected_attribute['index']],
        pos_label=label_map["positive"],
        neg_label=label_map["negative"],
        privileged_value=protected_attribute["privileged"],
        unprivileged_value=protected_attribute["unprivileged"],
        output_path="models/test_group_metrics.png"
    )
    save_kb_weights(kb, "models/kb.npz")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--dataset', help='Optional override for dataset CSV path')
    args = parser.parse_args()
    main(args)
