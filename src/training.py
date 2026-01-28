import sys

sys.path.append("../")
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import trange
import yaml
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

set_seed(42)


def save_kb_weights(kb, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    weights = {f"var_{i}": v.numpy() for i, v in enumerate(kb.trainable_variables)}
    np.savez(filepath, **weights)


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


    df, X_train, X_test, y_train, y_test  = prepare_dataset(df, conf["data"]["target_variable"],  protected_attribute)
    impliesOperator = get_implies_operator(conf["model"]["implies"])

    ltnOps = LTNOps(impliesOperator, conf["model"]["p_mean"], conf["model"]["aggregator_deviation"])
    kb = KnowledgeBase.KnowledgeBase(
            X_train, X_test,
            y_train, y_test,
            label_map,
            protected_attribute['privileged'],
            protected_attribute['unprivileged'],
            hidden_layer_sizes= conf["model"]["hidden_layer_sizes"],
            fuzzy_ops=ltnOps,
            sensitive_feature_index=protected_attribute['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    log_interval = max(1, conf["training"]["epochs"] // 100)
    epochs_hist = []
    test_acc_hist = []
    test_dpr_hist = []
    test_dpd_hist = []
    test_eor_hist = []
    test_eod_hist = []
    pbar = trange(conf["training"]["epochs"], desc="Training", dynamic_ncols=True)
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

    if epochs_hist:
        os.makedirs("models", exist_ok=True)

        def normalize_series(series):
            arr = np.array(series, dtype=float)
            if np.all(np.isnan(arr)):
                return arr
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
            if max_v - min_v == 0:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)

        fig = plt.figure(figsize=(10, 18))
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.1, 1.1])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[3, :])
        ax6 = fig.add_subplot(gs[4, :])

        ax0.plot(epochs_hist, test_acc_hist)
        ax0.set_title("Test Accuracy")
        ax1.plot(epochs_hist, test_dpr_hist)
        ax1.set_title("Test Demographic Parity Ratio")
        ax2.plot(epochs_hist, test_dpd_hist)
        ax2.set_title("Test Demographic Parity Difference")
        ax3.plot(epochs_hist, test_eor_hist)
        ax3.set_title("Test Equalized Odds Ratio")
        ax4.plot(epochs_hist, test_eod_hist)
        ax4.set_title("Test Equalized Odds Difference")

        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")

        ax5.plot(epochs_hist, test_acc_hist, label="test_acc")
        ax5.plot(epochs_hist, test_dpr_hist, label="test_dpr")
        ax5.plot(epochs_hist, test_dpd_hist, label="test_dpd")
        ax5.plot(epochs_hist, test_eor_hist, label="test_eor")
        ax5.plot(epochs_hist, test_eod_hist, label="test_eod")
        ax5.set_title("All Test Metrics")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Metric Value")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc="center right")

        ax6.plot(epochs_hist, normalize_series(test_acc_hist), label="test_acc_norm")
        ax6.plot(epochs_hist, normalize_series(test_dpr_hist), label="test_dpr_norm")
        ax6.plot(epochs_hist, normalize_series(test_dpd_hist), label="test_dpd_norm")
        ax6.plot(epochs_hist, normalize_series(test_eor_hist), label="test_eor_norm")
        ax6.plot(epochs_hist, normalize_series(test_eod_hist), label="test_eod_norm")
        ax6.set_title("All Test Metrics (Normalized)")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Normalized Value")
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc="center right")

        fig.suptitle("Test Metrics During Training")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig("models/test_metrics.png", dpi=150)
        plt.close(fig)

    save_kb_weights(kb, "models/kb.npz")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--dataset', help='Optional override for dataset CSV path')
    args = parser.parse_args()
    main(args)
