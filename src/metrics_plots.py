import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def normalize_series(series):
    arr = np.array(series, dtype=float)
    if np.all(np.isnan(arr)):
        return arr
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if max_v - min_v == 0:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def plot_training_metrics(histories, output_path):
    epochs_hist = histories["epochs"]
    if not epochs_hist:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(10, 18))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.1, 1.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[3, :])
    ax6 = fig.add_subplot(gs[4, :])

    ax0.plot(epochs_hist, histories["test_acc"])
    ax0.set_title("Test Accuracy")
    ax1.plot(epochs_hist, histories["test_dpr"])
    ax1.set_title("Test Demographic Parity Ratio")
    ax2.plot(epochs_hist, histories["test_dpd"])
    ax2.set_title("Test Demographic Parity Difference")
    ax3.plot(epochs_hist, histories["test_eor"])
    ax3.set_title("Test Equalized Odds Ratio")
    ax4.plot(epochs_hist, histories["test_eod"])
    ax4.set_title("Test Equalized Odds Difference")

    for ax in [ax0, ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")

    ax5.plot(epochs_hist, histories["test_acc"], label="test_acc")
    ax5.plot(epochs_hist, histories["test_dpr"], label="test_dpr")
    ax5.plot(epochs_hist, histories["test_dpd"], label="test_dpd")
    ax5.plot(epochs_hist, histories["test_eor"], label="test_eor")
    ax5.plot(epochs_hist, histories["test_eod"], label="test_eod")
    ax5.set_title("All Test Metrics")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Metric Value")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="center right")

    ax6.plot(epochs_hist, normalize_series(histories["test_acc"]), label="test_acc_norm")
    ax6.plot(epochs_hist, normalize_series(histories["test_dpr"]), label="test_dpr_norm")
    ax6.plot(epochs_hist, normalize_series(histories["test_dpd"]), label="test_dpd_norm")
    ax6.plot(epochs_hist, normalize_series(histories["test_eor"]), label="test_eor_norm")
    ax6.plot(epochs_hist, normalize_series(histories["test_eod"]), label="test_eod_norm")
    ax6.set_title("All Test Metrics (Normalized)")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Normalized Value")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="center right")

    fig.suptitle("Test Metrics During Training")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_metrics(y_true, y_pred, sensitive, pos_label, neg_label,
                       privileged_value, unprivileged_value, output_path):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
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
        cm, _, _ = compute_rates(y_true[mask], y_pred[mask])
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

    sel_rates = []
    tprs = []
    fprs = []
    for _, value in groups:
        mask = group_mask(value)
        sel_rates.append(np.mean(y_pred[mask] == pos_label) if np.any(mask) else np.nan)
        _, tpr, fpr = compute_rates(y_true[mask], y_pred[mask])
        tprs.append(tpr)
        fprs.append(fpr)

    axs[2].bar(["privileged", "unprivileged"], sel_rates, color=["#1f77b4", "#ff7f0e"])
    axs[2].set_title("Selection Rate")
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Rate")

    x = np.arange(len(groups))
    width = 0.35
    axs[3].bar(x - width / 2, tprs, width, label="TPR")
    axs[3].bar(x + width / 2, fprs, width, label="FPR")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(["privileged", "unprivileged"])
    axs[3].set_title("TPR / FPR by Group")
    axs[3].set_ylim(0, 1)
    axs[3].legend()

    fig.suptitle("Group Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
