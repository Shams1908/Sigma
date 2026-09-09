"""
M7-P: Reproducible Visualization Artifacts for Model Evaluation.

Generates:
  1. Confusion Matrix for M5
  2. Confusion Matrix for M6
  3. Accuracy vs SNR comparison curve
  4. Macro F1 vs SNR comparison curve
  5. Clean vs Multipath robustness comparison
  6. Confidence distribution (correct vs incorrect)
  7. Reliability diagram (Calibration)
"""
import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm_dict: Dict[str, Any],
    title: str,
    output_path: str,
    cmap: str = "Blues",
) -> str:
    """
    Renders and saves a formatted confusion matrix plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = cm_dict["rows"]
    cols = cm_dict["columns"]
    matrix = np.array(cm_dict["matrix"])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(cols)),
        yticks=np.arange(len(rows)),
        xticklabels=cols,
        yticklabels=rows,
        title=title,
        ylabel="Ground Truth Class",
        xlabel="Predicted Evaluation Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate counts inside cells
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(
                j, i, format(matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_snr_curves(
    snr_m5: Dict[str, Any],
    snr_m6: Dict[str, Any],
    metric_key: str,
    metric_name: str,
    output_path: str,
) -> str:
    """
    Plots M5 vs M6 performance curve across SNR levels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    snrs = sorted([int(k) for k in snr_m5.keys() if k in snr_m6])
    m5_vals = [snr_m5[str(s)][metric_key] for s in snrs]
    m6_vals = [snr_m6[str(s)][metric_key] for s in snrs]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(snrs, m5_vals, marker="o", linewidth=2.0, color="#1f77b4", label="M5 Baseline [I, Q]")
    ax.plot(snrs, m6_vals, marker="s", linewidth=2.0, color="#2ca02c", label="M6 Robust [I, Q, |z|]")

    ax.set_title(f"{metric_name} vs SNR (External Evaluation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_xticks(snrs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_clean_vs_multipath_bar(
    chan_m5: Dict[str, Any],
    chan_m6: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Generates bar chart comparing clean vs multipath accuracy and F1 for M5 and M6.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    categories = ["M5 Clean", "M5 Multipath", "M6 Clean", "M6 Multipath"]
    accs = [
        chan_m5["clean"]["accuracy"],
        chan_m5["multipath"]["accuracy"],
        chan_m6["clean"]["accuracy"],
        chan_m6["multipath"]["accuracy"],
    ]
    f1s = [
        chan_m5["clean"]["macro_f1"],
        chan_m5["multipath"]["macro_f1"],
        chan_m6["clean"]["macro_f1"],
        chan_m6["multipath"]["macro_f1"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#3470a3")
    rects2 = ax.bar(x + width/2, f1s, width, label="Macro F1", color="#e27c34")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("Channel Degradation: Clean AWGN vs Multipath Fading", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim([0, max(max(accs), max(f1s), 0.1) * 1.25])
    ax.legend()

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confidence_distribution(
    correct_confs: List[float],
    incorrect_confs: List[float],
    output_path: str,
) -> str:
    """
    Plots histogram comparing confidence distribution of correct vs incorrect predictions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bins = np.linspace(0.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if correct_confs:
        ax.hist(correct_confs, bins=bins, alpha=0.6, color="#2ca02c", label=f"Correct (N={len(correct_confs)})", density=True)
    if incorrect_confs:
        ax.hist(incorrect_confs, bins=bins, alpha=0.6, color="#d62728", label=f"Incorrect (N={len(incorrect_confs)})", density=True)

    ax.set_title("Prediction Confidence Distribution (Correct vs Incorrect)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence (Maximum Softmax Probability)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_reliability_diagram(
    ece_data: Dict[str, Any],
    title: str,
    output_path: str,
) -> str:
    """
    Plots Expected Calibration Error (ECE) reliability diagram.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bins = ece_data.get("bins", [])
    if not bins:
        return output_path

    confs = [b["confidence"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    ece_val = ece_data.get("ece", 0.0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.bar(
        [b["bin_lower"] + 0.05 for b in bins],
        accs,
        width=0.08,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="black",
        label=f"Outputs (ECE = {ece_val:.4f})",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path
