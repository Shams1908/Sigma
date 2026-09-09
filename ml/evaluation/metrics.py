"""
M7-D, E, F, I: Core Metrics, Confusion Matrix, SNR & Multipath Breakdowns, and Calibration (ECE).

Provides deterministic calculations for closed-set evaluation, condition breakdowns,
and Expected Calibration Error (ECE).
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ml.evaluation.taxonomy import (
    CLOSED_SET_CLASSES,
    map_prediction_to_eval_class,
    is_eval_prediction_correct,
)


def calculate_core_metrics(
    y_true_names: List[str],
    y_pred_raw_names: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification metrics on closed-set evaluation samples.
    
    Args:
        y_true_names: List of ground-truth external class names (e.g. ['BPSK', 'QAM', ...]).
        y_pred_raw_names: List of raw model predictions from 11-class model (e.g. ['BPSK', 'QAM16', ...]).
        classes: Target evaluation classes (default CLOSED_SET_CLASSES).
        
    Returns:
        Dict containing accuracy, macro/weighted precision/recall/f1, and per-class metrics.
    """
    if classes is None:
        classes = list(CLOSED_SET_CLASSES)

    n_samples = len(y_true_names)
    if n_samples == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "support_total": 0,
        }

    # Binary correctness per sample according to M7 taxonomy rules
    correct_mask = [
        is_eval_prediction_correct(pred, true)
        for pred, true in zip(y_pred_raw_names, y_true_names)
    ]
    accuracy = float(np.mean(correct_mask))

    # Per-class calculation
    per_class: Dict[str, Dict[str, float]] = {}
    class_precisions: List[float] = []
    class_recalls: List[float] = []
    class_f1s: List[float] = []
    class_supports: List[int] = []

    for cls in classes:
        # Ground truth positives for this class
        tp = sum(
            1 for pred, true in zip(y_pred_raw_names, y_true_names)
            if true == cls and is_eval_prediction_correct(pred, true)
        )
        # All samples with this ground truth
        support = sum(1 for true in y_true_names if true == cls)
        # All samples predicted as this class (after evaluation mapping)
        predicted_as_cls = sum(
            1 for pred in y_pred_raw_names
            if map_prediction_to_eval_class(pred) == cls
        )

        precision = float(tp / predicted_as_cls) if predicted_as_cls > 0 else 0.0
        recall = float(tp / support) if support > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
            "predicted_count": int(predicted_as_cls),
        }

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_supports.append(support)

    total_support = sum(class_supports)
    macro_precision = float(np.mean(class_precisions)) if class_precisions else 0.0
    macro_recall = float(np.mean(class_recalls)) if class_recalls else 0.0
    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    if total_support > 0:
        weighted_precision = float(sum(p * s for p, s in zip(class_precisions, class_supports)) / total_support)
        weighted_recall = float(sum(r * s for r, s in zip(class_recalls, class_supports)) / total_support)
        weighted_f1 = float(sum(f * s for f, s in zip(class_f1s, class_supports)) / total_support)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "support_total": total_support,
    }


def calculate_confusion_matrix(
    y_true_names: List[str],
    y_pred_raw_names: List[str],
    target_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Constructs a confusion matrix where rows are ground-truth closed classes
    and columns are mapped predicted classes (including OTHER_CLOSED).
    """
    if target_classes is None:
        target_classes = list(CLOSED_SET_CLASSES)

    col_classes = list(target_classes) + ["OTHER_CLOSED"]
    matrix = np.zeros((len(target_classes), len(col_classes)), dtype=int)

    for true_cls, pred_raw in zip(y_true_names, y_pred_raw_names):
        if true_cls not in target_classes:
            continue
        row_idx = target_classes.index(true_cls)
        mapped_pred = map_prediction_to_eval_class(pred_raw)
        if mapped_pred in target_classes:
            col_idx = target_classes.index(mapped_pred)
        else:
            col_idx = len(target_classes)  # OTHER_CLOSED

        matrix[row_idx, col_idx] += 1

    return {
        "rows": target_classes,
        "columns": col_classes,
        "matrix": matrix.tolist(),
    }


def calculate_snr_breakdown(
    y_true_names: List[str],
    y_pred_raw_names: List[str],
    snrs: List[int],
    allowed_snrs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Breaks down performance by SNR level (e.g. 20, 22, 24, 26, 28, 30 dB).
    """
    if allowed_snrs is None:
        allowed_snrs = sorted(list(set(snrs)))

    breakdown = {}
    for snr in allowed_snrs:
        indices = [i for i, s in enumerate(snrs) if s == snr]
        if not indices:
            continue
        sub_true = [y_true_names[i] for i in indices]
        sub_pred = [y_pred_raw_names[i] for i in indices]
        metrics = calculate_core_metrics(sub_true, sub_pred)
        breakdown[str(snr)] = {
            "snr_db": int(snr),
            "sample_count": len(indices),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": {cls: d["f1"] for cls, d in metrics["per_class"].items()},
        }

    return breakdown


def calculate_channel_breakdown(
    y_true_names: List[str],
    y_pred_raw_names: List[str],
    channels: List[int],
) -> Dict[str, Any]:
    """
    Compares clean (0) vs multipath (1) performance and computes the degradation delta.
    """
    clean_indices = [i for i, c in enumerate(channels) if c == 0]
    multi_indices = [i for i, c in enumerate(channels) if c == 1]

    clean_true = [y_true_names[i] for i in clean_indices]
    clean_pred = [y_pred_raw_names[i] for i in clean_indices]
    multi_true = [y_true_names[i] for i in multi_indices]
    multi_pred = [y_pred_raw_names[i] for i in multi_indices]

    clean_metrics = calculate_core_metrics(clean_true, clean_pred)
    multi_metrics = calculate_core_metrics(multi_true, multi_pred)

    acc_delta = float(clean_metrics["accuracy"] - multi_metrics["accuracy"])
    f1_delta = float(clean_metrics["macro_f1"] - multi_metrics["macro_f1"])
    acc_drop_pct = float(acc_delta / clean_metrics["accuracy"] * 100.0) if clean_metrics["accuracy"] > 0 else 0.0

    return {
        "clean": {
            "sample_count": len(clean_indices),
            "accuracy": clean_metrics["accuracy"],
            "macro_f1": clean_metrics["macro_f1"],
            "weighted_f1": clean_metrics["weighted_f1"],
        },
        "multipath": {
            "sample_count": len(multi_indices),
            "accuracy": multi_metrics["accuracy"],
            "macro_f1": multi_metrics["macro_f1"],
            "weighted_f1": multi_metrics["weighted_f1"],
        },
        "degradation": {
            "accuracy_drop_abs": acc_delta,
            "accuracy_drop_pct": acc_drop_pct,
            "macro_f1_drop_abs": f1_delta,
        },
    }


def calculate_ece(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculates Expected Calibration Error (ECE) and bin statistics.
    
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if len(confidences) == 0:
        return {"ece": 0.0, "bins": []}

    confs = np.array(confidences, dtype=np.float64)
    accs = np.array(accuracies, dtype=np.float64)
    N = len(confs)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_data = []
    total_ece = 0.0

    for b in range(num_bins):
        bin_lower = bin_boundaries[b]
        bin_upper = bin_boundaries[b + 1]
        
        if b == num_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accs[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            weight = bin_count / N
            total_ece += weight * bin_error
        else:
            bin_acc = 0.0
            bin_conf = float((bin_lower + bin_upper) / 2.0)
            bin_error = 0.0

        bin_data.append({
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "calibration_error": float(bin_error),
        })

    return {
        "ece": float(total_ece),
        "num_bins": num_bins,
        "sample_count": N,
        "bins": bin_data,
    }
