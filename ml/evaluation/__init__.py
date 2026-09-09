"""
ML Phase M7: Model Evaluation, Robustness, and Confidence-Reliability Package.
"""
from ml.evaluation.integrity import verify_dataset_integrity, DatasetIntegrityReport
from ml.evaluation.taxonomy import (
    EVALUATION_TAXONOMY,
    CLOSED_SET_CLASSES,
    OPEN_SET_CLASSES,
    map_prediction_to_eval_class,
    is_eval_prediction_correct,
)
from ml.evaluation.metrics import (
    calculate_core_metrics,
    calculate_confusion_matrix,
    calculate_snr_breakdown,
    calculate_channel_breakdown,
    calculate_ece,
)
from ml.evaluation.confidence import analyze_confidence_reliability, TemperatureScaler
from ml.evaluation.openset import OpenSetEvaluator, OpenSetThresholds
from ml.evaluation.benchmarking import benchmark_inference_latency
from ml.evaluation.evaluator import M7ModelEvaluator

__all__ = [
    "verify_dataset_integrity",
    "DatasetIntegrityReport",
    "EVALUATION_TAXONOMY",
    "CLOSED_SET_CLASSES",
    "OPEN_SET_CLASSES",
    "map_prediction_to_eval_class",
    "is_eval_prediction_correct",
    "calculate_core_metrics",
    "calculate_confusion_matrix",
    "calculate_snr_breakdown",
    "calculate_channel_breakdown",
    "calculate_ece",
    "analyze_confidence_reliability",
    "TemperatureScaler",
    "OpenSetEvaluator",
    "OpenSetThresholds",
    "benchmark_inference_latency",
    "M7ModelEvaluator",
]
