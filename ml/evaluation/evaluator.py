"""
M7-C, L, N, O: Unified Evaluation Orchestrator for M5 vs M6.

Executes side-by-side evaluation of M5 Baseline and M6 Robust CNN checkpoints
on identical samples. Evaluates both window-level and frame-level aggregations,
open-set rejection, confidence reliability, and inference latencies.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from ml.cnn_model.inference import get_cnn_model
from ml.dataset.external_dataset import (
    ExternalDatasetSplit,
    slice_frames_to_windows,
)
from ml.representations.transforms import compute_representation, RepresentationType
from ml.inference.aggregation import aggregate_window_probabilities
from ml.evaluation.integrity import verify_dataset_integrity, DatasetIntegrityReport
from ml.evaluation.taxonomy import (
    EXTERNAL_IDX_TO_NAME,
    CLOSED_SET_CLASSES,
    OPEN_SET_CLASSES,
    partition_closed_and_open_set_indices,
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
from ml.evaluation.confidence import (
    analyze_confidence_reliability,
    TemperatureScaler,
)
from ml.evaluation.openset import (
    OpenSetEvaluator,
    select_thresholds_on_dev,
)
from ml.evaluation.benchmarking import benchmark_inference_latency


class M7ModelEvaluator:
    """
    Unified evaluator for ML Phase M7.
    """
    def __init__(
        self,
        m5_checkpoint_path: str = "models/m5_iq_cnn.pt",
        m6_checkpoint_path: str = "models/m6_robust_cnn.pt",
    ):
        self.m5_checkpoint_path = m5_checkpoint_path
        self.m6_checkpoint_path = m6_checkpoint_path

        # Load models in eval mode
        self.model_m5, self.rms_m5 = get_cnn_model(m5_checkpoint_path)
        self.model_m6, self.rms_m6 = get_cnn_model(m6_checkpoint_path)
        self.model_m5.eval()
        self.model_m6.eval()

        self.labels_m5 = getattr(self.model_m5, "label_mapping", None)
        self.labels_m6 = getattr(self.model_m6, "label_mapping", None)

    def _run_batch_inference(
        self,
        model: torch.nn.Module,
        windows: np.ndarray,
        rms_factor: float,
        in_channels: int,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs batch inference on normalized windows.
        
        Args:
            windows: shape [M, 2, 128]
            rms_factor: RMS scaling
            in_channels: 2 or 3
            
        Returns:
            (logits, probs) of shape [M, num_classes]
        """
        logits_list = []
        probs_list = []
        num_windows = len(windows)

        with torch.no_grad():
            for i in range(0, num_windows, batch_size):
                batch_raw = windows[i : i + batch_size] / rms_factor
                if in_channels == 3:
                    batch_in = compute_representation(batch_raw, RepresentationType.IQ_AMPLITUDE)
                else:
                    batch_in = batch_raw
                batch_t = torch.tensor(batch_in, dtype=torch.float32)
                logits = model(batch_t)
                probs = torch.softmax(logits, dim=-1)
                logits_list.append(logits.numpy())
                probs_list.append(probs.numpy())

        all_logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 11))
        all_probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 11))
        return all_logits, all_probs

    def evaluate_split(
        self,
        split: ExternalDatasetSplit,
        dev_split_for_tuning: Optional[ExternalDatasetSplit] = None,
        fit_temperature_scaling: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes comprehensive evaluation of M5 and M6 on an external dataset split.
        
        Args:
            split: The test split to evaluate.
            dev_split_for_tuning: Optional isolated development split to fit calibration
                                  and open-set thresholds.
            fit_temperature_scaling: If True, fits temperature scaling on dev_split_for_tuning.
            
        Returns:
            Complete evaluation results dictionary.
        """
        # 1. Audit Dataset Integrity
        integrity_report = verify_dataset_integrity(
            split,
            reference_dev_split=dev_split_for_tuning,
            raise_on_error=True,
        )

        N_frames = len(split.X)
        y_mod = list(split.y_mod)
        y_chan = list(split.y_chan)
        y_snr = list(split.y_snr)
        true_class_names = [EXTERNAL_IDX_TO_NAME[m] for m in y_mod]

        # 2. Slice 1024-sample frames into 128-sample windows (8 per frame)
        windows = slice_frames_to_windows(split.X, window_length=128)
        num_windows_per_frame = 8

        # 3. Model Inference across all windows
        # M5
        logits_m5_win, probs_m5_win = self._run_batch_inference(
            self.model_m5, windows, self.rms_m5, in_channels=2
        )
        # M6
        logits_m6_win, probs_m6_win = self._run_batch_inference(
            self.model_m6, windows, self.rms_m6, in_channels=3
        )

        # 4. Window-level predictions
        pred_idx_m5_win = np.argmax(probs_m5_win, axis=1)
        pred_idx_m6_win = np.argmax(probs_m6_win, axis=1)
        pred_names_m5_win = [self.labels_m5[i] for i in pred_idx_m5_win]
        pred_names_m6_win = [self.labels_m6[i] for i in pred_idx_m6_win]
        confs_m5_win = [float(np.max(p)) for p in probs_m5_win]
        confs_m6_win = [float(np.max(p)) for p in probs_m6_win]

        # Expand frame truth to window truth
        y_mod_win = np.repeat(y_mod, num_windows_per_frame).tolist()
        y_chan_win = np.repeat(y_chan, num_windows_per_frame).tolist()
        y_snr_win = np.repeat(y_snr, num_windows_per_frame).tolist()
        true_names_win = [EXTERNAL_IDX_TO_NAME[m] for m in y_mod_win]

        # 5. Frame-level aggregation (mean softmax over the 8 windows of each frame)
        frame_probs_m5 = []
        frame_probs_m6 = []
        for f in range(N_frames):
            start = f * num_windows_per_frame
            end = start + num_windows_per_frame
            frame_probs_m5.append(np.mean(probs_m5_win[start:end], axis=0))
            frame_probs_m6.append(np.mean(probs_m6_win[start:end], axis=0))

        frame_probs_m5 = np.array(frame_probs_m5)
        frame_probs_m6 = np.array(frame_probs_m6)

        pred_names_m5_frame = [self.labels_m5[int(np.argmax(p))] for p in frame_probs_m5]
        pred_names_m6_frame = [self.labels_m6[int(np.argmax(p))] for p in frame_probs_m6]
        confs_m5_frame = [float(np.max(p)) for p in frame_probs_m5]
        confs_m6_frame = [float(np.max(p)) for p in frame_probs_m6]

        # 6. Partition Closed-Set vs Open-Set
        closed_frame_idx, open_frame_idx = partition_closed_and_open_set_indices(y_mod)
        closed_win_idx, open_win_idx = partition_closed_and_open_set_indices(y_mod_win)

        # 7. Closed-Set Metrics: Frame-Level
        c_true_frame = [true_class_names[i] for i in closed_frame_idx]
        c_pred_m5_frame = [pred_names_m5_frame[i] for i in closed_frame_idx]
        c_pred_m6_frame = [pred_names_m6_frame[i] for i in closed_frame_idx]
        c_chan_frame = [y_chan[i] for i in closed_frame_idx]
        c_snr_frame = [y_snr[i] for i in closed_frame_idx]
        c_conf_m5_frame = [confs_m5_frame[i] for i in closed_frame_idx]
        c_conf_m6_frame = [confs_m6_frame[i] for i in closed_frame_idx]

        m5_frame_core = calculate_core_metrics(c_true_frame, c_pred_m5_frame)
        m6_frame_core = calculate_core_metrics(c_true_frame, c_pred_m6_frame)

        m5_frame_cm = calculate_confusion_matrix(c_true_frame, c_pred_m5_frame)
        m6_frame_cm = calculate_confusion_matrix(c_true_frame, c_pred_m6_frame)

        m5_frame_snr = calculate_snr_breakdown(c_true_frame, c_pred_m5_frame, c_snr_frame)
        m6_frame_snr = calculate_snr_breakdown(c_true_frame, c_pred_m6_frame, c_snr_frame)

        m5_frame_chan = calculate_channel_breakdown(c_true_frame, c_pred_m5_frame, c_chan_frame)
        m6_frame_chan = calculate_channel_breakdown(c_true_frame, c_pred_m6_frame, c_chan_frame)

        # 8. Closed-Set Metrics: Window-Level
        c_true_win = [true_names_win[i] for i in closed_win_idx]
        c_pred_m5_win = [pred_names_m5_win[i] for i in closed_win_idx]
        c_pred_m6_win = [pred_names_m6_win[i] for i in closed_win_idx]

        m5_win_core = calculate_core_metrics(c_true_win, c_pred_m5_win)
        m6_win_core = calculate_core_metrics(c_true_win, c_pred_m6_win)

        # 9. Confidence Reliability & Calibration
        m5_acc_bool_frame = [is_eval_prediction_correct(p, t) for p, t in zip(c_pred_m5_frame, c_true_frame)]
        m6_acc_bool_frame = [is_eval_prediction_correct(p, t) for p, t in zip(c_pred_m6_frame, c_true_frame)]

        m5_conf_analysis = analyze_confidence_reliability(c_conf_m5_frame, m5_acc_bool_frame)
        m6_conf_analysis = analyze_confidence_reliability(c_conf_m6_frame, m6_acc_bool_frame)

        # 10. Optional Temperature Scaling (fitted on dev split ONLY)
        calibration_results = None
        if fit_temperature_scaling and dev_split_for_tuning is not None:
            calibration_results = self._fit_and_evaluate_calibration(
                dev_split_for_tuning,
                test_split=split,
                closed_frame_idx=closed_frame_idx,
                c_true_frame=c_true_frame,
            )

        # 11. Open-Set Rejection Evaluation
        # Select thresholds on dev split if available, otherwise use defaults
        if dev_split_for_tuning is not None:
            # Generate dev probabilities for threshold tuning
            dev_windows = slice_frames_to_windows(dev_split_for_tuning.X, window_length=128)
            _, dev_probs_m5_win = self._run_batch_inference(self.model_m5, dev_windows, self.rms_m5, in_channels=2)
            _, dev_probs_m6_win = self._run_batch_inference(self.model_m6, dev_windows, self.rms_m6, in_channels=3)

            dev_probs_m5_frame = np.array([
                np.mean(dev_probs_m5_win[f*8:(f+1)*8], axis=0) for f in range(len(dev_split_for_tuning.X))
            ])
            dev_probs_m6_frame = np.array([
                np.mean(dev_probs_m6_win[f*8:(f+1)*8], axis=0) for f in range(len(dev_split_for_tuning.X))
            ])

            thresh_m5 = select_thresholds_on_dev(dev_probs_m5_frame, list(dev_split_for_tuning.y_mod))
            thresh_m6 = select_thresholds_on_dev(dev_probs_m6_frame, list(dev_split_for_tuning.y_mod))
        else:
            from ml.evaluation.openset import OpenSetThresholds
            thresh_m5 = OpenSetThresholds(confidence_threshold=0.35, entropy_threshold=1.80, source="DEFAULT_FROZEN")
            thresh_m6 = OpenSetThresholds(confidence_threshold=0.35, entropy_threshold=1.80, source="DEFAULT_FROZEN")

        eval_openset_m5 = OpenSetEvaluator(thresh_m5).evaluate(frame_probs_m5, y_mod)
        eval_openset_m6 = OpenSetEvaluator(thresh_m6).evaluate(frame_probs_m6, y_mod)

        # 12. Inference Latency Benchmarks
        latency_m5 = benchmark_inference_latency(self.model_m5, self.rms_m5, in_channels=2)
        latency_m6 = benchmark_inference_latency(self.model_m6, self.rms_m6, in_channels=3)

        # 13. M5 vs M6 Direct Comparison
        comparison = {
            "frame_level_accuracy": {
                "m5": m5_frame_core["accuracy"],
                "m6": m6_frame_core["accuracy"],
                "delta_m6_minus_m5": float(m6_frame_core["accuracy"] - m5_frame_core["accuracy"]),
            },
            "frame_level_macro_f1": {
                "m5": m5_frame_core["macro_f1"],
                "m6": m6_frame_core["macro_f1"],
                "delta_m6_minus_m5": float(m6_frame_core["macro_f1"] - m5_frame_core["macro_f1"]),
            },
            "multipath_f1_retention": {
                "m5_multipath_f1": m5_frame_chan["multipath"]["macro_f1"],
                "m6_multipath_f1": m6_frame_chan["multipath"]["macro_f1"],
                "delta_m6_minus_m5": float(m6_frame_chan["multipath"]["macro_f1"] - m5_frame_chan["multipath"]["macro_f1"]),
            },
            "latency_p95_pipeline_ms": {
                "m5": latency_m5["full_frame_pipeline_8windows"]["p95_ms"],
                "m6": latency_m6["full_frame_pipeline_8windows"]["p95_ms"],
            },
        }

        # 14. Determine Evaluation Data Classification
        is_real_benchmark = (split.provenance == "REAL-WORLD-EXTERNAL-BENCHMARK")
        evaluation_data_type = (
            "independent_real_world_evaluation"
            if is_real_benchmark
            else "pipeline_verification_and_development_evaluation"
        )
        status = "M7 COMPLETE" if is_real_benchmark else "M7 PARTIAL"

        return {
            "status": status,
            "evaluation_data_type": evaluation_data_type,
            "dataset_integrity": integrity_report.to_dict(),
            "models": {
                "m5": {
                    "checkpoint": self.m5_checkpoint_path,
                    "in_channels": 2,
                    "representation": "RAW_IQ",
                    "rms_factor": self.rms_m5,
                    "frame_level_metrics": m5_frame_core,
                    "window_level_metrics": m5_win_core,
                    "confusion_matrix": m5_frame_cm,
                    "snr_breakdown": m5_frame_snr,
                    "channel_breakdown": m5_frame_chan,
                    "confidence_analysis": m5_conf_analysis,
                    "open_set_rejection": eval_openset_m5.to_dict(),
                    "latency": latency_m5,
                },
                "m6": {
                    "checkpoint": self.m6_checkpoint_path,
                    "in_channels": 3,
                    "representation": "IQ_AMPLITUDE",
                    "rms_factor": self.rms_m6,
                    "frame_level_metrics": m6_frame_core,
                    "window_level_metrics": m6_win_core,
                    "confusion_matrix": m6_frame_cm,
                    "snr_breakdown": m6_frame_snr,
                    "channel_breakdown": m6_frame_chan,
                    "confidence_analysis": m6_conf_analysis,
                    "open_set_rejection": eval_openset_m6.to_dict(),
                    "latency": latency_m6,
                },
            },
            "comparison": comparison,
            "temperature_scaling_calibration": calibration_results,
            "hypothesis_engine_interface": {
                "ml_role": "Provides candidate modulations and model probabilities as independent evidence.",
                "distinction": "MODEL PROBABILITY != HYPOTHESIS CONFIDENCE",
                "evidence_contract": {
                    "candidates": "Top-k predicted modulations mapped to canonical types.",
                    "model_probabilities": "Softmax distribution over modulation classes.",
                    "open_set_decision": "KNOWN, LOW_CONFIDENCE, or UNSUPPORTED.",
                },
            },
        }

    def _fit_and_evaluate_calibration(
        self,
        dev_split: ExternalDatasetSplit,
        test_split: ExternalDatasetSplit,
        closed_frame_idx: List[int],
        c_true_frame: List[str],
    ) -> Dict[str, Any]:
        """
        Fits Temperature Scaling on dev subset logits and evaluates on test split.
        """
        # Dev inference
        dev_win = slice_frames_to_windows(dev_split.X, window_length=128)
        dev_logits_m5_win, _ = self._run_batch_inference(self.model_m5, dev_win, self.rms_m5, in_channels=2)
        dev_logits_m6_win, _ = self._run_batch_inference(self.model_m6, dev_win, self.rms_m6, in_channels=3)

        dev_logits_m5_frame = np.array([
            np.mean(dev_logits_m5_win[f*8:(f+1)*8], axis=0) for f in range(len(dev_split.X))
        ])
        dev_logits_m6_frame = np.array([
            np.mean(dev_logits_m6_win[f*8:(f+1)*8], axis=0) for f in range(len(dev_split.X))
        ])

        # Dev closed targets: map to radioML label index if supported
        dev_y_mod = list(dev_split.y_mod)
        closed_dev_idx, _ = partition_closed_and_open_set_indices(dev_y_mod)
        
        # Fit on dev
        scaler_m5 = TemperatureScaler()
        scaler_m6 = TemperatureScaler()

        # Closed subset for calibration fitting
        dev_targets_m5 = []
        for idx in closed_dev_idx:
            ext_name = EXTERNAL_IDX_TO_NAME[dev_y_mod[idx]]
            if ext_name in ("BPSK", "QPSK", "WBFM"):
                dev_targets_m5.append(self.labels_m5.index(ext_name))
            elif ext_name == "QAM":
                dev_targets_m5.append(self.labels_m5.index("QAM16")) # Target anchor

        dev_targets_arr = np.array(dev_targets_m5, dtype=np.int64)
        t_m5 = scaler_m5.fit_on_dev(dev_logits_m5_frame[closed_dev_idx], dev_targets_arr)
        t_m6 = scaler_m6.fit_on_dev(dev_logits_m6_frame[closed_dev_idx], dev_targets_arr)

        # Now evaluate frozen scalers on test split
        test_win = slice_frames_to_windows(test_split.X, window_length=128)
        test_logits_m5_win, _ = self._run_batch_inference(self.model_m5, test_win, self.rms_m5, in_channels=2)
        test_logits_m6_win, _ = self._run_batch_inference(self.model_m6, test_win, self.rms_m6, in_channels=3)

        test_logits_m5_frame = np.array([
            np.mean(test_logits_m5_win[f*8:(f+1)*8], axis=0) for f in range(len(test_split.X))
        ])
        test_logits_m6_frame = np.array([
            np.mean(test_logits_m6_win[f*8:(f+1)*8], axis=0) for f in range(len(test_split.X))
        ])

        cal_probs_m5 = scaler_m5.calibrate_probs(test_logits_m5_frame[closed_frame_idx])
        cal_probs_m6 = scaler_m6.calibrate_probs(test_logits_m6_frame[closed_frame_idx])

        cal_confs_m5 = [float(np.max(p)) for p in cal_probs_m5]
        cal_confs_m6 = [float(np.max(p)) for p in cal_probs_m6]

        cal_preds_m5 = [self.labels_m5[int(np.argmax(p))] for p in cal_probs_m5]
        cal_preds_m6 = [self.labels_m6[int(np.argmax(p))] for p in cal_probs_m6]

        m5_accs = [is_eval_prediction_correct(p, t) for p, t in zip(cal_preds_m5, c_true_frame)]
        m6_accs = [is_eval_prediction_correct(p, t) for p, t in zip(cal_preds_m6, c_true_frame)]

        ece_cal_m5 = calculate_ece(cal_confs_m5, m5_accs, num_bins=10)
        ece_cal_m6 = calculate_ece(cal_confs_m6, m6_accs, num_bins=10)

        return {
            "fitting_split": dev_split.provenance,
            "fitted_temperature": {
                "m5": t_m5,
                "m6": t_m6,
            },
            "calibrated_ece": {
                "m5": ece_cal_m5["ece"],
                "m6": ece_cal_m6["ece"],
            },
            "ece_details_m5": ece_cal_m5,
            "ece_details_m6": ece_cal_m6,
        }
