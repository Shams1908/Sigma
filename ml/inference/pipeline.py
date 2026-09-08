"""
End-to-end signal analysis inference pipeline.

The public interface (analyze_file / SignalAnalysisResult) is unchanged.
M6 addition: representation-aware segment transformation before CNN inference,
selected from checkpoint metadata so that the existing M5 2-channel path is
not affected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ml.input.types import PipelineConfig, SignalMetadata
from ml.input.pipeline import process_file
from ml.cnn_model.inference import get_cnn_model, _apply_checkpoint_representation
from ml.inference.aggregation import aggregate_window_probabilities
from ml.dataset.labels import INDEX_TO_MODULATION


@dataclass
class SignalAnalysisResult:
    """Structured container for end-to-end signal prediction results."""

    # Input metadata
    filename: str
    detected_format: str
    sample_rate: Optional[float]
    original_sample_count: int
    canonical_iq_shape: List[int]

    # Model results
    predicted_class_index: int
    predicted_class_name: str
    confidence: float
    probability_vector: List[float]
    top_predictions: List[Tuple[str, float]]

    # Window results
    num_windows: int
    window_predictions: List[str]
    window_confidences: List[float]
    window_probabilities: List[List[float]]
    window_distribution: Dict[str, int]
    mean_prediction_entropy: float

    # Processing information
    model_checkpoint_path: str
    normalization_source: str
    segment_length: int
    processing_status: str


def get_top_k_predictions(
    probs: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Extract the sorted top-k (class_name, probability) pairs."""
    indices = np.argsort(probs)[::-1][:top_k]
    return [(INDEX_TO_MODULATION[int(idx)], float(probs[int(idx)])) for idx in indices]


def analyze_file(
    path: str,
    config: Optional[PipelineConfig] = None,
    model_path: str = "models/m5_iq_cnn.pt",
    top_k: int = 5,
) -> SignalAnalysisResult:
    """
    Full signal-analysis inference pipeline.

    Flow:
      1. Parse signal file into canonical [2, N] and [M, 2, L] segments.
      2. Load the cached checkpoint (model, rms_factor, representation).
      3. Apply the checkpoint's declared representation to each segment.
         Legacy 2-channel checkpoints (M5) skip this step entirely.
      4. Normalize segments with the checkpoint's training RMS factor.
      5. Run batch CNN inference.
      6. Aggregate window predictions into a file-level result.

    Args:
        path:       Path to the signal file.
        config:     Input pipeline configuration overrides (optional).
        model_path: Path to the CNN checkpoint.
        top_k:      Number of top predictions to include.

    Returns:
        SignalAnalysisResult.

    Raises:
        FileNotFoundError: Input file or model checkpoint not found.
        ValueError:        Parsing, validation, or inference failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found at: '{path}'")

    if config is None:
        config = PipelineConfig()

    filename = os.path.basename(path)

    # ── Step 1: Load and segment the signal ──────────────────────────────────
    segments, input_meta = process_file(path, config)

    if input_meta.validation_status == "ERROR":
        raise ValueError(
            f"Signal input validation error: {input_meta.error_message}"
        )

    if len(segments) == 0:
        raise ValueError(
            f"Parsed signal contains no valid segments of length "
            f"{config.segment_length}. Verification failed."
        )

    # ── Step 2: Load checkpoint ───────────────────────────────────────────────
    # get_cnn_model returns (model, rms_factor, representation_name)
    # representation_name is None for legacy 2-channel M5 checkpoints.
    model, rms_factor, representation = get_cnn_model(model_path)

    # ── Step 3: Apply representation transform ────────────────────────────────
    # segments shape: [M, 2, L] (raw IQ, 2 channels)
    # After transform: [M, C, L] where C = num_channels for the representation
    segments_repr = _apply_checkpoint_representation(segments, representation)

    # ── Step 4: Normalize ─────────────────────────────────────────────────────
    segments_norm = segments_repr / rms_factor

    # ── Step 5: CNN inference ─────────────────────────────────────────────────
    x_tensor = torch.tensor(segments_norm, dtype=torch.float32)

    try:
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()  # [M, 11]
    except Exception as exc:
        raise ValueError(f"CNN model inference failed: {exc}") from exc

    # ── Step 6: Aggregate ─────────────────────────────────────────────────────
    file_idx, file_name, file_conf, mean_probs, pred_dist, mean_ent = (
        aggregate_window_probabilities(probs)
    )

    window_preds = [INDEX_TO_MODULATION[int(idx)] for idx in np.argmax(probs, axis=1)]
    window_confidences = [float(c) for c in np.max(probs, axis=1)]
    window_probabilities = [list(p.astype(float)) for p in probs]
    top_preds = get_top_k_predictions(mean_probs, top_k=top_k)

    return SignalAnalysisResult(
        filename=filename,
        detected_format=input_meta.detected_format,
        sample_rate=input_meta.sample_rate,
        original_sample_count=input_meta.original_sample_count,
        canonical_iq_shape=input_meta.canonical_iq_shape,
        predicted_class_index=file_idx,
        predicted_class_name=file_name,
        confidence=file_conf,
        probability_vector=list(mean_probs.astype(float)),
        top_predictions=top_preds,
        num_windows=len(segments),
        window_predictions=window_preds,
        window_confidences=window_confidences,
        window_probabilities=window_probabilities,
        window_distribution=pred_dist,
        mean_prediction_entropy=mean_ent,
        model_checkpoint_path=model_path,
        normalization_source="checkpoint training RMS",
        segment_length=config.segment_length,
        processing_status="SUCCESS",
    )
