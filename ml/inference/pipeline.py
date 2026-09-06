import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

from ml.input.types import PipelineConfig, SignalMetadata
from ml.input.pipeline import process_file
from ml.cnn_model.inference import get_cnn_model
from ml.inference.aggregation import aggregate_window_probabilities
from ml.dataset.labels import INDEX_TO_MODULATION

@dataclass
class SignalAnalysisResult:
    """
    Structured container for end-to-end signal prediction results.
    """
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
    top_predictions: List[Tuple[str, float]] # List of (class_name, probability)
    
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

def get_top_k_predictions(probs: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Helper to extract the sorted top-k prediction classes and their scores.
    """
    indices = np.argsort(probs)[::-1][:top_k]
    return [(INDEX_TO_MODULATION[int(idx)], float(probs[int(idx)])) for idx in indices]

def analyze_file(
    path: str,
    config: Optional[PipelineConfig] = None,
    model_path: str = "models/m5_iq_cnn.pt",
    top_k: int = 5
) -> SignalAnalysisResult:
    """
    Ties the input parsing and model inference layer into a single unified API.
    
    Flow:
      1. Calls M7.1 pipeline to convert file to canonical [2, N] and segments.
      2. Validates output and handles processing errors.
      3. Loads cached frozen model and RMS scaling from the checkpoint.
      4. Normalizes and runs batch CNN inference on CPU.
      5. Aggregates window predictions into a single file-level result.
      
    Args:
        path (str): Path to the signal file.
        config (PipelineConfig, optional): Input pipeline configuration overrides.
        model_path (str): Path to the trained CNN model checkpoint.
        top_k (int): Number of top predictions to report. Defaults to 5.
        
    Returns:
        SignalAnalysisResult: Structured result object.
        
    Raises:
        FileNotFoundError: If input path or model file does not exist.
        ValueError: If input format detection, parsing, or inference fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found at: '{path}'")
        
    if config is None:
        config = PipelineConfig()
        
    filename = os.path.basename(path)
    
    # 1. Load, parse, validate, and segment signal using input package
    segments, input_meta = process_file(path, config)
    
    if input_meta.validation_status == "ERROR":
        raise ValueError(f"Signal input validation error: {input_meta.error_message}")
        
    if len(segments) == 0:
        raise ValueError(
            f"Parsed signal contains no valid segments of length {config.segment_length}. "
            "Verification failed."
        )
        
    # 2. Retrieve cached frozen CNN baseline model and stored training RMS
    # get_cnn_model raises FileNotFoundError internally if checkpoint is missing
    model, rms_factor = get_cnn_model(model_path)
    
    # 3. Normalize segments using the stored checkpoint scaling factor ONLY
    # This prevents inconsistent inference across files
    segments_norm = segments / rms_factor
    
    # 4. Perform PyTorch CNN Batch Inference on CPU
    x_tensor = torch.tensor(segments_norm, dtype=torch.float32)
    
    try:
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1).numpy() # shape [M, 11]
    except Exception as e:
        raise ValueError(f"CNN Model inference failed: {str(e)}")
        
    # 5. Aggregate predictions across windows
    file_idx, file_name, file_conf, mean_probs, pred_dist, mean_ent = aggregate_window_probabilities(probs)
    
    # Extract window details
    window_preds = [INDEX_TO_MODULATION[int(idx)] for idx in np.argmax(probs, axis=1)]
    window_confidences = [float(c) for c in np.max(probs, axis=1)]
    window_probabilities = [list(p.astype(float)) for p in probs]
    
    # Top-k predictions
    top_preds = get_top_k_predictions(mean_probs, top_k=top_k)
    
    # Construct structured result
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
        processing_status="SUCCESS"
    )
