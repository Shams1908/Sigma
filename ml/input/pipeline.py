import os
import numpy as np
from typing import Tuple, Optional

from ml.input.types import PipelineConfig, BinaryIQConfig, SignalMetadata, DEFAULT_MAX_FILE_SIZE_BYTES, DEFAULT_SEGMENT_LENGTH
from ml.input.detector import detect_file_format
from ml.input.wav import parse_wav_file
from ml.input.iq import parse_iq_file
from ml.input.segment import segment_canonical_iq

def detect_format(path: str, max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> str:
    """
    Exposes format detection for files.
    """
    return detect_file_format(path, max_size_bytes)

def load_signal(
    path: str,
    format_type: str,
    binary_config: Optional[BinaryIQConfig] = None
) -> Tuple[np.ndarray, Optional[float], Optional[int]]:
    """
    Loads raw files and converts them to a canonical float32 [2, N] array.
    """
    if format_type == "WAV":
        canonical_iq, sample_rate, _ = parse_wav_file(path)
        return canonical_iq, sample_rate, 2
    else:
        canonical_iq, _ = parse_iq_file(path, format_type, binary_config)
        return canonical_iq, None, None

def segment_iq(
    iq: np.ndarray,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    pad_short: bool = False
) -> np.ndarray:
    """
    Segments canonical [2, N] IQ data into non-overlapping [M, 2, segment_length] windows.
    """
    return segment_canonical_iq(iq, segment_length, pad_short)

def process_file(
    path: str,
    config: Optional[PipelineConfig] = None
) -> Tuple[np.ndarray, SignalMetadata]:
    """
    Executes the complete input adapter pipeline:
    file -> format detection -> parsing -> validation -> canonical IQ [2, N] -> segmentation [M, 2, L].
    
    Args:
        path (str): Path to the input file.
        config (PipelineConfig, optional): Configuration overrides.
        
    Returns:
        Tuple[np.ndarray, SignalMetadata]:
            - segments: Array of shape [M, 2, segment_length]
            - metadata: SignalMetadata object containing details about the execution
    """
    if config is None:
        config = PipelineConfig()
        
    filename = os.path.basename(path)
    
    try:
        # 1. Format detection & size validation
        fmt = detect_format(path, config.max_file_size_bytes)
        
        # 2. Parsing & target normalization to [2, N]
        canonical_iq, sample_rate, num_channels = load_signal(path, fmt, config.binary_config)
        
        # 3. Segmentation to [M, 2, L]
        segments = segment_iq(canonical_iq, config.segment_length, pad_short=False)
        
        # Assemble success metadata
        meta = SignalMetadata(
            source_filename=filename,
            detected_format=fmt,
            sample_rate=sample_rate,
            num_channels=num_channels,
            original_sample_count=canonical_iq.shape[1],
            canonical_iq_shape=list(canonical_iq.shape),
            num_generated_segments=len(segments),
            dtype=str(canonical_iq.dtype),
            validation_status="OK"
        )
        return segments, meta
        
    except Exception as e:
        # Assemble error metadata
        meta = SignalMetadata(
            source_filename=filename,
            detected_format="UNKNOWN",
            sample_rate=None,
            num_channels=None,
            original_sample_count=0,
            canonical_iq_shape=[0, 0],
            num_generated_segments=0,
            dtype="unknown",
            validation_status="ERROR",
            error_message=str(e)
        )
        return np.empty((0, 2, config.segment_length), dtype=np.float32), meta
