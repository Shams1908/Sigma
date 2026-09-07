from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

# Centralized constants to avoid magic numbers
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_SEGMENT_LENGTH = 128

@dataclass
class BinaryIQConfig:
    """
    Configuration required to parse raw interleaved binary IQ files (.bin, .dat).
    Prevents silent guessing.
    """
    dtype: str          # e.g., 'float32', 'float64', 'int16', 'int32'
    interleaved: bool   # True for I0, Q0, I1, Q1... ; False for I0, I1... then Q0, Q1...
    endianness: str = "little"  # 'little' or 'big'

@dataclass
class PipelineConfig:
    """
    Configuration for the input signal adapter.
    """
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    segment_length: int = DEFAULT_SEGMENT_LENGTH
    binary_config: Optional[BinaryIQConfig] = None

@dataclass
class SignalMetadata:
    """
    Downstream metadata dictionary representation for the parsed and validated signal.
    """
    source_filename: str
    detected_format: str
    sample_rate: Optional[float]
    num_channels: Optional[int]
    original_sample_count: int
    canonical_iq_shape: List[int]
    num_generated_segments: int
    dtype: str
    validation_status: str
    error_message: Optional[str] = None
