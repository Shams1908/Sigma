from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class SyntheticGroundTruth:
    """
    Metadata capturing the exact ground truth parameters of a synthetically generated signal.
    
    This is intended strictly to keep track of the noise-free physical settings 
    under which the signal was simulated.
    """
    modulation: str
    snr: Optional[float]
    sample_rate: float
    symbol_rate: float
    samples_per_symbol: int
    random_seed: int
    frequency_offset: Optional[float] = None
    phase_offset: Optional[float] = None
    timing_offset: Optional[float] = None
    rolloff: Optional[float] = None
    filter_span_symbols: Optional[int] = None
    dc_offset_i: Optional[float] = None
    dc_offset_q: Optional[float] = None
    iq_amplitude_imbalance: Optional[float] = None
    iq_phase_imbalance: Optional[float] = None


@dataclass(frozen=True)
class GeneratedSignal:
    """
    Container representing a synthetically generated signal before serialization.
    
    Attributes:
        samples (np.ndarray): IQ samples of shape [2, N] and dtype float32,
                              where samples[0] = I and samples[1] = Q.
        metadata (SyntheticGroundTruth): Ground truth parameters used during simulation.
    """
    samples: np.ndarray
    metadata: SyntheticGroundTruth

    def __post_init__(self):
        if not isinstance(self.samples, np.ndarray):
            raise TypeError(f"samples must be a numpy ndarray, got {type(self.samples)}")
        
        if self.samples.ndim != 2 or self.samples.shape[0] != 2:
            raise ValueError(
                f"samples shape must be of the canonical format [2, N], "
                f"got shape: {self.samples.shape}"
            )
            
        if self.samples.dtype != np.float32:
            raise TypeError(f"samples dtype must be float32, got {self.samples.dtype}")
            
        if not isinstance(self.metadata, SyntheticGroundTruth):
            raise TypeError(
                f"metadata must be a SyntheticGroundTruth instance, "
                f"got {type(self.metadata)}"
            )


def complex_to_iq(complex_samples: np.ndarray) -> np.ndarray:
    """
    Converts a 1D complex NumPy array to the project's canonical [2, N] IQ representation.
    
    Args:
        complex_samples (np.ndarray): Complex array of shape (N,).
        
    Returns:
        np.ndarray: Float32 array of shape (2, N) where:
                    - samples[0] = In-phase (I) channel
                    - samples[1] = Quadrature (Q) channel
                    
    Raises:
        TypeError: If complex_samples is not a NumPy array or is not complex.
    """
    if not isinstance(complex_samples, np.ndarray):
        raise TypeError(f"complex_samples must be a numpy ndarray, got {type(complex_samples)}")
        
    if not np.iscomplexobj(complex_samples):
        raise TypeError(
            f"complex_samples must have a complex dtype, "
            f"got: {complex_samples.dtype}"
        )
        
    N = len(complex_samples)
    iq = np.empty((2, N), dtype=np.float32)
    iq[0] = np.real(complex_samples).astype(np.float32)
    iq[1] = np.imag(complex_samples).astype(np.float32)
    return iq


def iq_to_complex(iq_samples: np.ndarray) -> np.ndarray:
    """
    Converts the project's canonical [2, N] IQ representation to a 1D complex array.
    
    Args:
        iq_samples (np.ndarray): Float32 array of shape (2, N).
        
    Returns:
        np.ndarray: Complex64 array of shape (N,).
        
    Raises:
        TypeError: If iq_samples is not a NumPy array.
        ValueError: If iq_samples shape is not [2, N].
    """
    if not isinstance(iq_samples, np.ndarray):
        raise TypeError(f"iq_samples must be a numpy ndarray, got {type(iq_samples)}")
        
    if iq_samples.ndim != 2 or iq_samples.shape[0] != 2:
        raise ValueError(
            f"iq_samples must have shape [2, N], "
            f"got shape: {iq_samples.shape}"
        )
        
    # Reconstruct complex numbers losslessly
    return iq_samples[0].astype(np.complex64) + 1j * iq_samples[1].astype(np.complex64)
