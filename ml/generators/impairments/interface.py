from typing import Protocol
import numpy as np
from ml.generators.config import GeneratorConfig

class Impairment(Protocol):
    """
    Abstract Protocol defining the interface for individual RF baseband impairments.
    """
    def __call__(self, waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        ...


def apply_impairments(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Composes and applies multiple RF impairments sequentially in a strict, documented order:
    
      1. Frequency offset (apply_frequency_offset)
      2. Phase offset (apply_phase_offset)
      3. IQ imbalance (apply_iq_imbalance)
      4. DC offset (apply_dc_offset)
      5. Timing offset (apply_timing_offset)
      
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array representing the impaired signal.
    """
    # Import impairment functions
    from ml.generators.impairments.frequency import apply_frequency_offset
    from ml.generators.impairments.phase import apply_phase_offset
    from ml.generators.impairments.iq_imbalance import apply_iq_imbalance
    from ml.generators.impairments.dc_offset import apply_dc_offset
    from ml.generators.impairments.timing import apply_timing_offset

    # Apply impairments in strict sequence order
    out = waveform
    out = apply_frequency_offset(out, config)
    out = apply_phase_offset(out, config)
    out = apply_iq_imbalance(out, config)
    out = apply_dc_offset(out, config)
    out = apply_timing_offset(out, config)
    
    return out.astype(np.complex64)
