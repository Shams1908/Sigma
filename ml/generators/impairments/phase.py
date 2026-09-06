import numpy as np
from ml.generators.config import GeneratorConfig

def apply_phase_offset(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies carrier phase offset rotation to a complex baseband signal.
    
    y[n] = x[n] * exp(j * phi)
    
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array with phase offset applied.
    """
    if config.phase_offset is None or config.phase_offset == 0.0:
        return waveform.astype(np.complex64)
        
    rotation_factor = np.exp(1j * config.phase_offset).astype(np.complex64)
    
    return (waveform * rotation_factor).astype(np.complex64)
