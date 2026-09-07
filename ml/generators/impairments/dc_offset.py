import numpy as np
from ml.generators.config import GeneratorConfig

def apply_dc_offset(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies complex baseband DC offset shift.
    
    y[n] = x[n] + (I_dc + j * Q_dc)
    
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array with DC offset applied.
    """
    dc_i = config.dc_offset_i if config.dc_offset_i is not None else 0.0
    dc_q = config.dc_offset_q if config.dc_offset_q is not None else 0.0
    
    if dc_i == 0.0 and dc_q == 0.0:
        return waveform.astype(np.complex64)
        
    dc_factor = np.complex64(dc_i + 1j * dc_q)
    
    return (waveform + dc_factor).astype(np.complex64)
