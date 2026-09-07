import numpy as np
from ml.generators.config import GeneratorConfig

def apply_frequency_offset(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies carrier frequency offset rotation to a complex baseband signal.
    
    y[n] = x[n] * exp(j * 2 * pi * delta_f * n / Fs)
    
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array with frequency offset applied.
        
    Raises:
        ValueError: If sample_rate is not positive or configuration properties are invalid.
    """
    if config.frequency_offset is None or config.frequency_offset == 0.0:
        return waveform.astype(np.complex64)
        
    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be greater than 0, got {config.sample_rate}")
        
    n = np.arange(len(waveform), dtype=np.float32)
    phase_rotation = 2.0 * np.pi * config.frequency_offset * n / config.sample_rate
    rotation_factor = np.exp(1j * phase_rotation).astype(np.complex64)
    
    return (waveform * rotation_factor).astype(np.complex64)
