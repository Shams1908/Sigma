import numpy as np
from ml.generators.config import GeneratorConfig

def apply_timing_offset(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies fractional timing offset to a complex baseband signal using linear interpolation.
    
    Timing Offset Sign/Direction Convention:
      - A positive timing offset (tau > 0) delays the signal (shifts it to the right).
      - E.g., tau = 1.0 shifts x[n] to x[n - 1].
      
    Interpolation algorithm:
      Let:
        k = floor(tau)
        d = tau - k
        
      The output signal y[n] is computed as:
        y[n] = (1 - d) * x[n - k] + d * x[n - k - 1]
        
      Out-of-range samples (indices < 0 or >= len(waveform)) are zero-padded.
      
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array with timing offset applied.
    """
    tau = config.timing_offset
    if tau is None or tau == 0.0:
        return waveform.astype(np.complex64)
        
    N = len(waveform)
    k = int(np.floor(tau))
    d = float(tau - k)
    
    # Secure zero-padded indexing helper
    def get_val(indices: np.ndarray) -> np.ndarray:
        mask = (indices >= 0) & (indices < N)
        clipped_indices = np.clip(indices, 0, N - 1)
        return np.where(mask, waveform[clipped_indices], 0.0 + 0.0j)
        
    n = np.arange(N, dtype=np.int32)
    val_k = get_val(n - k)
    val_k_1 = get_val(n - k - 1)
    
    y = (1.0 - d) * val_k + d * val_k_1
    return y.astype(np.complex64)
