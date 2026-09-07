import numpy as np

def extract_cumulant_features(x: np.ndarray) -> dict:
    """
    Extract normalized C40 cumulant features from a complex signal array.
    Supports both 1D arrays [128,] and 2D batch arrays [N, 128].
    
    Formula:
        C40_raw = E[x^4] - 3(E[x^2])^2
        C40_norm = C40_raw / E[|x|^2]^2
        
    Args:
        x (np.ndarray): Complex signal array.
        
    Returns:
        dict: Dictionary containing c40_real, c40_imag, and c40_magnitude.
    """
    # Force conversion to double precision for numerical stability
    x_complex = x.astype(np.complex128)
    
    # Calculate expectations
    E_x4 = np.mean(x_complex**4, axis=-1)
    E_x2 = np.mean(x_complex**2, axis=-1)
    E_abs_x2 = np.mean(np.abs(x_complex)**2, axis=-1)
    
    # Compute raw C40 cumulant
    C40_raw = E_x4 - 3.0 * (E_x2**2)
    
    # Normalize by the squared average power E[|x|^2]^2
    # Handle zero-power signals safely to avoid NaNs/Infs
    denom = E_abs_x2**2
    C40_norm = np.where(E_abs_x2 > 1e-9, C40_raw / np.maximum(denom, 1e-18), 0.0 + 0.0j)
    
    return {
        "c40_real": np.real(C40_norm).astype(np.float32),
        "c40_imag": np.imag(C40_norm).astype(np.float32),
        "c40_magnitude": np.abs(C40_norm).astype(np.float32)
    }
