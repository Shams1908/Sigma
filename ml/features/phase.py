import numpy as np

def extract_phase_features(x: np.ndarray) -> dict:
    """
    Extract phase-based features from a complex signal array.
    Supports both 1D arrays [128,] and 2D batch arrays [N, 128].
    
    Args:
        x (np.ndarray): Complex signal array (complex64 or complex128).
        
    Returns:
        dict: Dictionary containing extracted phase features.
    """
    # Force conversion to double precision for numerical stability
    x_complex = x.astype(np.complex128)
    
    # 1. Instantaneous Phase phi = angle(x)
    mag_x = np.abs(x_complex)
    phi = np.angle(x_complex)
    # Mask near-zero magnitude values to avoid undefined noisy phase
    phi = np.where(mag_x > 1e-9, phi, 0.0)
    
    # Wrapped Phase Variance
    var_phi = np.var(phi, axis=-1)
    
    # 2. Phase Differences dphi[n] = angle(x[n] * conj(x[n-1]))
    # Using ellipsis to support both 1D and 2D arrays automatically
    x_curr = x_complex[..., 1:]
    x_lag = x_complex[..., :-1]
    prod = x_curr * np.conj(x_lag)
    prod_mag = np.abs(prod)
    
    dphi = np.angle(prod)
    # Handle zero or near-zero magnitude cases in difference
    dphi = np.where(prod_mag > 1e-18, dphi, 0.0)
    
    # Statistics of phase differences
    mean_dphi = np.mean(dphi, axis=-1)
    var_dphi = np.var(dphi, axis=-1)
    
    # Excess Kurtosis of dphi
    if x.ndim == 2:
        dphi_diff = dphi - np.expand_dims(mean_dphi, axis=-1)
    else:
        dphi_diff = dphi - mean_dphi
        
    m4_dphi = np.mean(dphi_diff**4, axis=-1)
    kurtosis_dphi = np.where(var_dphi > 1e-9, m4_dphi / np.maximum(var_dphi**2, 1e-12) - 3.0, 0.0)
    
    # 3. Normalized 8-bin Phase Histogram covering [-pi, pi)
    # Shift to [0, 2*pi], scale to [0, 8)
    val = (phi + np.pi) / (2.0 * np.pi) * 8.0
    # Clip to [0, 8 - epsilon] and floor to get bin index 0..7
    val_clipped = np.clip(val, 0.0, 8.0 - 1e-9)
    bin_idx = np.floor(val_clipped).astype(np.int32)
    
    # Pack phase features
    features = {
        "phase_variance": var_phi.astype(np.float32),
        "phase_difference_mean": mean_dphi.astype(np.float32),
        "phase_difference_variance": var_dphi.astype(np.float32),
        "phase_difference_kurtosis": kurtosis_dphi.astype(np.float32),
    }
    
    # Populate the 8 bin features
    for k in range(8):
        features[f"phase_hist_bin_{k}"] = np.mean(bin_idx == k, axis=-1).astype(np.float32)
        
    return features
