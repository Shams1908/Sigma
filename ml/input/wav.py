import numpy as np
from typing import Tuple
from scipy.io import wavfile

def parse_wav_file(path: str) -> Tuple[np.ndarray, float, int]:
    """
    Parses a stereo WAV file. Extracts the sample rate, converts the data
    to canonical IQ format [2, N], and normalizes integer sample values to float32 [-1.0, 1.0].
    
    Args:
        path (str): Path to the WAV file.
        
    Returns:
        Tuple[np.ndarray, float, int]: 
            - canonical_iq: Array of shape [2, N] (I/Q channels)
            - sample_rate: The sample rate of the WAV file
            - original_sample_count: The number of frames (N)
            
    Raises:
        ValueError: If file is mono, empty, contains invalid data types, or contains NaNs/Infs.
    """
    try:
        sample_rate, data = wavfile.read(path)
    except Exception as e:
        raise ValueError(f"Failed to read WAV file structure: {str(e)}")
        
    # Verify stereo layout
    if data.ndim == 1:
        raise ValueError(
            "WAV layout is mono (1 channel). The signal input pipeline requires "
            "stereo WAV where channel 0 -> In-phase (I) and channel 1 -> Quadrature (Q)."
        )
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Unsupported WAV channel layout. Expected exactly 2 channels (stereo), "
            f"got {data.shape[1]} channels."
        )
        
    n_samples = data.shape[0]
    if n_samples == 0:
        raise ValueError("WAV file contains no signal samples.")
        
    # Normalize integers to float32 in range [-1.0, 1.0] dynamically depending on the type
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        # Using maximum absolute range (e.g. 32768 for int16)
        denom = max(abs(info.min), info.max)
        iq_data = data.astype(np.float32) / float(denom)
    elif np.issubdtype(data.dtype, np.floating):
        iq_data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample encoding data type: {data.dtype}")
        
    # Final NaN/Inf check
    if np.isnan(iq_data).any() or np.isinf(iq_data).any():
        raise ValueError("Parsed WAV samples contain NaNs or infinite values.")
        
    # Convert [N, 2] -> canonical shape [2, N]
    canonical_iq = iq_data.T
    
    return canonical_iq, float(sample_rate), n_samples
