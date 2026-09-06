import numpy as np

def segment_canonical_iq(
    iq: np.ndarray,
    segment_length: int = 128,
    pad_short: bool = False
) -> np.ndarray:
    """
    Segments a canonical [2, N] IQ array into non-overlapping windows of shape [M, 2, segment_length].
    
    Args:
        iq (np.ndarray): Canonical float32 IQ array of shape [2, N].
        segment_length (int): Expected window length. Defaults to 128.
        pad_short (bool): If True, pads signals shorter than segment_length with zeros.
                          If False, raises ValueError for short signals.
                          
    Returns:
        np.ndarray: Windowed segments of shape [M, 2, segment_length].
        
    Raises:
        ValueError: If signal is shorter than segment_length and pad_short is False.
    """
    if iq.ndim != 2 or iq.shape[0] != 2:
        raise ValueError(f"Input must be canonical shape [2, N], got shape {iq.shape}")
        
    n_samples = iq.shape[1]
    
    if n_samples < segment_length:
        if pad_short:
            # Pad with trailing zeros to reach segment_length
            padded = np.zeros((2, segment_length), dtype=np.float32)
            padded[:, :n_samples] = iq
            return np.expand_dims(padded, axis=0)
        else:
            raise ValueError(
                f"Signal length ({n_samples} samples) is shorter than the expected segment "
                f"length of {segment_length} samples. Silent padding is disabled by default."
            )
            
    # Calculate number of complete segments
    num_segments = n_samples // segment_length
    segments = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(iq[:, start:end])
        
    return np.array(segments, dtype=np.float32)
