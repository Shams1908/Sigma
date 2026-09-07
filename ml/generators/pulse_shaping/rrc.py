import numpy as np
from ml.generators.config import GeneratorConfig

def design_rrc_filter(
    samples_per_symbol: int,
    rolloff: float,
    filter_span_symbols: int
) -> np.ndarray:
    """
    Designs a symmetric Root Raised Cosine (RRC) filter.
    
    The filter is symmetric around zero with taps count = filter_span_symbols * samples_per_symbol + 1.
    Taps are energy-normalized to satisfy sum(h[n]^2) = 1.0.
    
    Singularities at t = 0 and t = ±T / (4 * rolloff) are explicitly handled.
    
    Args:
        samples_per_symbol (int): Number of samples per symbol (upsampling rate).
        rolloff (float): Rolloff factor, satisfying 0.0 <= rolloff <= 1.0.
        filter_span_symbols (int): Filter span in symbols (positive integer).
        
    Returns:
        np.ndarray: 1D float32 array containing the RRC filter taps.
        
    Raises:
        TypeError: If samples_per_symbol or filter_span_symbols is not an integer.
        ValueError: If configuration values are physically or mathematically invalid.
    """
    if not isinstance(samples_per_symbol, (int, np.integer)):
        raise TypeError(f"samples_per_symbol must be an integer, got {type(samples_per_symbol)}")
    if samples_per_symbol <= 0:
        raise ValueError(f"samples_per_symbol must be greater than 0, got {samples_per_symbol}")
        
    if not isinstance(filter_span_symbols, (int, np.integer)):
        raise TypeError(f"filter_span_symbols must be an integer, got {type(filter_span_symbols)}")
    if filter_span_symbols <= 0:
        raise ValueError(f"filter_span_symbols must be greater than 0, got {filter_span_symbols}")
        
    if not isinstance(rolloff, (int, float, np.number)):
        raise TypeError(f"rolloff must be numeric, got {type(rolloff)}")
    if not (0.0 <= rolloff <= 1.0):
        raise ValueError(f"rolloff must be between 0.0 and 1.0, got {rolloff}")

    # Tap calculations centered around zero index
    num_taps = filter_span_symbols * samples_per_symbol + 1
    half_taps = (num_taps - 1) // 2
    n = np.arange(-half_taps, half_taps + 1, dtype=np.float32)
    
    # Normalized time variable: tau = t / T = n / SPS
    tau = n / samples_per_symbol
    taps = np.empty(len(n), dtype=np.float32)
    
    for i, t in enumerate(tau):
        # 1. Singularity at t = 0
        if np.isclose(t, 0.0):
            taps[i] = 1.0 - rolloff + (4.0 * rolloff / np.pi)
            
        # 2. Singularity at t = ±T / (4 * rolloff)
        elif rolloff > 0.0 and np.isclose(np.abs(t), 1.0 / (4.0 * rolloff)):
            val = (rolloff / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
            )
            taps[i] = val
            
        # 3. Standard RRC tap calculation
        else:
            numerator = (
                np.sin(np.pi * t * (1.0 - rolloff)) + 
                4.0 * rolloff * t * np.cos(np.pi * t * (1.0 + rolloff))
            )
            denominator = np.pi * t * (1.0 - (4.0 * rolloff * t) ** 2)
            taps[i] = numerator / denominator

    # Normalize filter taps to unit energy: sum(h[n]^2) = 1.0
    energy = np.sum(taps ** 2)
    if energy > 0:
        taps = taps / np.sqrt(energy)
        
    return taps


def pulse_shape(symbols: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies zero-upsampling and Root Raised Cosine filtering to input baseband symbols.
    
    Args:
        symbols (np.ndarray): 1D complex-valued NumPy array of baseband symbols.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array containing the pulse-shaped waveform.
    """
    if config.rolloff is None or config.filter_span_symbols is None:
        raise ValueError(
            "Pulse shaping requires both 'rolloff' and 'filter_span_symbols' to be set in configuration."
        )

    # 1. Design RRC filter
    taps = design_rrc_filter(
        samples_per_symbol=config.samples_per_symbol,
        rolloff=config.rolloff,
        filter_span_symbols=config.filter_span_symbols
    )

    # 2. Upsample: insert (samples_per_symbol - 1) zeros after each symbol
    sps = config.samples_per_symbol
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    # 3. Filter using full linear convolution (preserves delay and transients)
    # Mode 'full' size = len(upsampled) + len(taps) - 1
    filtered = np.convolve(upsampled, taps, mode="full")

    return filtered.astype(np.complex64)
