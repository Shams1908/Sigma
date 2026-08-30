import numpy as np
from typing import Protocol, runtime_checkable, Dict, Optional
from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, SyntheticGroundTruth, complex_to_iq

@runtime_checkable
class Modulator(Protocol):
    """
    Abstract Protocol defining the interface for digital baseband modulators.
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        """
        Modulates a binary bit array into a complex-valued symbol sequence.
        
        Args:
            bits (np.ndarray): 1D array of binary bits (0 or 1).
            config (GeneratorConfig): Generator configuration.
            
        Returns:
            np.ndarray: 1D complex-valued array of symbols.
        """
        ...


def validate_bits(bits: np.ndarray, bits_per_symbol: int, modulation_name: str) -> None:
    """
    Validates input bits for binary correctness and length compatibility with modulation.
    
    Args:
        bits (np.ndarray): 1D NumPy array of bits.
        bits_per_symbol (int): Bits required per symbol.
        modulation_name (str): Modulation class name (for clear error messages).
        
    Raises:
        TypeError: If bits is not a NumPy array.
        ValueError: If bits is not 1D, contains values other than 0 and 1,
                    or its length is not divisible by bits_per_symbol.
    """
    if not isinstance(bits, np.ndarray):
        raise TypeError(f"bits must be a numpy ndarray, got {type(bits)}")
        
    if bits.ndim != 1:
        raise ValueError(f"bits must be a 1D array, got shape {bits.shape}")
        
    # Check that bits contain only 0 and 1
    # Expressed as vectorized logic: check if any element is not 0 and not 1
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"bits must contain only binary values 0 and 1.")
        
    # Reject incompatible bit lengths
    if len(bits) % bits_per_symbol != 0:
        raise ValueError(
            f"Bit length ({len(bits)}) is incompatible with {modulation_name} modulation. "
            f"Each symbol requires exactly {bits_per_symbol} bits. "
            f"Bit array length must be a multiple of {bits_per_symbol}."
        )


# Central registry of modulators
_MODULATORS: Dict[str, Modulator] = {}

def register_modulator(name: str, modulator: Modulator) -> None:
    """
    Registers a modulator for a specific modulation class.
    """
    _MODULATORS[name] = modulator

def get_modulator(name: str) -> Modulator:
    """
    Retrieves the registered modulator for the given modulation class name.
    """
    # Lazily import modulators to prevent any potential circular import issues
    if not _MODULATORS:
        from ml.generators.modulation.bpsk import BPSKModulator
        from ml.generators.modulation.qpsk import QPSKModulator
        from ml.generators.modulation.psk8 import EightPSKModulator
        from ml.generators.modulation.qam16 import QAM16Modulator
        from ml.generators.modulation.qam64 import QAM64Modulator

        register_modulator("BPSK", BPSKModulator())
        register_modulator("QPSK", QPSKModulator())
        register_modulator("8PSK", EightPSKModulator())
        register_modulator("QAM16", QAM16Modulator())
        register_modulator("QAM64", QAM64Modulator())

    if name not in _MODULATORS:
        raise ValueError(
            f"No modulator registered for '{name}'. "
            f"Supported modulations in M2.2: {list(_MODULATORS.keys())}"
        )
    return _MODULATORS[name]

def modulate(
    bits: np.ndarray, 
    config: GeneratorConfig, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Central dispatcher to modulate a sequence of bits using the configured modulation.
    
    Args:
        bits (np.ndarray): 1D array of binary bits (0 or 1).
        config (GeneratorConfig): Generator configuration.
        rng (np.random.Generator, optional): Unused here as modulation mapping is deterministic,
                                             but accepted for interface compliance.
                                             
    Returns:
        np.ndarray: 1D complex-valued array of symbols.
    """
    modulator = get_modulator(config.modulation)
    return modulator.modulate(bits, config)

def generate_modulated_signal(
    bits: np.ndarray, 
    config: GeneratorConfig,
    rng: Optional[np.random.Generator] = None
) -> GeneratedSignal:
    """
    Generates a modulated signal, converting baseband complex symbols to IQ representation
    and capturing the exact synthetic ground-truth parameters.
    
    Args:
        bits (np.ndarray): 1D array of binary bits (0 or 1).
        config (GeneratorConfig): Generator configuration.
        rng (Optional[np.random.Generator]): Seeded NumPy random generator.
        
    Returns:
        GeneratedSignal: Storable signal container.
    """
    # 1. Generate baseband complex symbols
    complex_symbols = modulate(bits, config)
    
    # 2. Apply pulse shaping if configured (with default fallback to ideal symbols)
    if config.rolloff is not None and config.filter_span_symbols is not None:
        from ml.generators.pulse_shaping.rrc import pulse_shape
        waveform = pulse_shape(complex_symbols, config)
    else:
        waveform = complex_symbols
        
    # 3. Apply AWGN channel if configured (after pulse shaping)
    if config.snr is not None:
        from ml.generators.channel.awgn import AWGNChannel
        channel_rng = rng if rng is not None else np.random.default_rng(config.random_seed)
        channel = AWGNChannel()
        waveform = channel.apply(waveform, config, channel_rng)

    # 4. Apply RF impairments composition
    from ml.generators.impairments import apply_impairments
    waveform = apply_impairments(waveform, config)

    # 5. Convert to project's canonical float32 shape [2, N]
    iq_samples = complex_to_iq(waveform)
    
    # 6. Create ground-truth metadata
    metadata = SyntheticGroundTruth(
        modulation=config.modulation,
        snr=config.snr,
        sample_rate=config.sample_rate,
        symbol_rate=config.symbol_rate,
        samples_per_symbol=config.samples_per_symbol,
        random_seed=config.random_seed,
        frequency_offset=config.frequency_offset,
        phase_offset=config.phase_offset,
        timing_offset=config.timing_offset,
        rolloff=config.rolloff,
        filter_span_symbols=config.filter_span_symbols,
        dc_offset_i=config.dc_offset_i,
        dc_offset_q=config.dc_offset_q,
        iq_amplitude_imbalance=config.iq_amplitude_imbalance,
        iq_phase_imbalance=config.iq_phase_imbalance
    )
    
    return GeneratedSignal(samples=iq_samples, metadata=metadata)
