import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import validate_bits

class BPSKModulator:
    """
    BPSK Baseband Modulator.
    
    Mapping convention:
      - Bit 0 -> -1.0 + 0.0j
      - Bit 1 -> +1.0 + 0.0j
      
    Average Symbol Power:
      - Exactly 1.0 (no normalization scaling required).
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        # Validate bits (1 bit per symbol)
        validate_bits(bits, bits_per_symbol=1, modulation_name="BPSK")
        
        # Vectorized mapping: 0 -> -1, 1 -> +1
        symbols = 2.0 * bits - 1.0
        
        # Cast to complex64 for baseband output representation
        return symbols.astype(np.complex64)
