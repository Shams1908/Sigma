import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import validate_bits

class QPSKModulator:
    """
    QPSK Baseband Modulator (Gray-coded).
    
    Mapping convention (2 bits per symbol grouped as (b0, b1)):
      - b0 b1 = 00 -> (+1 + 1j) / sqrt(2)
      - b0 b1 = 01 -> (-1 + 1j) / sqrt(2)
      - b0 b1 = 11 -> (-1 - 1j) / sqrt(2)
      - b0 b1 = 10 -> (+1 - 1j) / sqrt(2)
      
    Average Symbol Power:
      - Normalized to exactly 1.0 using the fixed constellation scaling factor: 1 / sqrt(2).
      
    This constellation is Gray-coded since adjacent quadrant states differ by exactly 1 bit.
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        # Validate bits (2 bits per symbol)
        validate_bits(bits, bits_per_symbol=2, modulation_name="QPSK")
        
        # Reshape to symbol groups
        grouped = bits.reshape(-1, 2)
        
        # Convert binary groups to integer index 0..3
        idx = grouped[:, 0] * 2 + grouped[:, 1]
        
        constellation = (np.array([
            1.0 + 1.0j,    # index 0 (00)
            -1.0 + 1.0j,   # index 1 (01)
            1.0 - 1.0j,    # index 2 (10)
            -1.0 - 1.0j    # index 3 (11)
        ], dtype=np.complex64) / np.sqrt(2.0)).astype(np.complex64)
        
        return constellation[idx]
