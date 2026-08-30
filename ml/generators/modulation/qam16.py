import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import validate_bits

class QAM16Modulator:
    """
    16QAM Baseband Modulator (Gray-coded).
    
    Mapping convention (4 bits per symbol grouped as (b0, b1, b2, b3)):
      - b0, b1 map to In-phase (I) level.
      - b2, b3 map to Quadrature (Q) level.
      
    For each 2-bit level mapper (Gray coded):
      - 00 (index 0) -> +3.0
      - 01 (index 1) -> +1.0
      - 11 (index 3) -> -1.0
      - 10 (index 2) -> -3.0
      
    Average Symbol Power:
      - Raw average power = E[I^2 + Q^2] = 5.0 + 5.0 = 10.0.
      - Normalized to exactly 1.0 using the fixed constellation scaling factor: 1 / sqrt(10).
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        # Validate bits (4 bits per symbol)
        validate_bits(bits, bits_per_symbol=4, modulation_name="QAM16")
        
        # Reshape to symbol groups
        grouped = bits.reshape(-1, 4)
        
        # Split into I and Q bits
        i_bits = grouped[:, 0:2]
        q_bits = grouped[:, 2:4]
        
        # Convert binary groups to integer index 0..3
        idx_i = i_bits[:, 0] * 2 + i_bits[:, 1]
        idx_q = q_bits[:, 0] * 2 + q_bits[:, 1]
        
        # Level lookup table where adjacent binary values map to adjacent levels
        levels = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float32)
        
        # Compute unnormalized symbols
        symbols = levels[idx_i] + 1j * levels[idx_q]
        
        # Scale to unit average power using fixed constellation constant
        return (symbols / np.sqrt(10.0)).astype(np.complex64)
