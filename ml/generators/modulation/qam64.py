import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import validate_bits

class QAM64Modulator:
    """
    64QAM Baseband Modulator (Gray-coded).
    
    Mapping convention (6 bits per symbol grouped as (b0, b1, b2, b3, b4, b5)):
      - b0, b1, b2 map to In-phase (I) level.
      - b3, b4, b5 map to Quadrature (Q) level.
      
    For each 3-bit level mapper (Gray coded):
      - 000 (index 0) -> +7.0
      - 001 (index 1) -> +5.0
      - 011 (index 3) -> +3.0
      - 010 (index 2) -> +1.0
      - 110 (index 6) -> -1.0
      - 111 (index 7) -> -3.0
      - 101 (index 5) -> -5.0
      - 100 (index 4) -> -7.0
      
    Average Symbol Power:
      - Raw average power = E[I^2 + Q^2] = 21.0 + 21.0 = 42.0.
      - Normalized to exactly 1.0 using the fixed constellation scaling factor: 1 / sqrt(42).
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        # Validate bits (6 bits per symbol)
        validate_bits(bits, bits_per_symbol=6, modulation_name="QAM64")
        
        # Reshape to symbol groups
        grouped = bits.reshape(-1, 6)
        
        # Split into I and Q bits
        i_bits = grouped[:, 0:3]
        q_bits = grouped[:, 3:6]
        
        # Convert binary groups to integer index 0..7
        idx_i = i_bits[:, 0] * 4 + i_bits[:, 1] * 2 + i_bits[:, 2]
        idx_q = q_bits[:, 0] * 4 + q_bits[:, 1] * 2 + q_bits[:, 2]
        
        # Level lookup table where adjacent binary values map to adjacent levels
        levels = np.array([7.0, 5.0, 1.0, 3.0, -7.0, -5.0, -1.0, -3.0], dtype=np.float32)
        
        # Compute unnormalized symbols
        symbols = levels[idx_i] + 1j * levels[idx_q]
        
        # Scale to unit average power using fixed constellation constant
        return (symbols / np.sqrt(42.0)).astype(np.complex64)
