import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import validate_bits

class EightPSKModulator:
    """
    8PSK Baseband Modulator (Gray-coded).
    
    Mapping convention (3 bits per symbol grouped as (b0, b1, b2)):
      - b0 b1 b2 = 000 (index 0) -> exp(1j * 0 * pi / 4)
      - b0 b1 b2 = 001 (index 1) -> exp(1j * 1 * pi / 4)
      - b0 b1 b2 = 011 (index 3) -> exp(1j * 2 * pi / 4)
      - b0 b1 b2 = 010 (index 2) -> exp(1j * 3 * pi / 4)
      - b0 b1 b2 = 110 (index 6) -> exp(1j * 4 * pi / 4)
      - b0 b1 b2 = 111 (index 7) -> exp(1j * 5 * pi / 4)
      - b0 b1 b2 = 101 (index 5) -> exp(1j * 6 * pi / 4)
      - b0 b1 b2 = 100 (index 4) -> exp(1j * 7 * pi / 4)
      
    Average Symbol Power:
      - Exactly 1.0 (as all symbols reside on the unit circle, no extra normalization is needed).
    """
    def modulate(self, bits: np.ndarray, config: GeneratorConfig) -> np.ndarray:
        # Validate bits (3 bits per symbol)
        validate_bits(bits, bits_per_symbol=3, modulation_name="8PSK")
        
        # Reshape to symbol groups
        grouped = bits.reshape(-1, 3)
        
        # Convert binary groups to integer index 0..7
        idx = grouped[:, 0] * 4 + grouped[:, 1] * 2 + grouped[:, 2]
        
        # Phase lookup map following Gray-coded sequence order
        phase_indices = np.array([0, 1, 3, 2, 7, 6, 4, 5], dtype=np.int32)
        
        # Compute angles
        angles = phase_indices[idx] * np.pi / 4.0
        
        # Return complex exp
        return (np.cos(angles) + 1j * np.sin(angles)).astype(np.complex64)
