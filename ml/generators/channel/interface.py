from typing import Protocol
import numpy as np
from ml.generators.config import GeneratorConfig

class Channel(Protocol):
    """
    Abstract Protocol defining the interface for channel models.
    
    Any channel model (e.g., AWGN, Fading, Freq/Phase offset) must implement
    this interface to integrate with the synthetic generator pipeline.
    """
    def apply(
        self,
        waveform: np.ndarray,
        config: GeneratorConfig,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Applies the channel model to the input complex baseband waveform.
        
        Args:
            waveform (np.ndarray): 1D complex-valued NumPy array.
            config (GeneratorConfig): Waveform generation configuration.
            rng (np.random.Generator): Seeded NumPy random generator.
            
        Returns:
            np.ndarray: 1D complex-valued NumPy array representing the noisy/distorted signal.
        """
        ...
