import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SignalExample:
    """
    Clear internal representation for one dataset example in SIGMA.
    
    Attributes:
        samples (np.ndarray): IQ samples of shape [2, 128] and dtype float32,
                              where samples[0] = I and samples[1] = Q.
        modulation (str): The modulation class name (e.g. '8PSK').
        snr (int): The Signal-to-Noise Ratio (SNR) value in dB.
        class_index (int): The stable integer class index mapping.
    """
    samples: np.ndarray
    modulation: str
    snr: int
    class_index: int

    def __post_init__(self):
        # Validate that samples is a numpy array
        if not isinstance(self.samples, np.ndarray):
            raise TypeError(f"samples must be a numpy ndarray, got {type(self.samples)}")
        
        # Verify shape is strictly [2, 128]
        if self.samples.shape != (2, 128):
            raise ValueError(f"samples shape must be exactly (2, 128), got {self.samples.shape}")
        
        # Verify type is float32
        if self.samples.dtype != np.float32:
            raise TypeError(f"samples dtype must be float32, got {self.samples.dtype}")
        
        # Verify SNR is an integer
        if not isinstance(self.snr, (int, np.integer)):
            raise TypeError(f"snr must be an integer, got {type(self.snr)}")

        # Verify modulation is a string
        if not isinstance(self.modulation, str):
            raise TypeError(f"modulation must be a string, got {type(self.modulation)}")

        # Verify class_index is an integer
        if not isinstance(self.class_index, (int, np.integer)):
            raise TypeError(f"class_index must be an integer, got {type(self.class_index)}")
