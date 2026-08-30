import numpy as np
from ml.generators.config import GeneratorConfig

def apply_iq_imbalance(waveform: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """
    Applies amplitude and phase imbalance between In-phase and Quadrature channels.
    
    Semantics of Amplitude Imbalance parameter A (config.iq_amplitude_imbalance):
      - Scaling gain for In-phase (I) channel is (1 + A).
      - Scaling gain for Quadrature (Q) channel is (1 - A).
      - E.g., for A = 0.1, I gain is 1.1 and Q gain is 0.9.
      - Relative I-to-Q amplitude scaling ratio is (1 + A) / (1 - A).
      
    Semantics of Phase Imbalance parameter theta (config.iq_phase_imbalance):
      - Represents the skew from orthogonality (in radians).
      
    Mathematical transformation model:
      [I_out]   [a11 a12] [I_in]
      [Q_out] = [a21 a22] [Q_in]
      
      where:
        a11 = (1 + A) * cos(theta / 2)
        a12 = -(1 + A) * sin(theta / 2)
        a21 = -(1 - A) * sin(theta / 2)
        a22 = (1 - A) * cos(theta / 2)
        
    If A = 0 and theta = 0, the matrix becomes the identity matrix, producing 
    no changes to the input waveform.
    
    Args:
        waveform (np.ndarray): 1D complex-valued NumPy array.
        config (GeneratorConfig): Waveform generation configuration.
        
    Returns:
        np.ndarray: 1D complex64 NumPy array representing the unbalanced signal.
    """
    A = config.iq_amplitude_imbalance if config.iq_amplitude_imbalance is not None else 0.0
    theta = config.iq_phase_imbalance if config.iq_phase_imbalance is not None else 0.0
    
    if A == 0.0 and theta == 0.0:
        return waveform.astype(np.complex64)
        
    # Split input complex waveform into real (I) and imaginary (Q) parts
    I_in = np.real(waveform)
    Q_in = np.imag(waveform)
    
    # Calculate 2x2 matrix coefficients
    a11 = (1.0 + A) * np.cos(theta / 2.0)
    a12 = -(1.0 + A) * np.sin(theta / 2.0)
    a21 = -(1.0 - A) * np.sin(theta / 2.0)
    a22 = (1.0 - A) * np.cos(theta / 2.0)
    
    # Apply 2x2 transformation matrix
    I_out = a11 * I_in + a12 * Q_in
    Q_out = a21 * I_in + a22 * Q_in
    
    return (I_out + 1j * Q_out).astype(np.complex64)
