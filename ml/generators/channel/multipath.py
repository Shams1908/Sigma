"""
Rayleigh Multipath and Frequency-Selective Fading Channel.

Simulates a discrete-time tapped-delay-line (TDL) multipath channel with
Rayleigh fading coefficients and an exponential power-delay profile.
"""
from typing import Optional, Union, Tuple
import numpy as np


class RayleighMultipathChannel:
    """
    Tapped-delay-line (TDL) Rayleigh multipath channel model.
    
    Attributes:
        num_taps: Number of discrete multipath taps (>= 1).
        decay_rate: Exponential power-delay profile decay factor tau_0 (> 0).
        random_seed: Optional integer seed for reproducible fading realizations.
    """
    def __init__(
        self,
        num_taps: int = 3,
        decay_rate: float = 1.5,
        random_seed: Optional[int] = None,
    ):
        if not isinstance(num_taps, (int, np.integer)):
            raise TypeError(f"num_taps must be an integer, got {type(num_taps)}")
        if num_taps < 1:
            raise ValueError(f"num_taps must be >= 1, got {num_taps}")
            
        if not isinstance(decay_rate, (int, float, np.number)):
            raise TypeError(f"decay_rate must be numeric, got {type(decay_rate)}")
        if decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
            
        self.num_taps = int(num_taps)
        self.decay_rate = float(decay_rate)
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def generate_taps(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generates complex FIR taps following a Rayleigh-distributed exponential profile.
        Normalized such that sum(|h_k|^2) = 1.0 (unit channel power).
        
        Returns:
            1D complex128 array of length num_taps.
        """
        active_rng = rng if rng is not None else self._rng
        
        # Exponential power delay profile: P(k) = exp(-k / decay_rate)
        indices = np.arange(self.num_taps, dtype=np.float64)
        power_profile = np.exp(-indices / self.decay_rate)
        
        # Complex Gaussian coefficients: h_k ~ CN(0, P(k))
        # Amplitude is Rayleigh, phase is uniform [0, 2pi)
        sigma = np.sqrt(power_profile / 2.0)
        h_real = active_rng.normal(0.0, sigma)
        h_imag = active_rng.normal(0.0, sigma)
        h = h_real + 1j * h_imag
        
        # Normalize total energy to 1.0
        total_energy = np.sum(np.abs(h)**2)
        if total_energy > 0:
            h = h / np.sqrt(total_energy)
        else:
            h[0] = 1.0 + 0j
            
        return h

    def apply(
        self,
        samples: np.ndarray,
        taps: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies multipath fading to input IQ samples.
        
        Args:
            samples: Input array of shape [2, N] float32.
            taps: Optional precomputed FIR taps. If None, generates new taps.
            rng: Optional explicit random generator for tap realization.
            
        Returns:
            Tuple of (faded_samples [2, N], applied_taps [num_taps]).
        """
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"samples must be a numpy ndarray, got {type(samples)}")
        if samples.ndim != 2 or samples.shape[0] != 2:
            raise ValueError(f"samples must have shape [2, N], got {samples.shape}")
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Input samples contain NaN or Infinite values.")

        if taps is None:
            taps = self.generate_taps(rng=rng)
            
        N = samples.shape[1]
        z = samples[0] + 1j * samples[1]
        
        # Discrete convolution: mode='same' keeps output centered at original sample points
        z_faded = np.convolve(z, taps, mode="same")
        
        out = np.empty((2, N), dtype=np.float32)
        out[0] = np.real(z_faded).astype(np.float32)
        out[1] = np.imag(z_faded).astype(np.float32)
        
        return out, taps
