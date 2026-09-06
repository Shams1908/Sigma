import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.channel.interface import Channel

class AWGNChannel(Channel):
    """
    Additive White Gaussian Noise (AWGN) channel model.
    
    Adds complex Gaussian noise to the input complex baseband signal.
    Noise power is computed dynamically based on the actual average power 
    of the input waveform and the requested SNR (in dB).
    """
    
    def apply(
        self,
        waveform: np.ndarray,
        config: GeneratorConfig,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Applies AWGN to the input complex baseband waveform.
        
        Args:
            waveform (np.ndarray): 1D complex-valued NumPy array.
            config (GeneratorConfig): Waveform generation configuration.
            rng (np.random.Generator): Seeded NumPy random generator.
            
        Returns:
            np.ndarray: 1D complex-valued NumPy array with AWGN noise added.
            
        Raises:
            TypeError: If input waveform is not a numpy array.
            ValueError: If waveform is not 1D, is empty, has zero power, or SNR is invalid.
        """
        if not isinstance(waveform, np.ndarray):
            raise TypeError(f"waveform must be a numpy array, got {type(waveform)}")
        if waveform.ndim != 1:
            raise ValueError(f"waveform must be 1D, got shape {waveform.shape}")
        if len(waveform) == 0:
            raise ValueError("waveform must not be empty")

        if config.snr is None:
            return waveform.astype(np.complex64)

        if not np.isfinite(config.snr):
            raise ValueError(f"snr must be a finite numeric value, got {config.snr}")

        # 1. Calculate actual average signal power
        # P_signal = mean(|x|^2)
        p_signal = float(np.mean(np.abs(waveform) ** 2))
        
        # Reject zero-power input
        if np.isclose(p_signal, 0.0) or p_signal <= 0.0:
            raise ValueError(
                "Input waveform has zero power. SNR is undefined for zero-power signals."
            )

        # 2. Calculate target total noise power
        # P_noise = P_signal / 10^(SNR_dB / 10)
        p_noise = p_signal / (10.0 ** (config.snr / 10.0))

        # 3. Generate complex Gaussian noise components
        # Each component (I and Q) receives exactly half of the total noise power
        # Var(n_I) = Var(n_Q) = P_noise / 2
        noise_std = np.sqrt(p_noise / 2.0)
        
        # Generate independent real and imaginary noise components
        noise_I = rng.normal(0.0, noise_std, size=len(waveform))
        noise_Q = rng.normal(0.0, noise_std, size=len(waveform))
        
        noise = (noise_I + 1j * noise_Q).astype(np.complex64)

        # 4. Return noisy complex waveform
        return (waveform + noise).astype(np.complex64)
