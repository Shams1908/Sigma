"""
generate_test_signals.py
 
Generates a small set of synthetic BPSK/QPSK test signals as stereo WAV
files (channel 0 = I, channel 1 = Q), matching exactly what
ml/input/wav.py expects: stereo, int16 PCM, normalized to [-1, 1].
 
Run this from your project root:
    python generate_test_signals.py
 
Output: a `test_signals/` folder containing several .wav files with
known ground truth (modulation + SNR baked into the filename).
 
Requires: numpy, scipy  (both already in your requirements.txt)
"""
 
import os
import numpy as np
from scipy.io import wavfile
 
 
def generate_bpsk_qpsk(modulation: str, num_symbols: int, sps: int,
                        snr_db: float, seed: int = 42) -> np.ndarray:
    """Generate a baseband complex IQ signal for BPSK or QPSK with AWGN."""
    rng = np.random.default_rng(seed)
 
    if modulation == "BPSK":
        bits = rng.integers(0, 2, num_symbols)
        symbols = np.where(bits == 0, -1.0, 1.0).astype(np.complex64)
 
    elif modulation == "QPSK":
        bits = rng.integers(0, 2, num_symbols * 2)
        symbols = np.zeros(num_symbols, dtype=np.complex64)
        norm = 1.0 / np.sqrt(2)
        # Gray mapping matching the project's documented convention:
        # 00 -> (1+1j), 01 -> (-1+1j), 11 -> (-1-1j), 10 -> (1-1j)
        for i in range(num_symbols):
            b0, b1 = bits[2 * i], bits[2 * i + 1]
            if b0 == 0 and b1 == 0:
                s = 1 + 1j
            elif b0 == 0 and b1 == 1:
                s = -1 + 1j
            elif b0 == 1 and b1 == 1:
                s = -1 - 1j
            else:
                s = 1 - 1j
            symbols[i] = s * norm
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
 
    # Simple rectangular pulse shaping (upsample by holding each symbol
    # for `sps` samples). Fine for a straightforward smoke test; swap in
    # your ml/generators RRC pulse shaping later if you want a closer
    # match to the real synthetic generator's output.
    tx = np.repeat(symbols, sps)
 
    # Add AWGN at the target SNR
    signal_power = np.mean(np.abs(tx) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = (
        rng.normal(0, np.sqrt(noise_power / 2), tx.shape)
        + 1j * rng.normal(0, np.sqrt(noise_power / 2), tx.shape)
    )
    rx = tx + noise
    return rx.astype(np.complex64)
 
 
def save_as_wav(iq: np.ndarray, sample_rate: int, path: str) -> None:
    """Save a complex IQ array as a stereo int16 WAV: ch0=I, ch1=Q."""
    i_channel = iq.real
    q_channel = iq.imag
 
    peak = max(np.max(np.abs(i_channel)), np.max(np.abs(q_channel)), 1e-9)
    i_norm = (i_channel / peak) * 0.9
    q_norm = (q_channel / peak) * 0.9
 
    stereo = np.stack([i_norm, q_norm], axis=1)
    int16_data = (stereo * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, int16_data)
 
 
def main():
    out_dir = "test_signals"
    os.makedirs(out_dir, exist_ok=True)
 
    sample_rate = 48000
    sps = 8              # samples per symbol
    num_symbols = 2000
 
    # (modulation, snr_db, filename)
    configs = [
        ("BPSK", 20, "bpsk_clean_snr20.wav"),
        ("BPSK", 5,  "bpsk_noisy_snr5.wav"),
        ("QPSK", 20, "qpsk_clean_snr20.wav"),
        ("QPSK", 5,  "qpsk_noisy_snr5.wav"),
    ]
 
    for modulation, snr_db, filename in configs:
        iq = generate_bpsk_qpsk(modulation, num_symbols, sps, snr_db)
        path = os.path.join(out_dir, filename)
        save_as_wav(iq, sample_rate, path)
        print(f"Saved {path}  (modulation={modulation}, "
              f"target_snr={snr_db}dB, sample_rate={sample_rate}, "
              f"symbol_rate={sample_rate/sps:.1f} baud)")
 
    # One deliberately broken file, to test the failure path:
    # random noise, no real modulation structure at all.
    rng = np.random.default_rng(1)
    garbage = (rng.normal(0, 1, num_symbols * sps)
               + 1j * rng.normal(0, 1, num_symbols * sps)).astype(np.complex64)
    save_as_wav(garbage, sample_rate, os.path.join(out_dir, "garbage_noise.wav"))
    print(f"Saved {out_dir}/garbage_noise.wav  (pure noise, "
          f"should trigger UNKNOWN / failed / low-confidence handling)")
 
 
if __name__ == "__main__":
    main()