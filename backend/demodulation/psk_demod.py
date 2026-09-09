"""
PSK (Phase Shift Keying) demodulator for SIGMA.
Implements symbol slicing, decision boundaries, and bit extraction for BPSK and QPSK.
"""
from typing import Tuple, Dict, Any
import numpy as np


def demodulate_psk(
    symbols: np.ndarray,
    modulation: str = "BPSK",
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Demodulates complex PSK symbols to hard decision bits and computes constellation agreement.

    Args:
        symbols: 1D complex or 2D [2, N] array of constellation symbols.
        modulation: Modulation name ('BPSK' or 'QPSK').

    Returns:
        Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
          - symbols: 1D complex64 symbol array
          - bits: 1D uint8 hard bit array
          - constellation_score: Normalized score in [0.0, 1.0]
          - metrics: Dictionary containing EVM and symbol count telemetry
    """
    if symbols is None or len(symbols) == 0:
        raise ValueError("Input symbols must not be empty")

    # Standardize to 1D complex
    if isinstance(symbols, np.ndarray) and symbols.ndim == 2 and symbols.shape[0] == 2 and not np.iscomplexobj(symbols):
        complex_syms = (symbols[0] + 1j * symbols[1]).astype(np.complex64)
    elif np.iscomplexobj(symbols):
        complex_syms = symbols.flatten().astype(np.complex64)
    else:
        complex_syms = symbols.flatten().astype(np.complex64)

    norm_mod = modulation.strip().upper()

    if norm_mod == "BPSK":
        # BPSK: Bit 0 -> -1.0, Bit 1 -> +1.0
        # Decision boundary: real > 0 -> 1, real <= 0 -> 0
        bits = (complex_syms.real > 0.0).astype(np.uint8)

        # Ideal constellation references
        ideal_syms = (2.0 * bits.astype(np.float32) - 1.0).astype(np.complex64)

        # EVM calculation
        err = complex_syms - ideal_syms
        evm = float(np.sqrt(np.mean(np.abs(err) ** 2)))
        score = max(0.0, min(1.0, 1.0 - evm))

        metrics = {
            "evm_rms": evm,
            "evm_db": float(20.0 * np.log10(max(1e-6, evm))),
            "num_symbols": len(complex_syms),
            "num_bits": len(bits),
            "bits_per_symbol": 1,
        }
        return complex_syms, bits, score, metrics

    elif norm_mod == "QPSK":
        # QPSK Gray-coded constellation matching ml.generators.modulation.qpsk:
        # b0 b1 = 00 -> (+1 + 1j) / sqrt(2)
        # b0 b1 = 01 -> (-1 + 1j) / sqrt(2)
        # b0 b1 = 10 -> (+1 - 1j) / sqrt(2)
        # b0 b1 = 11 -> (-1 - 1j) / sqrt(2)
        scale = 1.0 / np.sqrt(2.0)
        constellation = (np.array([
            1.0 + 1.0j,    # index 0 (00)
            -1.0 + 1.0j,   # index 1 (01)
            1.0 - 1.0j,    # index 2 (10)
            -1.0 - 1.0j    # index 3 (11)
        ], dtype=np.complex64) * scale).astype(np.complex64)

        # Vectorized minimum Euclidean distance slicing
        # distances shape: (N, 4)
        dists = np.abs(complex_syms[:, np.newaxis] - constellation[np.newaxis, :])
        min_idx = np.argmin(dists, axis=1)

        # Map min_idx (0..3) to bit pairs (b0, b1)
        b0 = (min_idx >= 2).astype(np.uint8)
        b1 = (min_idx % 2 == 1).astype(np.uint8)
        bits = np.column_stack([b0, b1]).flatten()

        ideal_syms = constellation[min_idx]
        err = complex_syms - ideal_syms
        evm = float(np.sqrt(np.mean(np.abs(err) ** 2)))
        score = max(0.0, min(1.0, 1.0 - evm))

        metrics = {
            "evm_rms": evm,
            "evm_db": float(20.0 * np.log10(max(1e-6, evm))),
            "num_symbols": len(complex_syms),
            "num_bits": len(bits),
            "bits_per_symbol": 2,
        }
        return complex_syms, bits, score, metrics

    else:
        raise NotImplementedError(f"Modulation '{modulation}' is not supported by psk_demod")
