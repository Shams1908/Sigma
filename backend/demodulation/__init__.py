"""
Demodulation package.

Supported modulations:
  BPSK, QPSK         — psk_demod.py    (hard-decision, matched to ML generators)
  8PSK               — psk8_demod.py   (hard-decision, matched to ML EightPSKModulator)
  16QAM, QAM16       — qam_demod.py    (hard-decision, matched to ML QAM16Modulator)
  64QAM, QAM64       — qam_demod.py    (hard-decision, matched to ML QAM64Modulator)
  2FSK               — fsk_demod.py    (non-coherent instantaneous-frequency)
  4FSK               — fsk_demod.py    (non-coherent instantaneous-frequency, Gray coded)

Public entry point: demodulate()
"""
from __future__ import annotations

import numpy as np

from demodulation.psk_demod import DemodResult, demodulate_psk, constellation_score
from demodulation.psk8_demod import demod_8psk
from demodulation.qam_demod import demod_qam16, demod_qam64, QAMDemodNotImplemented
from demodulation.fsk_demod import demod_2fsk, demod_4fsk

__all__ = [
    "demodulate",
    "DemodResult",
    "constellation_score",
    "demod_8psk",
    "demod_qam16",
    "demod_qam64",
    "demod_2fsk",
    "demod_4fsk",
    "QAMDemodNotImplemented",
]


def demodulate(symbols: np.ndarray, modulation: str, **kwargs) -> DemodResult:
    """
    Route to the appropriate demodulator based on modulation string.

    Supported: BPSK, QPSK, 8PSK, 16QAM (aliases: 16-QAM, QAM16), 64QAM
               (aliases: 64-QAM, QAM64), 2FSK, 4FSK.

    For FSK modulations, keyword arguments are forwarded to the FSK
    demodulator (e.g., ``sps``, ``tone_freqs``).

    Raises:
        ValueError: For unsupported modulation strings.
    """
    mod_upper = modulation.upper().replace("-", "").replace("_", "")

    if mod_upper in ("BPSK", "QPSK"):
        return demodulate_psk(symbols, mod_upper)  # type: ignore[arg-type]

    if mod_upper == "8PSK":
        return demod_8psk(symbols)

    if mod_upper in ("16QAM", "QAM16"):
        return demod_qam16(symbols)

    if mod_upper in ("64QAM", "QAM64"):
        return demod_qam64(symbols)

    if mod_upper == "2FSK":
        return demod_2fsk(symbols, **kwargs)

    if mod_upper == "4FSK":
        return demod_4fsk(symbols, **kwargs)

    raise ValueError(
        f"Unsupported modulation '{modulation}'. "
        "Supported: BPSK, QPSK, 8PSK, 16QAM/QAM16, 64QAM/QAM64, 2FSK, 4FSK."
    )
