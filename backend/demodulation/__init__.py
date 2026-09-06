"""
Demodulation package.

Supported modulations: BPSK, QPSK.
16-QAM: documented stub (qam_demod.py) — not yet implemented.
FSK:    out of scope.

Public entry point: demodulate()
"""
from __future__ import annotations

from demodulation.psk_demod import DemodResult, demodulate_psk, constellation_score

__all__ = ["demodulate", "DemodResult", "constellation_score"]


def demodulate(symbols, modulation: str) -> DemodResult:
    """
    Route to the appropriate demodulator based on modulation string.

    Supported: "BPSK", "QPSK"
    Unsupported strings raise ValueError with a clear message.
    """
    mod_upper = modulation.upper()

    if mod_upper in ("BPSK", "QPSK"):
        return demodulate_psk(symbols, mod_upper)  # type: ignore[arg-type]

    if mod_upper in ("16QAM", "16-QAM", "QAM16"):
        from demodulation.qam_demod import QAMDemodNotImplemented
        raise QAMDemodNotImplemented(
            "16-QAM demodulation is not yet implemented. "
            "Hypothesis will be marked demod_pass=False."
        )

    raise ValueError(
        f"Unsupported modulation '{modulation}'. "
        "Supported modulations: BPSK, QPSK."
    )
