"""
BPSK and QPSK demodulators.

Both operate on a complex symbol array that has already been through the
synchronisation chain (matched filter + carrier recovery + timing recovery).
The demodulator maps each complex symbol to the nearest constellation point
and returns the decoded bit sequence.

Constellation conventions (Gray-coded):
  BPSK:  +1 → 0,  -1 → 1
  QPSK:  Uses the standard Gray-coded mapping for the four quadrants.
         I > 0, Q > 0 → 00
         I < 0, Q > 0 → 10
         I < 0, Q < 0 → 11
         I > 0, Q < 0 → 01
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class DemodResult:
    bits: np.ndarray          # 1-D uint8 array of decoded bits
    symbols: np.ndarray       # Input complex symbols
    decisions: np.ndarray     # Hard-decided constellation points (complex)
    evm_rms: float            # RMS Error Vector Magnitude (linear, 0–1)
    modulation: str


# ── BPSK ─────────────────────────────────────────────────────────────────────

# BPSK constellation: index → symbol, symbol → bit
_BPSK_CONST = np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64)
_BPSK_BITS  = np.array([0, 1], dtype=np.uint8)


def demod_bpsk(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision BPSK demodulator.

    Decides on the sign of the real part; imaginary part is treated as noise.

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits, decisions, and EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)

    # Decision: threshold on real part
    bit_array = np.where(symbols.real >= 0.0, np.uint8(0), np.uint8(1))
    decisions = np.where(symbols.real >= 0.0, _BPSK_CONST[0], _BPSK_CONST[1])

    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bit_array,
        symbols=symbols,
        decisions=decisions.astype(np.complex64),
        evm_rms=evm,
        modulation="BPSK",
    )


# ── QPSK ─────────────────────────────────────────────────────────────────────

# Gray-coded QPSK constellation (normalised to unit average power)
_QPSK_NORM = 1.0 / np.sqrt(2.0)
# (re_sign, im_sign) → (2-bit Gray code)
_QPSK_DECISION_TABLE: dict[tuple[int, int], tuple[np.complex64, list[int]]] = {
    ( 1,  1): (np.complex64(( _QPSK_NORM + 1j * _QPSK_NORM)), [0, 0]),
    (-1,  1): (np.complex64((-_QPSK_NORM + 1j * _QPSK_NORM)), [1, 0]),
    (-1, -1): (np.complex64((-_QPSK_NORM - 1j * _QPSK_NORM)), [1, 1]),
    ( 1, -1): (np.complex64(( _QPSK_NORM - 1j * _QPSK_NORM)), [0, 1]),
}


def demod_qpsk(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision QPSK demodulator (Gray coded).

    Each symbol maps to 2 bits based on the quadrant of the complex plane.

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits, decisions, and EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)

    bits_list: list[int] = []
    decisions = np.zeros(n, dtype=np.complex64)

    for i, s in enumerate(symbols):
        re_sign = 1 if s.real >= 0.0 else -1
        im_sign = 1 if s.imag >= 0.0 else -1
        decision_point, two_bits = _QPSK_DECISION_TABLE[(re_sign, im_sign)]
        decisions[i] = decision_point
        bits_list.extend(two_bits)

    bit_array = np.array(bits_list, dtype=np.uint8)
    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bit_array,
        symbols=symbols,
        decisions=decisions,
        evm_rms=evm,
        modulation="QPSK",
    )


# ── Generic dispatcher ────────────────────────────────────────────────────────

def demodulate_psk(
    symbols: np.ndarray,
    modulation: Literal["BPSK", "QPSK"],
) -> DemodResult:
    """
    Dispatcher: route to the appropriate PSK demodulator.

    Args:
        symbols:    1-D complex symbol array.
        modulation: "BPSK" or "QPSK".

    Returns:
        DemodResult.

    Raises:
        ValueError: If an unsupported modulation is requested.
    """
    if modulation == "BPSK":
        return demod_bpsk(symbols)
    if modulation == "QPSK":
        return demod_qpsk(symbols)
    raise ValueError(
        f"Unsupported PSK modulation '{modulation}'. Supported: BPSK, QPSK."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rms_evm(received: np.ndarray, decisions: np.ndarray) -> float:
    """
    RMS Error Vector Magnitude (normalised to mean decision power).

    EVM_rms = sqrt( mean(|error|²) / mean(|decision|²) )

    Returns a value in [0, ∞); values < 0.3 indicate clean demodulation.
    """
    error_power = float(np.mean(np.abs(received - decisions) ** 2))
    ref_power = float(np.mean(np.abs(decisions) ** 2))
    if ref_power == 0.0:
        return 1.0
    return float(np.sqrt(error_power / ref_power))


def constellation_score(result: DemodResult) -> float:
    """
    Derive a 0–1 quality score from EVM.
    Score = max(0, 1 - 2 * evm_rms)
    At EVM=0 → score=1.0; at EVM≥0.5 → score=0.0 (demod failure).
    """
    return float(max(0.0, 1.0 - 2.0 * result.evm_rms))
