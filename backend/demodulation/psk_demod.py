"""
BPSK and QPSK demodulators.

Both operate on a complex symbol array that has already been through the
synchronisation chain (matched filter + carrier recovery + timing recovery).
The demodulator maps each complex symbol to the nearest constellation point
and returns the decoded bit sequence.

Constellation conventions match the ML generator canonical mappings exactly:

BPSK:
  bit 0 → -1 + 0j
  bit 1 → +1 + 0j
  Decision: real ≥ 0 → bit 1 (+1),  real < 0 → bit 0 (-1)

QPSK (Gray-coded, 2 bits per symbol as (b0, b1)):
  00 → (+1 + 1j) / sqrt(2)   [Q1: re>0, im>0]
  01 → (-1 + 1j) / sqrt(2)   [Q2: re<0, im>0]
  11 → (-1 - 1j) / sqrt(2)   [Q3: re<0, im<0]
  10 → (+1 - 1j) / sqrt(2)   [Q4: re>0, im<0]
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

# Canonical BPSK constellation matching the ML generator:
#   bit 0 → -1+0j  (index 0)
#   bit 1 → +1+0j  (index 1)
_BPSK_CONST = np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
_BPSK_BITS  = np.array([0, 1], dtype=np.uint8)


def demod_bpsk(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision BPSK demodulator.

    Canonical mapping (matches ML generator):
      real ≥ 0  →  decision = +1,  bit = 1
      real < 0  →  decision = -1,  bit = 0

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits, decisions, and EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)

    # real ≥ 0  →  bit 1  (+1),   real < 0  →  bit 0  (-1)
    positive = symbols.real >= 0.0
    bit_array = np.where(positive, np.uint8(1), np.uint8(0))
    decisions = np.where(positive, _BPSK_CONST[1], _BPSK_CONST[0])

    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bit_array,
        symbols=symbols,
        decisions=decisions.astype(np.complex64),
        evm_rms=evm,
        modulation="BPSK",
    )


# ── QPSK ─────────────────────────────────────────────────────────────────────

# Gray-coded QPSK constellation (normalised to unit average power).
# Table key: (sign(re), sign(im)) where +1 means ≥ 0.
# Matches ML generator index: b0*2 + b1 → symbol
#   idx 0 (b0=0,b1=0):  (+1+1j)/√2   Q1: re>0, im>0  → bits [0,0]
#   idx 1 (b0=0,b1=1):  (-1+1j)/√2   Q2: re<0, im>0  → bits [0,1]
#   idx 2 (b0=1,b1=0):  (+1-1j)/√2   Q4: re>0, im<0  → bits [1,0]
#   idx 3 (b0=1,b1=1):  (-1-1j)/√2   Q3: re<0, im<0  → bits [1,1]
_QPSK_NORM = 1.0 / np.sqrt(2.0)
_QPSK_DECISION_TABLE: dict[tuple[int, int], tuple[np.complex64, list[int]]] = {
    ( 1,  1): (np.complex64(( _QPSK_NORM + 1j * _QPSK_NORM)), [0, 0]),   # Q1: bits 00
    (-1,  1): (np.complex64((-_QPSK_NORM + 1j * _QPSK_NORM)), [0, 1]),   # Q2: bits 01
    (-1, -1): (np.complex64((-_QPSK_NORM - 1j * _QPSK_NORM)), [1, 1]),   # Q3: bits 11
    ( 1, -1): (np.complex64(( _QPSK_NORM - 1j * _QPSK_NORM)), [1, 0]),   # Q4: bits 10
}


def demod_qpsk(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision QPSK demodulator (Gray coded).

    Canonical mapping (matches ML generator):
      Q1 (re>0, im>0) → bits [0, 0]
      Q2 (re<0, im>0) → bits [0, 1]
      Q3 (re<0, im<0) → bits [1, 1]
      Q4 (re>0, im<0) → bits [1, 0]

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
