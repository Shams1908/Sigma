"""
QAM demodulators — 16-QAM and 64-QAM.

OUT-OF-SCOPE STUB removed; both are now implemented.

Canonical mappings exactly invert the ML QAM16Modulator and QAM64Modulator.

─────────────────────────────────────────────────────────────────────────────
16-QAM (4 bits per symbol: b0,b1 → I; b2,b3 → Q)
─────────────────────────────────────────────────────────────────────────────
Per-axis Gray-coded level mapping (applied independently to I and Q):
  bit-pair index = b_msb*2 + b_lsb
  idx=0 (00) → +3  (before /√10 normalisation)
  idx=1 (01) → +1
  idx=2 (10) → -3
  idx=3 (11) → -1

Decision boundaries (on the normalised signal, multiply by √10):
  level > +2/√10  → idx=0 (+3) → bits [0,0]
  0 < level ≤ +2/√10 → idx=1 (+1) → bits [0,1]
  -2/√10 < level ≤ 0 → idx=3 (-1) → bits [1,1]
  level ≤ -2/√10  → idx=2 (-3) → bits [1,0]

─────────────────────────────────────────────────────────────────────────────
64-QAM (6 bits per symbol: b0,b1,b2 → I; b3,b4,b5 → Q)
─────────────────────────────────────────────────────────────────────────────
Per-axis Gray-coded level mapping (3-bit, normalised by 1/√42):
  idx=0 (000) → +7
  idx=1 (001) → +5
  idx=2 (010) → +1
  idx=3 (011) → +3
  idx=4 (100) → -7
  idx=5 (101) → -5
  idx=6 (110) → -1
  idx=7 (111) → -3
"""
from __future__ import annotations

import numpy as np

from demodulation.psk_demod import DemodResult, _rms_evm


# ── 16-QAM ───────────────────────────────────────────────────────────────────

_QAM16_SCALE = 1.0 / np.sqrt(10.0)
_QAM16_BOUNDARY = 2.0 * _QAM16_SCALE   # threshold between outer and inner levels

# Per-axis: unnormalised level → (normalised_level, [b_msb, b_lsb])
# The generator levels array: levels[idx] = [3, 1, -3, -1]
# So the full constellation is levels[idx_i]/√10 + j*levels[idx_q]/√10
_QAM16_LEVELS = np.array([3.0, 1.0, -3.0, -1.0], dtype=np.float64) * _QAM16_SCALE
# bits for each index: idx → [b_msb, b_lsb]
_QAM16_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)

# Precompute the full 16-point constellation and corresponding 4-bit words
_QAM16_CONST = np.array(
    [(_QAM16_LEVELS[i] + 1j * _QAM16_LEVELS[j])
     for i in range(4) for j in range(4)],
    dtype=np.complex64,
)
_QAM16_ALL_BITS = np.array(
    [list(_QAM16_BITS[i]) + list(_QAM16_BITS[j])
     for i in range(4) for j in range(4)],
    dtype=np.uint8,
)  # shape [16, 4]


def _demod_qam16_axis(values: np.ndarray) -> np.ndarray:
    """
    Hard-decision per-axis 16-QAM slicer.

    Maps a real-valued array of normalised QAM levels to their 2-bit Gray
    codes using the thresholds at ±2/√10.

    Returns a (n, 2) uint8 array where each row is [b_msb, b_lsb].
    """
    # Boundary between outer (+3) and inner (+1) levels: +2/√10
    # Boundary between inner (+1) and inner (-1) levels: 0
    # Boundary between inner (-1) and outer (-3) levels: -2/√10
    bits = np.zeros((len(values), 2), dtype=np.uint8)
    bits[:, 0] = np.where(values < 0.0, 1, 0)            # b_msb: negative → 1
    # b_lsb: distinguishes outer (+3/-3) from inner (+1/-1)
    # outer:  |level| > 2/√10  → bit=0
    # inner:  |level| ≤ 2/√10  → bit=1
    bits[:, 1] = np.where(np.abs(values) <= _QAM16_BOUNDARY, 1, 0)
    return bits


def demod_qam16(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision 16-QAM demodulator.

    Canonical mapping (exactly inverts ML QAM16Modulator):
      I component → 2 bits [b0, b1]
      Q component → 2 bits [b2, b3]
      Total: 4 bits per symbol.

    Decision rule: independent per-axis slicing with three boundaries at
    −2/√10, 0, and +2/√10 on the normalised signal.

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits (4n bits for n symbols), decisions, EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)

    if n == 0:
        return DemodResult(
            bits=np.array([], dtype=np.uint8),
            symbols=symbols,
            decisions=np.array([], dtype=np.complex64),
            evm_rms=0.0,
            modulation="QAM16",
        )

    i_bits = _demod_qam16_axis(symbols.real.astype(np.float64))  # [n, 2]
    q_bits = _demod_qam16_axis(symbols.imag.astype(np.float64))  # [n, 2]

    # Reconstruct normalised decision levels from bits
    i_idx = i_bits[:, 0] * 2 + i_bits[:, 1]
    q_idx = q_bits[:, 0] * 2 + q_bits[:, 1]
    decisions = (_QAM16_LEVELS[i_idx] + 1j * _QAM16_LEVELS[q_idx]).astype(np.complex64)

    # Interleave: [b0,b1,b2,b3, b0,b1,b2,b3, ...]
    bits = np.concatenate([i_bits, q_bits], axis=1).ravel()  # 4n bits

    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bits,
        symbols=symbols,
        decisions=decisions,
        evm_rms=evm,
        modulation="QAM16",
    )


# ── 64-QAM ───────────────────────────────────────────────────────────────────

_QAM64_SCALE = 1.0 / np.sqrt(42.0)

# Generator level lookup (unnormalised): levels[idx] for idx 0..7
# levels = [7, 5, 1, 3, -7, -5, -1, -3]
_QAM64_LEVELS_RAW = np.array([7.0, 5.0, 1.0, 3.0, -7.0, -5.0, -1.0, -3.0],
                               dtype=np.float64)
_QAM64_LEVELS = _QAM64_LEVELS_RAW * _QAM64_SCALE

# 3-bit Gray words per index: idx → [b_msb, b_mid, b_lsb] = binary of idx
_QAM64_BITS = np.array(
    [[int(b) for b in f"{i:03b}"] for i in range(8)],
    dtype=np.uint8,
)

# Precompute all 64 constellation points and their 6-bit words
_QAM64_CONST = np.array(
    [(_QAM64_LEVELS[i] + 1j * _QAM64_LEVELS[j])
     for i in range(8) for j in range(8)],
    dtype=np.complex64,
)
_QAM64_ALL_BITS = np.array(
    [list(_QAM64_BITS[i]) + list(_QAM64_BITS[j])
     for i in range(8) for j in range(8)],
    dtype=np.uint8,
)  # shape [64, 6]


def _demod_qam64_axis(values: np.ndarray) -> np.ndarray:
    """
    Hard-decision per-axis 64-QAM slicer.

    Maps a real-valued array of normalised QAM levels to their 3-bit words
    by finding the nearest of the 8 unnormalised levels {±7,±5,±3,±1}/√42.

    Returns a (n, 3) uint8 array where each row is [b_msb, b_mid, b_lsb].
    """
    n = len(values)
    # Unnormalise for comparison
    vals_raw = values / _QAM64_SCALE   # back to ±1,±3,±5,±7 domain
    # Distance to each of the 8 levels
    dists = np.abs(vals_raw[:, np.newaxis] - _QAM64_LEVELS_RAW[np.newaxis, :])
    nearest_idx = np.argmin(dists, axis=1)
    return _QAM64_BITS[nearest_idx]


def demod_qam64(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision 64-QAM demodulator.

    Canonical mapping (exactly inverts ML QAM64Modulator):
      I component → 3 bits [b0, b1, b2]
      Q component → 3 bits [b3, b4, b5]
      Total: 6 bits per symbol.

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits (6n bits for n symbols), decisions, EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)

    if n == 0:
        return DemodResult(
            bits=np.array([], dtype=np.uint8),
            symbols=symbols,
            decisions=np.array([], dtype=np.complex64),
            evm_rms=0.0,
            modulation="QAM64",
        )

    i_bits = _demod_qam64_axis(symbols.real.astype(np.float64))  # [n, 3]
    q_bits = _demod_qam64_axis(symbols.imag.astype(np.float64))  # [n, 3]

    i_idx = np.argmin(
        np.abs(symbols.real.astype(np.float64)[:, np.newaxis] / _QAM64_SCALE
               - _QAM64_LEVELS_RAW[np.newaxis, :]),
        axis=1,
    )
    q_idx = np.argmin(
        np.abs(symbols.imag.astype(np.float64)[:, np.newaxis] / _QAM64_SCALE
               - _QAM64_LEVELS_RAW[np.newaxis, :]),
        axis=1,
    )
    decisions = (_QAM64_LEVELS[i_idx] + 1j * _QAM64_LEVELS[q_idx]).astype(np.complex64)

    bits = np.concatenate([i_bits, q_bits], axis=1).ravel()  # 6n bits

    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bits,
        symbols=symbols,
        decisions=decisions,
        evm_rms=evm,
        modulation="QAM64",
    )


# ── Error class for not-yet-implemented modulations ───────────────────────────

class QAMDemodNotImplemented(NotImplementedError):
    """Raised when a QAM variant that is not yet implemented is requested."""
