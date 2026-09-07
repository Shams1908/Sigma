"""
Synchronization package.

Provides the full sync chain:
  1. Matched filtering (RRC)         — matched_filter.py
  2. Carrier phase/freq recovery     — carrier_recovery.py  (Costas loop)
  3. Symbol timing recovery          — timing_recovery.py   (Gardner TED)

Public convenience function: synchronize() runs all three stages in order.
"""
from __future__ import annotations

import numpy as np

from synchronization.matched_filter import apply_matched_filter, downsample
from synchronization.carrier_recovery import costas_loop, estimate_residual_phase
from synchronization.timing_recovery import gardner_timing_recovery, coarse_timing_offset


def synchronize(
    iq: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    modulation_order: int = 4,
    rolloff: float = 0.35,
    use_gardner: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Full synchronisation chain for BPSK/QPSK signals.

    Stages:
        1. RRC matched filter
        2. Feedforward phase offset correction
        3. Costas loop carrier recovery
        4. Timing recovery (Gardner or coarse downsample)

    Args:
        iq:               1-D complex IQ array (float32 or complex64).
        sample_rate:      Sample rate (Hz).
        symbol_rate:      Estimated symbol rate (symbols/s).
        modulation_order: 2 = BPSK, 4 = QPSK.
        rolloff:          RRC roll-off factor β.
        use_gardner:      If True use Gardner TED; if False use coarse offset.

    Returns:
        symbols:   Complex symbol array (one sample per symbol).
        info:      Dict with diagnostics (sps, timing_errors, phase_err).
    """
    iq = np.asarray(iq, dtype=np.complex64)

    # Guard: need at least a few symbols
    sps_float = sample_rate / max(symbol_rate, 1.0)
    sps = max(2, int(round(sps_float)))

    # ── Stage 1: RRC matched filter ──────────────────────────────────────────
    try:
        mf_out = apply_matched_filter(iq, samples_per_symbol=sps, rolloff=rolloff)
    except Exception:  # noqa: BLE001
        mf_out = iq  # pass-through on failure

    # ── Stage 2: Feedforward phase correction ────────────────────────────────
    try:
        phi0 = estimate_residual_phase(mf_out, modulation_order)
        mf_out = (mf_out * np.exp(-1j * phi0)).astype(np.complex64)
    except Exception:  # noqa: BLE001
        phi0 = 0.0

    # ── Stage 3: Costas loop ─────────────────────────────────────────────────
    try:
        costas_out, phase_err = costas_loop(mf_out, modulation_order=modulation_order)
    except Exception:  # noqa: BLE001
        costas_out = mf_out
        phase_err = np.zeros(len(mf_out))

    # ── Stage 4: Timing recovery ─────────────────────────────────────────────
    timing_errors: np.ndarray = np.array([])

    if use_gardner and sps >= 2:
        try:
            symbols, timing_errors = gardner_timing_recovery(costas_out, sps)
        except Exception:  # noqa: BLE001
            offset = coarse_timing_offset(costas_out, sps)
            symbols = downsample(costas_out, sps, offset)
    else:
        offset = coarse_timing_offset(costas_out, sps)
        symbols = downsample(costas_out, sps, offset)

    info = {
        "sps": sps,
        "rolloff": rolloff,
        "initial_phase_offset_rad": float(phi0) if phi0 else 0.0,
        "timing_error_rms": float(np.sqrt(np.mean(timing_errors ** 2)))
        if len(timing_errors) > 0
        else 0.0,
    }

    return symbols.astype(np.complex64), info
