"""
Carrier phase / frequency recovery — Costas Loop.

A second-order Costas loop works for both BPSK (2nd-power phase detector)
and QPSK (4th-power phase detector).  The loop tracks residual carrier
frequency offset and phase after coarse CFO correction.

References:
  Proakis & Salehi, "Communication Systems Engineering", §6.3
  Rice, "Digital Communications — A Discrete-Time Approach", §8.3
"""
from __future__ import annotations

import numpy as np


def costas_loop(
    iq: np.ndarray,
    modulation_order: int = 4,
    loop_bw: float = 0.01,
    damping: float = 0.707,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a second-order Costas loop for carrier phase recovery.

    Supports BPSK (modulation_order=2) and QPSK (modulation_order=4).

    The loop BW and damping factor together determine the natural frequency:
        ωn = loop_bw * 2π
        K1 = 2 * damping * ωn   (proportional gain)
        K2 = ωn²                 (integral gain)

    Args:
        iq:               1-D complex IQ array (one sample per symbol, or
                          oversampled — loop operates sample-by-sample).
        modulation_order: 2 for BPSK, 4 for QPSK.
        loop_bw:          Normalised loop bandwidth (fraction of sample rate,
                          typically 0.001–0.05).
        damping:          Damping factor (0.707 = critically damped).

    Returns:
        corrected:  Phase-corrected complex IQ array (same length as input).
        phase_err:  Per-sample phase error trajectory (radians).
    """
    iq = np.asarray(iq, dtype=np.complex128)
    n = len(iq)

    # Second-order loop filter gains
    omega_n = loop_bw * 2.0 * np.pi
    K1 = 2.0 * damping * omega_n        # proportional
    K2 = omega_n ** 2                   # integral

    corrected = np.zeros(n, dtype=np.complex128)
    phase_err_trace = np.zeros(n, dtype=np.float64)

    phase = 0.0       # accumulated phase estimate (radians)
    freq = 0.0        # accumulated frequency error (radians/sample)
    integrator = 0.0  # integral branch accumulator

    for k in range(n):
        # Correct sample by rotating back by estimated phase
        corrected[k] = iq[k] * np.exp(-1j * phase)

        # Phase error detector: sgn(Re) * Im  for BPSK;
        # decision-directed 4th-power for QPSK
        err = _phase_error(corrected[k], modulation_order)
        phase_err_trace[k] = err

        # Loop filter (PI controller)
        integrator += K2 * err
        freq_update = K1 * err + integrator

        # Update phase accumulator
        phase += freq_update

    return corrected.astype(np.complex64), phase_err_trace


def _phase_error(sample: complex, order: int) -> float:
    """
    Decision-directed phase error detector.

    BPSK (order=2): e = sign(Re{s}) * Im{s}
    QPSK (order=4): e = sign(Re{s}) * Im{s} - sign(Im{s}) * Re{s}
                        (cross-product of hard decision and received sample)
    """
    re = sample.real
    im = sample.imag

    if order == 2:
        return float(np.sign(re) * im)

    # QPSK — four-quadrant cross-product detector
    return float(np.sign(re) * im - np.sign(im) * re)


def estimate_residual_phase(iq: np.ndarray, modulation_order: int = 4) -> float:
    """
    Quick feedforward residual phase estimate (no loop, single pass).
    Raises the signal to the Mth power, finds the dominant phase cluster,
    then divides by M to recover the carrier phase.

    Useful as an initial phase offset correction before the Costas loop.

    Returns estimated phase offset in radians.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    powered = iq ** modulation_order
    mean_phase = np.angle(np.mean(powered))
    return float(mean_phase / modulation_order)
