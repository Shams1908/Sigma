"""
FSK demodulators — 2-FSK and 4-FSK.

Algorithm
---------
Non-coherent M-FSK demodulation via instantaneous-frequency estimation.

The instantaneous frequency of a discrete-time signal is estimated as:

    f_inst[n] = angle(x[n] · conj(x[n-1])) / (2π · T_s)
              = angle(x[n] · conj(x[n-1])) · f_s / (2π)

where T_s = 1/f_s is the sample period.

For a signal at tone k: x[n] ≈ A·exp(j·2π·f_k·n/f_s), so
f_inst[n] ≈ f_k.

A per-symbol frequency estimate is obtained by averaging f_inst over one
symbol period of sps samples.  The symbol decision is made by comparing
the average instantaneous frequency to the expected tone frequencies.

Symbol → bit mapping conventions
---------------------------------
No ML generator exists for FSK in this project.  The following standard
orthogonal FSK conventions are used, consistent with common radio standards
(e.g., AFSK, Bell 202, and the project's CPFSK/GFSK modulation classes).

2-FSK (1 bit per symbol):
  bit 0  →  low tone   (most negative / lowest frequency)
  bit 1  →  high tone  (most positive / highest frequency)

  This is the conventional "space = 0, mark = 1" assignment.

4-FSK (2 bits per symbol, Gray-coded by frequency order):
  dibit  00  →  tone 0  (lowest frequency)
  dibit  01  →  tone 1
  dibit  11  →  tone 2
  dibit  10  →  tone 3  (highest frequency)

  The Gray coding means adjacent tones differ by 1 bit, minimising BER at
  moderate SNR.

Decision rule
-------------
The per-symbol average instantaneous frequency is compared against the set
of expected tone frequencies {f_k}.  The nearest tone wins.

When the caller does not provide explicit tone frequencies, the demodulator
operates in a tone-agnostic mode: it sorts the detected tones and assigns
them in order (lowest → highest), which is correct for any standard
orthogonal FSK signal where tones are symmetric around zero (or otherwise
equally spaced).

Input
-----
For correct operation, ``symbols`` should be a complex IQ array at 1 sample/
symbol (i.e., one sample per symbol decision epoch, already downsampled after
matched filtering / timing recovery), OR an oversampled IQ array at ``sps``
samples/symbol.

If ``sps > 1``, the demodulator averages f_inst over each sps-sample window.
If ``sps == 1``, a single instantaneous-frequency sample is used per symbol.

EVM for FSK
-----------
Conventional EVM (error vector magnitude) is not meaningful for FSK because
the signals are not constrained to a finite constellation in the complex
plane.  ``evm_rms`` is therefore reported as 0.0 for FSK results, and
``decisions`` contains the per-symbol frequency estimates as real-valued
complex numbers (Re = f_est, Im = 0).
"""
from __future__ import annotations

import numpy as np

from demodulation.psk_demod import DemodResult


# ── Gray code tables ──────────────────────────────────────────────────────────

# 4-FSK dibit assignment (Gray coded by tone index):
# tone_index → [b0, b1]
_4FSK_GRAY: dict[int, list[int]] = {
    0: [0, 0],   # lowest tone
    1: [0, 1],
    2: [1, 1],
    3: [1, 0],   # highest tone
}


# ── Instantaneous frequency helper ───────────────────────────────────────────

def _inst_freq(iq: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous frequency (in radians/sample) for each sample.

    f_inst[n] = angle(iq[n] · conj(iq[n-1]))

    The first output sample is 0.0 (no predecessor).

    Args:
        iq: 1-D complex array.

    Returns:
        Real-valued array of length len(iq), in radians/sample.
    """
    if len(iq) < 2:
        return np.zeros(len(iq), dtype=np.float64)
    phase_diff = np.angle(iq[1:].astype(np.complex128) * np.conj(iq[:-1].astype(np.complex128)))
    return np.concatenate([[0.0], phase_diff])


def _avg_freq_per_symbol(iq: np.ndarray, sps: int) -> np.ndarray:
    """
    Compute the average instantaneous frequency (rad/sample) over each
    sps-sample window.  Trailing samples that do not fill a full window
    are discarded.

    Args:
        iq:  1-D complex array.
        sps: Samples per symbol (≥ 1).

    Returns:
        Real-valued array of length floor(len(iq) / sps).
    """
    if sps < 1:
        raise ValueError(f"sps must be ≥ 1, got {sps}")
    n_sym = len(iq) // sps
    if n_sym == 0:
        return np.array([], dtype=np.float64)
    fi = _inst_freq(iq)
    # Trim to n_sym * sps samples and reshape to [n_sym, sps]
    fi_trimmed = fi[:n_sym * sps].reshape(n_sym, sps)
    # Average, ignoring the very first sample (0.0 sentinel) only for the
    # very first window.  Using mean of samples 1.. inside each window is
    # cleaner; for sps==1 all samples are used as-is.
    if sps == 1:
        return fi_trimmed[:, 0]
    return fi_trimmed[:, 1:].mean(axis=1)   # skip index-0 sentinel within window


# ── 2-FSK demodulator ─────────────────────────────────────────────────────────

def demod_2fsk(
    symbols: np.ndarray,
    sps: int = 1,
    tone_freqs: tuple[float, float] | None = None,
) -> DemodResult:
    """
    Non-coherent hard-decision 2-FSK demodulator.

    Bit assignment:
      bit 0  →  low tone  (most negative instantaneous frequency)
      bit 1  →  high tone (most positive instantaneous frequency)

    Args:
        symbols:     1-D complex array.  If ``sps > 1`` the array contains
                     multiple samples per symbol and averaging is applied.
        sps:         Samples per symbol.  Default 1 (already downsampled).
        tone_freqs:  Optional (f_low, f_high) in radians/sample.  If None,
                     the midpoint of the estimated frequency range is used
                     as the decision boundary (works for symmetric FSK).

    Returns:
        DemodResult with bits (1 bit/symbol), decisions, evm_rms=0.0,
        modulation="2FSK".
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if len(symbols) == 0:
        return DemodResult(
            bits=np.array([], dtype=np.uint8),
            symbols=symbols,
            decisions=np.array([], dtype=np.complex64),
            evm_rms=0.0,
            modulation="2FSK",
        )

    freq_est = _avg_freq_per_symbol(symbols, sps)   # rad/sample per symbol
    n_sym = len(freq_est)

    if tone_freqs is not None:
        f_low, f_high = float(tone_freqs[0]), float(tone_freqs[1])
        threshold = (f_low + f_high) / 2.0
    else:
        # Threshold at the midpoint of the observed frequency range.
        # This is unbiased for any symmetric orthogonal FSK.
        threshold = float(np.median(freq_est))

    bits = np.where(freq_est >= threshold, np.uint8(1), np.uint8(0))

    # Decision values: encode frequency estimate as complex (real = freq_rad)
    decisions = freq_est.astype(np.complex64)

    return DemodResult(
        bits=bits,
        symbols=symbols,
        decisions=decisions,
        evm_rms=0.0,
        modulation="2FSK",
    )


# ── 4-FSK demodulator ─────────────────────────────────────────────────────────

def demod_4fsk(
    symbols: np.ndarray,
    sps: int = 1,
    tone_freqs: tuple[float, float, float, float] | None = None,
) -> DemodResult:
    """
    Non-coherent hard-decision 4-FSK demodulator.

    Dibit assignment (Gray-coded by tone index, lowest → highest):
      dibit 00  →  tone 0 (lowest frequency)
      dibit 01  →  tone 1
      dibit 11  →  tone 2
      dibit 10  →  tone 3 (highest frequency)

    Args:
        symbols:     1-D complex array (``sps`` samples per symbol if > 1).
        sps:         Samples per symbol.  Default 1.
        tone_freqs:  Optional 4-tuple (f0, f1, f2, f3) in rad/sample in
                     ascending order.  If None, the four tones are estimated
                     from the observed frequency distribution by k-means-like
                     quantile partitioning.

    Returns:
        DemodResult with bits (2 bits/symbol), decisions, evm_rms=0.0,
        modulation="4FSK".
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if len(symbols) == 0:
        return DemodResult(
            bits=np.array([], dtype=np.uint8),
            symbols=symbols,
            decisions=np.array([], dtype=np.complex64),
            evm_rms=0.0,
            modulation="4FSK",
        )

    freq_est = _avg_freq_per_symbol(symbols, sps)   # [n_sym]
    n_sym = len(freq_est)

    if tone_freqs is not None:
        tones = np.asarray(tone_freqs, dtype=np.float64)
        if len(tones) != 4:
            raise ValueError(f"tone_freqs must have exactly 4 elements, got {len(tones)}")
        tones = np.sort(tones)
    else:
        # Estimate the 4 tone centres from the observed distribution.
        # Assign each sample to its nearest quartile centroid.
        # This is a simple, parameter-free approach that works when tone
        # occupancies are roughly equal (as in random data).
        q = np.percentile(freq_est, [12.5, 37.5, 62.5, 87.5])
        tones = q.astype(np.float64)

    # Decision: nearest tone
    # Shape: [n_sym, 4]
    dist_to_tones = np.abs(freq_est[:, np.newaxis] - tones[np.newaxis, :])
    tone_idx = np.argmin(dist_to_tones, axis=1)  # 0..3

    # Bits: Gray-coded lookup
    bits_2d = np.array([_4FSK_GRAY[ti] for ti in tone_idx], dtype=np.uint8)
    bits = bits_2d.ravel()

    decisions = freq_est.astype(np.complex64)

    return DemodResult(
        bits=bits,
        symbols=symbols,
        decisions=decisions,
        evm_rms=0.0,
        modulation="4FSK",
    )
