"""
Frame / preamble acquisition for the SIGMA synchronization pipeline.

Answers one question:
    "At what sample index (in the RRC-matched-filter output) does the known
    preamble frame begin, and how confident are we?"

Algorithm
---------
The acquisition is a normalized complex cross-correlation performed in the
RRC matched-filter output domain.  The correlation template is constructed
to match what ``apply_matched_filter`` actually produces when the preamble
is present at the start of the received buffer.

Template construction
---------------------
Given N_p preamble symbols at samples-per-symbol ``sps`` with roll-off ``β``
and filter span ``span``:

1.  Upsample preamble symbols by ``sps`` (insert sps-1 zeros between each).
    Length: N_p * sps.

2.  Convolve with RRC taps using ``mode="full"`` — the Tx pulse-shaping step.
    This mirrors ``ml.generators.pulse_shaping.rrc.pulse_shape``.
    Length: N_p * sps + span * sps.

3.  Convolve the result with the same RRC taps using ``mode="same"`` — the
    Rx matched-filtering step.
    This mirrors ``synchronization.matched_filter.apply_matched_filter``,
    which also uses ``mode="same"``.
    Length: N_p * sps + span * sps  (unchanged by same-mode convolution).

The ``mode="same"`` Rx step centres the filter impulse response, absorbing
the half-delay that ``mode="full"`` would otherwise introduce.  The combined
Tx-RRC (full) ⊗ Rx-RRC (same) template has its first-symbol energy peak
at sample index ``(len(rrc_taps) - 1) // 2 = span * sps // 2`` — but this
is internal to the template and is already accounted for by the correlation:
when the template is slid over the received MF output, the correlation peak
lag directly gives the position of the template's start in the signal.

What start_index means
----------------------
``start_index`` is the index in the **received RRC-MF-output signal** where
the first preamble symbol's energy peak lands, relative to the beginning of
the signal buffer.

More precisely: ``start_index`` is the lag at which the template most closely
matches the signal.  At that lag, ``signal[start_index : start_index + T]``
best matches the template ``template[0 : T]``, so the frame begins at
``start_index``.

If you then downsample the MF output at rate ``sps`` starting from
``start_index + (span * sps) // 2``, the first output sample corresponds to
the first preamble symbol.  (The ``(span * sps) // 2`` term accounts for the
half-delay of a single ``mode="same"`` RRC filter pass.)

Filter delay / transient compensation
--------------------------------------
Because the template is built using Tx ``mode="full"`` followed by Rx
``mode="same"``, the ``mode="same"`` step absorbs the delay of the Rx filter
exactly as the live signal processing does.  The remaining Tx ``mode="full"``
half-delay is baked into the template's shape.

Consequently ``template_delay = 0`` — there is **no** offset to subtract from
the raw correlation peak.  ``start_index = corr_peak_index``.

This was validated experimentally: with a noiseless signal at a known offset
``d``, the correlation peak falls at lag ``d``, so ``start_index = d`` exactly.

Detection threshold
-------------------
After normalizing the correlation to [0, 1] (dividing by the RMS of both
the template and the local signal window), any peak below ``threshold``
(default 0.3) is rejected as not found.  0.3 is a practical lower bound at
SNR ≈ 5 dB for a 16-symbol preamble.  Tune as needed.

No-match behavior
-----------------
If no correlation peak meets the threshold, ``FrameSyncResult.found`` is
``False``, ``start_index`` is ``-1``, and ``confidence`` holds the best-seen
normalized score (useful for diagnostics).

Short-input behavior
---------------------
If the signal is shorter than the template, no valid correlation window
exists.  The function returns ``found=False`` with ``confidence=0.0``
immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class FrameSyncResult:
    """
    Result of a frame/preamble acquisition attempt.

    Attributes:
        found:           True when a peak above ``threshold`` was found.
        start_index:     Lag (= sample index in the MF-output signal) where
                         the preamble template best matches.  -1 when not found.
        confidence:      Normalized correlation magnitude at the best candidate
                         position, in [0, 1].  Present even when not found.
        corr_peak_index: The raw argmax of the normalized correlation array.
                         Equal to ``start_index`` when found; -1 otherwise.
        template_delay:  Always 0 for this implementation.  Kept for API
                         symmetry and future extension.
    """
    found: bool
    start_index: int
    confidence: float
    corr_peak_index: int
    template_delay: int


# ── Template builder ──────────────────────────────────────────────────────────

def build_rc_template(
    preamble_symbols: np.ndarray,
    samples_per_symbol: int,
    rolloff: float = 0.35,
    filter_span_symbols: int = 8,
) -> tuple[np.ndarray, int]:
    """
    Build the expected matched-filter-output template for a known preamble.

    The template is constructed by:
      1. Upsampling the preamble symbols.
      2. Tx RRC filtering with ``mode="full"`` (matches ``pulse_shape``).
      3. Rx RRC filtering with ``mode="same"`` (matches ``apply_matched_filter``).

    This produces a template whose shape exactly mirrors what the live
    received-signal MF output looks like when the preamble is present at
    position 0 — giving ``template_delay = 0`` (no offset correction needed).

    Args:
        preamble_symbols:    1-D complex array of preamble symbols at
                             1 sample/symbol.  Must not be empty.
        samples_per_symbol:  Oversampling factor.  Must match Tx/Rx config.
        rolloff:             RRC roll-off factor β.  Must match Tx/Rx pair.
        filter_span_symbols: RRC filter span in symbols.  Must match Tx/Rx.

    Returns:
        template:       1-D complex64 RC-shaped template waveform.
        template_delay: 0 (no correction needed; returned for API symmetry).

    Raises:
        TypeError:  If ``preamble_symbols`` is not a numpy array.
        ValueError: If ``preamble_symbols`` is empty, not 1-D, or
                    ``samples_per_symbol`` < 1.
    """
    if not isinstance(preamble_symbols, np.ndarray):
        raise TypeError(
            f"preamble_symbols must be a numpy ndarray, "
            f"got {type(preamble_symbols).__name__}"
        )
    if preamble_symbols.ndim != 1:
        raise ValueError(
            f"preamble_symbols must be 1-D, got shape {preamble_symbols.shape}"
        )
    if len(preamble_symbols) == 0:
        raise ValueError("preamble_symbols must not be empty")
    if samples_per_symbol < 1:
        raise ValueError(
            f"samples_per_symbol must be ≥ 1, got {samples_per_symbol}"
        )

    from ml.generators.pulse_shaping.rrc import design_rrc_filter  # noqa: PLC0415

    sps = int(samples_per_symbol)
    span = int(filter_span_symbols)

    h_rrc = design_rrc_filter(
        samples_per_symbol=sps,
        rolloff=float(rolloff),
        filter_span_symbols=span,
    ).astype(np.float64)

    # Step 1: Upsample preamble symbols (N_p × sps samples)
    n_sym = len(preamble_symbols)
    upsampled = np.zeros(n_sym * sps, dtype=np.complex128)
    upsampled[::sps] = preamble_symbols.astype(np.complex128)

    # Step 2: Tx RRC shaping — mode="full", matches pulse_shape()
    #   Output length: N_p*sps + span*sps
    tx_shaped = np.convolve(upsampled, h_rrc, mode="full")

    # Step 3: Rx RRC matched filtering — mode="same", matches apply_matched_filter()
    #   Output length: same as tx_shaped = N_p*sps + span*sps
    rc_template = np.convolve(tx_shaped, h_rrc, mode="same")

    # template_delay = 0: the mode="same" Rx step absorbs its own half-delay
    # so no correction is needed when reading start_index from the correlation.
    return rc_template.astype(np.complex64), 0


# ── Frame synchronizer ────────────────────────────────────────────────────────

class BasicFrameSynchronizer:
    """
    Locate a known preamble in the RRC-matched-filter output of a received
    signal using normalized complex cross-correlation.

    Usage
    -----
    Construct once with the known preamble parameters, then call ``find()``
    on each received signal buffer.

    The received signal passed to ``find()`` should be in the RRC
    matched-filter output domain (after ``apply_matched_filter()``).
    Set ``apply_mf=True`` to have the class apply the matched filter itself
    before correlating.

    Parameters
    ----------
    preamble_symbols : np.ndarray
        1-D complex array of preamble symbols at 1 sample/symbol.
    samples_per_symbol : int
        Oversampling factor matching the Tx/Rx RRC filter design.
    rolloff : float
        RRC roll-off factor β.  Must match the Tx/Rx pair.
    filter_span_symbols : int
        RRC filter span in symbols.  Must match the Tx/Rx pair.
    threshold : float
        Normalized correlation magnitude threshold in [0, 1].  Default 0.3.
    """

    def __init__(
        self,
        preamble_symbols: np.ndarray,
        samples_per_symbol: int,
        rolloff: float = 0.35,
        filter_span_symbols: int = 8,
        threshold: float = 0.3,
    ) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )

        self._sps = int(samples_per_symbol)
        self._rolloff = float(rolloff)
        self._span = int(filter_span_symbols)
        self._threshold = float(threshold)

        self._template, self._template_delay = build_rc_template(
            preamble_symbols=preamble_symbols,
            samples_per_symbol=self._sps,
            rolloff=self._rolloff,
            filter_span_symbols=self._span,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def template(self) -> np.ndarray:
        """The correlation template (read-only)."""
        return self._template

    @property
    def template_delay(self) -> int:
        """Always 0 for this implementation."""
        return self._template_delay

    @property
    def threshold(self) -> float:
        """Normalized correlation threshold."""
        return self._threshold

    def find(
        self,
        signal: np.ndarray,
        apply_mf: bool = False,
    ) -> FrameSyncResult:
        """
        Search for the preamble in the received signal.

        Args:
            signal:    1-D complex array (RRC-MF output, or raw oversampled
                       signal if ``apply_mf=True``).
            apply_mf:  If True, apply the RRC matched filter before correlating.

        Returns:
            FrameSyncResult.

        Raises:
            TypeError:  If signal is not a numpy array.
            ValueError: If signal is not 1-D.
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError(
                f"signal must be a numpy ndarray, got {type(signal).__name__}"
            )
        if signal.ndim != 1:
            raise ValueError(
                f"signal must be 1-D, got shape {signal.shape}"
            )

        if apply_mf:
            from synchronization.matched_filter import apply_matched_filter  # noqa: PLC0415
            signal = apply_matched_filter(
                signal,
                samples_per_symbol=self._sps,
                rolloff=self._rolloff,
                filter_span_symbols=self._span,
            )

        signal = signal.astype(np.complex128)
        template = self._template.astype(np.complex128)
        t_len = len(template)

        # Signal shorter than the template: no valid correlation window.
        if len(signal) < t_len:
            return FrameSyncResult(
                found=False,
                start_index=-1,
                confidence=0.0,
                corr_peak_index=-1,
                template_delay=0,
            )

        # ── Normalized cross-correlation ──────────────────────────────────────
        #
        # np.correlate(signal, template, 'valid')[k]
        #   = sum_{n=0}^{T-1}  signal[k+n] * conj(template[n])
        #
        # The result has length  len(signal) - T + 1.
        # Lag k is where the template's index 0 aligns with signal index k.
        # The peak lag k_peak therefore means: the template best matches the
        # signal starting at position k_peak → start_index = k_peak.

        raw_corr = np.correlate(signal, template, mode="valid")

        template_energy = float(np.sum(np.abs(template) ** 2))
        template_norm = np.sqrt(max(template_energy, 1e-30))

        n_lags = len(raw_corr)
        sig_sq = np.abs(signal) ** 2
        cum_sq = np.concatenate([[0.0], np.cumsum(sig_sq)])
        window_energy = cum_sq[t_len : t_len + n_lags] - cum_sq[:n_lags]
        window_norm = np.sqrt(np.maximum(window_energy, 1e-30))

        normalized_corr = np.abs(raw_corr) / (window_norm * template_norm)

        corr_peak_index = int(np.argmax(normalized_corr))
        best_confidence = float(normalized_corr[corr_peak_index])

        # start_index = corr_peak_index because template_delay = 0
        start_index = corr_peak_index

        found = best_confidence >= self._threshold

        return FrameSyncResult(
            found=found,
            start_index=start_index if found else -1,
            confidence=best_confidence,
            corr_peak_index=corr_peak_index,
            template_delay=0,
        )


# ── Convenience function ──────────────────────────────────────────────────────

def find_frame_start(
    signal: np.ndarray,
    preamble_symbols: np.ndarray,
    samples_per_symbol: int,
    rolloff: float = 0.35,
    filter_span_symbols: int = 8,
    threshold: float = 0.3,
    apply_mf: bool = False,
) -> FrameSyncResult:
    """
    Convenience wrapper: build a BasicFrameSynchronizer and call find() once.

    Use BasicFrameSynchronizer directly when processing multiple frames with
    the same preamble to avoid rebuilding the template each time.
    """
    syncer = BasicFrameSynchronizer(
        preamble_symbols=preamble_symbols,
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        filter_span_symbols=filter_span_symbols,
        threshold=threshold,
    )
    return syncer.find(signal, apply_mf=apply_mf)
