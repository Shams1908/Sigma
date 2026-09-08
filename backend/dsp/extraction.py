"""
Signal extraction and isolation module for DSP pipeline.

Re-exports ExtractedSignal container and extract_signal pipeline function from
preprocessing.filtering.
"""

from __future__ import annotations

from preprocessing.filtering import ExtractedSignal, extract_signal, lowpass_filter

__all__ = ["ExtractedSignal", "extract_signal", "lowpass_filter"]
