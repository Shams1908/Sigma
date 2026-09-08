"""
PSD-based signal region detection and spectral segmentation module.

Provides tools to estimate noise floor, threshold PSD energy, and segment
wideband IQ signals into distinct occupied frequency regions.
"""

from __future__ import annotations

from dsp.detection import (
    SignalRegion,
    detect_signal_regions,
    find_contiguous_regions,
    merge_spectral_gaps,
)

__all__ = [
    "SignalRegion",
    "detect_signal_regions",
    "find_contiguous_regions",
    "merge_spectral_gaps",
]
