"""
Signal representation definitions for the SIGMA ML pipeline.

Supported representations:
    RAW_IQ          → [I, Q]                  2 channels   (existing baseline)
    IQ_AMPLITUDE    → [I, Q, |z|]             3 channels
    AMPLITUDE_PHASE → [|z|, Δφ]               2 channels   (phase-difference)
    IQ_AMP_PHASE    → [I, Q, |z|, Δφ]         4 channels

Nomenclature:
    |z| = instantaneous amplitude = sqrt(I² + Q²)
    Δφ  = instantaneous phase difference  = angle(z[n] · conj(z[n-1]))
          (zero-padded at sample 0)

Input conventions:
    Single sample: [2, N]  float32 canonical IQ
    Batch:         [B, 2, N]  float32 canonical IQ

Output:
    Single sample: [C, N]  float32 where C = channel count for the representation
    Batch:         [B, C, N]  float32

The I and Q channels in RAW_IQ, IQ_AMPLITUDE, and IQ_AMP_PHASE are taken
directly from the input without modification to ensure backward compatibility.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Representation(str, Enum):
    """Canonical representation identifiers."""

    RAW_IQ          = "RAW_IQ"
    IQ_AMPLITUDE    = "IQ_AMPLITUDE"
    AMPLITUDE_PHASE = "AMPLITUDE_PHASE"
    IQ_AMP_PHASE    = "IQ_AMP_PHASE"


class RepresentationInfo(NamedTuple):
    """Static metadata about a representation."""

    name: str
    num_channels: int
    description: str


# Registry of all supported representations
REPRESENTATION_INFO: dict[Representation, RepresentationInfo] = {
    Representation.RAW_IQ: RepresentationInfo(
        name="RAW_IQ",
        num_channels=2,
        description="Raw in-phase and quadrature channels [I, Q]",
    ),
    Representation.IQ_AMPLITUDE: RepresentationInfo(
        name="IQ_AMPLITUDE",
        num_channels=3,
        description="I, Q, and instantaneous amplitude [I, Q, |z|]",
    ),
    Representation.AMPLITUDE_PHASE: RepresentationInfo(
        name="AMPLITUDE_PHASE",
        num_channels=2,
        description="Instantaneous amplitude and phase-difference [|z|, Δφ]",
    ),
    Representation.IQ_AMP_PHASE: RepresentationInfo(
        name="IQ_AMP_PHASE",
        num_channels=4,
        description="I, Q, instantaneous amplitude, and phase-difference [I, Q, |z|, Δφ]",
    ),
}


def num_channels(rep: Representation | str) -> int:
    """Return the number of output channels for a given representation."""
    if isinstance(rep, str):
        rep = Representation(rep)
    return REPRESENTATION_INFO[rep].num_channels
