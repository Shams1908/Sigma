"""
ml.representations — signal representation utilities for SIGMA ML models.

Public API:
    Representation          enum of supported representation names
    RepresentationInfo      named-tuple with num_channels + description
    REPRESENTATION_INFO     dict[Representation, RepresentationInfo]
    num_channels(rep)       convenience helper

    apply_representation(iq, rep)   generic dispatcher
    to_raw_iq(iq)
    to_iq_amplitude(iq)
    to_amplitude_phase(iq)
    to_iq_amp_phase(iq)
"""
from ml.representations.definitions import (
    Representation,
    RepresentationInfo,
    REPRESENTATION_INFO,
    num_channels,
)
from ml.representations.transforms import (
    apply_representation,
    to_raw_iq,
    to_iq_amplitude,
    to_amplitude_phase,
    to_iq_amp_phase,
)

__all__ = [
    "Representation",
    "RepresentationInfo",
    "REPRESENTATION_INFO",
    "num_channels",
    "apply_representation",
    "to_raw_iq",
    "to_iq_amplitude",
    "to_amplitude_phase",
    "to_iq_amp_phase",
]
