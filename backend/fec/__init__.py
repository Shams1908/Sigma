"""
FEC package for SIGMA.
Exposes FECDecoder, ConvolutionalFECDecoder, and convolutional encoding helpers.
"""
from backend.fec.interface import FECDecoder
from backend.fec.convolutional import (
    ConvolutionalFECDecoder,
    encode_convolutional_r12_k7,
    POLY_G1,
    POLY_G2,
    CONSTRAINT_LENGTH,
)

__all__ = [
    "FECDecoder",
    "ConvolutionalFECDecoder",
    "encode_convolutional_r12_k7",
    "POLY_G1",
    "POLY_G2",
    "CONSTRAINT_LENGTH",
]
