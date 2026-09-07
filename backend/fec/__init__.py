"""
FEC package — Convolutional encoding and Viterbi decoding.

Implements a rate-1/2 constraint-length-7 (K=7) convolutional code,
the same standard used in many real-world BPSK/QPSK links (e.g. NASA,
CCSDS Telemetry).

Generators (octal): G1=0o171, G2=0o133  (standard CCSDS/NASA polynomials)

Public API:
    encode(bits)              → encoded_bits (rate-1/2, BPSK-ready)
    decode(soft_bits)         → decoded_bits (Viterbi, hard or soft input)
    check_bitstream(bits)     → BitstreamCheckResult (pass/fail + metrics)
"""
from fec.convolutional import encode, decode
from fec.bitstream_check import check_bitstream, BitstreamCheckResult

__all__ = ["encode", "decode", "check_bitstream", "BitstreamCheckResult"]
