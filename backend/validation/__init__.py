"""
Validation package for SIGMA.
Exposes CRC calculation engines and FrameValidator.
"""
from backend.validation.crc import (
    CRCParam,
    CRC_CATALOG,
    compute_crc_bytes,
    compute_crc_bits,
    check_crc_bits,
    append_crc_to_bits,
    get_crc_param,
)
from backend.validation.frame_validator import FrameValidator

__all__ = [
    "CRCParam",
    "CRC_CATALOG",
    "compute_crc_bytes",
    "compute_crc_bits",
    "check_crc_bits",
    "append_crc_to_bits",
    "get_crc_param",
    "FrameValidator",
]
