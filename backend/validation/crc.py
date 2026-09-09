"""
CRC (Cyclic Redundancy Check) calculation engine for SIGMA.
Implements exact mathematical bit-level and byte-level algorithms for:
  - CRC-16-CCITT (poly 0x1021, init 0xFFFF, refIn=False, refOut=False, xorOut=0x0000)
  - CRC-16-CCITT-0 (poly 0x1021, init 0x0000, refIn=False, refOut=False, xorOut=0x0000)
  - CRC-32 (IEEE 802.3, poly 0x04C11DB7, init 0xFFFFFFFF, refIn=True, refOut=True, xorOut=0xFFFFFFFF)
  - CRC-8 (ATM / SMBus, poly 0x07, init 0x00, refIn=False, refOut=False, xorOut=0x00)

Matches standard reference vectors (e.g. ASCII "123456789").
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class CRCParam:
    """Explicit parameters defining a CRC algorithm according to standard catalog conventions."""
    name: str
    width: int
    poly: int
    init: int
    ref_in: bool
    ref_out: bool
    xor_out: int


# Canonical CRC definitions
CRC_CATALOG: Dict[str, CRCParam] = {
    "crc-16-ccitt": CRCParam(
        name="CRC-16-CCITT",
        width=16,
        poly=0x1021,
        init=0xFFFF,
        ref_in=False,
        ref_out=False,
        xor_out=0x0000,
    ),
    "crc-16": CRCParam(
        name="CRC-16-CCITT",
        width=16,
        poly=0x1021,
        init=0xFFFF,
        ref_in=False,
        ref_out=False,
        xor_out=0x0000,
    ),
    "crc-16-ccitt-0": CRCParam(
        name="CRC-16-CCITT-0",
        width=16,
        poly=0x1021,
        init=0x0000,
        ref_in=False,
        ref_out=False,
        xor_out=0x0000,
    ),
    "crc-32": CRCParam(
        name="CRC-32",
        width=32,
        poly=0x04C11DB7,
        init=0xFFFFFFFF,
        ref_in=True,
        ref_out=True,
        xor_out=0xFFFFFFFF,
    ),
    "crc-8": CRCParam(
        name="CRC-8",
        width=8,
        poly=0x07,
        init=0x00,
        ref_in=False,
        ref_out=False,
        xor_out=0x00,
    ),
}


def _reflect(val: int, width: int) -> int:
    """Reverses the bit order of an integer of specified bit width."""
    res = 0
    for i in range(width):
        if (val >> i) & 1:
            res |= (1 << (width - 1 - i))
    return res


def get_crc_param(name: str) -> CRCParam:
    """Retrieves standard CRC parameters or raises KeyError."""
    norm = name.strip().lower()
    if norm not in CRC_CATALOG:
        raise KeyError(f"CRC algorithm '{name}' is not in the supported catalog: {list(CRC_CATALOG.keys())}")
    return CRC_CATALOG[norm]


def compute_crc_bytes(data: Union[bytes, bytearray, np.ndarray], crc_name: str = "CRC-16-CCITT") -> int:
    """
    Computes standard CRC integer over a sequence of bytes.

    Args:
        data: bytes-like object or 1D array of uint8 values.
        crc_name: Name of standard CRC algorithm in catalog.

    Returns:
        Integer checksum value.
    """
    param = get_crc_param(crc_name)
    raw_bytes = bytes(data)

    reg = param.init
    mask = (1 << param.width) - 1
    top_bit = 1 << (param.width - 1)

    for byte in raw_bytes:
        b = _reflect(byte, 8) if param.ref_in else byte
        reg ^= (b << (param.width - 8))
        for _ in range(8):
            if reg & top_bit:
                reg = ((reg << 1) ^ param.poly) & mask
            else:
                reg = (reg << 1) & mask

    if param.ref_out:
        reg = _reflect(reg, param.width)

    return (reg ^ param.xor_out) & mask


def compute_crc_bits(bits: np.ndarray, crc_name: str = "CRC-16-CCITT") -> int:
    """
    Computes CRC directly from a 1D binary bit array (0 or 1, MSB-first per byte).
    If bitstream length is not a multiple of 8, pads with trailing zeros to complete last byte.
    """
    if bits is None or len(bits) == 0:
        return 0

    arr = np.asarray(bits, dtype=np.uint8).flatten()
    pad_len = (8 - (len(arr) % 8)) % 8
    if pad_len > 0:
        arr = np.pad(arr, (0, pad_len), mode="constant", constant_values=0)

    # Pack bits to bytes (MSB first)
    byte_arr = np.packbits(arr)
    return compute_crc_bytes(byte_arr, crc_name=crc_name)


def crc_to_bits(crc_val: int, width: int) -> np.ndarray:
    """
    Converts an integer CRC value to a 1D uint8 bit array (MSB first).
    """
    bits = [(crc_val >> (width - 1 - i)) & 1 for i in range(width)]
    return np.array(bits, dtype=np.uint8)


def bits_to_crc(bits: np.ndarray) -> int:
    """
    Converts a 1D uint8 bit array (MSB first) to an integer.
    """
    val = 0
    for b in bits:
        val = (val << 1) | (int(b) & 1)
    return val


def append_crc_to_bits(
    payload_bits: np.ndarray,
    crc_name: str = "CRC-16-CCITT",
) -> np.ndarray:
    """
    Computes CRC over payload bits and appends trailing CRC bits.

    Returns:
        1D uint8 array: payload_bits concatenated with trailing CRC bits.
    """
    param = get_crc_param(crc_name)
    crc_val = compute_crc_bits(payload_bits, crc_name=crc_name)
    crc_bits = crc_to_bits(crc_val, param.width)
    return np.concatenate([np.asarray(payload_bits, dtype=np.uint8).flatten(), crc_bits])


def check_crc_bits(
    bits_with_crc: np.ndarray,
    crc_name: str = "CRC-16-CCITT",
) -> Tuple[bool, np.ndarray, int, int]:
    """
    Extracts trailing CRC bits, recalculates CRC over payload, and validates match.

    Args:
        bits_with_crc: 1D uint8 bit array containing payload + trailing CRC.
        crc_name: Name of standard CRC algorithm.

    Returns:
        Tuple[is_valid, payload_bits, expected_crc, observed_crc]
    """
    param = get_crc_param(crc_name)
    arr = np.asarray(bits_with_crc, dtype=np.uint8).flatten()

    if len(arr) < param.width:
        return False, arr, 0, 0

    payload = arr[:-param.width]
    observed_bits = arr[-param.width:]
    observed_crc = bits_to_crc(observed_bits)

    expected_crc = compute_crc_bits(payload, crc_name=crc_name)
    is_valid = (expected_crc == observed_crc)

    return is_valid, payload, expected_crc, observed_crc
