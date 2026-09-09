"""
Deterministic Rectangular Matrix Block Interleaver & Deinterleaver for SIGMA.
Implements:
  - Canonical matrix interleaving: write by rows, read by columns.
  - Canonical matrix deinterleaving: write by columns, read by rows.
  - Strict block length validation.
  - High-performance vectorized NumPy execution.
"""
import re
from typing import Optional, Tuple, Dict, Any
import numpy as np

from backend.interleaver.interface import Deinterleaver
try:
    from backend.hypothesis.decoder_contracts import InterleaverResult, DecoderStageStatus
except ImportError:
    from hypothesis.decoder_contracts import InterleaverResult, DecoderStageStatus


def parse_block_interleaver_config(config_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parses configuration strings like 'block_16x16', 'block_4x8', 'matrix_8x8'.
    Returns (rows, cols) if valid, or None.
    """
    if not config_str:
        return None
    m = re.match(r"^(?:block|matrix)_([0-9]+)x([0-9]+)$", str(config_str).strip().lower())
    if not m:
        return None
    rows, cols = int(m.group(1)), int(m.group(2))
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def interleave_block(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Applies rectangular matrix interleaving: writes by rows, reads by columns.

    Args:
        bits: 1D uint8 array of bits. Must be a multiple of rows * cols.
        rows: Matrix rows count.
        cols: Matrix columns count.

    Returns:
        1D uint8 array of interleaved bits.
    """
    if bits is None or len(bits) == 0:
        return np.empty(0, dtype=np.uint8)

    arr = np.asarray(bits, dtype=np.uint8).flatten()
    block_size = rows * cols
    if len(arr) % block_size != 0:
        raise ValueError(
            f"Bitstream length ({len(arr)}) is not an exact multiple of block size {rows}x{cols} = {block_size}"
        )

    # Reshape each block to (rows, cols), transpose to (cols, rows), and flatten
    num_blocks = len(arr) // block_size
    interleaved = arr.reshape(num_blocks, rows, cols).swapaxes(1, 2).flatten()
    return interleaved.astype(np.uint8)


def deinterleave_block(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Reverses rectangular matrix interleaving: writes by columns, reads by rows.

    Args:
        bits: 1D uint8 array of bits. Must be a multiple of rows * cols.
        rows: Matrix rows count.
        cols: Matrix columns count.

    Returns:
        1D uint8 array of deinterleaved bits.
    """
    if bits is None or len(bits) == 0:
        return np.empty(0, dtype=np.uint8)

    arr = np.asarray(bits, dtype=np.uint8).flatten()
    block_size = rows * cols
    if len(arr) % block_size != 0:
        raise ValueError(
            f"Bitstream length ({len(arr)}) is not an exact multiple of block size {rows}x{cols} = {block_size}"
        )

    # Received column-read data has shape (cols, rows) per block.
    # Swap axes back to (rows, cols) and flatten to recover row-wise order.
    num_blocks = len(arr) // block_size
    deinterleaved = arr.reshape(num_blocks, cols, rows).swapaxes(1, 2).flatten()
    return deinterleaved.astype(np.uint8)


class BlockDeinterleaver(Deinterleaver):
    """
    Modular block deinterleaver for block_RxC configurations.
    """

    def __init__(self, rows: int, cols: int):
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Rows and cols must be positive integers, got {rows}x{cols}")
        self.rows = rows
        self.cols = cols
        self.block_size = rows * cols

    def deinterleave(self, bits: np.ndarray) -> InterleaverResult:
        if bits is None or len(bits) == 0:
            return InterleaverResult.failed(
                failure_reason="No input bits provided for deinterleaving"
            )

        arr = np.asarray(bits, dtype=np.uint8).flatten()
        n_bits = len(arr)

        if n_bits % self.block_size != 0:
            return InterleaverResult.failed(
                failure_reason=(
                    f"Bitstream length ({n_bits}) is not a multiple of interleaver matrix "
                    f"block size ({self.rows}x{self.cols} = {self.block_size})"
                ),
                metrics={
                    "rows": self.rows,
                    "cols": self.cols,
                    "block_size": self.block_size,
                    "bitstream_length": n_bits,
                },
            )

        try:
            deinterleaved = deinterleave_block(arr, self.rows, self.cols)
            return InterleaverResult.create_success(
                deinterleaved_bits=deinterleaved,
                metrics={
                    "interleaver_type": "block",
                    "rows": self.rows,
                    "cols": self.cols,
                    "block_size": self.block_size,
                    "num_blocks": n_bits // self.block_size,
                },
            )
        except Exception as e:
            return InterleaverResult.failed(
                failure_reason=f"Block deinterleaving failed: {str(e)}"
            )
