"""
Bitstream and Frame Validator for SIGMA.
Validates received bitstreams against configured CRC polynomials and optional framing sync words.
Does NOT fabricate validation success or enforce arbitrary protocols unless explicitly requested.
"""
from typing import Optional, Dict, Any, Union
import numpy as np

from backend.validation.crc import (
    check_crc_bits,
    get_crc_param,
    CRC_CATALOG,
)
try:
    from backend.hypothesis.decoder_contracts import ValidationResult, DecoderStageStatus
except ImportError:
    from hypothesis.decoder_contracts import ValidationResult, DecoderStageStatus


class FrameValidator:
    """
    Validates decoded bitstreams against explicit framing and CRC rules.
    """

    def validate(
        self,
        bits: Optional[np.ndarray],
        crc_scheme: Optional[str] = None,
        sync_word: Optional[Union[str, np.ndarray, list]] = None,
        frame_length: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validates the bitstream.

        Args:
            bits: 1D uint8 array of decoded or demodulated bits.
            crc_scheme: Optional CRC algorithm name (e.g. 'CRC-16-CCITT', 'CRC-32', 'CRC-8').
            sync_word: Optional bit array or hex string for preamble/sync word search.
            frame_length: Optional expected frame length in bits.

        Returns:
            ValidationResult: Structured outcome with explicit validity and telemetry.
        """
        if bits is None or len(bits) == 0:
            return ValidationResult.failed(
                failure_reason="No bits available for validation",
                crc_status="not_evaluated",
            )

        raw_bits = np.asarray(bits, dtype=np.uint8).flatten()

        # If no validation scheme is requested, return NOT_EVALUATED (neutral)
        if not crc_scheme or str(crc_scheme).lower() in ("none", "not_evaluated"):
            return ValidationResult.not_evaluated(
                details={"reason": "No CRC scheme or validation rule configured"}
            )

        # Check if requested CRC scheme is supported
        norm_crc = str(crc_scheme).strip().lower()
        if norm_crc not in CRC_CATALOG:
            return ValidationResult.not_supported(
                failure_reason=f"CRC scheme '{crc_scheme}' is not supported",
                details={"supported_schemes": list(CRC_CATALOG.keys())},
            )

        param = get_crc_param(norm_crc)

        # 1. Sync word alignment (if requested)
        working_bits = raw_bits
        frame_status = None

        if sync_word is not None:
            sync_pattern = self._parse_sync_word(sync_word)
            if sync_pattern is not None and len(sync_pattern) > 0:
                idx = self._find_sync_word(raw_bits, sync_pattern)
                if idx < 0:
                    return ValidationResult.failed(
                        failure_reason="Sync word / preamble not found in bitstream",
                        crc_status="invalid",
                        metrics={"sync_word_len": len(sync_pattern)},
                    )
                # Slice frame body starting after sync word
                working_bits = raw_bits[idx + len(sync_pattern):]
                frame_status = f"aligned_at_bit_{idx}"

        # 2. Frame length check (if requested)
        if frame_length is not None and frame_length > 0:
            if len(working_bits) < frame_length:
                return ValidationResult(
                    status=DecoderStageStatus.FAILED,
                    is_valid=False,
                    crc_status="invalid",
                    frame_status=frame_status,
                    failure_reason=f"Extracted frame ({len(working_bits)} bits) is shorter than expected ({frame_length} bits)",
                )
            working_bits = working_bits[:frame_length]

        # 3. Minimum length for CRC
        if len(working_bits) <= param.width:
            return ValidationResult(
                status=DecoderStageStatus.FAILED,
                is_valid=False,
                crc_status="invalid",
                frame_status=frame_status,
                failure_reason=f"Bitstream length ({len(working_bits)}) is too short to contain {param.width}-bit CRC",
                metrics={"required_min_length": param.width + 1},
            )

        # 4. Compute and check CRC
        is_valid, payload, expected_crc, observed_crc = check_crc_bits(working_bits, crc_name=norm_crc)

        metrics = {
            "crc_poly": param.name,
            "crc_width": param.width,
            "expected_crc": hex(expected_crc),
            "observed_crc": hex(observed_crc),
            "payload_bits_count": len(payload),
            "frame_status": frame_status,
        }

        if is_valid:
            return ValidationResult.create_success(
                is_valid=True,
                crc_status="valid",
                frame_status=frame_status,
                validation_score=1.0,
                metrics=metrics,
            )
        else:
            return ValidationResult(
                status=DecoderStageStatus.FAILED,
                is_valid=False,
                crc_status="invalid",
                frame_status=frame_status,
                failure_reason=f"CRC checksum mismatch: expected {hex(expected_crc)}, got {hex(observed_crc)}",
                metrics=metrics,
            )

    def _parse_sync_word(self, sync_word: Union[str, np.ndarray, list]) -> Optional[np.ndarray]:
        """Parses sync word representations into a 1D uint8 array."""
        if isinstance(sync_word, str):
            clean = sync_word.strip()
            if clean.startswith("0x") or clean.startswith("0X"):
                # Hex string
                val = int(clean, 16)
                bit_len = (len(clean) - 2) * 4
                bits = [(val >> (bit_len - 1 - i)) & 1 for i in range(bit_len)]
                return np.array(bits, dtype=np.uint8)
            elif clean.startswith("0b") or clean.startswith("0B"):
                # Binary string
                bits = [int(c) for c in clean[2:] if c in ("0", "1")]
                return np.array(bits, dtype=np.uint8)
            elif all(c in ("0", "1") for c in clean):
                return np.array([int(c) for c in clean], dtype=np.uint8)
        return np.asarray(sync_word, dtype=np.uint8).flatten()

    def _find_sync_word(self, bits: np.ndarray, pattern: np.ndarray) -> int:
        """Finds the starting index of a pattern within bits using correlation search."""
        p_len = len(pattern)
        b_len = len(bits)
        if p_len > b_len:
            return -1

        for i in range(b_len - p_len + 1):
            if np.array_equal(bits[i : i + p_len], pattern):
                return i
        return -1
