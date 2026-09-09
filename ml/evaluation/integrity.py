"""
M7-A: Dataset Integrity and Split Isolation Auditor.

Enforces strict structural, type, and provenance invariants for external
and benchmark evaluation datasets. Fails loudly on any anomaly (no silent repairs).
"""
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import numpy as np

from ml.dataset.external_dataset import (
    EXTERNAL_MODULATION_CLASSES,
    EXTERNAL_CHANNEL_CONDITIONS,
    EXTERNAL_SNR_VALUES,
    ExternalDatasetSplit,
)


class DatasetIntegrityError(ValueError):
    """Raised when an evaluation dataset violates integrity or provenance invariants."""
    pass


@dataclass
class DatasetIntegrityReport:
    """Summary of dataset integrity checks."""
    is_valid: bool
    num_samples: int
    frame_length: int
    iq_channels: int
    dtype: str
    provenance: str
    unique_classes: List[int]
    unique_channels: List[int]
    unique_snrs: List[int]
    has_nans: bool
    has_infs: bool
    checks_passed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "num_samples": self.num_samples,
            "frame_length": self.frame_length,
            "iq_channels": self.iq_channels,
            "dtype": self.dtype,
            "provenance": self.provenance,
            "unique_classes": self.unique_classes,
            "unique_channels": self.unique_channels,
            "unique_snrs": self.unique_snrs,
            "has_nans": self.has_nans,
            "has_infs": self.has_infs,
            "checks_passed": self.checks_passed,
            "errors": self.errors,
        }


def compute_sample_fingerprints(X: np.ndarray, num_samples_to_hash: int = 100) -> Set[str]:
    """
    Computes cryptographic MD5 hashes for a subset of samples to verify split isolation.
    """
    hashes = set()
    step = max(1, len(X) // num_samples_to_hash)
    for i in range(0, len(X), step):
        sample_bytes = X[i].tobytes()
        hashes.add(hashlib.md5(sample_bytes).hexdigest())
    return hashes


def verify_dataset_integrity(
    split: ExternalDatasetSplit,
    expected_frame_length: int = 1024,
    expected_iq_channels: int = 2,
    allowed_classes: Optional[List[int]] = None,
    allowed_channels: Optional[List[int]] = None,
    allowed_snrs: Optional[List[int]] = None,
    reference_dev_split: Optional[ExternalDatasetSplit] = None,
    raise_on_error: bool = True,
) -> DatasetIntegrityReport:
    """
    Audits an evaluation split against strict M7 integrity criteria.
    
    Args:
        split: The ExternalDatasetSplit to audit.
        expected_frame_length: Expected temporal length of each frame (default 1024).
        expected_iq_channels: Expected number of channels (default 2 for I/Q).
        allowed_classes: Valid integer class indices (defaults to 0..6).
        allowed_channels: Valid channel conditions (defaults to [0, 1]).
        allowed_snrs: Valid SNR levels (defaults to [20, 22, 24, 26, 28, 30]).
        reference_dev_split: Optional dev split to check for accidental split contamination.
        raise_on_error: If True, raises DatasetIntegrityError on any violation.
        
    Returns:
        DatasetIntegrityReport summarizing the audit.
        
    Raises:
        DatasetIntegrityError: If raise_on_error=True and any check fails.
    """
    errors: List[str] = []
    checks_passed: List[str] = []

    if allowed_classes is None:
        allowed_classes = list(range(len(EXTERNAL_MODULATION_CLASSES)))
    if allowed_channels is None:
        allowed_channels = [0, 1]
    if allowed_snrs is None:
        allowed_snrs = list(EXTERNAL_SNR_VALUES)

    # 1. Type check
    if not isinstance(split.X, np.ndarray):
        errors.append(f"X must be a numpy ndarray, got {type(split.X)}")
    if not isinstance(split.y_mod, np.ndarray):
        errors.append(f"y_mod must be a numpy ndarray, got {type(split.y_mod)}")
    if not isinstance(split.y_chan, np.ndarray):
        errors.append(f"y_chan must be a numpy ndarray, got {type(split.y_chan)}")
    if not isinstance(split.y_snr, np.ndarray):
        errors.append(f"y_snr must be a numpy ndarray, got {type(split.y_snr)}")

    if errors:
        report = DatasetIntegrityReport(
            is_valid=False, num_samples=0, frame_length=0, iq_channels=0,
            dtype="unknown", provenance=getattr(split, "provenance", "unknown"),
            unique_classes=[], unique_channels=[], unique_snrs=[],
            has_nans=False, has_infs=False, checks_passed=[], errors=errors
        )
        if raise_on_error:
            raise DatasetIntegrityError(f"Dataset type validation failed: {'; '.join(errors)}")
        return report

    # 2. Length consistency check
    N = len(split.X)
    if N == 0:
        errors.append("Dataset is empty (N=0).")
    if len(split.y_mod) != N or len(split.y_chan) != N or len(split.y_snr) != N:
        errors.append(
            f"Length mismatch: len(X)={N}, len(y_mod)={len(split.y_mod)}, "
            f"len(y_chan)={len(split.y_chan)}, len(y_snr)={len(split.y_snr)}"
        )
    else:
        checks_passed.append("length_consistency")

    # 3. Shape and dimension checks: expect [N, 1024, 2] or [N, 2, 1024]
    frame_length = 0
    iq_channels = 0
    if split.X.ndim != 3:
        errors.append(f"Expected 3D array for X, got ndim={split.X.ndim} with shape {split.X.shape}")
    else:
        # Check canonical dimension arrangements
        if split.X.shape[1] == expected_frame_length and split.X.shape[2] == expected_iq_channels:
            frame_length = split.X.shape[1]
            iq_channels = split.X.shape[2]
            checks_passed.append("frame_dimensions_[N,1024,2]")
        elif split.X.shape[1] == expected_iq_channels and split.X.shape[2] == expected_frame_length:
            frame_length = split.X.shape[2]
            iq_channels = split.X.shape[1]
            checks_passed.append("frame_dimensions_[N,2,1024]")
        else:
            errors.append(
                f"Invalid dimensions: expected frame_length={expected_frame_length} and "
                f"iq_channels={expected_iq_channels}, got shape {split.X.shape}"
            )

    # 4. Floating point validity (no NaN or Inf)
    has_nans = bool(np.isnan(split.X).any())
    has_infs = bool(np.isinf(split.X).any())
    if has_nans:
        errors.append("X contains NaN (Not a Number) values.")
    if has_infs:
        errors.append("X contains infinite (Inf/-Inf) values.")
    if not has_nans and not has_infs:
        checks_passed.append("no_nans_or_infs")

    # 5. Label domain checks
    unique_mods = [int(v) for v in np.unique(split.y_mod)]
    invalid_mods = set(unique_mods) - set(allowed_classes)
    if invalid_mods:
        errors.append(f"Unexpected modulation classes found: {invalid_mods}. Allowed: {allowed_classes}")
    else:
        checks_passed.append("valid_modulation_classes")

    unique_chans = [int(v) for v in np.unique(split.y_chan)]
    invalid_chans = set(unique_chans) - set(allowed_channels)
    if invalid_chans:
        errors.append(f"Unexpected channel labels found: {invalid_chans}. Allowed: {allowed_channels}")
    else:
        checks_passed.append("valid_channel_conditions")

    unique_snrs = [int(v) for v in np.unique(split.y_snr)]
    invalid_snrs = set(unique_snrs) - set(allowed_snrs)
    if invalid_snrs:
        errors.append(f"Unexpected SNR values found: {invalid_snrs}. Allowed: {allowed_snrs}")
    else:
        checks_passed.append("valid_snr_levels")

    # 6. Provenance & Evaluation Isolation
    if not getattr(split, "is_evaluation_set", False):
        errors.append(f"Split {split.provenance} must have is_evaluation_set=True.")
    else:
        checks_passed.append("evaluation_isolation_flag")

    # 7. Contamination check between splits if dev reference is provided
    if reference_dev_split is not None and N > 0 and len(reference_dev_split.X) > 0:
        dev_hashes = compute_sample_fingerprints(reference_dev_split.X)
        test_hashes = compute_sample_fingerprints(split.X)
        overlap = dev_hashes.intersection(test_hashes)
        if overlap:
            errors.append(
                f"Split contamination detected! {len(overlap)} matching sample hashes found "
                f"between development and evaluation splits."
            )
        else:
            checks_passed.append("split_isolation_uncontaminated")

    is_valid = len(errors) == 0
    report = DatasetIntegrityReport(
        is_valid=is_valid,
        num_samples=N,
        frame_length=frame_length,
        iq_channels=iq_channels,
        dtype=str(split.X.dtype),
        provenance=str(getattr(split, "provenance", "UNKNOWN")),
        unique_classes=unique_mods,
        unique_channels=unique_chans,
        unique_snrs=unique_snrs,
        has_nans=has_nans,
        has_infs=has_infs,
        checks_passed=checks_passed,
        errors=errors,
    )

    if not is_valid and raise_on_error:
        raise DatasetIntegrityError(f"Dataset integrity verification failed with errors: {'; '.join(errors)}")

    return report
