"""
CNN inference utilities for the SIGMA ML pipeline.

Backward compatibility:
    Existing 2-channel M5 checkpoints load and run unchanged.
    Checkpoints that carry ``in_channels`` or ``representation`` metadata
    (M6+) cause the model to be instantiated with the correct channel count
    and the segments to be transformed by the declared representation.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ml.dataset.labels import INDEX_TO_MODULATION, MODULATION_CLASSES
from ml.cnn_model.architecture import RawIQCNN

# In-memory model cache to avoid repeated file loading
_MODEL_CACHE: Dict[str, Any] = {}


def get_cnn_model(
    model_path: str = "models/m5_iq_cnn.pt",
) -> Tuple[RawIQCNN, float, Optional[str]]:
    """
    Load and cache the CNN model and its checkpoint metadata.

    The cache is keyed by ``model_path`` so that different checkpoints are
    cached and returned independently.

    Returns:
        (model, rms_factor, representation_name)

        model:               RawIQCNN in eval mode.
        rms_factor:          Training RMS normalization factor.
        representation_name: Representation string from checkpoint metadata,
                             or None for legacy 2-channel checkpoints
                             (treated as RAW_IQ).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if model_path in _MODEL_CACHE:
        entry = _MODEL_CACHE[model_path]
        return entry["model"], entry["rms_factor"], entry.get("representation")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained CNN model checkpoint not found at '{model_path}'. "
            "Please run the training pipeline script `ml/cnn_model/train.py` first."
        )

    print(f"Loading cached PyTorch CNN model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # ── Determine channel count / representation from checkpoint ─────────────
    # Legacy M5 checkpoints have no 'in_channels' key → default to 2 (RAW_IQ).
    in_channels: int = int(checkpoint.get("in_channels", 2))
    representation: Optional[str] = checkpoint.get("representation", None)

    model = RawIQCNN(num_classes=11, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rms_factor = float(checkpoint["rms_factor"])

    _MODEL_CACHE[model_path] = {
        "model": model,
        "rms_factor": rms_factor,
        "representation": representation,
        "in_channels": in_channels,
    }

    return model, rms_factor, representation


def predict_iq(
    iq: np.ndarray,
    model_path: str = "models/m5_iq_cnn.pt",
) -> Dict[str, Any]:
    """
    Run prediction using the cached CNN model.

    Supports both single sample (shape [2, 128]) and batch inputs
    (shape [N, 2, 128]).

    The representation is selected automatically from checkpoint metadata:
        - Legacy 2-channel checkpoint → RAW_IQ (no transform).
        - 3-channel checkpoint with representation="IQ_AMPLITUDE" → applies
          the IQ_AMPLITUDE transform before inference.
        - Other M6 representations are handled via the representation module.

    Args:
        iq:         NumPy array of shape [2, 128] or [N, 2, 128] (raw I/Q).
        model_path: Path to the serialized model checkpoint.

    Returns:
        Dict with keys:
            class_index, class_name, probabilities, confidence
    """
    model, rms_factor, representation = get_cnn_model(model_path)

    if not isinstance(iq, np.ndarray):
        raise TypeError(
            f"Input features must be a numpy ndarray, got {type(iq)}"
        )

    if np.isnan(iq).any() or np.isinf(iq).any():
        raise ValueError(
            "Input features contain NaNs or infinite values."
        )

    is_batch = iq.ndim == 3

    if iq.ndim == 2:
        if iq.shape != (2, 128):
            raise ValueError(
                f"Expected shape (2, 128) for single IQ sample, got {iq.shape}"
            )
        X_raw = np.expand_dims(iq, axis=0)  # [1, 2, 128]
    elif iq.ndim == 3:
        if iq.shape[1] != 2 or iq.shape[2] != 128:
            raise ValueError(
                f"Expected shape [N, 2, 128] for batch, got {iq.shape}"
            )
        X_raw = iq
    else:
        raise ValueError(
            f"Input IQ must have ndim 2 or 3, got shape {iq.shape}"
        )

    # ── Apply representation transform if required by the checkpoint ─────────
    X = _apply_checkpoint_representation(X_raw, representation)

    # ── Normalize and infer ───────────────────────────────────────────────────
    X_normalized = X / rms_factor
    x_tensor = torch.tensor(X_normalized, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()  # [N, 11]

    class_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    class_names = np.array(
        [INDEX_TO_MODULATION[idx] for idx in class_indices], dtype=object
    )

    if is_batch:
        return {
            "class_index": class_indices.astype(np.int32),
            "class_name": class_names,
            "probabilities": probs.astype(np.float32),
            "confidence": confidences.astype(np.float32),
        }
    else:
        return {
            "class_index": int(class_indices[0]),
            "class_name": str(class_names[0]),
            "probabilities": probs[0].astype(np.float32),
            "confidence": float(confidences[0]),
        }


def _apply_checkpoint_representation(
    X_raw: np.ndarray,
    representation: Optional[str],
) -> np.ndarray:
    """
    Transform a batch of raw [B, 2, N] IQ segments into the representation
    declared by the checkpoint.

    Args:
        X_raw:          [B, 2, N] float32 raw IQ batch.
        representation: Representation name from checkpoint, or None for
                        legacy 2-channel RAW_IQ (no transform).

    Returns:
        [B, C, N] float32 array in the declared representation.
    """
    if representation is None or representation == "RAW_IQ":
        # Legacy path — no transform
        return X_raw.astype(np.float32)

    from ml.representations import apply_representation, Representation  # noqa: PLC0415

    try:
        rep_enum = Representation(representation)
    except ValueError:
        # Unknown representation in checkpoint → fall back to RAW_IQ and warn
        import warnings  # noqa: PLC0415
        warnings.warn(
            f"Unknown representation '{representation}' in checkpoint; "
            "falling back to RAW_IQ.",
            RuntimeWarning,
            stacklevel=4,
        )
        return X_raw.astype(np.float32)

    return apply_representation(X_raw, rep_enum)


def clear_cnn_cache() -> None:
    """Clear the in-memory CNN model cache."""
    _MODEL_CACHE.clear()
