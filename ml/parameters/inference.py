"""
M8: Parameter Estimation Inference API.

Provides unified interface estimate_parameters(...) combining DSP baselines,
ML neural estimators, and hybrid consensus.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import os
import numpy as np
import torch

from ml.parameters.dsp_estimators import (
    estimate_symbol_rate_dsp,
    estimate_snr_dsp,
    DSPSymbolRateResult,
    DSPSNRResult,
    _ensure_complex_1d,
)
from ml.parameters.architecture import ParameterEstimatorCNN

_M8_MODEL_CACHE: Dict[str, ParameterEstimatorCNN] = {}


@dataclass(frozen=True)
class SingleParameterResult:
    """Estimated value, confidence, and uncertainty for a single RF parameter."""
    estimate: float           # Final recommended estimate
    confidence: float         # 0.0 to 1.0 (calibrated confidence)
    uncertainty: float        # Standard deviation / margin of error
    method: str               # "hybrid_consensus", "ml_assisted", or "dsp_baseline"
    dsp_estimate: Optional[float] = None
    dsp_confidence: Optional[float] = None
    ml_estimate: Optional[float] = None
    ml_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": float(self.estimate),
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "method": self.method,
            "dsp_estimate": float(self.dsp_estimate) if self.dsp_estimate is not None else None,
            "dsp_confidence": float(self.dsp_confidence) if self.dsp_confidence is not None else None,
            "ml_estimate": float(self.ml_estimate) if self.ml_estimate is not None else None,
            "ml_confidence": float(self.ml_confidence) if self.ml_confidence is not None else None,
        }


@dataclass(frozen=True)
class ParameterAnalysisResult:
    """Structured container for all estimated signal parameters."""
    symbol_rate: SingleParameterResult
    snr_db: SingleParameterResult
    sample_rate: float
    signal_length_samples: int
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_rate": self.symbol_rate.to_dict(),
            "snr_db": self.snr_db.to_dict(),
            "sample_rate": float(self.sample_rate),
            "signal_length_samples": int(self.signal_length_samples),
            "duration_ms": float(self.duration_ms),
        }


def get_parameter_model(model_path: str = "models/m8_parameter_estimator.pt") -> Optional[ParameterEstimatorCNN]:
    """Loads and caches the ParameterEstimatorCNN model."""
    if model_path in _M8_MODEL_CACHE:
        return _M8_MODEL_CACHE[model_path]
    if not os.path.exists(model_path):
        return None

    model = ParameterEstimatorCNN(in_channels=2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _M8_MODEL_CACHE[model_path] = model
    return model


def estimate_parameters(
    signal: np.ndarray,
    sample_rate: float = 800000.0,
    model_path: str = "models/m8_parameter_estimator.pt",
) -> ParameterAnalysisResult:
    """
    Estimates symbol rate and SNR using hybrid DSP + ML architecture.
    
    Flow:
      1. Deterministic DSP extraction (Delay-and-multiply spectral line for Baud, M2M4 for SNR).
      2. ML neural estimation with uncertainty quantification.
      3. Consensus fusion: DSP acts as validator, ML proposes candidate interpretations.
      
    Args:
        signal: 1D complex array or 2D [2, N] float array.
        sample_rate: Sampling frequency in Hz (default 800 kHz).
        model_path: Path to serialized PyTorch ParameterEstimatorCNN.
        
    Returns:
        ParameterAnalysisResult dataclass.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    z = _ensure_complex_1d(signal)
    N = len(z)
    if N < 64:
        raise ValueError(f"Signal must contain at least 64 samples, got {N}")

    # Prepare standard [2, N] float32 array
    iq_2d = np.stack([np.real(z), np.imag(z)], axis=0).astype(np.float32)

    # -------------------------------------------------------------
    # 1. Deterministic DSP Baselines
    # -------------------------------------------------------------
    dsp_sr = estimate_symbol_rate_dsp(iq_2d, sample_rate)
    dsp_snr = estimate_snr_dsp(iq_2d)

    # -------------------------------------------------------------
    # 2. ML Neural Estimator
    # -------------------------------------------------------------
    model = get_parameter_model(model_path)
    ml_sr_val = None
    ml_sr_conf = None
    ml_sr_unc = None
    ml_snr_val = None
    ml_snr_conf = None
    ml_snr_unc = None

    if model is not None:
        # Segment into 128-sample windows (up to 32 windows for efficiency)
        num_windows = min(32, N // 128)
        if num_windows > 0:
            windows = []
            for w in range(num_windows):
                windows.append(iq_2d[:, w * 128 : (w + 1) * 128])
            win_tensor = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)

            with torch.no_grad():
                outs = model(win_tensor)
                mu_sps, log_var_sps = outs["sps"]
                mu_snr, log_var_snr = outs["snr"]

                # Mean predicted SPS and variance across windows
                pred_sps = float(torch.mean(mu_sps).item())
                sps_var = float(torch.mean(torch.exp(log_var_sps)).item())
                pred_sps_clamped = max(2.0, pred_sps)

                ml_sr_val = float(sample_rate / pred_sps_clamped)
                sps_std = np.sqrt(max(1e-6, sps_var))
                ml_sr_unc = float(ml_sr_val * (sps_std / pred_sps_clamped))
                ml_sr_conf = float(np.clip(np.exp(-sps_std / pred_sps_clamped), 0.10, 0.95))

                ml_snr_val = float(torch.mean(mu_snr).item())
                snr_std = float(torch.mean(torch.sqrt(torch.exp(log_var_snr))).item())
                ml_snr_unc = snr_std
                ml_snr_conf = float(np.clip(np.exp(-snr_std / 8.0), 0.10, 0.95))

    # -------------------------------------------------------------
    # 3. Hybrid Fusion Logic
    # -------------------------------------------------------------
    # Symbol Rate Fusion
    if ml_sr_val is not None and ml_sr_conf is not None:
        # Check agreement between DSP and ML
        rel_diff = abs(dsp_sr.estimate - ml_sr_val) / max(dsp_sr.estimate, ml_sr_val, 1.0)
        
        if dsp_sr.confidence >= 0.70 and rel_diff <= 0.15:
            # Strong agreement: hybrid consensus
            w_dsp = dsp_sr.confidence / max(1e-3, dsp_sr.uncertainty)
            w_ml = ml_sr_conf / max(1e-3, ml_sr_unc)
            fused_sr = float((w_dsp * dsp_sr.estimate + w_ml * ml_sr_val) / (w_dsp + w_ml))
            fused_conf = float(min(0.98, max(dsp_sr.confidence, ml_sr_conf) + 0.05))
            fused_unc = float(min(dsp_sr.uncertainty, ml_sr_unc))
            method_sr = "hybrid_consensus"
        elif dsp_sr.confidence < 0.45 and ml_sr_conf >= 0.50:
            # DSP is weak (low SNR or multipath): ML assists
            fused_sr = ml_sr_val
            fused_conf = ml_sr_conf
            fused_unc = ml_sr_unc
            method_sr = "ml_assisted"
        else:
            # Default to DSP baseline
            fused_sr = dsp_sr.estimate
            fused_conf = dsp_sr.confidence
            fused_unc = dsp_sr.uncertainty
            method_sr = "dsp_baseline"
    else:
        fused_sr = dsp_sr.estimate
        fused_conf = dsp_sr.confidence
        fused_unc = dsp_sr.uncertainty
        method_sr = "dsp_baseline"

    sr_result = SingleParameterResult(
        estimate=fused_sr,
        confidence=fused_conf,
        uncertainty=fused_unc,
        method=method_sr,
        dsp_estimate=dsp_sr.estimate,
        dsp_confidence=dsp_sr.confidence,
        ml_estimate=ml_sr_val,
        ml_confidence=ml_sr_conf,
    )

    # SNR Fusion
    if ml_snr_val is not None and ml_snr_conf is not None:
        diff_snr = abs(dsp_snr.estimate - ml_snr_val)
        if dsp_snr.confidence >= 0.60 and diff_snr <= 4.0:
            fused_snr = float(0.5 * dsp_snr.estimate + 0.5 * ml_snr_val)
            fused_snr_conf = float(min(0.98, max(dsp_snr.confidence, ml_snr_conf) + 0.05))
            fused_snr_unc = float(min(dsp_snr.uncertainty, ml_snr_unc))
            method_snr = "hybrid_consensus"
        elif dsp_snr.confidence < 0.35:
            fused_snr = ml_snr_val
            fused_snr_conf = ml_snr_conf
            fused_snr_unc = ml_snr_unc
            method_snr = "ml_assisted"
        else:
            fused_snr = dsp_snr.estimate
            fused_snr_conf = dsp_snr.confidence
            fused_snr_unc = dsp_snr.uncertainty
            method_snr = "dsp_baseline"
    else:
        fused_snr = dsp_snr.estimate
        fused_snr_conf = dsp_snr.confidence
        fused_snr_unc = dsp_snr.uncertainty
        method_snr = "dsp_baseline"

    snr_result = SingleParameterResult(
        estimate=fused_snr,
        confidence=fused_snr_conf,
        uncertainty=fused_snr_unc,
        method=method_snr,
        dsp_estimate=dsp_snr.estimate,
        dsp_confidence=dsp_snr.confidence,
        ml_estimate=ml_snr_val,
        ml_confidence=ml_snr_conf,
    )

    return ParameterAnalysisResult(
        symbol_rate=sr_result,
        snr_db=snr_result,
        sample_rate=float(sample_rate),
        signal_length_samples=N,
        duration_ms=float((N / sample_rate) * 1000.0),
    )
