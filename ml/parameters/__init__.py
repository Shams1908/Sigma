"""
ML Phase M8: ML-Assisted Parameter Estimation.

Provides deterministic DSP baselines, ML estimators with uncertainty quantification,
hybrid consensus mechanisms, and Hypothesis Engine adapters for:
  1. Symbol Rate Estimation (Baud rate)
  2. SNR Estimation (dB)
"""
from ml.parameters.dsp_estimators import (
    estimate_symbol_rate_dsp,
    estimate_snr_dsp,
    DSPSymbolRateResult,
    DSPSNRResult,
)
from ml.parameters.architecture import (
    ParameterEstimatorCNN,
    ParameterEstimatorHead,
    GaussianNLLLoss,
)
from ml.parameters.inference import (
    estimate_parameters,
    ParameterAnalysisResult,
    SingleParameterResult,
)

__all__ = [
    "estimate_symbol_rate_dsp",
    "estimate_snr_dsp",
    "DSPSymbolRateResult",
    "DSPSNRResult",
    "ParameterEstimatorCNN",
    "ParameterEstimatorHead",
    "GaussianNLLLoss",
    "estimate_parameters",
    "ParameterAnalysisResult",
    "SingleParameterResult",
]
