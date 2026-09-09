"""
Unit tests for M8 Parameter Estimator CNN and Loss (ml/parameters/architecture.py).
"""
import pytest
import torch

from ml.parameters.architecture import (
    ParameterEstimatorCNN,
    ParameterEstimatorHead,
    GaussianNLLLoss,
)


def test_parameter_estimator_forward():
    """Verifies forward pass through CNN backbone and dual heads."""
    model = ParameterEstimatorCNN(in_channels=2)
    x = torch.randn(8, 2, 128) # Batch of 8
    outs = model(x)

    assert "sps" in outs
    assert "snr" in outs

    mu_sps, log_var_sps = outs["sps"]
    mu_snr, log_var_snr = outs["snr"]

    assert mu_sps.shape == (8,)
    assert log_var_sps.shape == (8,)
    assert mu_snr.shape == (8,)
    assert log_var_snr.shape == (8,)


def test_gaussian_nll_loss():
    """Verifies Gaussian negative log-likelihood calculation."""
    criterion = GaussianNLLLoss()
    pred_mu = torch.tensor([10.0, 20.0])
    pred_log_var = torch.tensor([0.0, 0.0]) # var = 1.0
    target = torch.tensor([10.0, 20.0]) # exact match

    loss_zero_err = criterion(pred_mu, pred_log_var, target)
    # When error=0, loss = 0.5 * log_var = 0
    assert loss_zero_err.item() == pytest.approx(0.0)

    # When error increases, loss increases
    target_noisy = torch.tensor([12.0, 22.0])
    loss_with_err = criterion(pred_mu, pred_log_var, target_noisy)
    assert loss_with_err.item() > loss_zero_err.item()
