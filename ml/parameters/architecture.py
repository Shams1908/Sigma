"""
M8: Parameter Estimator CNN Architecture with Uncertainty Quantification.

Multi-head 1D CNN with heteroscedastic Gaussian Negative Log-Likelihood heads
for simultaneous parameter estimation and uncertainty quantification:
  Head 1: Symbol Rate (predicted as samples-per-symbol SPS and its uncertainty)
  Head 2: SNR (predicted in dB and its uncertainty)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any, Optional


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for heteroscedastic regression:
      loss = 0.5 * exp(-log_var) * (y - mu)^2 + 0.5 * log_var
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_mu: torch.Tensor, pred_log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp log_var for numerical stability
        log_var_clamped = torch.clamp(pred_log_var, min=-6.0, max=6.0)
        inv_var = torch.exp(-log_var_clamped)
        loss = 0.5 * (inv_var * ((target - pred_mu) ** 2) + log_var_clamped)
        return torch.mean(loss)


class ParameterEstimatorHead(nn.Module):
    """
    Regression head outputting mean (mu) and log-variance (log_var) for uncertainty.
    """
    def __init__(self, in_features: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2) # [mu, log_var]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mu = out[:, 0]
        log_var = out[:, 1]
        return mu, log_var


class ParameterEstimatorCNN(nn.Module):
    """
    1D CNN Backbone with dual parameter estimation heads.
    Input: [batch, in_channels, length] (default in_channels=2, length=128)
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.in_channels = in_channels

        # Conv Backbone
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 128 -> 64
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 64 -> 32
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Multi-Heads
        self.sps_head = ParameterEstimatorHead(in_features=256, hidden_dim=128)
        self.snr_head = ParameterEstimatorHead(in_features=256, hidden_dim=128)

    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            {
                "sps": (mu_sps, log_var_sps),
                "snr": (mu_snr, log_var_snr),
            }
        """
        feats = self.block1(x)
        feats = self.block2(feats)
        feats = self.block3(feats)

        # Global Average Pooling
        pooled = feats.mean(dim=-1) # [batch, 256]

        mu_sps, log_var_sps = self.sps_head(pooled)
        mu_snr, log_var_snr = self.snr_head(pooled)

        return {
            "sps": (mu_sps, log_var_sps),
            "snr": (mu_snr, log_var_snr),
        }
