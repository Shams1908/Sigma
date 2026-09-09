"""
M7-H, I: Confidence Reliability and Temperature Scaling Calibration.

Extracts confidence statistics for correct vs incorrect predictions, evaluates
overconfidence, and provides conditional Temperature Scaling fitted strictly on
the development subset.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.evaluation.metrics import calculate_ece


@dataclass
class ConfidenceDistribution:
    """Summary statistics for a confidence distribution."""
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    q90: float
    min: float
    max: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "q25": self.q25,
            "q75": self.q75,
            "q90": self.q90,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


def _compute_distribution(values: List[float]) -> Optional[ConfidenceDistribution]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return ConfidenceDistribution(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        q25=float(np.percentile(arr, 25)),
        q75=float(np.percentile(arr, 75)),
        q90=float(np.percentile(arr, 90)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        count=len(arr),
    )


def analyze_confidence_reliability(
    confidences: List[float],
    accuracies: List[bool],
    high_conf_threshold: float = 0.80,
) -> Dict[str, Any]:
    """
    Analyzes model confidence behavior: correct vs incorrect distributions,
    overconfidence rates, and uncalibrated ECE.
    """
    correct_confs = [c for c, a in zip(confidences, accuracies) if a]
    incorrect_confs = [c for c, a in zip(confidences, accuracies) if not a]

    correct_dist = _compute_distribution(correct_confs)
    incorrect_dist = _compute_distribution(incorrect_confs)

    total_incorrect = len(incorrect_confs)
    high_conf_errors = sum(1 for c in incorrect_confs if c >= high_conf_threshold)
    overconfidence_rate = float(high_conf_errors / total_incorrect) if total_incorrect > 0 else 0.0

    total_correct = len(correct_confs)
    high_conf_correct = sum(1 for c in correct_confs if c >= high_conf_threshold)
    high_conf_correct_rate = float(high_conf_correct / total_correct) if total_correct > 0 else 0.0

    ece_res = calculate_ece(confidences, accuracies, num_bins=10)

    return {
        "is_calibrated": False,
        "calibration_status": "Raw softmax output is not calibrated unless temperature scaling is applied.",
        "raw_ece": ece_res["ece"],
        "num_samples": len(confidences),
        "correct_distribution": correct_dist.to_dict() if correct_dist else None,
        "incorrect_distribution": incorrect_dist.to_dict() if incorrect_dist else None,
        "high_confidence_threshold": high_conf_threshold,
        "high_confidence_correct_count": high_conf_correct,
        "high_confidence_correct_rate": high_conf_correct_rate,
        "high_confidence_error_count": high_conf_errors,
        "overconfidence_rate": overconfidence_rate,
        "ece_details": ece_res,
    }


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibrator.
    
    Fits a single learned scalar T > 0 strictly on logits from the development subset.
    T is then frozen and applied to test logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.is_fitted = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by 1/T."""
        temp = self.temperature.clamp(min=0.01)
        return logits / temp

    def fit_on_dev(
        self,
        dev_logits: np.ndarray,
        dev_targets: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fits optimal temperature on the development subset ONLY using NLL loss.
        """
        if len(dev_logits) == 0:
            self.is_fitted = True
            return float(self.temperature.item())

        logits_tensor = torch.tensor(dev_logits, dtype=torch.float32)
        targets_tensor = torch.tensor(dev_targets, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_tensor)
            loss = criterion(scaled_logits, targets_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.is_fitted = True
        return float(self.temperature.detach().item())

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """
        Converts uncalibrated logits into calibrated probabilities.
        """
        self.eval()
        with torch.no_grad():
            t_logits = torch.tensor(logits, dtype=torch.float32)
            scaled = self.forward(t_logits)
            probs = torch.softmax(scaled, dim=-1).numpy()
        return probs
