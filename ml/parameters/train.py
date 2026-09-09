"""
M8: Training and Evaluation Pipeline for ML-Assisted Parameter Estimators.

Trains ParameterEstimatorCNN on synthetic waveforms with randomized symbol rates,
samples-per-symbol, and SNRs. Evaluates tolerance accuracy (±1%, ±5%, ±10%) and saves
serialized checkpoint models/m8_parameter_estimator.pt.
"""
import os
import time
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ml.parameters.architecture import ParameterEstimatorCNN, GaussianNLLLoss
from ml.parameters.dataset import generate_parameter_dataset, ParameterDatasetSplit


def evaluate_parameter_estimator(
    model: ParameterEstimatorCNN,
    val_split: ParameterDatasetSplit,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Evaluates parameter estimation performance against ground truth.
    Computes MAE, RMSE, and tolerance accuracy for symbol rate, and MAE/RMSE for SNR.
    """
    model.eval()
    X_val = torch.tensor(val_split.X, dtype=torch.float32)
    sps_true = val_split.sps
    sym_rate_true = val_split.symbol_rate
    snr_true = val_split.snr_db
    sample_rate = val_split.sample_rate

    pred_sps_list = []
    pred_snr_list = []
    unc_sps_list = []
    unc_snr_list = []

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch = X_val[i : i + batch_size]
            outs = model(batch)
            mu_sps, log_var_sps = outs["sps"]
            mu_snr, log_var_snr = outs["snr"]

            pred_sps_list.append(mu_sps.numpy())
            pred_snr_list.append(mu_snr.numpy())
            unc_sps_list.append(np.sqrt(np.exp(log_var_sps.numpy())))
            unc_snr_list.append(np.sqrt(np.exp(log_var_snr.numpy())))

    pred_sps = np.concatenate(pred_sps_list)
    pred_snr = np.concatenate(pred_snr_list)

    # Derive predicted symbol rate from predicted sps: sym_rate = sample_rate / sps
    # Clamp predicted sps to avoid division by zero
    pred_sps_clamped = np.clip(pred_sps, 2.0, 100.0)
    pred_sym_rate = sample_rate / pred_sps_clamped

    # 1. Symbol Rate Metrics
    sym_rate_err = np.abs(pred_sym_rate - sym_rate_true)
    sym_rate_rel_err = sym_rate_err / sym_rate_true

    tol_1pct = float(np.mean(sym_rate_rel_err <= 0.01))
    tol_5pct = float(np.mean(sym_rate_rel_err <= 0.05))
    tol_10pct = float(np.mean(sym_rate_rel_err <= 0.10))

    sym_rate_mae = float(np.mean(sym_rate_err))
    sym_rate_rmse = float(np.sqrt(np.mean(sym_rate_err ** 2)))
    sym_rate_median_ae = float(np.median(sym_rate_err))

    # 2. SNR Metrics
    snr_err = np.abs(pred_snr - snr_true)
    snr_mae = float(np.mean(snr_err))
    snr_rmse = float(np.sqrt(np.mean(snr_err ** 2)))
    snr_median_ae = float(np.median(snr_err))

    return {
        "symbol_rate": {
            "mae_baud": sym_rate_mae,
            "rmse_baud": sym_rate_rmse,
            "median_absolute_error_baud": sym_rate_median_ae,
            "relative_error_mean": float(np.mean(sym_rate_rel_err)),
            "tolerance_1pct_accuracy": tol_1pct,
            "tolerance_5pct_accuracy": tol_5pct,
            "tolerance_10pct_accuracy": tol_10pct,
        },
        "snr": {
            "mae_db": snr_mae,
            "rmse_db": snr_rmse,
            "median_absolute_error_db": snr_median_ae,
        },
        "num_val_samples": len(val_split.X),
    }


def train_parameter_estimator(
    num_train: int = 2500,
    num_val: int = 500,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_seed: int = 42,
    checkpoint_path: str = "models/m8_parameter_estimator.pt",
    metadata_path: str = "models/m8_parameter_estimator_metadata.json",
) -> Dict[str, Any]:
    """
    Trains and persists the M8 parameter estimator model.
    """
    print(f"Generating synthetic training ({num_train}) and validation ({num_val}) sets...")
    train_split = generate_parameter_dataset(
        num_samples=num_train, random_seed=random_seed, provenance="M8-TRAIN"
    )
    val_split = generate_parameter_dataset(
        num_samples=num_val, random_seed=random_seed + 100, provenance="M8-VAL", is_evaluation_set=True
    )

    torch.manual_seed(random_seed)
    model = ParameterEstimatorCNN(in_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_nll = GaussianNLLLoss()

    train_ds = TensorDataset(
        torch.tensor(train_split.X, dtype=torch.float32),
        torch.tensor(train_split.sps, dtype=torch.float32),
        torch.tensor(train_split.snr_db, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print("Training Parameter Estimator CNN...")
    start_time = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for bx, b_sps, b_snr in train_loader:
            optimizer.zero_grad()
            outs = model(bx)
            mu_sps, log_var_sps = outs["sps"]
            mu_snr, log_var_snr = outs["snr"]

            loss_sps = criterion_nll(mu_sps, log_var_sps, b_sps)
            loss_snr = criterion_nll(mu_snr, log_var_snr, b_snr)

            # Joint multi-task loss (SNR scaled by 0.1 for balanced gradient magnitude)
            loss = loss_sps + 0.1 * loss_snr
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        epoch_loss = total_loss / len(train_split.X)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    train_time = time.perf_counter() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    print("Evaluating on validation split...")
    metrics = evaluate_parameter_estimator(model, val_split)
    sr_m = metrics["symbol_rate"]
    snr_m = metrics["snr"]
    print(f"  Symbol Rate: MAE={sr_m['mae_baud']:.1f} Baud, ±5% Tol Acc={sr_m['tolerance_5pct_accuracy']*100:.1f}%, ±10% Tol Acc={sr_m['tolerance_10pct_accuracy']*100:.1f}%")
    print(f"  SNR: MAE={snr_m['mae_db']:.2f} dB, RMSE={snr_m['rmse_db']:.2f} dB")

    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "in_channels": 2,
        "parameters": ["symbol_rate", "snr"],
        "metrics": metrics,
        "training_time_s": train_time,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "random_seed": random_seed,
        },
    }, checkpoint_path)
    print(f"Saved M8 model checkpoint to: {checkpoint_path}")

    # Save metadata
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": checkpoint_path,
        "in_channels": 2,
        "parameters_estimated": ["symbol_rate", "snr"],
        "metrics": metrics,
        "training_time_s": train_time,
    }
    with open(metadata_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved M8 metadata to: {metadata_path}")

    return summary


if __name__ == "__main__":
    train_parameter_estimator()
