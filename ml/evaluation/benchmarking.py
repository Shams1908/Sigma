"""
M7-K: Inference Latency and Throughput Benchmarking.

Measures reproducible execution timings on CPU with warmups:
  1. Model-only forward pass (per-sample & batch)
  2. Preprocessing + Model inference
  3. Window segmentation + Aggregation pipeline
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from ml.representations.transforms import compute_representation, RepresentationType
from ml.inference.aggregation import aggregate_window_probabilities


def _calculate_timing_stats(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=float)
    mean_ms = float(np.mean(arr))
    throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "throughput_samples_per_sec": throughput,
    }


def benchmark_inference_latency(
    model: torch.nn.Module,
    rms_factor: float,
    in_channels: int = 2,
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Measures CPU latency across three isolated stages.
    
    Args:
        model: Loaded PyTorch model in eval mode.
        rms_factor: Training RMS scaling factor.
        in_channels: 2 for M5 (RAW_IQ) or 3 for M6 (IQ_AMPLITUDE).
        num_runs: Number of timed repetitions.
        warmup_runs: Number of untimed warmup repetitions.
        batch_size: Batch size for batched forward benchmarking.
        
    Returns:
        Dict with cleanly separated benchmarks.
    """
    model.eval()
    rng = np.random.default_rng(42)

    # Synthetic single sample: [2, 128]
    sample_raw = rng.standard_normal((2, 128)).astype(np.float32)
    # Synthetic batch of 128-sample windows: [batch_size, 2, 128]
    batch_raw = rng.standard_normal((batch_size, 2, 128)).astype(np.float32)
    # Synthetic 1024-sample frame (8 windows of 128)
    frame_raw = rng.standard_normal((2, 1024)).astype(np.float32)

    # Prepare preprocessed tensors for Stage 1 (Model-Only)
    if in_channels == 3:
        sample_tensor_in = torch.tensor(
            compute_representation(sample_raw[np.newaxis, :, :], RepresentationType.IQ_AMPLITUDE),
            dtype=torch.float32
        )
        batch_tensor_in = torch.tensor(
            compute_representation(batch_raw, RepresentationType.IQ_AMPLITUDE),
            dtype=torch.float32
        )
    else:
        sample_tensor_in = torch.tensor(sample_raw[np.newaxis, :, :], dtype=torch.float32)
        batch_tensor_in = torch.tensor(batch_raw, dtype=torch.float32)

    # -------------------------------------------------------------
    # Stage 1: Model-Only Forward Pass
    # -------------------------------------------------------------
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_tensor_in)
            _ = model(batch_tensor_in)

    # Single-sample model only
    latencies_model_single = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_tensor_in)
            t1 = time.perf_counter()
            latencies_model_single.append((t1 - t0) * 1000.0)

    # Batched model only (reported per sample)
    latencies_model_batch = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(batch_tensor_in)
            t1 = time.perf_counter()
            latencies_model_batch.append(((t1 - t0) * 1000.0) / batch_size)

    # -------------------------------------------------------------
    # Stage 2: Preprocessing + Model Inference
    # -------------------------------------------------------------
    latencies_prep_model = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            # 1. RMS normalize
            norm_sample = sample_raw / rms_factor
            # 2. Representation transform if required
            if in_channels == 3:
                x_in = compute_representation(norm_sample[np.newaxis, :, :], RepresentationType.IQ_AMPLITUDE)
            else:
                x_in = norm_sample[np.newaxis, :, :]
            # 3. Model forward
            t_in = torch.tensor(x_in, dtype=torch.float32)
            out = model(t_in)
            _ = torch.softmax(out, dim=-1).numpy()
            t1 = time.perf_counter()
            latencies_prep_model.append((t1 - t0) * 1000.0)

    # -------------------------------------------------------------
    # Stage 3: Window Segmentation + Batch Inference + Aggregation
    # -------------------------------------------------------------
    latencies_pipeline = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            # 1. Segment frame into 8 windows
            windows = []
            for w in range(8):
                windows.append(frame_raw[:, w * 128 : (w + 1) * 128])
            win_batch = np.stack(windows, axis=0) # [8, 2, 128]
            
            # 2. Normalize
            win_norm = win_batch / rms_factor
            if in_channels == 3:
                x_pipe = compute_representation(win_norm, RepresentationType.IQ_AMPLITUDE)
            else:
                x_pipe = win_norm
                
            # 3. Batch forward
            t_pipe = torch.tensor(x_pipe, dtype=torch.float32)
            out_pipe = model(t_pipe)
            probs_pipe = torch.softmax(out_pipe, dim=-1).numpy()
            
            # 4. Aggregation
            _ = aggregate_window_probabilities(probs_pipe)
            t1 = time.perf_counter()
            latencies_pipeline.append((t1 - t0) * 1000.0)

    return {
        "model_only_single_sample": _calculate_timing_stats(latencies_model_single),
        "model_only_batch_per_sample": _calculate_timing_stats(latencies_model_batch),
        "preprocessing_plus_inference": _calculate_timing_stats(latencies_prep_model),
        "full_frame_pipeline_8windows": _calculate_timing_stats(latencies_pipeline),
        "config": {
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "batch_size": batch_size,
            "in_channels": in_channels,
            "device": "cpu",
        }
    }
