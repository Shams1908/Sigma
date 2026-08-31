import os
import time
import numpy as np
import matplotlib.pyplot as plt

from ml.synthetic.generator import generate_synthetic_evaluation_dataset
from ml.synthetic.dataset import load_synthetic_dataset
from ml.generators.config import GeneratorConfig
from ml.generators import generate_bits, generate_modulated_signal

def run_diagnostic():
    print("==================================================")
    print("M6.1 SYNTHETIC EVALUATION DATASET DIAGNOSTIC")
    print("==================================================")
    
    # 1. Generate full evaluation dataset (default 100 examples per combination)
    t0 = time.time()
    X, y, metadata = generate_synthetic_evaluation_dataset(num_examples_per_cond=100)
    gen_time = time.time() - t0
    
    # 2. Print Dataset Statistics
    print("\nDataset Statistics:")
    print(f"  Total generated examples: {X.shape[0]}")
    print(f"  IQ Shape:                 {X.shape[1:]}")
    print(f"  Samples dtype:            {X.dtype}")
    print(f"  Labels shape/dtype:       {y.shape} / {y.dtype}")
    print(f"  Generation Time:          {gen_time:.2f} seconds")
    
    # Check finite values
    is_finite = np.isfinite(X).all()
    print(f"  Finite-value status:      {'PASSED' if is_finite else 'FAILED'}")
    
    # Count per modulation
    mods = [m["modulation"] for m in metadata]
    unique_mods, counts_mods = np.unique(mods, return_counts=True)
    print("\nModulation Class distribution:")
    for m_name, cnt in zip(unique_mods, counts_mods):
        print(f"  {m_name:<10}: {cnt} examples")
        
    # Count per experiment
    exps = [m["experiment"] for m in metadata]
    unique_exps, counts_exps = np.unique(exps, return_counts=True)
    print("\nExperiment distribution:")
    for e_name, cnt in zip(unique_exps, counts_exps):
        print(f"  {e_name:<25}: {cnt} examples")
        
    # 3. Seed Reproducibility Check
    print("\nVerifying seed reproducibility...")
    # Generate BPSK with timing offset 0.3 sample, seed 999
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        filter_span_symbols=8,
        rolloff=0.35,
        random_seed=999,
        snr=18.0,
        timing_offset=0.3
    )
    
    bits = generate_bits(8, seed_or_generator=999)
    
    rng1 = np.random.default_rng(999)
    sig1 = generate_modulated_signal(bits, config, rng=rng1)
    
    rng2 = np.random.default_rng(999)
    sig2 = generate_modulated_signal(bits, config, rng=rng2)
    
    reproducible = np.array_equal(sig1.samples, sig2.samples)
    print(f"  Seed reproducibility status: {'PASSED' if reproducible else 'FAILED'}")
    assert reproducible, "RNG reproduction failed: waveforms are not identical!"

    # 4. Generate Signal Visualizations
    print("\nGenerating representative signal plots...")
    os.makedirs("results/ml/m6", exist_ok=True)
    
    # Find a clean BPSK signal and an impaired BPSK signal
    bpsk_clean_idx = next(i for i, m in enumerate(metadata) if m["modulation"] == "BPSK" and m["experiment"] == "awgn" and m["sweep_value"] == 18.0)
    bpsk_noisy_idx = next(i for i, m in enumerate(metadata) if m["modulation"] == "BPSK" and m["experiment"] == "awgn" and m["sweep_value"] == -10.0)
    bpsk_freq_idx = next(i for i, m in enumerate(metadata) if m["modulation"] == "BPSK" and m["experiment"] == "frequency_offset" and m["sweep_value"] == 10000.0)
    
    plt.figure(figsize=(15, 10))
    
    # Clean BPSK
    plt.subplot(3, 1, 1)
    plt.plot(X[bpsk_clean_idx, 0], 'b-', label='I Channel', linewidth=1.5)
    plt.plot(X[bpsk_clean_idx, 1], 'r-', label='Q Channel', linewidth=1.5)
    plt.title("Representative BPSK - Clean (SNR = 18 dB)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    # Noisy BPSK
    plt.subplot(3, 1, 2)
    plt.plot(X[bpsk_noisy_idx, 0], 'b-', label='I Channel', linewidth=1.5)
    plt.plot(X[bpsk_noisy_idx, 1], 'r-', label='Q Channel', linewidth=1.5)
    plt.title("Representative BPSK - Noisy (SNR = -10 dB)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    # Frequency offset BPSK
    plt.subplot(3, 1, 3)
    plt.plot(X[bpsk_freq_idx, 0], 'b-', label='I Channel', linewidth=1.5)
    plt.plot(X[bpsk_freq_idx, 1], 'r-', label='Q Channel', linewidth=1.5)
    plt.title("Representative BPSK - Frequency Offset (10 kHz Offset, SNR = 18 dB)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "results/ml/m6/representative_signals.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Representative plots saved to: {plot_path}")
    print("==================================================")

if __name__ == "__main__":
    run_diagnostic()
