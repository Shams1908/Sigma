import os
import time
import numpy as np
from ml.dataset.loader import RadioMLDataset
from ml.features.extractor import extract_feature_matrix, get_feature_names
from ml.features.schema import NUM_FEATURES

def inspect_and_save_features():
    real_dataset_path = "datasets/raw/RML2016.10a_dict.pkl"
    if not os.path.exists(real_dataset_path):
        print(f"Error: Real dataset pickle not found at {real_dataset_path}.")
        return

    print("==================================================")
    print("M3 FEATURE ENGINEERING DIAGNOSTIC & VALIDATION")
    print("==================================================")

    # 1. Load Dataset Cache
    print("Loading real RadioML dataset into cache...")
    start_load = time.time()
    dataset = RadioMLDataset(pickle_path=real_dataset_path)
    load_time = time.time() - start_load
    print(f"Dataset loaded in {load_time:.2f} seconds. Size: {len(dataset)} examples.")

    # 2. Run Extraction on all 220,000 samples
    print("\nRunning feature extraction on all 220,000 samples...")
    start_extract = time.time()
    # Using chunk_size=50000 for memory safety
    X, y, snrs = extract_feature_matrix(dataset, chunk_size=50000)
    extract_time = time.time() - start_extract
    print(f"Feature extraction completed in {extract_time:.2f} seconds.")
    print(f"Processing rate: {len(dataset) / extract_time:.1f} samples/second.")

    # 3. Shape and Numeric Robustness Check
    print("\nVerifying output matrix and metadata...")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"SNRs shape: {snrs.shape}")

    # Check for NaNs and Infs
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN count in feature matrix: {nan_count}")
    print(f"Inf count in feature matrix: {inf_count}")

    # Confirm requirements
    assert X.shape == (220000, NUM_FEATURES), f"Shape mismatch: {X.shape}"
    assert nan_count == 0, "Error: NaN values found!"
    assert inf_count == 0, "Error: Infinite values found!"
    print("Numeric check PASSED (0 NaNs, 0 Infs).")

    # 4. Target Distribution and SNR Alignment Check
    print("\nAnalyzing class and SNR distribution...")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print("Modulation classes and counts:")
    for idx, count in zip(unique_classes, class_counts):
        # Resolve class name
        from ml.dataset.labels import INDEX_TO_MODULATION
        print(f"  - {INDEX_TO_MODULATION[idx]}: {count}")

    unique_snrs, snr_counts = np.unique(snrs, return_counts=True)
    print(f"SNR levels covered ({len(unique_snrs)} levels): {sorted(list(unique_snrs))}")
    print(f"Average examples per SNR: {np.mean(snr_counts):.1f}")

    # 5. Extract splits to get Training Split for Correlation Diagnostics
    print("\nGetting train/validation/test splits to analyze correlation on training data...")
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits(pickle_path=real_dataset_path, seed=42)
    print(f"Train split size: {len(train_ds)} examples.")
    
    # Slice the feature matrix to get the training split
    X_train = X[train_ds.indices]

    # Calculate highly correlated feature pairs (correlation coefficient > 0.95)
    print("\nFinding highly correlated feature pairs (|r| > 0.95) on training split...")
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    feature_names = get_feature_names()
    
    correlated_pairs = []
    for i in range(NUM_FEATURES):
        for j in range(i + 1, NUM_FEATURES):
            r = corr_matrix[i, j]
            if abs(r) > 0.95:
                correlated_pairs.append((feature_names[i], feature_names[j], r))

    if correlated_pairs:
        print(f"Found {len(correlated_pairs)} highly correlated feature pairs:")
        for name1, name2, r in sorted(correlated_pairs, key=lambda x: -abs(x[2])):
            print(f"  - {name1} <-> {name2}: r = {r:.4f}")
    else:
        print("No highly correlated feature pairs (|r| > 0.95) found.")

    # 6. Feature summary statistics
    print("\nFeature Summary Statistics:")
    print(f"{'Feature Name':<35} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 85)
    for idx, name in enumerate(feature_names):
        col = X[:, idx]
        print(f"{name:<35} | {np.min(col):<10.4f} | {np.max(col):<10.4f} | {np.mean(col):<10.4f} | {np.std(col):<10.4f}")

    # 7. Save to NPZ file
    output_dir = "datasets/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "RML2016.10a_features.npz")
    print(f"\nSaving processed features, labels, and SNR metadata to {output_path}...")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        snrs=snrs,
        feature_names=feature_names
    )
    print("Save complete!")
    print("==================================================")

if __name__ == "__main__":
    inspect_and_save_features()
