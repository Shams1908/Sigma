import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.labels import MODULATION_CLASSES, get_class_index, INDEX_TO_MODULATION
from ml.synthetic.dataset import load_synthetic_dataset
from ml.synthetic.evaluate import get_state_dict_hash
from ml.features.extractor import extract_batch_features
from ml.features.schema import FEATURE_NAMES
from ml.cnn_model.architecture import RawIQCNN

def calculate_signal_stats(X: np.ndarray) -> pd.DataFrame:
    """
    Calculates raw IQ waveform statistics for an array of samples.
    Input shape: [N, 2, 128]
    """
    N = X.shape[0]
    stats_list = []
    
    for i in range(N):
        I_chan = X[i, 0]
        Q_chan = X[i, 1]
        z = I_chan + 1j * Q_chan
        amp = np.abs(z)
        
        i_mean = float(np.mean(I_chan))
        q_mean = float(np.mean(Q_chan))
        i_std = float(np.std(I_chan))
        q_std = float(np.std(Q_chan))
        
        i_rms = float(np.sqrt(np.mean(I_chan**2)))
        q_rms = float(np.sqrt(np.mean(Q_chan**2)))
        complex_rms = float(np.sqrt(np.mean(amp**2)))
        sig_power = float(np.mean(amp**2))
        
        peak_amp = float(np.max(amp))
        amp_var = float(np.var(amp))
        amp_kurt = float(kurtosis(amp, fisher=True))
        
        par = (peak_amp**2) / (sig_power + 1e-9)
        
        stats_list.append({
            "I_mean": i_mean,
            "Q_mean": q_mean,
            "I_std": i_std,
            "Q_std": q_std,
            "I_rms": i_rms,
            "Q_rms": q_rms,
            "complex_rms": complex_rms,
            "signal_power": sig_power,
            "peak_amplitude": peak_amp,
            "amplitude_variance": amp_var,
            "amplitude_kurtosis": amp_kurt,
            "peak_to_average_ratio": par
        })
        
    return pd.DataFrame(stats_list)

def run_analysis():
    print("==================================================")
    print("M6.3 REAL-VS-SYNTHETIC DOMAIN ANALYSIS")
    print("==================================================")
    
    os.makedirs("results/ml/m6/domain_analysis", exist_ok=True)
    os.makedirs("results/ml/m6/domain_visuals", exist_ok=True)
    
    # 1. Matching Loader (100 examples per class x SNR combination)
    print("Subsampling matched dataset splits...")
    ds = RadioMLDataset() # loads real cache
    real_samples = DatasetCache.samples
    real_mods = DatasetCache.modulations
    real_snrs = DatasetCache.snrs
    
    syn_samples, syn_y, syn_metadata = load_synthetic_dataset()
    syn_experiments = np.array([m["experiment"] for m in syn_metadata])
    syn_mods = np.array([m["modulation"] for m in syn_metadata])
    syn_snrs = np.array([m["snr"] for m in syn_metadata])

    supported_classes = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]
    snr_levels = [-20, -10, 0, 10, 18]
    num_ex = 100
    
    real_matched_idx = []
    syn_matched_idx = []
    
    # Seeded deterministic matching
    for mod in supported_classes:
        for snr in snr_levels:
            # Subsample Real
            r_mask = np.where((real_mods == mod) & (real_snrs == snr))[0]
            real_matched_idx.extend(r_mask[:num_ex])
            
            # Subsample Synthetic (AWGN experiment only, no other impairments)
            s_mask = np.where((syn_experiments == "awgn") & (syn_mods == mod) & (syn_snrs == float(snr)))[0]
            syn_matched_idx.extend(s_mask[:num_ex])
            
    X_real = real_samples[real_matched_idx]
    y_real = DatasetCache.class_indices[real_matched_idx]
    snrs_real = real_snrs[real_matched_idx]
    mods_real = real_mods[real_matched_idx]
    
    X_syn = syn_samples[syn_matched_idx]
    y_syn = syn_y[syn_matched_idx]
    snrs_syn = syn_snrs[syn_matched_idx]
    mods_syn = syn_mods[syn_matched_idx]
    
    print(f"Matched datasets extracted. Size: Real={X_real.shape}, Synthetic={X_syn.shape}")

    # 2. Raw IQ Statistics comparison
    print("Calculating raw IQ waveform statistics...")
    df_stats_real = calculate_signal_stats(X_real)
    df_stats_syn = calculate_signal_stats(X_syn)
    
    raw_stats_summary = []
    for col in df_stats_real.columns:
        rmean = float(np.mean(df_stats_real[col]))
        smean = float(np.mean(df_stats_syn[col]))
        abs_diff = smean - rmean
        rel_diff = abs_diff / (abs(rmean) + 1e-9)
        
        raw_stats_summary.append({
            "statistic": col,
            "real_mean": rmean,
            "synthetic_mean": smean,
            "absolute_difference": abs_diff,
            "relative_difference": rel_diff
        })
    df_raw_summary = pd.DataFrame(raw_stats_summary)
    df_raw_summary.to_csv("results/ml/m6/domain_analysis/raw_iq_statistics.csv", index=False)

    # 3. Running M3 Feature Extractor and Cohen's d Effect Size
    print("Extracting 36 M3 features for both domains...")
    feats_real = extract_batch_features(X_real)
    feats_syn = extract_batch_features(X_syn)
    
    feature_comparison = []
    for i, f_name in enumerate(FEATURE_NAMES):
        rmean = float(np.mean(feats_real[:, i]))
        rstd = float(np.std(feats_real[:, i]))
        smean = float(np.mean(feats_syn[:, i]))
        sstd = float(np.std(feats_syn[:, i]))
        
        mean_diff = smean - rmean
        rel_diff = mean_diff / (abs(rmean) + 1e-9)
        
        pooled_std = np.sqrt((rstd**2 + sstd**2) / 2.0)
        cohen_d = mean_diff / (pooled_std + 1e-9)
        
        feature_comparison.append({
            "feature": f_name,
            "real_mean": rmean,
            "synthetic_mean": smean,
            "real_std": rstd,
            "synthetic_std": sstd,
            "mean_difference": mean_diff,
            "relative_difference": rel_diff,
            "effect_size": cohen_d
        })
        
    df_feats = pd.DataFrame(feature_comparison)
    # Rank by absolute Cohen's d descending
    df_feats["abs_effect_size"] = df_feats["effect_size"].abs()
    df_feats_ranked = df_feats.sort_values(by="abs_effect_size", ascending=False).drop(columns=["abs_effect_size"])
    df_feats_ranked.to_csv("results/ml/m6/domain_analysis/domain_feature_comparison.csv", index=False)

    # 4. Class-Conditional Analysis
    print("Executing class-conditional domain analysis...")
    class_gap_rows = []
    for mod in supported_classes:
        r_mod_mask = (mods_real == mod)
        s_mod_mask = (mods_syn == mod)
        
        feats_r_mod = feats_real[r_mod_mask]
        feats_s_mod = feats_syn[s_mod_mask]
        
        for i, f_name in enumerate(FEATURE_NAMES):
            rmean = float(np.mean(feats_r_mod[:, i]))
            smean = float(np.mean(feats_s_mod[:, i]))
            class_gap_rows.append({
                "class": mod,
                "feature": f_name,
                "real_mean": rmean,
                "synthetic_mean": smean,
                "difference": smean - rmean
            })
    df_class_gap = pd.DataFrame(class_gap_rows)
    df_class_gap.to_csv("results/ml/m6/domain_analysis/domain_gap_by_class.csv", index=False)

    # 5. Overlapping Histograms for Top 10 domain-gap features
    print("Plotting top 10 domain-gap features distributions...")
    top_10_features = df_feats_ranked["feature"].head(10).tolist()
    
    plt.figure(figsize=(15, 12))
    for idx, f_name in enumerate(top_10_features):
        f_idx = FEATURE_NAMES.index(f_name)
        plt.subplot(4, 3, idx + 1)
        
        data_real = feats_real[:, f_idx]
        data_syn = feats_syn[:, f_idx]
        
        # Robustly calculate finite range
        r_min = float(min(np.min(data_real), np.min(data_syn)))
        r_max = float(max(np.max(data_real), np.max(data_syn)))
        if np.isclose(r_min, r_max):
            r_min -= 0.1
            r_max += 0.1
            
        plt.hist(data_real, bins=30, range=(r_min, r_max), alpha=0.5, color="red", label="REAL", density=True)
        plt.hist(data_syn, bins=30, range=(r_min, r_max), alpha=0.5, color="blue", label="SYNTHETIC", density=True)
        plt.title(f_name, fontsize=10)
        plt.grid(True)
        plt.legend(fontsize=8)
        
    plt.tight_layout()
    plt.savefig("results/ml/m6/domain_analysis/domain_gap_features.png", dpi=150)
    plt.close()

    # 6. PCA Domain Separability plot
    print("Performing PCA separability analysis in feature space...")
    feats_combined = np.concatenate([feats_real, feats_syn], axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(feats_combined)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_proj[:2500, 0], pca_proj[:2500, 1], color="red", alpha=0.6, label="REAL", s=15)
    plt.scatter(pca_proj[2500:, 0], pca_proj[2500:, 1], color="blue", alpha=0.6, label="SYNTHETIC", s=15)
    plt.title("PCA Separability: REAL vs SYNTHETIC Feature Space")
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/ml/m6/domain_analysis/pca_domain_separability.png", dpi=150)
    plt.close()

    # 7. Visual Signal Comparison (Waveforms and envelopes)
    print("Generating representative waveform comparison plots...")
    for mod in ["BPSK", "QPSK"]:
        r_idx = next(i for i, m in enumerate(mods_real) if m == mod and snrs_real[i] == 18)
        s_idx = next(i for i, m in enumerate(mods_syn) if m == mod and snrs_syn[i] == 18)
        
        sample_r = X_real[r_idx]
        sample_s = X_syn[s_idx]
        
        amp_r = np.abs(sample_r[0] + 1j * sample_r[1])
        amp_s = np.abs(sample_s[0] + 1j * sample_s[1])
        
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Real waveform
        plt.subplot(2, 2, 1)
        plt.plot(sample_r[0], 'b-', label='I Channel')
        plt.plot(sample_r[1], 'r-', label='Q Channel')
        plt.title(f"REAL: Matched {mod} waveform (SNR=18 dB)")
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: Synthetic waveform
        plt.subplot(2, 2, 2)
        plt.plot(sample_s[0], 'b-', label='I Channel')
        plt.plot(sample_s[1], 'r-', label='Q Channel')
        plt.title(f"SYNTHETIC: Matched {mod} waveform (SNR=18 dB)")
        plt.grid(True)
        plt.legend()
        
        # Subplot 3: Envelope comparison
        plt.subplot(2, 2, 3)
        plt.plot(amp_r, 'r-', label='REAL Envelope', linewidth=1.5)
        plt.plot(amp_s, 'b--', label='SYNTHETIC Envelope', linewidth=1.5)
        plt.title("Amplitude Envelope Comparison")
        plt.grid(True)
        plt.legend()
        
        # Subplot 4: Constellations comparison
        plt.subplot(2, 2, 4)
        plt.scatter(sample_r[0], sample_r[1], color="red", alpha=0.7, label="REAL", s=20)
        plt.scatter(sample_s[0], sample_s[1], color="blue", alpha=0.7, label="SYNTHETIC", s=20)
        plt.title("Constellation Plot (I vs Q)")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/ml/m6/domain_visuals/{mod.lower()}_18db_comparison.png", dpi=150)
        plt.close()

    # 8. Spectral comparison
    print("Performing spectral comparison (Average PSD)...")
    plt.figure(figsize=(12, 5))
    for idx, mod in enumerate(["BPSK", "QPSK"]):
        r_mod_indices = [i for i, m in enumerate(mods_real) if m == mod and snrs_real[i] == 18]
        s_mod_indices = [i for i, m in enumerate(mods_syn) if m == mod and snrs_syn[i] == 18]
        
        # Average real spectrum
        r_spectra = []
        for i in r_mod_indices:
            sig = X_real[i, 0] + 1j * X_real[i, 1]
            spec = np.abs(np.fft.fftshift(np.fft.fft(sig)))**2
            # normalize sum to 1
            r_spectra.append(spec / (np.sum(spec) + 1e-9))
        r_spectra_avg = np.mean(r_spectra, axis=0)
        
        # Average synthetic spectrum
        s_spectra = []
        for i in s_mod_indices:
            sig = X_syn[i, 0] + 1j * X_syn[i, 1]
            spec = np.abs(np.fft.fftshift(np.fft.fft(sig)))**2
            s_spectra.append(spec / (np.sum(spec) + 1e-9))
        s_spectra_avg = np.mean(s_spectra, axis=0)
        
        freqs = np.linspace(-0.5, 0.5, 128)
        
        plt.subplot(1, 2, idx + 1)
        plt.plot(freqs, 10 * np.log10(r_spectra_avg + 1e-12), "r-", label="REAL", linewidth=2)
        plt.plot(freqs, 10 * np.log10(s_spectra_avg + 1e-12), "b--", label="SYNTHETIC", linewidth=2)
        plt.title(f"{mod} Average PSD at SNR=18 dB")
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Normalized Power (dB)")
        plt.grid(True)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig("results/ml/m6/domain_analysis/spectral_comparison.png", dpi=150)
    plt.close()

    # 9. Prediction Collapse Analysis
    print("Evaluating matched predictions to inspect collapse...")
    checkpoint_path = "models/m5_iq_cnn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RawIQCNN(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Run predictions on raw synthetic samples
    rms_factor = float(checkpoint["rms_factor"])
    X_syn_norm = X_syn / rms_factor
    
    with torch.no_grad():
        outputs = model(torch.tensor(X_syn_norm, dtype=torch.float32))
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    # Analyze predicted class counts
    pred_names = [INDEX_TO_MODULATION[p] for p in preds]
    unique_preds, counts_preds = np.unique(pred_names, return_counts=True)
    
    df_pred_collapse = pd.DataFrame({
        "predicted_class": unique_preds,
        "count": counts_preds,
        "percentage": counts_preds / len(preds)
    }).sort_values(by="count", ascending=False)
    df_pred_collapse.to_csv("results/ml/m6/domain_analysis/prediction_distribution.csv", index=False)
    
    print("\nSynthetic Prediction Distribution:")
    for _, row in df_pred_collapse.iterrows():
        print(f"  {row['predicted_class']:<10}: {row['count']} ({row['percentage']*100:.1f}%)")
        
    # Calculate entropy of prediction probabilities: H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    avg_entropy = float(np.mean(entropy))
    print(f"  Average entropy of predictions:   {avg_entropy:.4f}")

    # 10. Training Diagnostic Domain Classifier
    print("\nTraining diagnostic classifier: REAL vs SYNTHETIC...")
    # Label 0: Real, Label 1: Synthetic
    domain_labels = np.concatenate([np.zeros(2500), np.ones(2500)], axis=0).astype(np.int32)
    feats_combined = np.concatenate([feats_real, feats_syn], axis=0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        feats_combined, domain_labels, test_size=0.3, random_state=42, stratify=domain_labels
    )
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"  Domain Classifier Train Accuracy: {train_acc:.4f}")
    print(f"  Domain Classifier Val Accuracy:   {val_acc:.4f}")
    
    # Identify splits
    importances = clf.feature_importances_
    tree_importances = []
    for idx, imp in enumerate(importances):
        if imp > 0.0:
            tree_importances.append({
                "feature": FEATURE_NAMES[idx],
                "importance": float(imp)
            })
    df_tree_imp = pd.DataFrame(tree_importances).sort_values(by="importance", ascending=False)
    print("  Top features used to distinguish domains:")
    for _, row in df_tree_imp.iterrows():
         print(f"    {row['feature']:<30}: {row['importance']:.4f}")

    # 11. Write Diagnostic Summary JSON
    summary = {
        "model_integrity": {
            "initial_md5_hash": "8a69430254987d587872edd2faf9c61d",
            "final_md5_hash": get_state_dict_hash(model.state_dict()),
            "weights_frozen": get_state_dict_hash(model.state_dict()) == "8a69430254987d587872edd2faf9c61d"
        },
        "statistics": {
            "real_samples_matched": 2500,
            "synthetic_samples_matched": 2500,
            "examples_per_condition": num_ex
        },
        "amplitude_check": {
            "real_raw_rms": float(np.sqrt(np.mean(X_real**2))),
            "synthetic_raw_rms": float(np.sqrt(np.mean(X_syn**2))),
            "normalized_real_rms": float(np.sqrt(np.mean((X_real / rms_factor)**2))),
            "normalized_synthetic_rms": float(np.sqrt(np.mean((X_syn / rms_factor)**2)))
        },
        "domain_classifier": {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "splits": df_tree_imp.to_dict(orient="records")
        },
        "prediction_collapse": {
            "avg_entropy": avg_entropy,
            "top_predicted_class": str(df_pred_collapse.iloc[0]["predicted_class"]),
            "top_predicted_percentage": float(df_pred_collapse.iloc[0]["percentage"])
        }
    }
    
    with open("results/ml/m6/domain_analysis/diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Saved diagnostic summary to results/ml/m6/domain_analysis/diagnostic_summary.json")
    print("==================================================")

if __name__ == "__main__":
    run_analysis()
