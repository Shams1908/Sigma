import os
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ml.dataset.loader import RadioMLDataset
from ml.dataset.labels import MODULATION_CLASSES, INDEX_TO_MODULATION
from ml.features.schema import FEATURE_NAMES, NUM_FEATURES

def run_baseline_pipeline():
    print("==================================================")
    print("M4 CLASSICAL MACHINE LEARNING BASELINE PIPELINE")
    print("==================================================")

    # 1. Setup Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/ml/m4", exist_ok=True)

    # 2. Ingest Features and Labels
    npz_path = "datasets/processed/RML2016.10a_features.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Extracted feature file not found at: {npz_path}. Run M3 feature extraction first.")

    print("Loading pre-extracted feature dataset...")
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]
    feature_names = list(data["feature_names"])

    # 3. Retrieve Deterministic Split Indices
    print("Retrieving deterministic M1 split indices...")
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits()
    
    # Slice the datasets
    X_train, y_train, snrs_train = X[train_ds.indices], y[train_ds.indices], snrs[train_ds.indices]
    X_val, y_val, snrs_val = X[val_ds.indices], y[val_ds.indices], snrs[val_ds.indices]
    X_test, y_test, snrs_test = X[test_ds.indices], y[test_ds.indices], snrs[test_ds.indices]

    print(f"Data Splits: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    print(f"Feature count: {X_train.shape[1]}")

    # 4. Leakage Audit Check 1 (Basic Splits)
    train_set = set(train_ds.indices)
    val_set = set(val_ds.indices)
    test_set = set(test_ds.indices)
    assert len(train_set.intersection(val_set)) == 0, "Leakage: Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage: Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage: Val and Test overlap!"
    print("Leakage Audit Split Verification: PASSED (Zero split overlaps).")

    # 5. Define Candidate Classifier Configurations
    # Evaluate a small, controlled set of configurations
    rf_configs = [
        {"name": "RF_Config1", "n_estimators": 50, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
        {"name": "RF_Config2", "n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    ]

    hgb_configs = [
        {"name": "HGB_Config1", "max_iter": 100, "max_depth": 10, "learning_rate": 0.1, "min_samples_leaf": 20, "random_state": 42},
        {"name": "HGB_Config2", "max_iter": 150, "max_depth": 15, "learning_rate": 0.1, "min_samples_leaf": 20, "random_state": 42}
    ]

    candidate_results = []

    # 6. Train and Evaluate Random Forest Candidates
    for config in rf_configs:
        print(f"\nTraining Random Forest: {config['name']}...")
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=config["random_state"],
            n_jobs=config["n_jobs"]
        )
        
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        preds_val = model.predict(X_val)
        val_time = time.time() - t0
        
        acc = accuracy_score(y_val, preds_val)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds_val, average="macro")
        
        print(f"  Train time: {train_time:.2f}s | Val time: {val_time:.2f}s")
        print(f"  Validation Metrics: Accuracy={acc:.4f} | Macro F1={f1:.4f} | Macro Prec={prec:.4f} | Macro Rec={rec:.4f}")
        
        candidate_results.append({
            "name": config["name"],
            "model_type": "Random Forest",
            "model": model,
            "config": config,
            "train_time": train_time,
            "val_time": val_time,
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "val_macro_precision": prec,
            "val_macro_recall": rec
        })

    # 7. Train and Evaluate HistGradientBoosting Candidates
    for config in hgb_configs:
        print(f"\nTraining HistGradientBoosting: {config['name']}...")
        model = HistGradientBoostingClassifier(
            max_iter=config["max_iter"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=config["random_state"]
        )
        
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        preds_val = model.predict(X_val)
        val_time = time.time() - t0
        
        acc = accuracy_score(y_val, preds_val)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds_val, average="macro")
        
        print(f"  Train time: {train_time:.2f}s | Val time: {val_time:.2f}s")
        print(f"  Validation Metrics: Accuracy={acc:.4f} | Macro F1={f1:.4f} | Macro Prec={prec:.4f} | Macro Rec={rec:.4f}")
        
        candidate_results.append({
            "name": config["name"],
            "model_type": "HistGradientBoosting",
            "model": model,
            "config": config,
            "train_time": train_time,
            "val_time": val_time,
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "val_macro_precision": prec,
            "val_macro_recall": rec
        })

    # 8. Model Selection
    # Select the configuration that achieves the highest macro F1 on validation split
    best_candidate = max(candidate_results, key=lambda x: x["val_macro_f1"])
    print(f"\n==================================================")
    print(f"CHAMPION MODEL SELECTED: {best_candidate['name']} ({best_candidate['model_type']})")
    print(f"Validation Macro F1: {best_candidate['val_macro_f1']:.4f}")
    print(f"==================================================")

    # 9. Retrain selected model configuration
    # Retrain strictly on the train split to prevent test leakage and maintain validation split isolation.
    # Note: best_candidate['model'] is already fitted on the training split, so we can reuse it directly!
    champion_model = best_candidate["model"]

    # 10. Controlled Feature Ablation / Redundancy Experiment
    print("\nRunning Feature Ablation / Redundancy Experiment...")
    redundant_names = [
        "instantaneous_frequency_variance",
        "instantaneous_frequency_mean",
        "autocorr_lag_1_magnitude",
        "autocorr_lag_2_magnitude",
        "autocorr_lag_4_magnitude"
    ]
    reduced_indices = [idx for idx, name in enumerate(feature_names) if name not in redundant_names]
    reduced_feature_names = [feature_names[idx] for idx in reduced_indices]
    
    print(f"Full feature set size: {len(feature_names)} features.")
    print(f"Reduced feature set size: {len(reduced_indices)} features (removed: {redundant_names}).")
    
    # Slice features for ablation
    X_train_red = X_train[:, reduced_indices]
    X_val_red = X_val[:, reduced_indices]

    # Instantiate and fit champion configuration on reduced feature set
    if best_candidate["model_type"] == "Random Forest":
        ablation_model = RandomForestClassifier(
            n_estimators=best_candidate["config"]["n_estimators"],
            max_depth=best_candidate["config"]["max_depth"],
            min_samples_split=best_candidate["config"]["min_samples_split"],
            min_samples_leaf=best_candidate["config"]["min_samples_leaf"],
            max_features=best_candidate["config"]["max_features"],
            random_state=best_candidate["config"]["random_state"],
            n_jobs=best_candidate["config"]["n_jobs"]
        )
    else:
        ablation_model = HistGradientBoostingClassifier(
            max_iter=best_candidate["config"]["max_iter"],
            max_depth=best_candidate["config"]["max_depth"],
            learning_rate=best_candidate["config"]["learning_rate"],
            min_samples_leaf=best_candidate["config"]["min_samples_leaf"],
            random_state=best_candidate["config"]["random_state"]
        )
        
    t0 = time.time()
    ablation_model.fit(X_train_red, y_train)
    red_train_time = time.time() - t0
    
    preds_val_red = ablation_model.predict(X_val_red)
    red_acc = accuracy_score(y_val, preds_val_red)
    _, _, red_f1, _ = precision_recall_fscore_support(y_val, preds_val_red, average="macro")
    
    print(f"Reduced Feature Model Validation Macro F1: {red_f1:.4f} (vs Full Feature: {best_candidate['val_macro_f1']:.4f})")
    print(f"Reduced Feature Model Validation Accuracy: {red_acc:.4f} (vs Full Feature: {best_candidate['val_accuracy']:.4f})")
    
    ablation_diff = red_f1 - best_candidate["val_macro_f1"]
    if ablation_diff > 0.001:
        print(f"Redundancy ablation IMPROVED validation performance by {ablation_diff:.4f}")
    elif abs(ablation_diff) <= 0.001:
        print("Redundancy ablation MAINTAINED validation performance (within +/-0.001 F1).")
    else:
        print(f"Redundancy ablation HURT validation performance by {abs(ablation_diff):.4f}")

    # 11. Final Evaluation on Untouched Test Set (using full 36 features)
    print("\nRunning final evaluation on untouched Test split...")
    t0 = time.time()
    preds_test = champion_model.predict(X_test)
    test_inf_time = time.time() - t0
    
    test_acc = accuracy_score(y_test, preds_test)
    test_prec_macro, test_rec_macro, test_f1_macro, _ = precision_recall_fscore_support(y_test, preds_test, average="macro")
    _, _, test_f1_weighted, _ = precision_recall_fscore_support(y_test, preds_test, average="weighted")
    
    print(f"Final Test Metrics:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Macro Precision:  {test_prec_macro:.4f}")
    print(f"  Macro Recall:     {test_rec_macro:.4f}")
    print(f"  Macro F1:         {test_f1_macro:.4f}")
    print(f"  Weighted F1:      {test_f1_weighted:.4f}")

    # Per-class metrics
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(y_test, preds_test, labels=range(11))
    print("\nPer-Class Performance Table:")
    print(f"{'Class Name':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 60)
    per_class_list = []
    for i, name in enumerate(MODULATION_CLASSES):
        print(f"{name:<12} | {class_prec[i]:<10.4f} | {class_rec[i]:<10.4f} | {class_f1[i]:<10.4f} | {class_supp[i]:<10}")
        per_class_list.append({
            "class_name": name,
            "precision": float(class_prec[i]),
            "recall": float(class_rec[i]),
            "f1_score": float(class_f1[i]),
            "support": int(class_supp[i])
        })

    # Per-SNR metrics
    print("\nPer-SNR Performance Table:")
    print(f"{'SNR (dB)':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Support':<10}")
    print("-" * 48)
    unique_snrs = sorted(list(set(snrs_test)))
    per_snr_list = []
    for snr in unique_snrs:
        snr_mask = (snrs_test == snr)
        y_test_snr = y_test[snr_mask]
        preds_test_snr = preds_test[snr_mask]
        
        snr_acc = accuracy_score(y_test_snr, preds_test_snr)
        _, _, snr_f1, _ = precision_recall_fscore_support(y_test_snr, preds_test_snr, average="macro", zero_division=0)
        snr_support = int(np.sum(snr_mask))
        
        print(f"{snr:<10} | {snr_acc:<10.4f} | {snr_f1:<10.4f} | {snr_support:<10}")
        per_snr_list.append({
            "snr": int(snr),
            "accuracy": float(snr_acc),
            "macro_f1": float(snr_f1),
            "support": snr_support
        })

    # 12. Confusion Matrix Generation
    cm = confusion_matrix(y_test, preds_test)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Save raw confusion matrix CSV
    np.savetxt("results/ml/m4/confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print("\nSaved raw confusion matrix to results/ml/m4/confusion_matrix.csv")

    # Save normalized confusion matrix plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Normalized Confusion Matrix ({best_candidate['name']})")
    plt.colorbar()
    tick_marks = np.arange(len(MODULATION_CLASSES))
    plt.xticks(tick_marks, MODULATION_CLASSES, rotation=45)
    plt.yticks(tick_marks, MODULATION_CLASSES)
    
    # Text annotations in each cell
    fmt = ".2f"
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
                     
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_plot_path = "results/ml/m4/confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved normalized confusion matrix plot to {cm_plot_path}")

    # Plot Accuracy vs SNR & Macro F1 vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot([item["snr"] for item in per_snr_list], [item["accuracy"] for item in per_snr_list], "o-b", label="Accuracy", linewidth=2)
    plt.plot([item["snr"] for item in per_snr_list], [item["macro_f1"] for item in per_snr_list], "s-r", label="Macro F1", linewidth=2)
    plt.title(f"Performance vs SNR ({best_candidate['name']})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    snr_plot_path = "results/ml/m4/performance_vs_snr.png"
    plt.savefig(snr_plot_path, dpi=150)
    plt.close()
    print(f"Saved performance vs SNR plot to {snr_plot_path}")

    # 13. Feature Importance (Random Forest only)
    top_features_list = []
    if "Random Forest" in best_candidate["model_type"]:
        importances = champion_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 Feature Importances:")
        print(f"{'Rank':<5} | {'Feature Name':<35} | {'Importance':<10}")
        print("-" * 56)
        
        # Save to CSV
        with open("results/ml/m4/feature_importance.csv", "w") as f:
            f.write("rank,feature,importance\n")
            for rank in range(NUM_FEATURES):
                idx = indices[rank]
                name = feature_names[idx]
                imp = importances[idx]
                f.write(f"{rank+1},{name},{imp:.6f}\n")
                if rank < 15:
                    print(f"{rank+1:<5} | {name:<35} | {imp:<10.6f}")
                    top_features_list.append({
                        "rank": rank + 1,
                        "feature": name,
                        "importance": float(imp)
                    })
        print("Saved feature importances to results/ml/m4/feature_importance.csv")

        # Plot Feature Importances (Top 15)
        plt.figure(figsize=(10, 6))
        top_indices = indices[:15]
        plt.barh(range(15), importances[top_indices][::-1], align="center", color="teal")
        plt.yticks(range(15), [feature_names[i] for i in top_indices][::-1])
        plt.xlabel("Relative Importance")
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.tight_layout()
        fi_plot_path = "results/ml/m4/feature_importance.png"
        plt.savefig(fi_plot_path, dpi=150)
        plt.close()
        print(f"Saved feature importance plot to {fi_plot_path}")

    # 14. Leakage Audit Check 2
    # Verify no target leakage in features
    # Check X shape
    assert X_train.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {X_train.shape[1]}"
    # Verify no column contains class index or SNR values exactly
    for c in range(NUM_FEATURES):
        column = X_train[:, c]
        # target checks
        assert not np.array_equal(column, y_train), f"Leakage: Column {c} is identical to targets!"
        assert not np.array_equal(column, snrs_train), f"Leakage: Column {c} is identical to SNRs!"
    print("Leakage Audit Content Verification: PASSED (Zero data leakage into X).")

    # 15. Save Champion Model & Metadata
    model_name = "baseline_rf" if "Random Forest" in best_candidate["model_type"] else "baseline_hgb"
    model_file_path = f"models/{model_name}.joblib"
    print(f"\nSerializing champion model to {model_file_path}...")
    joblib.dump(champion_model, model_file_path)
    
    # Save Metadata
    metadata = {
        "model_type": best_candidate["model_type"],
        "feature_schema_version": "M3_DSP_v1",
        "feature_count": NUM_FEATURES,
        "feature_names": feature_names,
        "label_mapping": MODULATION_CLASSES,
        "random_seed": 42,
        "training_configuration": {k: v for k, v in best_candidate["config"].items() if k != "model"},
        "training_summary": {
            "dataset_used": "RadioML 2016.10A",
            "train_size": int(X_train.shape[0]),
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
            "train_time_sec": float(best_candidate["train_time"]),
            "val_inference_time_sec": float(best_candidate["val_time"]),
            "test_inference_time_sec": float(test_inf_time),
            "per_sample_latency_ms": float((test_inf_time / X_test.shape[0]) * 1000.0)
        },
        "performance_validation": {
            "accuracy": float(best_candidate["val_accuracy"]),
            "macro_f1": float(best_candidate["val_macro_f1"]),
            "macro_precision": float(best_candidate["val_macro_precision"]),
            "macro_recall": float(best_candidate["val_macro_recall"])
        },
        "performance_test": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1_macro),
            "macro_precision": float(test_prec_macro),
            "macro_recall": float(test_rec_macro),
            "weighted_f1": float(test_f1_weighted)
        }
    }
    
    metadata_file_path = f"models/{model_name}_metadata.json"
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_file_path}")

    # Save results files
    results_summary = {
        "candidate_results": [
            {
                "name": item["name"],
                "model_type": item["model_type"],
                "val_accuracy": float(item["val_accuracy"]),
                "val_macro_f1": float(item["val_macro_f1"]),
                "train_time": float(item["train_time"])
            } for item in candidate_results
        ],
        "champion_test_metrics": metadata["performance_test"],
        "ablation_experiment": {
            "full_features_f1": float(best_candidate["val_macro_f1"]),
            "reduced_features_f1": float(red_f1),
            "removed_features": redundant_names,
            "performance_diff": float(ablation_diff)
        },
        "per_class": per_class_list,
        "per_snr": per_snr_list,
        "top_features": top_features_list
    }
    
    results_json_path = "results/ml/m4/metrics.json"
    with open(results_json_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"Saved pipeline results JSON to {results_json_path}")
    print("==================================================")

if __name__ == "__main__":
    run_baseline_pipeline()
