import os
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from ml.dataset.loader import RadioMLDataset
from ml.dataset.labels import MODULATION_CLASSES, INDEX_TO_MODULATION
from ml.cnn_model.dataset import RadioMLPyTorchDataset
from ml.cnn_model.architecture import RawIQCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_cnn_pipeline():
    print("==================================================")
    print("M5 RAW IQ CNN BASELINE TRAINING PIPELINE")
    print("==================================================")

    # 1. Setup seeds and directories
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/ml/m5", exist_ok=True)

    # 2. Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # 3. Ingest raw dataset splits
    print("\nRetrieving M1 deterministic splits...")
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits()
    print(f"Split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 4. Calculate training RMS normalization factor (strictly training split only)
    train_samples = train_ds.samples # Shape [154000, 2, 128]
    train_rms = float(np.sqrt(np.mean(train_samples**2)))
    print(f"\nTraining RMS normalization factor: {train_rms:.9f}")

    # Verify that the resulting training-set RMS is approximately 1.0
    normalized_train_samples = train_samples / train_rms
    verified_rms = float(np.sqrt(np.mean(normalized_train_samples**2)))
    print(f"Verified normalized training RMS: {verified_rms:.9f} (approximately 1.0)")
    assert np.allclose(verified_rms, 1.0, atol=1e-4), "Verification failed: Normalized RMS is not 1.0!"

    # 5. Create PyTorch Datasets and Dataloaders
    train_dataset = RadioMLPyTorchDataset(train_ds, train_rms)
    val_dataset = RadioMLPyTorchDataset(val_ds, train_rms)
    test_dataset = RadioMLPyTorchDataset(test_ds, train_rms)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 6. Initialize Network, Loss, Optimizer
    model = RawIQCNN(num_classes=11).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 7. Training Loop with Early Stopping based on Val Macro F1
    epochs = 25
    patience = 5
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": []
    }

    print("\nStarting model training...")
    total_train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                running_val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        _, _, epoch_val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)

        # Log History
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f}")

        # Checkpoint selection based on Validation Macro F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rms_factor": train_rms,
                "val_macro_f1": epoch_val_f1,
                "val_accuracy": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "label_mapping": MODULATION_CLASSES,
                "hyperparameters": {
                    "batch_size": 128,
                    "learning_rate": 1e-3,
                    "optimizer": "Adam",
                    "device": str(device)
                }
            }
            torch.save(checkpoint, "models/m5_iq_cnn.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (no validation F1 improvement for {patience} epochs).")
                break

    total_train_time = time.time() - total_train_start
    print(f"\nTraining completed in {total_train_time:.2f} seconds. Best epoch: {best_epoch} with Val F1: {best_val_f1:.4f}")

    # 8. Plot Training History Curves
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Accuracy/F1 Curve
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.plot(history["val_macro_f1"], label="Val Macro F1")
    plt.title("Performance History")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/ml/m5/training_history.png", dpi=150)
    plt.close()
    print("Saved training history curves plot to results/ml/m5/training_history.png")

    # 9. Final Test Set Evaluation (using Best Checkpoint)
    print("\nLoading best model checkpoint for final test set evaluation...")
    checkpoint = torch.load("models/m5_iq_cnn.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []
    test_probs = []
    
    t0 = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    test_inf_time = time.time() - t0
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)

    test_acc = accuracy_score(test_targets, test_preds)
    test_prec_macro, test_rec_macro, test_f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    _, _, test_f1_weighted, _ = precision_recall_fscore_support(test_targets, test_preds, average="weighted", zero_division=0)

    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Macro Precision:  {test_prec_macro:.4f}")
    print(f"  Macro Recall:     {test_rec_macro:.4f}")
    print(f"  Macro F1:         {test_f1_macro:.4f}")
    print(f"  Weighted F1:      {test_f1_weighted:.4f}")

    # Per-class metrics
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(test_targets, test_preds, labels=range(11), zero_division=0)
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

    # Per-SNR metrics (SNR is stored in test_ds.snrs)
    print("\nPer-SNR Performance Table:")
    print(f"{'SNR (dB)':<10} | {'Accuracy':<10} | {'Macro F1':<10} | {'Support':<10}")
    print("-" * 48)
    snrs_test = test_ds.snrs
    unique_snrs = sorted(list(set(snrs_test)))
    per_snr_list = []
    for snr in unique_snrs:
        snr_mask = (snrs_test == snr)
        y_test_snr = test_targets[snr_mask]
        preds_test_snr = test_preds[snr_mask]
        
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

    # 10. Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Save raw confusion matrix CSV
    np.savetxt("results/ml/m5/confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Normalized Confusion Matrix (M5 CNN Baseline)")
    plt.colorbar()
    tick_marks = np.arange(len(MODULATION_CLASSES))
    plt.xticks(tick_marks, MODULATION_CLASSES, rotation=45)
    plt.yticks(tick_marks, MODULATION_CLASSES)
    
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
    plt.savefig("results/ml/m5/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix plots to results/ml/m5/")

    # Plot Accuracy vs SNR & Macro F1 vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot([item["snr"] for item in per_snr_list], [item["accuracy"] for item in per_snr_list], "o-b", label="Accuracy", linewidth=2)
    plt.plot([item["snr"] for item in per_snr_list], [item["macro_f1"] for item in per_snr_list], "s-r", label="Macro F1", linewidth=2)
    plt.title("CNN Performance vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ml/m5/performance_vs_snr.png", dpi=150)
    plt.close()
    print("Saved performance vs SNR plot to results/ml/m5/performance_vs_snr.png")

    # 11. M4 vs M5 Baseline Comparison Table
    # M4 Baseline metrics (HGB full-feature configuration)
    m4_metrics = {
        "accuracy": 0.5494,
        "precision_macro": 0.6410,
        "recall_macro": 0.5494,
        "f1_macro": 0.5656,
        "f1_weighted": 0.5656
    }
    
    print("\n==================================================")
    print("M4 VS M5 BASELINE COMPARISON TABLE")
    print("==================================================")
    print(f"{'Metric':<20} | {'M4 HGB (Handcrafted)':<22} | {'M5 CNN (Raw IQ)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<20} | {m4_metrics['accuracy']:<22.4f} | {test_acc:<20.4f}")
    print(f"{'Macro Precision':<20} | {m4_metrics['precision_macro']:<22.4f} | {test_prec_macro:<20.4f}")
    print(f"{'Macro Recall':<20} | {m4_metrics['recall_macro']:<22.4f} | {test_rec_macro:<20.4f}")
    print(f"{'Macro F1':<20} | {m4_metrics['f1_macro']:<22.4f} | {test_f1_macro:<20.4f}")
    print(f"{'Weighted F1':<20} | {m4_metrics['f1_weighted']:<22.4f} | {test_f1_weighted:<20.4f}")
    print("==================================================")

    # 12. Leakage Audit Check 2
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    test_indices = set(test_ds.indices)
    assert len(train_indices.intersection(val_indices)) == 0, "Leakage: train and val overlap!"
    assert len(train_indices.intersection(test_indices)) == 0, "Leakage: train and test overlap!"
    assert len(val_indices.intersection(test_indices)) == 0, "Leakage: val and test overlap!"
    print("Leakage Audit Split Verification: PASSED.")

    # 13. Save metrics summary file
    results_summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "training_rms_factor": train_rms,
        "val_metrics_at_best_epoch": {
            "loss": best_val_loss,
            "accuracy": best_val_acc,
            "macro_f1": best_val_f1
        },
        "test_metrics": {
            "accuracy": test_acc,
            "macro_precision": test_prec_macro,
            "macro_recall": test_rec_macro,
            "macro_f1": test_f1_macro,
            "weighted_f1": test_f1_weighted
        },
        "m4_comparison": {
            "m4": m4_metrics,
            "m5": {
                "accuracy": test_acc,
                "precision_macro": test_prec_macro,
                "recall_macro": test_rec_macro,
                "f1_macro": test_f1_macro,
                "f1_weighted": test_f1_weighted
            }
        },
        "per_class": per_class_list,
        "per_snr": per_snr_list,
        "timing": {
            "total_train_time_sec": total_train_time,
            "test_inference_time_sec": test_inf_time,
            "per_sample_latency_ms": (test_inf_time / len(test_dataset)) * 1000.0
        }
    }
    
    with open("results/ml/m5/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)
        
    with open("results/ml/m5/history.json", "w") as f:
        json.dump(history, f, indent=4)
    print("Saved pipeline results and history JSONs.")
    print("==================================================")

if __name__ == "__main__":
    run_cnn_pipeline()
