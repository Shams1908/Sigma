import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ml.cnn_model.architecture import RawIQCNN
from ml.synthetic_training.dataset import PyTorchSignalDataset, SUPPORTED_5_CLASSES

def evaluate_model_on_test_set(
    checkpoint_path: str,
    test_dataset: PyTorchSignalDataset,
    snrs_test: np.ndarray,
    device: torch.device,
    conf_matrix_save_path: str = None
) -> dict:
    """
    Evaluates the trained 5-class model on the untouched test set.
    Computes overall performance, class-conditional metrics, and SNR slice metrics.
    Generates and saves the normalized confusion matrix plot.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RawIQCNN(num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Use the RMS factor stored in the checkpoint for preprocessing consistency!
    rms_factor = float(checkpoint["rms_factor"])
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    preds = []
    probs_list = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Reapply correct scaling dynamically from the checkpoint
            # Note: test_dataset might have loaded with a default factor, but we divide 
            # by self.rms. We can just override the dataset's rms to the checkpoint's factor!
            test_dataset.rms = rms_factor
            
            bx = batch_x.to(device)
            outputs = model(bx)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds.extend(np.argmax(probs, axis=1))
            probs_list.extend(probs)
            targets.extend(batch_y.numpy())
            
    preds = np.array(preds)
    targets = np.array(targets)
    probs_array = np.array(probs_list)
    
    # Calculate overall metrics
    acc = accuracy_score(targets, preds)
    prec, rec, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2, 3, 4], average="weighted", zero_division=0
    )
    
    # Calculate prediction entropy
    entropy = -np.sum(probs_array * np.log(probs_array + 1e-12), axis=1)
    avg_entropy = float(np.mean(entropy))
    
    # Calculate class-conditional metrics
    c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    
    class_metrics = {}
    for i, name in enumerate(SUPPORTED_5_CLASSES):
        class_metrics[name] = {
            "precision": float(c_prec[i]),
            "recall": float(c_rec[i]),
            "f1_score": float(c_f1[i])
        }
        
    # Calculate SNR slice metrics
    unique_snrs = sorted(list(set(snrs_test)))
    snr_metrics = {}
    for snr in unique_snrs:
        snr_mask = (snrs_test == snr)
        snr_targets = targets[snr_mask]
        snr_preds = preds[snr_mask]
        
        snr_acc = accuracy_score(snr_targets, snr_preds)
        _, _, snr_f1, _ = precision_recall_fscore_support(
            snr_targets, snr_preds, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0
        )
        snr_metrics[str(snr)] = {
            "accuracy": float(snr_acc),
            "macro_f1": float(snr_f1)
        }
        
    # Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3, 4])
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    if conf_matrix_save_path is not None:
        os.makedirs(os.path.dirname(conf_matrix_save_path), exist_ok=True)
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(SUPPORTED_5_CLASSES))
        plt.xticks(tick_marks, SUPPORTED_5_CLASSES, rotation=45)
        plt.yticks(tick_marks, SUPPORTED_5_CLASSES)
        
        # Display numbers in matrix cells
        fmt = '.2f'
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, format(cm_norm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
                         
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(conf_matrix_save_path, dpi=150)
        plt.close()
        
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "average_entropy": avg_entropy,
        "class_metrics": class_metrics,
        "snr_metrics": snr_metrics,
        "predictions_distribution": {
            SUPPORTED_5_CLASSES[c]: int(np.sum(preds == c)) for c in [0, 1, 2, 3, 4]
        }
    }
