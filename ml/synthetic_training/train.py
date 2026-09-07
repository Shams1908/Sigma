import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from ml.cnn_model.architecture import RawIQCNN
from ml.synthetic_training.dataset import PyTorchSignalDataset, PyTorchSignalDataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(
    train_dataset: PyTorchSignalDataset,
    val_dataset: PyTorchSignalDataset,
    checkpoint_save_path: str,
    device: torch.device,
    max_epochs: int = 25,
    patience: int = 5,
    fine_tuning_state_dict: str = None
) -> float:
    """
    Trains the RawIQCNN model using standard Adam optimizer and early stopping based on Val Macro F1.
    If fine_tuning_state_dict is provided, loads the pre-trained weights first.
    """
    set_seed(42)
    os.makedirs(os.path.dirname(checkpoint_save_path), exist_ok=True)
    
    # Initialize network for 5 classes
    model = RawIQCNN(num_classes=5)
    
    if fine_tuning_state_dict is not None:
        print(f"  Loading pretrained weights for fine-tuning from {fine_tuning_state_dict}...")
        pretrained = torch.load(fine_tuning_state_dict, map_location="cpu")
        model.load_state_dict(pretrained["model_state_dict"])
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(bx := batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        # Calculate validation macro F1 over [0, 4] indices
        _, _, macro_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0
        )
        
        # Check early stopping
        if macro_f1 > best_val_f1 + 1e-4:
            best_val_f1 = macro_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Serialize checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rms_factor": train_dataset.rms,
                "best_val_macro_f1": best_val_f1
            }, checkpoint_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping triggered at Epoch {epoch}. Best Epoch: {best_epoch} with Val F1: {best_val_f1:.4f}")
            break
            
    print(f"  Training finished. Champion model saved at Epoch {best_epoch} with Val F1: {best_val_f1:.4f}")
    return best_val_f1
