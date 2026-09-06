import numpy as np
import torch
from torch.utils.data import Dataset
from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.synthetic.dataset import load_synthetic_dataset

SUPPORTED_5_CLASSES = ["8PSK", "BPSK", "QAM16", "QAM64", "QPSK"]
CLASS_5_TO_INDEX = {name: i for i, name in enumerate(SUPPORTED_5_CLASSES)}
INDEX_TO_CLASS_5 = {i: name for i, name in enumerate(SUPPORTED_5_CLASSES)}

class PyTorchSignalDataset(Dataset):
    """
    Standard PyTorch dataset wrapping raw signal arrays and mapping target class labels.
    Divides inputs by a dynamic RMS scaling factor.
    """
    def __init__(self, samples: np.ndarray, labels: np.ndarray, rms_factor: float):
        self.samples = samples.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.rms = float(rms_factor)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Apply scaling dynamically
        x = self.samples[idx] / self.rms
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_real_5_class_splits():
    """
    Loads M1 splits and filters them to keep only the 5 supported classes.
    Re-maps class labels to consecutive integers [0, 4].
    """
    # caches real dataset in cache
    ds_real = RadioMLDataset()
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits()
    
    real_samples = DatasetCache.samples
    real_mods = DatasetCache.modulations
    real_snrs = DatasetCache.snrs
    
    # Supported mask
    supported_mask = np.isin(real_mods, SUPPORTED_5_CLASSES)
    
    # Intersect splits with supported mask
    train_idx = np.intersect1d(train_ds.indices, np.where(supported_mask)[0])
    val_idx = np.intersect1d(val_ds.indices, np.where(supported_mask)[0])
    test_idx = np.intersect1d(test_ds.indices, np.where(supported_mask)[0])
    
    # Extract samples and construct target indices
    X_train = real_samples[train_idx]
    y_train = np.array([CLASS_5_TO_INDEX[m] for m in real_mods[train_idx]], dtype=np.int32)
    snrs_train = real_snrs[train_idx]
    
    X_val = real_samples[val_idx]
    y_val = np.array([CLASS_5_TO_INDEX[m] for m in real_mods[val_idx]], dtype=np.int32)
    snrs_val = real_snrs[val_idx]
    
    X_test = real_samples[test_idx]
    y_test = np.array([CLASS_5_TO_INDEX[m] for m in real_mods[test_idx]], dtype=np.int32)
    snrs_test = real_snrs[test_idx]
    
    return (X_train, y_train, snrs_train), (X_val, y_val, snrs_val), (X_test, y_test, snrs_test)


def load_synthetic_5_class_dataset(npz_path: str):
    """
    Loads a synthetic dataset from the specified NPZ path and retrieves labels.
    """
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y_class_indices = data["y"].astype(np.int32)
    
    # In M6 datasets, y contains indices from the 11-class alphabetically sorted MODULATION_CLASSES.
    # We must translate these indices to the 5-class alphabetic indexing [0, 4] safely.
    from ml.dataset.labels import INDEX_TO_MODULATION
    y_names = [INDEX_TO_MODULATION[idx] for idx in y_class_indices]
    
    # Map names to 5-class indices
    y_5 = np.array([CLASS_5_TO_INDEX[m] for m in y_names], dtype=np.int32)
    
    return X, y_5
