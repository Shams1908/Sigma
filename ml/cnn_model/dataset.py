import torch
from torch.utils.data import Dataset
import numpy as np
from ml.dataset.loader import RadioMLDataset

class RadioMLPyTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for the RadioML 2016.10A IQ dataset.
    Reads directly from pre-sliced memory cached arrays in RadioMLDataset for high performance.
    """
    def __init__(self, dataset: RadioMLDataset, rms_factor: float):
        """
        Args:
            dataset (RadioMLDataset): In-memory loaded dataset.
            rms_factor (float): The calculated RMS normalization factor of the training split.
        """
        self.samples = dataset.samples # Shape [N, 2, 128]
        self.labels = dataset.class_indices # Shape [N]
        self.rms_factor = rms_factor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            Tuple[torch.Tensor, int]: Normalized sample of shape [2, 128] and target class index.
        """
        # Retrieve and normalize the sample using the training-only RMS factor
        x = self.samples[idx] / self.rms_factor
        y = self.labels[idx]
        
        # Convert to torch tensors
        return torch.tensor(x, dtype=torch.float32), int(y)
