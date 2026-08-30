import os
import numpy as np
from typing import Optional, Tuple, List
from ml.dataset.schema import SignalExample
from ml.dataset.labels import MODULATION_TO_INDEX

class DatasetCache:
    """
    In-memory cache for loaded, validated, and flattened RadioML dataset arrays.
    
    This keeps memory usage optimal by storing data once at the class level.
    It can be easily refactored to support chunked or memory-mapped loading
    in the future without changing the public interfaces.
    """
    _cached_path: Optional[str] = None
    samples: Optional[np.ndarray] = None
    modulations: Optional[np.ndarray] = None
    snrs: Optional[np.ndarray] = None
    class_indices: Optional[np.ndarray] = None
    sorted_keys: Optional[List[Tuple[str, int]]] = None

    @classmethod
    def load(cls, pickle_path: str) -> None:
        """
        Loads, validates, and flattens the dataset pickle file if not already cached.
        
        Args:
            pickle_path (str): Path to the RML2016.10a_dict.pkl file.
        """
        if cls._cached_path == pickle_path and cls.samples is not None:
            return

        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"RadioML dataset pickle not found at: {pickle_path}")

        import pickle
        with open(pickle_path, "rb") as f:
            try:
                # RadioML 2016.10A dictionary is a python 2 pickle, requiring latin1 encoding
                raw_data = pickle.load(f, encoding="latin1")
            except Exception:
                f.seek(0)
                raw_data = pickle.load(f, encoding="bytes")
                # Normalize keys from bytes to str if needed
                normalized = {}
                for k, v in raw_data.items():
                    mod_key = k[0].decode("utf-8") if isinstance(k[0], bytes) else k[0]
                    snr_key = int(k[1])
                    normalized[(mod_key, snr_key)] = v
                raw_data = normalized

        # Determine examples per group (defaulting to 1000 if not inspectable)
        if isinstance(raw_data, dict) and len(raw_data) > 0:
            first_key = next(iter(raw_data.keys()))
            first_val = raw_data[first_key]
            examples_per_group = first_val.shape[0] if hasattr(first_val, "shape") else 1000
        else:
            examples_per_group = 1000

        # Validate the raw dataset structure using validation module
        from ml.dataset.validation import validate_raw_data
        validate_raw_data(raw_data, expected_examples_per_key=examples_per_group)

        # Deterministically sort the dictionary keys for stable indexing across runs
        cls.sorted_keys = sorted(
            [(str(k[0]), int(k[1])) for k in raw_data.keys()],
            key=lambda x: (x[0], x[1])
        )

        n_groups = len(cls.sorted_keys)
        total_examples = n_groups * examples_per_group

        # Allocate flat arrays
        cls.samples = np.empty((total_examples, 2, 128), dtype=np.float32)
        cls.snrs = np.empty(total_examples, dtype=np.int32)
        cls.class_indices = np.empty(total_examples, dtype=np.int32)
        mod_list = []

        for i, key in enumerate(cls.sorted_keys):
            offset = i * examples_per_group
            
            # Find the actual key in raw_data mapping (handling types safely)
            matching_key = None
            for rk in raw_data.keys():
                if str(rk[0]) == key[0] and int(rk[1]) == key[1]:
                    matching_key = rk
                    break

            val = raw_data[matching_key]
            
            # Write to flat arrays
            cls.samples[offset : offset + examples_per_group] = val.astype(np.float32)
            cls.snrs[offset : offset + examples_per_group] = key[1]
            cls.class_indices[offset : offset + examples_per_group] = MODULATION_TO_INDEX[key[0]]
            mod_list.extend([key[0]] * examples_per_group)

        cls.modulations = np.array(mod_list, dtype=object)
        cls._cached_path = pickle_path

    @classmethod
    def clear(cls) -> None:
        """
        Clears the cached arrays from memory.
        """
        cls._cached_path = None
        cls.samples = None
        cls.modulations = None
        cls.snrs = None
        cls.class_indices = None
        cls.sorted_keys = None


class RadioMLDataset:
    """
    Dataset loader for RadioML 2016.10A that yields signal examples.
    
    Uses index-based slicing on top of DatasetCache to represent splits 
    (train/validation/test) without copying large float32 sample arrays.
    """
    def __init__(self, pickle_path: str = "datasets/raw/RML2016.10a_dict.pkl", indices: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Args:
            pickle_path (str): Path to RML2016.10a_dict.pkl.
            indices (np.ndarray, optional): Array of specific indices to load. If None, loads all examples.
        """
        self.pickle_path = pickle_path
        DatasetCache.load(pickle_path)
        
        if indices is None:
            self.indices = np.arange(len(DatasetCache.samples), dtype=np.int32)
        else:
            self.indices = np.array(indices, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> SignalExample:
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): Subset index.
            
        Returns:
            SignalExample: Container holding samples, modulation name, SNR, and class index.
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
        
        real_idx = self.indices[idx]
        return SignalExample(
            samples=DatasetCache.samples[real_idx],
            modulation=str(DatasetCache.modulations[real_idx]),
            snr=int(DatasetCache.snrs[real_idx]),
            class_index=int(DatasetCache.class_indices[real_idx])
        )

    @property
    def samples(self) -> np.ndarray:
        """
        Returns the subset of samples array of shape [len(self), 2, 128].
        """
        return DatasetCache.samples[self.indices]

    @property
    def modulations(self) -> np.ndarray:
        """
        Returns the subset of modulations array.
        """
        return DatasetCache.modulations[self.indices]

    @property
    def snrs(self) -> np.ndarray:
        """
        Returns the subset of SNRs array.
        """
        return DatasetCache.snrs[self.indices]

    @property
    def class_indices(self) -> np.ndarray:
        """
        Returns the subset of encoded class indices.
        """
        return DatasetCache.class_indices[self.indices]

    @classmethod
    def get_splits(
        cls,
        pickle_path: str = "datasets/raw/RML2016.10a_dict.pkl",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple["RadioMLDataset", "RadioMLDataset", "RadioMLDataset"]:
        """
        Splits the dataset into train, validation, and test datasets in a reproducible stratified manner.
        
        Args:
            pickle_path (str): Path to RML2016.10a_dict.pkl.
            train_ratio (float): Fraction of dataset for training (default 0.70).
            val_ratio (float): Fraction of dataset for validation (default 0.15).
            test_ratio (float): Fraction of dataset for testing (default 0.15).
            seed (int): Configurable random seed for the split (default 42).
            
        Returns:
            Tuple[RadioMLDataset, RadioMLDataset, RadioMLDataset]: Train, Validation, and Test datasets.
        """
        DatasetCache.load(pickle_path)
        
        # Get examples per group from cache structure
        first_key = next(iter(DatasetCache.sorted_keys))
        # Group size is calculated by total samples / number of groups
        examples_per_key = len(DatasetCache.samples) // len(DatasetCache.sorted_keys)
        
        from ml.dataset.splitting import generate_splits
        train_idx, val_idx, test_idx = generate_splits(
            DatasetCache.sorted_keys,
            examples_per_key=examples_per_key,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        return (
            cls(pickle_path, train_idx),
            cls(pickle_path, val_idx),
            cls(pickle_path, test_idx)
        )
