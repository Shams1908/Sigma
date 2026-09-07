import os
import tempfile
import pickle
import pytest
import numpy as np

from ml.dataset.labels import (
    MODULATION_CLASSES,
    MODULATION_TO_INDEX,
    INDEX_TO_MODULATION,
    get_class_index,
    get_modulation_name,
)
from ml.dataset.metadata import get_dataset_metadata
from ml.dataset.schema import SignalExample
from ml.dataset.splitting import generate_splits
from ml.dataset.loader import RadioMLDataset, DatasetCache


@pytest.fixture(scope="module")
def mock_pickle_dataset():
    """
    Creates a small mock pickle file mimicking the structure of the RadioML 2016.10A dictionary.
    Includes all 11 classes and 20 SNRs, but with only 10 examples per key to ensure tests are extremely fast.
    """
    mock_data = {}
    modulations = MODULATION_CLASSES
    snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    examples_per_key = 10
    
    for mod in modulations:
        for snr in snrs:
            # Shape (10, 2, 128)
            mock_data[(mod, snr)] = np.random.randn(examples_per_key, 2, 128).astype(np.float32)
            
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(mock_data, tmp)
        tmp_path = tmp.name
        
    yield tmp_path
    
    # Cleanup file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# 1. Test Label Mapping Stability
def test_label_mapping_stability():
    assert len(MODULATION_CLASSES) == 11
    # Check that sorting is alphabetical and hardcoded stable
    sorted_classes = sorted(MODULATION_CLASSES)
    assert MODULATION_CLASSES == sorted_classes
    
    # Check mapping
    for i, name in enumerate(MODULATION_CLASSES):
        assert MODULATION_TO_INDEX[name] == i
        assert INDEX_TO_MODULATION[i] == name
        assert get_class_index(name) == i
        assert get_modulation_name(i) == name
        
    with pytest.raises(ValueError):
        get_class_index("UNKNOWN")
        
    with pytest.raises(ValueError):
        get_class_index("UNKNOWN_CLASS")

    with pytest.raises(ValueError):
        get_modulation_name(11)


# 2. Test Dataset Metadata
def test_dataset_metadata():
    metadata = get_dataset_metadata()
    assert metadata.dataset_name == "RadioML 2016.10A"
    assert metadata.dataset_version == "2016.10A"
    assert metadata.source == "RadioML 2016.10A / DeepSig"
    assert metadata.num_examples == 220000
    assert metadata.num_classes == 11
    assert metadata.class_names == MODULATION_CLASSES
    assert metadata.snr_values == [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert metadata.sample_shape == [2, 128]
    assert metadata.dtype == "float32"
    assert metadata.iq_channels == ["I", "Q"]


# 3. Test Loader Initialization (using Mock)
def test_loader_initialization_mock(mock_pickle_dataset):
    # Clear cache before running mock test to avoid contamination
    DatasetCache.clear()
    
    # Init mock dataset
    dataset = RadioMLDataset(pickle_path=mock_pickle_dataset)
    
    # Total examples: 11 classes * 20 SNRs * 10 examples = 2200
    assert len(dataset) == 2200
    
    # Test property getters
    samples = dataset.samples
    assert samples.shape == (2200, 2, 128)
    assert samples.dtype == np.float32
    
    modulations = dataset.modulations
    assert len(modulations) == 2200
    assert modulations[0] == MODULATION_CLASSES[0]
    
    snrs = dataset.snrs
    assert len(snrs) == 2200
    assert snrs[0] == -20
    
    class_indices = dataset.class_indices
    assert len(class_indices) == 2200
    assert class_indices[0] == 0

    # Test single example retrieval __getitem__
    example = dataset[0]
    assert isinstance(example, SignalExample)
    assert example.samples.shape == (2, 128)
    assert example.samples.dtype == np.float32
    assert example.modulation == MODULATION_CLASSES[0]
    assert example.snr == -20
    assert example.class_index == 0


# 4. Test Validation Logic with Corrupted Mock Dataset
def test_loader_validation_corrupted(mock_pickle_dataset):
    # Create invalid dataset structure to test validation rules
    # 4a. Wrong key structure (not a tuple)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump({"not_a_tuple_key": np.zeros((10, 2, 128))}, tmp)
        tmp_path = tmp.name
    try:
        DatasetCache.clear()
        with pytest.raises((TypeError, ValueError)):
            RadioMLDataset(pickle_path=tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4b. Missing modulation keys
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        # Save a dictionary with only 1 class
        pickle.dump({("QPSK", 0): np.zeros((10, 2, 128))}, tmp)
        tmp_path = tmp.name
    try:
        DatasetCache.clear()
        with pytest.raises(ValueError):
            RadioMLDataset(pickle_path=tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4c. Wrong shape of sample array
    mock_data = {}
    for mod in MODULATION_CLASSES:
        for snr in [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
            # Array shape is wrong: shape (10, 1, 128) instead of (10, 2, 128)
            mock_data[(mod, snr)] = np.zeros((10, 1, 128)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(mock_data, tmp)
        tmp_path = tmp.name
    try:
        DatasetCache.clear()
        with pytest.raises(ValueError):
            RadioMLDataset(pickle_path=tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# 5. Test Split Reproducibility and Overlap (using Mock)
def test_split_reproducibility_mock(mock_pickle_dataset):
    DatasetCache.clear()
    
    # 70% train, 15% validation, 15% test on mock (examples_per_key = 10)
    # n_train = 7, n_val = 1, n_test = 2
    # Total examples per key = 10
    # Total examples = 2200
    # Expected Train = 220 * 7 = 1540
    # Expected Val = 220 * 1 = 220
    # Expected Test = 220 * 2 = 440
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits(
        pickle_path=mock_pickle_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    assert len(train_ds) == 1540
    assert len(val_ds) == 220
    assert len(test_ds) == 440
    assert len(train_ds) + len(val_ds) + len(test_ds) == 2200
    
    # Verify no index overlap
    train_idx_set = set(train_ds.indices)
    val_idx_set = set(val_ds.indices)
    test_idx_set = set(test_ds.indices)
    
    assert len(train_idx_set.intersection(val_idx_set)) == 0
    assert len(train_idx_set.intersection(test_idx_set)) == 0
    assert len(val_idx_set.intersection(test_idx_set)) == 0
    
    # Verify reproducibility with same seed
    train_ds_2, val_ds_2, test_ds_2 = RadioMLDataset.get_splits(
        pickle_path=mock_pickle_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    assert np.array_equal(train_ds.indices, train_ds_2.indices)
    assert np.array_equal(val_ds.indices, val_ds_2.indices)
    assert np.array_equal(test_ds.indices, test_ds_2.indices)

    # Verify seed variation produces different splits
    train_ds_diff, _, _ = RadioMLDataset.get_splits(
        pickle_path=mock_pickle_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=100
    )
    assert not np.array_equal(train_ds.indices, train_ds_diff.indices)


# 6. Integration Test (Real Dataset)
# Only run if the real dataset is present locally
def test_real_dataset_integration():
    real_dataset_path = "datasets/raw/RML2016.10a_dict.pkl"
    if not os.path.exists(real_dataset_path):
        pytest.skip(f"Real dataset pickle not found at {real_dataset_path}. Skipping integration test.")
        
    DatasetCache.clear()
    
    dataset = RadioMLDataset(pickle_path=real_dataset_path)
    assert len(dataset) == 220000
    assert dataset.samples.shape == (220000, 2, 128)
    assert dataset.samples.dtype == np.float32
    
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits(
        pickle_path=real_dataset_path,
        seed=42
    )
    
    assert len(train_ds) == 154000
    assert len(val_ds) == 33000
    assert len(test_ds) == 33000
    
    # Check no overlap
    train_idx_set = set(train_ds.indices)
    val_idx_set = set(val_ds.indices)
    test_idx_set = set(test_ds.indices)
    
    assert len(train_idx_set.intersection(val_idx_set)) == 0
    assert len(train_idx_set.intersection(test_idx_set)) == 0
    assert len(val_idx_set.intersection(test_idx_set)) == 0
