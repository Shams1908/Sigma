from dataclasses import dataclass, field
from typing import List
from ml.dataset.labels import MODULATION_CLASSES

@dataclass(frozen=True)
class DatasetMetadata:
    """
    Dataset metadata representation containing standard RadioML 2016.10A facts.
    """
    dataset_name: str
    dataset_version: str
    source: str
    num_examples: int
    num_classes: int
    class_names: List[str]
    snr_values: List[int]
    sample_shape: List[int]
    dtype: str
    iq_channels: List[str]

def get_dataset_metadata() -> DatasetMetadata:
    """
    Retrieve the standard metadata for the RadioML 2016.10A dataset.
    """
    return DatasetMetadata(
        dataset_name="RadioML 2016.10A",
        dataset_version="2016.10A",
        source="RadioML 2016.10A / DeepSig",
        num_examples=220000,
        num_classes=11,
        class_names=MODULATION_CLASSES,
        snr_values=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        sample_shape=[2, 128],
        dtype="float32",
        iq_channels=["I", "Q"]
    )
