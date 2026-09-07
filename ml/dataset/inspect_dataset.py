import argparse
import sys
import numpy as np
from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.metadata import get_dataset_metadata

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="SIGMA RadioML 2016.10A Dataset Inspection Utility")
    parser.add_argument(
        "--path",
        type=str,
        default="datasets/raw/RML2016.10a_dict.pkl",
        help="Path to the RML2016.10a_dict.pkl file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation (default: 42)"
    )
    args = parser.parse_args()

    print_section("SIGMA DATASET INSPECTION")
    print(f"Target dataset path: {args.path}")

    # Load dataset
    try:
        dataset = RadioMLDataset(pickle_path=args.path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please place 'RML2016.10a_dict.pkl' in 'datasets/raw/'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    metadata = get_dataset_metadata()

    # Report details
    print(f"Dataset Name:        {metadata.dataset_name}")
    print(f"Dataset Version:     {metadata.dataset_version}")
    print(f"Dataset Source:      {metadata.source}")
    print(f"Underlying Object:   {type(DatasetCache.samples)}")
    print(f"Number of keys:      {len(DatasetCache.sorted_keys)}")
    print(f"Modulation classes:  {len(metadata.class_names)} classes")
    print(f"  Names:             {', '.join(metadata.class_names)}")
    print(f"SNR values (dB):     {len(metadata.snr_values)} levels")
    print(f"  Range:             {metadata.snr_values}")
    print(f"Total Examples:      {len(dataset)}")
    print(f"Sample Shape:        {metadata.sample_shape} [Channels (I, Q), Samples]")
    print(f"Data Type (dtype):   {DatasetCache.samples.dtype}")
    
    # Examples per modulation/SNR pair
    print(f"Examples per pair:   {len(dataset) // len(DatasetCache.sorted_keys)}")

    # Generate and report splits
    print_section("SPLIT STATISTICS ( reproducibility check )")
    print(f"Using split seed:    {args.seed}")
    
    try:
        train_ds, val_ds, test_ds = RadioMLDataset.get_splits(
            pickle_path=args.path,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed
        )
        
        print(f"Train set:           {len(train_ds):,} examples (70%)")
        print(f"Validation set:      {len(val_ds):,} examples (15%)")
        print(f"Test set:            {len(test_ds):,} examples (15%)")
        print(f"Total splits sum:    {len(train_ds) + len(val_ds) + len(test_ds):,} examples")

        # Double check overlap
        train_set = set(train_ds.indices)
        val_set = set(val_ds.indices)
        test_set = set(test_ds.indices)

        overlap_train_val = train_set.intersection(val_set)
        overlap_train_test = train_set.intersection(test_set)
        overlap_val_test = val_set.intersection(test_set)

        print("\nOverlap Check:")
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("[FAIL] Index overlaps found between splits!")
            print(f"  Train-Val overlap count:  {len(overlap_train_val)}")
            print(f"  Train-Test overlap count: {len(overlap_train_test)}")
            print(f"  Val-Test overlap count:   {len(overlap_val_test)}")
            sys.exit(1)
        else:
            print("[PASS] No index overlap between splits.")

        # Test index determinism by regenerating splits with same seed
        print("\nSeed Determinism Check:")
        t2_ds, v2_ds, te2_ds = RadioMLDataset.get_splits(
            pickle_path=args.path,
            seed=args.seed
        )
        
        train_match = np.array_equal(train_ds.indices, t2_ds.indices)
        val_match = np.array_equal(val_ds.indices, v2_ds.indices)
        test_match = np.array_equal(test_ds.indices, te2_ds.indices)
        
        if train_match and val_match and test_match:
            print(f"[PASS] Splits are perfectly reproducible with seed {args.seed}.")
        else:
            print("[FAIL] Splits generated from the same seed are not identical!")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed checking splits: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("=" * 60)
    print("Inspection complete.")

if __name__ == "__main__":
    main()
