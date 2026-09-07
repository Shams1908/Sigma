import sys
import argparse
from ml.input.pipeline import process_file
from ml.input.types import PipelineConfig, BinaryIQConfig

def main():
    parser = argparse.ArgumentParser(description="SIGMA Input Pipeline CLI Tool")
    parser.add_argument("file_path", type=str, help="Path to the signal file (WAV, NPY, NPZ, BIN, DAT)")
    parser.add_argument("--dtype", type=str, default=None, help="Dtype for raw binary files (e.g. float32, int16)")
    parser.add_argument("--interleaved", action="store_true", help="Set raw binary interleaving to True")
    parser.add_argument("--endianness", type=str, default="little", choices=["little", "big"], help="Endianness of binary file")
    parser.add_argument("--max-size-mb", type=int, default=50, help="Configure maximum file upload size in MB")
    
    args = parser.parse_args()
    
    binary_cfg = None
    if args.dtype is not None:
        binary_cfg = BinaryIQConfig(
            dtype=args.dtype,
            interleaved=args.interleaved,
            endianness=args.endianness
        )
        
    config = PipelineConfig(
        max_file_size_bytes=args.max_size_mb * 1024 * 1024,
        binary_config=binary_cfg
    )
    
    print("SIGMA Input Pipeline")
    print("--------------------")
    print(f"File: {args.file_path}")
    
    segments, meta = process_file(args.file_path, config)
    
    print(f"Detected format: {meta.detected_format}")
    if meta.num_channels is not None:
        print(f"Channels: {meta.num_channels}")
    if meta.sample_rate is not None:
        print(f"Sample rate: {meta.sample_rate:.1f} Hz")
    print(f"Original samples: {meta.original_sample_count}")
    
    print(f"\nParsing: {meta.validation_status}")
    if meta.validation_status == "OK":
        print("Validation: OK")
        print(f"Canonical IQ shape: {tuple(meta.canonical_iq_shape)}")
        print(f"Segments: {meta.num_generated_segments}")
        print(f"Segment shape: ({meta.num_generated_segments}, 2, 128)")
    else:
        print("Validation: ERROR")
        print(f"Error message: {meta.error_message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
