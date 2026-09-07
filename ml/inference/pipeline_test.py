import sys
import argparse
from ml.inference.pipeline import analyze_file
from ml.input.types import PipelineConfig, BinaryIQConfig

def main():
    parser = argparse.ArgumentParser(description="SIGMA End-to-End Prediction pipeline CLI")
    parser.add_argument("file_path", type=str, help="Path to the signal file to analyze")
    parser.add_argument("--model-path", type=str, default="models/m5_iq_cnn.pt", help="Path to the PyTorch model checkpoint")
    parser.add_argument("--dtype", type=str, default=None, help="Dtype for raw binary files (e.g., float32, int16)")
    parser.add_argument("--interleaved", action="store_true", help="Interleaving flag for raw binary files")
    parser.add_argument("--endianness", type=str, default="little", choices=["little", "big"], help="Endianness of binary files")
    parser.add_argument("--segment-length", type=int, default=128, help="Window segmentation length")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to display")
    
    args = parser.parse_args()
    
    binary_cfg = None
    if args.dtype is not None:
        binary_cfg = BinaryIQConfig(
            dtype=args.dtype,
            interleaved=args.interleaved,
            endianness=args.endianness
        )
        
    config = PipelineConfig(
        segment_length=args.segment_length,
        binary_config=binary_cfg
    )
    
    try:
        result = analyze_file(args.file_path, config, args.model_path, top_k=args.top_k)
    except Exception as e:
        print("SIGMA Signal Analysis")
        print("=====================")
        print(f"Error: End-to-end analysis failed: {str(e)}")
        sys.exit(1)
        
    print("SIGMA Signal Analysis")
    print("=====================")
    print("\nInput")
    print("-----")
    print(f"File: {result.filename}")
    print(f"Format: {result.detected_format}")
    if result.sample_rate is not None:
        print(f"Sample rate: {result.sample_rate:.1f} Hz")
    else:
        print("Sample rate: N/A")
    print(f"Original samples: {result.original_sample_count}")
    print(f"Canonical IQ shape: {tuple(result.canonical_iq_shape)}")
    print(f"Segments: {result.num_windows}")
    
    print("\nInference")
    print("---------")
    print("Model: M5 Raw IQ CNN")
    print(f"Normalization: {result.normalization_source}")
    print(f"Windows analyzed: {result.num_windows}")
    
    print("\nFinal Prediction")
    print("----------------")
    print(f"Modulation: {result.predicted_class_name}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    
    print("\nTop Predictions")
    print("---------------")
    for name, score in result.top_predictions:
        print(f"{name:<7} {score * 100:.1f}%")
        
    print("\nWindow Distribution")
    print("-------------------")
    for name, count in sorted(result.window_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count}")

if __name__ == "__main__":
    main()
