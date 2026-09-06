import numpy as np

from ml.dataset.loader import RadioMLDataset
from ml.features.extractor import extract_features
from ml.baselines.inference import predict
CLASS_NAMES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

def main():
    dataset = RadioMLDataset()

    sample = dataset[0]
    iq = sample.samples

    print("IQ shape:", iq.shape)
    print("True modulation:", sample.modulation)
    print("True SNR:", sample.snr)

    # M3 API:
    # (feature_names, feature_values)
    feature_names, features = extract_features(iq)

    print("Feature shape:", features.shape)
    print("Number of features:", len(feature_names))

    # Sanity checks.
    assert features.shape == (36,)
    assert len(feature_names) == 36
    assert np.all(np.isfinite(features))

    # M4 inference.
    result = predict(features)

    print("\nPrediction")
    print("----------")
    print("Predicted:", result["class_name"])
    print("Class index:", result["class_index"])
    print("Confidence:", result["confidence"])

    probabilities = np.asarray(result["probabilities"])

    print("\nTop 5 probabilities:")
    top_indices = np.argsort(probabilities)[::-1][:5]

    for idx in top_indices:
        print(
            f"  {CLASS_NAMES[idx]}: "
            f"{probabilities[idx]:.6f}"
        )


if __name__ == "__main__":
    main()