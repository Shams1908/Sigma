"""
API integration tests for Decoder Lab endpoints in backend.api.analysis.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.fec import encode_convolutional_r12_k7
from backend.validation import append_crc_to_bits

client = TestClient(app)


def _make_test_iq_lists(n_bits: int = 16):
    """Generates simple BPSK I and Q lists for API payloads."""
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1][:n_bits], dtype=np.uint8)
    bpsk = np.where(payload == 1, 1.0, -1.0).tolist()
    zeros = [0.0] * len(bpsk)
    return bpsk, zeros


def test_api_decoder_lab_post_with_2d_iq():
    """Verifies POST /api/v1/analysis/decoder-lab with 2D 'iq' array format."""
    i_samples, q_samples = _make_test_iq_lists(16)
    payload = {
        "iq": [i_samples, q_samples],
        "sample_rate": 1000.0,
        "symbol_rate": 1000.0,
        "ml_probabilities": {"BPSK": 0.85, "QPSK": 0.15},
        "max_candidates": 12,
    }

    response = client.post("/api/v1/analysis/decoder-lab", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["total_evaluated"] > 0
    assert data["total_evaluated"] <= 12
    assert len(data["ranked_hypotheses"]) > 0
    assert data["best_hypothesis_id"] is not None
    assert data["best_hypothesis_modulation"] == "BPSK"
    assert data["best_hypothesis_confidence"] > 0.0
    assert "search_summary" in data
    assert "diagnostics" in data


def test_api_decoder_lab_post_with_separate_samples():
    """Verifies POST /api/v1/analysis/decoder-lab with i_samples and q_samples."""
    i_samples, q_samples = _make_test_iq_lists(16)
    payload = {
        "i_samples": i_samples,
        "q_samples": q_samples,
        "sample_rate": 2000.0,
        "symbol_rate": 1000.0,
        "ml_probabilities": {"BPSK": 0.90},
    }

    response = client.post("/api/v1/analysis/decoder-lab", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["total_evaluated"] > 0
    assert data["best_hypothesis_id"] is not None
    assert data["best_hypothesis_modulation"] == "BPSK"


def test_api_decoder_lab_post_validation_error_mismatched_iq():
    """Verifies 422 HTTP error when IQ array lengths mismatch."""
    payload = {
        "iq": [[1.0, -1.0, 1.0], [0.0]],  # 3 I samples, 1 Q sample
        "sample_rate": 1000.0,
    }

    response = client.post("/api/v1/analysis/decoder-lab", json=payload)
    assert response.status_code == 422
    assert "mismatch" in response.text.lower()


def test_api_decoder_lab_post_missing_iq():
    """Verifies 422 HTTP error when neither 'iq' nor 'i_samples'/'q_samples' are supplied."""
    payload = {
        "sample_rate": 1000.0,
    }

    response = client.post("/api/v1/analysis/decoder-lab", json=payload)
    assert response.status_code == 422
    assert "missing iq signal" in response.text.lower()


def test_api_decoder_lab_post_candidate_bound():
    """Verifies that max_candidates cannot exceed 24."""
    i_samples, q_samples = _make_test_iq_lists(16)
    payload = {
        "iq": [i_samples, q_samples],
        "sample_rate": 1000.0,
        "max_candidates": 30,  # exceeds 24
    }

    response = client.post("/api/v1/analysis/decoder-lab", json=payload)
    # Pydantic schema validation should reject max_candidates > 24 with 422
    assert response.status_code == 422


def test_api_decoder_lab_demo_endpoint():
    """
    Verifies GET /api/v1/analysis/decoder-lab/demo runs deterministic
    synthetic BPSK+FEC+CRC frame and returns validated decode.
    """
    response = client.get("/api/v1/analysis/decoder-lab/demo")
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["has_validated_decode"] is True
    assert data["validated_decode"] is not None

    val_info = data["validated_decode"]
    assert val_info["modulation"] == "BPSK"
    assert val_info["fec_config"] == "conv_r1/2_k7"
    assert val_info["crc_scheme"] == "CRC-16-CCITT"
    assert val_info["bits_count"] > 0
    assert val_info["bits_preview"] is not None

    # Verify best hypothesis exists
    assert data["best_hypothesis_id"] is not None
    assert data["total_evaluated"] > 0
    assert data["successful_decodes_count"] >= 1
