"""
Root pytest configuration and shared fixtures.

Adds backend/ to sys.path so all backend packages resolve without
installing the project as a package.
"""
from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Also make sure the repo root is on the path (for ml.* imports)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Signal generation helpers ─────────────────────────────────────────────────

def make_bpsk_iq(
    n_symbols: int = 512,
    sps: int = 8,
    symbol_rate: float = 10_000.0,
    snr_db: float = 20.0,
    carrier_offset_hz: float = 500.0,
    rolloff: float = 0.35,
) -> tuple[np.ndarray, float]:
    """
    Generate a synthetic BPSK IQ signal as a [2, N] float32 array.

    Returns (iq_2d, sample_rate).
    """
    sample_rate = symbol_rate * sps
    bits = np.random.randint(0, 2, n_symbols).astype(np.float32)
    symbols = (2 * bits - 1).astype(np.complex64)  # BPSK: ±1

    # Simple rectangular pulse shaping (no RRC to avoid ml dependency in fixtures)
    upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    # Box filter
    rrc = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(upsampled, rrc, mode="full")[: n_symbols * sps]

    # Add carrier offset
    t = np.arange(len(iq)) / sample_rate
    iq = iq * np.exp(1j * 2 * np.pi * carrier_offset_hz * t).astype(np.complex64)

    # Add AWGN
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_std = np.sqrt(signal_power / (2.0 * snr_linear))
    noise = (noise_std * np.random.randn(len(iq)) +
             1j * noise_std * np.random.randn(len(iq))).astype(np.complex64)
    iq = iq + noise

    iq_2d = np.stack([iq.real, iq.imag], axis=0)
    return iq_2d.astype(np.float32), sample_rate


def make_qpsk_iq(
    n_symbols: int = 512,
    sps: int = 8,
    symbol_rate: float = 10_000.0,
    snr_db: float = 20.0,
    carrier_offset_hz: float = 200.0,
) -> tuple[np.ndarray, float]:
    """
    Generate a synthetic QPSK IQ signal as a [2, N] float32 array.

    Returns (iq_2d, sample_rate).
    """
    sample_rate = symbol_rate * sps
    _QPSK_NORM = 1.0 / np.sqrt(2.0)
    _QPSK_MAP = [
        _QPSK_NORM + 1j * _QPSK_NORM,   # 00
        -_QPSK_NORM + 1j * _QPSK_NORM,  # 10
        -_QPSK_NORM - 1j * _QPSK_NORM,  # 11
        _QPSK_NORM - 1j * _QPSK_NORM,   # 01
    ]
    dibit_indices = np.random.randint(0, 4, n_symbols)
    symbols = np.array([_QPSK_MAP[i] for i in dibit_indices], dtype=np.complex64)

    upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    rrc = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(upsampled, rrc, mode="full")[: n_symbols * sps]

    t = np.arange(len(iq)) / sample_rate
    iq = iq * np.exp(1j * 2 * np.pi * carrier_offset_hz * t).astype(np.complex64)

    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_std = np.sqrt(signal_power / (2.0 * snr_linear))
    noise = (noise_std * np.random.randn(len(iq)) +
             1j * noise_std * np.random.randn(len(iq))).astype(np.complex64)
    iq = iq + noise

    iq_2d = np.stack([iq.real, iq.imag], axis=0)
    return iq_2d.astype(np.float32), sample_rate


def write_wav(path: Path, iq_2d: np.ndarray, sample_rate: float) -> None:
    """Write a [2, N] float32 IQ array to a stereo WAV file (I=left, Q=right)."""
    # Normalise to int16 range
    data = iq_2d.T  # [N, 2]
    data_norm = data / (np.max(np.abs(data)) + 1e-9)
    data_int16 = (data_norm * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data_int16.tobytes())


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("sigma_tests")


@pytest.fixture(scope="session")
def bpsk_iq():
    np.random.seed(42)
    return make_bpsk_iq(n_symbols=512, sps=8, symbol_rate=10_000.0, snr_db=20.0)


@pytest.fixture(scope="session")
def qpsk_iq():
    np.random.seed(42)
    return make_qpsk_iq(n_symbols=512, sps=8, symbol_rate=10_000.0, snr_db=20.0)


@pytest.fixture(scope="session")
def bpsk_wav_path(tmp_dir, bpsk_iq):
    """WAV file containing a synthetic BPSK signal."""
    iq_2d, sample_rate = bpsk_iq
    path = tmp_dir / "test_bpsk.wav"
    write_wav(path, iq_2d, sample_rate)
    return path


@pytest.fixture(scope="session")
def qpsk_wav_path(tmp_dir, qpsk_iq):
    """WAV file containing a synthetic QPSK signal."""
    iq_2d, sample_rate = qpsk_iq
    path = tmp_dir / "test_qpsk.wav"
    write_wav(path, iq_2d, sample_rate)
    return path
