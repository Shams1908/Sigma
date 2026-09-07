"""
Integration tests for the FastAPI application.

These tests use HTTPX's AsyncClient against the real FastAPI app (in-process),
with the DB layer disabled (placeholder URI) so they work without Atlas.

Covers:
  - GET /health
  - POST /api/v1/upload  (success + bad file)
  - POST /api/v1/analysis/{signal_id}  (creates job, returns 202)
  - GET /api/v1/analysis/{analysis_id}/status  (polling)
  - GET /api/v1/results/{analysis_id}  (returns results)
  - GET /api/v1/results/{analysis_id}/report  (409 while pending, 200 after done)
  - Failure path: bad file → 422; missing signal → 404
"""
from __future__ import annotations

import asyncio
import io
import struct
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

# Backend on path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

# Import app AFTER path setup
from main import app  # noqa: E402

try:
    from httpx import AsyncClient, ASGITransport
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HTTPX_AVAILABLE, reason="httpx not installed"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav_bytes(n_samples: int = 2048, sample_rate: int = 80_000) -> bytes:
    """Generate a minimal stereo WAV file (I/Q channels) in memory."""
    np.random.seed(99)
    data = np.random.randn(n_samples, 2).astype(np.float32)
    data = (data / (np.max(np.abs(data)) + 1e-9) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def wav_bytes():
    return _make_wav_bytes()


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Health check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "db_connected" in body


# ── Upload ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_valid_wav(client, wav_bytes):
    r = await client.post(
        "/api/v1/upload",
        files={"file": ("test_signal.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "signal_id" in body
    assert "metadata" in body
    meta = body["metadata"]
    # Must match frontend SignalMetadata field names
    for field in ("fileName", "sampleRate", "centerFrequency", "duration", "fileSize", "ingestionTime"):
        assert field in meta, f"Missing field in metadata: {field}"


@pytest.mark.asyncio
async def test_upload_invalid_file(client):
    """Non-signal file should return 422 Unprocessable Entity."""
    garbage = b"this is not a valid IQ or WAV file at all"
    r = await client.post(
        "/api/v1/upload",
        files={"file": ("garbage.wav", garbage, "audio/wav")},
    )
    assert r.status_code in (422, 400), r.text


@pytest.mark.asyncio
async def test_upload_no_file(client):
    r = await client.post("/api/v1/upload")
    assert r.status_code == 422  # FastAPI validation error


# ── Analysis create ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analysis_create_returns_202(client, wav_bytes):
    # First upload
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("sig.wav", wav_bytes, "audio/wav")},
    )
    assert up.status_code == 201
    signal_id = up.json()["signal_id"]

    # Then trigger analysis
    r = await client.post(f"/api/v1/analysis/{signal_id}")
    assert r.status_code == 202, r.text
    body = r.json()
    assert "analysis_id" in body
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_analysis_unknown_signal(client):
    r = await client.post(f"/api/v1/analysis/{uuid.uuid4()}")
    assert r.status_code == 404


# ── Status polling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_pending_immediately(client, wav_bytes):
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("sig2.wav", wav_bytes, "audio/wav")},
    )
    signal_id = up.json()["signal_id"]
    cr = await client.post(f"/api/v1/analysis/{signal_id}")
    analysis_id = cr.json()["analysis_id"]

    st = await client.get(f"/api/v1/analysis/{analysis_id}/status")
    assert st.status_code == 200
    body = st.json()
    assert body["status"] in ("pending", "running", "done", "failed")
    assert body["analysis_id"] == analysis_id


@pytest.mark.asyncio
async def test_status_unknown_analysis(client):
    r = await client.get(f"/api/v1/analysis/{uuid.uuid4()}/status")
    assert r.status_code == 404


# ── Results ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_results_available_immediately(client, wav_bytes):
    """GET /results/{id} should return a response even while pending (partial data)."""
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("sig3.wav", wav_bytes, "audio/wav")},
    )
    signal_id = up.json()["signal_id"]
    cr = await client.post(f"/api/v1/analysis/{signal_id}")
    analysis_id = cr.json()["analysis_id"]

    r = await client.get(f"/api/v1/results/{analysis_id}")
    assert r.status_code == 200
    body = r.json()
    assert "analysis_id" in body
    assert "hypotheses" in body
    assert "status" in body


@pytest.mark.asyncio
async def test_results_unknown_analysis(client):
    r = await client.get(f"/api/v1/results/{uuid.uuid4()}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_report_409_while_pending(client, wav_bytes):
    """Report endpoint returns 409 while analysis hasn't completed yet."""
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("sig4.wav", wav_bytes, "audio/wav")},
    )
    signal_id = up.json()["signal_id"]
    cr = await client.post(f"/api/v1/analysis/{signal_id}")
    analysis_id = cr.json()["analysis_id"]

    # Immediately poll before background task can finish
    r = await client.get(f"/api/v1/results/{analysis_id}/report")
    # Should be 409 (not done yet) or 200 if the task completed very quickly
    assert r.status_code in (409, 200)


# ── Full pipeline (with wait) ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_full_pipeline_completes(client, wav_bytes):
    """
    Upload → trigger → poll until done/failed (or timeout).
    Validates the complete flow without requiring CNN model on disk.
    Analysis is expected to complete (possibly as 'failed' if CNN model
    is unavailable) — the key requirement is that it NEVER stays 'running'
    forever and NEVER crashes the server.
    """
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("sig5.wav", wav_bytes, "audio/wav")},
    )
    assert up.status_code == 201
    signal_id = up.json()["signal_id"]

    cr = await client.post(f"/api/v1/analysis/{signal_id}")
    assert cr.status_code == 202
    analysis_id = cr.json()["analysis_id"]

    # Poll for up to 30 seconds
    final_status = None
    for _ in range(60):
        await asyncio.sleep(0.5)
        st = await client.get(f"/api/v1/analysis/{analysis_id}/status")
        assert st.status_code == 200
        current = st.json()["status"]
        if current in ("done", "failed"):
            final_status = current
            break

    # Must have resolved to a terminal state
    assert final_status in ("done", "failed"), (
        f"Analysis did not complete within 30s; last status={final_status}"
    )

    # Results endpoint must work after completion
    r = await client.get(f"/api/v1/results/{analysis_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == final_status

    # Report endpoint must return 200 now
    rp = await client.get(f"/api/v1/results/{analysis_id}/report")
    assert rp.status_code == 200
    rp_body = rp.json()
    assert "analysis_id" in rp_body
    assert "status" in rp_body


# ── Response schema field names ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_metadata_field_names(client, wav_bytes):
    """Verify exact field names match frontend/src/types/index.ts SignalMetadata."""
    r = await client.post(
        "/api/v1/upload",
        files={"file": ("check.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 201
    meta = r.json()["metadata"]
    # These exact names are required by the frontend TypeScript interface
    required = {"fileName", "sampleRate", "centerFrequency", "duration", "fileSize", "ingestionTime"}
    missing = required - set(meta.keys())
    assert not missing, f"Metadata missing fields: {missing}"


@pytest.mark.asyncio
async def test_results_estimated_parameters_field_names(client, wav_bytes):
    """Verify EstimatedParameters field names match frontend types when present."""
    up = await client.post(
        "/api/v1/upload",
        files={"file": ("check2.wav", wav_bytes, "audio/wav")},
    )
    signal_id = up.json()["signal_id"]
    cr = await client.post(f"/api/v1/analysis/{signal_id}")
    analysis_id = cr.json()["analysis_id"]

    # Wait briefly for pipeline
    for _ in range(20):
        await asyncio.sleep(0.5)
        st = await client.get(f"/api/v1/analysis/{analysis_id}/status")
        if st.json()["status"] in ("done", "failed"):
            break

    r = await client.get(f"/api/v1/results/{analysis_id}")
    body = r.json()
    if body.get("parameters"):
        params = body["parameters"]
        required = {"snr", "bandwidth", "carrierOffset", "symbolRate"}
        missing = required - set(params.keys())
        assert not missing, f"EstimatedParameters missing fields: {missing}"
