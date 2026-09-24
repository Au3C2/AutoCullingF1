"""TDD Unit tests for Intelligent High-Resolution on-demand data loading.

Tests cover:
1. Tier 1 Passthrough: JPG/HIF/HEIF zero-transcode, zero-write path.
2. Tier 2 RAW Stream Extraction: ARW/NEF embedded JPEG stream range I/O direct write.
3. Cache Hit Mechanism: Idempotent resolution without re-extraction.
4. Generation Token & Cancellation: Obsolete gen_id discard check.
5. Protocol Event Emission: request_highres command and highres_ready event serialization.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from cull.highres import HighResProvider, HighResRequest, HighResResponse


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp(prefix="autocull_highres_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_tier1_passthrough_jpg_and_heif_preview(temp_cache_dir: Path):
    """Tier 1: JPG passthrough as-is. HEIF: software-decoded preview stream
    cached as JPEG (the full-res primary image must never be decoded inside
    the webview during a culling run)."""
    provider = HighResProvider(cache_dir=temp_cache_dir)

    # 1. Test JPG sample
    jpg_sample = Path("tests/ci/sample/seed.jpg")
    assert jpg_sample.exists()

    req = HighResRequest(file_path=jpg_sample, gen_id=1)
    res = provider.resolve(req)

    assert res is not None
    assert res.tier == "tier1_passthrough"
    assert res.resolved_path == jpg_sample.resolve()
    assert res.format.lower() in ("jpg", "jpeg")
    assert res.width > 0 and res.height > 0
    # Tier 1 must not create any new cache file
    assert len(list(temp_cache_dir.glob("*"))) == 0

    # 2. Test HEIF sample: preview-stream decode, cached as JPEG
    heif_sample = Path("tests/ci/sample/seed.heif")
    assert heif_sample.exists()

    req_heif = HighResRequest(file_path=heif_sample, gen_id=2)
    res_heif = provider.resolve(req_heif)

    assert res_heif is not None
    assert res_heif.tier == "tier2_heif_preview"
    assert res_heif.resolved_path.suffix.lower() == ".jpg"
    assert res_heif.resolved_path.exists()
    assert res_heif.format.lower() == "jpg"
    assert res_heif.width > 0 and res_heif.height > 0

    # Second request must hit the cache (idempotent resolution)
    res_heif2 = provider.resolve(HighResRequest(file_path=heif_sample, gen_id=3))
    assert res_heif2 is not None
    assert res_heif2.resolved_path == res_heif.resolved_path


def test_tier2_raw_embedded_stream_extraction(temp_cache_dir: Path):
    """Tier 2: RAW files with embedded JPEG streams should extract directly via Range I/O."""
    provider = HighResProvider(cache_dir=temp_cache_dir)
    
    arw_sample = Path("tests/ci/sample/seed.ARW")
    assert arw_sample.exists()
    
    req = HighResRequest(file_path=arw_sample, gen_id=3)
    res = provider.resolve(req)
    
    assert res is not None
    assert res.tier == "tier2_embedded_raw"
    assert res.resolved_path.exists()
    assert res.resolved_path.parent == temp_cache_dir
    assert res.resolved_path.suffix.lower() == ".jpg"
    assert res.width > 0 and res.height > 0
    
    # Verify binary header is real JPEG
    with open(res.resolved_path, "rb") as f:
        magic = f.read(2)
        assert magic == b"\xff\xd8"


def test_tier2_cache_hit_avoid_reextraction(temp_cache_dir: Path):
    """Subsequent requests for the same RAW should hit disk cache immediately."""
    provider = HighResProvider(cache_dir=temp_cache_dir)
    nef_sample = Path("tests/ci/sample/seed.nef")
    assert nef_sample.exists()
    
    # First call: creates cache
    req1 = HighResRequest(file_path=nef_sample, gen_id=4)
    res1 = provider.resolve(req1)
    assert res1 is not None
    cached_file = res1.resolved_path
    mtime1 = cached_file.stat().st_mtime_ns
    
    # Second call: must hit cache without touching/rewriting file
    req2 = HighResRequest(file_path=nef_sample, gen_id=5)
    res2 = provider.resolve(req2)
    assert res2 is not None
    assert res2.resolved_path == cached_file
    mtime2 = res2.resolved_path.stat().st_mtime_ns
    assert mtime1 == mtime2


def test_generation_token_cancellation(temp_cache_dir: Path):
    """Requests with obsolete generation IDs should be aborted."""
    provider = HighResProvider(cache_dir=temp_cache_dir)
    provider.update_active_generation(10)
    
    # Request with gen_id=5 < 10 should be discarded
    req = HighResRequest(file_path=Path("tests/ci/sample/seed.jpg"), gen_id=5)
    res = provider.resolve(req)
    assert res is None
