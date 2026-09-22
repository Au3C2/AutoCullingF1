"""Unit tests for XMP sidecar generation and DOM incremental updates in cull/xmp_writer.py.
Verifies Lightroom-compatible schema, preserving existing tags, and batch writes.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from cull.xmp_writer import write_xmp, write_xmp_batch
from cull.xmp_reader import read_xmp_rating


def test_write_xmp_new_file(tmp_path: Path):
    """Test creating a fresh XMP file next to an image."""
    img_path = tmp_path / "DSC01001.ARW"
    img_path.touch()

    xmp_path = write_xmp(img_path, rating=4, crop=(0.1, 0.15, 0.8, 0.85))
    assert xmp_path.exists()
    assert xmp_path == tmp_path / "DSC01001.xmp"

    # Verify via reader
    rating, pick = read_xmp_rating(img_path)
    assert rating == 4
    assert pick == 0

    # Parse XML and check tags
    content = xmp_path.read_text(encoding="utf-8")
    assert 'xmp:Rating="4"' in content or '<xmp:Rating>4</xmp:Rating>' in content
    assert 'crs:HasCrop="True"' in content
    assert 'crs:CropTop="0.100000"' in content


def test_write_xmp_incremental_preserves_existing_metadata(tmp_path: Path):
    """Test that existing metadata (Lightroom presets, copyright, etc.) is preserved."""
    img_path = tmp_path / "DSC02002.CR3"
    img_path.touch()
    xmp_path = tmp_path / "DSC02002.xmp"

    # Pre-populate an existing XMP with complex Lightroom metadata
    existing_xml = """<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmp:Rating="2"
      crs:Exposure2012="+0.50"
      crs:Contrast2012="+10">
      <dc:creator>
        <rdf:Seq>
          <rdf:li>Pro Motorsport Photo</rdf:li>
        </rdf:Seq>
      </dc:creator>
      <dc:rights>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">All Rights Reserved</rdf:li>
        </rdf:Alt>
      </dc:rights>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""
    xmp_path.write_text(existing_xml, encoding="utf-8")

    # Update rating to 5 with crop
    write_xmp(img_path, rating=5, crop=(0.05, 0.1, 0.95, 0.9))

    updated_content = xmp_path.read_text(encoding="utf-8")

    # Verify updated fields
    rating, _ = read_xmp_rating(img_path)
    assert rating == 5
    assert 'crs:HasCrop="True"' in updated_content

    # Crucial check: verify original custom metadata was NOT blown away
    assert "Pro Motorsport Photo" in updated_content
    assert "All Rights Reserved" in updated_content
    assert 'crs:Exposure2012="+0.50"' in updated_content
    assert 'crs:Contrast2012="+10"' in updated_content


def test_write_xmp_batch(tmp_path: Path):
    """Test batch write functionality."""
    img1 = tmp_path / "shot1.ARW"
    img2 = tmp_path / "shot2.ARW"
    img1.touch()
    img2.touch()

    items = [
        (img1, 3, (0.0, 0.0, 1.0, 1.0)),
        (img2, -1, None),
    ]

    written = write_xmp_batch(items)
    assert len(written) == 2

    r1, p1 = read_xmp_rating(img1)
    assert r1 == 3
    assert p1 == 0

    r2, p2 = read_xmp_rating(img2)
    assert r2 == -1
    assert p2 == -1
