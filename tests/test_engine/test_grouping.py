"""
tests/test_engine/test_grouping.py — Unit tests for burst grouping logic in cull/exif_reader.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from cull.exif_reader import ExifData, group_bursts


def _make_exif(
    name: str,
    dt: datetime | None = None,
    seq: int | None = None,
    burst_id: int | None = None,
    release_mode: str | None = None,
    shutter_count: int | None = None,
) -> ExifData:
    return ExifData(
        path=Path(name),
        datetime_original=dt,
        sequence_image_number=seq,
        burst_group_id=burst_id,
        release_mode=release_mode,
        shutter_count=shutter_count,
    )


def test_empty_list():
    assert group_bursts([]) == []


def test_single_shot_guard():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    files = [
        _make_exif("a.jpg", dt=t0, release_mode="Single"),
        _make_exif("b.jpg", dt=t0 + timedelta(milliseconds=100), release_mode="Single"),
        _make_exif("c.jpg", dt=t0 + timedelta(milliseconds=200), release_mode="Continuous"),
    ]
    groups = group_bursts(files)
    assert len(groups) == 3
    assert all(not g.is_burst for g in groups)


def test_nikon_burst_group_id():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    files = [
        _make_exif("n1.nef", dt=t0, burst_id=101),
        _make_exif("n2.nef", dt=t0 + timedelta(seconds=5), burst_id=101),
        _make_exif("n3.nef", dt=t0 + timedelta(seconds=6), burst_id=102),
    ]
    groups = group_bursts(files)
    assert len(groups) == 2
    assert [p.name for p in groups[0].frames] == ["n1.nef", "n2.nef"]
    assert groups[0].is_burst is True
    assert [p.name for p in groups[1].frames] == ["n3.nef"]
    assert groups[1].is_burst is False


def test_sony_sequence_and_missing_first_frame():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    # Burst 1 has seq 1, 2, 3
    # Burst 2 has first frame deleted (starts at seq 2, 3)
    files = [
        _make_exif("s1.arw", dt=t0, seq=1),
        _make_exif("s2.arw", dt=t0 + timedelta(milliseconds=100), seq=2),
        _make_exif("s3.arw", dt=t0 + timedelta(milliseconds=200), seq=3),
        _make_exif("s4.arw", dt=t0 + timedelta(milliseconds=300), seq=2),  # Missing seq=1, but 2 <= 3 -> split!
        _make_exif("s5.arw", dt=t0 + timedelta(milliseconds=400), seq=3),
    ]
    groups = group_bursts(files)
    assert len(groups) == 2
    assert [p.name for p in groups[0].frames] == ["s1.arw", "s2.arw", "s3.arw"]
    assert [p.name for p in groups[1].frames] == ["s4.arw", "s5.arw"]
    assert groups[0].is_burst is True
    assert groups[1].is_burst is True


def test_sony_shutter_count_discontinuity():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    # seq jumps by 1 (2 -> 3), but shutter count jumps by 5 -> split
    files = [
        _make_exif("s1.arw", dt=t0, seq=2, shutter_count=1000),
        _make_exif("s2.arw", dt=t0 + timedelta(milliseconds=100), seq=3, shutter_count=1005),
    ]
    groups = group_bursts(files)
    assert len(groups) == 2
    assert groups[0].frames == [Path("s1.arw")]
    assert groups[1].frames == [Path("s2.arw")]


def test_sony_continuous_gap_release():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    # Both in same sequence numbering seq=1, seq=2, but gap is 1.5s (> 0.8s) -> photographer released shutter
    files = [
        _make_exif("s1.arw", dt=t0, seq=1),
        _make_exif("s2.arw", dt=t0 + timedelta(seconds=1.5), seq=2),
    ]
    groups = group_bursts(files)
    assert len(groups) == 2


def test_generic_fallback_gap():
    t0 = datetime(2026, 3, 20, 10, 0, 0)
    files = [
        _make_exif("g1.jpg", dt=t0),
        _make_exif("g2.jpg", dt=t0 + timedelta(seconds=1.2)),
        _make_exif("g3.jpg", dt=t0 + timedelta(seconds=3.5)),
    ]
    groups = group_bursts(files)
    assert len(groups) == 2
    assert [p.name for p in groups[0].frames] == ["g1.jpg", "g2.jpg"]
    assert groups[0].is_burst is True
    assert [p.name for p in groups[1].frames] == ["g3.jpg"]
    assert groups[1].is_burst is False
