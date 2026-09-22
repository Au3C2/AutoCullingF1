"""Unit tests for automatic cropping logic in cull/cropper.py.
Verifies target aspect ratios, shift-first boundary overflow handling,
and safe margin preservation.
"""

from __future__ import annotations

import pytest
from cull.cropper import calculate_crop


def test_calculate_crop_centered():
    """Test standard centered landscape subject with 3:2 target."""
    # Box: x1=0.3, y1=0.35, x2=0.7, y2=0.65 (cx=0.5, cy=0.5, w=0.4, h=0.3)
    # Visual w = 0.4 * 1.5 = 0.6, visual h = 0.3
    # Target 3:2: expanded visual w = 0.6 * (4/3) = 0.8
    # Visual h = 0.8 / 1.5 = 0.5333...
    # Norm w = 0.8 / 1.5 = 0.5333..., Norm h = 0.5333...
    crop = calculate_crop(0.3, 0.35, 0.7, 0.65, img_ar=1.5)
    assert crop is not None
    top, left, bottom, right = crop

    assert 0.0 <= left < right <= 1.0
    assert 0.0 <= top < bottom <= 1.0
    # Center should remain around 0.5
    assert abs((left + right) / 2.0 - 0.5) < 1e-3
    assert abs((top + bottom) / 2.0 - 0.5) < 1e-3

    # Assert aspect ratio in pixel/visual terms is close to 3:2 (1.5)
    norm_w = right - left
    norm_h = bottom - top
    vis_ar = (norm_w * 1.5) / norm_h
    assert abs(vis_ar - 1.5) < 1e-3


def test_calculate_crop_edge_shift_not_shrink():
    """Test subject near edge: must shift to fit instead of shrinking smaller than detection."""
    # Bottom-right subject: car width=0.40, height=0.18, near bottom and right
    # x1=0.55, y1=0.80, x2=0.95, y2=0.98 (cx=0.75, cy=0.89)
    # The old algorithm forced (cx, cy) to stay at center of crop, so max_hh = min(0.89, 0.11) = 0.11.
    # Because target 3:2 height was ~0.35 (hh=0.175), scale shrank to 0.11/0.175 = 0.628!
    # As a result, crop width became 0.53 * 0.628 = 0.33, strictly CUTTING OFF the 0.40 wide car!
    crop = calculate_crop(0.55, 0.80, 0.95, 0.98, img_ar=1.5)
    assert crop is not None
    top, left, bottom, right = crop

    # All bounds must be within [0, 1]
    assert 0.0 <= left <= 1.0
    assert 0.0 <= right <= 1.0
    assert 0.0 <= top <= 1.0
    assert 0.0 <= bottom <= 1.0

    # The crop MUST strictly contain the detection box (left <= x1, right >= x2, top <= y1, bottom >= y2)
    # unless the detection itself exceeds full frame
    assert left <= 0.55 + 1e-4, f"Crop cut left edge of car: left={left} > x1=0.55"
    assert right >= 0.95 - 1e-4, f"Crop cut right edge of car: right={right} < x2=0.95"
    assert top <= 0.80 + 1e-4, f"Crop cut top edge of car: top={top} > y1=0.80"
    assert bottom >= 0.98 - 1e-4, f"Crop cut bottom edge of car: bottom={bottom} < y2=0.98"



def test_calculate_crop_portrait():
    """Test tall portrait detection box targeting 2:3."""
    # Tall box: x1=0.4, y1=0.2, x2=0.6, y2=0.8 (norm_w = 0.2, norm_h = 0.6)
    # vis_w = 0.2 * 1.5 = 0.3, vis_h = 0.6 -> Portrait box
    crop = calculate_crop(0.4, 0.2, 0.6, 0.8, img_ar=1.5)
    assert crop is not None
    top, left, bottom, right = crop

    norm_w = right - left
    norm_h = bottom - top
    vis_ar = (norm_w * 1.5) / norm_h
    # Target 2:3 = 0.6666...
    assert abs(vis_ar - (2.0 / 3.0)) < 1e-3


def test_calculate_crop_invalid():
    """Invalid box dimensions return None."""
    assert calculate_crop(0.5, 0.5, 0.5, 0.5) is None
    assert calculate_crop(0.6, 0.5, 0.4, 0.7) is None
