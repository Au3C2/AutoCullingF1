"""
cropper.py — Automatic cropping logic based on F1 car detections.
"""

import logging
from pathlib import Path
import re

log = logging.getLogger(__name__)

def calculate_crop(x1: float, y1: float, x2: float, y2: float, img_ar: float = 1.5) -> tuple[float, float, float, float] | None:
    """
    Calculate normalized crop coordinates (top, left, bottom, right) based on a detection box.
    
    Parameters:
        x1, y1, x2, y2: Normalized detection coordinates (0-1).
        img_ar: Aspect ratio of the original image (Width/Height). Defaults to 3:2 (1.5).
    """
    # Sanitize and clamp input coordinates to [0.0, 1.0]
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))

    w_box_norm = x2 - x1
    h_box_norm = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    if w_box_norm <= 1e-4 or h_box_norm <= 1e-4:
        return None

    # Convert normalized box dimensions to "visual" dimensions based on image AR
    # to determine if the box is landscape or portrait.
    w_box_vis = w_box_norm * img_ar
    h_box_vis = h_box_norm * 1.0 # reference
    
    if w_box_vis >= h_box_vis:
        # Landscape Detection Box -> Target 3:2
        target_ar = 3.0 / 2.0
        # Expand width by 1/6 on each side (total 4/3)
        w_new_vis = w_box_vis * (4.0 / 3.0)
        h_new_vis = w_new_vis / target_ar
    else:
        # Portrait Detection Box -> Target 2:3
        target_ar = 2.0 / 3.0
        # Expand height by 1/6 on each side (total 4/3)
        h_new_vis = h_box_vis * (4.0 / 3.0)
        w_new_vis = h_new_vis * target_ar

    # Convert visual dimensions back to normalized units
    w_new_norm = w_new_vis / img_ar
    h_new_norm = h_new_vis / 1.0

    # Ensure crop box size is not larger than full image in visual space
    # (i.e. scale down if it exceeds the entire frame dimensions)
    max_w_norm = 1.0
    max_h_norm = 1.0
    scale = 1.0
    if w_new_norm > max_w_norm:
        scale = min(scale, max_w_norm / w_new_norm)
    if h_new_norm > max_h_norm:
        scale = min(scale, max_h_norm / h_new_norm)

    w_final_norm = w_new_norm * scale
    h_final_norm = h_new_norm * scale

    # Shift-first bounds resolution:
    # Instead of forcing (cx, cy) to stay strictly at the center of the crop,
    # shift the center inside [hw, 1.0 - hw] and [hh, 1.0 - hh] to prevent clipping edges.
    hw = w_final_norm / 2.0
    hh = h_final_norm / 2.0

    cx_shifted = max(hw, min(1.0 - hw, cx)) if hw <= 0.5 else 0.5
    cy_shifted = max(hh, min(1.0 - hh, cy)) if hh <= 0.5 else 0.5

    left = cx_shifted - hw
    right = cx_shifted + hw
    top = cy_shifted - hh
    bottom = cy_shifted + hh

    # Safety margin guard: ensure detection box is strictly covered
    # If detection box boundary slightly exceeds because of extreme edge position,
    # clamp while maintaining valid box.
    left = min(left, x1)
    right = max(right, x2)
    top = min(top, y1)
    bottom = max(bottom, y2)

    # Final clamping to valid image coordinate bounds [0.0, 1.0]
    left = max(0.0, min(1.0, left))
    right = max(0.0, min(1.0, right))
    top = max(0.0, min(1.0, top))
    bottom = max(0.0, min(1.0, bottom))

    if left >= right or top >= bottom:
        return None
    
    return (top, left, bottom, right)

def has_crop_info(xmp_path: Path) -> bool:
    """Check if the XMP already contains crop information."""
    if not xmp_path.exists():
        return False
    try:
        content = xmp_path.read_text(encoding="utf-8", errors="ignore")
        return 'crs:HasCrop="True"' in content or '<crs:HasCrop>True</crs:HasCrop>' in content
    except Exception:
        return False

def update_xmp_with_crop(xmp_path: Path, crop: tuple[float, float, float, float]):
    """Update an existing XMP file with crop information as attributes."""
    top, left, bottom, right = crop
    
    if not xmp_path.exists():
        return

    try:
        content = xmp_path.read_text(encoding="utf-8", errors="ignore")
        
        # 1. Identify the tag where attributes live (rdf:Description)
        # We need to find the main <rdf:Description ... > tag.
        # It might span multiple lines.
        
        # Clean up ANY existing crop fields first (to avoid duplicates)
        fields = ["CropTop", "CropLeft", "CropBottom", "CropRight", "HasCrop", "CropAngle", "AlreadyApplied",
                  "CropConstrainToWarp", "CropConstrainToUnitSquare"]
        for f in fields:
            content = re.sub(fr'\s*crs:{f}="[^"]*"', '', content)
            content = re.sub(fr'\s*<crs:{f}>[^<]*</crs:{f}>', '', content)

        # 2. Add crs namespace if missing
        if 'xmlns:crs=' not in content:
            content = content.replace('<rdf:Description', '<rdf:Description\n    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"')
        
        # 3. Prepare new attribute string
        new_attr_str = (
            f'\n   crs:HasCrop="True"'
            f'\n   crs:CropTop="{top:.6f}"\n   crs:CropLeft="{left:.6f}"'
            f'\n   crs:CropBottom="{bottom:.6f}"\n   crs:CropRight="{right:.6f}"'
            f'\n   crs:CropAngle="0"\n   crs:AlreadyApplied="False"'
            f'\n   crs:CropConstrainToWarp="0"\n   crs:CropConstrainToUnitSquare="1"'
        )
        
        # 4. Insert into the first <rdf:Description tag
        # We look for the first occurance of <rdf:Description and insert after it
        content = re.sub(r'(<rdf:Description)', fr'\1 {new_attr_str}', content, count=1)
            
        xmp_path.write_text(content, encoding="utf-8")
        log.debug("Updated crop attributes in %s", xmp_path.name)
        
    except Exception as e:
        log.error("Failed to update XMP crop for %s: %s", xmp_path, e)
