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
    w_box_norm = x2 - x1
    h_box_norm = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    if w_box_norm <= 0 or h_box_norm <= 0:
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

    # Bounds check and shrink-to-fit while maintaining aspect ratio
    # We want to keep (cx, cy) fixed if possible.
    
    # Calculate half spans in normalized units
    hw = w_new_norm / 2
    hh = h_new_norm / 2
    
    # Check if we exceed image bounds relative to center
    max_hw = min(cx, 1.0 - cx)
    max_hh = min(cy, 1.0 - cy)
    
    # If the requested span is too large, we must shrink to fit the constraints
    scale = 1.0
    if hw > max_hw:
        scale = min(scale, max_hw / hw)
    if hh > max_hh:
        scale = min(scale, max_hh / hh)
        
    w_final_norm = w_new_norm * scale
    h_final_norm = h_new_norm * scale
    
    left = cx - w_final_norm / 2
    right = cx + w_final_norm / 2
    top = cy - h_final_norm / 2
    bottom = cy + h_final_norm / 2
    
    # Final clamping just for safety
    left = max(0.0, min(1.0, left))
    right = max(0.0, min(1.0, right))
    top = max(0.0, min(1.0, top))
    bottom = max(0.0, min(1.0, bottom))
    
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
