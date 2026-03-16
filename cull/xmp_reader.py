"""
xmp_reader.py — Read Lightroom rating and pick metadata from XMP sidecar files.
"""

import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Basic regex to extract Rating and Pick flag from XMP
# We use regex instead of full XML parser for speed and to handle slightly malformed XMP
_RATING_RE = re.compile(r'<xmp:Rating>([-0-9]+)</xmp:Rating>')
_PICK_RE = re.compile(r'<xmpDM:pick>([-0-9]+)</xmpDM:pick>')

def read_xmp_rating(image_path: Path) -> tuple[int | None, int | None]:
    """Read rating and pick status from the corresponding XMP sidecar.
    
    Returns:
        (rating, pick) where rating is 0-5 or -1, and pick is 1 (picked), -1 (rejected), 0/None (none).
        Returns (None, None) if XMP doesn't exist or doesn't contain valid data.
    """
    xmp_path = image_path.with_suffix(".xmp")
    if not xmp_path.exists():
        return None, None
    
    try:
        content = xmp_path.read_text(encoding="utf-8", errors="ignore")
        
        rating_match = _RATING_RE.search(content)
        pick_match = _PICK_RE.search(content)
        
        rating = int(rating_match.group(1)) if rating_match else None
        pick = int(pick_match.group(1)) if pick_match else None
        
        return rating, pick
    except Exception as e:
        log.debug("Failed to read XMP %s: %s", xmp_path, e)
        return None, None
