"""
xmp_writer.py — Write Lightroom-compatible XMP sidecar files.

Output format
-------------
Each image gets a same-directory, same-stem ``.xmp`` file, e.g.:
  DSC00827.HIF  →  DSC00827.xmp

The ``xmp:Rating`` field encodes the Lightroom star/reject value:
  -1  → Rejected (red X flag in Lightroom)
   1  → 1 star
   2  → 2 stars
   3  → 3 stars
   4  → 4 stars
   5  → 5 stars

Lightroom picks up these sidecars automatically when importing the original
RAW/HIF/NEF files from the same directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XMP template
# ---------------------------------------------------------------------------

_XMP_TEMPLATE = """\
<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      xmp:Rating="{rating}"
      xmpDM:pick="{pick}"
      {crop_attrs}>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_xmp(image_path: Path, rating: int, overwrite: bool = True, crop: tuple[float, float, float, float] | None = None) -> Path:
    """Write an XMP sidecar file next to *image_path*.

    Parameters
    ----------
    image_path:
        Path to the original image file.
    rating:
        Lightroom Rating value: -1 (reject) or 1-5 (stars).
    overwrite:
        When ``True`` (default), overwrite.
    crop:
        Optional (top, left, bottom, right) normalized coordinates.
    """
    if rating not in (-1, 0, 1, 2, 3, 4, 5):
        raise ValueError(f"Invalid rating {rating!r}; must be -1, 0 or 1-5.")

    xmp_path = image_path.with_suffix(".xmp")

    if xmp_path.exists() and not overwrite:
        log.debug("Skipping existing sidecar: %s", xmp_path)
        return xmp_path

    pick = "-1" if rating == -1 else "0"
    
    crop_attrs = ""
    if crop:
        t, l, b, r = crop
        crop_attrs = (
            f'crs:HasCrop="True" '
            f'crs:AlreadyApplied="False" '
            f'crs:CropTop="{t:.6f}" '
            f'crs:CropLeft="{l:.6f}" '
            f'crs:CropBottom="{b:.6f}" '
            f'crs:CropRight="{r:.6f}" '
            f'crs:CropAngle="0" '
            f'crs:CropConstrainToWarp="0" '
            f'crs:CropConstrainToUnitSquare="1"'
        )

    content = _XMP_TEMPLATE.format(rating=rating, pick=pick, crop_attrs=crop_attrs)
    
    # Cleanup empty lines and redundant spaces
    lines = [line for line in content.splitlines() if line.strip()]
    content = "\n".join(lines)

    xmp_path.write_text(content, encoding="utf-8")
    log.debug("Wrote %s  (Rating=%d, Crop=%s)", xmp_path.name, rating, "Yes" if crop else "No")
    return xmp_path


def write_xmp_batch(
    results: list[tuple[Path, int] | tuple[Path, int, tuple[float, float, float, float] | None]],
    overwrite: bool = True,
    dry_run: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for item in results:
        image_path = item[0]
        rating = item[1]
        crop = item[2] if len(item) > 2 else None
        
        xmp_path = image_path.with_suffix(".xmp")
        if dry_run:
            log.info("[dry-run] Would write %s  (Rating=%d, Crop=%s)", 
                     xmp_path, rating, "Yes" if crop else "No")
            written.append(xmp_path)
        else:
            try:
                written.append(write_xmp(image_path, rating, overwrite=overwrite, crop=crop))
            except Exception as exc:
                log.error("Failed to write sidecar for %s: %s", image_path.name, exc)
    return written
