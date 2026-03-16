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
      xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">
      <xmp:Rating>{rating}</xmp:Rating>
{pick_tag}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_xmp(image_path: Path, rating: int, overwrite: bool = True) -> Path:
    """Write an XMP sidecar file next to *image_path*.

    Parameters
    ----------
    image_path:
        Path to the original image file.  The sidecar is written alongside it.
    rating:
        Lightroom Rating value: -1 (reject) or 1-5 (stars).
    overwrite:
        When ``True`` (default), overwrite any existing sidecar.
        When ``False``, skip writing if the sidecar already exists.

    Returns
    -------
    Path
        Path to the written (or pre-existing) ``.xmp`` file.

    Raises
    ------
    ValueError
        If *rating* is not in {-1, 1, 2, 3, 4, 5}.
    """
    if rating not in (-1, 1, 2, 3, 4, 5):
        raise ValueError(f"Invalid rating {rating!r}; must be -1 or 1-5.")

    xmp_path = image_path.with_suffix(".xmp")

    if xmp_path.exists() and not overwrite:
        log.debug("Skipping existing sidecar: %s", xmp_path)
        return xmp_path

    pick_tag = '      <xmpDM:pick>-1</xmpDM:pick>' if rating == -1 else ''
    content = _XMP_TEMPLATE.format(rating=rating, pick_tag=pick_tag)
    # Remove empty line if pick_tag is empty
    if not pick_tag:
        content = content.replace("      <xmp:Rating>{rating}</xmp:Rating>\n\n", f"      <xmp:Rating>{rating}</xmp:Rating>\n")

    xmp_path.write_text(content, encoding="utf-8")
    log.debug("Wrote %s  (Rating=%d)", xmp_path.name, rating)
    return xmp_path


def write_xmp_batch(
    results: list[tuple[Path, int]],
    overwrite: bool = True,
    dry_run: bool = False,
) -> list[Path]:
    """Write XMP sidecars for multiple images.

    Parameters
    ----------
    results:
        List of (image_path, rating) tuples.
    overwrite:
        Passed through to :func:`write_xmp`.
    dry_run:
        If ``True``, log what would be written but do not create any files.

    Returns
    -------
    list[Path]
        Paths of sidecars that were written (or would be written in dry-run).
    """
    written: list[Path] = []
    for image_path, rating in results:
        xmp_path = image_path.with_suffix(".xmp")
        if dry_run:
            log.info("[dry-run] Would write %s  (Rating=%d)", xmp_path, rating)
            written.append(xmp_path)
        else:
            try:
                written.append(write_xmp(image_path, rating, overwrite=overwrite))
            except Exception as exc:
                log.error("Failed to write sidecar for %s: %s", image_path.name, exc)
    return written
