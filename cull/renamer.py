"""
renamer.py — Rename photo files based on EXIF capture time.

Supported format: IMG_YYYYMMDD_HHMMSS_MS.ext
Example: IMG_20240316_213037_123.jpg
"""

import logging
from datetime import datetime
from pathlib import Path
try:
    from .exif_reader import read_exif, ExifData
except (ImportError, ValueError):
    from exif_reader import read_exif, ExifData

log = logging.getLogger(__name__)

def setup_file_logging(log_dir: Path, name: str = "renamer"):
    """Setup logging to both console and a file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    root_log = logging.getLogger()
    # Clear existing handlers to prevent duplicates
    for handler in root_log.handlers[:]:
        root_log.removeHandler(handler)
        
    root_log.setLevel(logging.INFO)
    root_log.addHandler(file_handler)
    root_log.addHandler(console_handler)
    
    # Force real-time output by flushing the console handler
    console_handler.flush()
    
    log.info("Logging to %s", log_file)
    return log_file

def generate_new_name(exif: ExifData) -> str:
    """Generate a new filename based on EXIF timestamp."""
    dt = exif.datetime_original
    if not dt:
        return ""
    
    # Format: IMG_YYYYMMDD_HHMMSS
    base = dt.strftime("IMG_%Y%m%d_%H%M%S")
    
    # Add milliseconds (microsecond / 1000)
    ms = dt.microsecond // 1000
    return f"{base}_{ms:03d}{exif.path.suffix.lower()}"

def rename_images(paths: list[Path], dry_run: bool = False) -> dict[Path, Path]:
    """Rename a list of image files based on their EXIF timestamps.
    
    Returns a mapping of {original_path: new_path}.
    """
    if not paths:
        return {}

    log.info("Reading EXIF for renaming...")
    exif_list = read_exif(paths)
    
    path_map: dict[Path, Path] = {}
    used_names: set[str] = set()
    
    for exif in exif_list:
        old_path = exif.path
        new_name = generate_new_name(exif)
        
        if not new_name:
            log.warning("Could not determine capture time for %s, skipping rename", old_path.name)
            path_map[old_path] = old_path
            continue
            
        # Handle collisions (standard: append _1, _2...)
        stem = Path(new_name).stem
        ext = Path(new_name).suffix
        candidate = new_name
        counter = 1
        while candidate.lower() in used_names:
            candidate = f"{stem}_{counter}{ext}"
            counter += 1
        
        new_path = old_path.parent / candidate
        used_names.add(candidate.lower())
        path_map[old_path] = new_path

    # Execute move
    results: dict[Path, Path] = {}
    total = len(path_map)
    for i, (old_path, new_path) in enumerate(path_map.items(), 1):
        if i % 500 == 0 or i == total:
            log.info("  Renaming progress: %d/%d images...", i, total)
            
        if old_path == new_path:
            results[old_path] = old_path
            continue
            
        # Check for .xmp sidecar
        old_xmp = old_path.with_suffix(".xmp")
        has_xmp = old_xmp.exists()
        
        if dry_run:
            log.info("[dry-run] Rename: %s -> %s", old_path.name, new_path.name)
            if has_xmp:
                log.info("[dry-run] Rename XMP: %s -> %s", old_xmp.name, new_path.with_suffix(".xmp").name)
            results[old_path] = new_path
        else:
            try:
                # Use rename, overwrite if target exists (though we handle collisions)
                old_path.rename(new_path)
                log.debug("Renamed: %s -> %s", old_path.name, new_path.name)
                
                if has_xmp:
                    new_xmp = new_path.with_suffix(".xmp")
                    try:
                        old_xmp.rename(new_xmp)
                        log.debug("Renamed XMP: %s -> %s", old_xmp.name, new_xmp.name)
                    except Exception as e:
                        log.error("Failed to rename XMP %s: %s", old_xmp.name, e)
                
                results[old_path] = new_path
            except Exception as e:
                log.error("Failed to rename %s: %s", old_path.name, e)
                results[old_path] = old_path
                
    return results

if __name__ == "__main__":
    # Can be run as a standalone script
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Rename images by EXIF timestamp.")
    parser.add_argument("dir", type=Path, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't move files")
    args = parser.parse_args()
    
    if not args.dir.is_dir():
        print(f"Error: {args.dir} is not a directory")
        sys.exit(1)
        
    log_file = setup_file_logging(args.dir / "logs", name="renamer")
    
    # Simple collection logic for standalone use
    exts = {".jpg", ".jpeg", ".hif", ".heic", ".nef", ".arw", ".dng"}
    paths = [p for p in args.dir.iterdir() if p.suffix.lower() in exts]
    
    if args.dry_run:
        log.info("--- DRY RUN MODE (no files will be moved) ---")
        
    rename_images(paths, dry_run=args.dry_run)
    log.info("Done. Log saved to %s", log_file)
