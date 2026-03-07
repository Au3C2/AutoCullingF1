"""
make_preview.py — Build a preview directory tree with hardlinks.

Reads scores_preview.csv and test_set.csv, then creates:

  preview/
    {session}/
      kept/
        {filename}   <- hardlink to original HIF
      rejected/
        {filename}   <- hardlink to original HIF

Hardlinks have zero disk cost (no copy), and are fully transparent
to any viewer, Lightroom, etc.

Usage:
    python make_preview.py [--scores scores_preview.csv] [--out preview]
"""

import argparse
import csv
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scores", type=Path, default=Path("scores_preview.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("test_set.csv"))
    parser.add_argument("--out", type=Path, default=Path("preview"))
    args = parser.parse_args()

    # Build path map: filename → absolute HIF path
    log.info("Reading test set paths...")
    path_map: dict[str, Path] = {}
    with open(args.test_set, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path_map[row["filename"]] = Path(row["hif_dir"]) / row["filename"]

    # Read scores CSV
    log.info("Reading scores from %s ...", args.scores)
    rows = []
    with open(args.scores, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Remove old preview tree if exists
    if args.out.exists():
        import shutil
        shutil.rmtree(args.out)
        log.info("Removed old preview/ directory")

    kept = rejected = skipped = 0

    for row in rows:
        session = row["session"]
        filename = row["filename"]
        is_kept = int(row["rating"]) > 0

        src = path_map.get(filename)
        if src is None or not src.exists():
            skipped += 1
            continue

        subdir = "kept" if is_kept else "rejected"
        dst_dir = args.out / session / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Use .heif instead of original extension
        dst = dst_dir / Path(filename).with_suffix(".heif").name

        if dst.exists():
            dst.unlink()

        # Handle old DSC*.HIF if they exist (to clean up if previously run)
        old_dst = dst_dir / filename
        if old_dst.exists() and old_dst != dst:
            old_dst.unlink()

        try:
            os.link(src, dst)
            if is_kept:
                kept += 1
            else:
                rejected += 1
        except OSError as e:
            # Fallback: if hardlink fails cross-device, just copy
            log.warning("Hardlink failed (%s), copying instead...", e)
            import shutil
            shutil.copy2(src, dst)
            if is_kept:
                kept += 1
            else:
                rejected += 1

    log.info("Done! kept=%d  rejected=%d  skipped=%d", kept, rejected, skipped)
    log.info("Preview directory: %s", args.out.resolve())

    # Per-session summary
    sessions: dict[str, dict] = {}
    for row in rows:
        s = row["session"]
        if s not in sessions:
            sessions[s] = {"kept": 0, "rejected": 0}
        if int(row["rating"]) > 0:
            sessions[s]["kept"] += 1
        else:
            sessions[s]["rejected"] += 1

    print("\nPer-session breakdown:")
    print(f"  {'Session':<20} {'Kept':>6} {'Rejected':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*9}")
    for s, counts in sorted(sessions.items()):
        print(f"  {s:<20} {counts['kept']:>6} {counts['rejected']:>9}")


if __name__ == "__main__":
    main()
