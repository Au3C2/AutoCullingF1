"""Scan all XMP sidecar files for user-annotated ratings and picks."""
import os
import re
from collections import Counter

base_dir = r"E:\Users\Au3C2\Pictures\2025-03-21-23-F1 2025 中国大奖赛"

sessions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

rating_counts: Counter = Counter()
pick_counts: Counter = Counter()
total_xmp = 0
has_rating = 0

# Collect per-file data for later use
all_data: list[dict] = []

for sess in sorted(sessions):
    sess_path = os.path.join(base_dir, sess)
    xmp_files = [f for f in os.listdir(sess_path) if f.endswith(".xmp")]

    sess_ratings: Counter = Counter()
    sess_picks: Counter = Counter()

    for xf in xmp_files:
        total_xmp += 1
        path = os.path.join(sess_path, xf)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        rating_val = 0
        pick_val = None

        # xmp:Rating="N"
        m = re.search(r'xmp:Rating\s*=\s*"(-?\d+)"', content)
        if m:
            rating_val = int(m.group(1))

        # xmpDM:pick="N"
        m2 = re.search(r'xmpDM:pick\s*=\s*"(-?\d+)"', content)
        if m2:
            pick_val = int(m2.group(1))
            pick_counts[pick_val] += 1
            sess_picks[pick_val] += 1

        if rating_val > 0:
            has_rating += 1
            rating_counts[rating_val] += 1
            sess_ratings[rating_val] += 1

        all_data.append({
            "session": sess,
            "filename": xf,
            "stem": os.path.splitext(xf)[0],
            "rating": rating_val,
            "pick": pick_val,
        })

    if xmp_files:
        print(f"{sess}: {len(xmp_files)} xmp files")
        if sess_ratings:
            print(f"  Ratings: {dict(sorted(sess_ratings.items()))}")
        if sess_picks:
            print(f"  Picks: {dict(sorted(sess_picks.items()))}")

print(f"\n=== Total XMP Summary ===")
print(f"Total XMP files: {total_xmp}")
print(f"With Rating>0: {has_rating}")
print(f"Rating distribution: {dict(sorted(rating_counts.items()))}")
print(f"Pick distribution: {dict(sorted(pick_counts.items()))}")

# Also check if ARW files have embedded ratings
print("\n\n=== Checking ARW files for embedded ratings ===")
import subprocess

# Sample a few ARW files from different sessions
for sess in sorted(sessions)[:3]:
    sess_path = os.path.join(base_dir, sess)
    arw_files = [f for f in os.listdir(sess_path) if f.endswith(".ARW")][:3]
    for af in arw_files:
        path = os.path.join(sess_path, af)
        result = subprocess.run(
            ["exiftool", "-Rating", "-RatingPercent", "-s2", "-n", path],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout.strip()
        if out:
            print(f"  {sess}/{af}: {out}")

# Check HIF files too
print("\n=== Checking HIF files for embedded ratings ===")
for sess in sorted(sessions)[:3]:
    sess_path = os.path.join(base_dir, sess)
    hif_dir = os.path.join(sess_path, "HIF")
    if not os.path.isdir(hif_dir):
        continue
    hif_files = [f for f in os.listdir(hif_dir) if f.endswith(".HIF")][:3]
    for hf in hif_files:
        path = os.path.join(hif_dir, hf)
        result = subprocess.run(
            ["exiftool", "-Rating", "-RatingPercent", "-s2", "-n", path],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout.strip()
        if out:
            print(f"  {sess}/HIF/{hf}: {out}")
