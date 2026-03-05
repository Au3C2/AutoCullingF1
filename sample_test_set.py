"""
sample_test_set.py — Sample a ~2000-image test set from all valid sessions.

Sampling is done at the **burst-group level** to keep burst groups intact.
Uses SequenceImageNumber from EXIF (Sony A7C II) to identify burst boundaries.

Usage
-----
    python sample_test_set.py --target 2000 --seed 42 --output test_set.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sessions to include (skip ~100% keep-rate sessions)
# ---------------------------------------------------------------------------

_BASE = Path(r"E:\Users\Au3C2\Pictures\2025-03-21-23-F1 2025 中国大奖赛")

_SESSIONS: list[tuple[str, Path]] = [
    ("practice",     _BASE / "2025-03-21 上午 练习赛" / "HIF"),
    ("sprint_quali", _BASE / "2025-03-21 下午 冲刺排位赛" / "HIF"),
    ("sprint_race",  _BASE / "2025-03-22 上午 冲刺赛" / "HIF"),
    ("qualifying",   _BASE / "2025-03-22 下午 排位赛" / "HIF"),
    ("f1_academy2",  _BASE / "2025-03-23 上午 F1 学院 第二回合" / "HIF"),
    ("carrera_cup",  _BASE / "2025-03-23 下午 卡雷拉杯" / "HIF"),
    ("main_race",    _BASE / "2025-03-23 下午 正赛" / "HIF"),
]


# ---------------------------------------------------------------------------
# Burst grouping via EXIF SequenceImageNumber
# ---------------------------------------------------------------------------

def _read_seq_numbers(hif_dir: Path) -> dict[str, int]:
    """Read SequenceImageNumber for all HIF files using exiftool.

    Returns {filename: seq_num} dict.
    """
    files = sorted(hif_dir.glob("*.HIF"))
    if not files:
        return {}

    # Use stdin file-list mode to avoid command-line length limits
    file_list = "\n".join(str(f) for f in files)
    result = subprocess.run(
        ["exiftool", "-SequenceImageNumber", "-FileName", "-json", "-@", "-"],
        input=file_list,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"  WARNING: exiftool failed for {hif_dir}: {result.stderr[:200]}")
        return {}

    import json
    data = json.loads(result.stdout)
    seq_map: dict[str, int] = {}
    for entry in data:
        fname = Path(entry.get("FileName", entry.get("SourceFile", ""))).name
        seq = entry.get("SequenceImageNumber", 1)
        seq_map[fname] = int(seq)
    return seq_map


def _group_bursts(filenames: list[str], seq_map: dict[str, int]) -> list[list[str]]:
    """Group filenames into burst groups using SequenceImageNumber.

    SeqNum == 1 starts a new group.
    """
    groups: list[list[str]] = []
    current: list[str] = []

    for fname in filenames:
        seq = seq_map.get(fname, 1)
        if seq == 1 and current:
            groups.append(current)
            current = []
        current.append(fname)

    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_groups(
    groups: list[list[str]],
    target_count: int,
    rng: random.Random,
) -> list[list[str]]:
    """Randomly sample burst groups until reaching approximately target_count images."""
    shuffled = list(range(len(groups)))
    rng.shuffle(shuffled)

    selected: list[list[str]] = []
    total = 0
    for idx in shuffled:
        g = groups[idx]
        if total + len(g) > target_count * 1.05:  # allow 5% overshoot
            # Try to fit if close enough
            if total >= target_count * 0.95:
                break
        selected.append(g)
        total += len(g)
        if total >= target_count:
            break

    return selected


# ---------------------------------------------------------------------------
# ARW ground truth
# ---------------------------------------------------------------------------

def _find_arw_stems(session_hif_dir: Path) -> set[str]:
    """Find ARW file stems in the parent directory of HIF/."""
    parent = session_hif_dir.parent
    return {f.stem.lower() for f in parent.glob("*.ARW")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample ~2000 images from all sessions for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=int, default=2000, help="Target total images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("test_set.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Gather all sessions
    all_session_data: list[tuple[str, Path, list[list[str]], set[str]]] = []
    total_hif = 0
    for session_id, hif_dir in _SESSIONS:
        if not hif_dir.exists():
            print(f"WARNING: {hif_dir} does not exist, skipping.")
            continue
        files = sorted(f.name for f in hif_dir.glob("*.HIF"))
        print(f"[{session_id}] {len(files)} HIF files, reading EXIF...")
        seq_map = _read_seq_numbers(hif_dir)
        groups = _group_bursts(files, seq_map)
        arw_stems = _find_arw_stems(hif_dir)
        print(f"  {len(groups)} burst groups, {len(arw_stems)} ARW ground truth")
        all_session_data.append((session_id, hif_dir, groups, arw_stems))
        total_hif += len(files)

    print(f"\nTotal valid HIF: {total_hif}")

    # Proportional allocation
    allocations: list[tuple[str, Path, list[list[str]], set[str], int]] = []
    for session_id, hif_dir, groups, arw_stems in all_session_data:
        n_files = sum(len(g) for g in groups)
        target_session = round(n_files / total_hif * args.target)
        allocations.append((session_id, hif_dir, groups, arw_stems, target_session))
        print(f"  {session_id}: {n_files} HIF -> target {target_session}")

    # Sample from each session
    all_rows: list[dict] = []
    for session_id, hif_dir, groups, arw_stems, target_session in allocations:
        selected = _sample_groups(groups, target_session, rng)
        n_selected = sum(len(g) for g in selected)
        n_arw = sum(
            1 for g in selected for f in g if Path(f).stem.lower() in arw_stems
        )
        print(f"  {session_id}: sampled {n_selected} images "
              f"({len(selected)} groups), {n_arw} ARW")

        for g in selected:
            for fname in g:
                has_arw = 1 if Path(fname).stem.lower() in arw_stems else 0
                all_rows.append({
                    "session": session_id,
                    "hif_dir": str(hif_dir),
                    "filename": fname,
                    "has_arw": has_arw,
                })

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session", "hif_dir", "filename", "has_arw"])
        writer.writeheader()
        writer.writerows(all_rows)

    n_total = len(all_rows)
    n_arw = sum(r["has_arw"] for r in all_rows)
    print(f"\nSampled {n_total} images -> {args.output}")
    print(f"  ARW (keep): {n_arw} ({100*n_arw/n_total:.1f}%)")
    print(f"  No-ARW (discard): {n_total - n_arw} ({100*(n_total-n_arw)/n_total:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
