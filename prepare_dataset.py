"""
Dataset preparation script for F1 motorsport photo auto-culling.

This script organises raw image files (HIF/HEIF) from date-named session
subdirectories under `dataset/`, renames them using EXIF datetime
(format: yyyymmdd_hhmmss[_N]), moves them to `dataset/img/`, assigns binary
labels (1 = kept / selected ARW exists, 0 = not selected), writes metadata
to `data_info.csv`, and splits the labelled set into train/test CSVs using
a stratified split to handle class imbalance.

Usage
-----
    uv run prepare_dataset.py [--dataset-dir PATH] [--exiftool PATH]
                              [--test-size FLOAT] [--seed INT]
                              [--move] [--dry-run]

Requirements
------------
    exiftool must be available (pass path via --exiftool if not on PATH).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported image extensions inside the HIF sub-folder (case-insensitive)
HIF_EXTENSIONS: frozenset[str] = frozenset({".hif", ".heif"})

# EXIF tags to extract (in priority order for the datetime field)
DATETIME_TAGS: list[str] = [
    "DateTimeOriginal",
    "CreateDate",
    "FileModifyDate",
]

# Additional EXIF tags to store in the CSV
EXTRA_TAGS: list[str] = [
    "Make",
    "Model",
    "LensModel",
    "FocalLength",
    "FNumber",
    "ExposureTime",
    "ISO",
    "ShutterSpeedValue",
    "SubSecTimeOriginal",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_exiftool(hint: str | None) -> Path:
    """Locate the exiftool executable.

    Parameters
    ----------
    hint:
        Optional explicit path supplied via CLI argument.

    Returns
    -------
    Path
        Resolved path to an executable exiftool binary.

    Raises
    ------
    FileNotFoundError
        If exiftool cannot be located.
    """
    # Candidate paths to probe, in priority order
    candidates: list[Path] = []

    if hint:
        candidates.append(Path(hint))
    else:
        found = shutil.which("exiftool")
        if found:
            candidates.append(Path(found))
        # Well-known fallback locations (dev/CI setup)
        candidates += [
            Path("/tmp/Image-ExifTool-13.52/exiftool"),
            Path("/tmp/Image-ExifTool-13.10/exiftool"),
        ]

    for candidate in candidates:
        if not candidate.is_file():
            continue
        # Verify the candidate actually works by calling -ver
        try:
            result = subprocess.run(
                [str(candidate), "-ver"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return candidate
        except (subprocess.SubprocessError, OSError):
            continue

    if hint:
        raise FileNotFoundError(f"Specified exiftool not found or not working: {hint}")
    raise FileNotFoundError(
        "exiftool not found. Install it (e.g. apt install libimage-exiftool-perl) "
        "or pass --exiftool <path>."
    )


def _run_exiftool(exiftool: Path, paths: list[Path]) -> list[dict]:
    """Run exiftool in batch JSON mode on a list of files.

    Parameters
    ----------
    exiftool:
        Path to the exiftool binary.
    paths:
        Files to process.

    Returns
    -------
    list[dict]
        One dict per file as returned by ``exiftool -json``.
    """
    tags = DATETIME_TAGS + EXTRA_TAGS
    tag_args = [f"-{t}" for t in tags]
    cmd = [str(exiftool), "-json", "-n"] + tag_args + [str(p) for p in paths]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _parse_datetime(record: dict) -> str | None:
    """Extract a yyyymmdd_hhmmss string from an exiftool record.

    Tries tags in *DATETIME_TAGS* order.  Accepts the formats:
    ``YYYY:MM:DD HH:MM:SS`` and ``YYYY:MM:DD HH:MM:SS.sss±HH:MM``.

    Returns ``None`` if no usable tag is found.
    """
    for tag in DATETIME_TAGS:
        raw: str = record.get(tag, "")
        if not raw:
            continue
        # Strip sub-second and timezone parts
        raw = raw.split(".")[0].split("+")[0].split("-")[0].strip()
        parts = raw.split(" ")
        if len(parts) < 2:
            continue
        date_part = parts[0].replace(":", "")
        time_part = parts[1].replace(":", "")
        if len(date_part) == 8 and len(time_part) == 6:
            return f"{date_part}_{time_part}"
    return None


def _collect_session_dirs(dataset_dir: Path) -> list[Path]:
    """Return all immediate subdirectories of *dataset_dir* that contain
    a ``HIF`` subfolder (or HIF-extension files).

    Parameters
    ----------
    dataset_dir:
        Root dataset directory.

    Returns
    -------
    list[Path]
        Sorted list of session directories.
    """
    sessions = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "img":
            continue  # skip output folder
        hif_dir = child / "HIF"
        if hif_dir.is_dir():
            sessions.append(child)
        else:
            log.debug("Skipping %s — no HIF sub-folder", child.name)
    return sessions


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _build_file_index(
    sessions: list[Path],
    exiftool: Path,
    batch_size: int = 200,
) -> list[dict]:
    """Build a list of file records with EXIF metadata and ARW-kept label.

    For each session directory:
    - Collect all HIF/HEIF files from the ``HIF`` sub-folder.
    - Collect stems of ARW files in the session root (kept/selected images).
    - Run exiftool to get datetime + extra tags.
    - Assign ``label = 1`` if a same-stem ARW file exists, else ``label = 0``.

    Parameters
    ----------
    sessions:
        Session directories to process.
    exiftool:
        Path to the exiftool binary.
    batch_size:
        Number of files passed to exiftool per subprocess call.

    Returns
    -------
    list[dict]
        Records with keys:
        ``source_path``, ``stem``, ``session``, ``label``, ``datetime_str``,
        plus all EXTRA_TAGS fields.
    """
    records: list[dict] = []

    for session_dir in sessions:
        session_name = session_dir.name
        hif_dir = session_dir / "HIF"

        # Collect HIF images
        hif_files: list[Path] = [
            f
            for f in hif_dir.iterdir()
            if f.suffix.lower() in HIF_EXTENSIONS
        ]

        if not hif_files:
            log.warning("No HIF/HEIF files found in %s", hif_dir)
            continue

        # Collect selected ARW stems (case-insensitive)
        arw_stems: set[str] = {
            f.stem.upper()
            for f in session_dir.iterdir()
            if f.suffix.upper() == ".ARW"
        }

        log.info(
            "Session %-45s  HIF=%4d  ARW=%4d",
            f"'{session_name}'",
            len(hif_files),
            len(arw_stems),
        )

        # Batch EXIF extraction
        exif_map: dict[str, dict] = {}
        for i in range(0, len(hif_files), batch_size):
            batch = hif_files[i : i + batch_size]
            try:
                exif_records = _run_exiftool(exiftool, batch)
            except subprocess.CalledProcessError as exc:
                log.error("exiftool failed on batch: %s", exc.stderr[:200])
                exif_records = []
            for er in exif_records:
                src = er.get("SourceFile", "")
                exif_map[src] = er

        # Build records for this session
        for hif_file in hif_files:
            er = exif_map.get(str(hif_file), {})
            datetime_str = _parse_datetime(er)
            label = 1 if hif_file.stem.upper() in arw_stems else 0

            extra = {tag: er.get(tag, "") for tag in EXTRA_TAGS}

            records.append(
                {
                    "source_path": hif_file,
                    "stem": hif_file.stem,
                    "session": session_name,
                    "label": label,
                    "datetime_str": datetime_str,
                    **extra,
                }
            )

    return records


def _assign_unique_names(records: list[dict]) -> list[dict]:
    """Resolve name collisions by appending ``_1``, ``_2``, etc.

    Files whose datetime cannot be parsed fall back to their original stem.

    Parameters
    ----------
    records:
        Output of :func:`_build_file_index` — modified **in-place**.

    Returns
    -------
    list[dict]
        Same list, each record gains a ``filename`` key
        (e.g. ``20250322_135328.HIF`` or ``20250322_135328_1.HIF``).
    """
    counter: dict[str, int] = defaultdict(int)
    name_queue: dict[str, list[dict]] = defaultdict(list)

    # Preserve original extension (lower-cased for consistency)
    for rec in records:
        base = rec["datetime_str"] or rec["stem"]
        name_queue[base].append(rec)

    for base, group in name_queue.items():
        src: Path = group[0]["source_path"]
        ext = src.suffix  # keep original case (e.g. .HIF or .heif)

        if len(group) == 1:
            group[0]["filename"] = f"{base}{ext}"
        else:
            for idx, rec in enumerate(group, start=1):
                rec["filename"] = f"{base}_{idx}{ext}"

    # Sanity: ensure uniqueness across the whole dataset
    seen: set[str] = set()
    for rec in records:
        name = rec["filename"]
        if name in seen:
            # Extremely unlikely; append global counter as last resort
            counter[name] += 1
            stem_part, _, ext_part = name.rpartition(".")
            rec["filename"] = f"{stem_part}_{counter[name]}.{ext_part}"
        seen.add(rec["filename"])

    return records


def _move_files(
    records: list[dict],
    img_dir: Path,
    dry_run: bool,
    move: bool = False,
) -> None:
    """Transfer HIF files to *img_dir* with their new filenames.

    By default uses ``shutil.copy2`` (preserves timestamps).  When *move* is
    ``True`` the source file is relocated with ``shutil.move`` instead.

    Parameters
    ----------
    records:
        Records with ``source_path`` and ``filename`` populated.
    img_dir:
        Destination directory.
    dry_run:
        If *True*, log actions but do not touch the filesystem.
    move:
        If *True*, move files instead of copying them.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    action = "Moving" if move else "Copying"
    for rec in tqdm(records, desc=f"{action} files", unit="file"):
        src: Path = rec["source_path"]
        dst = img_dir / rec["filename"]
        rec["img_path"] = str(dst)
        if dry_run:
            log.debug("DRY-RUN (%s)  %s  ->  %s", action.lower(), src.name, dst)
        elif move:
            shutil.move(str(src), dst)
        else:
            shutil.copy2(src, dst)


def _write_csv(records: list[dict], path: Path) -> None:
    """Write a list of record dicts to a CSV file.

    Parameters
    ----------
    records:
        Rows to write.
    path:
        Output file path (parent must exist).
    """
    if not records:
        log.warning("No records to write to %s", path)
        return

    fieldnames = [
        "filename",
        "img_path",
        "label",
        "session",
        "datetime_str",
        "stem",
    ] + EXTRA_TAGS

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)

    log.info("Wrote %d rows to %s", len(records), path)


def _split_dataset(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that preserves label ratio.

    Uses :class:`sklearn.model_selection.StratifiedShuffleSplit` so that
    the minority class (label=1, kept images) is represented proportionally
    in both splits even when the dataset is heavily imbalanced.

    Parameters
    ----------
    df:
        Full labelled dataframe.
    test_size:
        Fraction of data to allocate to the test set (0 < test_size < 1).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(df, df["label"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _print_statistics(
    records: list[dict],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a summary of dataset statistics to stdout."""
    total = len(records)
    kept = sum(1 for r in records if r["label"] == 1)
    discarded = total - kept

    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total HIF images      : {total:>6}")
    print(f"  Label 1  (kept/ARW)   : {kept:>6}  ({kept/total*100:.1f}%)")
    print(f"  Label 0  (discarded)  : {discarded:>6}  ({discarded/total*100:.1f}%)")
    print(f"  Imbalance ratio       : {discarded/max(kept,1):.1f}:1  (neg:pos)")

    print("\n  Per-session breakdown:")
    session_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "kept": 0})
    for r in records:
        s = r["session"]
        session_stats[s]["total"] += 1
        session_stats[s]["kept"] += r["label"]
    for sname, stats in sorted(session_stats.items()):
        t, k = stats["total"], stats["kept"]
        print(f"    {sname:<48}  total={t:4d}  kept={k:4d}  ({k/t*100:.0f}%)")

    print("\n  Train/Test split:")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        t = len(df)
        k = int(df["label"].sum())
        print(
            f"    {name:<6}  total={t:4d}  kept={k:4d}  ({k/t*100:.1f}%)  "
            f"discarded={t-k:4d}  ({(t-k)/t*100:.1f}%)"
        )
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description=(
            "Organise F1 motorsport raw images: rename by EXIF datetime, "
            "move to img/, label against selected ARW files, and split into "
            "train/test CSVs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory containing date-named session folders.",
    )
    parser.add_argument(
        "--exiftool",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the exiftool binary (auto-detected if omitted).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test split (0 < x < 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splitting.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help=(
            "Move files to img/ instead of copying them. "
            "Source files are removed after a successful transfer."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse metadata and generate CSVs without copying files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry-point.

    Parameters
    ----------
    argv:
        Argument list (uses ``sys.argv`` if *None*).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        log.error("Dataset directory not found: %s", dataset_dir)
        return 1

    # Locate exiftool
    try:
        exiftool = _find_exiftool(args.exiftool)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("Using exiftool: %s", exiftool)

    img_dir = dataset_dir / "img"
    if not args.dry_run and img_dir.exists():
        log.warning(
            "Output directory '%s' already exists — existing files may be overwritten.",
            img_dir,
        )

    # 1. Discover sessions
    sessions = _collect_session_dirs(dataset_dir)
    if not sessions:
        log.error("No session directories with HIF sub-folders found under %s", dataset_dir)
        return 1
    log.info("Found %d session(s)", len(sessions))

    # 2. Build index with EXIF metadata
    log.info("Extracting EXIF metadata …")
    records = _build_file_index(sessions, exiftool)
    log.info("Indexed %d HIF/HEIF files total", len(records))

    # 3. Assign unique datetime-based filenames
    records = _assign_unique_names(records)

    # 4. Move / copy files to img/
    _move_files(records, img_dir, dry_run=args.dry_run, move=args.move)

    # 5. Write data_info.csv
    data_info_path = dataset_dir / "data_info.csv"
    _write_csv(records, data_info_path)

    # 6. Stratified train/test split
    df = pd.DataFrame(records)
    train_df, test_df = _split_dataset(df, test_size=args.test_size, seed=args.seed)

    train_csv = dataset_dir / "train_info.csv"
    test_csv = dataset_dir / "test_info.csv"

    _write_csv(train_df.to_dict("records"), train_csv)
    _write_csv(test_df.to_dict("records"), test_csv)

    # 7. Print statistics
    _print_statistics(records, train_df, test_df)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
