"""
engine.py — Core culling engine for Auto-Culling.
Handles file scanning, model loading, and the culling pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np
from cull.exif_reader import ExifData, BurstGroup, read_exif, group_bursts
from cull.detector import CloudF1Detector, load_f1_model, load_coco_model, detect
from cull.loader import load_image_rgb, update_image_metadata, RAW_EXTS, COOKED_EXTS, EXTENSIONS
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import ImageScore, score_image, select_best_n, SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
from cull.xmp_writer import write_xmp_batch
from cull.xmp_reader import read_xmp_rating
from cull.cropper import calculate_crop
from cull.renamer import rename_images

log = logging.getLogger(__name__)

# Frame log line shared by every scoring outcome (scored, manual metadata,
# decode failure) so protocol consumers see exactly one event per image.
FRAME_LOG_FMT = "  [%s]  sharp=%.3f  comp=%.3f  raw=%.2f  Rating=%+d%s"


def dedupe_raw_cooked(image_paths: list[Path]) -> tuple[list[Path], set[Path]]:
    """Collapse RAW/cooked pairs (same stem) onto the cooked file.

    Returns the deduplicated, sorted shot list plus the subset of cooked
    files that have no RAW sibling (metadata sync target).
    """
    stems: dict[str, Path] = {}
    has_raw: dict[str, bool] = {}
    for p in image_paths:
        stem = p.stem.lower()
        ext = p.suffix.lower()
        if ext in RAW_EXTS:
            has_raw[stem] = True
        if stem not in stems:
            stems[stem] = p
        else:
            prev = stems[stem]
            if ext in COOKED_EXTS and prev.suffix.lower() in RAW_EXTS:
                stems[stem] = p

    unique = sorted(stems.values())
    standalone = {p for stem, p in stems.items()
                  if p.suffix.lower() in COOKED_EXTS and not has_raw.get(stem)}
    return unique, standalone

@dataclass
class EngineConfig:
    """Configuration for the CullingEngine."""
    input_dir: Path
    recursive: bool = False
    f1_model_path: Path = Path("models/f1_yolov8n.onnx")
    rf_api_key: str | None = None
    top_n: int = 11
    sharp_thresh: float = SHARP_THRESH
    w_sharp: float = W_SHARP
    w_comp: float = W_COMP
    min_raw: float = MIN_RAW
    conf: float = 0.30
    dry_run: bool = False
    force: bool = False
    p4_policy: str = "auto"
    scale_width: int = 0
    autocrop: bool = True
    rename: bool = False
    workers: int = 4
    dump_scores: Path | None = None
    label_check: bool = False
    label_check_dir: Path | None = None

class CullingEngine:
    """
    Stateful engine that manages a culling session.
    """
    def __init__(self, config: EngineConfig):
        self.config = config
        self.image_paths: list[Path] = []
        self.exif_map: dict[Path, ExifData] = {}
        self.groups: list[BurstGroup] = []
        self.all_scores: list[ImageScore] = []
        self.f1_model = None
        self.coco_model = None
        self.cloud_f1 = None
        self.standalone_cooked: set[Path] = set()

    @staticmethod
    def collect_shots(input_dir: Path, recursive: bool = False) -> tuple[list[Path], set[Path]]:
        """Collect supported images and deduplicate RAW/cooked pairs.

        Shared by the engine scan and the sidecar ``scan`` command so the
        pending list shown before a run matches what the run will process.
        """
        found = CullingEngine._collect_images(input_dir, recursive)
        return dedupe_raw_cooked(found)


    def scan(self, progress_callback: Callable[[str, float], None] | None = None,
             cancel_event: threading.Event | None = None):
        """Scan input directory and group bursts."""
        if progress_callback:
            progress_callback("Collecting images...", 0.1)

        # 1. Collect images
        self.image_paths = self._collect_images(self.config.input_dir, self.config.recursive)
        log.info("Found %d raw image files", len(self.image_paths))

        if cancel_event and cancel_event.is_set():
            return

        if self.config.rename:
            if progress_callback:
                progress_callback("Renaming images...", 0.2)
            new_map = rename_images(self.image_paths, dry_run=self.config.dry_run)
            self.image_paths = sorted(list(new_map.values()))

        if cancel_event and cancel_event.is_set():
            return

        # 2. Prioritize JPG/HIF over RAW (same stems survive renaming)
        self.image_paths, self.standalone_cooked = dedupe_raw_cooked(self.image_paths)

        log.info("Processing %d unique shots", len(self.image_paths))

        if cancel_event and cancel_event.is_set():
            return

        # 3. Read EXIF & Grouping
        if progress_callback:
            progress_callback("Reading EXIF metadata...", 0.4)
        exif_list = read_exif(self.image_paths)
        self.exif_map = {e.path: e for e in exif_list}

        if cancel_event and cancel_event.is_set():
            return

        if progress_callback:
            progress_callback("Grouping burst sequences...", 0.6)
        self.groups = group_bursts(exif_list)
        log.info("Grouped into %d burst groups", len(self.groups))

    def load_models(self, progress_callback: Callable[[str, float], None] | None = None,
                    cancel_event: threading.Event | None = None):
        """Load detection models."""
        if cancel_event and cancel_event.is_set():
            return
        if progress_callback:
            progress_callback("Loading models...", 0.8)

        self.coco_model = load_coco_model()
        if self.config.rf_api_key:
            self.cloud_f1 = CloudF1Detector(self.config.rf_api_key)
        elif self.config.f1_model_path.exists():
            self.f1_model = load_f1_model(self.config.f1_model_path)
        else:
            log.warning("No F1 model available.")

    def run(self, progress_callback: Callable[[str, float], None] | None = None,
            cancel_event: threading.Event | None = None):
        """Execute the culling process.

        If *cancel_event* is provided, its flag is checked between frames;
        once set, scoring stops early, side effects (XMP writes / metadata
        sync) are skipped, and the partial scores collected so far are
        returned. The CLI passes no cancel_event and is unaffected.
        """
        self.scan(progress_callback, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set():
            if progress_callback:
                progress_callback("Cancelled", 0.0)
            return [], 0.0

        self.load_models(progress_callback, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set():
            if progress_callback:
                progress_callback("Cancelled", 0.0)
            return [], 0.0
        
        if progress_callback:
            progress_callback("Analyzing images...", 0.9)

        t_start = time.perf_counter()

        # DirectML's D3D12 device context is not thread-safe under concurrent
        # ThreadPoolExecutor execution on Windows; cap workers to 1 to ensure
        # 100% stability without access violations.
        from cull.detector import use_directml
        workers = 1 if use_directml() else self.config.workers

        # Parallel group processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            def _wrap_process(g_info):
                idx, group = g_info
                if cancel_event and cancel_event.is_set():
                    return []
                log.info("Processing Group %d/%d (%d frames)", idx, len(self.groups), len(group.frames))
                res = self._process_group_internal(group, cancel_event)
                for s in res:
                    s.burst_group = idx
                return res

            group_results = list(executor.map(_wrap_process, enumerate(self.groups, start=1)))
        
        self.all_scores = []
        for res in group_results:
            self.all_scores.extend(res)

        elapsed = time.perf_counter() - t_start

        if cancel_event and cancel_event.is_set():
            log.info("Cancelled after scoring %d/%d frames", len(self.all_scores), len(self.image_paths))
            if progress_callback:
                progress_callback("Cancelled", 0.0)
            return self.all_scores, elapsed
        
        if progress_callback:
            progress_callback("Saving metadata...", 0.95)
            
        # Write XMP sidecars for shots that need them (RAW or RAW+JPG pairs)
        # Skip standalone JPG/HIF files since we sync metadata directly into them.
        xmp_pairs = [(s.path, s.rating, s.crop) for s in self.all_scores 
                     if not s.is_manual and s.path not in self.standalone_cooked]
        write_xmp_batch(xmp_pairs, overwrite=True, dry_run=self.config.dry_run)

        if not self.config.dry_run:
            to_sync = [s for s in self.all_scores if s.path in self.standalone_cooked and not s.is_manual]
            if to_sync:
                log.info("Syncing metadata to %d standalone JPG/HIF files...", len(to_sync))
                with ThreadPoolExecutor(max_workers=min(8, self.config.workers)) as sync_executor:
                    for s in to_sync:
                        sync_executor.submit(update_image_metadata, s.path, s.rating, s.crop)

        if progress_callback:
            progress_callback("Done!", 1.0)
            
        return self.all_scores, elapsed

    def _process_group_internal(self, group: BurstGroup,
                                cancel_event: threading.Event | None = None) -> list[ImageScore]:
        """Core logic for processing a single burst group."""
        scores: list[ImageScore] = []
        prev_detections = None
        frames = group.frames
        
        # P4 Policy check
        check_p4 = self.config.p4_policy == "always"
        if self.config.p4_policy == "auto":
            dir_name = frames[0].parent.name.lower()
            keywords = ["f1", "gp", "grand prix", "race", "quali", "practice"]
            check_p4 = any(k in dir_name for k in keywords) and "sprint_quali" not in dir_name

        for frame_idx, frame_path in enumerate(frames):
            if cancel_event and cancel_event.is_set():
                log.info("Cancel requested; stopping group %s after %d frames",
                         group.group_id, frame_idx)
                break
            # Check for existing culling status
            xmp_rating, xmp_pick = read_xmp_rating(frame_path)
            exif = self.exif_map.get(frame_path)
            final_rating = xmp_rating if xmp_rating is not None else (exif.rating if exif else None)
            final_pick = xmp_pick if xmp_pick is not None else (exif.pick if exif else None)

            is_rating_set = final_rating is not None and final_rating != 0
            is_pick_set = final_pick is not None and final_pick != 0

            if not self.config.force and (is_rating_set or is_pick_set):
                if not is_rating_set:
                    final_rating = 1 if final_pick == 1 else (-1 if final_pick == -1 else 0)
                manual_score = ImageScore(
                    path=frame_path, s_sharp=1.0, s_comp=1.0,
                    raw_score=10.0 if final_rating > 0 else 0.0,
                    rating=final_rating, vetoed=(final_rating == -1),
                    veto_reason="manual_metadata", is_manual=True
                )
                scores.append(manual_score)
                log.info(FRAME_LOG_FMT, frame_path.name, manual_score.s_sharp,
                         manual_score.s_comp, manual_score.raw_score, manual_score.rating,
                         "  (manual_metadata)")
                continue

            # Load image
            img_rgb = load_image_rgb(frame_path, scale_width=self.config.scale_width)
            if img_rgb is None:
                log.info(FRAME_LOG_FMT, frame_path.name, 0.0, 0.0, 0.0, 0,
                         "  (decode_failed)")
                continue


            h, w = img_rgb.shape[:2]

            # Detection
            if self.cloud_f1:
                detections = self.cloud_f1.detect(img_rgb, conf=self.config.conf)
                if not detections:
                    detections = detect(img_rgb, None, self.coco_model, conf=self.config.conf)
            else:
                detections = detect(img_rgb, self.f1_model, self.coco_model, conf=self.config.conf)

            # Scoring
            s_sharp = score_sharpness(img_rgb, detections[0] if detections else None)
            s_comp = score_composition(detections, w, h, prev_detections, frame_idx == 0)
            
            img_score = score_image(
                path=frame_path, detections=detections, s_sharp=s_sharp, s_comp=s_comp,
                sharp_thresh=self.config.sharp_thresh, w_sharp=self.config.w_sharp,
                w_comp=self.config.w_comp, min_raw=self.config.min_raw,
                check_p4=check_p4, img_rgb=img_rgb, img_w=w, img_h=h
            )
            scores.append(img_score)
            
            log.info(
                FRAME_LOG_FMT, frame_path.name, s_sharp, s_comp, img_score.raw_score,
                img_score.rating, f"  ({img_score.veto_reason})" if img_score.vetoed else ""
            )

            prev_detections = detections if detections else None

        select_best_n(scores, top_n=self.config.top_n)
        
        if self.config.autocrop:
            for s in scores:
                if s.rating >= 1 and not s.is_manual and s.detections:
                    f1_cars = [d for d in s.detections if d.label == "f1_car"]
                    if len(f1_cars) == 1 and s.img_w and s.img_h:
                        d = f1_cars[0]
                        s.crop = calculate_crop(d.x1/s.img_w, d.y1/s.img_h, d.x2/s.img_w, d.y2/s.img_h, img_ar=s.img_w/s.img_h)
        return scores

    @staticmethod
    def _collect_images(input_dir: Path, recursive: bool) -> list[Path]:
        """Scan *input_dir* for supported image files, sorted by name."""
        import os
        found: list[Path] = []
        def _scan(target: str):
            try:
                with os.scandir(target) as it:
                    for entry in it:
                        if entry.is_file():
                            if entry.name.startswith("._"):
                                continue
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in EXTENSIONS:
                                found.append(Path(entry.path))
                        elif recursive and entry.is_dir():
                            if entry.name.startswith("."):
                                continue
                            _scan(entry.path)
            except PermissionError:
                pass
        _scan(str(input_dir))
        return sorted(found)

    def export_scores_csv(self, out_path: Path):
        """Write per-image scores to a CSV file."""
        import csv
        
        # Build ARW ground truth set if needed
        arw_stems: set[str] = set()
        for search_dir in [self.config.input_dir, self.config.input_dir.parent]:
            if search_dir.is_dir():
                for p in search_dir.iterdir():
                    if p.suffix.lower() == ".arw":
                        arw_stems.add(p.stem.lower())

        fieldnames = [
            "filename", "s_sharp", "s_comp", "raw_score", "rating",
            "vetoed", "veto_reason", "n_detections", "burst_group", "has_arw"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.all_scores:
                writer.writerow({
                    "filename": s.path.name,
                    "s_sharp": f"{s.s_sharp:.6f}",
                    "s_comp": f"{s.s_comp:.6f}",
                    "raw_score": f"{s.raw_score:.6f}",
                    "rating": s.rating,
                    "vetoed": int(s.vetoed),
                    "veto_reason": s.veto_reason,
                    "n_detections": s.n_detections,
                    "burst_group": s.burst_group,
                    "has_arw": int(s.path.stem.lower() in arw_stems)
                })
        log.info("Scores dumped to %s", out_path)

    def run_label_check(self, arw_dir: Path | None = None):
        """Compare system ratings against ARW ground truth."""
        log.info("\n" + "=" * 60)
        log.info("LABEL CHECK — comparing with ARW ground truth")
        log.info("=" * 60)

        arw_stems: set[str] = set()
        if arw_dir and arw_dir.is_dir():
            arw_stems = {p.stem.lower() for p in arw_dir.iterdir() if p.suffix.lower() == ".arw"}
        else:
            for search_dir in [self.config.input_dir, self.config.input_dir.parent]:
                if search_dir.is_dir():
                    for p in search_dir.iterdir():
                        if p.suffix.lower() == ".arw":
                            arw_stems.add(p.stem.lower())

        if not arw_stems:
            log.warning("No ARW files found — cannot run label check.")
            return

        tp = fp = fn = tn = 0
        mismatches = []
        for score in self.all_scores:
            stem = score.path.stem.lower()
            actual_keep = stem in arw_stems
            predicted_keep = score.rating > 0
            if predicted_keep and actual_keep: tp += 1
            elif predicted_keep and not actual_keep:
                fp += 1
                mismatches.append((score.path.name, "KEEP", "discard"))
            elif not predicted_keep and actual_keep:
                fn += 1
                mismatches.append((score.path.name, "REJECT", "keep"))
            else: tn += 1

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        log.info("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", accuracy, precision, recall, f1)
        if mismatches:
            log.info("Sample mismatches: %s", mismatches[:10])
