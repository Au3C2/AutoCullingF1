"""
cull_photos.py — Rule-based F1 photo culling pipeline CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from cull.engine import CullingEngine, EngineConfig
from cull.protocol import JsonLinesHandler, emit
from cull.scorer import MIN_RAW, SHARP_THRESH, W_COMP, W_SHARP

log = logging.getLogger(__name__)

if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # Suppress crash dialogs
    except Exception:
        pass


def subprocess_flags() -> dict:
    if platform.system() == "Windows":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.resolve()
    return base_path / relative_path


def select_folder(default_dir: Path) -> Path | None:
    """Prompt user to select a folder using native dialog."""
    if platform.system() == "Darwin":
        script = (
            f'tell application "System Events"\n'
            f'  activate\n'
            f'  set theFolder to choose folder with prompt "Select photo directory to cull" '
            f'default location POSIX file "{default_dir}"\n'
            f'  return POSIX path of theFolder\n'
            f'end tell'
        )
        try:
            out = subprocess.check_output(['osascript', '-e', script], text=True).strip()
            return Path(out) if out else None
        except Exception:
            return None
    elif platform.system() == "Windows":
        script = (
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$f.SelectedPath = '{default_dir}'; "
            f"$f.Description = 'Select photo directory to cull'; "
            f"$f.ShowNewFolderButton = $false; "
            f"if($f.ShowDialog() -eq 'OK') {{ $f.SelectedPath }}"
        )
        try:
            out = subprocess.check_output(['powershell', '-Command', script], text=True,
                                          **subprocess_flags()).strip()
            return Path(out) if out else None
        except Exception:
            return None
    return None


def setup_logging(base_dir: Path | None, console: bool = True) -> Path | None:
    log_dir = (base_dir or Path(tempfile.gettempdir())) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cull_{timestamp}.log"
    root_log = logging.getLogger()
    for h in root_log.handlers[:]:
        root_log.removeHandler(h)
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_log.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        root_log.addHandler(ch)
    else:
        root_log.addHandler(JsonLinesHandler())
    log.info("Logging to %s", log_file)
    return log_file


def _build_config(args: argparse.Namespace, input_dir: Path) -> EngineConfig:
    config = EngineConfig(
        input_dir=input_dir,
        recursive=args.recursive,
        f1_model_path=Path(args.f1_model),
        rf_api_key=args.rf_api_key,
        top_n=args.top_n,
        sharp_thresh=args.sharp_thresh,
        w_sharp=args.w_sharp,
        w_comp=args.w_comp,
        min_raw=args.min_raw,
        conf=args.conf,
        dry_run=args.dry_run,
        force=args.force,
        p4_policy=args.p4_policy,
        scale_width=args.scale_width,
        autocrop=getattr(args, "autocrop", not getattr(args, "crop_off", False)),
        rename=args.rename,
        workers=args.workers,
        consumer_threads=getattr(args, "consumer_threads", 1),
        dump_scores=Path(args.dump_scores) if args.dump_scores else None,
        label_check=args.label_check,
        label_check_dir=Path(args.label_check_dir) if args.label_check_dir else None,
        deterministic=getattr(args, "deterministic", False),
    )

    f1_model = config.f1_model_path
    if not f1_model.exists():
        bundled_f1 = get_resource_path(f"models/{f1_model.name}")
        if bundled_f1.exists():
            config.f1_model_path = bundled_f1
            log.debug("Using bundled F1 model: %s", bundled_f1)
    return config


def _sibling_ext_index(directory: Path, recursive: bool) -> dict[tuple[str, str], set[str]]:
    """One os.walk pass over *directory*: (parent dir, lowercase stem) → set
    of lowercase file extensions sharing that stem.

    The engine collapses RAW+cooked pairs onto the cooked file, so the shot
    list alone cannot tell the GUI which formats exist on disk. This index
    restores that (e.g. DSC00827.heif + DSC00827.ARW + DSC00827.xmp) so the
    GUI can render one format tag per sibling format.
    """
    index: dict[tuple[str, str], set[str]] = {}
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            stem, dot, ext = fname.rpartition(".")
            if not dot or not ext:
                continue
            index.setdefault((root.lower(), stem.lower()), set()).add(ext.lower())
        if not recursive:
            break
    return index


def run_json_lines(args: argparse.Namespace, input_dir: Path | None) -> int:
    """Resident JSON Lines engine for the Tauri GUI."""
    import base64
    import io
    import json
    import queue
    import threading
    from cull.engine import CullingEngine
    from cull.gui.preview import render_pil
    from cull.scorer import ImageScore

    setup_logging(input_dir, console=False)
    cancel_event = threading.Event()

    cmd_queue: "queue.Queue[dict]" = queue.Queue()
    last_scores: dict[Path, ImageScore] = {}
    last_engine = None

    def stdin_reader() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if line == "cancel":
                    cancel_event.set()
                    continue
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue
                if not isinstance(cmd, dict):
                    continue
                log.info("engine command: %s", cmd.get("cmd"))
                if cmd.get("cmd") == "cancel":
                    cancel_event.set()
                    continue
                if cmd.get("cmd") == "preview":
                    do_preview(cmd)
                    continue
                if cmd.get("cmd") == "run":
                    # Clear any stale cancel HERE — at the moment the run is
                    # requested — so a cancel sent after this point survives
                    # until engine.run() checks it. Clearing inside do_run
                    # instead races a cancel that arrives between queueing
                    # and execution (it would be wiped and the cancel lost).
                    cancel_event.clear()
                cmd_queue.put(cmd)
                if cmd.get("cmd") == "quit":
                    return
        except Exception:
            pass

    def _resolve_shot_meta(path: Path, exif_data: Any | None) -> tuple[int | None, str | None]:
        # Tier 1: Real EXIF datetime
        if exif_data and getattr(exif_data, "datetime_original", None):
            dt = exif_data.datetime_original
            return int(dt.timestamp() * 1000), dt.strftime("%H:%M:%S.%f")[:-3]

        # Tier 2: Filename pattern matching
        name = path.stem
        # Pattern 1: YYYYMMDD_HHMMSS_mmm or YYYYMMDD-HHMMSS-mmm
        m1 = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-](\d{2})(\d{2})(\d{2})[_-](\d{1,3})", name)
        if m1:
            Y, M, D, h, m_m, s, ms = m1.groups()
            try:
                dt = datetime(int(Y), int(M), int(D), int(h), int(m_m), int(s), int(ms.ljust(3, "0")[:3]) * 1000)
                return int(dt.timestamp() * 1000), dt.strftime("%H:%M:%S.%f")[:-3]
            except Exception:
                pass

        # Pattern 2: YYYYMMDD_HHMMSS
        m2 = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-](\d{2})(\d{2})(\d{2})", name)
        if m2:
            Y, M, D, h, m_m, s = m2.groups()
            try:
                dt = datetime(int(Y), int(M), int(D), int(h), int(m_m), int(s))
                return int(dt.timestamp() * 1000), dt.strftime("%H:%M:%S")
            except Exception:
                pass

        # Tier 3: File system stat fallback
        try:
            st = path.stat()
            mtime = getattr(st, "st_birthtime", st.st_mtime)
            dt = datetime.fromtimestamp(mtime)
            return int(dt.timestamp() * 1000), dt.strftime("%H:%M:%S")
        except Exception:
            pass

        return None, None

    def do_scan(cmd: dict) -> None:
        raw_dir = cmd.get("dir") or (str(input_dir) if input_dir else "")
        if not raw_dir:
            emit({"type": "scan_error", "message": "no directory provided for scan"})
            return
        directory = Path(raw_dir)
        recursive = bool(cmd.get("recursive", args.recursive))
        try:
            shots, _standalone = CullingEngine.collect_shots(directory, recursive)
            # Sibling format index: one fast walk, no file reads — lets the GUI
            # show every format present per shot ([HEIF][RAW][XML], ...).
            ext_index = _sibling_ext_index(directory, recursive)

            def _sibling_exts(p: Path) -> list[str]:
                key = (str(p.parent).lower(), p.stem.lower())
                exts = ext_index.get(key)
                return sorted(exts) if exts else [p.suffix.lower()]

            # Phase 1 — emit the file list IMMEDIATELY with fast metadata
            # (filename pattern / stat fallback) so the GUI renders the photo
            # table without waiting on exiftool. For a 10k-file directory the
            # EXIF pass below takes tens of seconds; blocking the list behind
            # it made the GUI appear frozen after a drag-and-drop.
            items = []
            for p in shots:
                ts, time_str = _resolve_shot_meta(p, None)
                items.append({"path": str(p), "name": p.name, "timestamp": ts,
                              "time_str": time_str, "exts": _sibling_exts(p)})
            emit({"type": "scanned", "dir": str(directory),
                  "count": len(shots),
                  "total": len(shots),
                  "paths": [str(p) for p in shots],
                  "items": items})

            # Phase 2 — EXIF timestamps + authoritative burst grouping in the
            # background; the result arrives as a separate 'scan_meta' event
            # that the GUI merges in place (no list reset, no scroll jump).
            def _exif_worker() -> None:
                try:
                    from cull.exif_reader import read_exif, group_bursts
                    # Exclude empty 0-byte files from exiftool call to prevent exiftool process failure
                    valid_shots = [p for p in shots if p.is_file() and p.stat().st_size > 0]
                    exif_list = read_exif(valid_shots) if valid_shots else []
                    exif_map = {e.path: e for e in exif_list}

                    group_map: dict[Path, str] = {}
                    try:
                        for g in group_bursts(exif_list):
                            for frame in g.frames:
                                group_map[frame] = g.group_id
                    except Exception as e_group:
                        log.debug("Scan burst grouping failed: %s", e_group)

                    meta_items = []
                    for p in shots:
                        e = exif_map.get(p)
                        ts, time_str = _resolve_shot_meta(p, e)
                        meta_items.append({
                            "path": str(p),
                            "timestamp": ts,
                            "time_str": time_str,
                            "burst_group": group_map.get(p),
                            "exts": _sibling_exts(p),
                        })
                    emit({"type": "scan_meta", "dir": str(directory), "items": meta_items})
                except Exception as e_exif:
                    log.debug("Scan EXIF read failed, keeping tiered meta: %s", e_exif)
                    emit({"type": "scan_meta", "dir": str(directory), "items": []})

            threading.Thread(target=_exif_worker, daemon=True).start()
        except Exception as exc:
            emit({"type": "scan_error", "message": str(exc)})

    def do_run(cmd: dict) -> None:
        nonlocal last_engine
        merged = argparse.Namespace(**vars(args))
        for key, value in cmd.get("config", {}).items():
            setattr(merged, key, value)
            if key == "autocrop":
                merged.crop_off = not value
        run_input_str = cmd.get("dir") or (str(merged.input_dir) if merged.input_dir else "")
        if not run_input_str:
            emit({"type": "error", "message": "no directory specified"})
            return
        run_input = Path(run_input_str)
        if not run_input.is_dir():
            emit({"type": "error", "message": f"input directory not found: {run_input}"})
            return

        config = _build_config(merged, run_input)
        # GUI protocol keys frames by full path (scan results do the same)
        config.log_full_paths = True
        engine = CullingEngine(config)

        def progress(msg: str, p: float) -> None:
            emit({"type": "stage", "message": msg, "msg": msg, "progress": p, "pct": p})

        try:
            scores, elapsed = engine.run(progress_callback=progress,
                                         cancel_event=cancel_event)
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
            return

        if cancel_event.is_set():
            last_scores.clear()
            for s in scores:
                last_scores[s.path] = s
            emit({"type": "cancelled", "count": len(scores)})
            return

        last_engine = engine
        last_scores.clear()
        for s in scores:
            last_scores[s.path] = s
        total = len(scores)
        keep = sum(1 for s in scores if s.rating > 0)
        failed = getattr(engine, "failed_count", 0)
        stars: dict[int, int] = {}
        for s in scores:
            if s.rating > 0:
                stars[s.rating] = stars.get(s.rating, 0) + 1
        emit({"type": "done", "elapsed": elapsed, "total": total, "keep": keep,
              "reject": total - keep, "failed": failed, "stars": stars})

    def do_preview(cmd: dict) -> None:
        def _worker():
            raw_path = cmd.get("path", "")
            try:
                p = Path(raw_path)
                score = last_scores.get(p)
                if score is None:
                    # Try matching by resolved absolute path or filename
                    resolved_p = p.resolve()
                    score = last_scores.get(resolved_p)
                    if score is None:
                        for s_path, s_obj in last_scores.items():
                            if s_path.name.lower() == p.name.lower():
                                score = s_obj
                                break

                if score is None:
                    score = ImageScore(path=p, s_sharp=0.0, s_comp=0.0, raw_score=0.0, rating=0)

                pil = render_pil(score, max_size=int(cmd.get("size", 640)))
            except Exception as e:
                log.warning("do_preview render error for %s: %s", raw_path, e)
                pil = None

            if pil is None:
                emit({"type": "preview", "path": raw_path, "data": None, "png": None})
            else:
                try:
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
                    raw_boxes = []
                    if hasattr(score, "detections") and score.detections:
                        for det in score.detections:
                            raw_boxes.append([float(det.x1), float(det.y1), float(det.x2), float(det.y2), str(det.label), float(det.conf)])
                    raw_crop = [float(x) for x in score.crop] if getattr(score, "crop", None) else None

                    emit({
                        "type": "preview",
                        "path": raw_path,
                        "data": b64_str,
                        "png": b64_str,
                        "width": pil.width,
                        "height": pil.height,
                        "boxes": raw_boxes,
                        "crop": raw_crop
                    })
                except Exception as e:
                    log.warning("do_preview serialization error for %s: %s", raw_path, e)
                    emit({"type": "preview", "path": raw_path, "data": None, "png": None})

        threading.Thread(target=_worker, daemon=True).start()

    def do_export_csv(cmd: dict) -> None:
        out_str = cmd.get("path") or (str(input_dir / "scores.csv") if input_dir else "")
        if not out_str:
            emit({"type": "error", "message": "no output path for CSV export"})
            return
        if last_engine is None or not last_engine.all_scores:
            emit({"type": "error", "message": "no scores available — run a culling first"})
            return
        try:
            last_engine.export_scores_csv(Path(out_str))
            emit({"type": "export_done", "path": str(out_str)})
        except Exception as exc:
            log.warning("CSV export failed: %s", exc)
            emit({"type": "error", "message": f"CSV export failed: {exc}"})

    def do_save_metadata(cmd: dict) -> None:
        items = cmd.get("items", [])
        if not items:
            emit({"type": "save_done", "count": 0, "status": "ok"})
            return

        def _save_worker() -> None:
            try:
                from cull.xmp_writer import write_xmp_batch
                from cull.loader import update_image_metadata_batch, COOKED_EXTS, RAW_EXTS, SIDECAR_EXTS

                xmp_list = []
                sync_list = []

                for it in items:
                    p_str = it.get("path")
                    if not p_str:
                        continue
                    p = Path(p_str)
                    rating = int(it.get("rating", 0))
                    crop = it.get("crop")
                    crop_tuple = tuple(crop) if crop and len(crop) == 4 else None

                    ext = p.suffix.lower()
                    # RAW and HEIF/HIF formats always use Lightroom-compatible .xmp sidecars.
                    # Only standalone JPEGs have metadata synced directly into the file.
                    has_raw_sibling = any(
                        p.with_suffix(re).exists() or p.with_suffix(re.upper()).exists()
                        for re in RAW_EXTS
                    )
                    if ext in SIDECAR_EXTS or has_raw_sibling:
                        xmp_list.append((p, rating, crop_tuple))
                    elif ext in COOKED_EXTS:
                        sync_list.append((p, rating, crop_tuple))
                    else:
                        xmp_list.append((p, rating, crop_tuple))

                if xmp_list:
                    write_xmp_batch(xmp_list, overwrite=True, dry_run=False)
                if sync_list:
                    update_image_metadata_batch(sync_list)

                emit({"type": "save_done", "count": len(items), "status": "ok"})
            except Exception as e:
                log.error("do_save_metadata failed: %s", e)
                emit({"type": "save_error", "message": str(e)})

        threading.Thread(target=_save_worker, daemon=True).start()

    from cull.highres import HighResProvider, HighResRequest
    highres_provider = HighResProvider()

    def do_highres(cmd: dict) -> None:
        def _highres_worker() -> None:
            raw_path = cmd.get("path", "")
            gen_id = int(cmd.get("gen_id", 0))
            roi = cmd.get("roi")
            try:
                highres_provider.update_active_generation(gen_id)
                res = highres_provider.resolve(HighResRequest(
                    file_path=Path(raw_path),
                    gen_id=gen_id,
                    roi=roi,
                ))
                if res is not None:
                    emit({
                        "type": "highres_ready",
                        "gen_id": res.gen_id,
                        "path": str(res.resolved_path),
                        "orig_path": raw_path,
                        "format": res.format,
                        "width": res.width,
                        "height": res.height,
                        "tier": res.tier,
                    })
                else:
                    emit({
                        "type": "highres_discarded",
                        "gen_id": gen_id,
                        "orig_path": raw_path,
                    })
            except Exception as e:
                log.warning("do_highres failed for %s: %s", raw_path, e)
                emit({
                    "type": "highres_error",
                    "gen_id": gen_id,
                    "orig_path": raw_path,
                    "message": str(e),
                })

        threading.Thread(target=_highres_worker, daemon=True).start()

    # Start the reader only after every handler exists — a command arriving
    # during the definition window used to raise NameError inside the reader
    # thread and kill it for the rest of the session.
    threading.Thread(target=stdin_reader, daemon=True).start()

    while True:
        cmd = cmd_queue.get()
        kind = cmd.get("cmd")
        if kind == "quit":
            return 0
        if kind == "scan":
            do_scan(cmd)
        elif kind == "run":
            do_run(cmd)
        elif kind == "preview":
            do_preview(cmd)
        elif kind == "export_csv":
            do_export_csv(cmd)
        elif kind == "save_metadata":
            do_save_metadata(cmd)
        elif kind == "highres":
            do_highres(cmd)



def run(args: argparse.Namespace) -> int:
    if getattr(args, "deterministic", False):
        import os as _os
        _os.environ["CULL_DETERMINISTIC"] = "1"

    input_dir = Path(args.input_dir) if args.input_dir else None
    if input_dir is None:
        if getattr(sys, 'frozen', False):
            default_dir = Path(sys.executable).parent
        else:
            default_dir = Path(__file__).parent.resolve()
        selected = select_folder(default_dir)
        if selected:
            input_dir = selected
            args.input_dir = selected
        else:
            print("No directory selected. Exiting.")
            return 0

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1

    setup_logging(input_dir)
    config = _build_config(args, input_dir)
    engine = CullingEngine(config)

    def progress(msg, p):
        log.info("[%d%%] %s", int(p * 100), msg)

    try:
        all_scores, elapsed = engine.run(progress_callback=progress)
    except Exception as e:
        log.exception("Engine failed: %s", e)
        return 1

    n_total = len(all_scores)
    n_reject = sum(1 for s in all_scores if s.rating == -1)
    n_keep = n_total - n_reject
    ips = n_total / elapsed if elapsed > 0 else float("inf")

    log.info(
        "\nDone in %.1fs  (%.1f img/s)  total=%d  keep=%d  reject=%d",
        elapsed, ips, n_total, n_keep, n_reject,
    )

    rating_dist: dict[int, int] = {}
    for s in all_scores:
        rating_dist[s.rating] = rating_dist.get(s.rating, 0) + 1
    for r in sorted(rating_dist):
        label = "Rejected" if r == -1 else f"{r}*"
        log.info("  %8s : %d", label, rating_dist[r])

    if config.dump_scores:
        engine.export_scores_csv(config.dump_scores)

    if config.label_check:
        engine.run_label_check(config.label_check_dir)

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cull_photos",
        description="Rule-based F1 photo culling pipeline (LITE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", type=Path, default=None, help="Directory to process.")
    parser.add_argument("--recursive", action="store_true", help="Recursive scan.")
    parser.add_argument("--f1-model", type=Path, default=Path("models/f1_yolov8n.onnx"), help="Path to F1 ONNX model.")
    parser.add_argument("--rf-api-key", default=None, help="Roboflow API key.")
    parser.add_argument("--crop-off", action="store_true", help="Disable auto-cropping.")
    parser.add_argument("--top-n", type=int, default=11, help="Max frames per burst.")
    parser.add_argument("--sharp-thresh", type=float, default=SHARP_THRESH)
    parser.add_argument("--w-sharp", type=float, default=W_SHARP)
    parser.add_argument("--w-comp", type=float, default=W_COMP)
    parser.add_argument("--min-raw", type=float, default=MIN_RAW)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--p4-policy", choices=["always", "never", "auto"], default="never")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--dump-scores", type=str, default=None)
    parser.add_argument("--label-check", action="store_true")
    parser.add_argument("--label-check-dir", type=Path, default=None)
    parser.add_argument("--scale-width", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--consumer-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json-lines", action="store_true",
                        help="JSON Lines engine mode for the Tauri GUI")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.json_lines:
        return run_json_lines(args, Path(args.input_dir) if args.input_dir else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return run(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
