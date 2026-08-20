"""
cull_photos.py — Rule-based F1 photo culling pipeline CLI.
(LITE VERSION — NO OpenCV)
"""

from __future__ import annotations

import argparse
import logging
import sys
import platform
import subprocess
import tempfile
from pathlib import Path

from cull.engine import CullingEngine, EngineConfig
from cull.loader import subprocess_flags
from cull.protocol import JsonLinesHandler, emit
from cull.scorer import SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
import time

log = logging.getLogger(__name__)

def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.resolve()
    
    return base_path / relative_path

def select_folder(default_dir: Path) -> Path | None:
    """Prompt user to select a folder using native dialog."""
    if platform.system() == "Darwin":
        # AppleScript for native macOS picker
        # We use a system-level confirm dialog to ensure it comes to front
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
        # PowerShell for native Windows picker
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
    # Resident sidecar mode starts with no base dir (the job dir comes per-run);
    # fall back to the system temp dir so logs are still captured.
    log_dir = (base_dir or Path(tempfile.gettempdir())) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cull_{timestamp}.log"
    root_log = logging.getLogger()
    for h in root_log.handlers[:]: root_log.removeHandler(h)
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_log.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        root_log.addHandler(ch)
    log.info("Logging to %s", log_file)
    return log_file

def _build_config(args: argparse.Namespace, input_dir: Path) -> EngineConfig:
    """Map argparse Namespace → EngineConfig (shared by CLI and JSON modes)."""
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
        autocrop=not args.crop_off,
        rename=args.rename,
        workers=args.workers,
        dump_scores=Path(args.dump_scores) if args.dump_scores else None,
        label_check=args.label_check,
        label_check_dir=Path(args.label_check_dir) if args.label_check_dir else None
    )

    # Resolve model paths for bundled version
    f1_model = config.f1_model_path
    if not f1_model.exists():
        # Try finding it in bundled data
        bundled_f1 = get_resource_path(f"models/{f1_model.name}")
        if bundled_f1.exists():
            config.f1_model_path = bundled_f1
            log.debug("Using bundled F1 model: %s", bundled_f1)
    return config


def run_json_lines(args: argparse.Namespace, input_dir: Path) -> int:
    """Resident JSON Lines sidecar for the Tauri GUI.

    Reads one JSON command per line from stdin and answers on stdout (one
    JSON object per line): ``scan`` lists the shots of a directory (filled
    into the GUI pending list before any run), ``run`` executes a culling
    job (stage/group/frame/log/done/cancelled/error events), ``cancel``
    aborts a running job, ``preview`` renders a thumbnail PNG, ``quit``
    exits. The process stays resident between commands so the GUI can
    rescan directories, rerun with different parameters, and preview
    without respawning the engine.
    """
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
    # Scores of the most recent run, keyed by path — previews render with the
    # full score so detection/crop overlays survive in the GUI.
    last_scores: dict[Path, ImageScore] = {}

    def stdin_reader() -> None:
        try:
            for line in sys.stdin:
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
                log.info("sidecar command: %s", cmd.get("cmd"))
                if cmd.get("cmd") == "cancel":
                    cancel_event.set()
                    continue
                if cmd.get("cmd") == "preview":
                    # Handle previews on the reader thread: the main command
                    # loop is blocked inside do_run for the whole run, so a
                    # queued preview would only render after the run ends.
                    do_preview(cmd)
                    continue
                cmd_queue.put(cmd)
                if cmd.get("cmd") == "quit":
                    return
        except Exception:
            pass

    threading.Thread(target=stdin_reader, daemon=True).start()

    def do_scan(cmd: dict) -> None:
        directory = Path(cmd.get("dir") or input_dir)
        recursive = bool(cmd.get("recursive", args.recursive))
        try:
            shots, _standalone = CullingEngine.collect_shots(directory, recursive)
            emit({"type": "scanned", "dir": str(directory),
                  "total": len(shots),
                  "paths": {p.name: str(p) for p in shots}})
        except Exception as exc:
            emit({"type": "scan_error", "message": str(exc)})

    def do_run(cmd: dict) -> None:
        merged = argparse.Namespace(**vars(args))
        for key, value in cmd.get("config", {}).items():
            if hasattr(merged, key):
                setattr(merged, key, value)
        run_input = Path(cmd.get("dir") or merged.input_dir)
        if not run_input.is_dir():
            emit({"type": "error", "message": f"input directory not found: {run_input}"})
            return

        config = _build_config(merged, run_input)
        cancel_event.clear()
        engine = CullingEngine(config)

        root = logging.getLogger()
        previous_level = root.level
        if previous_level > logging.INFO:
            root.setLevel(logging.INFO)
        handler = JsonLinesHandler(sys.stdout)
        root.addHandler(handler)

        def progress(msg: str, p: float) -> None:
            emit({"type": "stage", "msg": msg, "pct": p})

        try:
            scores, elapsed = engine.run(progress_callback=progress,
                                         cancel_event=cancel_event)
        except Exception as exc:
            emit({"type": "error", "message": str(exc)})
            return
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)

        if cancel_event.is_set():
            last_scores.clear()
            for s in scores:
                last_scores[s.path] = s
            emit({"type": "cancelled", "count": len(scores)})
            return

        last_scores.clear()
        for s in scores:
            last_scores[s.path] = s
        total = len(scores)
        keep = sum(1 for s in scores if s.rating > 0)
        stars: dict[int, int] = {}
        for s in scores:
            if s.rating > 0:
                stars[s.rating] = stars.get(s.rating, 0) + 1
        emit({"type": "done", "elapsed": elapsed, "total": total, "keep": keep,
              "reject": total - keep, "stars": stars})

    def do_preview(cmd: dict) -> None:
        try:
            # Prefer the full score of the most recent run so the thumbnail
            # carries the detection/crop overlays; fall back to a bare score.
            score = last_scores.get(Path(cmd["path"])) or ImageScore(
                path=Path(cmd["path"]), s_sharp=0.0, s_comp=0.0,
                raw_score=0.0, rating=0)
            pil = render_pil(score, max_size=int(cmd.get("size", 520)))
        except Exception:
            pil = None
        if pil is None:
            emit({"type": "preview", "path": cmd["path"], "png": None})
        else:
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            emit({"type": "preview", "path": cmd["path"],
                  "png": base64.b64encode(buf.getvalue()).decode("ascii")})

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


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1

    if args.json_lines:
        return run_json_lines(args, input_dir)

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

    # Summary statistics
    n_total   = len(all_scores)
    n_reject  = sum(1 for s in all_scores if s.rating == -1)
    n_keep    = n_total - n_reject
    ips       = n_total / elapsed if elapsed > 0 else float("inf")

    log.info(
        "\nDone in %.1fs  (%.1f img/s)  total=%d  keep=%d  reject=%d",
        elapsed, ips, n_total, n_keep, n_reject,
    )

    rating_dist: dict[int, int] = {}
    for s in all_scores:
        rating_dist[s.rating] = rating_dist.get(s.rating, 0) + 1
    for r in sorted(rating_dist):
        label = "Rejected" if r == -1 else f"{r}*"  # ASCII only, avoids GBK full-width artifacts
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

    parser.add_argument("--input-dir", type=Path, default=None, help="Directory to process. If omitted, a folder picker will open.")
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
    parser.add_argument("--p4-policy", choices=["always", "never", "auto"], default="always")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--dump-scores", type=str, default=None)
    parser.add_argument("--label-check", action="store_true")
    parser.add_argument("--label-check-dir", type=Path, default=None)
    parser.add_argument("--scale-width", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json-lines", action="store_true",
                        help="JSON Lines sidecar mode for the Tauri GUI (events on stdout, "
                             "cancel/preview commands on stdin)")

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # The resident sidecar starts with --json-lines and NO --input-dir; the
    # directory comes per-run via the `run` command, so JSON mode must bypass
    # both the interactive folder picker and the input-dir check.
    if args.json_lines:
        return run_json_lines(args, Path(args.input_dir) if args.input_dir else None)

    # CLI mode without --input-dir: show help rather than launching a GUI dialog
    if args.input_dir is None:
        parse_args(["--help"])
        return 0

    if not args.json_lines:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
    return run(args)

if __name__ == "__main__":
    sys.exit(main())
