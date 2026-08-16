# Auto-Culling — Agent Rules

## Project Overview

F1 motorsport burst-photo auto-culling tool. Groups burst shots via EXIF timestamps, scores them with a multi-stage pipeline (sharpness → composition → orientation/integrity), and writes Lightroom-compatible XMP sidecars with ratings and auto-crops. Runtime is ONNX Runtime + Pillow — **no Torch, no OpenCV** — and ships as a PyInstaller single-file executable. The torch-based ImageNet fine-tuning pipeline from earlier versions no longer exists; do not resurrect its artifacts.

## Language

- All code, comments, and docstrings: **English**
- All communication with the user: **Chinese**

## Environment

- Windows 10 (win32), shell is Git Bash; project root: `E:\Users\Au3C2\Documents\code\auto_culling`
- Python 3.10 (`.python-version`), managed by `uv` at `~/.local/bin/uv`
- venv is at `.venv/` (Windows layout): `source .venv/Scripts/activate`, or `uv run <script>`
- Code must stay Windows/macOS portable — the spec file builds for both platforms

## Tooling Conventions

- Path operations: always `pathlib.Path`, never string concatenation
- CLI arguments: always `argparse` with `ArgumentDefaultsHelpFormatter`
- No `cd <dir> && <cmd>` — use a workdir parameter or activate the venv instead

## Architecture

Core package is `cull/`; both the CLI and the packaged app drive it through `CullingEngine` (`cull/engine.py`, config via `EngineConfig`):

| Module | Role |
|---|---|
| `cull/engine.py` | `CullingEngine`, `EngineConfig` — scan, burst grouping, pipeline orchestration |
| `cull/loader.py` | Decode: HIF via FFmpeg, RAW via ExifTool, cooked formats via Pillow. NO OpenCV |
| `cull/exif_reader.py` | EXIF metadata + burst grouping via exiftool |
| `cull/detector.py` | ONNX YOLO detectors: `models/f1_yolov8n.onnx` (F1) + `models/yolov8n.onnx` (COCO) |
| `cull/p4_classifier.py`, `cull/fence_classifier.py` | ONNX classification models (orientation/integrity, fence veto) |
| `cull/scorer.py` | Scoring constants: `SHARP_THRESH=0.05`, `W_SHARP=1.5`, `W_COMP=2.5`, `MIN_RAW=3.1` |
| `cull/cropper.py`, `cull/xmp_writer.py`, `cull/xmp_reader.py`, `cull/renamer.py` | Auto-crop, XMP sidecar I/O, rename |
| `cull/gui/` | GUI helpers shared by both frontends: `worker.py` (in-process queue for the CustomTkinter app), `preview.py` (overlay thumbnails, also used by the sidecar), `settings.py`, `log_hook.py`, `app.py` (CustomTkinter window) |

Scoring: `raw_score = 1.5·S_sharp + 2.5·S_comp − penalty`. Veto rules (auto-reject): no target detected, sharpness < 0.05, car orientation is "Rear", or score < 3.1.

`CullingEngine.run(progress_callback=None, cancel_event=None)` — optional `threading.Event`; once set, scoring stops between frames and all side effects (XMP writes / metadata sync) are skipped, returning the partial scores. The CLI passes no event; the GUI uses it for its Stop button.

ONNX provider priority (`preferred_providers()` in `cull/detector.py`, reused by `p4_classifier.py`): win32 → CUDA → CPU; darwin → MLX → CoreML → CPU; other → CPU. The list is filtered against `ort.get_available_providers()`, so missing backends degrade to CPU automatically. Runtime deps in `pyproject.toml` are platform-conditional: `onnxruntime-gpu` + `nvidia-cudnn-cu12` (cuDNN 9 DLLs — ORT's GPU wheel bundles cublas/cudart but NOT cuDNN) on win32, `onnxruntime` elsewhere. `ensure_nvidia_runtime_on_path()` (`cull/detector.py`) prepends the `site-packages/nvidia/*/bin` DLL dirs to PATH before session creation — required for CUDA to activate from source; no-op in the packaged exe.

GUI specifics: the **Tauri 2 shell** (`src-tauri/`, frontend `ui/`) is the primary desktop UI — vanilla HTML/CSS/JS, no bundler; it spawns the CLI (`cull_photos.py --json-lines`) as a bundled sidecar and forwards its JSON Lines events to the webview. The CustomTkinter app (`cull_gui.py`) remains as a lightweight fallback. Both GUIs parse engine log lines (`Processing Group %d/%d (%d frames)` + per-frame rating lines) for fine-grained progress — the regexes live in `cull/protocol.py` (shared by `cull/gui/worker.py` and the sidecar `JsonLinesHandler`); `FRAME_RE` captures the veto reason greedily because it may contain nested parentheses (e.g. `(raw=2.728 < min_raw=3.100 (cut penalty applied))`). Phase progress is remapped via `_STAGE_SCALE` (0.35→0.96 spans the scoring phase). All subprocesses (exiftool, ffmpeg, folder picker) are launched with `CREATE_NO_WINDOW` + `stdin=DEVNULL` on Windows (`subprocess_flags(stdin_devnull=False)` in `cull/loader.py` when the call feeds the child via `input=`, as `exif_reader` does) — DEVNULL is required because children spawned from the Tauri sidecar inherit the GUI's stdin pipe, which never delivers data and makes perl/exiftool block forever.

Tauri gotchas (all found by live debugging): the global bundle exposes `event` at the TOP level (`window.__TAURI__.event.listen`), not under `core`; dialog `blocking_*` APIs must run via `tauri::async_runtime::spawn_blocking` (they call `block_on` internally and deadlock on the async runtime); `CommandChild::write` must append `\n` (the sidecar reads stdin line by line); the `preview` command must release the `preview_waiters` mutex before waiting on the reply channel or the read_loop deadlocks; a retired sidecar's `Terminated` event must not clear a newer spawn (guard with a generation counter — dropping `CommandChild` closes the stdin pipe and the new sidecar exits after its run). `gui.log` next to the exe receives Rust-side diagnostics. Windows GUI smoke tests use `Start-Process -WindowStyle Minimized` so the window never steals focus (e.g. from a fullscreen game).

Sidecar protocol (`cull_photos.py --json-lines`): stdout carries one JSON object per line — `total`, `paths`, `stage`, `group`, `frame`, `log`, `done`, `cancelled`, `error`; stdin accepts the line `cancel` (abort) and JSON commands `{"cmd":"preview","path":...,"size":...}` / `{"cmd":"quit"}`. After `done`/`cancelled` the process stays alive to answer preview requests (PNG base64), so the GUI never respawns the engine. Rust side (`src-tauri/src/main.rs`): `start_run`/`stop_run`/`preview`/`pick_directory`/`export_csv` commands; frame events are accumulated for CSV export; the sidecar is killed on app exit. `cull/gui/__init__.py` must stay dependency-free so the sidecar can import `cull.gui.preview` without customtkinter.

## Entry Point & CLI

- `cull_photos.py` — `parse_args()` maps argparse Namespace → `EngineConfig`. Omit `--input-dir` to open a native folder picker.
- Key flags: `--workers`, `--scale-width` (decode downscale), `--top-n`, `--force`, `--p4-policy` (`always`/`never`/`auto`), `--dump-scores`, `--dry-run`, `--rename`, `--label-check`.

## Packaging (PyInstaller) — gotchas

- `cull_photos.spec`: single-file CLI build bundling `models/*.onnx` + `external/exiftool/*`; output is `dist/cull_photos/auto_cull_v0.1_win_x64.exe` (win) / `auto_cull` (mac). Build with `pyinstaller cull_photos.spec --noconfirm`.
- `cull_gui.spec`: single-file CustomTkinter fallback GUI (entry `cull_gui.py`, `console=False`). Unlike the CLI spec it must NOT exclude `tkinter`/`PIL._imagingtk`; it adds `PIL.ImageTk` to hiddenimports, `collect_data_files('customtkinter')` for theme assets, and keeps `PySide6`/`PyQt5`/torch/cv2 excluded. Output: `dist/auto_cull_gui_v0.1_win_x64.exe` (win) / `auto_cull_gui` (mac).
- `cull_sidecar.spec`: windowed (`console=False`) single-file build of `cull_photos.py` for the Tauri sidecar; excludes customtkinter/darkdetect and strips GPU binaries. Output `dist/cull_sidecar.exe` must be copied to `src-tauri/binaries/cull-sidecar-<host-triple>.exe` before `tauri build`.
- Tauri build: `tauri build --no-bundle` (frontend `ui/` is static; `src-tauri/tauri.conf.json` sets `externalBin` to the sidecar). On Windows the host triple is `x86_64-pc-windows-gnu` when using the mingw toolchain (`rustup toolchain install stable-x86_64-pc-windows-gnu`; add the WinLibs mingw `bin/` dir to PATH). Output: `src-tauri/target/release/auto-culling.exe` + `cull-sidecar.exe`.
- Bundled-asset path resolution: `get_resource_path()` (handles `sys._MEIPASS` vs repo root) is **duplicated** in `detector.py`, `loader.py`, `p4_classifier.py`, `exif_reader.py`. Always resolve bundled assets (exiftool, ffmpeg, models) through it — never assume CWD.
- Tool lookup order: exiftool — bundled `external/exiftool/` → system PATH; ffmpeg — bundled `external/ffmpeg/` → Windows fallback `D:\ProgramData\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe` → system PATH.
- Never add torch/torchvision/ultralytics/cv2 imports to `cull/` or the entry scripts — the specs explicitly exclude them to keep the binary < 100 MB; `tests/test_package.py` verifies the packaged CLI binary without them.
- Both specs strip CUDA/TensorRT provider binaries (`_GPU_BIN_RE` on `a.binaries`): the onnxruntime-gpu wheel ships a 366MB CUDA provider DLL and PyInstaller then follows it into the system CUDA install (cublasLt/cublas/cufft/cudart — ~800MB total) that cannot load without a system cuDNN 9 anyway. The packaged exes are therefore CPU-only (provider list degrades gracefully); GPU runs apply to source mode. Keep the regex broad enough to catch cufft/curand/cusolver etc. or the exe balloons to >1GB.
- `tests/test_package.py` runs the built CLI executable (looked up as `auto_cull_v0.1_win_x64.exe` in the project root) against `tests/test_img/*.jpg` — run it after any spec/packaging change. It decodes subprocess output with `errors="replace"` because the binary may emit console-encoded (GBK) bytes on Windows.

## Tests

- `pytest tests/` — everything: `test_cull.py` (CLI end-to-end via `--dump-scores` CSV vs the BASELINE in `tests/test_cull.py`, shared with `test_package.py`), `test_cancel.py` (engine cancellation semantics), `test_json_protocol.py` (sidecar JSON Lines: full run, cancel via stdin, preview/quit loop), `test_gui_settings.py` / `test_gui_worker.py` (CustomTkinter GUI logic, no display needed), `test_gui_app.py` (view layer: real CustomTkinter window driven by event-loop pumping).
- GUI view-layer tests need a display and skip without one (headless CI: `xvfb-run pytest tests/`).
- `pyproject.toml` sets pytest `addopts = "--capture=sys"` — do not remove it: fd-level capture makes Tcl/Tk initialization fail intermittently on Windows.
- XMP sidecars are never committed (`.gitignore`); tests use the scores-CSV baseline.

## Training & Data (separate from runtime)

- `train/*.py` — torch/ultralytics scripts: `train_f1_yolo.py` (YOLO detectors), fence classifier, orientation classifier, P4 multitask. Torch lives only here, not in `pyproject.toml` runtime deps.
- `datasets/` — Roboflow YOLO datasets; `models/` — ONNX weights + CoreML packages (`f1_yolov8n.mlpackage` for macOS ANE).
- `results/comparison_report.md` — benchmark reports (macOS CoreML vs ONNX, Windows CUDA throughput). Update rather than duplicate.

## Docs

| File | Purpose |
|---|---|
| `cull_photos.py` | CLI entry point (argparse; omit `--input-dir` for a native folder picker) |
| `cull_gui.py` | CustomTkinter GUI entry point (fallback; no args) |
| `cull/engine.py` | `CullingEngine`, `EngineConfig` |
| `cull/protocol.py` | Shared JSON Lines protocol (regexes + `JsonLinesHandler`) |
| `cull/gui/` | GUI helpers: worker, preview, settings (shared by both frontends) |
| `ui/` + `src-tauri/` | Tauri 2 shell: static frontend + Rust app (primary GUI) |
| `cull_photos.spec` | PyInstaller config for the CLI binary |
| `cull_sidecar.spec` | PyInstaller config for the windowed Tauri sidecar |
| `cull_gui.spec` | PyInstaller config for the CustomTkinter GUI binary (console=False) |
| `external/exiftool`, `external/ffmpeg` | Bundled binaries (dev copies live outside git) |
| `models/*.onnx` | Production models |
| `tests/` | pytest suite (CLI + GUI + cancellation) |
| `README.md` / `README_zh.md` | User docs: usage, scoring logic, packaging, GUI |
| `results/comparison_report.md` | Benchmark / experiment comparison |
