# -*- mode: python ; coding: utf-8 -*-
"""
cull_photos.spec — single-file, dependency-free cull_photos binary.

Platform branches
-----------------
- macOS (darwin): ships the onnxsim-FROZEN graphs (f1_yolov8n_static 640px,
  yolov8n_static 640px, p4_car_model_static_ane 224px) UNDER THE PLAIN BASE
  NAMES. The darwin runtime prefers a ``*_static`` sibling when present and
  otherwise takes the base file with pinned CoreML options once its input
  dims are concrete (shape probe in cull/detector.py) — so the packaged
  pipeline runs the exact bytes the source pipeline runs. Exiftool is the
  Perl script + lib tree, executed through the macOS system Perl (ships
  with macOS, so the binary stays dependency-free).
- Windows: keeps the dynamic exports (Windows gates were locked on those)
  and the self-contained exiftool.exe.

Excludes keep the archive free of training/analysis stacks; the analysis
graph already pulls only the runtime closure (onnxruntime, cv2, av, Quartz,
scipy.fft, numpy, Pillow).
"""

import os
import shutil
import sys
from pathlib import Path

is_win = sys.platform == "win32"
# CULL_ONEDIR=1 produces the directory (unpackaged) form. The onefile form
# re-extracts to a fresh temp dir on every run, so macOS re-pays the kernel
# code-signature verification tax for every bundled Mach-O on each launch
# (~15-25 s; see packaging/build.py). The onedir form keeps stable inodes:
# the same tax is paid once per boot and cached afterwards, giving
# source-identical throughput. Both forms ship the identical data.
onedir = os.environ.get("CULL_ONEDIR") == "1"

_MODEL_STAGE = Path("build") / "_bundle_models"


def _stage_models() -> list[tuple[str, str]]:
    """(src, dest_dir) datas tuples for the three ONNX models.

    PyInstaller keeps the source basename in the archive, so frozen graphs
    are staged under the runtime base names (f1_yolov8n.onnx on darwin =
    the static 640px bytes; p4_car_model.onnx = the _static_ane bytes).
    """
    if is_win:
        return [
            ("models/f1_yolov8n.onnx", "models"),
            ("models/yolov8n.onnx", "models"),
            ("models/p4_car_model.onnx", "models"),
        ]
    _MODEL_STAGE.mkdir(parents=True, exist_ok=True)
    out = []
    for src_name, dst_name in (
        ("f1_yolov8n_static.onnx", "f1_yolov8n.onnx"),
        ("yolov8n_static.onnx", "yolov8n.onnx"),
        ("p4_car_model_static_ane.onnx", "p4_car_model.onnx"),
    ):
        src = Path("models") / src_name
        if not src.exists():
            # Fall back to the dynamic export when the frozen graph is absent
            # (dev checkouts without the latest models/ contents).
            src = Path("models") / dst_name
        dst = _MODEL_STAGE / dst_name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        out.append((str(dst), "models"))
    return out


def _exiftool_datas() -> list[tuple[str, str]]:
    """Bundled exiftool. Both platforms ship the perl-script form
    (perl.exe + exiftool.pl + lib on Windows; script + lib with system
    perl on macOS) — `_find_exiftool_path` priority 1 resolves it and the
    stay_open batch sessions (RAW extract + metadata write) work through
    the same proven invocation. The standalone exiftool.exe on this tree
    is NOT self-contained (it wants exiftool_files\\perl5*.dll) and dies
    at spawn, killing the batch metadata writer (measured 2026-08-30)."""
    if is_win:
        base = "external/exiftool"
        files = ["perl.exe", "perl532.dll", "exiftool.pl",
                 "libgcc_s_seh-1.dll", "liblzma-5__.dll",
                 "libstdc++-6.dll", "libwinpthread-1.dll"]
        return [(f"{base}/{f}", base) for f in files] \
            + [(f"{base}/lib", f"{base}/lib"),
               # win32-only perl core/XS modules (Encode, File::Glob, …),
               # split from lib/ so the darwin `-I lib` never sees them —
               # bundled perl gets both dirs via two -I flags.
               (f"{base}/lib-win32", f"{base}/lib-win32")]
    return [
        ("external/exiftool/exiftool", "external/exiftool"),
        ("external/exiftool/lib", "external/exiftool/lib"),
    ]


datas = _stage_models() + _exiftool_datas()

binaries = []

a = Analysis(
    ["cull_photos.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Training/analysis stacks and stdlib extras that the CLI never imports.
    # scipy is NOT excluded: cull/sharpness.py calls scipy.fft.rfft2
    # (0.34 ms/frame half-spectrum pocketfft; ~9x faster than cv2.dft on
    # Apple M4), and scipy.fft.__init__ hard-imports _fftlog ->
    # scipy.special -> scipy.linalg — the whole closure ships.
    excludes=[
        "torch", "torchvision", "ultralytics", "matplotlib", "pandas",
        "polars", "IPython", "tkinter", "PySide6", "PyQt5", "coremltools",
        "onnx", "sympy", "networkx", "requests", "coloredlogs", "humanfriendly",
        "PIL._imagingtk", "PIL._tkinter_finder",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

if is_win:
    exe_name = "auto_cull_v0.1_win_x64"
elif sys.platform == "darwin":
    exe_name = "auto_cull_v0.1_macos_arm64"
else:
    exe_name = "auto_cull"

# Single-file implementation: everything unpacks to a temp _MEIPASS dir.
# The onedir form keeps the bundle next to the executable (dist/<name>/).
if onedir:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=exe_name,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=exe_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )