# -*- mode: python ; coding: utf-8 -*-
# cull_sidecar.spec — windowed (console=False) single-file build used as the
# Tauri sidecar. Identical to cull_photos.spec except for the subsystem, so no
# console window flashes when the GUI spawns it; stdio pipes still work.
# Build: pyinstaller cull_sidecar.spec --noconfirm
# Output: dist/cull_sidecar.exe -> src-tauri/binaries/cull-sidecar-x86_64-pc-windows-gnu.exe
import sys
import os
import re
from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Strip the CUDA/TensorRT provider binaries and system CUDA runtime DLLs —
# they cannot load without a system cuDNN 9 and would add ~800MB (see
# cull_photos.spec for the full rationale). The sidecar runs CPU-only.
_GPU_BIN_RE = re.compile(
    r"(onnxruntime_providers_(cuda|tensorrt)|cublas|cublasLt|cudart|cudnn|cufft|"
    r"curand|cusolver|cusparse|nvrtc|nvjpeg|nvblas|npp[0-9_]+|nccl|nvml)", re.I)

is_win = sys.platform == "win32"
exiftool_exe = 'external/exiftool/exiftool.exe' if is_win else 'external/exiftool/exiftool'

datas = [
    ('models/f1_yolov8n.onnx', 'models'),
    ('models/yolov8n.onnx', 'models'),
    ('models/p4_car_model.onnx', 'models'),
    (exiftool_exe, 'external/exiftool'),
    ('external/exiftool/lib', 'external/exiftool/lib'),
]

if is_win:
    datas = [
        ('models/f1_yolov8n.onnx', 'models'),
        ('models/yolov8n.onnx', 'models'),
        ('models/p4_car_model.onnx', 'models'),
        ('external/exiftool/*', 'external/exiftool'),
    ]

a = Analysis(
    ['cull_photos.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['onnxruntime', 'numpy', 'PIL.Image', 'pillow_heif', 'pillow_avif'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'ultralytics', 'opencv-python', 'cv2', 'scipy',
              'matplotlib', 'pandas', 'polars', 'tkinter', 'PySide6', 'PyQt5',
              'IPython', 'customtkinter', 'darkdetect', 'packaging',
              'PIL._imagingtk', 'PIL._tkinter_finder'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.binaries = TOC([b for b in a.binaries if not _GPU_BIN_RE.search(b[0])])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cull_sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed: the Tauri shell pipes stdio, no console flash
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
