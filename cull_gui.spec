# -*- mode: python ; coding: utf-8 -*-
# cull_gui.spec — PyInstaller single-file build for the CustomTkinter GUI.
# Build: pyinstaller cull_gui.spec --noconfirm
# Output: dist/auto_cull_gui_v0.1_win_x64.exe (Windows) / dist/auto_cull_gui (macOS)
import re
import sys

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

is_win = sys.platform == "win32"

# The onnxruntime-gpu wheel ships the 366MB CUDA provider DLL, and PyInstaller
# additionally follows its dependencies into the system CUDA install
# (cublas/cublasLt/cudart — hundreds of MB). These cannot load in the packaged
# app without a system cuDNN 9 install anyway, and the provider list degrades
# to CPU automatically. Strip them to keep the single-file binary small.
_GPU_BIN_RE = re.compile(
    r"(onnxruntime_providers_(cuda|tensorrt)|cublas|cublasLt|cudart|cudnn|cufft|"
    r"curand|cusolver|cusparse|nvrtc|nvjpeg|nvblas|npp[0-9_]+|nccl|nvml)", re.I)

datas = [
    ('models/f1_yolov8n.onnx', 'models'),
    ('models/yolov8n.onnx', 'models'),
    ('models/p4_car_model.onnx', 'models'),
]
if is_win:
    datas += [('external/exiftool/*', 'external/exiftool')]
else:
    datas += [
        ('external/exiftool/exiftool', 'external/exiftool'),
        ('external/exiftool/lib', 'external/exiftool/lib'),
    ]

# CustomTkinter needs its theme resources and submodules at runtime.
datas += collect_data_files('customtkinter')
hiddenimports = ['onnxruntime', 'numpy', 'PIL.Image', 'PIL.ImageTk', 'pillow_heif',
                 'pillow_avif', 'darkdetect', 'packaging']
hiddenimports += collect_submodules('customtkinter')

a = Analysis(
    ['cull_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # tkinter is required by the GUI, so it is intentionally NOT excluded here
    # (unlike cull_photos.spec). Everything else stays pruned to keep the
    # binary small.
    excludes=['torch', 'torchvision', 'ultralytics', 'opencv-python', 'cv2', 'scipy',
              'matplotlib', 'pandas', 'polars', 'PySide6', 'PyQt5', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

a.binaries = TOC([b for b in a.binaries if not _GPU_BIN_RE.search(b[0])])

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe_name = 'auto_cull_gui_v0.1_win_x64' if is_win else 'auto_cull_gui'

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
    console=False,  # GUI build: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
