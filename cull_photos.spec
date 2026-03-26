# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Assets to bundle: Bundle the ONNX models!
is_win = sys.platform == "win32"
exiftool_exe = 'external/exiftool/exiftool.exe' if is_win else 'external/exiftool/exiftool'

datas = [
    ('models/f1_yolov8n.onnx', 'models'),
    ('models/yolov8n.onnx', 'models'),
    ('models/p4_car_model.onnx', 'models'),
    (exiftool_exe, 'external/exiftool'),
    ('external/exiftool/lib', 'external/exiftool/lib'),
]

# Include other exiftool files for the launcher if on Windows
if is_win:
     # We might need the other dlls and pl files from exiftool_files which we flattened
     # Let's just bundle the whole external/exiftool directory as a safe measure
     datas = [
        ('models/f1_yolov8n.onnx', 'models'),
        ('models/yolov8n.onnx', 'models'),
        ('models/p4_car_model.onnx', 'models'),
        ('external/exiftool/*', 'external/exiftool'),
     ]

binaries = []

a = Analysis(
    ['cull_photos.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['onnxruntime', 'numpy', 'PIL.Image', 'pillow_heif', 'pillow_avif'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude EVERYTHING unnecessary to keep it <100MB
    excludes=['torch', 'torchvision', 'ultralytics', 'opencv-python', 'cv2', 'scipy', 'matplotlib', 'pandas', 'polars', 'tkinter', 'PySide6', 'PyQt5', 'IPython', 'PIL._imagingtk', 'PIL._tkinter_finder'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single File implementation
exe_name = 'auto_cull_v0.1_win_x64' if is_win else 'auto_cull'

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
