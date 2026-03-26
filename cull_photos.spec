# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Assets to bundle: Bundle the ONNX models!
datas = [
    ('models/f1_yolov8n.onnx', 'models'),
    ('models/yolov8n.onnx', 'models'),
    ('models/p4_car_model.onnx', 'models'),
    ('external/exiftool/exiftool', 'external/exiftool'),
    ('external/exiftool/lib', 'external/exiftool/lib'),
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
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='auto_cull', # Rename to a cleaner name
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
