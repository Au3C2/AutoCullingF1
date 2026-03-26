# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Exclude models from internal data to keep engine small
datas = []

# Binaries: None needed for now (exiftool assumed in path for this step, 
# but we could bundle it if we want).
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
    excludes=['torch', 'torchvision', 'ultralytics', 'opencv-python', 'cv2', 'scipy', 'matplotlib', 'pandas', 'polars', 'tkinter', 'PySide6', 'PyQt5'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cull_photos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='cull_photos',
)
