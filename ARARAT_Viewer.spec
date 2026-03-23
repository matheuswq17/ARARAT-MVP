# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, collect_data_files

root = Path(os.getcwd()).resolve()
icon_path = root / "viewer" / "assets" / "ararat_logo.ico"

hiddenimports = [
    "matplotlib.backends.backend_tkagg",
    "PIL._tkinter_finder",
    "sklearn",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
    "joblib",
    "pandas",
    "numpy",
    "SimpleITK",
    "radiomics",
    "pykwalify",
    "ruamel.yaml",
]
hiddenimports += collect_submodules("radiomics")

binaries = []
binaries += collect_dynamic_libs("SimpleITK")

datas = [
    (str(root / "viewer" / "assets"), "viewer/assets"),
    (str(root / "inference" / "models" / "v1_prostatex"), "inference/models/v1_prostatex"),
    (str(root / "data" / "PROSTATEx" / "LABELS"), "data/PROSTATEx/LABELS"),
    (str(root / "data" / "PROSTATEx" / "LABELS" / "labels_merged.csv"), "data/PROSTATEx/LABELS"),
    (str(root / "vendor" / "radiomics" / "schemas"), "vendor/radiomics/schemas"),
    (str(root / "radiomics_params.yaml"), "."),
]
datas += collect_data_files("radiomics")

a = Analysis(
    ["viewer/viewer_app.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ARARAT_Viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)
