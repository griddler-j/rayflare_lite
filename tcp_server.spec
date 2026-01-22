# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

PARAM_SRC = Path(r"D:\Griddler\PV_circuit_model\PV_Circuit_Model\parameters")

# Collect every file under parameters/ recursively
datas = [(str(p), r"parameters") for p in PARAM_SRC.rglob("*") if p.is_file()]

a = Analysis(
    ['tcp_server.py'],
    pathex=[r"D:\Griddler\PV_circuit_model"],
    binaries=[],
    datas=datas,
    hiddenimports=['tqdm'],
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
    name='tcp_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
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
