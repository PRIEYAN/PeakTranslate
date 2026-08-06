"""Zero-dependency sys.path bootstrap for every pipeline entrypoint.

`venv/bin/python` in this project does not self-detect as a venv (its
`sys.prefix` resolves to the base uv-managed interpreter, so it never adds
`venv/lib/.../site-packages` on its own). Without this, every run required
manually exporting PYTHONPATH first. Importing this module first — before
any third-party import — makes `venv/bin/python pipeline/run_x.py` just
work on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ensure_venv_on_path() -> None:
    site_packages = (
        ROOT / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    for p in (str(ROOT), str(site_packages)):
        if p not in sys.path:
            sys.path.insert(0, p)
