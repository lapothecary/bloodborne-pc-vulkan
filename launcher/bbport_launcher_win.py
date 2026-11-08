#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Thin wrapper delegating to root Launcher.py."""

from pathlib import Path
import runpy
import sys

ROOT_LAUNCHER = Path(__file__).resolve().parent.parent / "Launcher.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_LAUNCHER.parent))
    runpy.run_path(str(ROOT_LAUNCHER), run_name="__main__")
