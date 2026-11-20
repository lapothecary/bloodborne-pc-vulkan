#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fast incremental cache check for prepare.py + link_libc.py + link_modules.py.

Avoids re-unpacking and re-relocating the 91 MB eboot.bin / libc.prx / libSceFios2.prx
(~2.3 s) on every launch when the underlying game binaries and linker scripts are unchanged.
Exit codes (check mode):
  0 = cache is valid (skip prepare/link)
  1 = cache miss / stale (run prepare/link, then invoke with --write)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import game_check

STAMP_VERSION = 1
STAMP_NAME = '.boot_prep.stamp.json'
REQUIRED_OUTPUTS = ('boot.bin', 'boot-linked.bin', 'analysis.json', 'link.json')
GAME_INPUTS = (
    'eboot.bin',
    'sce_module/libc.prx',
    'sce_module/libSceFios2.prx',
    'sce_sys/param.sfo',
)
SCRIPT_INPUTS = (
    'prepare.py',
    'link_libc.py',
    'link_modules.py',
    'game_check.py',
    'import_scopes.py',
)


def _stat_entry(path: Path):
    resolved = path.resolve()
    st = resolved.stat()
    return [str(resolved), st.st_size, st.st_mtime_ns]


def compute_signature(game: Path) -> str:
    scripts_dir = Path(__file__).resolve().parent
    payload = {
        'v': STAMP_VERSION,
        'skip_game_check': os.environ.get('BB_SKIP_GAME_CHECK', ''),
        'game_files': {},
        'scripts': {},
    }
    for rel in GAME_INPUTS:
        p = game_check.game_file(game, rel)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        payload['game_files'][rel] = _stat_entry(p)
    for name in SCRIPT_INPUTS:
        p = scripts_dir / name
        if p.is_file():
            payload['scripts'][name] = _stat_entry(p)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def is_cache_valid(game: Path, out: Path) -> bool:
    if os.environ.get('BB_FORCE_PREPARE', '') == '1':
        return False
    stamp_path = out / STAMP_NAME
    if not stamp_path.is_file():
        return False
    for name in REQUIRED_OUTPUTS:
        p = out / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    try:
        stored = json.loads(stamp_path.read_text(encoding='utf-8')).get('sig')
        return stored == compute_signature(game)
    except Exception:
        return False


def write_stamp(game: Path, out: Path) -> None:
    sig = compute_signature(game)
    out.mkdir(parents=True, exist_ok=True)
    (out / STAMP_NAME).write_text(json.dumps({'v': STAMP_VERSION, 'sig': sig}) + '\n',
                                  encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        write_stamp(args.game, args.out)
        return 0
    return 0 if is_cache_valid(args.game, args.out) else 1


if __name__ == '__main__':
    sys.exit(main())
