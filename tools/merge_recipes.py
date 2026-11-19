#!/usr/bin/env python3
"""
Merge, deduplicate, inspect, and compress Bloodborne PC Vulkan shader/PSO recipe catalogs
(`*.bbrc` / `unknown_recipes.bbrc`) into a compact zlib-compressed `data/shader_recipes.bbrc`.

100% DMCA-clean: records contain only mathematical GPU specialization state & Vulkan pipeline
keys (zero Sony shader bytecode, zero SPIR-V, zero game assets).

Usage:
  # Inspect a catalog or captured unknown_recipes.bbrc:
  python3 tools/merge_recipes.py --inspect data/shader_recipes.bbrc
  python3 tools/merge_recipes.py --inspect logs/recipes/unknown_recipes.bbrc

  # Merge any number of files or directories (e.g., player-submitted logs/ folders) into catalog:
  python3 tools/merge_recipes.py -o data/shader_recipes.bbrc data/shader_recipes.bbrc logs/recipes/ path/to/friend_logs/
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

MAGIC_RAW = b"BBRC"
MAGIC_ZLIB = b"BBRZ"
HEADER_FMT = "<4sIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 16 bytes
RECORD_FMT = "<BBHII"
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 12 bytes

KIND_STAGE = 1
KIND_GFX_PSO = 2
KIND_CS_PSO = 3


def fnv1a_32(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def load_bbrc_file(path: Path) -> Tuple[Tuple[int, int, int], List[Tuple[int, bytes]], int, int]:
    """Return ((fmt_ver, meta_ver, key_ver), [(kind, payload), ...], skipped_bad, raw_bytes_len)."""
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path}: file too small ({len(raw)} bytes)")
    magic, fmt_ver, meta_ver, key_ver = struct.unpack_from(HEADER_FMT, raw, 0)
    if magic == MAGIC_RAW:
        stream = raw[HEADER_SIZE:]
    elif magic == MAGIC_ZLIB:
        if len(raw) < HEADER_SIZE + 4:
            raise ValueError(f"{path}: truncated BBRZ payload")
        expected_len = struct.unpack_from("<I", raw, HEADER_SIZE)[0]
        stream = zlib.decompress(raw[HEADER_SIZE + 4 :])
        if len(stream) != expected_len:
            raise ValueError(f"{path}: inflated size mismatch ({len(stream)} != {expected_len})")
    else:
        raise ValueError(f"{path}: unknown magic {magic!r}")

    records: List[Tuple[int, bytes]] = []
    skipped = 0
    pos = 0
    end = len(stream)
    while pos + RECORD_SIZE <= end:
        kind, _flags, _res, payload_bytes, expected_crc = struct.unpack_from(RECORD_FMT, stream, pos)
        pos += RECORD_SIZE
        if payload_bytes == 0 or pos + payload_bytes > end:
            skipped += 1
            break
        payload = stream[pos : pos + payload_bytes]
        pos += payload_bytes
        if fnv1a_32(payload) != expected_crc:
            skipped += 1
            continue
        records.append((kind, payload))

    return (fmt_ver, meta_ver, key_ver), records, skipped, len(stream)


def record_dedup_key(kind: int, payload: bytes) -> Tuple[int, int] | None:
    if kind == KIND_STAGE:
        # Layout: u64 pgm_hash, u64 perm_hash, u64 spec_digest, ...
        if len(payload) < 24:
            return None
        spec_digest = struct.unpack_from("<Q", payload, 16)[0]
        return (KIND_STAGE, spec_digest)
    if kind in (KIND_GFX_PSO, KIND_CS_PSO):
        # Layout: u64 key_hash, ...
        if len(payload) < 8:
            return None
        key_hash = struct.unpack_from("<Q", payload, 0)[0]
        return (kind, key_hash)
    return None


def collect_inputs(paths: List[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith(".bbrc"):
                        found.append(Path(root) / fn)
    return found


def write_bbrc_file(
    out_path: Path,
    versions: Tuple[int, int, int],
    records: List[Tuple[int, bytes]],
    compress: bool = True,
) -> Tuple[int, int]:
    buf = bytearray()
    for kind, payload in records:
        buf.extend(struct.pack(RECORD_FMT, kind, 0, 0, len(payload), fnv1a_32(payload)))
        buf.extend(payload)
    raw_bytes = bytes(buf)
    fmt_ver, meta_ver, key_ver = versions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        comp = zlib.compress(raw_bytes, level=9)
        header = struct.pack(HEADER_FMT, MAGIC_ZLIB, fmt_ver, meta_ver, key_ver)
        out_data = header + struct.pack("<I", len(raw_bytes)) + comp
    else:
        header = struct.pack(HEADER_FMT, MAGIC_RAW, fmt_ver, meta_ver, key_ver)
        out_data = header + raw_bytes
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(out_data)
    tmp.replace(out_path)
    return len(raw_bytes), len(out_data)


def count_kinds(records: List[Tuple[int, bytes]]) -> Tuple[int, int, int, int]:
    st = gfx = cs = extra_perms = 0
    for kind, payload in records:
        if kind == KIND_STAGE:
            st += 1
            if len(payload) >= 28:
                perm_idx = struct.unpack_from("<I", payload, 24)[0]
                if perm_idx >= 1:
                    extra_perms += 1
        elif kind == KIND_GFX_PSO:
            gfx += 1
        elif kind == KIND_CS_PSO:
            cs += 1
    return st, extra_perms, gfx, cs


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge or inspect Bloodborne shader/PSO recipe catalogs.")
    ap.add_argument("--inspect", metavar="FILE", help="Inspect a .bbrc catalog file and print stats")
    ap.add_argument("-o", "--output", metavar="OUT", help="Output merged catalog path (e.g. data/shader_recipes.bbrc)")
    ap.add_argument("--raw", action="store_true", help="Write uncompressed BBRC instead of zlib BBRZ")
    ap.add_argument("inputs", nargs="*", help="Input .bbrc files or directories to scan recursively")
    args = ap.parse_args()

    if args.inspect:
        p = Path(args.inspect)
        vers, recs, skipped, raw_len = load_bbrc_file(p)
        st, extra, gfx, cs = count_kinds(recs)
        print(
            f"{p}: format=v{vers[0]} meta_ver={vers[1]} key_ver={vers[2]} | "
            f"stages={st} (perm>=1: {extra}) gfx_psos={gfx} cs_psos={cs} skipped={skipped} | "
            f"raw={raw_len/1024:.1f} KiB file={p.stat().st_size/1024:.1f} KiB"
        )
        return 0

    if not args.output or not args.inputs:
        ap.print_help()
        return 2

    files = collect_inputs(args.inputs)
    if not files:
        print("No .bbrc files found in inputs.", file=sys.stderr)
        return 1

    target_vers: Tuple[int, int, int] | None = None
    dedup: Dict[Tuple[int, int], Tuple[int, bytes]] = {}
    order: List[Tuple[int, int]] = []

    for f in files:
        try:
            vers, recs, skipped, _ = load_bbrc_file(f)
        except Exception as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
            continue
        if target_vers is None:
            target_vers = vers
        elif vers != target_vers:
            print(
                f"[skip] {f}: version mismatch {vers} != {target_vers}",
                file=sys.stderr,
            )
            continue
        added = 0
        for kind, payload in recs:
            dk = record_dedup_key(kind, payload)
            if dk is None or dk in dedup:
                continue
            dedup[dk] = (kind, payload)
            order.append(dk)
            added += 1
        print(f"  + {f}: {len(recs)} records ({added} new, {skipped} bad/truncated)")

    if target_vers is None or not order:
        print("No valid records to write.", file=sys.stderr)
        return 1

    merged = [dedup[k] for k in order]
    out_path = Path(args.output)
    raw_sz, comp_sz = write_bbrc_file(out_path, target_vers, merged, compress=not args.raw)
    st, extra, gfx, cs = count_kinds(merged)
    print(
        f"Wrote {out_path}: stages={st} (perm>=1: {extra}), gfx_psos={gfx}, cs_psos={cs} | "
        f"raw={raw_sz/1024:.1f} KiB -> disk={comp_sz/1024:.1f} KiB "
        f"({100.0 * comp_sz / max(1, raw_sz):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
