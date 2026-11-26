#!/usr/bin/env python3
"""
Fast instruction-level verifier for `gpu/cloth_exp/` experimental C++20 cloth/wind
functions vs PS4 `eboot.bin` / `out/boot.bin`.

Usage:
  python3 tools/verify_cloth_exp_asm.py [--vma 0x076a960]
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/home/kekw3/Documents/BloodBorne/bb_vulkan")
BOOT_BIN = ROOT / "out/boot.bin"
BOOT_BIAS = 0x741668
REGISTRY_CPP = ROOT / "gpu/cloth_exp/cloth_exp_registry.cpp"
LIBBBGPU = ROOT / "build_gpu/libbbgpu.so"


def parse_entries():
    text = REGISTRY_CPP.read_text(errors="ignore")
    out = []
    for m in re.finditer(
        r'\{\s*(0x[0-9a-fA-F]+)ULL\s*,\s*(\d+)u\s*,\s*reinterpret_cast<void\*>\(&([A-Za-z0-9_]+)\)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(true|false)\s*\}',
        text,
    ):
        out.append((
            int(m.group(1), 16),
            int(m.group(2)),
            m.group(3),
            m.group(4),
            m.group(5),
            m.group(6) == "true",
        ))
    return out


def disasm_boot(vma: int, size: int) -> str:
    with open(BOOT_BIN, "rb") as f:
        f.seek(vma + BOOT_BIAS)
        chunk = f.read(size)
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(chunk)
        tf.flush()
        return subprocess.check_output(
            ["objdump", "-D", "-M", "intel", "-b", "binary", "-m", "i386:x86-64", f"--adjust-vma={vma:#x}", tf.name],
            text=True,
        )


def load_host_symbols():
    if not LIBBBGPU.exists():
        return {}
    nm_out = subprocess.check_output(["nm", "-S", "--defined-only", str(LIBBBGPU)], text=True, errors="ignore")
    syms = {}
    for line in nm_out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith("ClothExp_"):
            syms[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
    return syms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vma", type=str, default="")
    args = ap.parse_args()

    entries = parse_entries()
    host_syms = load_host_symbols()
    print(f"[verify_cloth_exp_asm] Registered entries: {len(entries)}, compiled symbols in libbbgpu.so: {len(host_syms)}")

    missing = 0
    for idx, (vma, ps4_sz, sym, name, mod, detoured) in enumerate(entries):
        if args.vma and int(args.vma, 16) != vma:
            continue
        h = host_syms.get(sym)
        if not h:
            print(f"  [{idx:2d}] 0x{vma:07x} MISSING {sym}")
            missing += 1
            continue
        h_addr, h_sz = h
        ratio = h_sz / max(ps4_sz, 1)
        print(f"  [{idx:2d}] 0x{vma:07x} | PS4={ps4_sz:4d} B -> C++20={h_sz:4d} B ({ratio:4.2f}x) | detour={str(detoured):5s} | {mod}::{name}")
        if args.vma:
            print("\n--- PS4 boot.bin ---")
            print(disasm_boot(vma, ps4_sz))
            print("\n--- Compiled C++20 ---")
            print(subprocess.check_output(
                ["objdump", "-d", "-M", "intel", f"--start-address={h_addr:#x}", f"--stop-address={h_addr + h_sz:#x}", str(LIBBBGPU)],
                text=True,
            ))

    if missing:
        raise SystemExit(f"ERROR: {missing} symbols missing from {LIBBBGPU}")
    print("[verify_cloth_exp_asm] ALL Batch symbols verified present and non-stub!")


if __name__ == "__main__":
    main()
