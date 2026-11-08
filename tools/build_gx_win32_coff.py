#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Builds our_cpp_100000_source/build_win32/libbb_gx_engine_native.a (Windows x86_64 COFF archive).

Why this two-stage Clang pipeline exists:
1. All 2,940 decompiled C++20 functions in `our_cpp_100000_source/` (102,597 LOC) and their 1,496
   vtable/function-pointer calls use the PS4 Orbis / System V AMD64 ABI (`rdi, rsi, rdx, rcx, r8, r9`,
   `xmm0..7`, `rax/rdx`, `xmm6..15` caller-saved, and LP64 Itanium C++ mangling where `uint64_t`/`size_t`
   mangles as `m`).
2. Stage 1 (`clang++ -target x86_64-pc-linux-gnu -S`) compiles each `.cpp` file to x86_64 assembly
   with 100% System V AMD64 ABI and LP64 mangling, while `-DGX_TARGET_WINDOWS_MINGW=1` selects the
   Windows TEB TLS slot (`gs:[g_win32_guest_tls_teb_offset]`).
3. Stage 2 converts ELF section/symbol directives to PE/COFF directives, rewrites the 13 external
   standard C/math calls (`memcpy` -> `__gx_sysv_memcpy`, etc.) so they go through `sysv_abi`
   wrappers in `gpu/shim/bb_golden_engine.cpp`, and assembles each file into a native Windows
   `coff-x86-64` `.obj` via `clang++ -target x86_64-w64-mingw32 -c`.
4. Finally, it generates `gx_win32_external_trampolines.obj` for all external `eboot.bin` guest
   imports not defined in the 48 engine translation units and packs `libbb_gx_engine_native.a`.
"""

import concurrent.futures
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GX_DIR = ROOT / "our_cpp_100000_source"
INC_DIR = GX_DIR / "include"
BUILD_WIN32 = GX_DIR / "build_win32"
OBJ_DIR = BUILD_WIN32 / "obj"

CLANGXX = os.environ.get("CLANGXX", "clang++")
LLVM_AR = os.environ.get("LLVM_AR", "llvm-ar")
LLVM_NM = os.environ.get("LLVM_NM", "llvm-nm")

SYSV_LIBC_REWRITES = {
    "memcpy": "__gx_sysv_memcpy",
    "memmove": "__gx_sysv_memmove",
    "memset": "__gx_sysv_memset",
    "memcmp": "__gx_sysv_memcmp",
    "bcmp": "__gx_sysv_memcmp",
    "strcmp": "__gx_sysv_strcmp",
    "getenv": "__gx_sysv_getenv",
    "strtoul": "__gx_sysv_strtoul",
    "__isoc23_strtoul": "__gx_sysv_strtoul",
    "cosf": "__gx_sysv_cosf",
    "sinf": "__gx_sysv_sinf",
    "tanf": "__gx_sysv_tanf",
    "exp2f": "__gx_sysv_exp2f",
    "log2f": "__gx_sysv_log2f",
    "powf": "__gx_sysv_powf",
}

CALL_JMP_RE = re.compile(
    r"^(\s*(?:callq|jmp|jmpq)\s+)([A-Za-z0-9_$.]+)(?:@PLT)?(\s*(?:#.*)?)$"
)
WEAK_RE = re.compile(r"^\s*\.weak\s+([A-Za-z0-9_$.]+)\s*$")
GLBL_RE = re.compile(r"^\s*\.globl\s+([A-Za-z0-9_$.]+)\s*$")


def convert_elf_asm_to_coff(asm_text: str, drop_weak_defs: set[str]) -> str:
    lines = asm_text.splitlines()
    out: list[str] = []
    skipping_func = False

    for line in lines:
        s = line.strip()

        # Check if we are entering a function label that should be dropped (duplicate weak stub)
        if s.endswith(":") and not s.startswith("."):
            label = s[:-1].strip()
            if label in drop_weak_defs:
                skipping_func = True
                continue
            else:
                skipping_func = False

        if skipping_func:
            continue

        if s.startswith(
            (
                ".type\t",
                ".type ",
                ".size\t",
                ".size ",
                ".ident\t",
                ".ident ",
                ".section\t.note",
                '.section\t".note',
                ".addrsig",
                ".file\t",
                ".file ",
            )
        ):
            continue
        if s.startswith(
            (
                ".hidden\t",
                ".hidden ",
                ".protected\t",
                ".protected ",
                ".internal\t",
                ".internal ",
                ".local\t",
                ".local ",
            )
        ):
            continue

        m_weak = WEAK_RE.match(line)
        if m_weak:
            sym = m_weak.group(1)
            if sym in drop_weak_defs or sym.startswith(("_ZN12gx_direct_vk", "_ZGVZ", "_ZZ")):
                continue
            out.append(f"\t.globl\t{sym}")
            continue

        m_glbl = GLBL_RE.match(line)
        if m_glbl and m_glbl.group(1) in drop_weak_defs:
            continue

        if s.startswith((".section\t.rodata", ".section .rodata", ".section\t.data.rel.ro", ".section .data.rel.ro")):
            out.append('\t.section\t.rdata,"dr"')
            continue
        if s.startswith((".section\t.text.", ".section .text.")):
            out.append("\t.text")
            continue
        if s.startswith((".section\t.data.", ".section .data.")):
            out.append("\t.data")
            continue
        if s.startswith((".section\t.bss.", ".section .bss.")):
            out.append("\t.bss")
            continue
        if s.startswith((".section\t.init_array", ".section .init_array")):
            out.append('\t.section\t.CRT$XCU,"dr"')
            continue

        m_call = CALL_JMP_RE.match(line)
        if m_call:
            prefix, target, suffix = m_call.groups()
            target = SYSV_LIBC_REWRITES.get(target, target)
            out.append(f"{prefix}{target}{suffix}")
            continue

        out.append(line.replace("@PLT", ""))

    return "\n".join(out) + "\n"


def compile_one_source(src_path: Path, index: int) -> Path:
    stem = f"{index:02d}_{src_path.stem}"
    s_elf = OBJ_DIR / f"{stem}.elf.s"
    s_coff = OBJ_DIR / f"{stem}.coff.s"
    o_coff = OBJ_DIR / f"{stem}.obj"

    hdr = INC_DIR / "gx_runtime_vma_and_tls.h"
    cmd_s = [
        CLANGXX,
        "-target",
        "x86_64-pc-linux-gnu",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-march=x86-64-v3",
        "-mtune=znver4",
        "-fno-omit-frame-pointer",
        "-fshort-wchar",
        "-fno-strict-aliasing",
        "-fno-stack-protector",
        "-fno-threadsafe-statics",
        "-fno-exceptions",
        "-fno-asynchronous-unwind-tables",
        "-ffp-contract=off",
        "-mno-red-zone",
        "-fPIC",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-Wno-unused-parameter",
        "-Wno-unused-function",
        "-Wno-address",
        "-Wno-address-of-packed-member",
        "-DGX_TARGET_WINDOWS_MINGW=1",
        "-I",
        str(INC_DIR),
        "-include",
        str(hdr),
        "-S",
        str(src_path),
        "-o",
        str(s_elf),
    ]
    subprocess.run(cmd_s, check=True)

    # Drop duplicate weak dummy definitions in GXGnmTextureAndShaderStageInitializer.cpp
    drop_weaks = {"__cyg_profile_func_enter", "__cyg_profile_func_exit"}
    if src_path.name in (
        "GXGnmCommandContextAndDeviceFactory.cpp",
        "GXGnmTextureAndShaderStageInitializer.cpp",
    ):
        drop_weaks.update(
            {
                "BbGoldenEngine_NotifyGpuSurfaceWrite",
                "BbGoldenEngine_OnDrawCall",
                "BbGoldenEngine_OnDispatchCall",
                "BbGoldenEngine_UpgradeResourceCache",
            }
        )

    coff_asm = convert_elf_asm_to_coff(s_elf.read_text(), drop_weaks)
    s_coff.write_text(coff_asm)

    cmd_c = [
        CLANGXX,
        "-target",
        "x86_64-w64-mingw32",
        "-c",
        str(s_coff),
        "-o",
        str(o_coff),
    ]
    subprocess.run(cmd_c, check=True)
    return o_coff


def parse_external_imports_header() -> tuple[dict[str, int], dict[str, int]]:
    """Returns (func_trampolines: sym -> vma, data_trampolines: sym -> vma) from gx_eboot_external_imports.h."""
    hdr_text = (INC_DIR / "gx_eboot_external_imports.h").read_text()
    func_map: dict[str, int] = {}
    data_map: dict[str, int] = {}

    entry_re = re.compile(
        r"\{\s*0x([0-9a-fA-F]+)ULL\s*,\s*\"([^\"]+)\"\s*,\s*GXExternalSubsystem::[A-Za-z0-9_]+\s*,\s*(true|false)\s*,"
    )
    for m in entry_re.finditer(hdr_text):
        vma = int(m.group(1), 16)
        sym = m.group(2)
        is_data = m.group(3) == "true"
        if vma == 0 or sym in (
            "CGMemoryGnm_Memcpy",
            "_ZN19GXGraphicsFramework29CGMemoryGnm_FastCopyGpuMemoryEPvPKvj",
        ):
            continue
        if is_data:
            data_map[sym] = vma
        else:
            func_map[sym] = vma
    return func_map, data_map


def main() -> int:
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path(p) for p in glob.glob(str(GX_DIR / "src/**/*.cpp"), recursive=True))
    if len(sources) != 48:
        print(f"ERROR: expected 48 source files, found {len(sources)}", file=sys.stderr)
        return 1

    print(f"[build_gx_win32_coff] Compiling {len(sources)} C++20 engine files to Windows COFF64...")
    obj_files: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        futures = [ex.submit(compile_one_source, src, idx) for idx, src in enumerate(sources)]
        for fut in concurrent.futures.as_completed(futures):
            obj_files.append(fut.result())
    obj_files.sort()

    # Check defined and undefined symbols across all 48 COFF .obj files
    defined: set[str] = set()
    undefined: set[str] = set()
    defined_owner: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []

    for obj in obj_files:
        out = subprocess.check_output([LLVM_NM, str(obj)], text=True)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            sym = parts[-1]
            kind = parts[-2]
            if sym.startswith("."):
                continue
            if kind in ("T", "D", "B", "R", "C", "S"):
                if sym in defined and defined_owner[sym] != obj.name:
                    duplicates.append((sym, defined_owner[sym], obj.name))
                defined.add(sym)
                defined_owner[sym] = obj.name
            elif kind == "U":
                undefined.add(sym)

    if duplicates:
        print(f"[build_gx_win32_coff] Found {len(duplicates)} duplicate strong symbols across .obj files:")
        for sym, o1, o2 in duplicates[:20]:
            print(f"  {sym}: {o1} vs {o2}")

    # Generate COFF assembly for all external eboot.bin trampolines needed by the 48 .obj files
    func_map, data_map = parse_external_imports_header()
    needed_ext = sorted(undefined - defined)

    tramp_lines = [
        "\t.text",
        "\t.globl\tg_eboot_live_slide",
        "\t.globl\tg_win32_guest_tls_teb_offset",
        "\t.globl\tg_gx_offline_scratch",
        "\t.globl\tg_gx_offline_tls_area",
    ]
    tramp_count = 0
    for sym in needed_ext:
        vma = func_map.get(sym) or data_map.get(sym)
        if vma is not None:
            tramp_lines.extend(
                [
                    f"\t.globl\t{sym}",
                    "\t.p2align\t4",
                    f"{sym}:",
                    "\tmovq\tg_eboot_live_slide(%rip), %r11",
                    "\ttestq\t%r11, %r11",
                    "\tjz\t1f",
                    f"\taddq\t$0x{vma:x}, %r11",
                    "\tjmpq\t*%r11",
                    "1:",
                    "\txorl\t%eax, %eax",
                    "\tretq",
                ]
            )
            tramp_count += 1

    # Also emit global state variables in .data / .bss
    tramp_lines.extend(
        [
            "\t.data",
            "\t.p2align\t3",
            "g_eboot_live_slide:",
            "\t.quad\t0",
            "g_win32_guest_tls_teb_offset:",
            "\t.long\t0x1480",
            "\t.bss",
            "\t.p2align\t6",
            "g_gx_offline_scratch:",
            "\t.zero\t65536",
            "g_gx_offline_tls_area:",
            "\t.zero\t16384",
        ]
    )

    tramp_s = OBJ_DIR / "99_gx_win32_external_trampolines.coff.s"
    tramp_o = OBJ_DIR / "99_gx_win32_external_trampolines.obj"
    tramp_s.write_text("\n".join(tramp_lines) + "\n")
    subprocess.run(
        [CLANGXX, "-target", "x86_64-w64-mingw32", "-c", str(tramp_s), "-o", str(tramp_o)],
        check=True,
    )
    obj_files.append(tramp_o)

    out_lib = BUILD_WIN32 / "libbb_gx_engine_native.a"
    if out_lib.exists():
        out_lib.unlink()
    subprocess.run([LLVM_AR, "rcs", str(out_lib)] + [str(o) for o in obj_files], check=True)

    # Final audit of remaining undefined symbols in libbb_gx_engine_native.a
    defined_all: set[str] = set()
    undef_all: set[str] = set()
    out = subprocess.check_output([LLVM_NM, str(out_lib)], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sym = parts[-1]
        kind = parts[-2]
        if kind in ("T", "D", "B", "R", "C", "S"):
            defined_all.add(sym)
        elif kind == "U":
            undef_all.add(sym)

    final_ext = sorted(undef_all - defined_all)
    print(
        f"[build_gx_win32_coff] Built {out_lib} ({out_lib.stat().st_size // 1024} KiB): "
        f"{len(defined_all)} defined symbols, {tramp_count} eboot trampolines, "
        f"{len(final_ext)} host imports:"
    )
    for s in final_ext:
        print(f"  U {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
