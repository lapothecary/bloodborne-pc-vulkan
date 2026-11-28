#!/usr/bin/env python3
"""
Audit 2.0 — Automated Machine-Level Differential Auditor for dantelion_gx (Modules 01..07).

Mechanically cross-checks all 2,941 registered C++20 functions in
`dantelion_gx/build/CMakeFiles/bb_gx_engine_native.dir/src/**/*.o`
against original PS4 `eboot.bin` (`file_offset = vma + 0x4000`), plus
source-level bug-pattern scanning and optional strict clang++ warning sweeps.

Usage:
  python3 bb_vulkan/tools/audit_dantelion_gx.py [--module 01] [--warnings] [--vma 0x21916d0] [--json out.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/kekw3/Documents/BloodBorne/bb_vulkan")
EBOOT_BIN = Path("/home/kekw3/Documents/bloodborne engine decompile project/original file/eboot.bin")
EBOOT_FILE_OFFSET = 0x4000

REGISTRY_H = ROOT / "dantelion_gx/include/gx_decomp_registry.h"
IMPORTS_H = ROOT / "dantelion_gx/include/gx_eboot_external_imports.h"
GOLDEN_CPP = ROOT / "gpu/shim/bb_golden_engine.cpp"
SRC_DIR = ROOT / "dantelion_gx/src"
OBJ_DIR = ROOT / "dantelion_gx/build/CMakeFiles/bb_gx_engine_native.dir/src"
CLANGXX = Path("/home/kekw3/Documents/BloodBorne/AI_WORK_FOR_PORTING/toolchain/bin/clang++")
if not CLANGXX.exists():
    CLANGXX = Path("clang++")

# PS4 CRT / compiler-runtime helpers that clang -O3 legitimately replaces/inlines/eliminates.
CRT_IGNORE_VMAS = {
    0x2bbe4f8,  # __stack_chk_fail
    0x2bbe578,  # memcpy
    0x2bbe658,  # memmove
    0x2bbe688,  # memset
    0x2bbe518,  # sinf / cosf / math CRT
    0x2bbe6a8,  # memcmp / strcmp
    0x2bbe548,  # powf / math
    0x2bbe558,  # math
    0x2bbe568,  # math
    0x2bbe598,  # math
    0x2bbe5c8,  # math
    0x2bbe608,  # math
    0x2bbe628,  # math
    0x2bbe638,  # math
    0x2bbe678,  # math
    0x2bbe6b8,  # math
    0x2bbe6d8,  # math
    0x2bbe718,  # math
    0x2bbe748,  # math
    0x2bbe798,  # math
    0x2bbe7d8,  # math
    0x2bbe818,  # math
    0x2bbe858,  # math
    0x2bbe898,  # math
    0x2bbe8d8,  # math
    0x2bbe918,  # math
    0x2bbe958,  # math
    0x2bbe998,  # math
    0x2bbea00,  # math / libc range checked below
    0x2bc0de8,  # sceKernelReadTsc (replaced by __rdtsc())
}
CRT_VMA_RANGE = (0x2bbe000, 0x2bc1000)  # PS4 Orbis libc/libm/libkernel PLT stubs

STACK_GUARD_GLOBAL = 0x53e3be0  # __stack_chk_guard

# Known host/DirectVk symbols indicating intentional PC Vulkan fast-path additions.
DIRECT_VK_SYM_PREFIXES = (
    "bb_direct_vk_",
    "gx_direct_vk",
    "_ZN12gx_direct_vk",
    "BbGoldenEngine_",
    "bb_golden_",
    "__gx_sysv_",
    "__cyg_profile_",
    "bb_guest_tls",
)

# VMAs intentionally skipped in bb_golden_engine.cpp::ShouldInstallEntry
SKIPPED_VMAS_STATIC = {
    0x28ce7b0,
    0x28ceec0,
    0x22777d0,
    0x22c39f0,
    0x22da690,
    0x2313870,
    0x2315d40,
}

TRIVIAL_FLOATS = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0}


def is_crt_vma(vma: int) -> bool:
    return vma in CRT_IGNORE_VMAS or (CRT_VMA_RANGE[0] <= vma < CRT_VMA_RANGE[1])


def parse_registry():
    txt = REGISTRY_H.read_text(errors="ignore")
    alias_to_elf = {}
    for m in re.finditer(r'void\s+(gx_decomp_sym_[0-9a-fA-F]+)\(\)\s+__asm__\("([^"]+)"\)', txt):
        alias_to_elf[m.group(1)] = m.group(2)

    entries = []
    elf_to_vma = {}
    vma_to_entry = {}
    for m in re.finditer(
        r'\{\s*(0x[0-9a-fA-F]+)ULL\s*,\s*(\d+)u\s*,\s*reinterpret_cast<void\*>\(&(gx_decomp_sym_[0-9a-fA-F]+)\)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}',
        txt,
    ):
        vma = int(m.group(1), 16)
        size = int(m.group(2))
        alias = m.group(3)
        name = m.group(4)
        mod = m.group(5)
        elf_sym = alias_to_elf.get(alias, name)
        active = size >= 5 and vma not in SKIPPED_VMAS_STATIC
        rec = {
            "vma": vma,
            "size": size,
            "elf_sym": elf_sym,
            "name": name,
            "module": mod,
            "active": active,
        }
        entries.append(rec)
        # Map ONLY actual ELF linker symbols (never human display names, which
        # can collide with extern "C" trampoline symbols in gx_eboot_external_imports.h).
        elf_to_vma[elf_sym] = vma
        vma_to_entry[vma] = rec
    return entries, elf_to_vma, vma_to_entry


def parse_external_imports():
    txt = IMPORTS_H.read_text(errors="ignore")
    sym_to_vma = {}
    data_syms = set()
    for m in re.finditer(
        r'\{\s*(0x[0-9a-fA-F]+)ULL\s*,\s*"([^"]+)"\s*,\s*GXExternalSubsystem::[A-Za-z0-9_]+\s*,\s*(true|false)',
        txt,
    ):
        vma = int(m.group(1), 16)
        sym = m.group(2)
        is_data = m.group(3) == "true"
        if vma != 0:
            sym_to_vma[sym] = vma
            if is_data:
                data_syms.add(sym)
    return sym_to_vma, data_syms


def extract_vma_from_symbol(sym: str, elf_to_vma: dict[str, int], imp_to_vma: dict[str, int]) -> int | None:
    clean = sym.split("@")[0].split("+")[0].split("-")[0]
    if clean in elf_to_vma:
        return elf_to_vma[clean]
    if clean in imp_to_vma:
        return imp_to_vma[clean]
    m = re.match(r'^(?:sub|fn|g|g_vtable|g_str|g_rodata|g_global|loc|dat|unk)_([0-9a-fA-F]{6,7})$', clean)
    if m:
        return int(m.group(1), 16)
    m2 = re.search(r'_([0-9a-fA-F]{6,7})$', clean)
    if m2:
        v = int(m2.group(1), 16)
        if 0x00800000 <= v <= 0x05900000:
            return v
    return None


def disassemble_eboot_ranges(entries: list[dict], eboot_bytes: bytes) -> dict[int, list[tuple[int, str]]]:
    """Batch-disassemble eboot.bin by coalescing nearby functions into contiguous chunks."""
    sorted_e = sorted(entries, key=lambda e: e["vma"])
    # Coalesce spans separated by <= 64 KiB into a single objdump call
    chunks = []
    for e in sorted_e:
        s, end = e["vma"], e["vma"] + max(e["size"], 1)
        if chunks and s - chunks[-1][1] <= 0x10000:
            chunks[-1][1] = max(chunks[-1][1], end)
            chunks[-1][2].append(e)
        else:
            chunks.append([s, end, [e]])

    func_insns: dict[int, list[tuple[int, str]]] = {}

    def run_chunk(ch):
        c_start, c_end, ents = ch
        raw = eboot_bytes[c_start + EBOOT_FILE_OFFSET : c_end + EBOOT_FILE_OFFSET]
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(raw)
            tname = tf.name
        try:
            out = subprocess.check_output(
                [
                    "objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                    f"--adjust-vma={hex(c_start)}", tname,
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        finally:
            os.unlink(tname)

        # Parse (addr, asm_text) lines
        parsed = []
        line_re = re.compile(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*(.+)$')
        for line in out.splitlines():
            m = line_re.match(line)
            if m:
                parsed.append((int(m.group(1), 16), m.group(2).strip()))

        # Slice parsed instructions to each entry [vma, vma + size)
        res = {}
        idx = 0
        n = len(parsed)
        for e in ents:
            lo, hi = e["vma"], e["vma"] + e["size"]
            while idx < n and parsed[idx][0] < lo:
                idx += 1
            j = idx
            lst = []
            while j < n and parsed[j][0] < hi:
                lst.append(parsed[j])
                j += 1
            res[e["vma"]] = lst
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for part in ex.map(run_chunk, chunks):
            func_insns.update(part)
    return func_insns


def analyze_eboot_func(vma: int, size: int, insns: list[tuple[int, str]], eboot_bytes: bytes) -> dict:
    direct_calls: set[int] = set()
    tail_jmps: set[int] = set()
    globals_5x: set[int] = set()
    vtable_slots: Counter[int] = Counter()
    rodata_floats: set[float] = set()
    alloc_sizes: list[int] = []
    real_insn_cnt = 0

    call_re = re.compile(r'^(?:call|jmp)\s+0x([0-9a-f]+)$')
    rip_re = re.compile(r'#\s+0x([0-9a-f]+)')
    vtbl_re = re.compile(r'^(?:call|jmp)\s+QWORD PTR \[(r[a-z0-9]+)(?:\+0x([0-9a-f]+))?\]$')
    scalar_fp_ops = ("vmovss", "movss", "vaddss", "vsubss", "vmulss", "vdivss", "vucomiss", "vcomiss", "vminss", "vmaxss")

    last_esi_imm = None
    for addr, asm in insns:
        op = asm.split()[0] if asm else ""
        if op in ("nop", "data16", "xchg", "int3", ".byte"):
            continue
        real_insn_cnt += 1

        # Track immediate loaded into esi (2nd arg to allocateAligned(alloc, size, align))
        m_esi = re.match(r'^mov\s+esi,0x([0-9a-f]+)$', asm)
        if m_esi:
            last_esi_imm = int(m_esi.group(1), 16)
        elif op in ("call", "jmp"):
            pass
        elif "esi" in asm or "rsi" in asm:
            last_esi_imm = None

        mc = call_re.match(asm)
        if mc:
            dst = int(mc.group(1), 16)
            if 0x00010000 <= dst < 0x03600000 and not (vma <= dst < vma + size):
                if op == "call":
                    if not is_crt_vma(dst):
                        direct_calls.add(dst)
                else:
                    if not is_crt_vma(dst):
                        tail_jmps.add(dst)

        mv = vtbl_re.match(asm)
        if mv:
            slot = int(mv.group(2), 16) if mv.group(2) else 0
            if slot <= 0x200 and (slot % 8 == 0):
                vtable_slots[slot] += 1
                if slot == 0x58 and last_esi_imm is not None and 0x10 <= last_esi_imm <= 0x10000:
                    alloc_sizes.append(last_esi_imm)

        mr = rip_re.search(asm)
        if mr:
            g = int(mr.group(1), 16)
            if 0x5000000 <= g < 0x5800000:
                if g != STACK_GUARD_GLOBAL:
                    globals_5x.add(g)
            elif 0x4700000 <= g < 0x4e00000 and op in scalar_fp_ops:
                foff = g + EBOOT_FILE_OFFSET
                if 0 <= foff <= len(eboot_bytes) - 4:
                    fv = struct.unpack_from("<f", eboot_bytes, foff)[0]
                    if math.isfinite(fv) and abs(fv) > 1e-6 and abs(fv) < 1e7:
                        if not any(abs(fv - t) < 1e-6 for t in TRIVIAL_FLOATS):
                            rodata_floats.add(round(fv, 5))

    return {
        "insn_count": real_insn_cnt,
        "calls": direct_calls | tail_jmps,
        "globals_5x": globals_5x,
        "vtable_slots": vtable_slots,
        "rodata_floats": rodata_floats,
        "alloc_sizes": sorted(set(alloc_sizes)),
    }


def parse_object_file(obj_path: Path, elf_to_vma: dict[str, int], imp_to_vma: dict[str, int]) -> dict[str, dict]:
    """Disassemble one compiled .cpp.o with relocations and extract per-function facts."""
    rel_src = str(obj_path.relative_to(OBJ_DIR)).removesuffix(".o")
    out = subprocess.check_output(
        ["objdump", "-d", "-r", "-M", "intel", str(obj_path)],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    # Extract all float32 constants from .rodata* sections of this .o file
    ro_out = subprocess.run(
        ["objdump", "-s", "-j", ".rodata.cst4", "-j", ".rodata.cst8", "-j", ".rodata.cst16", "-j", ".rodata", str(obj_path)],
        capture_output=True,
        text=True,
    ).stdout
    obj_floats: set[float] = set()
    hex_line_re = re.compile(r'^\s*[0-9a-f]+\s+([0-9a-f ]{1,36})\s')
    raw_bytes = bytearray()
    for line in ro_out.splitlines():
        if line.startswith("Contents of section"):
            if len(raw_bytes) >= 4:
                for i in range(0, len(raw_bytes) - 3, 4):
                    fv = struct.unpack_from("<f", raw_bytes, i)[0]
                    if math.isfinite(fv) and 1e-6 < abs(fv) < 1e7:
                        obj_floats.add(round(fv, 5))
            raw_bytes.clear()
            continue
        mh = hex_line_re.match(line)
        if mh:
            parts = mh.group(1).strip().split()
            for p in parts:
                if len(p) == 8:
                    try:
                        raw_bytes.extend(bytes.fromhex(p))
                    except ValueError:
                        pass
    if len(raw_bytes) >= 4:
        for i in range(0, len(raw_bytes) - 3, 4):
            fv = struct.unpack_from("<f", raw_bytes, i)[0]
            if math.isfinite(fv) and 1e-6 < abs(fv) < 1e7:
                obj_floats.add(round(fv, 5))

    funcs: dict[str, dict] = {}
    cur_sym = None
    cur_insns = 0
    cur_calls_vma: set[int] = set()
    cur_calls_syms: set[str] = set()
    cur_local_callees: set[str] = set()
    cur_globals_vma: set[int] = set()
    cur_vma_imms: set[int] = set()
    cur_vtable_slots: Counter[int] = Counter()
    cur_alloc_sizes: list[int] = []
    cur_has_direct_vk = False
    cur_last_op = ""
    cur_last_esi = None

    hdr_re = re.compile(r'^[0-9a-f]+ <([^>]+)>:$')
    insn_re = re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*(.+)$')
    reloc_re = re.compile(r'^\s*[0-9a-f]+:\s+R_X86_64_\S+\s+(\S+)$')
    vtbl_re = re.compile(r'^(?:call|jmp)\s+QWORD PTR \[(r[a-z0-9]+)(?:\+0x([0-9a-f]+))?\]$')
    imm_vma_re = re.compile(r'\b0x([0-9a-f]{6,7})\b')

    def flush():
        if cur_sym is None:
            return
        funcs[cur_sym] = {
            "src_file": rel_src,
            "insn_count": cur_insns,
            "calls_vma": cur_calls_vma,
            "calls_syms": cur_calls_syms,
            "local_callees": cur_local_callees,
            "globals_5x": cur_globals_vma,
            "vma_imms": cur_vma_imms,
            "vtable_slots": cur_vtable_slots,
            "alloc_sizes": sorted(set(cur_alloc_sizes)),
            "has_direct_vk": cur_has_direct_vk,
            "obj_floats": obj_floats,
        }

    for line in out.splitlines():
        mh = hdr_re.match(line)
        if mh:
            flush()
            cur_sym = mh.group(1)
            cur_insns = 0
            cur_calls_vma = set()
            cur_calls_syms = set()
            cur_local_callees = set()
            cur_globals_vma = set()
            cur_vma_imms = set()
            cur_vtable_slots = Counter()
            cur_alloc_sizes = []
            cur_has_direct_vk = False
            cur_last_op = ""
            cur_last_esi = None
            continue
        if cur_sym is None:
            continue

        mr = reloc_re.match(line)
        if mr:
            rsym = mr.group(1)
            base_sym = re.sub(r'[+-]0x[0-9a-fA-F]+$', '', rsym)
            # Strip section prefix `.text.` used by clang for static/anon-ns local helpers
            clean_local = base_sym[6:] if base_sym.startswith(".text.") else base_sym
            if any(clean_local.startswith(pref) or pref in clean_local for pref in DIRECT_VK_SYM_PREFIXES):
                cur_has_direct_vk = True
            v = extract_vma_from_symbol(clean_local, elf_to_vma, imp_to_vma)
            if v is not None:
                if 0x5000000 <= v < 0x5800000:
                    cur_globals_vma.add(v)
                elif not is_crt_vma(v):
                    cur_calls_vma.add(v)
            if cur_last_op in ("call", "jmp"):
                cur_calls_syms.add(clean_local)
                if v is None:
                    cur_local_callees.add(clean_local)
            continue

        mi = insn_re.match(line)
        if mi:
            asm = mi.group(1).strip()
            op = asm.split()[0] if asm else ""
            cur_last_op = op
            if op in ("nop", "data16", "xchg", "int3", ".byte"):
                continue
            cur_insns += 1

            m_esi = re.match(r'^mov\s+esi,0x([0-9a-f]+)$', asm)
            if m_esi:
                cur_last_esi = int(m_esi.group(1), 16)
            elif op not in ("call", "jmp") and ("esi" in asm or "rsi" in asm):
                cur_last_esi = None

            mv = vtbl_re.match(asm)
            if mv:
                slot = int(mv.group(2), 16) if mv.group(2) else 0
                if slot <= 0x200 and (slot % 8 == 0):
                    cur_vtable_slots[slot] += 1
                    if slot == 0x58 and cur_last_esi is not None and 0x10 <= cur_last_esi <= 0x10000:
                        cur_alloc_sizes.append(cur_last_esi)

            # Capture literal VMA immediates:
            # - Code VMAs (0x00800000..0x03500000): only on add/lea/mov when NOT a repeated byte mask (e.g. 0x1010101)
            # - Global/VTable VMAs (0x05000000..0x05800000) & Rodata float VMAs (0x04700000..0x04d00000): on add/lea/mov/movabs
            if op in ("add", "lea", "mov", "movabs"):
                for mi_v in imm_vma_re.finditer(asm):
                    val = int(mi_v.group(1), 16)
                    if 0x05000000 <= val < 0x05800000:
                        if val != STACK_GUARD_GLOBAL:
                            cur_globals_vma.add(val)
                            cur_vma_imms.add(val)
                    elif 0x04700000 <= val < 0x04d00000:
                        cur_vma_imms.add(val)
                    elif op in ("add", "lea") and 0x00800000 <= val < 0x03500000:
                        if not is_crt_vma(val):
                            cur_vma_imms.add(val)
                            cur_calls_vma.add(val)
    flush()

    # Recursively merge intra-TU static/anonymous-namespace helper functions into their callers
    memo: dict[str, dict] = {}

    def resolve_merged(sym: str, depth: int = 0) -> dict:
        if sym in memo:
            return memo[sym]
        base = funcs[sym]
        if depth >= 4 or not base["local_callees"]:
            return base
        insns = base["insn_count"]
        calls_vma = set(base["calls_vma"])
        globals_5x = set(base["globals_5x"])
        vslots = Counter(base["vtable_slots"])
        alloc_sizes = set(base["alloc_sizes"])
        has_dvk = base["has_direct_vk"]
        for lc in base["local_callees"]:
            if lc in funcs and lc != sym:
                sub = resolve_merged(lc, depth + 1)
                insns += sub["insn_count"]
                calls_vma |= sub["calls_vma"]
                globals_5x |= sub["globals_5x"]
                vslots += sub["vtable_slots"]
                alloc_sizes |= set(sub["alloc_sizes"])
                has_dvk = has_dvk or sub["has_direct_vk"]
        merged = dict(base)
        merged["insn_count"] = insns
        merged["calls_vma"] = calls_vma
        merged["globals_5x"] = globals_5x
        merged["vtable_slots"] = vslots
        merged["alloc_sizes"] = sorted(alloc_sizes)
        merged["has_direct_vk"] = has_dvk
        memo[sym] = merged
        return merged

    return {s: resolve_merged(s) for s in funcs}


def scan_source_patterns(module_filter: str) -> list[dict]:
    """Scan all 47 .cpp files for high-signal decompilation bug patterns (including ilIdx vs vsIdx)."""
    hits = []
    for cpp in sorted(SRC_DIR.rglob("*.cpp")):
        rel = str(cpp.relative_to(SRC_DIR))
        if module_filter and module_filter not in rel:
            continue
        lines = cpp.read_text(errors="ignore").splitlines()
        for i, line in enumerate(lines):
            lno = i + 1
            # Pattern 1: if (arr[idxA] ...) { ... arr[idxB] ... } within 3 lines (the exact 0x21916d0 bug class!)
            m_if = re.search(r'if\s*\(\s*([A-Za-z0-9_>.\-]+)\[([A-Za-z_][A-Za-z0-9_]*)\]\s*(?:!=\s*nullptr|!=\s*0)?\s*\)', line)
            if m_if:
                arr_name, idx_a = m_if.group(1), m_if.group(2)
                window = "\n".join(lines[i + 1 : min(len(lines), i + 5)])
                for m_use in re.finditer(re.escape(arr_name) + r'\[([A-Za-z_][A-Za-z0-9_]*)\]', window):
                    idx_b = m_use.group(1)
                    if idx_b != idx_a:
                        hits.append({
                            "kind": "SRC_INDEX_MISMATCH",
                            "severity": "P1",
                            "file": rel,
                            "line": lno,
                            "detail": f"Null-check on `{arr_name}[{idx_a}]` followed by `{arr_name}[{idx_b}]`",
                        })
                        break

            # Pattern 2: transposed memset(ptr, non_zero_size, 0)
            m_ms = re.search(r'memset\s*\(\s*[^,]+,\s*([^,]+),\s*0[uUlL]?\s*\)', line)
            if m_ms:
                hits.append({
                    "kind": "SRC_MEMSET_ZERO_LEN",
                    "severity": "P0",
                    "file": rel,
                    "line": lno,
                    "detail": f"memset with 0 length: `{line.strip()[:100]}`",
                })

            # Pattern 3: 32-bit shift by >= 32 (1u << 32 etc.)
            m_sh = re.search(r'\b1u?\s*<<\s*(\d+)', line)
            if m_sh and int(m_sh.group(1)) >= 32:
                hits.append({
                    "kind": "SRC_SHIFT_OVERFLOW",
                    "severity": "P0",
                    "file": rel,
                    "line": lno,
                    "detail": f"32-bit shift overflow `1 << {m_sh.group(1)}`",
                })

            # Pattern 4: sizeof(pointer) in memset/memcpy
            if re.search(r'(?:memset|memcpy|memmove)\s*\([^;]*sizeof\s*\(\s*[A-Za-z0-9_]+\s*\*\s*\)', line):
                # ignore intentional pointer-array sizing `count * sizeof(void*)`
                if not re.search(r'\*\s*sizeof\s*\(', line):
                    hits.append({
                        "kind": "SRC_SIZEOF_PTR",
                        "severity": "P2",
                        "file": rel,
                        "line": lno,
                        "detail": f"Check sizeof(T*) in memop: `{line.strip()[:100]}`",
                    })

            # Pattern 5: passing &scalar_4byte to UpdateBufferData (sub_26d73f0) — 16-byte CB over-read class
            if "26d73f0" in line and "&" in line:
                m_arg = re.search(r'&([A-Za-z0-9_]+)', line)
                if m_arg:
                    vname = m_arg.group(1)
                    decl_win = "\n".join(lines[max(0, i - 15) : i])
                    if re.search(rf'\b(?:uint32_t|int32_t|float|uint16_t|uint8_t)\s+{re.escape(vname)}\b(?!\s*\[)', decl_win):
                        hits.append({
                            "kind": "SRC_SCALAR_CB_OVERREAD",
                            "severity": "P1",
                            "file": rel,
                            "line": lno,
                            "detail": f"4-byte scalar `&{vname}` passed to 16B CB upload `sub_26d73f0`",
                        })
    return hits


def run_strict_warnings(module_filter: str) -> list[dict]:
    """Run clang++ -fsyntax-only with strict bug-hunting warnings across all 47 .cpp files."""
    cpp_files = [
        p for p in sorted(SRC_DIR.rglob("*.cpp"))
        if not module_filter or module_filter in str(p.relative_to(SRC_DIR))
    ]
    flags = [
        str(CLANGXX),
        "-fsyntax-only",
        "-std=c++20",
        "-fPIC",
        "-O2",
        "-DNDEBUG",
        "-march=x86-64-v3",
        "-mno-avx512f",
        "-fshort-wchar",
        f"-I{ROOT / 'dantelion_gx/include'}",
        "-Wall",
        "-Wextra",
        "-Wuninitialized",
        "-Wsometimes-uninitialized",
        "-Wtype-limits",
        "-Wsizeof-pointer-memaccess",
        "-Wmemset-transposed-args",
        "-Wlogical-op-parentheses",
        "-Wbool-compare",
        "-Wshift-count-overflow",
        "-Wtautological-compare",
        "-Wself-assign",
        "-Wduplicate-enum",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-unused-but-set-variable",
        "-Wno-missing-field-initializers",
        "-Wno-sign-compare",
        "-Wno-strict-aliasing",
        "-Wno-address-of-packed-member",
    ]
    warn_re = re.compile(r'^(.+?):(\d+):(\d+):\s+warning:\s+(.+?)\s+\[([^\]]+)\]$')

    def check_one(cpp: Path):
        rel = str(cpp.relative_to(SRC_DIR))
        r = subprocess.run(flags + [str(cpp)], capture_output=True, text=True)
        out = []
        for line in r.stderr.splitlines():
            m = warn_re.match(line)
            if m and rel in m.group(1):
                out.append({
                    "file": rel,
                    "line": int(m.group(2)),
                    "msg": m.group(4),
                    "flag": m.group(5),
                })
        return out

    warnings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        for res in ex.map(check_one, cpp_files):
            warnings.extend(res)
    return warnings


def main():
    ap = argparse.ArgumentParser(description="Audit 2.0: Automated 01..07 binary cross-checker")
    ap.add_argument("--module", default="", help="Filter by module/path substring (e.g. 01, 02, 04, 05, 06, 07)")
    ap.add_argument("--vma", default="", help="Inspect single VMA in detail (hex, e.g. 0x21916d0)")
    ap.add_argument("--warnings", action="store_true", help="Also run strict clang++ -fsyntax-only warning sweep")
    ap.add_argument("--json", default="", help="Write full structured JSON report to path")
    ap.add_argument("--top", type=int, default=60, help="Top findings to display")
    args = ap.parse_args()

    entries, elf_to_vma, vma_to_entry = parse_registry()
    imp_to_vma, data_syms = parse_external_imports()

    # Map friendly module prefix ("01".."07") to registry module_name / folder
    mod_filter = args.module
    if args.vma:
        tv = int(args.vma, 16)
        target_entries = [e for e in entries if e["vma"] == tv]
    else:
        target_entries = [
            e for e in entries
            if not mod_filter or mod_filter in e["module"]
        ]

    eboot_bytes = EBOOT_BIN.read_bytes()
    eboot_dis = disassemble_eboot_ranges(entries, eboot_bytes)
    eboot_facts = {
        e["vma"]: analyze_eboot_func(e["vma"], e["size"], eboot_dis.get(e["vma"], []), eboot_bytes)
        for e in entries
    }

    # Parse all compiled .o files in parallel
    obj_files = sorted(OBJ_DIR.rglob("*.cpp.o"))
    compiled_map: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        for part in ex.map(lambda p: parse_object_file(p, elf_to_vma, imp_to_vma), obj_files):
            compiled_map.update(part)

    # Build 1-level callee expansion set per eboot function (to absorb compiler inlining of small helpers!)
    def expanded_eboot_calls(vma: int, depth: int = 1) -> set[int]:
        base = set(eboot_facts[vma]["calls"])
        if depth <= 0:
            return base
        out = set(base)
        for c in base:
            if c in eboot_facts and vma_to_entry[c]["size"] <= 512:
                out |= eboot_facts[c]["calls"]
        return out

    def expanded_eboot_globals(vma: int) -> set[int]:
        out = set(eboot_facts[vma]["globals_5x"])
        for c in eboot_facts[vma]["calls"]:
            if c in eboot_facts and vma_to_entry[c]["size"] <= 512:
                out |= eboot_facts[c]["globals_5x"]
        return out

    findings = []
    missing_symbols = 0
    checked_active = 0

    for e in target_entries:
        vma = e["vma"]
        if not e["active"] and not args.vma:
            continue
        checked_active += 1
        ef = eboot_facts[vma]
        cf = compiled_map.get(e["elf_sym"])
        if cf is None:
            missing_symbols += 1
            findings.append({
                "severity": "P0",
                "score": 100,
                "vma": hex(vma),
                "size": e["size"],
                "module": e["module"],
                "file": "?",
                "name": e["name"],
                "direct_vk": False,
                "issues": ["MISSING_ELF_SYMBOL"],
            })
            continue

        # Check if this function is a clean 1-line delegate/wrapper to an identical twin VMA in registry!
        # (e.g. COMDAT-duplicated template instantiations in eboot.bin consolidated in C++20)
        if cf["insn_count"] <= 16 and len(cf["calls_vma"]) == 1:
            twin = next(iter(cf["calls_vma"]))
            if twin in eboot_facts and twin != vma:
                tf = eboot_facts[twin]
                # Verify the twin actually has matching calls & vtable slots in eboot.bin!
                if (
                    abs(tf["insn_count"] - ef["insn_count"]) <= max(8, int(ef["insn_count"] * 0.15))
                    and ef["calls"] == tf["calls"]
                    and set(ef["vtable_slots"].keys()) == set(tf["vtable_slots"].keys())
                ):
                    if not args.vma:
                        continue

        # Also absorb 1-level callees of compiled functions (in case eboot inlined a small helper we called out-of-line!)
        comp_calls = set(cf["calls_vma"])
        comp_globals = set(cf["globals_5x"])
        for c in list(comp_calls):
            if c in eboot_facts and vma_to_entry[c]["size"] <= 256:
                comp_calls |= eboot_facts[c]["calls"]
                comp_globals |= eboot_facts[c]["globals_5x"]
        # Also absorb inlined small callees of eboot into "expected" sets
        eb_calls_exp = expanded_eboot_calls(vma, 1)
        eb_globals_exp = expanded_eboot_globals(vma)

        issues = []
        score = 0

        # 1. Missing direct calls from eboot.bin (not inlined into our compiled code)
        miss_calls = sorted(ef["calls"] - comp_calls)
        # Filter out small intra-registry callees that clang likely inlined (we check if their own callees/globals appear or size <= 160B)
        real_miss_calls = []
        for mc in miss_calls:
            if mc in vma_to_entry and vma_to_entry[mc]["size"] <= 256:
                # Could have been inlined by clang -O3! Only flag if compiled insn_count is smaller than eboot!
                if cf["insn_count"] >= ef["insn_count"] * 0.75:
                    continue
            real_miss_calls.append(mc)

        # Extra literal eboot calls in compiled code that aren't in eboot (even after 1-level expansion)
        extra_calls = sorted(cf["calls_vma"] - eb_calls_exp - {vma})
        # Filter out known harmless helper calls (e.g. DLAllocator / ResolveVMA runtime / recursive)
        if real_miss_calls:
            pts = 35 if not cf["has_direct_vk"] else 10
            score += pts * min(len(real_miss_calls), 3)
            issues.append(f"MISSING_CALL({[hex(x) for x in real_miss_calls[:6]]})")
        if extra_calls:
            pts = 30 if not cf["has_direct_vk"] else 8
            score += pts * min(len(extra_calls), 3)
            issues.append(f"EXTRA_CALL({[hex(x) for x in extra_calls[:6]]})")

        # 2. Global / VTable 0x5xxxxxx parity (absorb constant-folded vtable+0x10 / +0x08 and TLS 0x53e4xxx)
        def global_covered(g: int, pool: set[int]) -> bool:
            return g in pool or (g + 0x10) in pool or (g - 0x10) in pool or (g + 0x08) in pool or (g - 0x08) in pool

        TLS_RANGE = range(0x53e4300, 0x53e4700)
        miss_globs = sorted(g for g in ef["globals_5x"] if not global_covered(g, comp_globals) and g not in TLS_RANGE)
        extra_globs = sorted(g for g in cf["globals_5x"] if not global_covered(g, eb_globals_exp) and g not in TLS_RANGE)
        if miss_globs:
            pts = 40 if not cf["has_direct_vk"] else 12
            score += pts * min(len(miss_globs), 3)
            issues.append(f"MISSING_GLOBAL({[hex(x) for x in miss_globs[:6]]})")
        if extra_globs:
            pts = 35 if not cf["has_direct_vk"] else 8
            score += pts * min(len(extra_globs), 3)
            issues.append(f"EXTRA_GLOBAL({[hex(x) for x in extra_globs[:6]]})")

        # 3. Vtable slot offset parity (ignore if function has DirectVk or small inlined helpers)
        eb_slots = set(ef["vtable_slots"].keys())
        cf_slots = set(cf["vtable_slots"].keys())
        miss_slots = sorted(eb_slots - cf_slots)
        extra_slots = sorted(cf_slots - eb_slots)
        if miss_slots and not cf["has_direct_vk"]:
            score += 25 * min(len(miss_slots), 2)
            issues.append(f"MISSING_VSLOT({[hex(x) for x in miss_slots[:5]]})")
        if extra_slots and not cf["has_direct_vk"] and not extra_calls:
            # Exclude slots coming from inlined small callees
            inlined_slots = set()
            for c in ef["calls"]:
                if c in eboot_facts and vma_to_entry[c]["size"] <= 256:
                    inlined_slots |= set(eboot_facts[c]["vtable_slots"].keys())
            true_extra_slots = sorted(set(extra_slots) - inlined_slots)
            if true_extra_slots:
                score += 20 * min(len(true_extra_slots), 2)
                issues.append(f"EXTRA_VSLOT({[hex(x) for x in true_extra_slots[:5]]})")

        # 4. Alloc size mismatch
        if ef["alloc_sizes"] and cf["alloc_sizes"]:
            eb_as = set(ef["alloc_sizes"])
            cf_as = set(cf["alloc_sizes"])
            if eb_as != cf_as and not cf["has_direct_vk"]:
                score += 45
                issues.append(f"ALLOC_SIZE(eboot={[hex(x) for x in sorted(eb_as)]}, cpp={[hex(x) for x in sorted(cf_as)]})")

        # 5. Severe code truncation (potential stub or dropped branch)
        if e["size"] >= 120 and ef["insn_count"] >= 25 and not cf["has_direct_vk"]:
            ratio = cf["insn_count"] / max(ef["insn_count"], 1)
            if ratio < 0.35:
                score += 55
                issues.append(f"TRUNCATED(eboot={ef['insn_count']}i, cpp={cf['insn_count']}i, ratio={ratio:.2f})")
            elif ratio > 3.5 and ef["insn_count"] >= 40:
                score += 12
                issues.append(f"BLOATED(eboot={ef['insn_count']}i, cpp={cf['insn_count']}i, ratio={ratio:.2f})")

        # 6. Non-trivial float constants missing from entire .o rodata (also accept direct GX_VMA(0x49xxxxx) loads!)
        if ef["rodata_floats"]:
            vma_floats = set()
            for iv in cf["vma_imms"]:
                if 0x04700000 <= iv < 0x04d00000:
                    foff = iv + EBOOT_FILE_OFFSET
                    if 0 <= foff <= len(eboot_bytes) - 4:
                        fv2 = struct.unpack_from("<f", eboot_bytes, foff)[0]
                        if math.isfinite(fv2):
                            vma_floats.add(round(fv2, 5))
            all_cf_floats = cf["obj_floats"] | vma_floats
            miss_f = []
            for fv in sorted(ef["rodata_floats"]):
                if not any(abs(fv - of) <= max(1e-4, abs(fv) * 1e-4) for of in all_cf_floats):
                    miss_f.append(fv)
            if miss_f and not cf["has_direct_vk"]:
                score += 18 * min(len(miss_f), 2)
                issues.append(f"MISSING_FLOAT({miss_f[:5]})")

        if args.vma:
            print(f"=== VMA {hex(vma)} ({e['name']}) [{e['module']}] file={cf['src_file']} ===")
            print(f"  eboot: {e['size']}B, {ef['insn_count']} insns, calls={[hex(x) for x in sorted(ef['calls'])]}")
            print(f"  eboot globals_5x: {[hex(x) for x in sorted(ef['globals_5x'])]}")
            print(f"  eboot vtable_slots: { {hex(k): v for k, v in sorted(ef['vtable_slots'].items())} }")
            print(f"  eboot alloc_sizes: {[hex(x) for x in ef['alloc_sizes']]}, floats: {sorted(ef['rodata_floats'])}")
            print(f"  cpp:   {cf['insn_count']} insns, direct_vk={cf['has_direct_vk']}, calls={[hex(x) for x in sorted(cf['calls_vma'])]}")
            print(f"  cpp   globals_5x: {[hex(x) for x in sorted(cf['globals_5x'])]}")
            print(f"  cpp   vtable_slots: { {hex(k): v for k, v in sorted(cf['vtable_slots'].items())} }")
            print(f"  cpp   alloc_sizes: {[hex(x) for x in cf['alloc_sizes']]}")
            print(f"  score={score}, issues={issues}")
            return

        if issues:
            sev = "P0" if score >= 55 else ("P1" if score >= 30 else "P2")
            is_debug_gui = any(
                k in e["name"]
                for k in ("DebugMenu", "DebugGui", "DebugProperty", "ParamEdit", "PropertyTree", "RegisterYebis", "PopulateDebug")
            )
            findings.append({
                "severity": sev,
                "score": score,
                "vma": hex(vma),
                "size": e["size"],
                "module": e["module"],
                "file": cf["src_file"],
                "name": e["name"],
                "direct_vk": cf["has_direct_vk"],
                "debug_gui": is_debug_gui,
                "eboot_insns": ef["insn_count"],
                "cpp_insns": cf["insn_count"],
                "issues": issues,
            })

    findings.sort(key=lambda x: (x.get("debug_gui", False), -x["score"], -x["size"]))
    src_hits = scan_source_patterns(mod_filter)
    comp_warns = run_strict_warnings(mod_filter) if args.warnings else []

    live_findings = [f for f in findings if not f.get("debug_gui")]
    dbg_findings = [f for f in findings if f.get("debug_gui")]
    by_sev = Counter(f["severity"] for f in live_findings)
    by_mod = Counter(f["module"] for f in live_findings if f["severity"] in ("P0", "P1"))
    print(f"=== AUDIT 2.0 SUMMARY (active checked: {checked_active}, missing ELF syms: {missing_symbols}) ===")
    print(f"  LIVE runtime flags: total={len(live_findings)}  P0={by_sev['P0']}  P1={by_sev['P1']}  P2={by_sev['P2']}")
    print(f"  DebugGUI-only flags: {len(dbg_findings)}")
    print(f"  LIVE P0+P1 by module: {dict(sorted(by_mod.items()))}")
    print(f"  Source bug-pattern hits: {len(src_hits)}")
    if args.warnings:
        print(f"  Strict clang++ warnings: {len(comp_warns)}")

    print(f"\n=== TOP {min(args.top, len(live_findings))} LIVE RUNTIME BINARY DIFF SUSPECTS ===")
    for f in live_findings[: args.top]:
        dvk = " [DirectVk]" if f["direct_vk"] else ""
        print(f"  [{f['severity']} {f['score']:3d}] {f['vma']} ({f['size']:5d}B) [{f['module']}] {f['file']} :: {f['name']}{dvk}")
        print(f"           {'; '.join(f['issues'])}")

    if src_hits:
        print(f"\n=== SOURCE PATTERN HITS ({len(src_hits)}) ===")
        for h in src_hits[:40]:
            print(f"  [{h['severity']} {h['kind']}] {h['file']}:{h['line']} — {h['detail']}")

    if comp_warns:
        print(f"\n=== STRICT COMPILER WARNINGS ({len(comp_warns)}) ===")
        by_flag = Counter(w["flag"] for w in comp_warns)
        print("  By flag:", dict(by_flag.most_common(15)))
        for w in comp_warns[:50]:
            print(f"  [{w['flag']}] {w['file']}:{w['line']} — {w['msg']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "checked_active": checked_active,
                    "summary_by_sev": dict(by_sev),
                    "summary_p0p1_by_mod": dict(by_mod),
                    "findings": findings,
                    "source_hits": src_hits,
                    "compiler_warnings": comp_warns,
                },
                indent=2,
            )
        )
        print(f"\nWrote JSON report -> {args.json}")


if __name__ == "__main__":
    main()
