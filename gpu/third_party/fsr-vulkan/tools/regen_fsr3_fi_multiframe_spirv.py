#!/usr/bin/env python3
"""Regenerate the modified FSR 3.1.4 Frame Interpolation SPIR-V blobs natively via glslangValidator.

Compiles the five FI compute passes parameterized by cbFI.fInterpolationTime across all four
variants (base, wave64, 16bit, wave64_16bit), validates each binary with spirv-val (vulkan1.2),
verifies entry point, UBO member layout at offset 156, and descriptor bindings match the existing
reflection tables 1-to-1, then updates the _size and _data[] arrays in-place.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SDK_DIR = os.path.join(MODULE_DIR, "upstream", "ffx-1.1.4", "sdk")
GPU_DIR = os.path.join(SDK_DIR, "include", "FidelityFX", "gpu")
FI_INC_DIR = os.path.join(GPU_DIR, "frameinterpolation")
GLSL_DIR = os.path.join(SDK_DIR, "src", "backends", "vk", "shaders", "frameinterpolation")
GEN_DIR = os.path.join(MODULE_DIR, "generated", "ffx-1.1.4", "vk", "fsr3framegeneration")

PASSES = [
    "ffx_frameinterpolation_reconstruct_previous_depth_pass",
    "ffx_frameinterpolation_game_motion_vector_field_pass",
    "ffx_frameinterpolation_optical_flow_vector_field_pass",
    "ffx_frameinterpolation_disocclusion_mask_pass",
    "ffx_frameinterpolation_pass",
]

VARIANTS = [
    ("", 0),
    ("_wave64", 0),
    ("_16bit", 1),
    ("_wave64_16bit", 1),
]


def parse_old_spv(hdr_text: str) -> bytes:
    m = re.search(r"static const unsigned char \w+_data\[\]\s*=\s*\{([^}]+)\};", hdr_text, re.S)
    if not m:
        raise RuntimeError("Could not locate _data[] array in header")
    return bytes(int(tok.strip(), 0) for tok in m.group(1).split(",") if tok.strip())


def extract_descriptor_bindings(dis_text: str) -> set[tuple[str, str, int]]:
    sets: dict[str, int] = {}
    bindings: dict[str, int] = {}
    for line in dis_text.splitlines():
        m = re.search(r"OpDecorate\s+(%\S+)\s+DescriptorSet\s+(\d+)", line)
        if m:
            sets[m.group(1)] = int(m.group(2))
        m = re.search(r"OpDecorate\s+(%\S+)\s+Binding\s+(\d+)", line)
        if m:
            bindings[m.group(1)] = int(m.group(2))
    out: set[tuple[str, str, int]] = set()
    for name, b in bindings.items():
        if name not in sets:
            raise RuntimeError(f"Missing DescriptorSet for {name}")
        out.add((name, f"set={sets[name]}", b))
    return out


def verify_spirv(old_spv: bytes, new_spv_path: str, label: str) -> None:
    subprocess.check_call(["spirv-val", "--target-env", "vulkan1.2", new_spv_path])

    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tmp_old:
        tmp_old.write(old_spv)
        old_spv_path = tmp_old.name
    try:
        old_dis = subprocess.check_output(["spirv-dis", old_spv_path], text=True)
    finally:
        os.unlink(old_spv_path)
    new_dis = subprocess.check_output(["spirv-dis", new_spv_path], text=True)

    if 'OpEntryPoint GLCompute %main "main"' not in new_dis:
        raise RuntimeError(f"{label}: missing GLCompute entry point 'main'")

    old_b = extract_descriptor_bindings(old_dis)
    new_b = extract_descriptor_bindings(new_dis)
    if old_b != new_b:
        raise RuntimeError(f"{label}: descriptor bindings mismatch:\n  old={sorted(old_b)}\n  new={sorted(new_b)}")

    m_name = re.search(r'OpMemberName\s+%cbFI_t\s+(\d+)\s+"fInterpolationTime"', new_dis)
    if not m_name:
        raise RuntimeError(f"{label}: cbFI_t is missing member fInterpolationTime")
    member_idx = m_name.group(1)
    if f"OpMemberDecorate %cbFI_t {member_idx} Offset 156" not in new_dis:
        raise RuntimeError(f"{label}: cbFI_t member {member_idx} (fInterpolationTime) offset is not 156")


def format_c_array(data: bytes) -> str:
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        line = ",".join(f"0x{b:02x}" for b in chunk)
        if i + 16 < len(data):
            line += ","
        lines.append(line)
    return "\n".join(lines) + "\n"


def compile_and_update() -> int:
    updated_blobs = 0
    for pass_name in PASSES:
        src_glsl = os.path.join(GLSL_DIR, f"{pass_name}.glsl")
        for suffix, ffx_half in VARIANTS:
            perm_path = os.path.join(GEN_DIR, f"{pass_name}{suffix}_permutations.h")
            with open(perm_path, "r", encoding="utf-8") as f:
                perm_txt = f.read()

            incs = re.findall(r'#include "([^"]+)"', perm_txt)
            tbl_m = re.search(r"IndirectionTable\[\]\s*=\s*\{([^}]+)\}", perm_txt, re.S)
            if not tbl_m:
                raise RuntimeError(f"Missing IndirectionTable in {perm_path}")
            indirection = [int(x.strip()) for x in tbl_m.group(1).split(",") if x.strip()]
            if len(indirection) != 8:
                raise RuntimeError(f"Unexpected indirection table length in {perm_path}: {len(indirection)}")

            # Bit 2 of key.index is FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH.
            blob_inv: dict[int, set[int]] = {}
            for key_idx, table_idx in enumerate(indirection):
                inv = (key_idx >> 2) & 1
                blob_inv.setdefault(table_idx, set()).add(inv)

            for table_idx, inc_name in enumerate(incs):
                inv_set = blob_inv.get(table_idx)
                if not inv_set:
                    raise RuntimeError(f"{perm_path}: unused blob index {table_idx}")
                # If both 0 and 1 share one blob, INVERTED_DEPTH is invariant; compile with 0.
                inv = 0 if len(inv_set) > 1 else next(iter(inv_set))

                hdr_path = os.path.join(GEN_DIR, inc_name)
                with open(hdr_path, "r", encoding="utf-8") as f:
                    hdr_txt = f.read()
                old_spv = parse_old_spv(hdr_txt)

                with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tmp_spv:
                    spv_path = tmp_spv.name
                try:
                    cmd = [
                        "glslangValidator",
                        "-V",
                        "--target-env",
                        "vulkan1.2",
                        "-S",
                        "comp",
                        "-Os",
                        "-DFFX_GPU=1",
                        "-DFFX_GLSL=1",
                        f"-DFFX_HALF={ffx_half}",
                        "-DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=0",
                        "-DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0",
                        f"-DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH={inv}",
                        f"-I{GPU_DIR}",
                        f"-I{FI_INC_DIR}",
                        "-o",
                        spv_path,
                        src_glsl,
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise RuntimeError(
                            f"glslangValidator failed for {inc_name}:\n{res.stdout}\n{res.stderr}"
                        )
                    verify_spirv(old_spv, spv_path, inc_name)
                    with open(spv_path, "rb") as f:
                        new_spv = f.read()
                finally:
                    if os.path.exists(spv_path):
                        os.unlink(spv_path)

                size_re = re.compile(
                    r"(static const uint32_t \w+_size\s*=\s*)\d+(\s*;)"
                )
                data_re = re.compile(
                    r"(static const unsigned char \w+_data\[\]\s*=\s*\{\s*\n)[^}]+(\};)",
                    re.S,
                )
                new_hdr, n_size = size_re.subn(rf"\g<1>{len(new_spv)}\2", hdr_txt, count=1)
                if n_size != 1:
                    raise RuntimeError(f"Failed to replace _size in {hdr_path}")
                formatted = format_c_array(new_spv)
                new_hdr, n_data = data_re.subn(
                    lambda m: f"{m.group(1)}{formatted}{m.group(2)}", new_hdr, count=1
                )
                if n_data != 1:
                    raise RuntimeError(f"Failed to replace _data[] in {hdr_path}")

                with open(hdr_path, "w", encoding="utf-8") as f:
                    f.write(new_hdr)
                updated_blobs += 1
                print(
                    f"OK {inc_name} (half={ffx_half}, inv={inv}): {len(old_spv)} -> {len(new_spv)} bytes"
                )
    return updated_blobs


if __name__ == "__main__":
    count = compile_and_update()
    print(f"Successfully regenerated and validated {count} SPIR-V blob headers.")
    sys.exit(0)
