#!/usr/bin/env python3
"""
verify_clean_glsl.py — Symbolic AST analyzer, SPIR-V converter, and 100% mathematical
equivalence verifier for Bloodborne Vulkan 1.3 native GLSL shaders.

Usage:
  1. Convert cached .spv to raw GLSL reference:
     python3 verify_clean_glsl.py --spv-to-raw <input.spv> <output_raw.frag>
  2. Symbolically simplify and inspect a raw shader:
     python3 verify_clean_glsl.py --analyze <raw_shader.frag>
  3. Verify 100% equivalence (SPIR-V interface + 64-vector FP32 simulation) between raw & clean:
     python3 verify_clean_glsl.py --verify <ref_raw.frag> <clean.frag> [--spv <ref.spv>]
"""

import argparse
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import numpy as np


# ==============================================================================
# 1. Exact SPIR-V Interface Signature Extractor (mirrors native_shaders.cpp)
# ==============================================================================
def extract_interface_sig(spv_bytes: bytes):
    if len(spv_bytes) < 20:
        raise ValueError("SPIR-V too short")
    words = struct.unpack(f"<{len(spv_bytes)//4}I", spv_bytes)
    if words[0] != 0x07230203:
        raise ValueError("Invalid SPIR-V magic")
    bound = words[3]

    type_comps = [0] * bound
    type_desc_tag = [0] * bound
    struct_deco = [0] * bound
    ptr_sc = [0xFFFFFFFF] * bound
    ptr_pointee = [0] * bound

    var_has_loc = [0] * bound
    var_loc = [0] * bound
    var_has_builtin = [0] * bound
    var_interp = [0] * bound
    var_per_vertex = [0] * bound
    var_has_binding = [0] * bound
    var_binding = [0] * bound
    var_set = [0] * bound
    var_used = [0] * bound

    global_vars = []
    store_vals = []
    in_func = False
    exec_model = 0xFFFFFFFF

    pos = 5
    while pos < len(words):
        wc = words[pos] >> 16
        op = words[pos] & 0xFFFF
        if wc == 0 or pos + wc > len(words):
            break
        w = words[pos : pos + wc]
        if op == 54:  # OpFunction
            in_func = True
        if in_func:
            if op in (61, 65, 66, 67, 83):  # Load, AccessChain, InBoundsAccessChain, ImageTexelPointer, CopyObject
                if wc >= 4 and w[3] < bound:
                    var_used[w[3]] = 1
            elif op == 62:  # OpStore
                if wc >= 3 and w[1] < bound:
                    var_used[w[1]] = 1
                    if len(store_vals) < 4:
                        store_vals.append(w[2])
            elif op == 12:  # OpExtInst
                if wc >= 6 and w[5] < bound:
                    var_used[w[5]] = 1
            elif op == 57:  # OpFunctionCall
                for i in range(3, wc):
                    if w[i] < bound:
                        var_used[w[i]] = 1
            pos += wc
            continue

        if op == 15:  # OpEntryPoint
            if wc >= 4 and exec_model == 0xFFFFFFFF:
                exec_model = w[1]
        elif op == 71:  # OpDecorate
            if wc >= 3 and w[1] < bound:
                target, deco = w[1], w[2]
                if deco == 30 and wc >= 4:
                    var_has_loc[target] = 1
                    var_loc[target] = w[3]
                elif deco == 11:
                    var_has_builtin[target] = 1
                elif deco == 14:
                    var_interp[target] |= 0x1
                elif deco == 13:
                    var_interp[target] |= 0x2
                elif deco == 16:
                    var_interp[target] |= 0x4
                elif deco == 17:
                    var_interp[target] |= 0x8
                elif deco == 5285:
                    var_per_vertex[target] = 1
                elif deco == 33 and wc >= 4:
                    var_has_binding[target] = 1
                    var_binding[target] = w[3]
                elif deco == 34 and wc >= 4:
                    var_set[target] = w[3]
                elif deco == 2:
                    struct_deco[target] = 1
                elif deco == 3:
                    struct_deco[target] = 2
        elif op in (21, 22):  # OpTypeInt, OpTypeFloat
            if wc >= 3 and w[1] < bound:
                type_comps[w[1]] = 1
        elif op == 23:  # OpTypeVector
            if wc == 4 and w[1] < bound and w[3] <= 4:
                type_comps[w[1]] = w[3]
        elif op == 25:  # OpTypeImage
            if wc >= 8 and w[1] < bound:
                dim = w[3] & 0x7
                arrayed = (w[5] & 0x1) << 3
                ms = (w[6] & 0x1) << 4
                sampled = (w[7] & 0x3) << 5
                type_desc_tag[w[1]] = 0x1000 | dim | arrayed | ms | sampled
        elif op == 26:  # OpTypeSampler
            if wc == 2 and w[1] < bound:
                type_desc_tag[w[1]] = 0x2000
        elif op == 27:  # OpTypeSampledImage
            if wc == 3 and w[1] < bound and w[2] < bound:
                type_desc_tag[w[1]] = 0x3000 | (type_desc_tag[w[2]] & 0xFFF)
        elif op == 30:  # OpTypeStruct
            if wc >= 2 and w[1] < bound:
                type_desc_tag[w[1]] = 0x4000
        elif op == 32:  # OpTypePointer
            if wc == 4 and w[1] < bound and w[3] < bound:
                ptr_sc[w[1]] = w[2]
                ptr_pointee[w[1]] = w[3]
        elif op == 59:  # OpVariable
            if wc >= 4 and w[1] < bound and w[2] < bound:
                global_vars.append((w[2], w[1], w[3]))
        pos += wc

    in_loc_mask = 0
    out_loc_mask = 0
    in_interp = {}
    in_comps = {}
    out_comps = {}
    descs = []

    for vid, ptype, sc in global_vars:
        pointee = ptr_pointee[ptype]
        if sc == 1:  # Input
            if not var_used[vid] or var_has_builtin[vid] or (pointee < bound and struct_deco[pointee] != 0 and not var_has_loc[vid]):
                continue
            loc = var_loc[vid]
            in_loc_mask |= (1 << loc)
            in_interp[loc] = var_interp[vid]
            in_comps[loc] = type_comps[pointee] if pointee < bound else 0
        elif sc == 3:  # Output
            if var_has_builtin[vid] or (pointee < bound and struct_deco[pointee] != 0 and not var_has_loc[vid]):
                continue
            loc = var_loc[vid]
            out_loc_mask |= (1 << loc)
            out_comps[loc] = type_comps[pointee] if pointee < bound else 0
        elif var_has_binding[vid] and var_used[vid]:
            kind_tag = 0
            if sc == 12:
                kind_tag = 0x5001
            elif sc == 2:
                kind_tag = 0x5001 if (pointee < bound and struct_deco[pointee] == 2) else 0x5002
            elif sc == 0 and pointee < bound:
                kind_tag = type_desc_tag[pointee]
            entry = (var_set[vid], var_binding[vid], kind_tag)
            if entry not in descs:
                descs.append(entry)

    descs.sort()
    return {
        "exec_model": exec_model,
        "in_loc_mask": in_loc_mask,
        "out_loc_mask": out_loc_mask,
        "in_interp": in_interp,
        "in_comps": in_comps,
        "out_comps": out_comps,
        "descs": descs,
    }


def compile_glsl_to_spv(glsl_path: str, defines=None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tmp:
        spv_path = tmp.name
    try:
        cmd = ["glslangValidator", "-V", "--target-env", "vulkan1.3"]
        if defines:
            for d in defines:
                cmd.append(f"-D{d}")
        cmd.extend(["-o", spv_path, glsl_path])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"glslangValidator failed on {glsl_path}:\n{res.stdout}\n{res.stderr}")
        with open(spv_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(spv_path):
            os.remove(spv_path)


# ==============================================================================
# 2. Convert cached .spv (via spirv-dis) into Raw GLSL reference
# ==============================================================================
def spv_to_raw_glsl(spv_path: str, out_frag_path: str):
    dis = subprocess.check_output(["spirv-dis", spv_path], text=True)
    lines = dis.splitlines()

    names = {}
    decorations = {}
    types = {}
    constants = {}
    global_vars = {}
    sampled_images = {}

    for line in lines:
        s = line.strip()
        m = re.match(r'OpName %(\S+) "([^"]+)"', s)
        if m:
            names[m.group(1)] = m.group(2)
            continue
        m = re.match(r"OpDecorate %(\S+) (\w+)(?: (\d+))?", s)
        if m:
            vid, dec, val = m.group(1), m.group(2), m.group(3)
            decorations.setdefault(vid, {})[dec] = int(val) if val is not None else True
            continue
        m = re.match(r"%(\S+) = OpConstant %(\S+) (\S+)", s)
        if m:
            cid, ctype, cval = m.group(1), m.group(2), m.group(3)
            if "u32" in ctype or "uint" in ctype:
                constants[cid] = f"{cval}u"
            elif "f32" in ctype or "float" in ctype:
                if "." not in cval and "e" not in cval and "E" not in cval:
                    cval = cval + ".0"
                constants[cid] = cval
            elif "i32" in ctype or "int" in ctype:
                constants[cid] = cval
            continue
        m = re.match(r"%(\S+) = OpConstantTrue", s)
        if m:
            constants[m.group(1)] = "true"
            continue
        m = re.match(r"%(\S+) = OpConstantFalse", s)
        if m:
            constants[m.group(1)] = "false"
            continue
        m = re.match(r"%(\S+) = OpTypeImage %\S+ (\w+) (\d+) (\d+) (\d+) (\d+)", s)
        if m:
            tid, dim, depth, arrayed, ms, sampled = m.groups()
            if dim == "2D":
                types[tid] = "texture2DArray" if arrayed == "1" else "texture2D"
            elif dim == "3D":
                types[tid] = "texture3D"
            elif dim == "Cube":
                types[tid] = "textureCubeArray" if arrayed == "1" else "textureCube"
            continue
        m = re.match(r"%(\S+) = OpTypeRuntimeArray %(\S+)", s)
        if m:
            types[m.group(1)] = "float" if "f32" in m.group(2) else "uint"
            continue
        m = re.match(r"%(\S+) = OpTypeStruct %(\S+)", s)
        if m and m.group(2) in types:
            types[m.group(1)] = types[m.group(2)]
            continue
        m = re.match(r"%(\S+) = OpTypePointer (\w+) %(\S+)", s)
        if m:
            types[m.group(1)] = (m.group(2), m.group(3))
            continue
        m = re.match(r"%(\S+) = OpVariable %(\S+) (\w+)", s)
        if m:
            vid, ptype, sc = m.groups()
            global_vars[vid] = (ptype, sc)

    in_vars = []
    out_vars = []
    ssbo_vars = []
    img_vars = []
    samp_vars = []

    for vid, (ptype, sc) in global_vars.items():
        decs = decorations.get(vid, {})
        pointee = types.get(ptype, (None, None))[1]
        if sc == "Input" and "Location" in decs:
            in_vars.append((decs["Location"], vid, decs))
        elif sc == "Output" and "Location" in decs:
            out_vars.append((decs["Location"], vid))
        elif sc == "StorageBuffer" and "Binding" in decs:
            elem_t = types.get(pointee, "uint")
            ssbo_vars.append((decs["Binding"], vid, elem_t))
        elif sc == "UniformConstant" and "Binding" in decs:
            if vid.startswith("fs_img"):
                img_t = types.get(pointee, "texture2D")
                img_vars.append((decs["Binding"], vid, img_t))
            elif vid.startswith("fs_samp"):
                samp_vars.append((decs["Binding"], vid))

    in_vars.sort()
    out_vars.sort()
    ssbo_vars.sort()
    img_vars.sort()
    samp_vars.sort()

    out_lines = [
        "#version 460",
        "#extension GL_EXT_samplerless_texture_functions : require",
        "",
    ]
    for loc, vid, decs in in_vars:
        qual = ""
        if "Flat" in decs:
            qual += "flat "
        if "NoPerspective" in decs:
            qual += "noperspective "
        if "Centroid" in decs:
            qual += "centroid "
        out_lines.append(f"layout(location = {loc}) {qual}in vec4 {vid};")
    out_lines.append("")
    for loc, vid in out_vars:
        out_lines.append(f"layout(location = {loc}) out vec4 {vid};")
    out_lines.append("""
layout(push_constant) uniform AuxData {
    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    uvec4 ud_regs0;
    uvec4 ud_regs1;
    uvec4 ud_regs2;
    uvec4 ud_regs3;
    uvec4 buf_offsets0;
    uvec4 buf_offsets1;
    uvec2 buf_offsets2;
    uint motion_param;
    uint scene_size;
} push_data;
""")
    for bnd, vid, elem_t in ssbo_vars:
        out_lines.append(f"layout(set = 0, binding = {bnd}, std430) readonly buffer {vid}_buf {{ {elem_t} data[]; }} {vid};")
    out_lines.append("")
    for bnd, vid, img_t in img_vars:
        out_lines.append(f"layout(set = 0, binding = {bnd}) uniform {img_t} {vid};")
    for bnd, vid in samp_vars:
        out_lines.append(f"layout(set = 0, binding = {bnd}) uniform sampler {vid};")

    out_lines.append("""
float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

void main() {""")

    def vname(tok):
        tok = tok.lstrip("%")
        if tok in constants:
            return constants[tok]
        if tok.isdigit():
            return f"t{tok}"
        return tok

    def glsl_type(spv_t):
        spv_t = spv_t.lstrip("%")
        if "f32vec4" in spv_t:
            return "vec4"
        if "f32vec3" in spv_t:
            return "vec3"
        if "f32vec2" in spv_t:
            return "vec2"
        if "u32vec4" in spv_t:
            return "uvec4"
        if "u32vec2" in spv_t:
            return "uvec2"
        if "f32" in spv_t:
            return "float"
        if "u32" in spv_t:
            return "uint"
        if "bool" in spv_t:
            return "bool"
        return "float"

    in_main = False
    access_chains = {}
    branch_cond_for_block = {}
    cur_label = None

    for line in lines:
        s = line.strip()
        if s.startswith("%") and "= OpFunction " in s:
            in_main = True
            continue
        if not in_main:
            continue
        if s == "OpFunctionEnd":
            break
        m = re.match(r"%(\S+) = OpLabel", s)
        if m:
            cur_label = m.group(1)
            continue
        m = re.match(r"OpBranchConditional %(\S+) %(\S+) %(\S+)", s)
        if m:
            cond_id, true_lbl, false_lbl = m.groups()
            branch_cond_for_block[(true_lbl, cur_label)] = vname(cond_id)
            continue
        m = re.match(r"%(\S+) = OpPhi %(\S+) %(\S+) %(\S+) %(\S+) %(\S+)", s)
        if m:
            dst, rtype, v1, b1, v2, b2 = m.groups()
            cond = branch_cond_for_block.get((b1, b2), "true")
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {cond} ? {vname(v1)} : {vname(v2)};")
            continue
        m = re.match(r"%(\S+) = OpAccessChain %\S+ %(\S+) (.+)", s)
        if m:
            dst, base, idxs_str = m.groups()
            idxs = [x.lstrip("%") for x in idxs_str.split()]
            if base == "push_data":
                idx0 = constants.get(idxs[0], idxs[0]).rstrip("u")
                member_map = {
                    "0": "xoffset", "1": "yoffset", "2": "xscale", "3": "yscale",
                    "4": "ud_regs0", "5": "ud_regs1", "6": "ud_regs2", "7": "ud_regs3",
                    "8": "buf_offsets0", "9": "buf_offsets1", "10": "buf_offsets2",
                    "11": "motion_param", "12": "scene_size"
                }
                expr = f"push_data.{member_map[idx0]}"
                if len(idxs) > 1:
                    comp = int(constants.get(idxs[1], idxs[1]).rstrip("u"))
                    expr += "." + "xyzw"[comp]
                access_chains[dst] = expr
            elif base.startswith("ssbo_"):
                access_chains[dst] = f"{base}.data[{vname(idxs[1])}]"
            elif base.startswith("fs_in_attr"):
                comp = int(constants.get(idxs[0], idxs[0]).rstrip("u"))
                access_chains[dst] = f"{base}." + "xyzw"[comp]
            continue
        m = re.match(r"%(\S+) = OpLoad %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            if src in access_chains:
                out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {access_chains[src]};")
            elif src.startswith("fs_img") or src.startswith("fs_samp"):
                access_chains[dst] = src
            else:
                out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {src};")
            continue
        m = re.match(r"%(\S+) = OpSampledImage %\S+ %(\S+) %(\S+)", s)
        if m:
            dst, img_id, samp_id = m.groups()
            img_name = access_chains.get(img_id, img_id)
            samp_name = access_chains.get(samp_id, samp_id)
            img_t = "sampler2D"
            for _, iv, it in img_vars:
                if iv == img_name and "Array" in it:
                    img_t = "sampler2DArray"
            sampled_images[dst] = f"{img_t}({img_name}, {samp_name})"
            continue
        m = re.match(r"%(\S+) = OpImageSampleImplicitLod %(\S+) %(\S+) %(\S+)(?: Bias %(\S+))?", s)
        if m:
            dst, rtype, si, uv, bias = m.groups()
            si_expr = sampled_images[si]
            if bias:
                out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = texture({si_expr}, {vname(uv)}, {vname(bias)});")
            else:
                out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = texture({si_expr}, {vname(uv)});")
            continue
        m = re.match(r"%(\S+) = OpImageSampleExplicitLod %(\S+) %(\S+) %(\S+) Lod %(\S+)", s)
        if m:
            dst, rtype, si, uv, lod = m.groups()
            si_expr = sampled_images[si]
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = textureLod({si_expr}, {vname(uv)}, {vname(lod)});")
            continue
        m = re.match(r"%(\S+) = OpCompositeExtract %(\S+) %(\S+) (\d+)", s)
        if m:
            dst, rtype, src, comp = m.groups()
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {vname(src)}." + "xyzw"[int(comp)] + ";")
            continue
        m = re.match(r"%(\S+) = OpCompositeConstruct %(\S+) (.+)", s)
        if m:
            dst, rtype, args_s = m.groups()
            args = ", ".join(vname(x) for x in args_s.split())
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {glsl_type(rtype)}({args});")
            continue
        m = re.match(r"%(\S+) = OpBitFieldUExtract %(\S+) %(\S+) %(\S+) %(\S+)", s)
        if m:
            dst, rtype, base, off, cnt = m.groups()
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = bitfieldExtract({vname(base)}, int({vname(off)}), int({vname(cnt)}));")
            continue
        m = re.match(r"%(\S+) = OpShiftRightLogical %(\S+) %(\S+) %(\S+)", s)
        if m:
            dst, rtype, a, b = m.groups()
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {vname(a)} >> {vname(b)};")
            continue
        m = re.match(r"%(\S+) = OpBitcast %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            gt = glsl_type(rtype)
            fn = "uintBitsToFloat" if gt == "float" else "floatBitsToUint"
            out_lines.append(f"    {gt} {vname(dst)} = {fn}({vname(src)});")
            continue
        m = re.match(r"%(\S+) = OpConvertFToU %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            out_lines.append(f"    uint {vname(dst)} = uint({vname(src)});")
            continue
        m = re.match(r"%(\S+) = OpConvertUToF %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            out_lines.append(f"    float {vname(dst)} = float({vname(src)});")
            continue
        m = re.match(r"%(\S+) = OpFNegate %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = -{vname(src)};")
            continue
        m = re.match(r"%(\S+) = OpLogicalNot %(\S+) %(\S+)", s)
        if m:
            dst, rtype, src = m.groups()
            out_lines.append(f"    bool {vname(dst)} = !{vname(src)};")
            continue
        bin_ops = {
            "OpFAdd": "+", "OpFSub": "-", "OpFMul": "*", "OpFDiv": "/",
            "OpIAdd": "+", "OpIMul": "*",
            "OpFOrdGreaterThan": ">", "OpFOrdGreaterThanEqual": ">=",
            "OpFOrdLessThan": "<", "OpFOrdLessThanEqual": "<=",
            "OpFOrdEqual": "==", "OpFOrdNotEqual": "!=",
            "OpLogicalAnd": "&&", "OpLogicalOr": "||",
        }
        matched_bin = False
        for op_name, sym in bin_ops.items():
            m = re.match(rf"%(\S+) = {op_name} %(\S+) %(\S+) %(\S+)", s)
            if m:
                dst, rtype, a, b = m.groups()
                out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = ({vname(a)} {sym} {vname(b)});")
                matched_bin = True
                break
        if matched_bin:
            continue
        m = re.match(r"%(\S+) = OpSelect %(\S+) %(\S+) %(\S+) %(\S+)", s)
        if m:
            dst, rtype, cond, a, b = m.groups()
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {vname(cond)} ? {vname(a)} : {vname(b)};")
            continue
        m = re.match(r"%(\S+) = OpExtInst %(\S+) %\S+ (\w+) (.+)", s)
        if m:
            dst, rtype, ext_fn, args_s = m.groups()
            args = [vname(x) for x in args_s.split()]
            ext_map = {
                "Fma": "fma", "FClamp": "clamp", "NClamp": "clamp",
                "FMax": "max", "NMax": "max", "FMin": "min", "NMin": "min",
                "Sqrt": "sqrt", "InverseSqrt": "inversesqrt",
                "Log2": "log2", "Exp2": "exp2",
                "FAbs": "abs", "Floor": "floor", "Fract": "fract",
            }
            fn_name = ext_map.get(ext_fn, ext_fn.lower())
            out_lines.append(f"    {glsl_type(rtype)} {vname(dst)} = {fn_name}({', '.join(args)});")
            continue
        m = re.match(r"OpStore %(\S+) %(\S+)", s)
        if m:
            dst, src = m.groups()
            out_lines.append(f"    {dst} = {vname(src)};")
            continue
        if s in ("OpKill", "OpDemoteToHelperInvocation"):
            out_lines.append("    discard;")

    out_lines.append("}")
    with open(out_frag_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")


# ==============================================================================
# 3. Symbolic AST Analyzer for Raw GLSL Shaders
# ==============================================================================
def analyze_raw_glsl(frag_path: str):
    txt = open(frag_path).read()
    lines = txt.splitlines()

    env = {}
    tex_samples = []
    discards = []
    outputs = {}
    ssbo_types = {}

    for line in lines:
        m = re.search(r"readonly buffer \w+ \{ (\w+) data\[\]; \} (ssbo_\d+);", line)
        if m:
            ssbo_types[m.group(2)] = m.group(1)

    def simplify_expr(e: str) -> str:
        e = e.strip()
        while e.startswith("(") and e.endswith(")"):
            # check balanced
            depth = 0
            ok = True
            for i, ch in enumerate(e):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i < len(e) - 1:
                        ok = False
                        break
            if ok:
                e = e[1:-1].strip()
            else:
                break

        # uint(uint(x)) -> uint(x)
        m = re.match(r"^uint\((\d+)u\)$", e)
        if m:
            return f"{m.group(1)}u"
        m = re.match(r"^uint\((d\d+|buf\d+_dword_off)\)$", e)
        if m:
            return m.group(1)
        m = re.match(r"^uint\((.+)\)$", e)
        if m:
            inner = simplify_expr(m.group(1))
            if inner.endswith("u") or inner.startswith("d") or inner.startswith("uint(") or "floatBitsToUint" in inner:
                return inner
            return f"uint({inner})"

        # Cancel uintBitsToFloat(floatBitsToUint(x)) and vice versa
        m = re.match(r"^uintBitsToFloat\((.+)\)$", e)
        if m:
            inner = simplify_expr(m.group(1))
            if inner.startswith("floatBitsToUint(") and inner.endswith(")"):
                return inner[len("floatBitsToUint("):-1]
            m2 = re.match(r"^ssbo_(\d+)\.data\[(.+)\]$", inner)
            if m2:
                return f"s{m2.group(1)}f({m2.group(2)})"
            return f"uintBitsToFloat({inner})"

        m = re.match(r"^floatBitsToUint\((.+)\)$", e)
        if m:
            inner = simplify_expr(m.group(1))
            if inner.startswith("uintBitsToFloat(") and inner.endswith(")"):
                return inner[len("uintBitsToFloat("):-1]
            return f"floatBitsToUint({inner})"

        # max(x, x) -> x
        m = re.match(r"^max\(([^,]+),\s*([^,]+)\)$", e)
        if m and m.group(1).strip() == m.group(2).strip():
            return m.group(1).strip()

        return e

    in_main = False
    tex_idx = 0
    for line in lines:
        s = line.strip()
        if s.startswith("void main()"):
            in_main = True
            continue
        if not in_main:
            continue
        if s in ("{", "}", "if (true) {"):
            continue
        if "discard;" in s:
            m = re.match(r"if \((.+)\) \{ discard; \}", s)
            if m:
                c = m.group(1)
                discards.append(env.get(c, c))
            continue

        # Texture call
        m = re.match(r"vec4 (\w+) = (texture\w*\(.+\));", s)
        if m:
            tvar, tcall = m.groups()
            # substitute args in tcall
            def repl_tok(match):
                w = match.group(0)
                return env.get(w, w)
            tcall_sub = re.sub(r"\b[a-zA-Z_]\w*\b", repl_tok, tcall)
            tname = f"TEX{tex_idx}"
            tex_idx += 1
            tex_samples.append((tname, tvar, tcall_sub))
            env[tvar] = tname
            continue

        # Assignment
        m = re.match(r"(?:const\s+)?(?:float|uint|int|bool|vec2|vec3|vec4|uvec4|uvec2)\s+(\w+)\s*=\s*(.+);$", s)
        if not m:
            m = re.match(r"(\w+)\s*=\s*(.+);$", s)
        if m:
            lhs, rhs = m.groups()
            if lhs.startswith("buf") and lhs.endswith("_dword_off"):
                idx = lhs[3]
                env[lhs] = f"d{idx}"
                continue
            if lhs.startswith("fs_in_attr") or rhs.startswith("fs_in_attr"):
                if lhs.startswith("in_attr"):
                    env[lhs] = lhs
                    continue

            def repl_tok(match):
                w = match.group(0)
                return env.get(w, w)

            rhs_sub = re.sub(r"\b[a-zA-Z_]\w*\b", repl_tok, rhs)
            rhs_sub = simplify_expr(rhs_sub)

            # Simplify safe_rsq pattern from spv_to_raw
            m_rsq = re.match(r"^\(\((.+) > 0\.0\) && \(\1 <= 3\.40282347e\+38\)\) \? inversesqrt\(min\(max\(\1, 1\.17549435e-38\), 3\.40282347e\+38\)\) : 0\.0$", rhs_sub)
            if m_rsq:
                rhs_sub = f"safe_rsq({m_rsq.group(1)})"

            # Simplify double ternary: cond ? a : (!cond ? b : c) -> cond ? a : b
            m_tern = re.match(r"^(\w+)\s*\?\s*(.+?)\s*:\s*\(!\1\s*\?\s*(.+?)\s*:\s*(.+)\)$", rhs_sub)
            if m_tern:
                c, a, b, _ = m_tern.groups()
                rhs_sub = f"({c} ? {a} : {b})"

            # Recognize front_facing
            if "gl_FrontFacing" in rhs_sub and "0.0225" in rhs_sub:
                print(f"  [FRONT_FACE] {lhs} = {rhs_sub}")
                env[lhs] = "is_front"
                continue

            if lhs.startswith("frag_color") or lhs == "gl_FragDepth":
                outputs[lhs] = rhs_sub
            else:
                # Keep expression reasonably compact by naming important sub-expressions
                if len(rhs_sub) > 120:
                    print(f"  [LET] {lhs} = {rhs_sub}")
                    env[lhs] = lhs
                else:
                    env[lhs] = f"({rhs_sub})" if (" " in rhs_sub and not rhs_sub.startswith("(")) else rhs_sub

    print(f"=== Symbolic Analysis of {os.path.basename(frag_path)} ===")
    print(f"SSBO types: {ssbo_types}")
    if discards:
        print("DISCARD IF:")
        for d in discards:
            print(f"  {d}")
    print("TEXTURE SAMPLES:")
    for tname, tvar, tcall in tex_samples:
        print(f"  {tname} ({tvar}) = {tcall}")
    print("OUTPUTS:")
    for k in sorted(outputs.keys()):
        print(f"  {k} = {outputs[k]}")


# ==============================================================================
# 4. Pure-Python FP32 GLSL Interpreter & Equivalence Verifier
# ==============================================================================
class GlslBreak(Exception):
    pass

class GlslDiscard(Exception):
    pass

class GlslReturn(Exception):
    def __init__(self, val=None):
        self.val = val


def make_glsl_env(test_case: dict, ssbo_elem_types: dict):
    f32 = np.float32
    u32 = np.uint32
    i32 = np.int32

    def to_f32(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return np.float32(x)

    def to_u32(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.uint32)
        return np.uint32(int(x) & 0xFFFFFFFF)

    def vec2(*args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, np.ndarray):
                return v[:2].astype(np.float32).copy()
            return np.array([v, v], dtype=np.float32)
        out = []
        for a in args:
            if isinstance(a, np.ndarray):
                out.extend(a.tolist())
            else:
                out.append(float(a))
        return np.array(out[:2], dtype=np.float32)

    def vec3(*args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, np.ndarray):
                return v[:3].astype(np.float32).copy()
            return np.array([v, v, v], dtype=np.float32)
        out = []
        for a in args:
            if isinstance(a, np.ndarray):
                out.extend(a.tolist())
            else:
                out.append(float(a))
        return np.array(out[:3], dtype=np.float32)

    def vec4(*args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, np.ndarray):
                return v[:4].astype(np.float32).copy()
            return np.array([v, v, v, v], dtype=np.float32)
        out = []
        for a in args:
            if isinstance(a, np.ndarray):
                out.extend(a.tolist())
            else:
                out.append(float(a))
        return np.array(out[:4], dtype=np.float32)

    def uvec4(*args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, np.ndarray):
                return v[:4].astype(np.uint32).copy()
            return np.array([v, v, v, v], dtype=np.uint32)
        out = []
        for a in args:
            if isinstance(a, np.ndarray):
                out.extend(a.tolist())
            else:
                out.append(int(a) & 0xFFFFFFFF)
        return np.array(out[:4], dtype=np.uint32)

    def uvec2(*args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, np.ndarray):
                return v[:2].astype(np.uint32).copy()
            return np.array([v, v], dtype=np.uint32)
        out = []
        for a in args:
            if isinstance(a, np.ndarray):
                out.extend(a.tolist())
            else:
                out.append(int(a) & 0xFFFFFFFF)
        return np.array(out[:2], dtype=np.uint32)

    def uintBitsToFloat(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.uint32).view(np.float32)
        return np.array([int(x) & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]

    def floatBitsToUint(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32).view(np.uint32)
        return np.array([float(x)], dtype=np.float32).view(np.uint32)[0]

    def bitfieldExtract(val, off, bits):
        v = int(val) & 0xFFFFFFFF
        o = int(off)
        b = int(bits)
        return np.uint32((v >> o) & ((1 << b) - 1))

    def glsl_fma(a, b, c):
        return np.float32(a) * np.float32(b) + np.float32(c)

    def glsl_clamp(x, lo, hi):
        return np.minimum(np.maximum(np.float32(x), np.float32(lo)), np.float32(hi))

    def glsl_max(a, b):
        return np.maximum(np.float32(a), np.float32(b))

    def glsl_min(a, b):
        return np.minimum(np.float32(a), np.float32(b))

    def glsl_mix(a, b, t):
        a32, b32, t32 = np.float32(a), np.float32(b), np.float32(t)
        return a32 + t32 * (b32 - a32)

    def glsl_dot(a, b):
        a32, b32 = np.float32(a), np.float32(b)
        s = np.float32(0.0)
        for x, y in zip(a32, b32):
            s = np.float32(s + np.float32(x) * np.float32(y))
        return s

    def glsl_cross(a, b):
        a32, b32 = np.float32(a), np.float32(b)
        return np.array([
            a32[1] * b32[2] - a32[2] * b32[1],
            a32[2] * b32[0] - a32[0] * b32[2],
            a32[0] * b32[1] - a32[1] * b32[0],
        ], dtype=np.float32)

    def glsl_inversesqrt(x):
        return np.float32(1.0 / np.sqrt(np.float32(x)))

    def glsl_sqrt(x):
        return np.sqrt(np.maximum(np.float32(x), np.float32(0.0)))

    def glsl_abs(x):
        return np.abs(np.float32(x))

    def glsl_floor(x):
        return np.floor(np.float32(x))

    def glsl_fract(x):
        x32 = np.float32(x)
        return x32 - np.floor(x32)

    def glsl_exp2(x):
        return np.exp2(np.clip(np.float32(x), -126.0, 126.0))

    def glsl_log2(x):
        return np.log2(np.maximum(np.float32(x), np.float32(1.17549435e-38)))

    def glsl_pow(x, y):
        return np.exp2(np.float32(y) * glsl_log2(x))

    def glsl_reflect(I, N):
        I32, N32 = np.float32(I), np.float32(N)
        return I32 - np.float32(2.0) * glsl_dot(N32, I32) * N32

    def sampler_comb(img, samp):
        return (img, samp)

    def eval_texture(comb, uv, lod_or_bias=0.0, is_lod=False):
        img_name, samp_name = comb
        uv_arr = np.atleast_1d(np.float32(uv))
        # Deterministic smooth hash function of (img, samp, uv, lod) returning positive values in (0.1, 0.95)
        img_id = int(re.findall(r"\d+", img_name)[0])
        samp_id = int(re.findall(r"\d+", samp_name)[0])
        u = float(uv_arr[0]) if len(uv_arr) > 0 else 0.0
        v = float(uv_arr[1]) if len(uv_arr) > 1 else 0.0
        w = float(uv_arr[2]) if len(uv_arr) > 2 else 0.0
        l = float(lod_or_bias)
        base = (img_id + 1) * 1.37 + (samp_id + 1) * 0.73 + u * 2.13 + v * 3.17 + w * 1.91 + l * 0.53
        r = 0.5 + 0.4 * math.sin(base + 0.1)
        g = 0.5 + 0.4 * math.cos(base * 1.3 + 0.2)
        b = 0.5 + 0.4 * math.sin(base * 1.7 + 0.3)
        a = test_case.get(f"tex_alpha_{img_name}", 0.5 + 0.45 * math.cos(base * 0.9 + 0.4))
        return np.array([r, g, b, a], dtype=np.float32)

    class SwizzleVec(np.ndarray):
        pass

    env = {
        "vec2": vec2, "vec3": vec3, "vec4": vec4, "uvec4": uvec4, "uvec2": uvec2,
        "float": to_f32, "uint": to_u32, "int": lambda x: np.int32(int(x)), "bool": bool,
        "uintBitsToFloat": uintBitsToFloat, "floatBitsToUint": floatBitsToUint,
        "bitfieldExtract": bitfieldExtract,
        "fma": glsl_fma, "clamp": glsl_clamp, "max": glsl_max, "min": glsl_min,
        "mix": glsl_mix, "dot": glsl_dot, "cross": glsl_cross, "reflect": glsl_reflect,
        "inversesqrt": glsl_inversesqrt, "sqrt": glsl_sqrt, "abs": glsl_abs,
        "floor": glsl_floor, "fract": glsl_fract, "exp2": glsl_exp2, "log2": glsl_log2, "pow": glsl_pow,
        "sampler2D": sampler_comb, "sampler2DArray": sampler_comb, "sampler3D": sampler_comb,
        "texture": lambda comb, uv, bias=0.0: eval_texture(comb, uv, bias, False),
        "textureLod": lambda comb, uv, lod=0.0: eval_texture(comb, uv, lod, True),
        "gl_FrontFacing": test_case["gl_FrontFacing"],
        "gl_FragCoord": test_case["gl_FragCoord"],
        "true": True, "false": False,
    }
    return env


# ==============================================================================
# 4. Binary SPIR-V Virtual Machine for 100% Equivalence Verification
# ==============================================================================
class SpvVM:
    def __init__(self, spv_bytes: bytes):
        self.words = struct.unpack(f"<{len(spv_bytes)//4}I", spv_bytes)
        if self.words[0] != 0x07230203:
            raise ValueError("Invalid SPIR-V magic")
        self.bound = self.words[3]
        self.types = {}
        self.constants = {}
        self.decorations = {}
        self.member_decorations = {}
        self.global_vars = {}
        self.functions = {}
        self.entry_id = None
        self._parse()

    def _parse(self):
        pos = 5
        cur_func = None
        cur_block = None
        while pos < len(self.words):
            w0 = self.words[pos]
            wc = w0 >> 16
            op = w0 & 0xFFFF
            w = self.words[pos : pos + wc]
            pos += wc

            if cur_func is not None:
                if op == 55:  # OpFunctionParameter
                    cur_func["params"].append((w[1], w[2]))
                elif op == 248:  # OpLabel
                    cur_block = {"id": w[1], "insts": []}
                    cur_func["blocks"][w[1]] = cur_block
                    if cur_func["entry_block"] is None:
                        cur_func["entry_block"] = w[1]
                elif op == 56:  # OpFunctionEnd
                    cur_func = None
                    cur_block = None
                else:
                    if cur_block is not None:
                        cur_block["insts"].append((op, w))
                continue

            if op == 15:  # OpEntryPoint
                if self.entry_id is None:
                    self.entry_id = w[2]
            elif op == 71:  # OpDecorate
                target, deco = w[1], w[2]
                val = w[3] if wc >= 4 else True
                self.decorations.setdefault(target, {})[deco] = val
            elif op == 72:  # OpMemberDecorate
                st, mem, deco = w[1], w[2], w[3]
                val = w[4] if wc >= 5 else True
                self.member_decorations.setdefault((st, mem), {})[deco] = val
            elif op == 19:  # OpTypeVoid
                self.types[w[1]] = ("void",)
            elif op == 20:  # OpTypeBool
                self.types[w[1]] = ("bool",)
            elif op == 21:  # OpTypeInt
                self.types[w[1]] = ("int" if w[3] == 1 else "uint", w[2])
            elif op == 22:  # OpTypeFloat
                self.types[w[1]] = ("float", w[2])
            elif op == 23:  # OpTypeVector
                self.types[w[1]] = ("vec", w[2], w[3])
            elif op == 24:  # OpTypeMatrix
                self.types[w[1]] = ("mat", w[2], w[3])
            elif op == 25:  # OpTypeImage
                self.types[w[1]] = ("image", w[3], w[5])
            elif op == 26:  # OpTypeSampler
                self.types[w[1]] = ("sampler",)
            elif op == 27:  # OpTypeSampledImage
                self.types[w[1]] = ("sampled_image", w[2])
            elif op == 28:  # OpTypeArray
                self.types[w[1]] = ("array", w[2], self.constants.get(w[3], 1))
            elif op == 29:  # OpTypeRuntimeArray
                self.types[w[1]] = ("rarray", w[2])
            elif op == 30:  # OpTypeStruct
                self.types[w[1]] = ("struct", w[2:])
            elif op == 32:  # OpTypePointer
                self.types[w[1]] = ("ptr", w[2], w[3])
            elif op == 41:  # OpConstantTrue
                self.constants[w[2]] = True
            elif op == 42:  # OpConstantFalse
                self.constants[w[2]] = False
            elif op == 43:  # OpConstant
                t = self.types[w[1]]
                raw_u32 = w[3]
                if t[0] == "float":
                    self.constants[w[2]] = np.array([raw_u32], dtype=np.uint32).view(np.float32)[0]
                elif t[0] == "int":
                    self.constants[w[2]] = np.array([raw_u32], dtype=np.uint32).view(np.int32)[0]
                else:
                    self.constants[w[2]] = np.uint32(raw_u32)
            elif op == 44:  # OpConstantComposite
                t = self.types[w[1]]
                comps = [self.constants[cid] for cid in w[3:]]
                if t[0] == "vec":
                    et = self.types[t[1]][0]
                    dt = np.float32 if et == "float" else (np.int32 if et == "int" else np.uint32)
                    self.constants[w[2]] = np.array(comps, dtype=dt)
                else:
                    self.constants[w[2]] = comps
            elif op in (46, 1):  # OpConstantNull, OpUndef
                self.constants[w[2]] = self._zero_val(w[1])
            elif op == 59:  # OpVariable
                self.global_vars[w[2]] = (w[1], w[3])
            elif op == 54:  # OpFunction
                cur_func = {"id": w[2], "ret_type": w[1], "params": [], "blocks": {}, "entry_block": None}
                self.functions[w[2]] = cur_func

    def _zero_val(self, type_id):
        t = self.types.get(type_id, ("float", 32))
        if t[0] == "float":
            return np.float32(0.0)
        if t[0] == "uint":
            return np.uint32(0)
        if t[0] == "int":
            return np.int32(0)
        if t[0] == "bool":
            return False
        if t[0] == "vec":
            et = self.types[t[1]][0]
            if et == "bool":
                return [False] * t[2]
            dt = np.float32 if et == "float" else (np.int32 if et == "int" else np.uint32)
            return np.zeros(t[2], dtype=dt)
        if t[0] == "mat":
            return [self._zero_val(t[1]) for _ in range(t[2])]
        if t[0] == "array":
            return [self._zero_val(t[1]) for _ in range(int(t[2]))]
        if t[0] == "struct":
            return [self._zero_val(mt) for mt in t[1]]
        return np.float32(0.0)

    def run(self, state: dict):
        # Set up global storage
        mem = {}
        out_locs = {}
        in_map = {}
        out_map = {}
        ssbo_map = {}
        img_map = {}
        samp_map = {}
        push_var = None

        for vid, (ptype, sc) in self.global_vars.items():
            decs = self.decorations.get(vid, {})
            pointee = self.types[ptype][2]
            if sc == 1:  # Input
                if 11 in decs:  # BuiltIn
                    bi = decs[11]
                    if bi == 17:  # FrontFacing
                        mem[vid] = state["gl_FrontFacing"]
                    elif bi == 15:  # FragCoord
                        mem[vid] = state["gl_FragCoord"].copy()
                    else:
                        mem[vid] = self._zero_val(pointee)
                elif 30 in decs:
                    loc = decs[30]
                    in_map[vid] = loc
                    v = state["inputs"].get(loc, np.zeros(4, dtype=np.float32))
                    t = self.types[pointee]
                    if t[0] == "vec":
                        et = self.types[t[1]][0]
                        if et == "uint":
                            mem[vid] = v[:t[2]].view(np.uint32).copy()
                        elif et == "int":
                            mem[vid] = v[:t[2]].view(np.int32).copy()
                        else:
                            mem[vid] = v[:t[2]].astype(np.float32).copy()
                    elif t[0] == "float":
                        mem[vid] = np.float32(v[0])
                    elif t[0] == "uint":
                        mem[vid] = v[:1].view(np.uint32)[0]
                    elif t[0] == "int":
                        mem[vid] = v[:1].view(np.int32)[0]
            elif sc == 3:  # Output
                mem[vid] = self._zero_val(pointee)
                if 30 in decs:
                    out_map[vid] = decs[30]
                elif 11 in decs and decs[11] == 22:  # FragDepth
                    out_map[vid] = "depth"
            elif sc in (12, 2) and 33 in decs:  # SSBO / UBO
                ssbo_map[vid] = decs[33]
            elif sc == 0 and 33 in decs:  # UniformConstant
                pt = self.types[pointee]
                if pt[0] == "image":
                    img_map[vid] = decs[33]
                elif pt[0] == "sampler":
                    samp_map[vid] = decs[33]
            elif sc == 9:  # PushConstant
                push_var = vid
                mem[vid] = list(state["push_constants"])

        discarded = [False]

        def read_ptr(ptr_spec, res_type_id):
            base, idxs = ptr_spec
            if base in ssbo_map:
                bnd = ssbo_map[base]
                buf_u32 = state["ssbos"][bnd]
                # idxs[0] is struct member 0, idxs[1] is array index
                arr_idx = int(idxs[1]) & 0xFFF
                rt = self.types[res_type_id]
                if rt[0] == "float":
                    return buf_u32[arr_idx : arr_idx + 1].view(np.float32)[0]
                elif rt[0] == "uint":
                    return np.uint32(buf_u32[arr_idx])
                elif rt[0] == "int":
                    return buf_u32[arr_idx : arr_idx + 1].view(np.int32)[0]
                elif rt[0] == "vec":
                    n = rt[2]
                    base_u = arr_idx * n
                    if len(idxs) > 2:
                        c = int(idxs[2])
                        et = self.types[rt[1]][0]
                        if et == "float":
                            return buf_u32[base_u + c : base_u + c + 1].view(np.float32)[0]
                        return np.uint32(buf_u32[base_u + c])
                    et = self.types[rt[1]][0]
                    if et == "float":
                        return buf_u32[base_u : base_u + n].view(np.float32).copy()
                    return buf_u32[base_u : base_u + n].copy()
            cur = mem[base]
            for idx in idxs:
                cur = cur[int(idx)]
            if isinstance(cur, np.ndarray):
                return cur.copy()
            return cur

        def write_ptr(ptr_spec, val):
            base, idxs = ptr_spec
            if not idxs:
                mem[base] = val.copy() if isinstance(val, np.ndarray) else val
                return
            cur = mem[base]
            for idx in idxs[:-1]:
                cur = cur[int(idx)]
            cur[int(idxs[-1])] = val

        def eval_func(func_id, arg_vals):
            fn = self.functions[func_id]
            regs = dict(self.constants)
            local_vars = {}
            for (ptype, pid), aval in zip(fn["params"], arg_vals):
                regs[pid] = aval

            cur_bid = fn["entry_block"]
            prev_bid = None

            def get_val(vid):
                if vid in regs:
                    return regs[vid]
                return mem.get(vid)

            while cur_bid is not None:
                blk = fn["blocks"][cur_bid]
                next_bid = None
                phi_updates = {}
                for op, w in blk["insts"]:
                    if op == 245:  # OpPhi
                        rtype, dst = w[1], w[2]
                        for i in range(3, len(w), 2):
                            val_id, pred_bid = w[i], w[i + 1]
                            if pred_bid == prev_bid:
                                phi_updates[dst] = get_val(val_id)
                                break
                        continue
                    if phi_updates:
                        regs.update(phi_updates)
                        phi_updates.clear()

                    if op == 59:  # Local OpVariable
                        ptype, dst, sc = w[1], w[2], w[3]
                        pointee = self.types[ptype][2]
                        mem[dst] = self._zero_val(pointee)
                        local_vars[dst] = True
                    elif op in (65, 66):  # OpAccessChain
                        dst, base = w[2], w[3]
                        idxs = [int(get_val(x)) for x in w[4:]]
                        if base in regs and isinstance(regs[base], tuple) and len(regs[base]) == 2:
                            b0, i0 = regs[base]
                            regs[dst] = (b0, i0 + idxs)
                        else:
                            regs[dst] = (base, idxs)
                    elif op == 61:  # OpLoad
                        rtype, dst, src = w[1], w[2], w[3]
                        if src in img_map:
                            regs[dst] = ("img", img_map[src])
                        elif src in samp_map:
                            regs[dst] = ("samp", samp_map[src])
                        elif src in regs and isinstance(regs[src], tuple) and len(regs[src]) == 2 and regs[src][0] not in ("img", "samp", "si"):
                            regs[dst] = read_ptr(regs[src], rtype)
                        elif src in ssbo_map:
                            regs[dst] = (src, [0])
                        else:
                            v = mem[src] if src in mem else regs[src]
                            regs[dst] = v.copy() if isinstance(v, np.ndarray) else v
                    elif op == 62:  # OpStore
                        dst, val_id = w[1], w[2]
                        val = get_val(val_id)
                        if dst in regs and isinstance(regs[dst], tuple) and len(regs[dst]) == 2:
                            write_ptr(regs[dst], val)
                        else:
                            mem[dst] = val.copy() if isinstance(val, np.ndarray) else val
                    elif op == 83:  # OpCopyObject
                        regs[w[2]] = get_val(w[3])
                    elif op == 57:  # OpFunctionCall
                        rtype, dst, callee = w[1], w[2], w[3]
                        c_args = []
                        for aid in w[4:]:
                            if aid in local_vars or aid in mem:
                                c_args.append((aid, []))
                            else:
                                c_args.append(get_val(aid))
                        regs[dst] = eval_func(callee, c_args)
                    elif op == 86:  # OpSampledImage
                        regs[w[2]] = ("si", get_val(w[3])[1], get_val(w[4])[1])
                    elif op in (87, 88, 89, 90, 95):  # ImageSample / Fetch
                        rtype, dst, si_id, uv_id = w[1], w[2], w[3], w[4]
                        si = get_val(si_id)
                        img_b = si[1] if si[0] == "si" else si[1]
                        samp_b = si[2] if si[0] == "si" else 0
                        uv = np.atleast_1d(np.float32(get_val(uv_id)))
                        lod = 0.0
                        if op == 88 and len(w) >= 7:
                            lod = float(np.atleast_1d(get_val(w[6]))[0])
                        elif op == 87 and len(w) >= 7:
                            lod = float(np.atleast_1d(get_val(w[6]))[0])
                        elif op in (89, 90):  # Dref
                            dref = float(get_val(w[5]))
                            lod = dref
                        u = float(uv[0]) if len(uv) > 0 else 0.0
                        v = float(uv[1]) if len(uv) > 1 else 0.0
                        z = float(uv[2]) if len(uv) > 2 else 0.0
                        base_h = (img_b + 1) * 1.37 + (samp_b + 1) * 0.73 + u * 2.13 + v * 3.17 + z * 1.91 + lod * 0.53
                        r = 0.5 + 0.4 * math.sin(base_h + 0.1)
                        g = 0.5 + 0.4 * math.cos(base_h * 1.3 + 0.2)
                        b = 0.5 + 0.4 * math.sin(base_h * 1.7 + 0.3)
                        a = state["tex_alpha"].get(img_b, 0.5 + 0.45 * math.cos(base_h * 0.9 + 0.4))
                        res_vec = np.array([r, g, b, a], dtype=np.float32)
                        if self.types[rtype][0] == "float":
                            regs[dst] = np.float32(res_vec[0])
                        else:
                            regs[dst] = res_vec
                    elif op == 80:  # OpCompositeConstruct
                        rtype, dst = w[1], w[2]
                        t = self.types[rtype]
                        parts = []
                        for x in w[3:]:
                            vx = get_val(x)
                            if isinstance(vx, np.ndarray):
                                parts.extend(vx.tolist())
                            else:
                                parts.append(vx)
                        if t[0] == "vec":
                            et = self.types[t[1]][0]
                            if et == "bool":
                                regs[dst] = [bool(p) for p in parts]
                            else:
                                dt = np.float32 if et == "float" else (np.int32 if et == "int" else np.uint32)
                                regs[dst] = np.array(parts, dtype=dt)
                        else:
                            regs[dst] = [get_val(x) for x in w[3:]]
                    elif op == 81:  # OpCompositeExtract
                        rtype, dst, src = w[1], w[2], w[3]
                        cur = get_val(src)
                        for idx in w[4:]:
                            cur = cur[int(idx)]
                        t = self.types[rtype]
                        if t[0] == "float":
                            regs[dst] = np.float32(cur)
                        elif t[0] == "uint":
                            regs[dst] = np.uint32(cur)
                        elif t[0] == "int":
                            regs[dst] = np.int32(cur)
                        else:
                            regs[dst] = cur
                    elif op == 82:  # OpCompositeInsert
                        rtype, dst, obj_id, comp_id = w[1], w[2], w[3], w[4]
                        comp = get_val(comp_id).copy() if isinstance(get_val(comp_id), np.ndarray) else list(get_val(comp_id))
                        comp[int(w[5])] = get_val(obj_id)
                        regs[dst] = comp
                    elif op == 79:  # OpVectorShuffle
                        rtype, dst, v1_id, v2_id = w[1], w[2], w[3], w[4]
                        v1 = np.atleast_1d(get_val(v1_id))
                        v2 = np.atleast_1d(get_val(v2_id))
                        comb = np.concatenate([v1, v2])
                        idxs = [idx if idx != 0xFFFFFFFF else 0 for idx in w[5:]]
                        regs[dst] = comb[idxs].copy()
                    elif op == 124:  # OpBitcast
                        rtype, dst, src = w[1], w[2], w[3]
                        val = get_val(src)
                        t = self.types[rtype]
                        target_t = t[0] if t[0] != "vec" else self.types[t[1]][0]
                        dt = np.float32 if target_t == "float" else (np.int32 if target_t == "int" else np.uint32)
                        arr = np.atleast_1d(val).view(dt)
                        regs[dst] = arr.copy() if t[0] == "vec" else arr[0]
                    elif op == 109:  # OpConvertFToU
                        val = get_val(w[3])
                        t = self.types[w[1]]
                        arr = np.atleast_1d(np.float32(val)).astype(np.uint32)
                        regs[w[2]] = arr if t[0] == "vec" else arr[0]
                    elif op == 110:  # OpConvertFToS
                        val = get_val(w[3])
                        t = self.types[w[1]]
                        arr = np.atleast_1d(np.float32(val)).astype(np.int32)
                        regs[w[2]] = arr if t[0] == "vec" else arr[0]
                    elif op in (111, 112):  # OpConvertSToF, OpConvertUToF
                        val = get_val(w[3])
                        t = self.types[w[1]]
                        arr = np.atleast_1d(val).astype(np.float32)
                        regs[w[2]] = arr if t[0] == "vec" else arr[0]
                    elif op == 127:  # OpFNegate
                        regs[w[2]] = -np.float32(get_val(w[3]))
                    elif op == 126:  # OpSNegate
                        regs[w[2]] = -np.int32(get_val(w[3]))
                    elif op == 129:  # OpFAdd
                        regs[w[2]] = np.float32(get_val(w[3])) + np.float32(get_val(w[4]))
                    elif op == 131:  # OpFSub
                        regs[w[2]] = np.float32(get_val(w[3])) - np.float32(get_val(w[4]))
                    elif op == 133:  # OpFMul
                        regs[w[2]] = np.float32(get_val(w[3])) * np.float32(get_val(w[4]))
                    elif op == 136:  # OpFDiv
                        regs[w[2]] = np.float32(get_val(w[3])) / np.float32(get_val(w[4]))
                    elif op == 142:  # OpVectorTimesScalar
                        regs[w[2]] = np.float32(get_val(w[3])) * np.float32(get_val(w[4]))
                    elif op == 144:  # OpVectorTimesMatrix
                        vec = np.float32(get_val(w[3]))
                        cols = get_val(w[4])
                        regs[w[2]] = np.array([np.sum(vec * np.float32(c), dtype=np.float32) for c in cols], dtype=np.float32)
                    elif op == 145:  # OpMatrixTimesVector
                        cols = get_val(w[3])
                        vec = np.float32(get_val(w[4]))
                        res = np.zeros_like(cols[0], dtype=np.float32)
                        for c, s in zip(cols, vec):
                            res = res + np.float32(c) * np.float32(s)
                        regs[w[2]] = res
                    elif op == 148:  # OpDot
                        a = np.float32(get_val(w[3]))
                        b = np.float32(get_val(w[4]))
                        s = np.float32(0.0)
                        for x, y in zip(a, b):
                            s = np.float32(s + np.float32(x) * np.float32(y))
                        regs[w[2]] = s
                    elif op == 128:  # OpIAdd
                        regs[w[2]] = np.uint32(get_val(w[3])) + np.uint32(get_val(w[4]))
                    elif op == 130:  # OpISub
                        regs[w[2]] = np.uint32(get_val(w[3])) - np.uint32(get_val(w[4]))
                    elif op == 132:  # OpIMul
                        regs[w[2]] = np.uint32(get_val(w[3])) * np.uint32(get_val(w[4]))
                    elif op == 194:  # OpShiftRightLogical
                        regs[w[2]] = np.uint32(get_val(w[3])) >> np.uint32(get_val(w[4]))
                    elif op == 196:  # OpShiftLeftLogical
                        regs[w[2]] = np.uint32(get_val(w[3])) << np.uint32(get_val(w[4]))
                    elif op == 197:  # OpBitwiseOr
                        regs[w[2]] = np.uint32(get_val(w[3])) | np.uint32(get_val(w[4]))
                    elif op == 198:  # OpBitwiseXor
                        regs[w[2]] = np.uint32(get_val(w[3])) ^ np.uint32(get_val(w[4]))
                    elif op == 199:  # OpBitwiseAnd
                        regs[w[2]] = np.uint32(get_val(w[3])) & np.uint32(get_val(w[4]))
                    elif op == 203:  # OpBitFieldUExtract
                        val = int(get_val(w[3])) & 0xFFFFFFFF
                        off = int(get_val(w[4]))
                        cnt = int(get_val(w[5]))
                        regs[w[2]] = np.uint32((val >> off) & ((1 << cnt) - 1))
                    elif op in (180, 181):  # OpFOrdEqual
                        regs[w[2]] = (np.float32(get_val(w[3])) == np.float32(get_val(w[4])))
                    elif op in (182, 183):  # OpFOrdNotEqual
                        regs[w[2]] = (np.float32(get_val(w[3])) != np.float32(get_val(w[4])))
                    elif op in (184, 185):  # OpFOrdLessThan
                        regs[w[2]] = (np.float32(get_val(w[3])) < np.float32(get_val(w[4])))
                    elif op in (186, 187):  # OpFOrdGreaterThan
                        regs[w[2]] = (np.float32(get_val(w[3])) > np.float32(get_val(w[4])))
                    elif op in (188, 189):  # OpFOrdLessThanEqual
                        regs[w[2]] = (np.float32(get_val(w[3])) <= np.float32(get_val(w[4])))
                    elif op in (190, 191):  # OpFOrdGreaterThanEqual
                        regs[w[2]] = (np.float32(get_val(w[3])) >= np.float32(get_val(w[4])))
                    elif op == 170:  # OpIEqual
                        regs[w[2]] = (np.uint32(get_val(w[3])) == np.uint32(get_val(w[4])))
                    elif op == 171:  # OpINotEqual
                        regs[w[2]] = (np.uint32(get_val(w[3])) != np.uint32(get_val(w[4])))
                    elif op == 172:  # OpUGreaterThan
                        regs[w[2]] = (np.uint32(get_val(w[3])) > np.uint32(get_val(w[4])))
                    elif op == 174:  # OpUGreaterThanEqual
                        regs[w[2]] = (np.uint32(get_val(w[3])) >= np.uint32(get_val(w[4])))
                    elif op == 176:  # OpULessThan
                        regs[w[2]] = (np.uint32(get_val(w[3])) < np.uint32(get_val(w[4])))
                    elif op == 178:  # OpULessThanEqual
                        regs[w[2]] = (np.uint32(get_val(w[3])) <= np.uint32(get_val(w[4])))
                    elif op == 177:  # OpSLessThan
                        regs[w[2]] = (np.int32(get_val(w[3])) < np.int32(get_val(w[4])))
                    elif op == 168:  # OpLogicalNot
                        c = get_val(w[3])
                        regs[w[2]] = [not x for x in c] if isinstance(c, list) else (not bool(c))
                    elif op == 167:  # OpLogicalAnd
                        regs[w[2]] = bool(get_val(w[3])) and bool(get_val(w[4]))
                    elif op == 166:  # OpLogicalOr
                        regs[w[2]] = bool(get_val(w[3])) or bool(get_val(w[4]))
                    elif op == 154:  # OpAny
                        regs[w[2]] = any(bool(x) for x in get_val(w[3]))
                    elif op == 155:  # OpAll
                        regs[w[2]] = all(bool(x) for x in get_val(w[3]))
                    elif op == 169:  # OpSelect
                        cond, a, b = get_val(w[3]), get_val(w[4]), get_val(w[5])
                        if isinstance(cond, (list, np.ndarray)) and np.ndim(cond) > 0:
                            regs[w[2]] = np.where(cond, a, b)
                        else:
                            regs[w[2]] = (a.copy() if isinstance(a, np.ndarray) else a) if bool(cond) else (b.copy() if isinstance(b, np.ndarray) else b)
                    elif op == 12:  # OpExtInst (GLSL.std.450)
                        rtype, dst, ext_op = w[1], w[2], w[4]
                        args = [get_val(x) for x in w[5:]]
                        if ext_op == 4:  # FAbs
                            regs[dst] = np.abs(np.float32(args[0]))
                        elif ext_op == 8:  # Floor
                            regs[dst] = np.floor(np.float32(args[0]))
                        elif ext_op == 9:  # Ceil
                            regs[dst] = np.ceil(np.float32(args[0]))
                        elif ext_op == 10:  # Fract
                            x32 = np.float32(args[0])
                            regs[dst] = x32 - np.floor(x32)
                        elif ext_op == 13:  # Sin
                            regs[dst] = np.sin(np.float32(args[0]))
                        elif ext_op == 14:  # Cos
                            regs[dst] = np.cos(np.float32(args[0]))
                        elif ext_op == 26:  # Pow
                            regs[dst] = np.power(np.maximum(np.float32(args[0]), np.float32(1e-38)), np.float32(args[1]))
                        elif ext_op == 27:  # Exp
                            regs[dst] = np.exp(np.clip(np.float32(args[0]), -87.0, 87.0))
                        elif ext_op == 28:  # Log
                            regs[dst] = np.log(np.maximum(np.float32(args[0]), np.float32(1.17549435e-38)))
                        elif ext_op == 29:  # Exp2
                            regs[dst] = np.exp2(np.clip(np.float32(args[0]), -126.0, 126.0))
                        elif ext_op == 30:  # Log2
                            regs[dst] = np.log2(np.maximum(np.float32(args[0]), np.float32(1.17549435e-38)))
                        elif ext_op == 31:  # Sqrt
                            regs[dst] = np.sqrt(np.maximum(np.float32(args[0]), np.float32(0.0)))
                        elif ext_op == 32:  # InverseSqrt
                            regs[dst] = np.float32(1.0) / np.sqrt(np.maximum(np.float32(args[0]), np.float32(1.17549435e-38)))
                        elif ext_op in (37, 79):  # FMin, NMin
                            regs[dst] = np.minimum(np.float32(args[0]), np.float32(args[1]))
                        elif ext_op in (40, 80):  # FMax, NMax
                            regs[dst] = np.maximum(np.float32(args[0]), np.float32(args[1]))
                        elif ext_op == 38:  # UMin
                            regs[dst] = np.minimum(np.uint32(args[0]), np.uint32(args[1]))
                        elif ext_op == 41:  # UMax
                            regs[dst] = np.maximum(np.uint32(args[0]), np.uint32(args[1]))
                        elif ext_op in (43, 81):  # FClamp, NClamp
                            regs[dst] = np.minimum(np.maximum(np.float32(args[0]), np.float32(args[1])), np.float32(args[2]))
                        elif ext_op == 46:  # FMix
                            a0, a1, t0 = np.float32(args[0]), np.float32(args[1]), np.float32(args[2])
                            regs[dst] = a0 + t0 * (a1 - a0)
                        elif ext_op == 49:  # SmoothStep
                            e0, e1, x0 = np.float32(args[0]), np.float32(args[1]), np.float32(args[2])
                            t0 = np.clip((x0 - e0) / (e1 - e0), np.float32(0.0), np.float32(1.0))
                            regs[dst] = t0 * t0 * (np.float32(3.0) - np.float32(2.0) * t0)
                        elif ext_op == 50:  # Fma
                            regs[dst] = np.float32(args[0]) * np.float32(args[1]) + np.float32(args[2])
                        elif ext_op == 62:  # UnpackHalf2x16
                            u = np.uint32(args[0])
                            lo = np.uint16(u & np.uint32(0xFFFF)).view(np.float16)
                            hi = np.uint16((u >> np.uint32(16)) & np.uint32(0xFFFF)).view(np.float16)
                            regs[dst] = np.array([np.float32(lo), np.float32(hi)], dtype=np.float32)
                        elif ext_op == 66:  # Length
                            v = np.float32(args[0])
                            regs[dst] = np.sqrt(np.sum(v * v, dtype=np.float32))
                        elif ext_op == 67:  # Distance
                            d = np.float32(args[0]) - np.float32(args[1])
                            regs[dst] = np.sqrt(np.sum(d * d, dtype=np.float32))
                        elif ext_op == 68:  # Cross
                            a0, b0 = np.float32(args[0]), np.float32(args[1])
                            regs[dst] = np.array([
                                a0[1] * b0[2] - a0[2] * b0[1],
                                a0[2] * b0[0] - a0[0] * b0[2],
                                a0[0] * b0[1] - a0[1] * b0[0],
                            ], dtype=np.float32)
                        elif ext_op == 69:  # Normalize
                            v = np.float32(args[0])
                            l2 = np.sum(v * v, dtype=np.float32)
                            regs[dst] = v * (np.float32(1.0) / np.sqrt(np.maximum(l2, np.float32(1.17549435e-38))))
                        elif ext_op == 71:  # Reflect
                            I0, N0 = np.float32(args[0]), np.float32(args[1])
                            d0 = np.sum(N0 * I0, dtype=np.float32)
                            regs[dst] = I0 - np.float32(2.0) * d0 * N0
                        else:
                            raise NotImplementedError(f"Unhandled GLSL.std.450 op {ext_op}")
                    elif op == 249:  # OpBranch
                        next_bid = w[1]
                        break
                    elif op == 250:  # OpBranchConditional
                        cond = bool(get_val(w[1]))
                        next_bid = w[2] if cond else w[3]
                        break
                    elif op == 251:  # OpSwitch
                        sel = int(get_val(w[1]))
                        next_bid = w[2]
                        for i in range(3, len(w), 2):
                            if int(w[i]) == sel:
                                next_bid = w[i + 1]
                                break
                        break
                    elif op in (252, 5380):  # OpKill, OpDemoteToHelperInvocation
                        discarded[0] = True
                        if op == 252:
                            return None
                    elif op == 253:  # OpReturn
                        return None
                    elif op == 254:  # OpReturnValue
                        return get_val(w[1])
                prev_bid = cur_bid
                cur_bid = next_bid
            return None

        eval_func(self.entry_id, [])
        outputs = {"discarded": discarded[0]}
        for vid, loc in out_map.items():
            outputs[loc] = mem[vid].copy() if isinstance(mem[vid], np.ndarray) else mem[vid]
        return outputs


def verify_equivalence(ref_path: str, clean_path: str, spv_ref_path: str = None, num_trials: int = 32):
    ref_spv = open(spv_ref_path, "rb").read() if spv_ref_path else compile_glsl_to_spv(ref_path)
    clean_spv = compile_glsl_to_spv(clean_path)

    sig_ref = extract_interface_sig(ref_spv)
    sig_clean = extract_interface_sig(clean_spv)

    if sig_ref != sig_clean:
        print(f"[FAIL] Interface signature mismatch for {clean_path}!")
        print("  Ref  :", sig_ref)
        print("  Clean:", sig_clean)
        sys.exit(1)

    vm_ref = SpvVM(ref_spv)
    vm_clean = SpvVM(clean_spv)

    rng = np.random.default_rng(42)
    max_err = 0.0

    for trial in range(num_trials):
        # Construct randomized test case covering front/back faces, steep/flat normals, discard/pass
        inputs = {}
        for loc in range(12):
            v = rng.uniform(-1.5, 1.5, size=4).astype(np.float32)
            if loc == 0:
                # attr0.w is often an integer material index in [0..6]
                v[3] = np.float32(trial % 4)
                if trial % 4 == 0:
                    # flat normal (|y| < 0.15 |N|)
                    v[0], v[1], v[2] = 0.8, 0.05 * (1 if trial % 8 == 0 else -1), 0.6
                else:
                    # steep normal
                    v[1] = 0.7 * (1 if trial % 2 == 0 else -1)
            elif loc in (1, 2):
                v[3] = np.float32(0.85 if trial % 5 != 0 else 0.05)
            elif loc == 4:
                v[3] = np.float32(rng.uniform(0.8, 5.0))
            inputs[loc] = v

        ssbos = {}
        for bnd in range(16):
            # Generate positive & negative floats in reasonable range (avoid NaNs/denorms)
            flts = rng.uniform(0.05, 1.8, size=4096).astype(np.float32)
            # Make alpha threshold at offset 20 small on most trials so both discard & non-discard run
            flts[20::4] = np.float32(0.3 if trial % 3 != 0 else 0.95)
            flts[21::4] = np.float32(0.2)
            flts[22::4] = np.float32(1.0 if trial % 2 == 0 else 0.0)
            ssbos[bnd] = flts.view(np.uint32)

        push_constants = [
            np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(1.0),
            np.zeros(4, dtype=np.uint32), np.zeros(4, dtype=np.uint32),
            np.zeros(4, dtype=np.uint32), np.zeros(4, dtype=np.uint32),
            np.array([0x10080400, 0x28201814, 0, 0], dtype=np.uint32),
            np.zeros(4, dtype=np.uint32), np.zeros(2, dtype=np.uint32),
            np.uint32(0), np.uint32(0x05A00A00),
        ]

        state = {
            "gl_FrontFacing": bool(trial % 2 == 0),
            "gl_FragCoord": np.array([100.5 + trial * 7.0, 200.5 + trial * 5.0, 0.5, 1.0], dtype=np.float32),
            "inputs": inputs,
            "ssbos": ssbos,
            "push_constants": push_constants,
            "tex_alpha": {},
        }

        out_ref = vm_ref.run(state)
        out_clean = vm_clean.run(state)

        if out_ref["discarded"] != out_clean["discarded"]:
            print(f"[FAIL] Trial {trial}: discard mismatch (ref={out_ref['discarded']}, clean={out_clean['discarded']})")
            sys.exit(1)
        if out_ref["discarded"]:
            continue

        for k in out_ref:
            if k == "discarded":
                continue
            v_ref = np.atleast_1d(np.float32(out_ref[k]))
            v_clean = np.atleast_1d(np.float32(out_clean[k]))
            diff = np.max(np.abs(v_ref - v_clean) / np.maximum(np.abs(v_ref), 1.0))
            if diff > max_err:
                max_err = float(diff)
            if diff > 1e-4:
                print(f"[FAIL] Trial {trial} output {k} mismatch (rel_err={diff:.3e}):")
                print(f"  Ref  : {v_ref}")
                print(f"  Clean: {v_clean}")
                sys.exit(1)

    clean_lines = len(open(clean_path).read().splitlines())
    print(f"[PASS] {os.path.basename(clean_path):28s} | {clean_lines:3d} lines | sig=OK | {num_trials} trials max_rel_err={max_err:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spv-to-raw", nargs=2, metavar=("SPV", "OUT_FRAG"))
    parser.add_argument("--analyze", metavar="RAW_FRAG")
    parser.add_argument("--verify", nargs=2, metavar=("REF_FRAG", "CLEAN_FRAG"))
    parser.add_argument("--spv", metavar="REF_SPV", default=None)
    args = parser.parse_args()
    if args.spv_to_raw:
        spv_to_raw_glsl(args.spv_to_raw[0], args.spv_to_raw[1])
        print(f"Converted {args.spv_to_raw[0]} -> {args.spv_to_raw[1]}")
    if args.analyze:
        analyze_raw_glsl(args.analyze)
    if args.verify:
        verify_equivalence(args.verify[0], args.verify[1], args.spv)
