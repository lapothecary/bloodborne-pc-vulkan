#!/usr/bin/env python3
"""Full-game Havok Cloth (.hkx) census & schema verifier across Bloodborne (CUSA03173).

Scans all `parts/*.partsbnd.dcx`, `chr/*.chrbnd.dcx`, and `obj/*.objbnd.dcx` archives in parallel,
extracting every `hclClothContainer` packfile and cataloging:
  - Total cloth archives, `hclSimClothData` layers, particle counts, and substeps/damping/adapt flags
  - Every `hcl*` class (`ConstraintSet`, `Operator`, `Shape`, `Action`) used anywhere in the game
  - Exact collider & constraint breakdown of `bd_m_1300` Layer 2 (the 24-particle Hunter belt strap).
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import struct
import sys
import zlib

GAME_ROOT = "/home/kekw3/Documents/BloodBorne/CUSA03173/dvdroot_ps4"


def parse_hkx_packfile(hkx: bytes):
    sh_off = hkx.find(b"__classnames__")
    if sh_off < 0 or sh_off + 3 * 0x40 > len(hkx):
        return None
    cn_start = struct.unpack_from("<I", hkx, sh_off + 0x14)[0]
    d_start, l_fix, g_fix, v_fix, exp_off, _, _ = struct.unpack_from(
        "<7I", hkx, sh_off + 2 * 0x40 + 0x14
    )
    if d_start >= len(hkx) or d_start + exp_off > len(hkx):
        return None

    ptr_map = {}
    p = d_start + l_fix
    lim_l = min(len(hkx) - 8, d_start + g_fix)
    while p <= lim_l:
        src, dst = struct.unpack_from("<2I", hkx, p)
        if src == 0xFFFFFFFF:
            break
        ptr_map[src] = dst
        p += 8

    p = d_start + g_fix
    lim_g = min(len(hkx) - 12, d_start + v_fix)
    while p <= lim_g:
        src, sec, dst = struct.unpack_from("<3I", hkx, p)
        if src == 0xFFFFFFFF:
            break
        if sec == 2:
            ptr_map[src] = dst
        p += 12

    objs = []
    p = d_start + v_fix
    lim_v = min(len(hkx) - 12, d_start + exp_off)
    while p <= lim_v:
        data_off, _, cname_off = struct.unpack_from("<3I", hkx, p)
        if data_off == 0xFFFFFFFF:
            break
        s0 = cn_start + cname_off
        s1 = hkx.find(b"\x00", s0)
        if s0 < len(hkx) and s1 > s0:
            cname = hkx[s0:s1].decode("ascii", errors="replace")
            objs.append((data_off, cname))
        p += 12
    return d_start, ptr_map, objs


def scan_archive(args):
    cat, path = args
    try:
        raw = open(path, "rb").read()
        idx = raw.find(b"\x78\xda")
        if idx < 0:
            idx = raw.find(b"\x78\x9c")
        if idx < 0:
            return None
        bnd = zlib.decompress(raw[idx:])
    except Exception:
        return None

    if b"hclClothData" not in bnd and b"hclSimClothData" not in bnd:
        return None

    class_counts = Counter()
    sim_layers = 0
    total_particles = 0
    adapt_true = 0
    substeps_hist = Counter()
    shape_types = Counter()
    pinch_layers = 0
    anti_pinch_csets = 0

    pos = 0
    hkx_files = 0
    while True:
        off = bnd.find(b"\x57\xe0\xe0\x57", pos)
        if off < 0:
            break
        pos = off + 16
        # Bound packfile slice via __data__ end offset if valid:
        sh_off = bnd.find(b"__classnames__", off, off + 0x400)
        if sh_off < 0:
            continue
        end_data = struct.unpack_from("<2I", bnd, sh_off + 2 * 0x40 + 0x14 + 20)
        d_start = struct.unpack_from("<I", bnd, sh_off + 2 * 0x40 + 0x14)[0]
        end_off = d_start + end_data[1]
        hkx = bnd[off : off + max(end_off, 0x1000)]
        if b"hclSimClothData" not in hkx:
            continue

        parsed = parse_hkx_packfile(hkx)
        if not parsed:
            continue
        hkx_files += 1
        ds, pmap, objs = parsed
        for doff, cname in objs:
            if cname.startswith("hcl"):
                class_counts[cname] += 1
            base = ds + doff
            if cname == "hclSimClothData" and base + 0x180 <= len(hkx):
                sim_layers += 1
                n_p = struct.unpack_from("<i", hkx, base + 0x40)[0]
                n_ap = struct.unpack_from("<i", hkx, base + 0xD0)[0]
                pinch_en = hkx[base + 0x2C]
                total_particles += max(0, n_p)
                if pinch_en != 0:
                    pinch_layers += 1
                anti_pinch_csets += max(0, n_ap)
            elif cname == "hclSimulateOperator" and base + 0x40 <= len(hkx):
                sub = struct.unpack_from("<i", hkx, base + 0x20)[0]
                adp = hkx[base + 0x38]
                substeps_hist[sub] += 1
                if adp != 0:
                    adapt_true += 1
            elif cname.endswith("Shape") and cname.startswith("hcl") and base + 0x10 <= len(hkx):
                stype = struct.unpack_from("<I", hkx, base + 0x0C)[0]
                shape_types[f"{cname}(type={stype})"] += 1

    if hkx_files == 0:
        return None
    return {
        "cat": cat,
        "name": os.path.basename(path),
        "hkx_files": hkx_files,
        "sim_layers": sim_layers,
        "particles": total_particles,
        "classes": class_counts,
        "adapt_true": adapt_true,
        "substeps": substeps_hist,
        "shapes": shape_types,
        "pinch_layers": pinch_layers,
        "anti_pinch": anti_pinch_csets,
    }


def dump_bd_m_1300_belt():
    path = f"{GAME_ROOT}/parts/bd_m_1300.partsbnd.dcx"
    raw = open(path, "rb").read()
    idx = raw.find(b"\x78\xda")
    if idx < 0:
        idx = raw.find(b"\x78\x9c")
    bnd = zlib.decompress(raw[idx:])
    hkx = bnd[bnd.find(b"\x57\xe0\xe0\x57") :]
    ds, pmap, objs = parse_hkx_packfile(hkx)
    obj_by_off = dict(objs)
    sim_offs = [o for o, c in objs if c == "hclSimClothData"]
    print("\n=== Detailed Collider & Pinch Inspection of bd_m_1300 (All 5 Layers) ===")
    for li, sd in enumerate(sim_offs):
        b = ds + sd
        n_p = struct.unpack_from("<i", hkx, b + 0x40)[0]
        n_scol = struct.unpack_from("<i", hkx, b + 0xB0)[0]
        pinch_flags = list(hkx[b + 0x2C : b + 0x30])
        col_names = []
        if (sd + 0xA8) in pmap and n_scol > 0:
            arr = pmap[sd + 0xA8]
            for ci in range(n_scol):
                if (arr + ci * 8) in pmap:
                    co = pmap[arr + ci * 8]
                    cb = ds + co
                    pinch_c = hkx[cb + 0x80]
                    sh_o = pmap.get(co + 0x88, 0)
                    sh_cls = obj_by_off.get(sh_o, "?")
                    nm_o = pmap.get(co + 0x10, 0)
                    nm = (
                        hkx[ds + nm_o : hkx.find(b"\x00", ds + nm_o)].decode("ascii", "replace")
                        if nm_o
                        else "?"
                    )
                    col_names.append(f"{nm}:{sh_cls}(pinch={pinch_c})")
        masks_nz = 0
        if (sd + 0xF8) in pmap and n_p > 0:
            mp = ds + pmap[sd + 0xF8]
            masks = struct.unpack_from(f"<{n_p}I", hkx, mp)
            masks_nz = sum(1 for m in masks if m != 0)
        print(
            f"  Layer {li} (n_part={n_p:3d}): staticCols={n_scol:2d}, maskedParts={masks_nz:3d}/{n_p:3d}, cfgFlags={pinch_flags}"
        )
        if li == 2:
            for cn in col_names:
                print(f"     -> Belt Collider: {cn}")


def main():
    tasks = []
    for sub, pat in [
        ("parts", "*.partsbnd.dcx"),
        ("chr", "*.chrbnd.dcx"),
        ("obj", "*.objbnd.dcx"),
    ]:
        for f in sorted(glob.glob(f"{GAME_ROOT}/{sub}/{pat}")):
            if f.endswith("_l.partsbnd.dcx"):
                continue
            tasks.append((sub, f))

    print(f"Scanning {len(tasks)} primary archives across parts/, chr/, obj/ ...")
    by_cat = {k: [] for k in ("parts", "chr", "obj")}
    all_classes = Counter()
    all_shapes = Counter()
    all_substeps = Counter()
    tot_adapt = 0
    tot_pinch = 0
    tot_ap = 0

    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(scan_archive, tasks, chunksize=8):
            if not res:
                continue
            by_cat[res["cat"]].append(res)
            all_classes.update(res["classes"])
            all_shapes.update(res["shapes"])
            all_substeps.update(res["substeps"])
            tot_adapt += res["adapt_true"]
            tot_pinch += res["pinch_layers"]
            tot_ap += res["anti_pinch"]

    for cat in ("parts", "chr", "obj"):
        items = by_cat[cat]
        layers = sum(x["sim_layers"] for x in items)
        parts = sum(x["particles"] for x in items)
        samples = ", ".join(f"{x['name']}({x['sim_layers']}L/{x['particles']}p)" for x in items[:6])
        print(
            f"  [{cat:5s}] {len(items):3d} cloth archives | {layers:4d} SimCloth layers | {parts:6d} particles | e.g. {samples}"
        )

    print("\n=== Complete Global Census of hcl* Classes Across Entire Game + DLC ===")
    for k, v in sorted(all_classes.items()):
        print(f"  {k:40s}: {v:5d}")

    print("\n=== Collider Shapes & Substep / Adapt / Pinch Summary ===")
    print("  Shapes:", dict(all_shapes))
    print("  Substeps:", dict(sorted(all_substeps.items())))
    print(f"  adaptConstraintStiffness!=0 ops: {tot_adapt} | pinch_layers: {tot_pinch} | antiPinch_csets: {tot_ap}")

    dump_bd_m_1300_belt()


if __name__ == "__main__":
    sys.exit(main())
