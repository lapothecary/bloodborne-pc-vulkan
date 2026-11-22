#!/usr/bin/env python3
"""Bundle portable Linux shared libraries into out/gpu/ and enforce ISA / TLS gates.

Guarantees that prebuilt Linux releases:
  1. Have ZERO x86-64-v4 (AVX-512) GNU property requirements in any executable or .so.
  2. Have ZERO AVX-512 (zmm) instructions in our compiled binaries (works on Ryzen 1000-5000,
     Intel Haswell-14th gen / Core Ultra, Steam Deck, etc.).
  3. Keep libbbgpu.so PT_TLS <= 0x124e8 so PS4 guest TLS fits safely.
  4. Bundle distro-varying shared libs (libSDL3, libfmt, libSPIRV-Tools, libxxhash, FFmpeg,
     protobuf/absl, miniupnpc) into out/gpu/ with $ORIGIN RPATH so the archive runs across
     Ubuntu, CachyOS, Arch, Fedora, Bazzite, and SteamOS without SONAME mismatches.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
GPU_DIR = OUT / "gpu"

# Core OS, glibc, C++ runtime, graphics driver, display server, and audio server libraries
# that MUST ALWAYS come from the host OS (never bundle these!).
SYSTEM_LIB_PREFIXES = (
    "linux-vdso",
    "ld-linux",
    "libc.so",
    "libm.so",
    "libmvec.so",
    "libpthread.so",
    "libdl.so",
    "librt.so",
    "libresolv.so",
    "libanl.so",
    "libutil.so",
    "libnsl.so",
    "libnss_",
    "libgcc_s.so",
    "libstdc++.so",
    "libgomp.so",
    "libatomic.so",
    "libz.so.1",
    "libvulkan.so",
    "libGL",
    "libEGL",
    "libGLES",
    "libdrm",
    "libgbm.so",
    "libX11",
    "libXext",
    "libXrender",
    "libXfixes",
    "libXcursor",
    "libXi",
    "libXrandr",
    "libXss",
    "libXau",
    "libXdmcp",
    "libXdamage",
    "libXcomposite",
    "libXxf86vm",
    "libxcb",
    "libxkbcommon",
    "libwayland",
    "libasound.so",
    "libpulse",
    "libpipewire",
    "libjack",
    "libsndio",
    "libdbus",
    "libglib-",
    "libgobject-",
    "libgio-",
    "libgmodule-",
    "libgthread-",
    "libgtk-",
    "libgdk",
    "libatk",
    "libpango",
    "libcairo",
    "libharfbuzz",
    "libfreetype",
    "libfontconfig",
    "libfribidi",
    "libthai",
    "libpixman",
    "libwebkit",
    "libjavascriptcore",
    "libsoup",
    "libexpat",
    "libffi",
    "libpcre",
    "libmount",
    "libblkid",
    "libselinux",
    "libsepol",
    "libsystemd",
    "libudev",
    "libcap.so",
    "libseccomp",
    "libaudit",
    "libpam",
    "libssl.so",
    "libcrypto.so",
    "libp11-kit",
    "libgpg-error",
    "libgcrypt",
    "libkrb5",
    "libk5crypto",
    "libgssapi",
    "libcom_err",
    "libkeyutils",
    "libOpenCL",
    "libva.so",
    "libva-drm",
    "libva-x11",
    "libvdpau",
    "libnvidia",
    "libcuda",
    "libbbgpu.so",
    "libbbnet20.so",
    "libbbport_dlss.so",
)


def is_system_lib(soname: str) -> bool:
    return any(soname.startswith(p) for p in SYSTEM_LIB_PREFIXES)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def parse_ldd(path: Path) -> list[tuple[str, Path]]:
    res = run(["ldd", str(path)], check=False)
    out: list[tuple[str, Path]] = []
    for line in res.stdout.splitlines():
        m = re.match(r"^\s*(\S+)\s*=>\s*(\S+)\s*\(", line)
        if not m:
            continue
        soname, resolved = m.group(1), m.group(2)
        if resolved == "not" or not os.path.isfile(resolved):
            continue
        out.append((soname, Path(resolved)))
    return out


def bundle_deps(targets: list[Path]) -> list[Path]:
    queue: list[Path] = [t for t in targets if t.is_file()]
    visited: set[Path] = set()
    bundled: dict[str, Path] = {}

    while queue:
        cur = queue.pop(0)
        real_cur = cur.resolve()
        if real_cur in visited:
            continue
        visited.add(real_cur)
        for soname, resolved in parse_ldd(cur):
            if is_system_lib(soname):
                continue
            dst = GPU_DIR / soname
            if soname not in bundled:
                if resolved.resolve() != dst.resolve():
                    shutil.copy2(resolved, dst)
                    os.chmod(dst, 0o755)
                bundled[soname] = dst
                queue.append(dst)

    print(f"[bundle_linux_deps] Bundled {len(bundled)} shared libraries into {GPU_DIR}")
    for name in sorted(bundled):
        print(f"  + {name}")
    return list(bundled.values())


def patch_rpaths(executables: list[Path], libs: list[Path]) -> None:
    if not shutil.which("patchelf"):
        print("[bundle_linux_deps] WARNING: patchelf not found, skipping RPATH patching")
        return
    for exe in executables:
        if exe.is_file():
            run(["patchelf", "--force-rpath", "--set-rpath", "$ORIGIN/gpu:$ORIGIN", str(exe)])
    for lib in libs:
        if lib.is_file():
            run(["patchelf", "--force-rpath", "--set-rpath", "$ORIGIN", str(lib)])


def verify_isa_and_tls(own_binaries: list[Path], all_check_files: list[Path]) -> None:
    errors: list[str] = []

    # 1. Check GNU_PROPERTY_X86_ISA_1_NEEDED across every binary and bundled library.
    for f in all_check_files:
        if not f.is_file():
            continue
        res = run(["readelf", "-n", str(f)], check=False)
        for line in res.stdout.splitlines():
            if "x86 ISA needed:" in line and "x86-64-v4" in line:
                errors.append(f"FORBIDDEN x86-64-v4 ISA needed in {f}: {line.strip()}")

    # 2. Check our own compiled binaries for any AVX-512 (zmm) instructions.
    for f in own_binaries:
        if not f.is_file():
            continue
        res = run(["objdump", "-d", str(f)], check=False)
        for line in res.stdout.splitlines():
            if re.search(r"%zmm\d+", line):
                errors.append(f"FORBIDDEN AVX-512 zmm instruction in {f}: {line.strip()}")
                break

    # 3. Check PT_TLS <= 0x124e8 on out/gpu/libbbgpu.so.
    bbgpu = GPU_DIR / "libbbgpu.so"
    if bbgpu.is_file():
        res = run(["readelf", "-lW", str(bbgpu)], check=False)
        for line in res.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "TLS" and len(parts) >= 6:
                memsz = int(parts[5], 16)
                print(f"[bundle_linux_deps] libbbgpu.so PT_TLS = {memsz:#x} (limit 0x124e8)")
                if memsz > 0x124E8:
                    errors.append(f"libbbgpu.so PT_TLS {memsz:#x} exceeds safe limit 0x124e8!")

    if errors:
        print("[bundle_linux_deps] VERIFICATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print("[bundle_linux_deps] ISA & TLS verification PASSED (0 x86-64-v4, 0 zmm, PT_TLS OK)")


def main() -> None:
    own_exes = [OUT / "bb-vulkan", OUT / "bb-gpu-capabilities"]
    own_libs = [GPU_DIR / "libbbgpu.so", GPU_DIR / "libbbnet20.so"]
    launcher = OUT / "bb-launcher"

    bundled_libs = bundle_deps(own_exes + [l for l in own_libs if l.is_file()])
    patch_rpaths(own_exes, [l for l in own_libs if l.is_file()] + bundled_libs)

    own_all = [p for p in own_exes + [launcher] + own_libs if p.is_file()]
    verify_isa_and_tls(own_all, own_all + bundled_libs)


if __name__ == "__main__":
    main()
