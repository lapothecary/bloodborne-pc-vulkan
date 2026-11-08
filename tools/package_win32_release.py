#!/usr/bin/env python3
"""
Builds the Windows WPF Launcher (Bloodborne.exe), freezes the PyInstaller runner (bbport-run.exe),
collects MSYS2 CLANG64 runtime DLLs via PE32+ import table parsing, and packages
bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip + .sha256.
"""
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


def emit_error(msg: str) -> None:
    for line in msg.splitlines():
        line = line.strip()
        if line:
            print(f"::error::{line}", flush=True)


def run_checked(cmd: list[str], label: str) -> None:
    print(f"=== [{label}] Running: {' '.join(cmd)} ===", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        emit_error(f"{label} failed (exit {proc.returncode}):\n{tail}")
        sys.exit(proc.returncode)


def parse_pe_imports(pe_path: Path) -> list[str]:
    """Parse PE32/PE32+ import directory table to extract imported DLL names."""
    data = pe_path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        return []
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        return []
    num_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
    opt_header_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    opt_offset = pe_offset + 24
    if opt_offset + opt_header_size > len(data):
        return []
    magic = struct.unpack_from("<H", data, opt_offset)[0]
    if magic == 0x20B:  # PE32+ (64-bit)
        data_dir_offset = opt_offset + 112
    elif magic == 0x10B:  # PE32 (32-bit)
        data_dir_offset = opt_offset + 96
    else:
        return []
    if data_dir_offset + 16 > opt_offset + opt_header_size:
        return []
    import_rva, import_size = struct.unpack_from("<II", data, data_dir_offset + 8)
    if import_rva == 0 or import_size == 0:
        return []

    sec_table_offset = opt_offset + opt_header_size
    sections = []
    for i in range(num_sections):
        s_off = sec_table_offset + i * 40
        if s_off + 40 > len(data):
            break
        vsize, vrva, raw_size, raw_ptr = struct.unpack_from("<IIII", data, s_off + 8)
        sections.append((vrva, max(vsize, raw_size), raw_ptr, raw_size))

    def rva_to_offset(rva: int) -> int | None:
        for vrva, span, raw_ptr, raw_size in sections:
            if vrva <= rva < vrva + span:
                off = rva - vrva + raw_ptr
                return off if off < len(data) else None
        return None

    imp_off = rva_to_offset(import_rva)
    if imp_off is None:
        return []

    dlls = []
    while imp_off + 20 <= len(data):
        ilt_rva, ts, fwd, name_rva, iat_rva = struct.unpack_from("<IIIII", data, imp_off)
        if ilt_rva == 0 and name_rva == 0 and iat_rva == 0:
            break
        name_off = rva_to_offset(name_rva)
        if name_off is not None:
            end = data.find(b"\x00", name_off)
            if end != -1:
                dll_name = data[name_off:end].decode("ascii", errors="ignore").strip()
                if dll_name:
                    dlls.append(dll_name)
        imp_off += 20
    return dlls


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    # 1. Freeze Unified Python Launcher (Bloodborne.exe + bbport-run.exe) via PyInstaller
    run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "pyinstaller", "pillow"], "PIP_PYINSTALLER")

    hidden_modules = [
        "argparse", "base64", "bisect", "collections", "contextlib", "glob", "hashlib", "json",
        "math", "re", "shutil", "struct", "tempfile", "zlib", "io", "runpy", "xml.etree.ElementTree",
        "ctypes.wintypes", "socket", "urllib.request", "urllib.error", "http.server",
        "socketserver", "secrets", "uuid", "traceback", "datetime", "threading",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
    ]
    pyi_dist = root / "dist_win" / "pyi-dist"
    pyi_work = root / "dist_win" / "pyi-work"
    for d in (pyi_dist, pyi_work):
        if d.exists():
            shutil.rmtree(d)

    pyi_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--log-level", "WARN", "--windowed",
        "--name", "Bloodborne",
        "--icon", str(root / "packaging" / "bbport.ico"),
        "--distpath", str(pyi_dist),
        "--workpath", str(pyi_work),
        "--specpath", str(pyi_work),
        "--paths", str(root),
        "--paths", str(root / "scripts"),
        "--hidden-import", "run",
    ]
    for mod in hidden_modules:
        pyi_cmd.extend(["--hidden-import", mod])
    pyi_cmd.append(str(root / "launcher" / "bbport_launcher_win.py"))
    run_checked(pyi_cmd, "PYINSTALLER")

    bbport_run_dir = pyi_dist / "Bloodborne"
    if not (bbport_run_dir / "Bloodborne.exe").is_file():
        emit_error(f"Bloodborne.exe was not generated in {bbport_run_dir}")
        sys.exit(1)

    # 2. Assemble staging directory
    stage = root / "dist_win" / "bloodborne-pc-vulkan-windows"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bin").mkdir(parents=True, exist_ok=True)
    (stage / "launcher").mkdir(parents=True, exist_ok=True)
    (stage / "scripts" / "bbsc_gameserver").mkdir(parents=True, exist_ok=True)

    shutil.copytree(bbport_run_dir, stage, dirs_exist_ok=True)
    shutil.copy2(stage / "Bloodborne.exe", stage / "bbport-run.exe")
    for fname in ("play.bat", "run.py", "LICENSE"):
        src = root / fname
        if src.is_file():
            shutil.copy2(src, stage / fname)
    readme_win = root / "packaging" / "windows" / "README-Windows.txt"
    if readme_win.is_file():
        shutil.copy2(readme_win, stage / "README-Windows.txt")
    for icon_name in ("bloodborne.ico", "bloodborne.png"):
        src = root / "launcher" / icon_name
        if src.is_file():
            shutil.copy2(src, stage / "launcher" / icon_name)

    for py_file in (root / "scripts").glob("*.py"):
        shutil.copy2(py_file, stage / "scripts" / py_file.name)
    bbsc_dir = root / "scripts" / "bbsc_gameserver"
    if bbsc_dir.is_dir():
        for item in bbsc_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, stage / "scripts" / "bbsc_gameserver" / item.name)

    if (root / "patches").is_dir():
        shutil.copytree(root / "patches", stage / "patches", dirs_exist_ok=True)
    if (root / "fsr4_shaders").is_dir():
        shutil.copytree(root / "fsr4_shaders", stage / "fsr4_shaders", dirs_exist_ok=True)

    # 4. Copy & strip bb-probe.exe and bb-gpu-capabilities.exe + collect MSYS2 CLANG64 DLLs
    build_gpu = root / "build_win32_gpu"
    candidates = []
    if os.environ.get("MSYS2_CLANG64_BIN"):
        candidates.append(Path(os.environ["MSYS2_CLANG64_BIN"]))
    if os.environ.get("RUNNER_TEMP"):
        candidates.append(Path(os.environ["RUNNER_TEMP"]) / "msys64" / "clang64" / "bin")
    candidates.extend([
        Path("D:/a/_temp/msys64/clang64/bin"),
        Path("C:/msys64/clang64/bin"),
    ])
    clang64_bin = next((c for c in candidates if c.is_dir()), Path("C:/msys64/clang64/bin"))
    print(f"=== Using MSYS2 CLANG64 bin directory: {clang64_bin} (exists={clang64_bin.is_dir()}) ===", flush=True)
    if not clang64_bin.is_dir():
        emit_error(f"Could not locate MSYS2 clang64/bin directory among {[str(c) for c in candidates]}")
        sys.exit(1)

    llvm_strip = clang64_bin / "llvm-strip.exe"

    for exe_name in ("bb-probe.exe", "bb-gpu-capabilities.exe"):
        src_exe = build_gpu / exe_name
        dst_exe = stage / "bin" / exe_name
        if not src_exe.is_file():
            emit_error(f"Missing built executable: {src_exe}")
            sys.exit(1)
        if llvm_strip.is_file():
            res = subprocess.run([str(llvm_strip), "--strip-debug", "-o", str(dst_exe), str(src_exe)])
            if res.returncode != 0:
                shutil.copy2(src_exe, dst_exe)
        else:
            shutil.copy2(src_exe, dst_exe)

    # Map case-insensitive DLLs in clang64/bin
    clang_dll_map: dict[str, Path] = {}
    for p in clang64_bin.iterdir():
        if p.is_file() and p.suffix.lower() == ".dll":
            clang_dll_map[p.name.lower()] = p

    queue = [stage / "bin" / "bb-probe.exe", stage / "bin" / "bb-gpu-capabilities.exe"]
    seen: set[str] = {"vulkan-1.dll"}
    copied_dlls = []
    while queue:
        cur = queue.pop(0)
        for dll in parse_pe_imports(cur):
            key = dll.lower()
            if key in seen:
                continue
            seen.add(key)
            if key in clang_dll_map:
                src_dll = clang_dll_map[key]
                dst_dll = stage / "bin" / src_dll.name
                shutil.copy2(src_dll, dst_dll)
                copied_dlls.append(src_dll.name)
                queue.append(dst_dll)

    print(f"=== Bundled {len(copied_dlls)} MSYS2 CLANG64 runtime DLLs: {', '.join(sorted(copied_dlls))} ===", flush=True)
    if len(copied_dlls) < 8 or not (stage / "bin" / "SDL3.dll").is_file() or not (stage / "bin" / "libc++.dll").is_file():
        emit_error(f"Failed to bundle required MSYS2 DLLs (only {len(copied_dlls)} found in {clang64_bin})")
        sys.exit(1)

    # 4.5 Build NVIDIA DLSS 4.5 / DLAA 4 Bridge (bbport_dlss.dll) + Bundle nvngx_dlss.dll
    dlss_sdk_dir = root / "dist_win" / "dlss_sdk"
    dlss_build_dir = root / "dist_win" / "dlss_bridge"
    if not (dlss_sdk_dir / "include" / "nvsdk_ngx_vk.h").is_file():
        if dlss_sdk_dir.exists():
            shutil.rmtree(dlss_sdk_dir)
        run_checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/DLSS.git", str(dlss_sdk_dir)], "GIT_DLSS_SDK")
    if dlss_build_dir.exists():
        shutil.rmtree(dlss_build_dir)
    vulkan_inc = root / "dist_win" / "vk_headers_inc"
    if vulkan_inc.exists():
        shutil.rmtree(vulkan_inc)
    vulkan_inc.mkdir(parents=True, exist_ok=True)
    for vk_sub in ("vulkan", "vk_video"):
        src_vk = clang64_bin.parent / "include" / vk_sub
        if src_vk.is_dir():
            shutil.copytree(src_vk, vulkan_inc / vk_sub)
    run_checked([
        "cmake", "-S", str(root / "gpu" / "dlss_bridge"), "-B", str(dlss_build_dir),
        "-A", "x64",
        f"-DDLSS_SDK_ROOT={dlss_sdk_dir.as_posix()}",
        f"-DVULKAN_INCLUDE={vulkan_inc.as_posix()}",
    ], "CMAKE_DLSS_BRIDGE")
    run_checked(["cmake", "--build", str(dlss_build_dir), "--config", "Release"], "BUILD_DLSS_BRIDGE")
    built_bridge = dlss_build_dir / "Release" / "bbport_dlss.dll"
    if not built_bridge.is_file():
        built_bridge = dlss_build_dir / "bbport_dlss.dll"
    ngx_dll = dlss_sdk_dir / "lib" / "Windows_x86_64" / "rel" / "nvngx_dlss.dll"
    if not built_bridge.is_file() or not ngx_dll.is_file():
        emit_error(f"Missing DLSS files: bridge={built_bridge.is_file()}, nvngx_dlss={ngx_dll.is_file()}")
        sys.exit(1)
    shutil.copy2(built_bridge, stage / "bin" / "bbport_dlss.dll")
    shutil.copy2(ngx_dll, stage / "bin" / "nvngx_dlss.dll")
    dlss_lic = dlss_sdk_dir / "LICENSE.txt"
    if dlss_lic.is_file():
        shutil.copy2(dlss_lic, stage / "NVIDIA-DLSS-LICENSE.txt")
    print("=== Bundled NVIDIA DLSS 4.5 / DLAA 4 Bridge: bin/bbport_dlss.dll + bin/nvngx_dlss.dll ===", flush=True)

    # 5. Create zip archive + sha256
    zip_path = root / "bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip"
    sha_path = root / "bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip.sha256"
    if zip_path.exists():
        zip_path.unlink()

    print(f"=== Creating {zip_path.name} ===", flush=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(stage.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(stage.parent).as_posix()
                zf.write(file_path, arcname)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"=== Successfully packaged {zip_path.name} ({size_mb:.2f} MB), SHA256={digest} ===", flush=True)


if __name__ == "__main__":
    main()
