#!/usr/bin/env python3
"""
Freezes the unified Python launcher & game runner (Bloodborne.exe + Launcher.exe) via PyInstaller,
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
        err_lines = [l for l in proc.stdout.splitlines() if any(k in l for k in ("error ", "fatal error", "LNK", "Error:"))]
        tail = "\n".join((err_lines or proc.stdout.splitlines())[-20:])
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

    # 1. Freeze preparation/updater helper (bin/bb-helper.exe) via PyInstaller
    #    During normal gameplay, native Bloodborne.exe runs solo with 0% Python in RAM!
    run_checked([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "pyinstaller", "pillow"], "PIP_PYINSTALLER")

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
        "--name", "bb-helper",
        "--icon", str(root / "packaging" / "bbport.ico"),
        "--distpath", str(pyi_dist),
        "--workpath", str(pyi_work),
        "--specpath", str(pyi_work),
        "--paths", str(root),
        "--paths", str(root / "scripts"),
        "--paths", str(root / "launcher"),
        "--hidden-import", "run",
        "--hidden-import", "bbport_updater",
    ]
    for mod in hidden_modules:
        pyi_cmd.extend(["--hidden-import", mod])
    pyi_cmd.append(str(root / "run.py"))
    run_checked(pyi_cmd, "PYINSTALLER")

    bb_helper_dir = pyi_dist / "bb-helper"
    if not (bb_helper_dir / "bb-helper.exe").is_file():
        emit_error(f"bb-helper.exe was not generated in {bb_helper_dir}")
        sys.exit(1)

    # 2. Build native C++20 WebView2 Launcher (Launcher.exe) & zero-Python starter (Bloodborne.exe)
    import urllib.request
    wv2_dir = root / "dist_win" / "webview2_sdk"
    if not (wv2_dir / "build" / "native" / "include" / "WebView2.h").is_file():
        if wv2_dir.exists():
            shutil.rmtree(wv2_dir)
        wv2_dir.mkdir(parents=True, exist_ok=True)
        nupkg_path = root / "dist_win" / "webview2.nupkg"
        wv2_url = "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.2903.40"
        print(f"=== Downloading Microsoft.Web.WebView2 SDK ({wv2_url}) ===", flush=True)
        req = urllib.request.Request(wv2_url, headers={"User-Agent": "BloodbornePCVulkan-CI/1.1.9"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            nupkg_path.write_bytes(resp.read())
        with zipfile.ZipFile(nupkg_path, "r") as zf:
            zf.extractall(wv2_dir)
        nupkg_path.unlink(missing_ok=True)

    launcher_build = root / "dist_win" / "launcher_build"
    if launcher_build.exists():
        shutil.rmtree(launcher_build)
    run_checked([
        "cmake", "-S", str(root / "launcher_v2"), "-B", str(launcher_build),
        "-A", "x64",
        f"-DWEBVIEW2_SDK_DIR={wv2_dir.as_posix()}",
    ], "CMAKE_LAUNCHER_V2")
    run_checked([
        "cmake", "--build", str(launcher_build), "--config", "Release", "--parallel",
        "--target", "Launcher", "BloodborneStarter",
    ], "BUILD_LAUNCHER_V2")
    built_launcher_exe = launcher_build / "Release" / "Launcher.exe"
    built_starter_exe = launcher_build / "Release" / "Bloodborne.exe"
    if not built_launcher_exe.is_file() or not built_starter_exe.is_file():
        emit_error(f"Native C++20 executables missing: Launcher={built_launcher_exe.is_file()}, Bloodborne={built_starter_exe.is_file()}")
        sys.exit(1)

    # 3. Assemble staging directory
    stage = root / "dist_win" / "bloodborne-pc-vulkan-windows"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bin").mkdir(parents=True, exist_ok=True)
    (stage / "launcher").mkdir(parents=True, exist_ok=True)
    (stage / "launcher_v2").mkdir(parents=True, exist_ok=True)
    (stage / "scripts" / "bbsc_gameserver").mkdir(parents=True, exist_ok=True)

    # Keep frozen Python helper + _internal tucked neatly inside bin/ (spotless root!)
    shutil.copytree(bb_helper_dir, stage / "bin", dirs_exist_ok=True)
    shutil.copy2(built_launcher_exe, stage / "Launcher.exe")
    shutil.copy2(built_starter_exe, stage / "Bloodborne.exe")
    shutil.copytree(root / "launcher_v2" / "ui", stage / "launcher_v2" / "ui", dirs_exist_ok=True)
    for fname in ("play.bat", "run.py", "LICENSE"):
        src = root / fname
        if src.is_file():
            shutil.copy2(src, stage / fname)
    if (root / "launcher" / "bbport_updater.py").is_file():
        shutil.copy2(root / "launcher" / "bbport_updater.py", stage / "launcher" / "bbport_updater.py")
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
    if (root / "data").is_dir():
        shutil.copytree(root / "data", stage / "data", dirs_exist_ok=True)

    # 4. Copy & strip native Bloodborne.exe engine and bb-gpu-capabilities.exe + collect MSYS2 CLANG64 DLLs
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

    for exe_name in ("Bloodborne.exe", "bb-gpu-capabilities.exe"):
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

    queue = [stage / "bin" / "Bloodborne.exe", stage / "bin" / "bb-gpu-capabilities.exe"]
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
    ngx_dlssg_dll = dlss_sdk_dir / "lib" / "Windows_x86_64" / "rel" / "nvngx_dlssg.dll"
    if not built_bridge.is_file() or not ngx_dll.is_file() or not ngx_dlssg_dll.is_file():
        emit_error(
            f"Missing DLSS files: bridge={built_bridge.is_file()}, "
            f"nvngx_dlss={ngx_dll.is_file()}, nvngx_dlssg={ngx_dlssg_dll.is_file()}"
        )
        sys.exit(1)
    shutil.copy2(built_bridge, stage / "bin" / "bbport_dlss.dll")
    shutil.copy2(ngx_dll, stage / "bin" / "nvngx_dlss.dll")
    shutil.copy2(ngx_dlssg_dll, stage / "bin" / "nvngx_dlssg.dll")
    dlss_lic = dlss_sdk_dir / "LICENSE.txt"
    if dlss_lic.is_file():
        shutil.copy2(dlss_lic, stage / "NVIDIA-DLSS-LICENSE.txt")
    print("=== Bundled NVIDIA DLSS 4.5 / DLSS-G Bridge: bin/bbport_dlss.dll + bin/nvngx_dlss.dll + bin/nvngx_dlssg.dll ===", flush=True)

    # 5. Write .bbport-version.json, create zip archive + sha256 + windows-manifest.json
    sys.path.insert(0, str(root / "launcher"))
    import bbport_updater

    tag_name = os.environ.get("GITHUB_REF_NAME") or "v1.1.9"
    git_sha = os.environ.get("GITHUB_SHA") or ""
    bbport_updater.write_build_manifest(
        archive_path=Path(""),
        manifest_out_path=Path(""),
        version_out_path=stage / ".bbport-version.json",
        stage_root=stage,
        tag=tag_name,
        commit=git_sha,
        is_windows=True,
    )

    zip_path = root / "bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip"
    sha_path = root / "bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip.sha256"
    manifest_path = root / "windows-manifest.json"
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
    bbport_updater.write_build_manifest(
        archive_path=zip_path,
        manifest_out_path=manifest_path,
        version_out_path=stage / ".bbport-version.json",
        stage_root=stage,
        tag=tag_name,
        commit=git_sha,
        is_windows=True,
    )
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"=== Successfully packaged {zip_path.name} ({size_mb:.2f} MB), SHA256={digest} ===", flush=True)


if __name__ == "__main__":
    main()
