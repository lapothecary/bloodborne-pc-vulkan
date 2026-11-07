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

    # 1. Build Windows WPF Launcher (Bloodborne.exe)
    os.environ["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    os.environ["DOTNET_NOLOGO"] = "1"
    wpf_out = root / "dist_win" / "wpf"
    if wpf_out.exists():
        shutil.rmtree(wpf_out)
    run_checked([
        "dotnet", "publish", "launcher/windows",
        "-c", "Release", "-r", "win-x64", "--self-contained", "true",
        "-p:PublishSingleFile=true",
        "-p:IncludeNativeLibrariesForSelfExtract=true",
        "-p:EnableCompressionInSingleFile=true",
        "-p:DebugType=none",
        "-o", str(wpf_out),
    ], "DOTNET_WPF")

    bloodborne_exe = wpf_out / "Bloodborne.exe"
    if not bloodborne_exe.is_file():
        emit_error(f"Bloodborne.exe was not generated in {wpf_out}")
        sys.exit(1)

    # 2. Build PyInstaller runner (bbport-run.exe)
    run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "pyinstaller", "pillow"], "PIP_PYINSTALLER")

    hidden_modules = [
        "argparse", "base64", "bisect", "collections", "contextlib", "glob", "hashlib", "json",
        "math", "re", "shutil", "struct", "tempfile", "zlib", "io", "runpy", "xml.etree.ElementTree",
        "ctypes.wintypes", "socket", "urllib.request", "urllib.error", "http.server",
        "socketserver", "secrets", "uuid", "traceback", "datetime", "threading",
    ]
    pyi_dist = root / "dist_win" / "pyi-dist"
    pyi_work = root / "dist_win" / "pyi-work"
    for d in (pyi_dist, pyi_work):
        if d.exists():
            shutil.rmtree(d)

    pyi_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--log-level", "WARN", "--console",
        "--name", "bbport-run",
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

    bbport_run_dir = pyi_dist / "bbport-run"
    if not (bbport_run_dir / "bbport-run.exe").is_file():
        emit_error(f"bbport-run.exe was not generated in {bbport_run_dir}")
        sys.exit(1)

    # 3. Assemble staging directory
    stage = root / "dist_win" / "bloodborne-pc-vulkan-windows"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bin").mkdir(parents=True, exist_ok=True)
    (stage / "launcher").mkdir(parents=True, exist_ok=True)
    (stage / "scripts" / "bbsc_gameserver").mkdir(parents=True, exist_ok=True)

    shutil.copytree(bbport_run_dir, stage, dirs_exist_ok=True)
    shutil.copy2(bloodborne_exe, stage / "Bloodborne.exe")
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
    clang64_bin = Path("C:/msys64/clang64/bin")
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

    # Map case-insensitive DLLs in C:/msys64/clang64/bin
    clang_dll_map: dict[str, Path] = {}
    if clang64_bin.is_dir():
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
