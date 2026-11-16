#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""One-click GitHub Release self-updater for Bloodborne PC Vulkan (Windows & Linux).

Verifies updates cryptographically by SHA-256 (both per-binary core file hashes and full
archive SHA-256), so both new tags (v1.1.6+) and hotfix rebuilds under the same tag (v1.1.5)
are detected accurately without re-downloading when already up to date.

User data (`user/savedata`, `bbport.ini`, `launcher-win.json`, `mods/`, `mods.json`,
`patches.json`, `logs/`, and `CUSA*` game folders) is strictly preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

GITHUB_REPO = "lapothecary/bloodborne-pc-vulkan"
USER_AGENT = "BloodbornePCVulkan-Updater/1.1.6"
VERSION_STATE_FILE = ".bbport-version.json"

LINUX_ARCHIVE = "bloodborne-pc-vulkan-linux-x86_64-prebuilt.tar.gz"
LINUX_MANIFEST = "linux-manifest.json"
WINDOWS_ARCHIVE = "bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip"
WINDOWS_MANIFEST = "windows-manifest.json"

LINUX_CORE_FILES = (
    "out/gpu/libbbgpu.so",
    "out/bb-probe",
    "Launcher.py",
    "launcher/bbport_launcher.py",
)
WINDOWS_CORE_FILES = (
    "bin/bb-probe.exe",
    "Launcher.py",
    "run.py",
)

PROTECTED_NAMES = {
    "user",
    "mods",
    "logs",
    "bbport.ini",
    "launcher-win.json",
    "mods.json",
    "patches.json",
    ".update_tmp",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(256 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest().lower()
    except OSError:
        return ""


def _http_get_bytes(url: str, timeout: float = 12.0, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def read_local_state(port_dir: Path) -> dict:
    p = Path(port_dir) / VERSION_STATE_FILE
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def local_core_fingerprint(port_dir: Path, is_windows: bool | None = None) -> str:
    if is_windows is None:
        is_windows = sys.platform == "win32"
    port_dir = Path(port_dir)
    primary = port_dir / ("bin/bb-probe.exe" if is_windows else "out/gpu/libbbgpu.so")
    digest = sha256_file(primary)
    return digest[:12] if digest else "unknown"


def _parse_sha256_text(text: str) -> str:
    for line in text.splitlines():
        parts = line.strip().split()
        if parts and len(parts[0]) == 64 and all(c in "0123456789abcdefABCDEF" for c in parts[0]):
            return parts[0].lower()
    return ""


def check_for_update(port_dir: Path, is_windows: bool | None = None) -> dict:
    """Checks GitHub Releases (`latest`) and compares SHA-256 hashes against `port_dir`.

    Returns dict with keys:
      ok: bool
      up_to_date: bool
      tag: str
      commit: str
      archive_name: str
      archive_url: str
      archive_sha256: str
      archive_size: int
      local_short_hash: str
      remote_short_hash: str
      published_at: str
      error: str (when ok is False)
    """
    if is_windows is None:
        is_windows = sys.platform == "win32"
    port_dir = Path(port_dir)
    archive_name = WINDOWS_ARCHIVE if is_windows else LINUX_ARCHIVE
    manifest_name = WINDOWS_MANIFEST if is_windows else LINUX_MANIFEST
    base_dl = f"https://github.com/{GITHUB_REPO}/releases/latest/download"

    local_state = read_local_state(port_dir)
    local_fp = local_core_fingerprint(port_dir, is_windows)

    manifest = {}
    try:
        raw_m = _http_get_bytes(f"{base_dl}/{manifest_name}", timeout=10.0)
        parsed = json.loads(raw_m.decode("utf-8", errors="replace"))
        if isinstance(parsed, dict):
            manifest = parsed
    except Exception:
        manifest = {}

    tag = str(manifest.get("tag") or "")
    commit = str(manifest.get("commit") or "")
    archive_sha256 = str(manifest.get("archive_sha256") or "").lower()
    archive_size = int(manifest.get("archive_size") or 0)
    archive_url = f"{base_dl}/{archive_name}"
    published_at = str(manifest.get("published_at") or "")
    remote_core: dict[str, str] = {}
    if isinstance(manifest.get("core_files"), dict):
        remote_core = {str(k): str(v).lower() for k, v in manifest["core_files"].items() if v}

    # Fallback / enrichment via .sha256 sidecar or GitHub REST API if manifest was absent.
    if not archive_sha256:
        try:
            sha_txt = _http_get_bytes(f"{base_dl}/{archive_name}.sha256", timeout=10.0).decode(
                "utf-8", errors="replace"
            )
            archive_sha256 = _parse_sha256_text(sha_txt)
        except Exception:
            pass

    if not tag or not archive_sha256 or archive_size <= 0:
        try:
            api_raw = _http_get_bytes(
                f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
                timeout=10.0,
                accept="application/vnd.github+json",
            )
            rel = json.loads(api_raw.decode("utf-8", errors="replace"))
            tag = tag or str(rel.get("tag_name") or "")
            published_at = published_at or str(rel.get("published_at") or "")
            for asset in rel.get("assets") or []:
                if asset.get("name") == archive_name:
                    archive_url = str(asset.get("browser_download_url") or archive_url)
                    archive_size = archive_size or int(asset.get("size") or 0)
                    digest = str(asset.get("digest") or "")
                    if not archive_sha256 and digest.startswith("sha256:"):
                        archive_sha256 = digest.split(":", 1)[1].strip().lower()
        except urllib.error.URLError as exc:
            if not archive_sha256:
                return {
                    "ok": False,
                    "up_to_date": False,
                    "local_short_hash": local_fp,
                    "error": f"Network error: {exc.reason if hasattr(exc, 'reason') else exc}",
                }
        except Exception as exc:
            if not archive_sha256:
                return {
                    "ok": False,
                    "up_to_date": False,
                    "local_short_hash": local_fp,
                    "error": str(exc),
                }

    if not archive_sha256:
        return {
            "ok": False,
            "up_to_date": False,
            "local_short_hash": local_fp,
            "error": f"Release asset {archive_name} (or its SHA-256) was not found on GitHub.",
        }

    # Determine whether local files already match the remote build 1:1.
    core_matches = False
    if remote_core:
        core_matches = True
        for rel_path, expected_hash in remote_core.items():
            actual_hash = sha256_file(port_dir / rel_path)
            if not actual_hash or actual_hash != expected_hash:
                core_matches = False
                break

    recorded_archive_sha = str(local_state.get("archive_sha256") or "").lower()
    if remote_core:
        # Both freshly unpacked archives and updater-installed copies have matching core_files!
        up_to_date = core_matches and (not recorded_archive_sha or recorded_archive_sha == archive_sha256)
    else:
        up_to_date = bool(recorded_archive_sha and recorded_archive_sha == archive_sha256)

    primary_key = "bin/bb-probe.exe" if is_windows else "out/gpu/libbbgpu.so"
    remote_fp = (remote_core.get(primary_key) or archive_sha256)[:12]

    return {
        "ok": True,
        "up_to_date": up_to_date,
        "tag": tag or "latest",
        "commit": commit[:7] if commit else "",
        "archive_name": archive_name,
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "core_files": remote_core,
        "local_short_hash": local_fp,
        "remote_short_hash": remote_fp,
        "published_at": published_at,
        "error": "",
    }


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")
        zf.extractall(dest_dir)


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
            if member.issym() or member.islnk():
                link_target = (target.parent / member.linkname).resolve()
                if link_target != dest_resolved and dest_resolved not in link_target.parents:
                    raise RuntimeError(f"Unsafe symlink in tar archive: {member.name}")
        try:
            tf.extractall(dest_dir, filter="tar")
        except TypeError:
            tf.extractall(dest_dir)


def _unwrap_single_root(staged: Path) -> Path:
    entries = [p for p in staged.iterdir()]
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name not in ("out", "bin", "launcher", "scripts"):
        return entries[0]
    return staged


def download_and_apply_update(
    port_dir: Path,
    info: dict,
    progress_cb=None,
    is_windows: bool | None = None,
) -> dict:
    """Downloads the release archive, verifies SHA-256, and applies the update safely.

    progress_cb(fraction: float, stage: str) is called periodically from the worker thread.
    Returns {"ok": True, "restarting_external": bool} or {"ok": False, "error": str}.
    """
    if is_windows is None:
        is_windows = sys.platform == "win32"
    port_dir = Path(port_dir).resolve()
    tmp_root = port_dir / ".update_tmp"
    archive_name = str(info["archive_name"])
    archive_url = str(info["archive_url"])
    expected_sha = str(info["archive_sha256"]).lower()
    expected_size = int(info.get("archive_size") or 0)

    def report(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, frac)), msg)

    try:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        archive_path = tmp_root / archive_name

        report(0.01, "Connecting to GitHub Releases…")
        req = urllib.request.Request(archive_url, headers={"User-Agent": USER_AGENT})
        hasher = hashlib.sha256()
        downloaded = 0
        t0 = time.monotonic()

        with urllib.request.urlopen(req, timeout=30.0) as resp, archive_path.open("wb") as out_f:
            total = int(resp.headers.get("Content-Length") or expected_size or 0)
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                elapsed = max(0.001, time.monotonic() - t0)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                dl_mb = downloaded / (1024 * 1024)
                if total > 0:
                    tot_mb = total / (1024 * 1024)
                    frac = 0.02 + 0.78 * (downloaded / total)
                    report(frac, f"Downloading: {dl_mb:.1f} / {tot_mb:.1f} MB ({speed_mb:.1f} MB/s)")
                else:
                    report(0.40, f"Downloading: {dl_mb:.1f} MB ({speed_mb:.1f} MB/s)")

        report(0.82, "Verifying SHA-256 checksum…")
        actual_sha = hasher.hexdigest().lower()
        if actual_sha != expected_sha:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return {
                "ok": False,
                "error": (
                    f"SHA-256 mismatch! Expected {expected_sha[:16]}…, "
                    f"got {actual_sha[:16]}…. Update aborted safely."
                ),
            }

        report(0.86, "Extracting verified package…")
        extract_dir = tmp_root / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if archive_name.endswith(".zip"):
            _safe_extract_zip(archive_path, extract_dir)
        else:
            _safe_extract_tar(archive_path, extract_dir)
        archive_path.unlink(missing_ok=True)

        staged = _unwrap_single_root(extract_dir)
        for pname in PROTECTED_NAMES:
            cand = staged / pname
            if cand.is_symlink() or cand.is_file():
                cand.unlink(missing_ok=True)
            elif cand.is_dir():
                shutil.rmtree(cand, ignore_errors=True)

        core_list = WINDOWS_CORE_FILES if is_windows else LINUX_CORE_FILES
        core_hashes = {}
        for rel in core_list:
            p = staged / rel
            if p.is_file():
                core_hashes[rel] = sha256_file(p)

        version_state = {
            "tag": info.get("tag") or "latest",
            "commit": info.get("commit") or "",
            "archive_name": archive_name,
            "archive_sha256": actual_sha,
            "core_files": core_hashes,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staged / VERSION_STATE_FILE).write_text(
            json.dumps(version_state, indent=2) + "\n", encoding="utf-8"
        )

        if is_windows:
            report(0.96, "Applying update & restarting Launcher…")
            pid = os.getpid()
            relaunch_exe = Path(sys.executable).resolve()
            if relaunch_exe.name.lower().startswith("python"):
                relaunch_cmd = f'start "" "{relaunch_exe}" "{port_dir / "Launcher.py"}"'
            elif (port_dir / "Launcher.exe").is_file():
                relaunch_cmd = f'start "" "{port_dir / "Launcher.exe"}"'
            else:
                relaunch_cmd = f'start "" "{relaunch_exe}" --launcher'

            bat_path = tmp_root / "apply_update.cmd"
            bat_content = (
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                "setlocal\r\n"
                f"for /L %%i in (1,1,20) do (\r\n"
                f"    tasklist /FI \"PID eq {pid}\" 2>nul | find \"{pid}\" >nul\r\n"
                "    if errorlevel 1 goto :copy_files\r\n"
                "    timeout /t 1 /nobreak >nul\r\n"
                ")\r\n"
                ":copy_files\r\n"
                f"robocopy \"{staged}\" \"{port_dir}\" /E /IS /IT /R:8 /W:1 "
                "/XF bbport.ini launcher-win.json mods.json patches.json "
                "/XD user mods logs .update_tmp >nul\r\n"
                f"if exist \"{port_dir / 'out' / 'patches.bin'}\" del /F /Q \"{port_dir / 'out' / 'patches.bin'}\" >nul 2>&1\r\n"
                f"{relaunch_cmd}\r\n"
                f"rmdir /S /Q \"{tmp_root}\" >nul 2>&1\r\n"
            )
            bat_path.write_text(bat_content, encoding="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
            subprocess.Popen(["cmd.exe", "/c", str(bat_path)], cwd=str(port_dir), creationflags=flags)
            report(1.0, "Restarting…")
            return {"ok": True, "restarting_external": True}

        report(0.94, "Installing files atomically…")
        exec_names = {
            "play.sh",
            "launcher.sh",
            "run.sh",
            "Launcher.py",
            "bbport_launcher.py",
            "bb-launcher.sh",
            "bb-probe",
            "probe",
            "bb-gpu-capabilities",
        }
        for src_item in staged.rglob("*"):
            rel = src_item.relative_to(staged)
            if rel.parts and rel.parts[0] in PROTECTED_NAMES:
                continue
            dst_item = port_dir / rel
            if src_item.is_symlink():
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                tmp_link = dst_item.parent / f".{dst_item.name}.upd"
                tmp_link.unlink(missing_ok=True)
                os.symlink(os.readlink(src_item), tmp_link)
                os.replace(tmp_link, dst_item)
            elif src_item.is_dir():
                if not dst_item.is_symlink():
                    dst_item.mkdir(parents=True, exist_ok=True)
            elif src_item.is_file():
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                tmp_dst = dst_item.parent / f".{dst_item.name}.upd"
                shutil.copy2(src_item, tmp_dst)
                if src_item.name in exec_names:
                    tmp_dst.chmod(tmp_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                os.replace(tmp_dst, dst_item)

        # Keep out/probe synced with out/bb-probe when out/probe exists in local layout.
        bb_probe = port_dir / "out" / "bb-probe"
        legacy_probe = port_dir / "out" / "probe"
        if bb_probe.is_file() and (legacy_probe.is_file() or not legacy_probe.exists()):
            tmp_p = port_dir / "out" / ".probe.upd"
            shutil.copy2(bb_probe, tmp_p)
            tmp_p.chmod(tmp_p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.replace(tmp_p, legacy_probe)

        # Invalidate cached binary patch table so updated scripts/patches.py applies on next launch.
        (port_dir / "out" / "patches.bin").unlink(missing_ok=True)

        shutil.rmtree(tmp_root, ignore_errors=True)
        report(1.0, "Update complete!")
        return {"ok": True, "restarting_external": False}
    except Exception as exc:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return {"ok": False, "error": str(exc)}


def write_build_manifest(
    archive_path: Path,
    manifest_out_path: Path,
    version_out_path: Path,
    stage_root: Path,
    tag: str,
    commit: str,
    is_windows: bool,
) -> None:
    """Helper called by CI packaging scripts to emit `<platform>-manifest.json` and `.bbport-version.json`."""
    core_list = WINDOWS_CORE_FILES if is_windows else LINUX_CORE_FILES
    core_hashes = {}
    for rel in core_list:
        p = Path(stage_root) / rel
        if p.is_file():
            core_hashes[rel] = sha256_file(p)

    # First write .bbport-version.json if caller builds the archive after this, or update manifest after archive.
    version_payload = {
        "tag": tag,
        "commit": commit,
        "core_files": core_hashes,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if version_out_path:
        Path(version_out_path).write_text(json.dumps(version_payload, indent=2) + "\n", encoding="utf-8")

    if archive_path and Path(archive_path).is_file() and manifest_out_path:
        arc = Path(archive_path)
        manifest_payload = {
            "tag": tag,
            "commit": commit,
            "archive_name": arc.name,
            "archive_sha256": sha256_file(arc),
            "archive_size": arc.stat().st_size,
            "core_files": core_hashes,
            "published_at": version_payload["built_at"],
        }
        Path(manifest_out_path).write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build manifest & SHA-256 updater helper")
    parser.add_argument("--write-version", metavar="OUT_JSON", help="Write .bbport-version.json for stage root")
    parser.add_argument(
        "--write-manifest",
        nargs=3,
        metavar=("ARCHIVE", "MANIFEST_JSON", "VERSION_JSON"),
        help="Write <platform>-manifest.json + .bbport-version.json after packaging archive",
    )
    parser.add_argument("--stage-root", default=".", help="Root directory containing packaged files")
    parser.add_argument("--windows", action="store_true", help="Use Windows core file list")
    args = parser.parse_args()

    tag_val = os.environ.get("GITHUB_REF_NAME") or "v1.1.6"
    sha_val = os.environ.get("GITHUB_SHA") or ""
    if args.write_version:
        write_build_manifest(
            Path(""), Path(""), Path(args.write_version), Path(args.stage_root), tag_val, sha_val, args.windows
        )
    if args.write_manifest:
        arc_p, man_p, ver_p = args.write_manifest
        write_build_manifest(
            Path(arc_p), Path(man_p), Path(ver_p), Path(args.stage_root), tag_val, sha_val, args.windows
        )

