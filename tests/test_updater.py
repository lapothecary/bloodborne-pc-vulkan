#!/usr/bin/env python3
"""Regression tests for launcher/bbport_updater.py (tag pinning & TOCTOU SHA-256 recovery)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launcher"))

import bbport_updater  # noqa: E402


def _make_tar_gz(files: dict[str, bytes]) -> tuple[bytes, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    return raw, hashlib.sha256(raw).hexdigest().lower()


class _FakeResp:
    def __init__(self, payload: bytes):
        self._bio = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        return self._bio.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class UpdaterTests(unittest.TestCase):
    def test_archive_url_pinned_to_manifest_tag(self):
        with tempfile.TemporaryDirectory() as td:
            port_dir = Path(td)
            manifest = {
                "tag": "v1.1.6",
                "commit": "8e6a032dc1d38c639899fc26feb6422445cb01e4",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_sha256": "0" * 64,
                "archive_size": 1234,
                "core_files": {"out/gpu/libbbgpu.so": "1" * 64},
            }
            seen_urls: list[str] = []

            def fake_get(url: str, **_kw) -> bytes:
                seen_urls.append(url)
                return json.dumps(manifest).encode("utf-8")

            with mock.patch.object(bbport_updater, "_http_get_bytes", side_effect=fake_get):
                res = bbport_updater.check_for_update(port_dir, is_windows=False)

            self.assertTrue(res["ok"])
            self.assertIn("?_=", seen_urls[0])
            self.assertEqual(
                res["archive_url"],
                f"https://github.com/{bbport_updater.GITHUB_REPO}/releases/download/v1.1.6/{bbport_updater.LINUX_ARCHIVE}",
            )

    def test_stale_v115_info_refreshes_to_v116_before_download(self):
        """Reproduces screenshot bug: launcher cached v1.1.5 SHA, then user clicked Update after v1.1.6 shipped."""
        with tempfile.TemporaryDirectory() as td:
            port_dir = Path(td)
            (port_dir / "out" / "gpu").mkdir(parents=True)
            (port_dir / "out" / "gpu" / "libbbgpu.so").write_bytes(b"old-v115-so")

            new_so = b"new-v116-so"
            new_so_sha = hashlib.sha256(new_so).hexdigest()
            arc_bytes, arc_sha = _make_tar_gz({"out/gpu/libbbgpu.so": new_so})

            stale_info = {
                "ok": True,
                "up_to_date": False,
                "tag": "v1.1.5",
                "commit": "4e53dee",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_url": f"https://github.com/{bbport_updater.GITHUB_REPO}/releases/latest/download/{bbport_updater.LINUX_ARCHIVE}",
                "archive_sha256": "3562ddd10c592606c55fb95f5fbcdd40a7d1285eecf6f0850e2bfe942f889209",
                "archive_size": 51408478,
            }
            fresh_manifest = {
                "tag": "v1.1.6",
                "commit": "8e6a032dc1d38c639899fc26feb6422445cb01e4",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_sha256": arc_sha,
                "archive_size": len(arc_bytes),
                "core_files": {"out/gpu/libbbgpu.so": new_so_sha},
            }

            def fake_urlopen(req, timeout=30.0):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "linux-manifest.json" in url:
                    return _FakeResp(json.dumps(fresh_manifest).encode("utf-8"))
                self.assertIn("/releases/download/v1.1.6/", url)
                return _FakeResp(arc_bytes)

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                res = bbport_updater.download_and_apply_update(port_dir, stale_info, is_windows=False)

            self.assertTrue(res["ok"], res)
            self.assertEqual((port_dir / "out" / "gpu" / "libbbgpu.so").read_bytes(), new_so)
            state = bbport_updater.read_local_state(port_dir)
            self.assertEqual(state["tag"], "v1.1.6")
            self.assertEqual(state["archive_sha256"], arc_sha)

    def test_corrupted_download_still_aborts_safely(self):
        with tempfile.TemporaryDirectory() as td:
            port_dir = Path(td)
            (port_dir / "out" / "gpu").mkdir(parents=True)
            (port_dir / "out" / "gpu" / "libbbgpu.so").write_bytes(b"original-good")

            good_arc, good_sha = _make_tar_gz({"out/gpu/libbbgpu.so": b"good"})
            corrupted_arc = good_arc + b"CORRUPTED"
            manifest = {
                "tag": "v1.1.6",
                "commit": "8e6a032",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_sha256": good_sha,
                "archive_size": len(good_arc),
                "core_files": {"out/gpu/libbbgpu.so": "a" * 64},
            }
            info = {
                "ok": True,
                "up_to_date": False,
                "tag": "v1.1.6",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_url": f"https://github.com/{bbport_updater.GITHUB_REPO}/releases/download/v1.1.6/{bbport_updater.LINUX_ARCHIVE}",
                "archive_sha256": good_sha,
                "archive_size": len(good_arc),
            }

            def fake_urlopen(req, timeout=30.0):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "linux-manifest.json" in url:
                    return _FakeResp(json.dumps(manifest).encode("utf-8"))
                return _FakeResp(corrupted_arc)

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                res = bbport_updater.download_and_apply_update(port_dir, info, is_windows=False)

            self.assertFalse(res["ok"])
            self.assertIn("SHA-256 mismatch", res["error"])
            self.assertEqual((port_dir / "out" / "gpu" / "libbbgpu.so").read_bytes(), b"original-good")

    def test_windows_powershell_script_handles_cyrillic_and_no_cmd_find(self):
        import base64
        staged = Path(r"C:\Users\Иван O'Connor\Загрузки\bb\bloodborne-pc-vulkan-windows\.update_tmp\extracted")
        port_dir = Path(r"C:\Users\Иван O'Connor\Загрузки\bb\bloodborne-pc-vulkan-windows")
        tmp_root = Path(r"C:\Users\Иван O'Connor\Загрузки\bb\bloodborne-pc-vulkan-windows\.update_tmp")
        relaunch_exe = Path(r"C:\Users\Иван O'Connor\Загрузки\bb\bloodborne-pc-vulkan-windows\Launcher.exe")

        ps, encoded = bbport_updater._build_windows_apply_ps_script(
            staged=staged,
            port_dir=port_dir,
            tmp_root=tmp_root,
            relaunch_exe=relaunch_exe,
            relaunch_args=[],
            wait_pids=[8564, 4120],
        )
        decoded = base64.b64decode(encoded).decode("utf-16-le")
        self.assertEqual(decoded, ps)
        # Cyrillic & apostrophe escaping preserved cleanly in UTF-16LE
        self.assertIn("Загрузки", decoded)
        self.assertIn("O''Connor", decoded)
        # Never invokes cmd.exe / tasklist / find / timeout
        self.assertNotIn("tasklist", decoded)
        self.assertNotIn("timeout ", decoded)
        self.assertIn("Get-Process -Id $wp", decoded)
        self.assertIn("Remove-Item Env:_MEIPASS2", decoded)
        self.assertIn("& 'robocopy.exe' @rcArgs", decoded)
        self.assertIn("Remove-Item -LiteralPath $tmpRoot -Recurse -Force", decoded)

    def test_progress_cb_two_arg_contract(self):
        with tempfile.TemporaryDirectory() as td:
            port_dir = Path(td)
            (port_dir / "out" / "gpu").mkdir(parents=True)
            (port_dir / "out" / "gpu" / "libbbgpu.so").write_bytes(b"old")
            new_so = b"new-v118"
            arc_bytes, arc_sha = _make_tar_gz({"out/gpu/libbbgpu.so": new_so})
            manifest = {
                "tag": "v1.1.8",
                "commit": "c360570",
                "archive_name": bbport_updater.LINUX_ARCHIVE,
                "archive_sha256": arc_sha,
                "archive_size": len(arc_bytes),
                "core_files": {"out/gpu/libbbgpu.so": hashlib.sha256(new_so).hexdigest()},
            }
            info = dict(manifest, ok=True, up_to_date=False,
                        archive_url=f"https://github.com/{bbport_updater.GITHUB_REPO}/releases/download/v1.1.8/{bbport_updater.LINUX_ARCHIVE}")
            events: list[tuple[int, str]] = []

            def cb(frac, detail):
                events.append((int(round(float(frac) * 100.0)), str(detail)))

            def fake_urlopen(req, timeout=30.0):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "linux-manifest.json" in url:
                    return _FakeResp(json.dumps(manifest).encode("utf-8"))
                return _FakeResp(arc_bytes)

            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                res = bbport_updater.download_and_apply_update(port_dir, info, progress_cb=cb, is_windows=False)

            self.assertTrue(res["ok"], res)
            self.assertGreaterEqual(len(events), 4)
            self.assertEqual(events[-1][0], 100)


if __name__ == "__main__":
    unittest.main()
