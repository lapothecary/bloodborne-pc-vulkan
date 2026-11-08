#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bloodborne PC Vulkan — Standalone Settings Launcher (Launcher.py).

Mirrors the Linux GTK4/Libadwaita 8-page launcher design in pure Python (100% English UI).
Edits bbport.ini and launcher-win.json / settings.json with live 1-second two-way sync:
- Changes made in Launcher.py are immediately written to bbport.ini and remembered by Bloodborne.exe.
- Changes made live inside the game (System -> Display / Graphics / Controls or F8) update bbport.ini
  and are immediately reflected in Launcher.py.
- Clicking Play launches Bloodborne.exe (on Windows) or run.sh (on Linux).
"""

import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    HAS_TK = True
except ImportError:
    class _DummyTk:
        Canvas = object
        Frame = object
    tk = _DummyTk()
    filedialog = messagebox = ttk = None
    HAS_TK = False

if getattr(sys, "frozen", False):
    PORT_DIR = Path(sys.executable).resolve().parent
else:
    _here = Path(__file__).resolve().parent
    PORT_DIR = _here.parent if _here.name == "launcher" else _here

SCRIPTS_DIR = PORT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DATA_DIR = Path(os.environ.get("BB_DATA_DIR") or PORT_DIR)
INI_FILE = Path(os.environ.get("BB_CONFIG") or DATA_DIR / "bbport.ini")
WIN_SETTINGS_FILE = DATA_DIR / "launcher-win.json"
LINUX_SETTINGS_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "bb_vulkan-launcher"
    / "settings.json"
)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MAX_LOG_LINES = 5000

# --- Choices & Schema (100% English, matching Screenshot 2) ---------------------------

ASPECTS = ["auto", "4:3", "16:10", "16:9", "21:9", "32:9"]
ASPECT_RATIOS = {"4:3": 4 / 3, "16:10": 16 / 10, "16:9": 16 / 9, "21:9": 64 / 27, "32:9": 32 / 9}
OUTPUTS = {
    "4:3": [(640, 480), (800, 600), (1024, 768), (1280, 960), (1440, 1080), (1600, 1200), (2048, 1536), (2880, 2160)],
    "16:10": [(640, 400), (960, 600), (1280, 800), (1440, 900), (1680, 1050), (1920, 1200), (2560, 1600), (3840, 2400)],
    "16:9": [(640, 360), (854, 480), (960, 540), (1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (3200, 1800), (3840, 2160)],
    "21:9": [(860, 360), (1280, 540), (1720, 720), (2560, 1080), (3440, 1440), (3840, 1600), (5120, 2160)],
    "32:9": [(1280, 360), (1920, 540), (2560, 720), (3840, 1080), (5120, 1440), (7680, 2160)],
}

FPS_PRESET_CHOICES = [
    ("Unlimited", 0),
    ("30 FPS (PS4)", 30),
    ("60 FPS", 60),
    ("90 FPS", 90),
    ("120 FPS", 120),
    ("144 FPS", 144),
    ("165 FPS", 165),
    ("170 FPS", 170),
    ("240 FPS", 240),
    ("360 FPS", 360),
    ("Custom", -1),
]

PRESENT_MODES = [
    ("FastSync / Mailbox (Default)", "Mailbox",
     "Tear-free without locking FPS to monitor refresh rate — recommended"),
    ("VSync On (FIFO)", "Fifo",
     "Classic vertical sync: locks FPS to the monitor refresh rate"),
    ("Adaptive VSync (FIFO Relaxed)", "FifoRelaxed",
     "Syncs at full monitor refresh rate, presents immediately on frame drops"),
    ("VSync Off (Immediate)", "Immediate",
     "Presents frames immediately with lowest latency; screen tearing may occur"),
]

ANTI_ALIASING = [
    ("FXAA (Subpixel AA)", "fxaa"),
    ("SMAA 1x (Morphological Ultra)", "smaa"),
    ("SMAA T2x (Morphological + Temporal 2x)", "smaa_t2x"),
    ("TAA (Temporal AA)", "taa"),
    ("DLAA 4 (Native DLSS 100% Transformer)", "dlaa"),
    ("DLAA 3 (Native DLSS 100% CNN Fast)", "dlaa3"),
    ("Off", "off"),
]

UPSCALERS = [
    ("Off (Native 100%)", "off"),
    ("DLSS (NVIDIA RTX)", "dlss"),
    ("FSR 3.1", "fsr3"),
    ("FSR 4", "fsr4"),
    ("FSR 4.1.1", "fsr411"),
]

PRESETS = [
    ("Native 100% / DLAA", "0"),
    ("Quality (67% Scale)", "1"),
    ("Balanced (58% Scale)", "2"),
    ("Performance (50% Scale)", "3"),
    ("Ultra Performance (33% Scale)", "4"),
]

DLSS_PRESETS = [
    ("DLSS 4.5 (Preset M — 2nd Gen Transformer)", "13"),
    ("DLSS 4.5 (Preset L — 2nd Gen Transformer)", "12"),
    ("DLSS 4 (Preset K — Transformer)", "11"),
    ("DLSS (Auto / Default Preset)", "0"),
]

ANISOTROPIC_LEVELS = [
    ("16x", "16"),
    ("8x", "8"),
    ("4x", "4"),
    ("2x", "2"),
    ("Off (1x)", "1"),
]

MODEL_LOD = [
    ("Game default (0)", "0"),
    ("Highest (-2)", "-2"),
    ("Lower (1)", "1"),
    ("Lowest (2)", "2"),
]

DRAW_PIPE = [
    ("Auto (Enabled on 8+ CPU threads)", ""),
    ("Enabled", "1"),
    ("Disabled (Safer, slower)", "0"),
]

LANGUAGES = [
    ("English (US)", "1"),
    ("English (UK)", "18"),
    ("Russian", "8"),
    ("Japanese", "0"),
    ("French", "2"),
    ("Spanish (Spain)", "3"),
    ("Spanish (Latin America)", "20"),
    ("German", "4"),
    ("Italian", "5"),
    ("Dutch", "6"),
    ("Portuguese (Portugal)", "7"),
    ("Portuguese (Brazil)", "17"),
    ("Polish", "16"),
    ("Turkish", "19"),
    ("Korean", "9"),
    ("Chinese (Simplified)", "11"),
    ("Chinese (Traditional)", "10"),
    ("Finnish", "12"),
    ("Swedish", "13"),
    ("Danish", "14"),
    ("Norwegian", "15"),
]

EFFECTS = [
    ("effect_chromatic_aberration", "Chromatic aberration", True,
     "Red-blue color fringing near screen edges (disabling makes the image noticeably cleaner)"),
    ("effect_dof", "Depth of field (DoF)", True,
     "Cinematic blur on distant backgrounds and out-of-focus objects"),
    ("effect_motion_blur", "Motion blur", True,
     "Camera rotation and character dodge-roll motion blur"),
    ("effect_ssao", "SSAO ambient occlusion", True,
     "Soft contact shading in corners, cobblestone crevices, and cloth folds"),
    ("effect_dynamic_shadows", "Dynamic light shadows", True,
     "Shadows cast by lanterns, torches, bonfires, and firearm flashes"),
    ("effect_ssr", "Screen-space reflections (SSR, live in-game)", False,
     "Hidden PS4 screen-space raycast reflections on puddles, blood, and wet stone"),
    ("effect_bloom", "Bloom", True,
     "Soft light glow around lanterns, fire, moonlight, and arcane effects"),
    ("skip_intro", "Skip startup intro logos", True,
     "Jump straight to the title screen when launching the game"),
    ("debug_camera", "Free camera (Cross + L3 / F + Ctrl)", False,
     "Detaches the camera from the player character for free fly-through"),
    ("debug_menu", "Debug menu (left touchpad / Tab; requires fonts)", False,
     "Requires the adhoc/font folder from Nexus mod #253 in dvdroot_ps4 or as a mod"),
]

REMASTER_ENHANCEMENTS = [
    ("ssao_quality", "Enhanced dual-scale GTAO / noise-free SSAO", True,
     "Replaces PS4 64x64 dither grid with Interleaved Gradient Noise and adds micro-crevice shading"),
    ("contact_shadows", "Screen-space contact shadows & 12-Tap Vogel PCF", True,
     "Upgrades shadow filtering from PS4 4-tap to 12-tap Vogel disk and traces screen-space micro-shadows"),
    ("ssr_quality", "High-precision Ultra SSR (2x ray density & 30 Hz Cubemap)", True,
     "Doubles SSR raymarch density (1.5 px/step vs 3.0) on wet cobblestone and puddles"),
    ("true_black_hdr", "True Black / OLED anti-haze & color richness", True,
     "Removes the 2015 LCD raised-black gray veil after 3D LUT with a soft S-curve and rich blood/fire tones"),
]

CONTROLS = [
    ("cross", "Cross (Interact / Confirm)", "F, Enter", "a"),
    ("circle", "Circle (Dodge / Sprint)", "", "b"),
    ("dodge", "Dodge (Short Circle press; Shift+Space jumps)", "Space", None),
    ("sprint", "Sprint (Hold Circle)", "Left Shift", None),
    ("walk", "Slow Walk (Hold)", "Left Alt", None),
    ("square", "Square (Use Quick Item)", "R, Tab", "x"),
    ("triangle", "Triangle (Blood Vial)", "Q", "y"),
    ("l1", "L1 (Transform Weapon)", "E, Mouse 5", "leftshoulder"),
    ("r1", "R1 (Light Attack)", "Mouse Left", "rightshoulder"),
    ("l2", "L2 (Firearm / Parry)", "Mouse Right", "lefttrigger"),
    ("r2", "R2 (Heavy Attack)", "Left Shift+Mouse Left, Mouse 4", "righttrigger"),
    ("l3", "L3 (Jump / Sort)", "Left Ctrl", "leftstick"),
    ("r3", "R3 (Lock-On / Reset Camera)", "C, Mouse Middle", "rightstick"),
    ("options", "Options (Main Menu; Esc goes back in menus)", "Escape", "start"),
    ("touchpad", "Touchpad Left (Gestures)", "G", "back, touchpad"),
    ("touchpad_right", "Touchpad Right (Personal Effects)", "Backspace", ""),
    ("up", "D-Pad Up (Blood Bullets / Wheel Up)", "Wheel Up, 3", "dpup"),
    ("down", "D-Pad Down (Cycle Quick Item / Wheel Down)", "Wheel Down, 4", "dpdown"),
    ("left", "D-Pad Left (Left-Hand Weapon)", "1", "dpleft"),
    ("right", "D-Pad Right (Right-Hand Weapon)", "2", "dpright"),
    ("move_up", "Move Forward", "W", None),
    ("move_down", "Move Back", "S", None),
    ("move_left", "Move Left", "A", None),
    ("move_right", "Move Right", "D", None),
    ("look_up", "Camera Up", "I", None),
    ("look_down", "Camera Down", "K", None),
    ("look_left", "Camera Left", "J", None),
    ("look_right", "Camera Right", "L", None),
]


def discover_game_dir():
    env_dir = os.environ.get("BB_GAME_DIR", "")
    if env_dir and (Path(env_dir) / "eboot.bin").is_file():
        return str(Path(env_dir).resolve())
    candidates = [
        PORT_DIR / "CUSA03173",
        PORT_DIR / "game",
        PORT_DIR.parent / "CUSA03173",
        PORT_DIR.parent / "game",
        PORT_DIR.parent / "Bloodborne" / "CUSA03173",
        PORT_DIR.parent / "Bloodborne" / "game",
    ]
    for c in candidates:
        if (c / "eboot.bin").is_file():
            return str(c.resolve())
    for parent in (PORT_DIR, PORT_DIR.parent):
        try:
            for sub in parent.iterdir():
                if sub.is_dir():
                    if (sub / "eboot.bin").is_file():
                        return str(sub.resolve())
                    for sub2 in ("CUSA03173", "game", "dvdroot_ps4"):
                        p2 = sub / sub2
                        if (p2 / "eboot.bin").is_file():
                            return str(p2.resolve())
        except OSError:
            pass
    return ""


DEFAULTS = {
    "ui_language": "en",
    "game_dir": discover_game_dir(),
    "user_dir": "",
    "mods_dir": "",
    "mods_enabled": True,
    "patches_dir": "",
    "language": "1",
    "fullscreen": False,
    "hdr": False,
    "present_mode": "Mailbox",
    "gamepad": "",
    "gamepad_name": "",
    "display": "",
    "gpu": "",
    "skip_network_choice": True,
    "fps_mode": "uncap",
    "fps_limit": 0,
    "draw_pipe": "",
    "hw_occlusion_cull": True,
    "native_shaders": True,
    "mangohud": False,
    "save_log": True,
    "gpu_profile": False,
    "vk_validation": False,
    "extra_env": "",
    "online": False,
    "online_bbsc": True,
    "online_community": "http://play.seamlesshunt.com:20443",
    "online_server": "play.seamlesshunt.com:31313",
    "online_webapi": "",
    "online_npid": "",
    "online_password": "",
    "online_upnp": True,
    "online_pool": "seamless",
    "thd_server": "",
    "thd_online_id": "",
    "thd_token": "",
}

INI_DEFAULTS = {
    "aa_mode": "smaa_t2x",
    "upscaler": "off",
    "preset": "0",
    "dlss_preset": "11",
    "anisotropic": "16",
    "ui_upscale": "1",
    "fps_limit": "0",
    "sharpen": "1",
    "sharpness": "0.50",
    "object_motion": "0",
    "show_fps": "1",
    "aspect_ratio": "auto",
    "output_res": "2560x1440",
    "model_lod": "0",
    "live_resolution": "0",
    "effect_game_aa": "0",
    "mouse_look": "1",
    "mouse_sensitivity": "1.00",
    "mouse_invert_y": "0",
    **{key: "1" if default else "0" for key, _, default, *_ in EFFECTS + REMASTER_ENHANCEMENTS},
}


def load_settings():
    s = dict(DEFAULTS)
    for path in (WIN_SETTINGS_FILE, LINUX_SETTINGS_FILE):
        if path.is_file():
            with contextlib.suppress(OSError, ValueError):
                s.update(json.loads(path.read_text(encoding="utf-8")))
                break
    if not s.get("game_dir") or not (Path(s["game_dir"]) / "eboot.bin").is_file():
        found = discover_game_dir()
        if found:
            s["game_dir"] = found
    return s


def save_settings(s):
    with contextlib.suppress(OSError):
        WIN_SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            LINUX_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            LINUX_SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ini_mtime_ns():
    try:
        return INI_FILE.stat().st_mtime_ns
    except OSError:
        return 0


def load_ini():
    values = dict(INI_DEFAULTS)
    lines = []
    try:
        lines = INI_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        pass
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values, lines


def save_ini(values, lines):
    written = set()
    out = []
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in values:
                if values[key] is not None:
                    out.append(f"{key}={values[key]}")
                written.add(key)
                continue
        out.append(line)
    if not lines:
        out.append("# bbport settings (in-game menu: Insert / L3+R3)")
    for key, value in values.items():
        if key not in written and value is not None:
            out.append(f"{key}={value}")
    with contextlib.suppress(OSError):
        INI_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_size(text, default=(2560, 1440)):
    if not text or text == "auto":
        return default
    try:
        w, h = (int(v) for v in text.lower().replace("×", "x").split("x"))
        return (w, h) if w > 0 and h > 0 else default
    except ValueError:
        return default


def resolved_aspect(aspect, monitor):
    if aspect in OUTPUTS:
        return aspect
    if not monitor:
        return "16:9"
    ratio = monitor[0] / monitor[1]
    return min(OUTPUTS, key=lambda a: abs(math.log(ratio / ASPECT_RATIOS[a])))


def output_sizes(aspect, monitor):
    sizes = list(OUTPUTS[resolved_aspect(aspect, monitor)])
    if (aspect == "auto" and monitor and monitor not in sizes
            and 640 <= monitor[0] <= 7680 and 360 <= monitor[1] <= 4320):
        sizes = sorted(sizes + [monitor], key=lambda s: (s[1], s[0]))
    return sizes


def cap_tool_path():
    exe = "bb-gpu-capabilities.exe" if sys.platform == "win32" else "bb-gpu-capabilities"
    for d in ("bin", "out"):
        p = PORT_DIR / d / exe
        if p.is_file():
            return p
    return PORT_DIR / "bin" / exe


def connected_gpus():
    tool = cap_tool_path()
    env = {k: v for k, v in os.environ.items() if k != "BB_GPU_ID"}
    try:
        res = subprocess.run([str(tool), "--gpus"], capture_output=True, text=True,
                             timeout=5, env=env, creationflags=NO_WINDOW)
        rows = [line.split("\t") for line in res.stdout.splitlines() if line.count("\t") == 4]
        return [(int(idx), vendor, kind, name, best == "1")
                for idx, vendor, kind, name, best in rows if idx.isdigit()]
    except Exception:
        return []


def connected_displays():
    tool = cap_tool_path()
    try:
        res = subprocess.run([str(tool), "--displays"], capture_output=True, text=True,
                             timeout=5, creationflags=NO_WINDOW)
        rows = [line.split("\t") for line in res.stdout.splitlines() if line.count("\t") == 2]
        names = [name for name, _, _ in rows]
        return [
            (name if names.count(name) == 1 else str(num),
             f"{num}: {name} ({size})" + (", primary" if pri == "1" else ""),
             parse_size(size, None),
             pri == "1")
            for num, (name, size, pri) in enumerate(rows, 1)
        ]
    except Exception:
        return []


def connected_gamepads():
    tool = cap_tool_path()
    try:
        res = subprocess.run([str(tool), "--gamepads"], capture_output=True, text=True,
                             timeout=5, creationflags=NO_WINDOW)
        return [tuple(line.split("\t", 1)) for line in res.stdout.splitlines() if "\t" in line]
    except Exception:
        return []


# --- Adwaita Dark UI Widgets (Mirroring Screenshot 2) --------------------------------

BG_WINDOW = "#1e1e1e"
BG_HEADER = "#242424"
BG_SIDEBAR = "#242424"
BG_SIDEBAR_SEL = "#383838"
BG_SIDEBAR_HOV = "#2e2e2e"
BG_CARD = "#2b2b2b"
BORDER_CARD = "#3a3a3a"
FG_TITLE = "#ffffff"
FG_SUB = "#9a9a9a"
ACCENT_BLUE = "#3584e4"
ACCENT_BLUE_HOV = "#4a94ec"
ACCENT_RED = "#e01b24"
SWITCH_OFF = "#4a4a4a"
BTN_BG = "#383838"
BTN_HOV = "#464646"


class AdwSwitch(tk.Canvas):
    """Smooth Libadwaita pill toggle switch (blue when active, dark gray when inactive)."""
    def __init__(self, parent, active=False, on_change=None):
        super().__init__(parent, width=46, height=26, bg=BG_CARD, highlightthickness=0, bd=0, cursor="hand2")
        self._active = bool(active)
        self._enabled = True
        self._on_change = on_change
        self.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self):
        self.delete("all")
        fill = ACCENT_BLUE if self._active else SWITCH_OFF
        if not self._enabled:
            fill = "#333333"
        # Pill background (rounded rectangle via two circles + rect)
        r = 12
        self.create_oval(2, 1, 2 + 2 * r, 1 + 2 * r, fill=fill, outline=fill)
        self.create_oval(44 - 2 * r, 1, 44, 1 + 2 * r, fill=fill, outline=fill)
        self.create_rectangle(2 + r, 1, 44 - r, 1 + 2 * r, fill=fill, outline=fill)
        # Thumb circle
        knob_x = 32 if self._active else 14
        knob_color = "#ffffff" if self._enabled else "#888888"
        self.create_oval(knob_x - 9, 4, knob_x + 9, 22, fill=knob_color, outline=knob_color)

    def _toggle(self, _event=None):
        if not self._enabled:
            return
        self._active = not self._active
        self._draw()
        if self._on_change:
            self._on_change(self._active)

    def get_active(self):
        return self._active

    def set_active(self, val, notify=False):
        val = bool(val)
        if self._active != val:
            self._active = val
            self._draw()
            if notify and self._on_change:
                self._on_change(self._active)

    def set_sensitive(self, enabled):
        self._enabled = bool(enabled)
        self.configure(cursor="hand2" if self._enabled else "arrow")
        self._draw()


class AdwPreferencesGroup(tk.Frame):
    """Adwaita PreferencesGroup: bold header title + gray description + rounded card container."""
    def __init__(self, parent, title="", description=""):
        super().__init__(parent, bg=BG_WINDOW)
        self.pack(fill=tk.X, pady=(0, 22))
        if title:
            lbl = tk.Label(self, text=title, bg=BG_WINDOW, fg=FG_TITLE,
                           font=("Segoe UI", 11, "bold"), anchor="w")
            lbl.pack(fill=tk.X, pady=(0, 2 if description else 8))
        if description:
            desc = tk.Label(self, text=description, bg=BG_WINDOW, fg=FG_SUB,
                            font=("Segoe UI", 9), anchor="w", justify="left", wraplength=680)
            desc.pack(fill=tk.X, pady=(0, 8))
        self.card_border = tk.Frame(self, bg=BORDER_CARD, padx=1, pady=1)
        self.card_border.pack(fill=tk.X)
        self.card = tk.Frame(self.card_border, bg=BG_CARD)
        self.card.pack(fill=tk.X)
        self._row_count = 0

    def add_row_frame(self):
        if self._row_count > 0:
            sep = tk.Frame(self.card, bg=BORDER_CARD, height=1)
            sep.pack(fill=tk.X)
        row = tk.Frame(self.card, bg=BG_CARD, padx=16, pady=12)
        row.pack(fill=tk.X)
        self._row_count += 1
        return row


def make_flat_btn(parent, text, command, width=None):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=BTN_BG, fg=FG_TITLE, activebackground=BTN_HOV, activeforeground=FG_TITLE,
        relief="flat", bd=0, padx=10, pady=4, font=("Segoe UI", 9), cursor="hand2"
    )
    if width:
        btn.configure(width=width)
    return btn


class AdwRowBase:
    def __init__(self, group, title, subtitle=""):
        self.frame = group.add_row_frame()
        self.left = tk.Frame(self.frame, bg=BG_CARD)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))
        self.title_lbl = tk.Label(
            self.left, text=title, bg=BG_CARD, fg=FG_TITLE,
            font=("Segoe UI", 10), anchor="w", justify="left"
        )
        self.title_lbl.pack(fill=tk.X)
        self.sub_lbl = tk.Label(
            self.left, text=subtitle or "", bg=BG_CARD, fg=FG_SUB,
            font=("Segoe UI", 8), anchor="w", justify="left", wraplength=470
        )
        if subtitle:
            self.sub_lbl.pack(fill=tk.X, pady=(2, 0))
        self.right = tk.Frame(self.frame, bg=BG_CARD)
        self.right.pack(side=tk.RIGHT)

    def set_title(self, text):
        self.title_lbl.configure(text=text)

    def set_subtitle(self, text):
        self.sub_lbl.configure(text=text or "")
        if text:
            if not self.sub_lbl.winfo_ismapped():
                self.sub_lbl.pack(fill=tk.X, pady=(2, 0))
        else:
            self.sub_lbl.pack_forget()


class AdwSwitchRow(AdwRowBase):
    def __init__(self, group, title, subtitle="", active=False, on_change=None):
        super().__init__(group, title, subtitle)
        self.switch = AdwSwitch(self.right, active=active, on_change=on_change)
        self.switch.pack(side=tk.RIGHT)

    def get_active(self):
        return self.switch.get_active()

    def set_active(self, val, notify=False):
        self.switch.set_active(val, notify=notify)

    def set_sensitive(self, enabled):
        self.switch.set_sensitive(enabled)


class AdwComboRow(AdwRowBase):
    def __init__(self, group, title, subtitle="", choices=(), current=None, on_change=None, width=26):
        super().__init__(group, title, subtitle)
        self._base_subtitle = subtitle or ""
        self._on_change = on_change
        self.values = []
        self.labels = []
        self.notes = []
        self.var = tk.StringVar()
        self.combo = ttk.Combobox(
            self.right, textvariable=self.var, state="readonly", width=width, font=("Segoe UI", 9)
        )
        self.combo.pack(side=tk.LEFT, padx=(4, 0))
        self.combo.bind("<<ComboboxSelected>>", self._selected)
        if choices:
            self.set_choices(choices, current)

    def add_suffix_btn(self, text, command):
        btn = make_flat_btn(self.right, text, command)
        btn.pack(side=tk.LEFT, padx=(6, 0))
        return btn

    def set_choices(self, choices, current=None):
        self.labels = [c[0] for c in choices]
        self.values = [c[1] for c in choices]
        self.notes = [c[2] if len(c) > 2 else None for c in choices]
        self.combo["values"] = self.labels
        idx = self.values.index(current) if current in self.values else 0
        if self.labels:
            self.combo.current(idx)
            self._update_note(idx)

    def _update_note(self, idx):
        note = self.notes[idx] if 0 <= idx < len(self.notes) else None
        parts = [p for p in (self._base_subtitle, note) if p]
        self.set_subtitle("\n".join(parts))

    def _selected(self, _event=None):
        idx = self.combo.current()
        self._update_note(idx)
        if self._on_change:
            self._on_change(self.get_value())

    def get_selected(self):
        return max(0, self.combo.current())

    def set_selected(self, idx):
        if 0 <= idx < len(self.labels):
            self.combo.current(idx)
            self._update_note(idx)

    def get_value(self):
        idx = self.combo.current()
        return self.values[idx] if 0 <= idx < len(self.values) else (self.values[0] if self.values else "")

    def set_value(self, val):
        if val in self.values:
            self.set_selected(self.values.index(val))

    def set_sensitive(self, enabled):
        self.combo.configure(state="readonly" if enabled else "disabled")


class AdwSpinRow(AdwRowBase):
    """Adwaita SpinRow with value display and (-) (+) buttons just like Screenshot 2."""
    def __init__(self, group, title, subtitle="", min_v=0.0, max_v=5000.0, step=1.0, digits=0, value=0.0, on_change=None):
        super().__init__(group, title, subtitle)
        self.min_v = min_v
        self.max_v = max_v
        self.step = step
        self.digits = digits
        self._val = float(value)
        self._on_change = on_change
        self._enabled = True

        self.var = tk.StringVar(value=self._fmt(self._val))
        self.entry = tk.Entry(
            self.right, textvariable=self.var, width=6, bg="#222222", fg=FG_TITLE,
            insertbackground=FG_TITLE, relief="flat", justify="center", font=("Segoe UI", 10)
        )
        self.entry.pack(side=tk.LEFT, padx=(0, 6), ipady=3)
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        self.minus_btn = make_flat_btn(self.right, "−", lambda: self._step_by(-1), width=2)
        self.minus_btn.pack(side=tk.LEFT, padx=2)
        self.plus_btn = make_flat_btn(self.right, "+", lambda: self._step_by(1), width=2)
        self.plus_btn.pack(side=tk.LEFT, padx=2)

    def _fmt(self, v):
        return f"{v:.{self.digits}f}" if self.digits > 0 else f"{int(round(v))}"

    def _step_by(self, d):
        if not self._enabled:
            return
        self.set_value(self._val + d * self.step, notify=True)

    def _on_entry(self, _event=None):
        try:
            v = float(self.var.get().strip())
            self.set_value(v, notify=True)
        except ValueError:
            self.var.set(self._fmt(self._val))

    def get_value(self):
        try:
            v = float(self.var.get().strip())
            return max(self.min_v, min(self.max_v, v))
        except ValueError:
            return self._val

    def set_value(self, val, notify=False):
        self._val = max(self.min_v, min(self.max_v, float(val)))
        self.var.set(self._fmt(self._val))
        if notify and self._on_change:
            self._on_change(self._val)

    def set_sensitive(self, enabled):
        self._enabled = bool(enabled)
        self.entry.configure(state="normal" if enabled else "disabled")


class AdwEntryRow(AdwRowBase):
    def __init__(self, group, title, subtitle="", text="", password=False, on_change=None, width=30):
        super().__init__(group, title, subtitle)
        self.var = tk.StringVar(value=text)
        self.entry = tk.Entry(
            self.right, textvariable=self.var, width=width, bg="#222222", fg=FG_TITLE,
            insertbackground=FG_TITLE, relief="flat", font=("Segoe UI", 9),
            show="•" if password else ""
        )
        self.entry.pack(side=tk.LEFT, ipady=4, padx=4)
        if on_change:
            self.entry.bind("<FocusOut>", lambda _e: on_change(self.get_text()))
            self.entry.bind("<Return>", lambda _e: on_change(self.get_text()))

    def get_text(self):
        return self.var.get()

    def set_text(self, val):
        self.var.set(val)


class AdwLauncherWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Bloodborne PC Vulkan")
        self.root.geometry("1040x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG_WINDOW)

        # Configure dark Combobox / Scrollbar styling
        style = ttk.Style(self.root)
        with contextlib.suppress(Exception):
            style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground="#363636",
            background="#363636",
            foreground=FG_TITLE,
            arrowcolor=FG_TITLE,
            borderwidth=0,
            padding=5,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#363636"), ("disabled", "#262626")],
            foreground=[("readonly", FG_TITLE), ("disabled", "#777777")],
            selectbackground=[("readonly", "#363636")],
            selectforeground=[("readonly", FG_TITLE)],
        )
        self.root.option_add("*TCombobox*Listbox.background", "#2b2b2b")
        self.root.option_add("*TCombobox*Listbox.foreground", FG_TITLE)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_BLUE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", FG_TITLE)

        self.settings = load_settings()
        self.ini, self.ini_lines = load_ini()
        if not INI_FILE.is_file():
            save_ini(self.ini, self.ini_lines)
            self.ini, self.ini_lines = load_ini()
        self._ini_mtime_ns = ini_mtime_ns()
        self._reloading_ini = False
        self._syncing_fps = False
        self.process = None
        self._stopping_manually = False

        self.effect_rows = {}
        self.control_rows = {}
        self.sidebar_items = {}
        self.page_frames = {}
        self.page_titles = {}
        self.current_page = "display"

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(1000, self.check_ini_reload)

    def _build_ui(self):
        # 1. Top HeaderBar (matching Screenshot 2)
        header = tk.Frame(self.root, bg=BG_HEADER, height=52)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg=BG_HEADER)
        title_box.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            title_box, text="Bloodborne PC Vulkan", bg=BG_HEADER, fg=FG_TITLE,
            font=("Segoe UI", 10, "bold")
        ).pack()
        self.header_sub_lbl = tk.Label(
            title_box, text="Display and FPS", bg=BG_HEADER, fg=FG_SUB,
            font=("Segoe UI", 8)
        )
        self.header_sub_lbl.pack()

        self.play_btn = tk.Button(
            header, text="   Play   ", command=self.on_launch,
            bg=ACCENT_BLUE, fg=FG_TITLE, activebackground=ACCENT_BLUE_HOV, activeforeground=FG_TITLE,
            relief="flat", bd=0, padx=16, pady=5, font=("Segoe UI", 10, "bold"), cursor="hand2"
        )
        self.play_btn.pack(side=tk.RIGHT, padx=16, pady=9)

        tk.Frame(self.root, bg="#363636", height=1).pack(fill=tk.X, side=tk.TOP)

        # 2. Main Split Body: Left Sidebar + Right Scrollable Stack
        body = tk.Frame(self.root, bg=BG_WINDOW)
        body.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(body, bg=BG_SIDEBAR, width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Frame(body, bg="#363636", width=1).pack(side=tk.LEFT, fill=tk.Y)

        # Right area with Scrollable Canvas for settings pages + non-scrolled Log page
        self.right_host = tk.Frame(body, bg=BG_WINDOW)
        self.right_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_host = tk.Frame(self.right_host, bg=BG_WINDOW)
        self.scroll_host.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.scroll_host, bg=BG_WINDOW, highlightthickness=0, bd=0)
        self.vscroll = tk.Scrollbar(self.scroll_host, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content_wrap = tk.Frame(self.canvas, bg=BG_WINDOW)
        self.canvas_win = self.canvas.create_window((0, 0), window=self.content_wrap, anchor="nw")

        self.content_wrap.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda _e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda _e: self.canvas.yview_scroll(1, "units"))

        pages_spec = [
            ("display", "Display and FPS", "🖥", self.build_display_page),
            ("upscaler", "Upscaler and Anti-Aliasing", "◎", self.build_upscaler_page),
            ("effects", "Graphics Effects", "☾", self.build_effects_page),
            ("engine", "Engine and Debug", "⚙", self.build_engine_page),
            ("controls", "Controls", "🎮", self.build_controls_page),
            ("game", "Game and Mods", "📁", self.build_game_page),
            ("online", "Online", "🌐", self.build_online_page),
            ("log", "Log", ">_", self.build_log_page),
        ]

        tk.Frame(self.sidebar, bg=BG_SIDEBAR, height=8).pack(fill=tk.X)
        for key, title, icon, builder in pages_spec:
            self.page_titles[key] = title
            item = tk.Frame(self.sidebar, bg=BG_SIDEBAR, padx=12, pady=9, cursor="hand2")
            item.pack(fill=tk.X, padx=8, pady=2)
            ic_lbl = tk.Label(item, text=icon, bg=BG_SIDEBAR, fg=FG_TITLE, font=("Segoe UI", 10), width=3, anchor="w")
            ic_lbl.pack(side=tk.LEFT)
            tx_lbl = tk.Label(item, text=title, bg=BG_SIDEBAR, fg=FG_TITLE, font=("Segoe UI", 10), anchor="w")
            tx_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            for w in (item, ic_lbl, tx_lbl):
                w.bind("<Button-1>", lambda _e, k=key: self.switch_page(k))
            self.sidebar_items[key] = (item, ic_lbl, tx_lbl)

            if key == "log":
                page_frame = tk.Frame(self.right_host, bg=BG_WINDOW)
            else:
                page_frame = tk.Frame(self.content_wrap, bg=BG_WINDOW, padx=48, pady=28)
            self.page_frames[key] = page_frame
            builder(page_frame)

        self.switch_page("display")
        self.update_launch_button()
        self.update_game_status()
        self.update_user_status()
        self.update_upscaler_status()

    def _on_canvas_resize(self, event):
        w = max(560, event.width)
        self.canvas.itemconfigure(self.canvas_win, width=w)

    def _on_mousewheel(self, event):
        if self.current_page == "log":
            return
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta, "units")

    def switch_page(self, name):
        self.current_page = name
        self.header_sub_lbl.configure(text=self.page_titles.get(name, ""))
        for k, (item, ic_lbl, tx_lbl) in self.sidebar_items.items():
            bg = BG_SIDEBAR_SEL if k == name else BG_SIDEBAR
            item.configure(bg=bg)
            ic_lbl.configure(bg=bg)
            tx_lbl.configure(bg=bg, font=("Segoe UI", 10, "bold" if k == name else "normal"))

        for k, frame in self.page_frames.items():
            frame.pack_forget()

        if name == "log":
            self.scroll_host.pack_forget()
            self.page_frames["log"].pack(fill=tk.BOTH, expand=True)
        else:
            self.page_frames["log"].pack_forget()
            if not self.scroll_host.winfo_ismapped():
                self.scroll_host.pack(fill=tk.BOTH, expand=True)
            self.page_frames[name].pack(fill=tk.BOTH, expand=True)
            self.canvas.yview_moveto(0.0)
            self.content_wrap.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # --- Page 1: Display and FPS --------------------------------------------------------

    def build_display_page(self, page):
        screen = AdwPreferencesGroup(
            page,
            title="Display and GPU",
            description="Output resolution, aspect ratio, monitor, and GPU selection",
        )
        self.gpu_row = AdwComboRow(
            screen,
            title="GPU",
            subtitle="On a laptop with two GPUs, pick the discrete one; applies when the game starts",
            on_change=lambda _v: self.store(),
        )
        self.gpu_row.add_suffix_btn("↻", self.fill_gpus)
        self.fill_gpus()

        self.display_row = AdwComboRow(
            screen, title="Monitor", on_change=lambda _v: self.on_display_changed()
        )
        self.display_row.add_suffix_btn("↻", self.fill_displays)
        self.fill_displays()

        self.output_row = AdwComboRow(
            screen,
            title="Output resolution",
            subtitle="Display, UI, and render resolution (G-Buffer scales live in-game; Yebis2 post-FX rebuilds on startup)",
            on_change=lambda _v: (self.update_upscaler_status(), self.store()),
        )
        self.aspect_row = AdwComboRow(
            screen,
            title="Aspect ratio",
            subtitle="Wider than 16:9 sees more to the sides, taller more above and below; the UI keeps to the screen edges",
            on_change=lambda _v: (self.fill_outputs(parse_size(self.output_row.get_value())), self.store()),
        )
        aspect = self.ini.get("aspect_ratio", "auto")
        self.fill_aspects(aspect if aspect in ASPECTS else "auto")
        self.fill_outputs(parse_size(self.ini.get("output_res", "2560x1440")))

        self.fullscreen_row = AdwSwitchRow(
            screen, title="Fullscreen",
            active=self.ini.get("fullscreen", "1" if self.settings["fullscreen"] else "0") == "1",
            on_change=lambda _v: self.store(),
        )
        self.hdr_row = AdwSwitchRow(
            screen, title="Allow HDR", active=self.settings["hdr"], on_change=lambda _v: self.store()
        )

        frames = AdwPreferencesGroup(
            page,
            title="Frame Rate and VSync",
            description="FPS cap and vertical sync; preset and exact cap switch live in-game",
        )
        ini_fps_limit = int(self.ini.get("fps_limit", "0") or 0)
        cur_fps = ini_fps_limit if ini_fps_limit > 0 else int(self.settings.get("fps_limit", 0) or 0)
        preset_vals = [v for _, v in FPS_PRESET_CHOICES if v >= 0]

        self.fps_row = AdwComboRow(
            frames,
            title="Frame rate preset",
            subtitle="Switches live in-game (System → Display or F8); FixedBlockPool ceiling unlocked",
            choices=FPS_PRESET_CHOICES,
            current=cur_fps if cur_fps in preset_vals else -1,
            on_change=self._on_fps_preset,
        )
        self.limit_row = AdwSpinRow(
            frames,
            title="Exact FPS cap",
            subtitle="0: true unlimited (0.0 ms frame pacer target); stored in bbport.ini",
            min_v=0, max_v=5000, step=1, digits=0, value=cur_fps,
            on_change=self._on_fps_spin,
        )
        self.present_row = AdwComboRow(
            frames,
            title="Present mode",
            choices=PRESENT_MODES,
            current=self.ini.get("present_mode") or self.settings["present_mode"],
            on_change=lambda _v: self.store(),
        )

        telemetry = AdwPreferencesGroup(page, title="Telemetry Overlay")
        self.show_fps_row = AdwSwitchRow(
            telemetry,
            title="Telemetry and Debug Overlay (FPS / DirectVk Stats)",
            subtitle="Shows FPS, 10-pass GPU & CPU pipeline timings, Draw Calls, and VRAM telemetry (F8 / F10)",
            active=self.ini.get("show_fps", "1") == "1",
            on_change=lambda _v: self.store(),
        )

    def _on_fps_preset(self, val):
        if self._syncing_fps:
            return
        if val >= 0:
            self._syncing_fps = True
            self.limit_row.set_value(val, notify=False)
            self._syncing_fps = False
        self.store()

    def _on_fps_spin(self, val):
        if self._syncing_fps:
            return
        ival = int(round(val))
        preset_vals = [v for _, v in FPS_PRESET_CHOICES if v >= 0]
        target = ival if ival in preset_vals else -1
        self._syncing_fps = True
        self.fps_row.set_value(target)
        self._syncing_fps = False
        self.store()

    # --- Page 2: Upscaler and Anti-Aliasing ---------------------------------------------

    def build_upscaler_page(self, page):
        upscaler = AdwPreferencesGroup(
            page,
            title="Anti-Aliasing and Upscaler",
            description="All settings below switch live in-game via System → Display or the F8 overlay",
        )
        raw_up = self.ini.get("upscaler", "off")
        raw_preset = str(self.ini.get("preset", "0") or "0")
        raw_aa = self.ini.get("aa_mode", "smaa_t2x")
        init_aa = raw_aa if raw_aa in ("off", "fxaa", "smaa", "smaa_t2x", "taa", "dlaa", "dlaa3") else "smaa_t2x"
        init_up = raw_up if raw_up in ("off", "dlss", "fsr3", "fsr4", "fsr411") else "off"

        self.aa_row = AdwComboRow(
            upscaler,
            title="Anti-aliasing",
            subtitle="FXAA: subpixel, SMAA 1x/T2x: morphological + temporal, TAA: temporal, DLAA 4/3: native 100% DLSS (Transformer / CNN)",
            choices=ANTI_ALIASING,
            current=init_aa,
            on_change=lambda _v: (self.update_upscaler_status(), self.store()),
            width=34,
        )
        self.upscaler_row = AdwComboRow(
            upscaler,
            title="Upscaler",
            choices=UPSCALERS,
            current=init_up,
            on_change=lambda _v: (self.update_upscaler_status(), self.store()),
            width=34,
        )
        self.preset_row = AdwComboRow(
            upscaler,
            title="Upscaler quality mode",
            choices=PRESETS,
            current=raw_preset,
            on_change=lambda _v: (self.update_upscaler_status(), self.store()),
            width=34,
        )
        self.dlss_preset_row = AdwComboRow(
            upscaler,
            title="DLSS Model (Preset)",
            subtitle="Preset M/L: 2nd Gen Transformer (DLSS 4.5); Preset K: DLSS 4.0 Transformer",
            choices=DLSS_PRESETS,
            current=self.ini.get("dlss_preset", "11"),
            on_change=lambda _v: self.store(),
            width=34,
        )
        self.dlss_status_row = AdwRowBase(upscaler, title="DLSS: NVIDIA library")
        self.motion_row = AdwSwitchRow(
            upscaler,
            title="Object motion vectors",
            subtitle="Eliminates ghosting on characters and cloaks when an upscaler (DLSS / FSR / TAA) is active; costs ~10% FPS",
            active=self.ini.get("object_motion") == "1",
            on_change=lambda _v: self.store(),
        )

        clarity = AdwPreferencesGroup(
            page,
            title="Texture Clarity, UI Icons, and Sharpening",
            description="Anisotropic filtering for floors/walls, 2x/4x UI icon reconstruction, and RCAS sharpening",
        )
        self.aniso_row = AdwComboRow(
            clarity,
            title="Anisotropic filtering",
            subtitle="Floor and wall texture clarity at oblique angles (switches live in-game)",
            choices=ANISOTROPIC_LEVELS,
            current=self.ini.get("anisotropic", "16"),
            on_change=lambda _v: self.store(),
        )
        self.ui_upscale_row = AdwSwitchRow(
            clarity,
            title="UI and Icon Upscaling (AI / SDF 2x–4x)",
            subtitle="2x/4x GPU reconstruction of the Bloodborne logo, weapon/armor/item icons, and HUD (switches live in-game)",
            active=self.ini.get("ui_upscale", "1") != "0",
            on_change=lambda _v: self.store(),
        )
        self.sharpen_row = AdwSwitchRow(
            clarity,
            title="Sharpening (RCAS)",
            active=self.ini.get("sharpen", "1") == "1",
            on_change=lambda _v: self.store(),
        )
        self.sharpness_row = AdwSpinRow(
            clarity,
            title="Sharpening strength",
            min_v=0.0, max_v=2.0, step=0.05, digits=2,
            value=float(self.ini.get("sharpness", "0.50") or 0.50),
            on_change=lambda _v: self.store(),
        )

    # --- Page 3: Graphics Effects -------------------------------------------------------

    def build_effects_page(self, page):
        effects = AdwPreferencesGroup(
            page,
            title="Geometry and Visual Effects",
            description="All graphics effects run natively in C++20 / DirectVk and switch live in-game",
        )
        self.lod_row = AdwComboRow(
            effects,
            title="Model detail (LOD)",
            subtitle="Draw distance for full-detail geometry (Highest -2 disables distant low-poly LOD pop-in)",
            choices=MODEL_LOD,
            current=self.ini.get("model_lod", "0"),
            on_change=lambda _v: self.store(),
        )
        for key, title, default, subtitle in EFFECTS:
            if key in ("skip_intro", "debug_camera", "debug_menu"):
                continue
            row = AdwSwitchRow(
                effects,
                title=title,
                subtitle=subtitle,
                active=self.ini.get(key, "1" if default else "0") == "1",
                on_change=lambda _v: self.store(),
            )
            self.effect_rows[key] = row

        remaster = AdwPreferencesGroup(
            page,
            title="Remaster Graphics Enhancements (Vulkan 1.3)",
            description="Exclusive Vulkan 1.3 GLSL shader enhancements (switch live in-game, < 0.25 ms GPU cost)",
        )
        for key, title, default, subtitle in REMASTER_ENHANCEMENTS:
            row = AdwSwitchRow(
                remaster,
                title=title,
                subtitle=subtitle,
                active=self.ini.get(key, "1" if default else "0") == "1",
                on_change=lambda _v: self.store(),
            )
            self.effect_rows[key] = row

    # --- Page 4: Engine and Debug -------------------------------------------------------

    def build_engine_page(self, page):
        perf = AdwPreferencesGroup(
            page,
            title="Performance (C++20 Engine and DirectVk 1.3)",
            description="Decompiled C++20 graphics engine (2,940 / 2,940 functions) and DirectVk 1.3 backend with Split PC VRAM (0 page-faults/s)",
        )
        self.pipe_row = AdwComboRow(
            perf,
            title="Two-stage GPU draw pipeline",
            subtitle="20–30% faster on multi-core CPUs; disable if unstable",
            choices=DRAW_PIPE,
            current=self.settings.get("draw_pipe", ""),
            on_change=lambda _v: self.store(),
        )
        self.occlusion_cull_row = AdwSwitchRow(
            perf,
            title="Hardware 3D Occlusion Culler",
            subtitle="Culls wall-occluded meshes and micro-shadows on the GPU (−35..45% redundant Draw Calls)",
            active=self.settings.get("hw_occlusion_cull", True),
            on_change=lambda _v: self.store(),
        )
        self.native_shaders_row = AdwSwitchRow(
            perf,
            title="Native Vulkan 1.3 GLSL Shaders (109 shaders)",
            subtitle="Replaces translated PS4 GCN shaders with hand-crafted vectorized Vulkan 1.3 GLSL 460 shaders",
            active=self.settings.get("native_shaders", True),
            on_change=lambda _v: self.store(),
        )

        dev = AdwPreferencesGroup(
            page,
            title="Developer and Diagnostics",
            description="Logging, GPU profiling, in-game debug menu, and environment variables",
        )
        for key, title, default, subtitle in EFFECTS:
            if key in ("debug_camera", "debug_menu"):
                row = AdwSwitchRow(
                    dev,
                    title=title,
                    subtitle=subtitle,
                    active=self.ini.get(key, "1" if default else "0") == "1",
                    on_change=lambda _v: self.store(),
                )
                self.effect_rows[key] = row

        self.save_log_row = AdwSwitchRow(
            dev,
            title="Save log and frame stats to file",
            subtitle="Writes logs/YYYYMMDD_HHMMSS.log and .frames.csv for stutter and crash diagnosis",
            active=self.settings.get("save_log", True),
            on_change=lambda _v: self.store(),
        )
        self.profile_row = AdwSwitchRow(
            dev,
            title="GPU profile in log",
            subtitle="BB_GPU_PROFILE",
            active=self.settings.get("gpu_profile", False),
            on_change=lambda _v: self.store(),
        )
        self.validation_row = AdwSwitchRow(
            dev,
            title="Vulkan validation layers",
            subtitle="Slows rendering down significantly",
            active=self.settings.get("vk_validation", False),
            on_change=lambda _v: self.store(),
        )
        self.extra_row = AdwEntryRow(
            dev,
            title="Extra environment variables (KEY=value space-separated)",
            text=self.settings.get("extra_env", ""),
            on_change=lambda _v: self.store(),
        )

    # --- Page 5: Controls ---------------------------------------------------------------

    def build_controls_page(self, page):
        device_group = AdwPreferencesGroup(page, title="Controller and Mouse Camera")
        self.gamepad_row = AdwComboRow(
            device_group, title="Controller", on_change=lambda _v: self.store()
        )
        self.gamepad_row.add_suffix_btn("↻", self.fill_gamepads)
        self.fill_gamepads()

        self.mouse_look_row = AdwSwitchRow(
            device_group,
            title="Mouse camera look",
            subtitle="Click in the game window to capture the mouse; press F1 to release it",
            active=self.ini.get("mouse_look", "1") == "1",
            on_change=lambda _v: self.store(),
        )
        self.mouse_sensitivity_row = AdwSpinRow(
            device_group,
            title="Mouse sensitivity",
            min_v=0.1, max_v=10.0, step=0.05, digits=2,
            value=float(self.ini.get("mouse_sensitivity", "1.00") or 1.00),
            on_change=lambda _v: self.store(),
        )
        self.mouse_invert_row = AdwSwitchRow(
            device_group,
            title="Invert vertical mouse look",
            active=self.ini.get("mouse_invert_y") == "1",
            on_change=lambda _v: self.store(),
        )

        kb_group = AdwPreferencesGroup(
            page,
            title="Keyboard and Mouse Bindings",
            description="Key and mouse button bindings; also configurable live in-game under System → Controls",
        )
        for name, label, key_default, _pad_default in CONTROLS:
            if key_default is None:
                continue
            row = AdwRowBase(kb_group, title=label)
            make_flat_btn(row.right, "Reset", lambda k="key", n=name: self.set_control(k, n, None)).pack(side=tk.RIGHT, padx=(4, 0))
            make_flat_btn(row.right, "Assign", lambda k="key", n=name: self.assign_control(k, n)).pack(side=tk.RIGHT)
            self.control_rows[("key", name)] = (row, key_default)
            self.show_control("key", name)

        pad_group = AdwPreferencesGroup(
            page,
            title="Gamepad Button Bindings",
            description="SDL gamepad button bindings; also configurable live in-game under System → Controls",
        )
        for name, label, _key_default, pad_default in CONTROLS:
            if pad_default is None:
                continue
            row = AdwRowBase(pad_group, title=label)
            make_flat_btn(row.right, "Reset", lambda k="pad", n=name: self.set_control(k, n, None)).pack(side=tk.RIGHT, padx=(4, 0))
            make_flat_btn(row.right, "Assign", lambda k="pad", n=name: self.assign_control(k, n)).pack(side=tk.RIGHT)
            self.control_rows[("pad", name)] = (row, pad_default)
            self.show_control("pad", name)


    # --- Page 6: Game and Mods ----------------------------------------------------------

    def build_game_page(self, page):
        game = AdwPreferencesGroup(
            page,
            title="Game and Language",
            description="Game and save data folders, in-game language, and startup patches",
        )
        self.game_row = AdwRowBase(game, title="Game folder (CUSA03173 / eboot.bin)")
        make_flat_btn(self.game_row.right, "Open Folder", lambda: self.open_folder(self.game_dir())).pack(side=tk.RIGHT, padx=(6, 0))
        make_flat_btn(self.game_row.right, "Browse…", self.on_choose_game).pack(side=tk.RIGHT)

        self.user_row = AdwRowBase(game, title="Save data folder")
        make_flat_btn(self.user_row.right, "Reset", self.on_reset_user).pack(side=tk.RIGHT, padx=(6, 0))
        make_flat_btn(self.user_row.right, "Open Folder", lambda: self.open_folder(self.user_dir())).pack(side=tk.RIGHT, padx=(6, 0))
        make_flat_btn(self.user_row.right, "Browse…", self.on_choose_user).pack(side=tk.RIGHT)

        self.language_row = AdwComboRow(
            game,
            title="System language",
            choices=LANGUAGES,
            current=self.ini.get("language") or str(self.settings.get("language", "1")),
            on_change=lambda _v: self.store(),
        )

        for key, title, default, subtitle in EFFECTS:
            if key == "skip_intro":
                row = AdwSwitchRow(
                    game,
                    title=title,
                    subtitle=subtitle,
                    active=self.ini.get(key, "1" if default else "0") == "1",
                    on_change=lambda _v: self.store(),
                )
                self.effect_rows[key] = row

        self.skip_network_row = AdwSwitchRow(
            game,
            title="Skip 'Play Online / Play Offline' prompt",
            subtitle="Opens the main menu directly in offline mode (game patch)",
            active=self.ini.get("skip_network_choice", "1" if self.settings.get("skip_network_choice", True) else "0") != "0",
            on_change=lambda _v: self.store(),
        )

        mods_group = AdwPreferencesGroup(
            page,
            title="Mods",
            description="Unpack each mod into its own subfolder in mods/ (with dvdroot_ps4 or chr/, parts/, etc.). Lower mods override higher ones. Applied on startup.",
        )
        self.mods_enabled_row = AdwSwitchRow(
            mods_group,
            title="Enable mods",
            active=self.settings.get("mods_enabled", True),
            on_change=lambda _v: self.store(),
        )
        self.mods_folder_row = AdwRowBase(mods_group, title="Mods folder", subtitle=str(self.mods_dir()))
        make_flat_btn(self.mods_folder_row.right, "Open Folder", lambda: self.open_folder(self.mods_dir())).pack(side=tk.RIGHT, padx=(6, 0))
        make_flat_btn(self.mods_folder_row.right, "Browse…", self.on_choose_mods).pack(side=tk.RIGHT)

        patches_group = AdwPreferencesGroup(
            page,
            title="External Patches",
            description="shadPS4-format 01.09 XML patches in patches/. Applied on startup.",
        )
        self.patches_folder_row = AdwRowBase(patches_group, title="Patches folder", subtitle=str(self.patches_dir()))
        make_flat_btn(self.patches_folder_row.right, "Open Folder", lambda: self.open_folder(self.patches_dir())).pack(side=tk.RIGHT, padx=(6, 0))
        make_flat_btn(self.patches_folder_row.right, "Browse…", self.on_choose_patches).pack(side=tk.RIGHT)

    # --- Page 7: Online -----------------------------------------------------------------

    def build_online_page(self, page):
        play = AdwPreferencesGroup(page, title="Network Play")
        self.online_row = AdwSwitchRow(
            play,
            title="Play online",
            active=self.settings.get("online", False),
            on_change=lambda _v: self.store(),
        )
        self.pool_row = AdwComboRow(
            play,
            title="Server pool",
            choices=[
                ("Seamless Hunt (seamlesshunt.com)", "seamless",
                 "BBSC server: seamless co-op, bells, and summons; register on the website"),
                ("The Hunter's Dream (thehuntersdream.com)", "huntersdream",
                 "Co-op via The Hunter's Dream; sign in with Discord"),
            ],
            current="huntersdream" if self.settings.get("online_pool") == "huntersdream" else "seamless",
            on_change=lambda _v: self.store(),
            width=34,
        )

        game_srv = AdwPreferencesGroup(page, title="Game Server (Messages, Bloodstains, Ghosts)")
        self.community_row = AdwEntryRow(
            game_srv, title="Game server", text=self.settings.get("online_community", ""),
            on_change=lambda _v: self.store(),
        )

        coop = AdwPreferencesGroup(
            page,
            title="Co-op Server (Bells and Summons)",
            description="play.seamlesshunt.com:31313 is the Seamless Hunt (BBSC) server. Leave WebAPI empty to use port 31315 on the server host.",
        )
        self.server_row = AdwEntryRow(
            coop, title="Server address", text=self.settings.get("online_server", ""),
            on_change=lambda _v: self.store(),
        )
        self.webapi_row = AdwEntryRow(
            coop, title="WebAPI address", text=self.settings.get("online_webapi", ""),
            on_change=lambda _v: self.store(),
        )
        self.bbsc_row = AdwSwitchRow(
            coop,
            title="Seamless Co-op (BBSC)",
            subtitle="Session persists across deaths, lamps, area transitions, and bosses; up to 5 hunters",
            active=self.settings.get("online_bbsc", True),
            on_change=lambda _v: self.store(),
        )

        account = AdwPreferencesGroup(
            page,
            title="Account",
            description="shadNet account name (NPID), not your email. Register at seamlesshunt.com.",
        )
        self.npid_row = AdwEntryRow(
            account, title="Online ID (NPID)", text=self.settings.get("online_npid", ""),
            on_change=lambda _v: self.store(),
        )
        self.password_row = AdwEntryRow(
            account, title="Password", text=self.settings.get("online_password", ""),
            password=True, on_change=lambda _v: self.store(),
        )

        conn = AdwPreferencesGroup(
            page,
            title="Connection",
            description="All players in a session must run game version 1.09 on the same server.",
        )
        self.upnp_row = AdwSwitchRow(
            conn,
            title="UPnP (forward port on router automatically)",
            subtitle="Disable when using Tailscale or another VPN",
            active=self.settings.get("online_upnp", True),
            on_change=lambda _v: self.store(),
        )

    # --- Page 8: Log --------------------------------------------------------------------

    def build_log_page(self, page):
        top = tk.Frame(page, bg="#141414")
        top.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            top, bg="#141414", fg="#e0e0e0", insertbackground="#ffffff",
            font=("Consolas" if sys.platform == "win32" else "Monospace", 9),
            relief="flat", bd=0, padx=14, pady=12, wrap="word"
        )
        lscroll = tk.Scrollbar(top, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=lscroll.set)
        lscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bar = tk.Frame(page, bg=BG_HEADER, height=44)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        make_flat_btn(bar, "Export Log…", self.on_export_log).pack(side=tk.RIGHT, padx=(6, 16), pady=8)
        make_flat_btn(bar, "Copy", self.on_copy_log).pack(side=tk.RIGHT, padx=6, pady=8)
        make_flat_btn(bar, "Clear", lambda: self.log_text.delete("1.0", tk.END)).pack(side=tk.RIGHT, padx=6, pady=8)

    # --- Helpers & Dynamic Lists --------------------------------------------------------

    def fill_gpus(self):
        current = self.settings.get("gpu", "")
        gpus = connected_gpus()
        best_name = next((name for _i, _v, _k, name, best in gpus if best), None)
        auto_label = f"Auto ({best_name})" if best_name else "Auto"
        choices = [(auto_label, "")] + [(name, name) for _i, _v, _k, name, _best in gpus]
        if current and current not in [v for _, v in choices]:
            choices.append((f"{current} (not found)", current))
        self.gpu_row.set_choices(choices, current)

    def fill_displays(self):
        current = self.settings.get("display", "")
        displays = connected_displays()
        self.display_sizes = {val: size for val, _lbl, size, _pri in displays if size}
        pri_size = next((size for _v, _l, size, pri in displays if pri and size), None)
        if pri_size:
            self.display_sizes[""] = pri_size
        choices = [("Primary", "")] + [(lbl, val) for val, lbl, _sz, _pri in displays]
        if current and current not in [v for _, v in choices]:
            choices.append((f"{current} (disconnected)", current))
        self.display_row.set_choices(choices, current)

    def monitor_size(self):
        val = self.display_row.get_value() if hasattr(self, "display_row") else ""
        sizes = getattr(self, "display_sizes", {})
        return sizes.get(val) or sizes.get("") or (2560, 1440)

    def on_display_changed(self):
        keep = parse_size(self.output_row.get_value())
        self.fill_aspects(self.aspect_row.get_value())
        self.fill_outputs(keep)
        self.store()

    def fill_aspects(self, current):
        auto_aspect = resolved_aspect("auto", self.monitor_size())
        choices = [(f"Auto ({auto_aspect})", "auto")] + [(a, a) for a in ASPECTS[1:]]
        self.aspect_row.set_choices(choices, current if current in ASPECTS else "auto")

    def fill_outputs(self, keep):
        aspect = self.aspect_row.get_value() or "auto"
        sizes = output_sizes(aspect, self.monitor_size())
        choices = [(f"{w}×{h}", f"{w}x{h}") for w, h in sizes]
        target = min(sizes, key=lambda s: abs(s[1] - keep[1])) if keep not in sizes else keep
        self.output_row.set_choices(choices, f"{target[0]}x{target[1]}")

    def fill_gamepads(self):
        current = self.settings.get("gamepad", "")
        choices = [("Auto (First connected)", "")]
        for guid, name in connected_gamepads():
            choices.append((name, guid))
        if current and current not in [v for _, v in choices]:
            choices.append((f"{current} (disconnected)", current))
        self.gamepad_row.set_choices(choices, current)

    def show_control(self, kind, name):
        row, default = self.control_rows[(kind, name)]
        val = self.ini.get(f"{kind}.{name}")
        if val is None:
            row.set_subtitle(f"{default} (default)" if default else "Unbound")
        else:
            row.set_subtitle(val or "Unbound")

    def set_control(self, kind, name, value):
        self.ini[f"{kind}.{name}"] = value
        self.show_control(kind, name)
        self.store()

    def assign_control(self, kind, name):
        tool = cap_tool_path()
        row, _default = self.control_rows[(kind, name)]
        row.set_subtitle("Press a key or button… (Esc to cancel)")

        def work():
            try:
                res = subprocess.run([str(tool), "--read-input", kind],
                                     capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
                out = (res.stdout or "").strip()
            except Exception:
                out = ""

            def finish():
                words = out.split(" ", 1)
                if len(words) == 2 and words[0] == kind:
                    self.set_control(kind, name, "" if words[1] == "Delete" else words[1])
                else:
                    self.show_control(kind, name)
            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def game_dir(self):
        return Path(self.settings.get("game_dir") or "").expanduser()

    def user_dir(self):
        return Path(self.settings["user_dir"]).expanduser() if self.settings.get("user_dir") else DATA_DIR / "user"

    def mods_dir(self):
        return Path(self.settings.get("mods_dir") or DATA_DIR / "mods").expanduser()

    def patches_dir(self):
        return Path(self.settings.get("patches_dir") or DATA_DIR / "patches").expanduser()

    def open_folder(self, path):
        p = Path(path)
        with contextlib.suppress(OSError):
            p.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            with contextlib.suppress(OSError):
                os.startfile(str(p))
        else:
            with contextlib.suppress(OSError):
                subprocess.Popen(["xdg-open", str(p)])

    def on_choose_game(self):
        chosen = filedialog.askdirectory(
            title="Select Bloodborne game folder (containing eboot.bin)",
            initialdir=str(self.game_dir() if self.game_dir().is_dir() else PORT_DIR.parent),
        )
        if not chosen:
            return
        p = Path(chosen)
        for cand in (p, p / "CUSA03173", p / "game", p / "dvdroot_ps4"):
            if (cand / "eboot.bin").is_file():
                p = cand
                break
        self.settings["game_dir"] = str(p.resolve())
        self.update_game_status()
        self.store()

    def on_choose_user(self):
        chosen = filedialog.askdirectory(title="Select save data folder", initialdir=str(self.user_dir()))
        if chosen:
            self.settings["user_dir"] = str(Path(chosen).resolve())
            self.update_user_status()
            self.store()

    def on_reset_user(self):
        self.settings["user_dir"] = ""
        self.update_user_status()
        self.store()

    def on_choose_mods(self):
        chosen = filedialog.askdirectory(title="Select mods folder", initialdir=str(self.mods_dir()))
        if chosen:
            self.settings["mods_dir"] = str(Path(chosen).resolve())
            self.mods_folder_row.set_subtitle(str(self.mods_dir()))
            self.store()

    def on_choose_patches(self):
        chosen = filedialog.askdirectory(title="Select patches folder", initialdir=str(self.patches_dir()))
        if chosen:
            self.settings["patches_dir"] = str(Path(chosen).resolve())
            self.patches_folder_row.set_subtitle(str(self.patches_dir()))
            self.store()

    def update_game_status(self):
        path = self.game_dir()
        ok = bool(self.settings.get("game_dir")) and (path / "eboot.bin").is_file()
        if ok:
            self.game_row.set_subtitle(f"{path}  ✓ (eboot.bin found)")
        else:
            self.game_row.set_subtitle(f"{path if self.settings.get('game_dir') else 'Not selected'}  ⚠ (No eboot.bin found)")

    def update_user_status(self):
        path = self.user_dir()
        custom = bool(self.settings.get("user_dir"))
        saves = path / "savedata"
        found = saves.is_dir() and any(saves.iterdir())
        badge = "✓ Saves found" if found else "New save folder"
        self.user_row.set_subtitle(f"{path if custom else f'Default: {path}'}  ({badge})")

    def update_upscaler_status(self):
        up = self.upscaler_row.get_value()
        aa = self.aa_row.get_value()
        bindir = PORT_DIR / ("bin" if (PORT_DIR / "bin").is_dir() else "out")
        if sys.platform == "win32":
            bridge = (bindir / "bbport_dlss.dll").is_file()
            ngx = (bindir / "nvngx_dlss.dll").is_file() or (self.user_dir() / "dlss" / "nvngx_dlss.dll").is_file()
        else:
            bridge = (bindir / "libbbport_dlss.so").is_file()
            ngx = any(bindir.glob("libnvidia-ngx-dlss.so.*"))
        if hasattr(self, "dlss_status_row"):
            if bridge and ngx:
                self.dlss_status_row.set_subtitle("Bundled & Ready (NVIDIA NGX DLSS 4.5 / DLAA 4 Transformer)")
            else:
                self.dlss_status_row.set_subtitle("Not found in bin/")

        if up == "fsr4":
            ok = (PORT_DIR / "fsr4_shaders").is_dir()
            self.upscaler_row.set_subtitle("FSR 4 shader assets found" if ok else "Missing fsr4_shaders/ folder")
        elif up == "dlss":
            self.upscaler_row.set_subtitle("DLSS ready (requires NVIDIA RTX GPU)" if (bridge and ngx) else "DLSS library missing; will fall back to FSR 3.1")
        else:
            self.upscaler_row.set_subtitle("")

        self.preset_row.set_sensitive(up != "off")
        self.dlss_preset_row.set_sensitive(up == "dlss" or aa == "dlaa")

    def update_launch_button(self):
        running = self.process is not None
        self.play_btn.configure(
            text="   Stop   " if running else "   Play   ",
            bg=ACCENT_RED if running else ACCENT_BLUE,
            activebackground="#c01820" if running else ACCENT_BLUE_HOV,
        )

    # --- Two-Way Persistence (bbport.ini <-> Launcher.py <-> Bloodborne.exe) ------------

    def store(self):
        if self._reloading_ini:
            return
        s = self.settings
        s["gpu"] = self.gpu_row.get_value()
        s["display"] = self.display_row.get_value()
        s["fullscreen"] = self.fullscreen_row.get_active()
        s["hdr"] = self.hdr_row.get_active()
        s["present_mode"] = self.present_row.get_value()
        s["fps_limit"] = int(round(self.limit_row.get_value()))
        s["draw_pipe"] = self.pipe_row.get_value()
        s["hw_occlusion_cull"] = self.occlusion_cull_row.get_active()
        s["native_shaders"] = self.native_shaders_row.get_active()
        s["save_log"] = self.save_log_row.get_active()
        s["gpu_profile"] = self.profile_row.get_active()
        s["vk_validation"] = self.validation_row.get_active()
        s["extra_env"] = self.extra_row.get_text().strip()
        s["gamepad"] = self.gamepad_row.get_value()
        s["language"] = self.language_row.get_value()
        s["skip_network_choice"] = self.skip_network_row.get_active()
        s["mods_enabled"] = self.mods_enabled_row.get_active()
        s["online"] = self.online_row.get_active()
        s["online_pool"] = self.pool_row.get_value()
        s["online_community"] = self.community_row.get_text().strip()
        s["online_server"] = self.server_row.get_text().strip()
        s["online_webapi"] = self.webapi_row.get_text().strip()
        s["online_bbsc"] = self.bbsc_row.get_active()
        s["online_npid"] = self.npid_row.get_text().strip()
        s["online_password"] = self.password_row.get_text()
        s["online_upnp"] = self.upnp_row.get_active()
        save_settings(s)

        aa_val = self.aa_row.get_value()
        up_val = self.upscaler_row.get_value()
        self.ini.update({
            "game_dir": s.get("game_dir", ""),
            "aa_mode": aa_val,
            "upscaler": up_val,
            "preset": str(self.preset_row.get_value()),
            "dlss_preset": str(self.dlss_preset_row.get_value()),
            "anisotropic": str(self.aniso_row.get_value()),
            "ui_upscale": "1" if self.ui_upscale_row.get_active() else "0",
            "fps_limit": str(int(round(self.limit_row.get_value()))),
            "language": str(self.language_row.get_value()),
            "fullscreen": "1" if self.fullscreen_row.get_active() else "0",
            "present_mode": str(self.present_row.get_value()),
            "skip_network_choice": "1" if self.skip_network_row.get_active() else "0",
            "sharpen": "1" if self.sharpen_row.get_active() else "0",
            "sharpness": f"{self.sharpness_row.get_value():.2f}",
            "object_motion": "1" if self.motion_row.get_active() else "0",
            "show_fps": "1" if self.show_fps_row.get_active() else "0",
            "aspect_ratio": self.aspect_row.get_value(),
            "output_res": self.output_row.get_value(),
            "model_lod": self.lod_row.get_value(),
            "live_resolution": "0",
            "effect_game_aa": "1" if aa_val == "fxaa" else "0",
            "mouse_look": "1" if self.mouse_look_row.get_active() else "0",
            "mouse_sensitivity": f"{self.mouse_sensitivity_row.get_value():.2f}",
            "mouse_invert_y": "1" if self.mouse_invert_row.get_active() else "0",
            **{key: "1" if row.get_active() else "0" for key, row in self.effect_rows.items()},
        })
        save_ini(self.ini, self.ini_lines)
        self.ini, self.ini_lines = load_ini()
        self._ini_mtime_ns = ini_mtime_ns()

    def check_ini_reload(self):
        self.reload_from_ini()
        self.root.after(1000, self.check_ini_reload)

    def reload_from_ini(self, force=False):
        if self._reloading_ini:
            return
        mtime = ini_mtime_ns()
        if not force and mtime == self._ini_mtime_ns:
            return
        self._reloading_ini = True
        try:
            self.ini, self.ini_lines = load_ini()
            self._ini_mtime_ns = mtime
            s = self.settings
            if self.ini.get("game_dir") and (Path(self.ini["game_dir"]) / "eboot.bin").is_file():
                s["game_dir"] = self.ini["game_dir"]
            if "language" in self.ini:
                s["language"] = self.ini["language"]
            if "fullscreen" in self.ini:
                s["fullscreen"] = self.ini["fullscreen"] == "1"
            if "present_mode" in self.ini:
                s["present_mode"] = self.ini["present_mode"]
            if "skip_network_choice" in self.ini:
                s["skip_network_choice"] = self.ini["skip_network_choice"] != "0"
            if "fps_limit" in self.ini:
                with contextlib.suppress(ValueError):
                    s["fps_limit"] = int(self.ini.get("fps_limit", "0") or 0)
            save_settings(s)

            aspect = self.ini.get("aspect_ratio", "auto")
            self.fill_aspects(aspect if aspect in ASPECTS else "auto")
            self.fill_outputs(parse_size(self.ini.get("output_res", "2560x1440")))
            self.fullscreen_row.set_active(bool(s["fullscreen"]))
            self.present_row.set_value(s["present_mode"])
            self.language_row.set_value(s["language"])
            self.skip_network_row.set_active(bool(s.get("skip_network_choice", True)))

            raw_up = self.ini.get("upscaler", "off")
            raw_preset = str(self.ini.get("preset", "0") or "0")
            raw_aa = self.ini.get("aa_mode", "smaa_t2x")
            self.aa_row.set_value(raw_aa if raw_aa in ("off", "fxaa", "smaa", "smaa_t2x", "taa", "dlaa", "dlaa3") else "smaa_t2x")
            self.upscaler_row.set_value(raw_up if raw_up in ("off", "dlss", "fsr3", "fsr4", "fsr411") else "off")
            self.dlss_preset_row.set_value(self.ini.get("dlss_preset", "11"))
            self.preset_row.set_value(raw_preset)
            self.aniso_row.set_value(self.ini.get("anisotropic", "16"))
            self.ui_upscale_row.set_active(self.ini.get("ui_upscale", "1") != "0")
            self.sharpen_row.set_active(self.ini.get("sharpen", "1") == "1")
            with contextlib.suppress(ValueError):
                self.sharpness_row.set_value(float(self.ini.get("sharpness", "0.50")))
            self.motion_row.set_active(self.ini.get("object_motion") == "1")
            self.show_fps_row.set_active(self.ini.get("show_fps", "1") == "1")
            self.lod_row.set_value(self.ini.get("model_lod", "0"))
            for key, _title, default, *_ in EFFECTS + REMASTER_ENHANCEMENTS:
                if key in self.effect_rows:
                    self.effect_rows[key].set_active(self.ini.get(key, "1" if default else "0") == "1")

            cur_fps = int(s.get("fps_limit", 0) or 0)
            preset_vals = [v for _, v in FPS_PRESET_CHOICES if v >= 0]
            self._syncing_fps = True
            self.fps_row.set_value(cur_fps if cur_fps in preset_vals else -1)
            self.limit_row.set_value(cur_fps)
            self._syncing_fps = False

            self.mouse_look_row.set_active(self.ini.get("mouse_look", "1") == "1")
            with contextlib.suppress(ValueError):
                self.mouse_sensitivity_row.set_value(float(self.ini.get("mouse_sensitivity", "1.00")))
            self.mouse_invert_row.set_active(self.ini.get("mouse_invert_y") == "1")
            for kind, name in self.control_rows:
                self.show_control(kind, name)
            self.update_game_status()
            self.update_upscaler_status()
        finally:
            self._reloading_ini = False

    # --- Launch & Stop Bloodborne.exe / run.sh ------------------------------------------

    def append_log(self, text):
        def _do():
            self.log_text.insert(tk.END, text)
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > MAX_LOG_LINES:
                self.log_text.delete("1.0", f"{lines - MAX_LOG_LINES}.0")
            self.log_text.see(tk.END)
        self.root.after(0, _do)

    def on_copy_log(self):
        txt = self.log_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)

    def on_export_log(self):
        txt = self.log_text.get("1.0", tk.END)
        chosen = filedialog.asksaveasfilename(
            title="Export Log",
            initialfile=f"bbport-{time.strftime('%Y%m%d_%H%M%S')}.log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")],
        )
        if chosen:
            with contextlib.suppress(OSError):
                Path(chosen).write_text(txt, encoding="utf-8")

    def on_launch(self):
        if self.process is not None:
            self.stop_game()
            return
        self.store()
        gdir = self.game_dir()
        if not self.settings.get("game_dir") or not (gdir / "eboot.bin").is_file():
            self.switch_page("game")
            messagebox.showwarning(
                "Bloodborne PC Vulkan",
                "Please select your Bloodborne game folder containing eboot.bin (e.g. CUSA03173 or game)."
            )
            self.on_choose_game()
            gdir = self.game_dir()
            if not (gdir / "eboot.bin").is_file():
                return

        self.log_text.delete("1.0", tk.END)
        self._stopping_manually = False

        env = dict(os.environ)
        env["BB_CONFIG"] = str(INI_FILE)
        env["BB_GAME_DIR"] = str(gdir)
        if self.settings.get("user_dir"):
            env["BB_USER_DIR"] = str(self.settings["user_dir"])
        env["BB_LANGUAGE"] = str(self.settings.get("language", "1"))
        env["BB_MENU_LANGUAGE"] = "en"
        env["BB_FULLSCREEN"] = "1" if self.settings.get("fullscreen") else "0"
        env["BB_PRESENT_MODE"] = str(self.settings.get("present_mode", "Mailbox"))
        env["BB_FPS"] = "uncap"
        env["BB_FPS_LIMIT"] = str(self.settings.get("fps_limit", 0) or 0)
        env["BB_HW_OCCLUSION_CULL"] = "1" if self.settings.get("hw_occlusion_cull", True) else "0"
        env["BB_MICRO_SHADOW_CULL"] = "1" if self.settings.get("hw_occlusion_cull", True) else "0"
        env["BB_NATIVE_SHADERS"] = "1" if self.settings.get("native_shaders", True) else "0"
        env.setdefault("BB_READBACKS", "0")
        env.setdefault("BB_BREADCRUMBS", "0")
        env["BB_PC_MODEL"] = "0"
        env["BB_GX_MODULES"] = "all"

        if sys.platform == "win32":
            bb_exe = PORT_DIR / "Bloodborne.exe"
            if bb_exe.is_file():
                cmd = [str(bb_exe)]
            else:
                cmd = [sys.executable, str(PORT_DIR / "run.py")]
        else:
            cmd = ["bash", str(PORT_DIR / "run.sh")]

        try:
            self.process = subprocess.Popen(
                cmd, env=env, cwd=str(PORT_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW,
            )
        except OSError as err:
            messagebox.showerror("Bloodborne PC Vulkan", f"Could not start game: {err}")
            return

        self.update_launch_button()

        def reader():
            proc = self.process
            if proc and proc.stdout:
                for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace")
                    self.append_log(line)
            code = proc.wait() if proc else -1

            def on_done():
                stopped = self._stopping_manually
                self._stopping_manually = False
                self.process = None
                self.reload_from_ini(force=True)
                self.update_launch_button()
                self.append_log(f"\n— the game exited (code {code}) —\n")
                if code != 0 and not stopped:
                    self.switch_page("log")

            self.root.after(0, on_done)

        threading.Thread(target=reader, daemon=True).start()

    def stop_game(self):
        if not self.process:
            return
        self._stopping_manually = True
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                    creationflags=NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                self.process.terminate()

    def on_close(self):
        self.reload_from_ini()
        self.store()
        if self.process:
            self.stop_game()
        self.root.destroy()


def main():
    if sys.platform != "win32" and not os.environ.get("BB_FORCE_TK_LAUNCHER"):
        gtk_launcher = PORT_DIR / "launcher" / "bbport_launcher.py"
        if gtk_launcher.is_file():
            try:
                import gi
                gi.require_version("Gtk", "4.0")
                gi.require_version("Adw", "1")
                import runpy
                sys.path.insert(0, str(gtk_launcher.parent))
                runpy.run_path(str(gtk_launcher), run_name="__main__")
                return
            except Exception:
                pass
    if not HAS_TK:
        print("Error: neither GTK4/Libadwaita nor tkinter is installed.", file=sys.stderr)
        sys.exit(1)
    root = tk.Tk()
    AdwLauncherWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
