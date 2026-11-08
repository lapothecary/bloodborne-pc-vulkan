#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bloodborne PC Vulkan Launcher for Windows (Unified Python / Tkinter).

1:1 feature, setting, and default parity with launcher/bbport_launcher.py (Linux):
- All 7 Anti-Aliasing modes (SMAA T2x, SMAA 1x, DLAA 4 Transformer, DLAA 3 CNN Fast, TAA, FXAA, Off)
- All Upscalers (Off Native 100%, DLSS 4.5 / 4 NVIDIA RTX, FSR 3.1, FSR 4, FSR 4.1.1)
- All 10 Visual Effects + 4 Vulkan 1.3 Remaster Enhancements + 28 Keyboard/Mouse & Gamepad Controls
- 10-Category Hardware GPU Render-Pass Breakdown HUD (F8 / F10, show_fps=1 by default) & Live Log
- Frozen directly into Bloodborne.exe and bbport-run.exe via PyInstaller.
"""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading

FROZEN = getattr(sys, 'frozen', False)
PACKAGE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(PACKAGE_DIR / 'scripts'))
DATA_DIR = Path(os.environ.get('BB_DATA_DIR', PACKAGE_DIR))
CONFIG_DIR = Path(os.environ.get('APPDATA', Path.home())) / 'bb_vulkan-launcher'
LEGACY_CONFIG_DIR = Path(os.environ.get('APPDATA', Path.home())) / 'bbport-launcher'
CONFIG_FILE = CONFIG_DIR / 'settings.json'
INI_FILE = DATA_DIR / 'bbport.ini'
MAX_LOG_LINES = 5000

ANTI_ALIASING = [
    ('SMAA T2x (Морфологическое + темпоральное 2x)', 'smaa_t2x'),
    ('SMAA 1x (Морфологическое сглаживание Ultra)', 'smaa'),
    ('DLAA 4 (Нативное DLSS 100% — Transformer Preset K)', 'dlaa'),
    ('DLAA 3 (Нативное DLSS 100% — CNN Fast Preset F)', 'dlaa3'),
    ('TAA (Темпоральное сглаживание)', 'taa'),
    ('FXAA (Субпиксельное сглаживание)', 'fxaa'),
    ('Выкл.', 'off'),
]
UPSCALERS = [
    ('Выключен (Native 100%)', 'off'),
    ('DLSS (NVIDIA RTX)', 'dlss'),
    ('FSR 3.1', 'fsr3'),
    ('FSR 4', 'fsr4'),
    ('FSR 4.1.1', 'fsr411'),
]
DLSS_PRESETS = [
    ('DLSS 4 (Preset K — Transformer, рекомендуется)', '11'),
    ('DLSS 4.5 (Preset M — 2nd Gen Transformer)', '13'),
    ('DLSS 4.5 (Preset L — 2nd Gen Transformer)', '12'),
    ('DLSS (Auto / Default Preset)', '0'),
]
PRESETS = [
    ('Native 100% / DLAA', '0'),
    ('Quality (67% Scale)', '1'),
    ('Balanced (58% Scale)', '2'),
    ('Performance (50% Scale)', '3'),
    ('Ultra Performance (33% Scale)', '4'),
]
ANISOTROPIC_LEVELS = [('16x', '16'), ('8x', '8'), ('4x', '4'), ('2x', '2'), ('Выкл. (1x)', '1')]
ASPECTS = [('Авто (по монитору)', 'auto'), ('16:9', '16:9'), ('16:10', '16:10'),
           ('21:9 (Ultrawide)', '21:9'), ('32:9 (Super Ultrawide)', '32:9'), ('4:3', '4:3')]
OUTPUTS = [
    'auto', '1280x720', '1600x900', '1920x1080', '2560x1440', '3200x1800', '3840x2160',
    '1920x1200', '2560x1600', '2560x1080', '3440x1440', '3840x1600', '5120x1440', '5120x2160',
]
FPS_PRESET_CHOICES = [
    ('Без лимита (Анлимит)', '0'),
    ('30 FPS (PS4)', '30'),
    ('60 FPS', '60'),
    ('90 FPS', '90'),
    ('120 FPS', '120'),
    ('144 FPS', '144'),
    ('165 FPS', '165'),
    ('170 FPS', '170'),
    ('240 FPS', '240'),
    ('360 FPS', '360'),
]
PRESENT_MODES = [
    ('FastSync / Mailbox (Без разрывов и без зажима FPS)', 'Mailbox'),
    ('VSync Вкл. (FIFO)', 'Fifo'),
    ('Адаптивный VSync (FIFO Relaxed)', 'FifoRelaxed'),
    ('VSync Выкл. (Immediate)', 'Immediate'),
]
DRAW_PIPE = [
    ('Авто (Включён при 8+ потоках CPU)', ''),
    ('Включён', '1'),
    ('Выключен', '0'),
]
MODEL_LOD = [
    ('Как в игре (0)', '0'),
    ('Максимальная (-2, без упрощения вдали)', '-2'),
    ('Ниже (1)', '1'),
    ('Минимальная (2)', '2'),
]
LOW_LATENCIES = [
    ('Включена (NVIDIA Reflex / AMD Anti-Lag 2)', 'on'),
    ('Включена + Boost (NVIDIA Reflex Boost)', 'boost'),
    ('Выключена', 'off'),
]
TEXTURE_QUALITY = [
    ('Авто (по объёму VRAM)', 'auto'),
    ('Полное (Full)', 'full'),
    ('Половина (Half)', 'half'),
    ('Четверть (Quarter)', 'quarter'),
]
LANGUAGES = [
    ('Английский', '1'), ('Английский (Великобритания)', '18'), ('Русский', '8'),
    ('Японский', '0'), ('Французский', '2'), ('Испанский', '3'),
    ('Испанский (Латинская Америка)', '20'), ('Немецкий', '4'), ('Итальянский', '5'),
    ('Нидерландский', '6'), ('Португальский', '7'), ('Португальский (Бразилия)', '17'),
    ('Польский', '16'), ('Турецкий', '19'), ('Корейский', '9'),
    ('Китайский (упрощённый)', '11'), ('Китайский (традиционный)', '10'),
    ('Финский', '12'), ('Шведский', '13'), ('Датский', '14'), ('Норвежский', '15'),
]
EFFECTS = [
    ('effect_chromatic_aberration', 'Хроматическая аберрация (красно-синяя окантовка по краям)', True),
    ('effect_dof', 'Глубина резкости (DoF — размытие дальнего фона)', True),
    ('effect_motion_blur', 'Размытие в движении (Motion Blur)', True),
    ('effect_ssao', 'Затенение SSAO (контактные тени в углах и складках)', True),
    ('effect_dynamic_shadows', 'Тени от динамических источников (фонари, факелы, костры)', True),
    ('effect_ssr', 'Экранные отражения SSR (лужи, кровь, мокрый камень — меняются на лету)', False),
    ('effect_bloom', 'Свечение (Bloom вокруг фонарей, огня и луны)', True),
    ('skip_intro', 'Пропуск заставок при запуске (сразу к титульному экрану)', True),
    ('debug_camera', 'Свободная камера (Cross + L3 / F + Ctrl)', False),
    ('debug_menu', 'Debug menu (левый touchpad / Tab; нужны шрифты adhoc)', False),
]
REMASTER_ENHANCEMENTS = [
    ('ssao_quality', 'Улучшенный двухмасштабный GTAO / SSAO без шума (Interleaved Gradient Noise)', True),
    ('contact_shadows', 'Контактные микро-тени (Screen-Space) и мягкий 12-Tap Vogel PCF', True),
    ('ssr_quality', 'Высокоточный Ultra SSR (2x плотность лучей и 30 Гц Cubemap)', True),
    ('true_black_hdr', 'Глубокий чёрный (True Black / OLED), анти-дымка и сочность', True),
]
CONTROLS = [
    ('cross', 'Крест (взаимодействие)', 'F, Enter', 'a'),
    ('circle', 'Круг (уклонение / бег)', '', 'b'),
    ('dodge', 'Кувырок (короткое нажатие Круга; Shift+Space — прыжок)', 'Space', None),
    ('sprint', 'Бег (удержание Круга)', 'Left Shift', None),
    ('walk', 'Медленный шаг (удержание — 40% скорости)', 'Left Alt', None),
    ('square', 'Квадрат (быстрый предмет)', 'R, Tab', 'x'),
    ('triangle', 'Треугольник (флакон крови)', 'Q', 'y'),
    ('l1', 'L1 (трансформация оружия)', 'E, Mouse 5', 'leftshoulder'),
    ('r1', 'R1 (атака)', 'Mouse Left', 'rightshoulder'),
    ('l2', 'L2 (выстрел)', 'Mouse Right', 'lefttrigger'),
    ('r2', 'R2 (сильная атака)', 'Left Shift+Mouse Left, Mouse 4', 'righttrigger'),
    ('l3', 'L3 (прыжок / шаг / сортировка)', 'Left Ctrl', 'leftstick'),
    ('r3', 'R3 (захват цели)', 'C, Mouse Middle', 'rightstick'),
    ('options', 'Options (меню игры / пропуск катсцен)', 'Escape', 'start'),
    ('touchpad', 'Тачпад, левая половина (жесты)', 'G', 'back, touchpad'),
    ('touchpad_right', 'Тачпад, правая половина (личные вещи)', 'Backspace', ''),
    ('up', 'Крестовина вверх (кровавые пули / колесо вверх)', 'Wheel Up, 3', 'dpup'),
    ('down', 'Крестовина вниз (смена предмета / колесо вниз)', 'Wheel Down, 4', 'dpdown'),
    ('left', 'Крестовина влево (оружие левой руки)', '1', 'dpleft'),
    ('right', 'Крестовина вправо (оружие правой руки)', '2', 'dpright'),
    ('move_up', 'Движение вперёд', 'W', None),
    ('move_down', 'Движение назад', 'S', None),
    ('move_left', 'Движение влево', 'A', None),
    ('move_right', 'Движение вправо', 'D', None),
    ('look_up', 'Камера вверх', 'I', None),
    ('look_down', 'Камера вниз', 'K', None),
    ('look_left', 'Камера влево', 'J', None),
    ('look_right', 'Камера вправо', 'L', None),
]


def _discover_default_game_dir() -> str:
    env_dir = os.environ.get('BB_GAME_DIR', '').strip()
    if env_dir and (Path(env_dir) / 'eboot.bin').is_file():
        return env_dir
    candidates = [
        PACKAGE_DIR / 'CUSA03173',
        PACKAGE_DIR / 'game',
        PACKAGE_DIR.parent / 'CUSA03173',
        PACKAGE_DIR.parent / 'game',
        PACKAGE_DIR.parent / 'Bloodborne' / 'CUSA03173',
        PACKAGE_DIR.parent / 'Bloodborne' / 'game',
    ]
    for c in candidates:
        if (c / 'eboot.bin').is_file():
            return str(c)
    try:
        for sub in PACKAGE_DIR.parent.iterdir():
            if sub.is_dir():
                if (sub / 'eboot.bin').is_file():
                    return str(sub)
                for sub2 in sub.iterdir():
                    if sub2.is_dir() and (sub2 / 'eboot.bin').is_file():
                        return str(sub2)
    except OSError:
        pass
    return ''


DEFAULTS = {
    'ui_language': 'ru',
    'game_dir': _discover_default_game_dir(),
    'user_dir': '',
    'mods_dir': '',
    'mods_enabled': True,
    'patches_dir': '',
    'language': '8',
    'fullscreen': False,
    'hdr': False,
    'present_mode': 'Mailbox',
    'gamepad': '',
    'display': '',
    'gpu_id': '',
    'skip_network_choice': True,
    'fps_mode': 'uncap',
    'fps_limit': 0,
    'draw_pipe': '',
    'hw_occlusion_cull': True,
    'native_shaders': True,
    'save_log': True,
    'gpu_profile': False,
    'vk_validation': False,
    'extra_env': '',
    'online': False,
    'online_bbsc': True,
    'online_pool': 'seamless',
    'online_community': '',
    'online_server': '',
    'online_webapi': '',
    'online_npid': '',
    'online_password': '',
    'online_upnp': True,
    'thd_server': '',
    'thd_online_id': '',
    'thd_token': '',
    'trophy_key': '',
}

INI_DEFAULTS = {
    'aa_mode': 'smaa_t2x',
    'upscaler': 'off',
    'preset': '0',
    'dlss_preset': '11',
    'anisotropic': '16',
    'ui_upscale': '1',
    'fps_limit': '0',
    'sharpen': '1',
    'sharpness': '0.50',
    'object_motion': '0',
    'show_fps': '1',
    'aspect_ratio': 'auto',
    'output_res': 'auto',
    'model_lod': '0',
    'live_resolution': '0',
    'effect_game_aa': '1',
    'low_latency': 'on',
    'texture_quality': 'auto',
    'mouse_look': '1',
    'mouse_sensitivity': '1.00',
    'mouse_invert_y': '0',
    'menu_language': 'ru',
    **{key: '1' if default else '0' for key, _, default in EFFECTS + REMASTER_ENHANCEMENTS},
}


def load_settings():
    settings = dict(DEFAULTS)
    for cfg in (LEGACY_CONFIG_DIR / 'settings.json', CONFIG_FILE):
        try:
            if cfg.is_file():
                settings.update(json.loads(cfg.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            pass
    if not settings.get('game_dir') or not (Path(settings['game_dir']) / 'eboot.bin').is_file():
        auto_dir = _discover_default_game_dir()
        if auto_dir:
            settings['game_dir'] = auto_dir
    return settings


def save_settings(settings):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding='utf-8')


def read_ini():
    values = dict(INI_DEFAULTS)
    file_keys = set()
    try:
        for line in INI_FILE.read_text(encoding='utf-8', errors='replace').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                k = key.strip()
                file_keys.add(k)
                values[k] = value.strip()
    except OSError:
        pass
    # Migrate legacy Windows launcher defaults (upscaler=fsr4, preset=3, missing aa_mode / show_fps)
    if 'aa_mode' not in file_keys:
        if values.get('upscaler') == 'fsr4' and values.get('preset') == '3':
            values['upscaler'] = 'off'
            values['preset'] = '0'
        values['aa_mode'] = 'smaa_t2x'
        values['show_fps'] = '1'
    return values


def write_ini(changes):
    """Updates keys in bbport.ini while preserving all other lines and comments."""
    lines = []
    try:
        lines = INI_FILE.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        lines = ['# bbport settings (in-game menu: Insert / L3+R3, HUD: F8 / F10)']
    left = dict(changes)
    out = []
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in left:
                val = left.pop(key)
                if val is not None:
                    out.append(f'{key}={val}')
                continue
        out.append(line)
    for key, value in left.items():
        if value is not None:
            out.append(f'{key}={value}')
    INI_FILE.write_text('\n'.join(out) + '\n', encoding='utf-8')


def game_environment(s):
    """Environment for run.py from the launcher settings (1:1 with Linux launcher)."""
    env = dict(os.environ)
    env['BB_DATA_DIR'] = str(DATA_DIR)
    env['BB_CONFIG'] = str(INI_FILE)
    env['BB_GAME_DIR'] = str(s.get('game_dir') or '')
    if s.get('user_dir'):
        env['BB_USER_DIR'] = str(s['user_dir'])
    env['BB_LANGUAGE'] = str(s.get('language') or '8')
    env['BB_MENU_LANGUAGE'] = 'ru' if (s.get('ui_language') or 'ru') == 'ru' else 'en'
    env['BB_FULLSCREEN'] = '1' if s.get('fullscreen') else '0'
    if s.get('hdr'):
        env['BB_HDR'] = '1'
    env['BB_PRESENT_MODE'] = str(s.get('present_mode') or 'Mailbox')
    if s.get('display'):
        env['BB_DISPLAY'] = str(s['display'])
    if s.get('gamepad'):
        env['BB_GAMEPAD'] = str(s['gamepad'])
    env['BB_SKIP_NETWORK_CHOICE'] = '1' if s.get('skip_network_choice', True) else '0'
    env['BB_FPS'] = 'uncap'
    fps_lim = int(s.get('fps_limit') or 0)
    env['BB_FPS_LIMIT'] = str(fps_lim) if fps_lim > 0 else '0'
    if s.get('draw_pipe'):
        env['BB_DRAW_PIPE'] = str(s['draw_pipe'])
    env['BB_HW_OCCLUSION_CULL'] = '1' if s.get('hw_occlusion_cull', True) else '0'
    env['BB_MICRO_SHADOW_CULL'] = '1' if s.get('hw_occlusion_cull', True) else '0'
    env['BB_NATIVE_SHADERS'] = '1' if s.get('native_shaders', True) else '0'
    env['BB_PC_MODEL'] = os.environ.get('BB_PC_MODEL', '0')
    env['BB_GUEST_IN_PLACE'] = os.environ.get('BB_GUEST_IN_PLACE', env['BB_PC_MODEL'])
    env['BB_GX_MODULES'] = os.environ.get('BB_GX_MODULES', 'all')
    if str(s.get('gpu_id', '')).strip():
        env['BB_GPU_ID'] = str(s['gpu_id']).strip()
    if s.get('save_log', True):
        env['BB_SAVE_LOG'] = '1'
        env['BB_FRAME_STATS'] = '1'
    if s.get('gpu_profile'):
        env['BB_GPU_PROFILE'] = '1'
    if s.get('vk_validation'):
        env['BB_VK_VALIDATION'] = '1'
    env['BB_MODS_DIR'] = str(s.get('mods_dir') or DATA_DIR / 'mods')
    env['BB_MODS_ENABLED'] = '0' if s.get('mods_enabled') in (False, '0', 0) else '1'
    env['BB_PATCHES_DIR'] = str(s.get('patches_dir') or DATA_DIR / 'patches')
    if s.get('online') in (True, '1', 1):
        import online
        server = str(s.get('online_server') or '').strip() or online.SHADNET_SERVER
        community = online.community_address(s.get('online_community'))
        env['BB_ONLINE'] = '1'
        env['BB_SKIP_NETWORK_CHOICE'] = '0'
        env['BB_SHADNET_SERVER'] = server
        env['BB_SHADNET_WEBAPI'] = online.webapi_setting(s.get('online_webapi'), server, community)
        env['BB_SHADNET_NPID'] = str(s.get('online_npid') or '').strip()
        env['BB_SHADNET_PASSWORD'] = str(s.get('online_password') or '')
        env['BB_UPNP'] = '0' if s.get('online_upnp') in (False, '0', 0) else '1'
        env['BB_COMMUNITY_SERVER'] = community
        if s.get('online_pool') == 'huntersdream':
            env['BB_NET_BACKEND'] = 'thd'
            env['BB_COMMUNITY_SERVER'] = online.thd_base(s.get('thd_server', ''))
            env['BB_THD_ONLINE_ID'] = str(s.get('thd_online_id') or '')
            env['BB_THD_TOKEN'] = str(s.get('thd_token') or '')
        elif s.get('online_bbsc', True) not in (False, '0', 0):
            env['BB_BBSC'] = '1'
    trophy_key = str(s.get('trophy_key') or '').strip()
    if re.fullmatch(r'[0-9A-Fa-f]{32}', trophy_key):
        env['BB_TROPHY_KEY'] = trophy_key
    for item in str(s.get('extra_env') or '').split():
        if '=' in item:
            key, value = item.split('=', 1)
            env[key] = value
    return env


def play(settings, on_line=None):
    import run
    return run.run(env=game_environment(settings), on_line=on_line)


def _dlss_status_text() -> str:
    bindir = PACKAGE_DIR / 'bin'
    bridge = (bindir / 'bbport_dlss.dll').is_file()
    ngx = (bindir / 'nvngx_dlss.dll').is_file() or (DATA_DIR / 'user' / 'dlss' / 'nvngx_dlss.dll').is_file()
    if bridge and ngx:
        return 'DLSS 4.5 / DLAA 4 (Transformer): Активен (bbport_dlss.dll + nvngx_dlss.dll найдены)'
    missing = []
    if not bridge:
        missing.append('bbport_dlss.dll')
    if not ngx:
        missing.append('nvngx_dlss.dll')
    return f'DLSS не найден в bin\\ ({", ".join(missing)})'


def main_window():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    settings = load_settings()
    ini = read_ini()
    # Ensure bbport.ini exists with clean defaults on first launch
    if not INI_FILE.is_file():
        write_ini(ini)

    root = tk.Tk()
    version_file = PACKAGE_DIR / 'VERSION'
    version = version_file.read_text().strip() if version_file.is_file() else 'v1.1.0'
    root.title(f'Bloodborne PC Vulkan ({version} — C++20 Engine / DirectVk 1.3)')
    root.geometry('1040x780')
    icon = PACKAGE_DIR / 'launcher' / 'bloodborne.ico'
    if icon.is_file():
        try:
            root.iconbitmap(str(icon))
        except tk.TclError:
            pass

    top_frame = ttk.Frame(root)
    top_frame.pack(fill='x', padx=8, pady=(8, 4))
    ttk.Label(
        top_frame,
        text='Bloodborne PC Vulkan — 2,940 C++20 Engine | 109 Clean GLSL Shaders | F8/F10: 10-Pass GPU HUD | Insert: Меню в игре',
        font=('Segoe UI', 10, 'bold'),
    ).pack(side='left')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='x', padx=8, pady=4)
    variables = {}

    def row(parent, r, title, widget):
        ttk.Label(parent, text=title).grid(row=r, column=0, sticky='w', padx=8, pady=3)
        widget.grid(row=r, column=1, sticky='we', padx=8, pady=3)
        parent.columnconfigure(1, weight=1)

    def combo(parent, r, title, choices, value, key, store):
        labels = [label for label, _ in choices]
        current = next((label for label, v in choices if str(v) == str(value)), labels[0])
        var = tk.StringVar(value=current)
        cb = ttk.Combobox(parent, textvariable=var, values=labels, state='readonly')
        row(parent, r, title, cb)
        variables[key] = (store, lambda: next(v for label, v in choices if label == var.get()), var, choices)
        return cb, var

    def check(parent, r, title, value, key, store):
        var = tk.BooleanVar(value=bool(value))
        ttk.Checkbutton(parent, text=title, variable=var).grid(
            row=r, column=0, columnspan=2, sticky='w', padx=8, pady=2
        )
        variables[key] = (store, var.get, var, None)
        return var

    def entry(parent, r, title, value, key, store):
        var = tk.StringVar(value=str(value))
        row(parent, r, title, ttk.Entry(parent, textvariable=var))
        variables[key] = (store, var.get, var, None)
        return var

    # 1. Экран и FPS
    tab_display = ttk.Frame(notebook, padding=6)
    notebook.add(tab_display, text='1. Экран и FPS')
    out_choices = [('Авто (разрешение монитора)', 'auto')] + [(o, o) for o in OUTPUTS if o != 'auto']
    if ini.get('output_res') and ini['output_res'] not in [v for _, v in out_choices]:
        out_choices.append((ini['output_res'], ini['output_res']))
    combo(tab_display, 0, 'Разрешение вывода', out_choices, ini.get('output_res', 'auto'), 'output_res', 'i')
    combo(tab_display, 1, 'Соотношение сторон', ASPECTS, ini.get('aspect_ratio', 'auto'), 'aspect_ratio', 'i')
    cur_fps_lim = str(ini.get('fps_limit') or settings.get('fps_limit') or '0')
    _, fps_preset_var = combo(
        tab_display, 2, 'Пресет частоты кадров (меняется на лету)',
        FPS_PRESET_CHOICES, cur_fps_lim if cur_fps_lim in [v for _, v in FPS_PRESET_CHOICES] else '0',
        '_fps_preset', 'noop'
    )
    fps_limit_var = entry(tab_display, 3, 'Точное ограничение FPS (0 = честный анлимит)', cur_fps_lim, 'fps_limit', 'both_fps')

    def _on_fps_preset(*_):
        val = next((v for label, v in FPS_PRESET_CHOICES if label == fps_preset_var.get()), '0')
        fps_limit_var.set(val)

    fps_preset_var.trace_add('write', _on_fps_preset)
    combo(tab_display, 4, 'Режим вертикальной синхронизации (Present Mode)', PRESENT_MODES, settings['present_mode'], 'present_mode', 's')
    check(tab_display, 5, 'Полноэкранный режим (Fullscreen)', settings['fullscreen'], 'fullscreen', 's')
    check(tab_display, 6, 'Разрешить HDR', settings.get('hdr', False), 'hdr', 's')
    check(tab_display, 7, 'Оверлей телеметрии и 10 рендер-пассов GPU (F8 / F10 в игре; также пишет в лог каждые 5 с)',
          ini.get('show_fps', '1') == '1', 'show_fps', 'ib')
    entry(tab_display, 8, 'Видеокарта (индекс в Vulkan, пусто = авто дискретная)', settings.get('gpu_id', ''), 'gpu_id', 's')

    # 2. Апскейл и Сглаживание
    tab_upscale = ttk.Frame(notebook, padding=6)
    notebook.add(tab_upscale, text='2. Апскейл и Сглаживание')
    raw_aa = ini.get('aa_mode', 'smaa_t2x')
    if raw_aa not in ('smaa_t2x', 'smaa', 'dlaa', 'dlaa3', 'taa', 'fxaa', 'off'):
        raw_aa = 'smaa_t2x'
    combo(tab_upscale, 0, 'Сглаживание (Anti-Aliasing, при выключенном апскейлере)', ANTI_ALIASING, raw_aa, 'aa_mode', 'i')
    combo(tab_upscale, 1, 'Апскейлер (Upscaler)', UPSCALERS, ini.get('upscaler', 'off'), 'upscaler', 'i')
    combo(tab_upscale, 2, 'Режим масштабирования (Scaling Preset)', PRESETS, ini.get('preset', '0'), 'preset', 'i')
    combo(tab_upscale, 3, 'Модель DLSS / DLAA (Transformer / CNN)', DLSS_PRESETS, ini.get('dlss_preset', '11'), 'dlss_preset', 'i')
    combo(tab_upscale, 4, 'Анизотропная фильтрация текстур', ANISOTROPIC_LEVELS, ini.get('anisotropic', '16'), 'anisotropic', 'i')
    check(tab_upscale, 5, 'Апскейл иконок и интерфейса (AI / SDF 2x–4x)', ini.get('ui_upscale', '1') != '0', 'ui_upscale', 'ib')
    check(tab_upscale, 6, 'Фильтр адаптивной резкости (Fused Zero-Pass RCAS)', ini.get('sharpen', '1') == '1', 'sharpen', 'ib')
    entry(tab_upscale, 7, 'Сила резкости RCAS (0.00 .. 2.00)', ini.get('sharpness', '0.50'), 'sharpness', 'i')
    check(tab_upscale, 8, 'Векторы движения объектов (убирает шлейфы при апскейле; ~10% FPS)', ini.get('object_motion', '0') == '1', 'object_motion', 'ib')
    combo(tab_upscale, 9, 'Низкая задержка (NVIDIA Reflex / AMD Anti-Lag 2)', LOW_LATENCIES, ini.get('low_latency', 'on'), 'low_latency', 'i')
    combo(tab_upscale, 10, 'Качество текстур в VRAM', TEXTURE_QUALITY, ini.get('texture_quality', 'auto'), 'texture_quality', 'i')
    ttk.Label(tab_upscale, text=_dlss_status_text(), foreground='#1a6b2b').grid(
        row=11, column=0, columnspan=2, sticky='w', padx=8, pady=4
    )

    # 3. Эффекты и Ремастер
    tab_fx = ttk.Frame(notebook, padding=6)
    notebook.add(tab_fx, text='3. Эффекты и Ремастер')
    combo(tab_fx, 0, 'Детализация моделей (LOD)', MODEL_LOD, ini.get('model_lod', '0'), 'model_lod', 'i')
    r_idx = 1
    for key, title, default in EFFECTS:
        if key in ('skip_intro', 'debug_camera', 'debug_menu'):
            continue
        check(tab_fx, r_idx, title, ini.get(key, '1' if default else '0') == '1', key, 'ib')
        r_idx += 1
    ttk.Separator(tab_fx, orient='horizontal').grid(row=r_idx, column=0, columnspan=2, sticky='we', padx=8, pady=4)
    r_idx += 1
    ttk.Label(tab_fx, text='Улучшения графики ремастера (Vulkan 1.3 GLSL, меняются на лету):', font=('Segoe UI', 9, 'bold')).grid(
        row=r_idx, column=0, columnspan=2, sticky='w', padx=8, pady=2
    )
    r_idx += 1
    for key, title, default in REMASTER_ENHANCEMENTS:
        check(tab_fx, r_idx, title, ini.get(key, '1' if default else '0') == '1', key, 'ib')
        r_idx += 1

    # 4. Движок и Отладка
    tab_eng = ttk.Frame(notebook, padding=6)
    notebook.add(tab_eng, text='4. Движок и Отладка')
    combo(tab_eng, 0, 'Двухстадийный конвейер GPU (Stage A + Stage B)', DRAW_PIPE, settings.get('draw_pipe', ''), 'draw_pipe', 's')
    check(tab_eng, 1, 'Аппаратное 3D-отсечение невидимых объектов (Hardware Occlusion Culler)', settings.get('hw_occlusion_cull', True), 'hw_occlusion_cull', 's')
    check(tab_eng, 2, 'Нативные шейдеры Vulkan 1.3 GLSL (109 ручных шейдеров без регистровых дампов)', settings.get('native_shaders', True), 'native_shaders', 's')
    check(tab_eng, 3, 'Свободная камера (Cross + L3 / F + Ctrl)', ini.get('debug_camera', '0') == '1', 'debug_camera', 'ib')
    check(tab_eng, 4, 'Debug menu (левый touchpad / Tab; нужны шрифты adhoc)', ini.get('debug_menu', '0') == '1', 'debug_menu', 'ib')
    check(tab_eng, 5, 'Сохранять подробный журнал и статистику кадров в logs\\', settings.get('save_log', True), 'save_log', 's')
    check(tab_eng, 6, 'Профиль рендер-пассов GPU в журнале (BB_GPU_PROFILE)', settings.get('gpu_profile', False), 'gpu_profile', 's')
    check(tab_eng, 7, 'Слои валидации Vulkan (BB_VK_VALIDATION — сильно замедляет)', settings.get('vk_validation', False), 'vk_validation', 's')
    entry(tab_eng, 8, 'Доп. переменные окружения (KEY=VALUE через пробел)', settings.get('extra_env', ''), 'extra_env', 's')

    # 5. Управление
    tab_ctrl = ttk.Frame(notebook, padding=6)
    notebook.add(tab_ctrl, text='5. Управление')
    check(tab_ctrl, 0, 'Камера мышью (щелчок в окне захватывает мышь, F1 — отпускает)', ini.get('mouse_look', '1') == '1', 'mouse_look', 'ib')
    entry(tab_ctrl, 1, 'Чувствительность мыши (0.10 .. 10.00)', ini.get('mouse_sensitivity', '1.00'), 'mouse_sensitivity', 'i')
    check(tab_ctrl, 2, 'Инвертировать мышь по вертикали', ini.get('mouse_invert_y', '0') == '1', 'mouse_invert_y', 'ib')
    ttk.Label(
        tab_ctrl,
        text='Назначение клавиш и кнопок настраивается интерактивно прямо в игре: System -> Controls (или по клавише Insert).\n'
             'Медленный шаг: Left Alt | Кувырок: Space | Бег: Left Shift | Пропуск видео-катсцен: Enter / Escape / Start.',
        foreground='#444444',
    ).grid(row=3, column=0, columnspan=2, sticky='w', padx=8, pady=6)

    # 6. Игра и Моды
    tab_game = ttk.Frame(notebook, padding=6)
    notebook.add(tab_game, text='6. Игра и Моды')
    game_var = tk.StringVar(value=settings['game_dir'])
    folder = ttk.Frame(tab_game)
    ttk.Entry(folder, textvariable=game_var).pack(side='left', fill='x', expand=True)

    def _choose_game_dir():
        chosen = filedialog.askdirectory(title='Выберите папку с eboot.bin (например CUSA03173 или game)')
        if chosen:
            p = Path(chosen)
            if not (p / 'eboot.bin').is_file():
                for sub in ('CUSA03173', 'game', 'dvdroot_ps4'):
                    if (p / sub / 'eboot.bin').is_file():
                        p = p / sub
                        break
                    if sub == 'dvdroot_ps4' and (p.parent / 'eboot.bin').is_file():
                        p = p.parent
                        break
            game_var.set(str(p))

    ttk.Button(folder, text='Выбрать папку…', command=_choose_game_dir).pack(side='left', padx=4)
    row(tab_game, 0, 'Папка игры (с файлом eboot.bin)', folder)
    variables['game_dir'] = ('s', game_var.get, game_var, None)
    combo(tab_game, 1, 'Язык текста игры', LANGUAGES, settings['language'], 'language', 's')
    check(tab_game, 2, 'Пропускать заставки при запуске игры (Skip Intro)', ini.get('skip_intro', '1') == '1', 'skip_intro', 'ib')
    check(tab_game, 3, 'Пропускать выбор «онлайн / оффлайн» при входе', settings.get('skip_network_choice', True), 'skip_network_choice', 's')
    entry(tab_game, 4, 'Папка сохранений (пусто = папка user\\ рядом с лаунчером)', settings.get('user_dir', ''), 'user_dir', 's')
    check(tab_game, 5, 'Загружать моды из папки mods\\', settings.get('mods_enabled', True), 'mods_enabled', 's')
    entry(tab_game, 6, 'Своя папка модов (пусто = mods\\ рядом с лаунчером)', settings.get('mods_dir', ''), 'mods_dir', 's')
    ttk.Label(
        tab_game,
        text=f'Папка порта: {DATA_DIR}\nФайл настроек игры: {INI_FILE}\nНастройки лаунчера: {CONFIG_FILE}',
        foreground='gray',
    ).grid(row=7, column=0, columnspan=2, sticky='w', padx=8, pady=6)

    # Bottom Live Log + Action Bar
    log_frame = ttk.LabelFrame(root, text='Журнал запуска и телеметрия рендер-пассов (Live Log — обновляется каждые 5 секунд)', padding=4)
    log_frame.pack(fill='both', expand=True, padx=8, pady=4)
    log_scroll = ttk.Scrollbar(log_frame)
    log_scroll.pack(side='right', fill='y')
    log = tk.Text(
        log_frame, height=14, wrap='none', state='disabled',
        font=('Consolas', 9), bg='#111418', fg='#e6e1d6',
        yscrollcommand=log_scroll.set,
    )
    log.pack(fill='both', expand=True)
    log_scroll.config(command=log.yview)

    buttons = ttk.Frame(root)
    buttons.pack(fill='x', padx=8, pady=8)
    lines = queue.Queue()

    def collect():
        for key, (store, get, _var, _choices) in variables.items():
            if store == 'noop':
                continue
            value = get()
            if store == 'both_fps':
                try:
                    lim = max(0, int(str(value).strip() or 0))
                except ValueError:
                    lim = 0
                settings['fps_limit'] = lim
                ini['fps_limit'] = str(lim)
            elif store == 's':
                settings[key] = value
            elif store == 'i':
                ini[key] = str(value)
            elif store == 'ib':
                ini[key] = '1' if value else '0'
        ini['effect_game_aa'] = '1' if ini.get('aa_mode') == 'fxaa' else '0'
        ini['language'] = str(settings.get('language') or '8')
        ini['fullscreen'] = '1' if settings.get('fullscreen') else '0'
        ini['present_mode'] = str(settings.get('present_mode') or 'Mailbox')
        ini['skip_network_choice'] = '1' if settings.get('skip_network_choice', True) else '0'
        save_settings(settings)
        write_ini(ini)

    def append(text):
        log.configure(state='normal')
        log.insert('end', text + '\n')
        excess = int(log.index('end-1c').split('.')[0]) - MAX_LOG_LINES
        if excess > 0:
            log.delete('1.0', f'{excess}.0')
        log.see('end')
        log.configure(state='disabled')

    def pump():
        try:
            while True:
                item = lines.get_nowait()
                if item is None:
                    play_button.configure(state='normal', text='ЗАПУСТИТЬ ИГРУ')
                else:
                    append(item)
        except queue.Empty:
            pass
        root.after(80, pump)

    def start():
        collect()
        gdir = Path(settings.get('game_dir') or '')
        if not (gdir / 'eboot.bin').is_file():
            for sub in ('CUSA03173', 'game'):
                if (gdir / sub / 'eboot.bin').is_file():
                    gdir = gdir / sub
                    settings['game_dir'] = str(gdir)
                    game_var.set(str(gdir))
                    save_settings(settings)
                    break
        if not (gdir / 'eboot.bin').is_file():
            messagebox.showerror(
                'Bloodborne PC Vulkan',
                'В выбранной папке не найден eboot.bin!\n'
                'Перейдите на вкладку «6. Игра и Моды» и укажите папку игры (CUSA03173 или game).',
            )
            notebook.select(tab_game)
            return
        play_button.configure(state='disabled', text='ИГРА ЗАПУЩЕНА…')

        def worker():
            import run
            original = run.log

            def to_window(line):
                original(line)
                lines.put(line)

            run.log = to_window
            try:
                play(settings)
            except Exception as error:
                lines.put(f'STOP: {error!r}')
            finally:
                run.log = original
                lines.put(None)

        threading.Thread(target=worker, daemon=True).start()

    def open_logs():
        (DATA_DIR / 'logs').mkdir(exist_ok=True)
        os.startfile(str(DATA_DIR / 'logs'))  # noqa: S606

    def copy_log():
        root.clipboard_clear()
        root.clipboard_append(log.get('1.0', 'end-1c'))

    play_button = ttk.Button(buttons, text='ЗАПУСТИТЬ ИГРУ', command=start)
    play_button.pack(side='right', ipadx=14, ipady=4)
    ttk.Button(buttons, text='Сохранить настройки', command=collect).pack(side='right', padx=6, ipady=4)
    ttk.Button(buttons, text='Открыть папку logs\\', command=open_logs).pack(side='left', ipady=4)
    ttk.Button(buttons, text='Скопировать лог в буфер', command=copy_log).pack(side='left', padx=6, ipady=4)

    append(f'Готов к запуску. {_dlss_status_text()}')
    append('Горячие клавиши в игре: [F8] или [F10] — Оверлей FPS и 10 рендер-пассов GPU | [Insert] — Меню настроек | [F3] — Лок 30 FPS | [F1] — Курсор мыши')
    root.after(80, pump)
    root.protocol('WM_DELETE_WINDOW', lambda: (collect(), root.destroy()))
    root.mainloop()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--run-script':
        import runpy
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        sys.path.insert(0, str(Path(script).resolve().parent))
        try:
            runpy.run_path(script, run_name='__main__')
        except SystemExit as e:
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
                return 1
            return e.code or 0
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == '--build-fsr411':
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.path.insert(0, str(PACKAGE_DIR / 'tools' / 'fsr4cap'))
        os.environ.setdefault('BB_DATA_DIR', str(DATA_DIR))
        import build_assets_win
        return build_assets_win.main(sys.argv[2:])
    if '--play' in sys.argv[1:]:
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        s = load_settings()
        ini = read_ini()
        write_ini(ini)
        if not (Path(s.get('game_dir') or '') / 'eboot.bin').is_file():
            main_window()
            return 0
        return play(s)
    main_window()
    return 0


if __name__ == '__main__':
    sys.exit(main())
