#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Starts the game on Windows (run.sh is the Linux one; this follows it step for step).

The game's files are prepared (scripts/prepare.py, link_libc.py, link_modules.py,
content_profile.py, patches.py) into out/, then bin/bb-probe.exe runs the original eboot. Everything
the port writes lives next to it (portable): bbport.ini, user/ (saves), out/, logs/.

Environment: BB_GAME_DIR (the game's folder, with eboot.bin), BB_DATA_DIR (writable folder for the
port's files; default: this folder), BB_FPS (uncap, 90, 60, 30), BB_SAVE_LOG=1 (frame statistics
in logs/ too), and the BB_* variables of run.sh. The launcher (launcher/bbport_launcher_win.py)
calls run() in its own process when frozen into Bloodborne.exe, so no Python is installed.
"""
import contextlib
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

PORT_DIR = Path(getattr(sys, '_MEIPASS', '')) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
# A frozen launcher keeps the scripts next to Bloodborne.exe (the package's folder).
PACKAGE_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else PORT_DIR
SCRIPTS = PACKAGE_DIR / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def script(name, *args):
    """Runs scripts/<name> with args; its standard output (some print a result) is returned."""
    path = SCRIPTS / name
    if getattr(sys, 'frozen', False):
        out = io.StringIO()
        argv = sys.argv
        sys.argv = [str(path), *map(str, args)]
        try:
            with contextlib.redirect_stdout(out):
                runpy.run_path(str(path), run_name='__main__')
        except SystemExit as stop:
            if stop.code not in (None, 0):
                raise RuntimeError(f'{name} failed ({stop.code}): {out.getvalue().strip()}') from None
        finally:
            sys.argv = argv
        text = out.getvalue()
    else:
        done = subprocess.run([sys.executable, str(path), *map(str, args)], stdout=subprocess.PIPE,
                              text=True, encoding='utf-8')
        text = done.stdout
        if done.returncode != 0:
            raise RuntimeError(f'{name} failed ({done.returncode}): {text.strip()}')
    for line in text.splitlines():
        if line.strip():
            log(f'  {line}')
    return text


_log_file = None


def log(line):
    if sys.stdout:  # a windowed Bloodborne.exe has none
        print(line, flush=True)
    if _log_file:
        _log_file.write(line + '\n')
        _log_file.flush()


def setdefault(env, name, value):
    if not env.get(name):
        env[name] = value


def online(env, user, bindir):
    """Online play (BB_ONLINE=1, the launcher's Online section), as run.sh sets it up: the online
    module online/bbnet20.dll (BB_NET_MODULE overrides it) and the addresses of the pool: Seamless
    Hunt (the shadNet server and its WebAPI, the game server in host_overrides.json) or The
    Hunter's Dream (BB_NET_BACKEND=thd: the account from the launcher's Discord sign-in, its
    overrides). BBSC's local game server fallback is Linux-only for now."""
    if env.get('BB_ONLINE') != '1':
        return
    module = Path(env.get('BB_NET_MODULE') or PACKAGE_DIR / 'online' / 'bbnet20.dll')
    if not module.is_file():
        log(f'Online: no {module}: offline play')
        env.pop('BB_ONLINE', None)
        return
    env['BB_NET_MODULE'] = str(module)
    if env.get('BB_NET_BACKEND') == 'thd':
        env.pop('BB_BBSC', None)
        if not env.get('BB_THD_TOKEN') or not env.get('BB_THD_ONLINE_ID'):
            raise RuntimeError("The Hunter's Dream: not signed in (launcher: Online -> Sign in with Discord)")
        os.environ.update(env)
        lines = script('online.py', 'thd-overrides', '--user', user).strip().splitlines()
        if len(lines) < 2:
            raise RuntimeError("The Hunter's Dream: no host overrides")
        env['SHADPS4_HTTP_HOST_OVERRIDES_JSON'] = lines[-2].strip()
        base = lines[-1].strip()
        setdefault(env, 'BB_THD_SERVER', base + ':18671')
        setdefault(env, 'BB_THD_AUTH', base)
        for name in ('BB_SHADNET_SERVER', 'BB_SHADNET_WEBAPI', 'BB_SHADNET_NPID', 'BB_SHADNET_PASSWORD',
                     'BB_SHADNET_TOKEN'):
            env.pop(name, None)
            os.environ.pop(name, None)
        log(f"Online: {module.name}, The Hunter's Dream {env['BB_THD_SERVER']} as {env['BB_THD_ONLINE_ID']}")
        return
    setdefault(env, 'BB_SHADNET_SERVER', 'play.seamlesshunt.com:31313')
    setdefault(env, 'BB_SHADNET_WEBAPI', 'http://' + env['BB_SHADNET_SERVER'].rsplit(':', 1)[0] + ':31315')
    os.environ.update(env)
    if env.get('BB_BBSC') == '1':
        bbsc_local_server(env, user)
    try:
        path = script('online.py', 'overrides', '--user', user).strip().splitlines()
    except RuntimeError as error:
        log(f'Online: no host overrides ({error})')
        path = []
    if path:
        env['SHADPS4_HTTP_HOST_OVERRIDES_JSON'] = path[-1].strip()
    log(f'Online: {module.name}, shadNet {env["BB_SHADNET_SERVER"]}, WebAPI {env["BB_SHADNET_WEBAPI"]}'
        + (', BBSC' if env.get('BB_BBSC') == '1' else ''))


_bbsc_server = None
_mod_games = []


def bbsc_local_server(env, user):
    """BBSC: its game server on this PC when the server's own does not answer (online.py
    bbsc-local; run.sh does the same): scripts/bbsc_gameserver run by this interpreter (the frozen
    bbport-run.exe --run-script, or Python), for as long as the game runs."""
    global _bbsc_server
    server_script = SCRIPTS / 'bbsc_gameserver' / 'bbsc_gameserver.py'
    if not server_script.is_file():
        return
    try:
        shadnet = script('online.py', 'bbsc-local').strip()
    except RuntimeError:
        shadnet = ''
    if not shadnet:
        return
    import socket
    try:
        socket.create_connection(('127.0.0.1', 20443), timeout=1).close()
        log('Co-op: a game server answers on 127.0.0.1:20443 already: used')
    except OSError:
        folder = Path(user) / 'bbsc'
        (folder / 'logs').mkdir(parents=True, exist_ok=True)
        if getattr(sys, 'frozen', False):
            command = [sys.executable, '--run-script', str(server_script)]
        else:
            command = [sys.executable, str(server_script)]
        command += ['--shadnet', shadnet, '--host', '127.0.0.1', '--port', '20443', '--relay-port', '0',
                    '--no-bbhost', '--no-playlog', '--data', str(folder / 'online-data.json'),
                    '--log-dir', str(folder / 'logs')]
        console = open(folder / 'logs' / 'gameserver-console.log', 'w', encoding='utf-8', errors='replace')
        try:
            _bbsc_server = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                                            cwd=str(server_script.parent), creationflags=NO_WINDOW)
        except OSError as error:
            log(f'Co-op: the local BBSC game server did not start ({error})')
            return
        finally:
            console.close()
        log(f'Co-op: BBSC game server on 127.0.0.1:20443 (summons -> {shadnet}:31315)')
    env['BB_COMMUNITY_SERVER'] = 'http://127.0.0.1'
    os.environ['BB_COMMUNITY_SERVER'] = env['BB_COMMUNITY_SERVER']


def stop_bbsc_local_server():
    global _bbsc_server
    if _bbsc_server is not None:
        with contextlib.suppress(OSError):
            _bbsc_server.terminate()
        with contextlib.suppress(Exception):
            _bbsc_server.wait(timeout=5)
        _bbsc_server = None


def large_scene_memory(env, scene):
    """More direct memory only for the game's larger heaps (a scene above 1920x1080, see run.sh)."""
    width, _, height = scene.partition('x')
    if width.isdigit() and height.isdigit() and int(width) * int(height) > 1920 * 1080:
        setdefault(env, 'BB_DMEM_MB', '9152')


def _discover_game_dir(base_dir):
    candidates = [
        base_dir / 'CUSA03173',
        base_dir / 'game',
        base_dir.parent / 'CUSA03173',
        base_dir.parent / 'game',
        base_dir.parent / 'Bloodborne' / 'CUSA03173',
        base_dir.parent / 'Bloodborne' / 'game',
    ]
    for c in candidates:
        if (c / 'eboot.bin').is_file():
            return str(c.resolve())
    for parent in (base_dir, base_dir.parent):
        try:
            for sub in parent.iterdir():
                if sub.is_dir():
                    if (sub / 'eboot.bin').is_file():
                        return str(sub.resolve())
                    for sub2 in ('CUSA03173', 'game', 'dvdroot_ps4'):
                        p2 = sub / sub2
                        if (p2 / 'eboot.bin').is_file():
                            return str(p2.resolve())
        except OSError:
            pass
    return ''


def _load_saved_launcher_and_ini(env, data):
    """Loads saved settings from launcher-win.json and bbport.ini so running Bloodborne.exe
    directly applies all settings saved by Launcher.py or changed live in-game."""
    import json
    ini_path = Path(env.get('BB_CONFIG') or data / 'bbport.ini')
    ini = {}
    ini_lines = []
    if ini_path.is_file():
        try:
            ini_lines = ini_path.read_text(encoding='utf-8', errors='replace').splitlines()
            for line in ini_lines:
                if '=' in line and not line.lstrip().startswith('#'):
                    k, v = line.split('=', 1)
                    ini[k.strip()] = v.strip()
        except OSError:
            pass
    else:
        default_ini = (
            "# bbport settings (in-game menu: Insert / L3+R3)\n"
            "aa_mode=smaa_t2x\n"
            "upscaler=off\n"
            "preset=0\n"
            "dlss_preset=11\n"
            "anisotropic=16\n"
            "ui_upscale=1\n"
            "fps_limit=0\n"
            "sharpen=1\n"
            "sharpness=0.50\n"
            "object_motion=0\n"
            "show_fps=1\n"
            "aspect_ratio=auto\n"
            "output_res=auto\n"
            "window_mode=fullscreen\n"
            "fullscreen=1\n"
            "model_lod=0\n"
            "live_resolution=0\n"
            "effect_game_aa=0\n"
            "mouse_look=1\n"
            "mouse_sensitivity=1.00\n"
            "mouse_invert_y=0\n"
            "skip_intro=1\n"
            "skip_network_choice=1\n"
        )
        with contextlib.suppress(OSError):
            ini_path.write_text(default_ini, encoding='utf-8')
            for line in default_ini.splitlines():
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    ini[k.strip()] = v.strip()

    s = {}
    json_candidates = [
        data / 'launcher-win.json',
        Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'bb_vulkan-launcher' / 'settings.json',
    ]
    if os.environ.get('APPDATA'):
        json_candidates.append(Path(os.environ['APPDATA']) / 'bbport-launcher' / 'settings.json')
    for jf in json_candidates:
        if jf.is_file():
            with contextlib.suppress(OSError, ValueError):
                s = json.loads(jf.read_text(encoding='utf-8'))
                break

    # In-game menu writes directly to bbport.ini; let bbport.ini override launcher json for shared keys
    game_dir = env.get('BB_GAME_DIR') or ini.get('game_dir') or s.get('game_dir') or ''
    if not game_dir or not (Path(game_dir) / 'eboot.bin').is_file():
        discovered = _discover_game_dir(PACKAGE_DIR)
        if discovered:
            game_dir = discovered
    if game_dir:
        setdefault(env, 'BB_GAME_DIR', str(game_dir))
    if s.get('user_dir'):
        setdefault(env, 'BB_USER_DIR', str(s['user_dir']))
    if s.get('mods_dir'):
        setdefault(env, 'BB_MODS_DIR', str(s['mods_dir']))
    setdefault(env, 'BB_MODS_CONFIG', str(data / 'mods.json'))
    setdefault(env, 'BB_MODS_ENABLED', '0' if s.get('mods_enabled') is False else '1')
    if s.get('patches_dir'):
        setdefault(env, 'BB_PATCHES_DIR', str(s['patches_dir']))
    setdefault(env, 'BB_PATCHES_CONFIG', str(data / 'patches.json'))

    lang = ini.get('language') or str(s.get('language', '1'))
    setdefault(env, 'BB_LANGUAGE', lang)
    setdefault(env, 'BB_MENU_LANGUAGE', 'en')

    window_mode = ini.get('window_mode') or s.get('window_mode')
    fullscreen = ini.get('fullscreen')
    if window_mode in ('fullscreen', 'borderless', 'windowed'):
        setdefault(env, 'BB_WINDOW_MODE', window_mode)
        setdefault(env, 'BB_FULLSCREEN', '0' if window_mode == 'windowed' else '1')
    else:
        if fullscreen is None:
            fullscreen = '1' if s.get('fullscreen', True) else '0'
        setdefault(env, 'BB_FULLSCREEN', fullscreen)

    present_mode = ini.get('present_mode') or s.get('present_mode') or 'Mailbox'
    setdefault(env, 'BB_PRESENT_MODE', present_mode)

    if s.get('gamepad'):
        setdefault(env, 'BB_GAMEPAD', str(s['gamepad']))
    if s.get('display'):
        setdefault(env, 'BB_DISPLAY', str(s['display']))

    skip_net = ini.get('skip_network_choice')
    if skip_net is None:
        skip_net = '0' if s.get('skip_network_choice') is False else '1'
    setdefault(env, 'BB_SKIP_NETWORK_CHOICE', skip_net)

    if s.get('hdr'):
        setdefault(env, 'BB_HDR', '1')

    fps_limit = ini.get('fps_limit')
    if fps_limit is None:
        fps_limit = str(s.get('fps_limit', 0) or 0)
    setdefault(env, 'BB_FPS', 'uncap')
    setdefault(env, 'BB_FPS_LIMIT', str(fps_limit))

    if s.get('draw_pipe'):
        setdefault(env, 'BB_DRAW_PIPE', str(s['draw_pipe']))
    setdefault(env, 'BB_HW_OCCLUSION_CULL', '0' if s.get('hw_occlusion_cull') is False else '1')
    setdefault(env, 'BB_MICRO_SHADOW_CULL', '0' if s.get('hw_occlusion_cull') is False else '1')
    setdefault(env, 'BB_NATIVE_SHADERS', '0' if s.get('native_shaders') is False else '1')
    if s.get('save_log'):
        setdefault(env, 'BB_SAVE_LOG', '1')
    if s.get('gpu_profile'):
        setdefault(env, 'BB_GPU_PROFILE', '1')
    if s.get('vk_validation'):
        setdefault(env, 'BB_VK_VALIDATION', '1')
    for item in str(s.get('extra_env', '')).split():
        if '=' in item:
            k, v = item.split('=', 1)
            setdefault(env, k, v)


def _prompt_game_dir_once(data, ini_path):
    """If Bloodborne.exe is launched directly and eboot.bin cannot be found, ask once via a native folder dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Bloodborne PC Vulkan",
            "Please select your Bloodborne game folder (the folder containing eboot.bin, e.g. CUSA03173 or game)."
        )
        chosen = filedialog.askdirectory(title="Select Bloodborne folder (containing eboot.bin)")
        root.destroy()
        if not chosen:
            return ''
        p = Path(chosen)
        for cand in (p, p / 'CUSA03173', p / 'game', p / 'dvdroot_ps4'):
            if (cand / 'eboot.bin').is_file():
                resolved = str(cand.resolve())
                # Save into launcher-win.json and bbport.ini so future launches remember it
                import json
                jf = data / 'launcher-win.json'
                s = {}
                if jf.is_file():
                    with contextlib.suppress(OSError, ValueError):
                        s = json.loads(jf.read_text(encoding='utf-8'))
                s['game_dir'] = resolved
                with contextlib.suppress(OSError):
                    jf.write_text(json.dumps(s, indent=2), encoding='utf-8')
                with contextlib.suppress(OSError):
                    txt = ini_path.read_text(encoding='utf-8') if ini_path.is_file() else ''
                    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith('game_dir=')]
                    lines.append(f'game_dir={resolved}')
                    ini_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                return resolved
    except Exception:
        pass
    return ''


def run(extra_args=(), env=None, on_line=None):
    """Prepares and runs the game; the exit code. on_line(text) gets every line of output."""
    global _log_file
    env = dict(os.environ if env is None else env)
    # A test build's diagnostics: test-env.txt next to the package, KEY=VALUE lines (# comments),
    # under what the player set.
    test_env = PACKAGE_DIR / 'test-env.txt'
    test_keys = []
    if test_env.is_file():
        for line in test_env.read_text(encoding='utf-8', errors='replace').splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and not key.startswith('#'):
                test_keys.append(key.strip())
                setdefault(env, key.strip(), value.strip())
    data = Path(env.get('BB_DATA_DIR') or PACKAGE_DIR)
    out = data / 'out'
    out.mkdir(parents=True, exist_ok=True)
    setdefault(env, 'BB_CONFIG', str(data / 'bbport.ini'))
    _load_saved_launcher_and_ini(env, data)
    user = Path(env.get('BB_USER_DIR') or data / 'user')
    setdefault(env, 'BB_GPU_USER_DIR', str(user))
    # Every run is logged on Windows (an issue report needs it): logs/<time>.log.
    logs = data / 'logs'
    logs.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    _log_file = open(logs / f'{stamp}.log', 'w', encoding='utf-8', errors='replace')
    if env.get('BB_SAVE_LOG') == '1':
        env['BB_FRAME_STATS'] = '1'
        if env.get('BB_CSV_LOG') == '1':
            setdefault(env, 'BB_FRAME_LOG', str(logs / f'{stamp}.frames.csv'))
    elif sys.platform == 'win32':
        # Windows builds: log lightweight 10 s summaries and slow image creations (20 ms+);
        # heavy per-frame .frames.csv is opt-in via BB_CSV_LOG=1 (matching run.sh).
        if env.get('BB_FRAME_STATS') != 'off' and env.get('BB_SAVE_LOG') != '0':
            setdefault(env, 'BB_FRAME_STATS', '1')
            if env.get('BB_CSV_LOG') == '1':
                setdefault(env, 'BB_FRAME_LOG', str(logs / f'{stamp}.frames.csv'))
            setdefault(env, 'BB_SLOW_BIND_LOG', '20000')
        elif env.get('BB_FRAME_STATS') == 'off':
            del env['BB_FRAME_STATS']
    if sys.platform == 'win32':
        # The FSR 4 v07 set the package brings, and the FSR 4.1.1 set in the data folder (the
        # launcher copies a built one there), as run.sh looks them up: the game runs in the data
        # folder, which BB_DATA_DIR may move away from the package.
        if (PACKAGE_DIR / 'fsr4_shaders').is_dir():
            setdefault(env, 'BB_FSR4_DIR', str(PACKAGE_DIR / 'fsr4_shaders'))
        if (data / 'fsr4_411').is_dir():
            setdefault(env, 'BB_FSR411_DIR', str(data / 'fsr4_411'))
    log(f'Log: {logs / f"{stamp}.log"}')
    if (PACKAGE_DIR / 'VERSION').is_file():  # issue #307
        log(f"bbport {(PACKAGE_DIR / 'VERSION').read_text().strip()}")
    if test_keys:
        log(f'Test settings ({test_env.name}): ' + ' '.join(f'{key}={env[key]}' for key in test_keys))
    try:
        game = Path(env.get('BB_GAME_DIR') or PACKAGE_DIR.parent / 'CUSA03173')
        import game_check
        base_cand, patch_cand = game_check.resolve_patch_pair(game)
        if not (game / 'eboot.bin').is_file() and (base_cand / 'eboot.bin').is_file():
            game = base_cand
        if not (game / 'eboot.bin').is_file():
            chosen = _prompt_game_dir_once(data, Path(env['BB_CONFIG']))
            if chosen:
                env['BB_GAME_DIR'] = chosen
                game = Path(chosen)
                base_cand, patch_cand = game_check.resolve_patch_pair(game)
                if not (game / 'eboot.bin').is_file() and (base_cand / 'eboot.bin').is_file():
                    game = base_cand
        if not (game / 'eboot.bin').is_file():
            log(f'No eboot.bin in {game} (choose the game folder in Launcher.py).')
            return 1
        if patch_cand is not None:
            log(f'Game: {base_cand} + update {patch_cand}')
        original_game = game  # the game's own files (shader recipes read program code there)
        os.environ.update(env)  # the scripts read BB_* from the environment
        game = Path(script('mods.py', game, '--out', out, '--mods-dir', env.get('BB_MODS_DIR', data / 'mods'),
                           '--config', env.get('BB_MODS_CONFIG', data / 'mods.json'),
                           '--enabled', env.get('BB_MODS_ENABLED', '1'),
                           '--saves', user / 'savedata', '--state', data / 'mods.last').strip().splitlines()[-1])
        if game != original_game and game.name.startswith('mod-game-'):
            _mod_games.append(game)  # the game with the mods' files linked in: removed at the end
        script('prepare.py', game, '--out', out)
        script('link_libc.py', game, '--out', out)
        script('link_modules.py', game, '--out', out)
        script('content_profile.py', game, '--out', out, '--sku', env.get('BB_CONTENT_SKU', 'full'))
        if env.get('BB_AUTO_RENDER_RES') == '1':
            for name in ('BB_RENDER_RES', 'BB_OUTPUT_RES', 'BB_AUTO_RENDER_RES'):
                env.pop(name, None)
        fps = env.get('BB_FPS', 'uncap')
        bindir = PACKAGE_DIR / 'bin'
        caps = bindir / 'bb-gpu-capabilities.exe'
        scaled_render = scaled_output = ''
        ini_lines = []
        if Path(env['BB_CONFIG']).is_file():
            ini_lines = Path(env['BB_CONFIG']).read_text(encoding='utf-8', errors='replace').splitlines()
        ini = dict(line.split('=', 1) for line in ini_lines if '=' in line and not line.startswith('#'))
        # output_res=auto (and no key, the default): the monitor's resolution (BB_DISPLAY: its name or
        # number, else the primary one).
        if ini.get('output_res', 'auto') == 'auto' and not env.get('BB_MONITOR_SIZE'):
            try:
                rows = subprocess.run([str(caps), '--displays'], stdout=subprocess.PIPE, text=True, env=env,
                                      creationflags=NO_WINDOW).stdout.splitlines()
            except OSError:
                rows = []
            wanted = env.get('BB_DISPLAY', '')
            for number, row in enumerate(rows, 1):
                parts = row.split('\t')
                if len(parts) == 3 and ((wanted and wanted in (parts[0], str(number))) or
                                        (not wanted and parts[2] == '1')):
                    env['BB_MONITOR_SIZE'] = parts[1]
            if env.get('BB_MONITOR_SIZE'):
                log(f'Output: the monitor\'s resolution, {env["BB_MONITOR_SIZE"]} (output_res=auto)')
        # patches.py reads BB_MONITOR_SIZE from the environment: the output stayed 1920x1080 on any
        # other monitor, stretched over an ultrawide one (3440x1440, 2026-10-27).
        os.environ.update(env)
        if not env.get('BB_RENDER_RES'):
            printed = script('patches.py', '--print-scaled', '--settings', env['BB_CONFIG']).split()
            if len(printed) == 2:
                scaled_render, scaled_output = printed
        # Match run.sh + play.sh: respect `live_resolution` from bbport.ini (default 0 = native
        # startup patch with zero RunUiOnly proxy overhead) instead of forcing BB_LIVE_HYBRID=1.
        live = env.get('BB_LIVE_RES', '')
        if not live:
            live = ini.get('live_resolution', '0').strip()
            if live == 'auto':
                try:
                    live = subprocess.run([str(caps), '--live-resolution'], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, env=env,
                                          creationflags=NO_WINDOW).stdout.strip() or '0'
                except OSError:
                    live = '0'
        live = '1' if live == '1' else '0'
        if live == '1' and not scaled_output and not env.get('BB_RENDER_RES') and env.get('BB_LIVE_HYBRID', '1') == '1':
            printed = script('patches.py', '--print-scaled', '--align', '8', '--settings', env['BB_CONFIG']).split()
            if len(printed) == 2:
                scaled_render, scaled_output = printed
                env['BB_LIVE_RES'] = '1'
        hybrid = live == '1' and bool(scaled_output) and env.get('BB_LIVE_HYBRID', '1') == '1'
        if hybrid:
            # Live preset changes over the startup patch (see run.sh), the scene a multiple of 8.
            scaled_render, scaled_output = script('patches.py', '--print-scaled', '--align', '8', '--settings',
                                                  env['BB_CONFIG']).split()
            env.update(BB_RENDER_RES=scaled_render, BB_OUTPUT_RES=scaled_output, BB_AUTO_RENDER_RES='1',
                       BB_LIVE_HYBRID='1')
            large_scene_memory(env, scaled_render)
            log(f'Output {scaled_output}: scene {scaled_render}, live preset changes over it')
        elif live == '1' and scaled_output:
            log(f'Output {scaled_output}: live resolution changes (BB_LIVE_HYBRID=0: proxies only)')
        elif scaled_output:
            env.update(BB_RENDER_RES=scaled_render, BB_OUTPUT_RES=scaled_output, BB_AUTO_RENDER_RES='1')
            large_scene_memory(env, scaled_render)
            log(f'Output {scaled_output}: scene {scaled_render}, direct memory {env.get("BB_DMEM_MB") or 9152} MiB')
        # Texture packs fill the game's graphics heaps (#240, see run.sh): the full larger heaps
        # and their direct memory. BB_LARGE_HEAPS=1/0 forces it.
        large_heaps = env.get('BB_LARGE_HEAPS', '')
        pack = ''
        if large_heaps not in ('0', '1'):
            try:
                found = script('game_check.py', '--texture-pack', game, out / 'texture-pack.json').split(maxsplit=1)
            except RuntimeError:
                found = []
            large_heaps = found[0] if found and found[0] == '1' else '0'
            pack = found[1].strip() if len(found) > 1 else ''
        if large_heaps == '1':
            setdefault(env, 'BB_DMEM_MB', '9152')
            log(f'Graphics heaps: larger ({pack or "BB_LARGE_HEAPS=1"}), direct memory {env["BB_DMEM_MB"]} MiB '
                '(BB_LARGE_HEAPS=0: the game\'s own sizes)')
        os.environ.update(env)
        script('patches.py', '--out', out, '--fps', fps, '--extra', env.get('BB_PATCHES', ''),
               '--settings', env['BB_CONFIG'], '--game-dir', game, '--render-res', env.get('BB_RENDER_RES', ''),
               '--output-res', env.get('BB_OUTPUT_RES', ''), '--patches-dir', env.get('BB_PATCHES_DIR', data / 'patches'),
               '--patches-config', env.get('BB_PATCHES_CONFIG', data / 'patches.json'), '--large-heaps', large_heaps)
        setdefault(env, '__GL_SHADER_DISK_CACHE', '1')
        setdefault(env, '__GL_SHADER_DISK_CACHE_SKIP_CLEANUP', '1')
        setdefault(env, '__GL_SHADER_DISK_CACHE_SIZE', '10737418240')
        setdefault(env, 'BB_PC_MODEL', '0')
        setdefault(env, 'BB_GUEST_IN_PLACE', env['BB_PC_MODEL'])
        setdefault(env, 'BB_PREUPLOAD', '2')
        setdefault(env, 'BB_GX_MODULES', 'all')
        setdefault(env, 'BB_HW_OCCLUSION_CULL', '1')
        setdefault(env, 'BB_MICRO_SHADOW_CULL', '1')
        setdefault(env, 'BB_NATIVE_SHADERS', '1')
        setdefault(env, 'BB_READBACKS', '0')
        setdefault(env, 'BB_BREADCRUMBS', '0')
        setdefault(env, 'BB_INDIRECT_GUARD', '0')
        setdefault(env, 'BB_OBJECT_MOTION', '0')
        setdefault(env, 'BB_LIVE_RES', live)
        setdefault(env, 'BB_COPY_GPU_BUFFERS', '1')
        setdefault(env, 'BB_GPU_WRITE_TWINS', '1')
        setdefault(env, 'BB_GPU_WRITE_TWINS_MAX', '65536')
        setdefault(env, 'BB_VBLANK_HZ', {'uncap': '1000', '90': '90'}.get(fps, '60'))
        setdefault(env, 'BB_FPS', fps)
        # Shader recipes (gpu/render/shader_rebuild.h): the shaders of the recorded game, translated
        # for this GPU by a background process at start, the code read from the game's own files.
        if 'BB_SHADER_RECIPES' not in env and (PACKAGE_DIR / 'shader_recipes.bin').is_file():
            env['BB_SHADER_RECIPES'] = str(PACKAGE_DIR / 'shader_recipes.bin')
        # Image shapes (gpu/textures/image_spares.h): spare textures made ahead from the first session on.
        if 'BB_IMAGE_SHAPES' not in env and (PACKAGE_DIR / 'image_shapes.txt').is_file():
            env['BB_IMAGE_SHAPES'] = str(PACKAGE_DIR / 'image_shapes.txt')
        setdefault(env, 'BB_SHADER_GAME_DIR', str(game))
        online(env, user, bindir)
        probe = Path(env.get('BB_PROBE') or bindir / 'bb-probe.exe')
        args = [str(probe), str(out / 'boot-linked.bin'), '--content-profile', str(out / 'content.bin'),
                '--patches', str(out / 'patches.bin'), '--app0', str(game), '--user', str(user),
                '--timeout', env.get('BB_TIMEOUT', '0'), *extra_args]
        log('Starting ' + ' '.join(args))
        with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              cwd=str(data), creationflags=NO_WINDOW) as game_process:
            for raw in game_process.stdout:
                line = raw.decode('utf-8', 'replace').rstrip('\r\n')
                log(line)
                if on_line:
                    on_line(line)
            code = game_process.wait()
        log(f'— the game exited (code {code}) —')
        return code
    except (OSError, RuntimeError) as error:
        log(f'STOP: {error}')
        return 1
    finally:
        stop_bbsc_local_server()
        # As run.sh: the linked game folder goes (links and junctions only, never what they point to).
        while _mod_games:
            shutil.rmtree(_mod_games.pop(), ignore_errors=True)
        _log_file.close()
        _log_file = None


NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--run-script':
        target_script = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_path(target_script, run_name='__main__')
        sys.exit(0)
    if Path(sys.executable).stem.lower().startswith('launcher') or '--launcher' in sys.argv[1:]:
        runpy.run_path(str(PACKAGE_DIR / 'Launcher.py'), run_name='__main__')
        sys.exit(0)
    sys.exit(run(sys.argv[1:]))
