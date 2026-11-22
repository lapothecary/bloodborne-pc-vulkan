# Bloodborne PC Vulkan (Windows & Linux)

<p align="center">
  <a href="https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest"><b>📦 Download Latest Release</b></a> •
  <a href="#-quick-start"><b>⚡ Quick Start</b></a> •
  <a href="#-key-features"><b>✨ Key Features</b></a> •
  <a href="#-more-details"><b>🔍 Comparisons & Specs</b></a> •
  <a href="#-документация-на-русском-языке-ru"><b>🇷🇺 На русском</b></a>
</p>

An optimization-focused **Windows 10/11** and **Linux (`x86_64`)** PC port for **Bloodborne** (`CUSA03173` / `CUSA00900` v1.09), built on top of [`bbport`](https://github.com/deadinside28/bloodborne_pc) and [`shadPS4`](https://github.com/shadps4-emu/shadPS4).

Instead of emulating raw PS4 GPU memory traps and heavy legacy shaders on every frame, this project refactors the core graphics pipeline of FromSoftware's **Dantelion2** engine into native **C++20** and **Vulkan 1.3** (`DirectVk`) — delivering much higher frame rates, lower VRAM/RAM usage, stutter-free gameplay, and a native PC experience right inside Bloodborne's own menus.

---

## ⚡ Quick Start

> **Legal Notice:** This repository contains **zero** Sony or FromSoftware game assets or executables (`eboot.bin`). Provide your own legally dumped *Bloodborne* v1.09 folder (`CUSA03173` or `CUSA00900`; split `CUSAxxxxx` + `CUSAxxxxx-patch` folders are detected automatically).

| Platform | Download from [Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest) | How to Play |
| :--- | :--- | :--- |
| 🪟 **Windows 10/11 (`x86_64`)** | `bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip` | Extract next to `CUSA03173` and run **`Bloodborne.exe`** (or **`Launcher.exe`** for mods/updates/pre-bake) |
| 🐧 **Linux (`x86_64`)** | `bloodborne-pc-vulkan-linux-x86_64-prebuilt.tar.gz` | Extract next to `CUSA03173` and run **`./play.sh`** (or **`./launcher.sh`**) |

```text
BloodBorne/
├── CUSA03173/                      # Your dumped Bloodborne v1.09 folder (eboot.bin, dvdroot_ps4/, ...)
└── bloodborne-pc-vulkan/           # Extracted release folder:
    ├── Bloodborne.exe / play.sh    # Direct launch (zero Python in RAM during gameplay)
    ├── Launcher.exe / launcher.sh  # Lightweight native C++20 Launcher
    └── ...
```

**Handy In-Game Shortcuts:** `F4` cycles Camera Follow Mode (`Original` / `Balanced` / `Locked`) • `F3` cycles FPS cap • `F8` / `F9` toggles performance HUD • `Insert` opens the overlay • `Alt+Enter` toggles Fullscreen / Borderless (all rebindable in `System -> Controls`).

---

## ✨ Key Features

- **Authentic Native Launcher (Zero Bundled Game Assets)** — Fast, frameless C++20 launcher with 1-click SHA-256 updates, Mods, Online (`bbnet20`), and Shader Pre-Bake. On first run it decodes Bloodborne's own title art, official Victorian font (with full Cyrillic support), menu SFX, and ambient music (*Hunter's Dream* & *Omen*) straight from your local game files so it feels like an official native companion app.
- **Full-Game Shader Pre-Bake (`5,761` Shaders)** — Pre-compile all game shaders once in the launcher (or let the quiet background auto-baker finish while you play) so you don't hit shader-compile stutters during gameplay.
- **Silky-Smooth Frame Pacing & `1000+ Hz` Mouse Camera** — Overhauled frame/present pacing and event-driven mouse input remove camera stutter when looking around, making playthroughs feel crisp and responsive at 60–300+ FPS.
- **Smooth `Balanced` Camera Mode (`F4`)** — Inspired by *Lies of P*'s smooth camera feel and refined for Bloodborne: strafing, sprinting, and climbing stairs flow smoothly without jarring camera jerks. Switch live anytime between `Original (PS4)`, `Balanced`, and `Locked (1:1)`.
- **Complete In-Game PC Menu & Controls Refactor** — Rebind any keyboard/mouse key (including mouse wheel, alternate keys, and `Left Alt` Slow Walk) and adjust resolution, live VSync, upscalers, AA, and effects right inside Bloodborne's own `System` menu — complete with gothic scrollbars, instant mouse-wheel scrolling, and `Exit Game` on the Title Screen.
- **4–5× Faster Location & Respawn Loading** — Accelerated decompression and loading-screen pacing drop typical warps and respawns from ~11–12 s down to ~2–3 s, plus fast cached repeat startups.
- **Modern Upscalers & Optional Remaster Visuals** — Live switching for `DLSS 4.5` / `DLSS 4` / `DLAA` / `FSR 3.1 & 4` / `SMAA T2x`, free `RCAS` sharpening, `2×/4×` GPU UI & icon upscaling, fixed high-FPS foliage wind & blood splatters, plus 4 optional remaster toggles (`GTAO/SSAO`, `Contact Shadows`, `Adaptive SSR`, and `True Black / OLED`).
- **Single-Process Native Execution** — Only `Bloodborne.exe` / `bb-vulkan` stays in memory during gameplay with zero Python overhead and native Hunter's Mark taskbar icons on Windows & Linux Wayland.

---

> **Hardware Requirements:** Because this port executes Bloodborne's original PS4 (`AMD Jaguar x86_64`) code natively on your CPU, it requires a **Vulkan 1.3 GPU** and an **x86-64-v3 CPU** supporting `AVX`, `BMI1`, `MOVBE`, `LZCNT`, and `POPCNT` (**Intel Haswell 4th-gen Core 2013+** or **AMD Ryzen / Steam Deck**; older pre-2013 CPUs such as Ivy Bridge `i7-3770K` or Sandy Bridge are not supported).  
> **Linux Launcher Audio Tip (Arch / CachyOS):** Launcher BGM/SFX uses WebKitGTK + GStreamer — if launcher audio is silent on minimal installs, install `gst-plugins-base gst-plugins-good gst-plugin-pipewire`.

## 🆕 Latest Release (`v1.2.1`)

- **Fixed Linux `CPU ISA level is lower than required` & `Shader baker exited with code 127`** — Linux prebuilt releases are now compiled cleanly in CI as portable **`x86-64-v3`** (zero AVX-512 / `x86-64-v4`) with bundled distro-independent libraries (`out/gpu/`), working out of the box on Ryzen 1000–5000 (`5700X3D`/`5800X3D`), Intel 4th–14th gen / Core Ultra, and Steam Deck across CachyOS, Arch, Ubuntu, Fedora, Bazzite, and SteamOS.
- **Includes everything from `v1.2.0`** — +35–55% CPU FPS breakthrough (330–375+ FPS in Central Yharnam), 1.3 s death respawns, 57× faster Linux Shader Pre-Bake (~4 s), high-FPS curb & Havok Cloth fixes, and crisp 2×–4× UI upscaling.

*(See [Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases) for clean per-version changelogs.)*

---

## 🔍 More Details

<details>
<summary><b>1. 🆚 How Does This Compare to <code>shadPS4</code> and Upstream <code>bbport</code>?</b></summary>
<br/>

| Area | `shadPS4` | Upstream `bbport` | **`bloodborne-pc-vulkan`** |
| :--- | :--- | :--- | :--- |
| **Execution Model** | General-purpose PS4 emulator | Host runtime + PS4 graphics translation | `bbport` host runtime + **refactored C++20 Dantelion graphics engine** (`2,940` functions) wired into **DirectVk 1.3** |
| **VRAM & Memory** | Unified memory page traps | Shared host RAM + hot VRAM mirrors | **True split PC VRAM (`DEVICE_LOCAL`)** — `0` page faults/s in gameplay, much lower RAM & VRAM footprint |
| **Shaders & Stutter** | Runtime translation | Runtime translation + recipe cache | **Full `5,761`-shader pre-baker**, quiet background auto-baker, fast async shader fallback, and **`129` native Vulkan GLSL shaders** |
| **Performance Features** | Frame-rate patches | Two-stage draw pipe + Vulkan occlusion | **DirectVk command stream**, **3D occlusion culling (`27–41%` skipped)**, front-to-back G-Buffer sorting, light-tile/UI batching, multi-core L3 cache pinning |
| **PC Polish & UX** | External emulator UI | Upscalers (`DLSS`/`FSR`/`FG`), Ultrawide, raw mouse, basic in-game menus, online | **Full live in-game `Display`/`Effects`/`Controls` menus (28 actions)**, **smooth `Balanced` camera (`F4`)**, **4–5× faster loading**, **dual-stage frame pacing**, **4 optional remaster shaders**, **`2×/4×` UI upscaling**, **authentic C++20 native launcher**, and **1-click updates** |

</details>

<details>
<summary><b>2. 📊 Reference Performance (Ryzen 9 7950X + RTX 4090 @ 2560×1440 Native)</b></summary>
<br/>

> Performance varies by CPU, GPU, resolution, and scene. Measured at **2560×1440 Native (100%)** with all graphics effects enabled (`SSR`, `SSAO`, `Bloom`, `DoF`, `Motion Blur`, `Contact Shadows`):

| Scenario (`2560×1440` Native) | Frame Rate | Pure GPU Time | Per-Process VRAM |
| :--- | :---: | :---: | :---: |
| **Heavy Central Yharnam Vista** (uncapped / all FX on) | **340 – 370 FPS** *(or locked `170.0 FPS`, `0.01–0.15 ms` jitter)* | `2.48 – 2.64 ms` | `~3.1 – 3.9 GB` (`0` faults/s) |
| **Enclosed Areas (e.g. Hunter's Dream)** (uncapped) | **480 – 700+ FPS** | `1.31 – 1.95 ms` | `~2.6 – 3.1 GB` (`0` faults/s) |

</details>

<details>
<summary><b>3. 🛠️ Building from Source & Diagnostic Flags</b></summary>
<br/>

```bash
# 1. Build the refactored C++20 graphics engine (dantelion_gx):
cmake -S dantelion_gx -B dantelion_gx/build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C dantelion_gx/build

# 2. Build the Vulkan 1.3 RHI library (Linux):
ninja -C build_gpu && cp -f build_gpu/libbbgpu.so out/gpu/libbbgpu.so

# 3. Build the native C++20 WebView Launcher (Linux):
cmake -S launcher_v2 -B launcher_v2/build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C launcher_v2/build && cp -f launcher_v2/build/bb-launcher out/bb-launcher
```

**Useful Runtime Flags (No Rebuild Needed):**
```bash
BB_HW_OCCLUSION_CULL=0 ./play.sh        # Disable 3D hardware occlusion culling
BB_NATIVE_SHADERS=0 ./play.sh           # Fall back from native GLSL to translated SPIR-V
BB_GX_SKIP_MODS="05,06" ./play.sh       # Skip specific C++20 graphics modules for debugging
```

</details>

<details>
<summary><b>4. 🇷🇺 Документация на русском языке (RU)</b></summary>
<br/>

### О проекте
Оптимизированный ПК-порт **Bloodborne** (`CUSA03173` / `CUSA00900` v1.09) для **Windows 10/11** и **Linux (`x86_64`)**, созданный на базе [`bbport`](https://github.com/deadinside28/bloodborne_pc) и [`shadPS4`](https://github.com/shadps4-emu/shadPS4). Вместо эмуляции тяжёлого PS4-рендера на каждом кадре критическая графическая часть движка **Dantelion2** переписана в нативный **C++20** и **Vulkan 1.3** (`DirectVk`), а самые тяжёлые шейдеры заменены чистыми Vulkan-шейдерами.

### Быстрый старт (30 секунд)
1. Скачайте архив для **Windows** (`.zip`) или **Linux** (`.tar.gz`) из раздела **[Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest)**.
2. Распакуйте его рядом со своей папкой игры **`CUSA03173`** (`v1.09`).
3. Запустите **`Bloodborne.exe`** / **`Launcher.exe`** (Windows) или **`./play.sh`** / **`./launcher.sh`** (Linux).

### Что делает этот порт удобным для игры
- **Аутентичный нативный лаунчер** — быстрый C++20 лаунчер с автообновлением в 1 клик, модами, онлайном и предкомпиляцией шейдеров. Не хранит защищённых файлов Sony: при первом запуске сам декодирует прямо из вашей папки игры родной арт, официальный шрифт Bloodborne (с кириллицей), звуки меню и музыку *Hunter's Dream* / *Omen*.
- **Полный пре-бейк всех `5,761` шейдеров** — один раз прогоняется в лаунчере (или тихо в фоне во время игры), чтобы убрать статтеры при компиляции шейдеров.
- **Плавная камера и фреймпейсинг** — событийный опрос мыши `1000+ Гц`, ровный фреймпейсинг без рывков при вращении камеры и новый режим плавной камеры **`Balanced`** (`F4`) в духе *Lies of P* (переключается на лету между `Original`, `Balanced` и `Locked`).
- **Полноценные ПК-настройки прямо в игре** — переназначение любых клавиш и мыши, живой `VSync`, разрешение, апскейлеры (`DLSS 4.5` / `DLAA` / `FSR 3.1 & 4` / `SMAA T2x`), скроллбары, мгновенная прокрутка колесом и `Exit Game` прямо в главном меню.
- **Загрузки в 4–5 раз быстрее** — переходы и возрождения занимают ~2–3 секунды вместо 11–12, плюс ускоренный повторный запуск игры.
- **Опциональные ремастер-улучшения** — `GTAO/SSAO`, контактные микро-тени, адаптивный `SSR`, `True Black / OLED` и `2×/4×` апскейл интерфейса и иконок.

</details>

---

## 📜 Credits & License

- **[`bbport` (`deadinside28/bloodborne_pc`)](https://github.com/deadinside28/bloodborne_pc)** — Host ELF/PE loader, PS4 kernel/system runtime, Windows port foundation, upscaler bridges (`DLSS`, `FSR 3.1/4`, `Frame Gen`), Ultrawide, Havok cloth fixes, raw mouse look, and online (`bbnet20`).
- **[`shadPS4` (`shadps4-emu/shadPS4`)](https://github.com/shadps4-emu/shadPS4)** — Original PS4 Vulkan video core and GCN-to-SPIR-V shader recompiler foundation (`GPL-2.0-or-later`).
- **[`yharnam-pc` (`Jgamrot/yharnam-pc`)](https://github.com/Jgamrot/yharnam-pc)** — UX layout inspiration for the slim-rail launcher and first-time setup flow (`GPL-2.0`).
- **`bloodborne-pc-vulkan` ([@lapothecary](https://github.com/lapothecary))** — Refactored C++20 Dantelion graphics engine, DirectVk split VRAM & occlusion/batching pipeline, `129` native Vulkan shaders, full `5,761`-shader pre-baker, frame pacing & `Balanced` camera, expanded live in-game menus, 4–5× fast loading, optional remaster shaders, and the native C++20 runtime-decoding launcher.
- **License:** **[GNU General Public License v3.0](LICENSE)** (`GPL-3.0`).
