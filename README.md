# Bloodborne PC Port (Vulkan 1.3) — Refactored C++20 Graphics Pipeline (Windows & Linux)

<p align="center">
  <a href="https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest"><b>📦 Download Latest Prebuilt Release (Windows & Linux)</b></a> •
  <a href="#-quick-start-windows--linux"><b>⚡ Quick Start</b></a> •
  <a href="#-clickable-sections--technical-details"><b>🔍 Comparisons & Architecture</b></a> •
  <a href="#-документация-на-русском-языке-ru"><b>🇷🇺 На русском</b></a>
</p>

An optimization-focused PC port and translation layer for **Bloodborne** (`CUSA03173` / `CUSA00900` v1.09) on **Windows 10/11** and **Linux (`x86_64`)**, built on top of [`bbport`](https://github.com/deadinside28/bloodborne_pc) and [`shadPS4`](https://github.com/shadps4-emu/shadPS4).

Instead of translating raw PS4 GPU command buffers and unified-memory page faults on every draw, this project refactors the critical graphics bottleneck of FromSoftware's **Dantelion2** engine (`FD4` / `GX` / `Yebis 2` rendering modules and core shaders) into native **C++20** and **Vulkan 1.3** (`DirectVk`), while keeping the original gameplay, AI, Havok physics, and PS4 runtime layer intact.

---

## ⚡ Quick Start (Windows & Linux)

> **Legal Notice:** This repository contains **no** proprietary Sony or FromSoftware game assets or executables (`eboot.bin`). You must provide your own legally dumped *Bloodborne* v1.09 folder (`CUSA03173` or `CUSA00900`; split `CUSA03173` + `CUSA03173-patch` folders are also supported automatically).

| Platform | Archive from [Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest) | How to Launch |
| :--- | :--- | :--- |
| 🪟 **Windows 10/11 (`x86_64`)** | `bloodborne-pc-vulkan-windows-x86_64-prebuilt.zip` | Extract next to `CUSA03173` and double-click **`Bloodborne.exe`** (or **`Launcher.exe`** for settings) |
| 🐧 **Linux (`x86_64`)** | `bloodborne-pc-vulkan-linux-x86_64-prebuilt.tar.gz` | Extract next to `CUSA03173` and run **`./play.sh`** (or **`./launcher.sh`** for settings) |

```text
BloodBorne/
├── CUSA03173/                   # Your unpacked Bloodborne v1.09 game folder (contains eboot.bin, dvdroot_ps4/, ...)
└── bloodborne-pc-vulkan/        # Extracted release folder:
    ├── Bloodborne.exe / play.sh # Direct game launch
    ├── Launcher.exe / Launcher.py # 8-page GUI Settings Configurator
    └── ...
```

* In-game shortcuts: **`F8`** toggles the hardware GPU/CPU telemetry HUD (`F9` toggles per-pass X-Ray breakdown), **`Insert`** opens the ImGui overlay, **`Alt+Enter`** toggles Fullscreen / Borderless.

---

## 🔍 Clickable Sections & Technical Details

Click any topic below to expand its details:

<details open>
<summary><b>1. 🆚 How Does This Differ from <code>shadPS4</code> and Upstream <code>bbport</code>?</b></summary>
<br/>

| Area | `shadPS4` (PS4 Emulator) | Upstream `bbport` (Host Translation Layer) | **`bloodborne-pc-vulkan` (This Project)** |
| :--- | :--- | :--- | :--- |
| **Core Execution Model** | Full general-purpose PS4 emulator for many PS4 titles | Runs relinked `eboot.bin` directly on host OS with a dedicated PS4 kernel/GNM compatibility layer | Builds on `bbport`'s host runtime, replacing the heavy **Dantelion graphics path** with decompiled **C++20** modules wired into **DirectVk 1.3** |
| **VRAM & Memory Model** | Emulates PS4 unified memory via page protection traps (`mprotect` / `VEH`) | Tracks guest GPU memory writes and maintains VRAM copies / host-imported chunks | **True Split PC VRAM (`DirectBuffers` & `DirectTextures`):** allocates native `DEVICE_LOCAL` `VkBuffer` / `VkImage` objects with `4 MiB` in-pass dynamic buffer renaming (`0` write-faults/s in steady gameplay) |
| **Shader Pipeline** | Runtime GCN → SPIR-V translation | Runtime GCN → SPIR-V translation + recipe cache | **Clean Vectorized GLSL 460 Shaders** for the heaviest rendering passes (G-Buffer, Deferred PBR, 2-phase Adaptive SSR, SSAO, Yebis 2 Bloom/DoF/LUT, CSM/VSM shadows, two-sided foliage translucency) |
| **CPU Submission Path** | Multi-thread emulated command queues | Two-stage `Stage A → Stage B` draw pipe & multi-thread command segmenting | **Direct-Inline Single-Stage Recording (`Stage B = 0.00 ms`)** with epoch memoization (`pso_sig_epoch`, static `DirectBuffers` vertex memoization, unified push-descriptor layout `DscSav: 54–62%`) + Multi-CCD / P-Core thread pinning |
| **Culling & Redundant Passes** | Original PS4 draw stream | Original PS4 draw stream | **3D Hardware `SceneOcclusionCuller`** (skips occluded buildings/props/instanced meshes), trimmed SSR/Bloom mip cascades (`vkCmdCopyImage`), and dirty-gated gamma/DoF LUT uploads |
| **High-FPS Physics & Controls** | Frame-rate patches | Havok cloth stiffness & frame-pacing hooks, base in-game settings | Adds **frame-rate-independent FLVER foliage/flower wind sway (`HookFoliageSway`)**, zero-save inline `HitPushBack` guard, **native Scaleform PC menus** (`Display`, `Effects`, `26-action Controls`), `SMAA T2x`, `DLAA 4 (Preset K)`, fused `RCAS`, and `2x/4x` UI icon super-resolution |

</details>

<details>
<summary><b>2. ⚙️ Architecture: Refactored Dantelion Modules vs. PS4 Runtime</b></summary>
<br/>

* **Engine overview:** *Bloodborne* runs on FromSoftware's proprietary **Dantelion2** engine alongside **Havok** (physics/cloth/ragdolls), **Morpheme** (character animation), **FMOD** (audio), and **Scaleform GFx** (UI).
* **What still runs from the original PS4 binary (`eboot.bin`):**
  * Gameplay logic, AI, combat state machines, world streaming, Havok physics & cloth simulation, Morpheme animation graphs, FMOD audio mixing, and parts of the PS4 system/GNM context still execute from the relinked original binary over the `bbport` POSIX/Win32 runtime layer.
* **What was refactored and replaced (~90% of the critical graphics bottleneck):**
  * The core Dantelion graphics & post-processing modules (`01_CoreGraphics2_GNM`, `02_GX_GraphicsFramework`, `03_TPF_GNF_ShaderContainers`, `04_SceneRenderer`, `05_PostProcess_Yebis2`, `06_DeferredLightingAndAO`, `07_SFX_ParticleGpuRenderer`) were decompiled into native **C++20** (`our_cpp_100000_source/`) and wired directly into the **DirectVk Vulkan 1.3** backend (`gpu/`).
  * The heaviest machine-translated shaders were replaced with clean, human-readable **GLSL 460** shaders (`gpu/shader/backend/spirv/native/`), verified mathematically against the original shader outputs.
  * Remaining non-graphics bottlenecks at high frame rates (Havok cloth stiffness above 30 FPS, procedural wind sway on bushes/flowers, and PS4 thread scheduling on modern multi-CCD / hybrid CPUs) are handled via targeted binary hooks and OS thread-topology pinning.

</details>

<details>
<summary><b>3. 📊 Performance Examples (Ryzen 9 7950X + RTX 4090 @ 2560×1440)</b></summary>
<br/>

> **Note:** Frame rate depends on your specific CPU (especially L3 cache and single-core speed), GPU, RAM, resolution, and OS. The numbers below were measured on a single test system and are shown as a concrete reference example rather than a universal baseline for every PC.

**Test Configuration:**
* **CPU:** `AMD Ryzen 9 7950X` (16 cores / 32 threads, Dual-CCD)
* **GPU:** `NVIDIA GeForce RTX 4090` (`24 GB`)
* **Resolution & Settings:** `2560×1440` Native (`100%` render scale), all in-game graphics effects enabled (`SSR`, `SSAO`, `Bloom`, `Depth of Field`, `Motion Blur`, `Contact Shadows`), `DLAA 3 / DLAA 4 / SMAA T2x + RCAS`.

| In-Game Scenario (`2560×1440` Native) | Observed Frame Rate | Pure HW GPU Busy (All 10 Passes) | CPU Stage A (DirectVk RHI) | VRAM / System RAM |
| :--- | :---: | :---: | :---: | :---: |
| **Heavy Central Yharnam Street / Carriage Scene** (`~1,200–1,760` active draws after occlusion culling, `DLAA + RCAS`, All Post-FX On) | **`166 – 168 FPS`** (`~5.97 ms`, capped at `170`) | **`2.80 – 2.98 ms`** (`~47%` GPU busy, `~235 W`) | `4.09 – 5.29 ms` (`Stage B: 0.00 ms`) | `~6.2 GB` VRAM / `~3.8 GB` RAM (`0` Faults/s) |
| **Heaviest Found Multi-Shadow Indoor/Outdoor Portal** (`1,976` draws, `10` compute dispatches, `DLAA + RCAS`, All Post-FX On) | **`154 – 160 FPS`** (`~6.23 – 6.51 ms`) | **`3.69 – 3.73 ms`** (`~57%` GPU busy, `~241 W`) | `6.06 – 6.41 ms` (`Stage B: 0.00 ms`) | `~6.6 GB` VRAM / `~4.0 GB` RAM (`0` Faults/s) |
| **Lighter / Enclosed Areas (e.g. Hunter's Dream)** (`2560×1440` Native, Uncapped, `Off + RCAS`) | **`500 – 650+ FPS`** (`~1.54 – 1.95 ms`) | **`1.17 – 1.60 ms`** | `1.11 – 1.24 ms` (`Stage B: 0.00 ms`) | `~2.3 – 3.4 GB` VRAM / `~2.8 GB` RAM (`0` Faults/s) |

*(Additional side-by-side scene comparisons will be added as testing progresses).*

</details>

<details>
<summary><b>4. 🎮 PC Graphics, Anti-Aliasing, Upscalers & Controls</b></summary>
<br/>

* **Anti-Aliasing & Upscalers (`System -> Display`):**
  * **7 AA Modes:** `SMAA T2x` (default — Morphological Ultra + 2× Temporal Resolve, `~0.24 ms`), `SMAA 1x Ultra` (`~0.16 ms`), `DLAA 4 (Transformer Preset K)`, `DLAA 3 (CNN Fast)`, `TAA`, `FXAA`, or `Off`.
  * **Upscalers:** `NVIDIA DLSS 4.5 / 4`, `AMD FSR 4`, `AMD FSR 3.1` (`Quality`, `Balanced`, `Performance`, `Ultra Perf.`), or `Off (Native 100%)`.
  * **Zero-Pass `RCAS` Sharpening:** Fused directly into the final present shader (`0.00 ms` extra render pass cost).
* **Window Modes & Display:**
  * Live **3-State Window Mode** (`Fullscreen`, `Borderless Windowed`, `Windowed` + `Alt+Enter`), live Resolution (`720p`–`4K` with automatic monitor detection `output_res=auto`), Ultrawide (`21:9` / `32:9`), `16×` Anisotropic Filtering, and **`2x / 4x` GPU UI & Icon Super-Resolution**.
  * On Windows NVIDIA GPUs, automatically configures the driver profile for **DXGI Flip-Model Swapchain** (`0x20D690F8 = 1`) and **Prefer Maximum Performance** (`0x1057EB71 = 1`), preventing `dwm.exe` Desktop Window Manager composition overhead and enabling G-Sync / Independent Flip in Borderless mode.
* **Live Graphics & Remaster Toggles (`System -> Effects`, 0 restarts):**
  * Toggle Bloom/Glare, Depth of Field, Motion Blur, Chromatic Aberration, Vignette, Film Grain, Distant Blur, and Model LOD on the fly, plus 4 shader enhancements (`Multi-Scale GTAO/SSAO`, `Screen-Space Contact Micro-Shadows + 12-Tap Vogel PCF`, `Adaptive SSR + 30 Hz Cubemap Fix`, `True Black / OLED Contrast De-Haze`).
* **Mouse & Keyboard (`System -> Controls`):**
  * 26 rebindable actions, raw Mouse Look, Menu Mouse Cursor, Mouse Wheel bindings (`Wheel Up / Down`), secondary `+Alt Key` slots, **`Left Alt` Slow Walk (`40%` speed for stealth backstabs)**, and **`Enter / Esc / Options` Cutscene Skipping**.

</details>

<details>
<summary><b>5. 🛠️ Building from Source & Diagnostic Flags</b></summary>
<br/>

```bash
# 1. Build the C++20 Graphics Engine (our_cpp_100000_source -> libbb_gx_engine_native.a):
cmake -S our_cpp_100000_source -B our_cpp_100000_source/build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C our_cpp_100000_source/build

# 2. Build the Vulkan 1.3 RHI Library & Runtime (Linux):
ninja -C build_gpu && cp -f build_gpu/libbbgpu.so out/gpu/libbbgpu.so
```

**Runtime Diagnostic & Isolation Variables (No Rebuild Needed):**
```bash
# Disable 3D Hardware Occlusion Culler & Micro-Shadow Culler:
BB_HW_OCCLUSION_CULL=0 BB_MICRO_SHADOW_CULL=0 ./play.sh

# Disable hand-crafted native GLSL shaders (fall back to raw translated SPIR-V):
BB_NATIVE_SHADERS=0 BB_NATIVE_SHADER_MIN=999 ./play.sh

# Isolate specific C++20 graphics modules (01..07) or skip a VMA range for bisection:
BB_GX_MODULES=01,02,03,04,05,06,07 ./play.sh
BB_GX_SKIP_MIN=0x22900a0 BB_GX_SKIP_MAX=0x2292000 ./play.sh
```

</details>

<details>
<summary><b>6. 🇷🇺 Документация на русском языке (О проекте, архитектура, отличия и запуск)</b></summary>
<br/>

### О проекте
Это оптимизированный порт и слой трансляции **Bloodborne** (`CUSA03173` / `CUSA00900` v1.09) для **Windows 10/11** и **Linux (`x86_64`)**, созданный на базе [`bbport`](https://github.com/deadinside28/bloodborne_pc) и [`shadPS4`](https://github.com/shadps4-emu/shadPS4).

* **Архитектура движка (что отрефакторено и что осталось от PS4):**
  * *Bloodborne* работает на внутреннем движке FromSoftware **Dantelion2** в связке с физикой **Havok**, анимацией **Morpheme**, звуком **FMOD** и интерфейсом **Scaleform GFx**.
  * Игровая логика, ИИ, боевая система, физика Havok, анимации и часть системного контекста PS4 выполняются из перелинкованного оригинального бинарника (`eboot.bin`) поверх хост-рантайма `bbport`.
  * Основная работа в этом проекте сосредоточена на **критической графической части Dantelion (~90% графического конвейера)** — модули рендера `FD4` / `GX` / `CoreGraphics2_GNM` / `Yebis 2` декомпилированы в нативный **C++20** и подключены напрямую к **Vulkan 1.3 (`DirectVk`)**, а наиболее тяжёлые шейдеры переписаны в чистый векторизованный **GLSL 460** без эмуляции общей памяти PS4 и страничных прерываний.
* **Отличия от `shadPS4` и базового `bbport`:**
  1. **Раздельная видеопамять ПК (`DirectBuffers` и `DirectTextures`):** буферы и текстуры размещаются напрямую в `DEVICE_LOCAL` памяти видеокарты с динамическим переименованием до `4 МБ` внутри RenderPass (`0` страничных ошибок `Faults/s`, снижено потребление VRAM и ОЗУ).
  2. **Одностадийная запись команд (`Direct-Inline Stage A`, `Stage B = 0.00 мс`)** + мемоизация состояний (`pso_sig_epoch`, мемоизация вершинных буферов `DirectBuffers`, дедупликация дескрипторов `DscSav: 54–62%`) + привязка потоков к L3-кэшу (`CCD0` / `P-cores`) на Windows и Linux.
  3. **Аппаратное отсечение невидимой геометрии (`SceneOcclusionCuller`),** урезание лишних проходов мип-пирамид SSR/Bloom (`vkCmdCopyImage`) и отключение холостых перезаливок таблиц гаммы/DoF LUT.
  4. **Шейдеры GLSL 460 и фиксы физики на высоком FPS:** двустороннее подповерхностное освещение листвы, сглаживание шва каскадов теней CSM, плавная физика плащей Havok выше 30 FPS и **корректное качание травы, кустов и цветов на любом FPS (`HookFoliageSway`)**.
  5. **Встроенное ПК-меню (`Display`, `Effects`, `Controls`),** `SMAA T2x`, `DLAA 4 (Preset K)`, `DLSS 4.5`, `FSR 4`, однопроходный `RCAS`, 3 режима окна (`Fullscreen / Borderless / Windowed`), автоматическое включение **DXGI Flip-Model Swapchain** в драйвере NVIDIA на Windows (убирает нагрузку `dwm.exe` в безрамочном режиме) и профайлер `F8` / `F9`.

### Примеры производительности (`Ryzen 9 7950X + RTX 4090`, `2560×1440 Native`)
> **Примечание:** итоговый FPS зависит от конкретного процессора (особенно объёма L3-кэша и частоты на ядро), видеокарты, памяти и разрешения. Ниже приведены результаты замеров на тестовом ПК (`AMD Ryzen 9 7950X + NVIDIA RTX 4090, 2560×1440 Native`, все эффекты включены, `DLAA + RCAS`) для ориентира:
* **Уличные сцены Центрального Ярнама (`~1200–1760` дровов):** **`166–168 FPS`** (лимит `170`), чистое время работы GPU по всем 10 проходам — **`2.80–2.98 мс`** (`~47%` загрузки GPU), `~6.2 ГБ` VRAM, `~3.8 ГБ` ОЗУ.
* **Сложный портал с наложением каскадных и точечных теней (`1976` дровов):** **`154–160 FPS`** (`~6.2–6.5 мс`), чистое время работы GPU — **`3.69–3.73 мс`** (`~57%` загрузки GPU).
* **Закрытые локации (Сон Охотника, без лимита FPS и без DLAA):** **`500–650+ FPS`** (`~1.17–1.60 мс` GPU).

### Как запустить за 30 секунд:
1. Скачайте архив для **Windows** (`.zip`) или **Linux** (`.tar.gz`) из раздела **[Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest)**.
2. Распакуйте его рядом с папкой вашей игры **`CUSA03173`** (`v1.09` GOTY).
3. На **Windows** запустите **`Bloodborne.exe`** (или **`Launcher.exe`** для настроек); на **Linux** — **`./play.sh`** (или **`./launcher.sh`**).

</details>

---

## 📜 Credits & License

* **Upstream Foundation:**
  * **[`bbport` (`deadinside28/bloodborne_pc`)](https://github.com/deadinside28/bloodborne_pc)** — Host ELF/PE loader, PS4 kernel/system runtime (`src/`), base in-game Display menu & upscaler bridges (`FSR 3.1/4`, `DLSS`, `TAA`), and frame-pacing hooks.
  * **[`shadPS4` (`shadps4-emu/shadPS4`)](https://github.com/shadps4-emu/shadPS4)** — Original PS4 Vulkan video core and shader recompiler foundation (`GPL-2.0-or-later`).
* **`bloodborne-pc-vulkan` Engineering ([@lapothecary](https://github.com/lapothecary)):**
  * Refactored C++20 Dantelion graphics & post-FX pipeline ([`our_cpp_100000_source/`](our_cpp_100000_source), [`bb_golden_engine.cpp`](gpu/shim/bb_golden_engine.cpp)), DirectVk 1.3 split VRAM architecture ([`direct_buffers.cpp`](gpu/buffers/direct_buffers.cpp), [`direct_textures.cpp`](gpu/textures/direct_textures.cpp)), clean vectorized GLSL 460 shader suite ([`gpu/shader/backend/spirv/native/`](gpu/shader/backend/spirv/native)), 3D hardware `SceneOcclusionCuller`, Direct-Inline Stage A memoization, high-FPS foliage sway & physics fixes, sealed Scaleform GFx menu engine ([`bbport_menu_engine.h`](gpu/shim/bbport_menu_engine.h)), `SMAA T2x` / `SMAA 1x Ultra`, `2x/4x` UI super-resolution, 21-subpass X-Ray GPU/CPU profiler HUD, and Windows/Linux L3/CCD thread topology & NVAPI DXGI swapchain integration.
* **License:** **[GNU General Public License v3.0](LICENSE)** (`GPL-3.0`).
