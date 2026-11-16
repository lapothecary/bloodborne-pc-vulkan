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

* In-game shortcuts: **`F8`** toggles the `Debug` HUD (`F9` toggles `Additional Debug` sub-pass breakdown), **`F4`** cycles **Camera Follow Mode** (`Original (PS4)` / `Balanced` / `Locked (1:1)`), **`F3`** cycles the FPS cap (all hotkeys are rebindable in `System -> Controls`), **`Insert`** opens the ImGui overlay, **`Alt+Enter`** toggles Fullscreen / Borderless.

---

## 🆕 What's New in `v1.1.5`

* **Dual-Stage Monotonic Frame & Present Pacing (`Jit: 2.50 ms -> 0.01–0.15 ms`):** Replaced the relative guest `usleep` cap path with a monotonic phase-locked deadline pacer (`HookPacerWait`, `0x2034964`) paired with post-render output `Present Pacing` on `bb:Present`, `5`-image `Mailbox`/`Immediate` swapchain buffering, and a zero-drift `sim_dt` stabilizer — eliminating FPS cap sag (`170 -> 145–155 FPS`), early burst presents, and fixed-refresh micro-stutter (`p50 = 5.87 ms`, `p95 = 5.88 ms`).
* **Event-Driven `1000+ Hz` Mouse Input:** Removed the `2 ms` window poll sleep quantum (`SDL_WaitEventTimeout`) and eliminated `imgui_mutex` contention on raw `SDL_EVENT_MOUSE_MOTION` events when the overlay is closed, keeping high-polling-rate mouse camera movement smooth at high refresh rates.
* **3 Camera Follow Modes (`F4` Hotkey) & *Lies of P*-Style `Balanced` Camera:** Added live switching between **`Original (PS4)`**, **`Balanced`** (default), and **`Locked (1:1)`** camera follow modes (`F4` with on-screen OSD toast, `System -> Controls`, `Insert` overlay, and GUI Launchers). The **`Balanced`** preset brings a ***Lies of P*-inspired** physical spring-arm feel calibrated for *Bloodborne*: direct world-space `XZ` pivot inertia from `0.00 s` (no harsh camera jerk on lateral `A/D` or diagonal movement), smooth quadratic (`y ~ t^2`) sprint/jog pull-back with native Havok wall collision, a crisp `~0.55 s` stop return, stair/cobblestone vertical smoothing, and `0%` mouse rotation interference.
* **`4–5×` Faster Location, Warp & Respawn Loading (`BB_FAST_LOAD`) + Streaming `GFX` Heap Stability Fixes:** Redirected both statically linked PS4 `zlib` copies (`1.2.7` & `1.2.3`, `11/11` detours) to host SIMD `zlib` (`zlib-ng` / `zlib1`) with a `64 KiB` `DcxDflt` streaming chunk, unlocked a `500 Hz` non-3D loading-screen pacer (`Core.Res.*` pinned to fast worker cores), and moved post-rename `SPRJ0005` save `fsync` flushes off the critical path (`~11–12 s -> ~2.0–2.5 s` respawn reload) — paired with a full thread-safety & destructor teardown overhaul across all `48` C++20 `dantelion_gx` modules that eliminates `GFX` heap streaming crashes (`DLRegularHeap::free` / `GXWorker` command wrappers) on sector transitions.
* **Universal Crash "Black Box" (`logs/crash-*.log`), Native `"Exit Game"` & Input Prompt Fixes:** Added a cross-platform Windows/Linux crash analyzer (`logs/crash-YYYYMMDD_HHMMSS.log` with `.pdb`-free C++20/guest hybrid stack traces, `DLRegularHeap` corruption inspection, and a 128-event Flight Recorder), a localized **`Exit Game`** / **`Выйти из игры`** row on the Title Screen, and Primary-slot (`Slot 1`) + `Mouse 4/5` HUD prompt icon fixes.
* *(See [Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest) for full `v1.1.5` & `v1.1.4` changelogs, including `SceneOcclusionCuller` 2.0, `decommission_pm4`, and `123` native GLSL 460 shaders).*

---

## 🔍 Clickable Sections & Technical Details

Click any topic below to expand its details:

<details open>
<summary><b>1. 🆚 How Does This Differ from <code>shadPS4</code> and Upstream <code>bbport</code>?</b></summary>
<br/>

| Area | `shadPS4` (PS4 Emulator) | Upstream `bbport` (Host Translation Layer) | **`bloodborne-pc-vulkan` (This Project)** |
| :--- | :--- | :--- | :--- |
| **Core Execution Model** | Full general-purpose PS4 emulator for many PS4 titles | Runs relinked `eboot.bin` directly on host OS with a dedicated PS4 kernel/GNM compatibility layer | Builds on `bbport`'s host runtime, replacing the heavy **Dantelion graphics path** with decompiled **C++20** modules wired into **DirectVk 1.3** |
| **VRAM & Memory Model** | Emulates PS4 unified memory via page protection traps (`mprotect` / `VEH`) | Shares guest RAM in-place (`VK_EXT_external_memory_host`) with mirrored VRAM copies for hot blocks | **True Split PC VRAM (`DirectBuffers` & `DirectTextures`):** intercepts engine buffer/texture allocators (`GXGnmBuffer`, `.flver`, `.tpf`/`.gnf`) into native `DEVICE_LOCAL` `VkBuffer` / `VkImage` objects with `4 MiB` in-pass dynamic buffer renaming (`0` write-faults/s in steady gameplay) |
| **Shader Pipeline** | Runtime GCN → SPIR-V translation | Runtime GCN → SPIR-V translation + background recipe cache | **123 Hand-Crafted Vectorized GLSL 460 Shaders** replacing the heaviest rendering passes (G-Buffer, early-out 2-sided foliage/cutout, Deferred PBR, Atmospheric Fog, 2-phase Adaptive SSR, SSAO, Yebis 2 Bloom/DoF/LUT, CSM/VSM shadows, Scaleform UI & light-tile batching) + `SPIRV-Tools-opt` pass |
| **CPU Submission Path** | Single GPU command thread emulating the command stream | Two-stage `Stage A → Stage B` draw pipe & multi-thread command segmenting | **DirectVk Command Stream (`decommission_pm4`) + L3-Co-Located `Stage A → Stage B → VkRecorder` Pipeline** (zero-heap `DrawPipe` & inline `CopyBatch` packets, `O(1)` `2 MiB` VMA TLB + `1 MiB` `PendingReads` + `64 KiB` CB write bitmasks, immutable `.flver` buffer fast-skip, `Stage B`-hosted `SceneOcclusionCuller` 2.0 at `0.00 ms` `Stage A` cost, and opt-in `VkMtDirectRecorder` secondary command buffers) |
| **Culling, Batching & Redundant Passes** | Original PS4 draw stream | Original PS4 draw stream + Vulkan occlusion queries | **3D Hardware `SceneOcclusionCuller` 2.0** (`27–41%` G-Buffer culling + `45%` wall-occluded distant skeleton culling + `7-MRT` upscaler support), **`LightTileBatcher`** (`100%` `84/84` `GXLightAcc` deferred light-tile coalescing, `-40..55%` draws in dense lighting), **`ObjectVelocityCuller`**, **`VsmShadowMemo`**, **`VfxUiBatcher`**, staggered far-cascade CSM, and idle-scene **Motion Blur skip** |
| **High-FPS Physics, Fast Load & Controls** | Frame-rate patches | Havok cloth stiffness & frame-pacing hooks, base in-game `Display` / `Effects` pages, `SMAA 1x` & `RCAS` | Adds **dual-stage monotonic `PacerWait` + output `Present Pacing`** (`0.01–0.15 ms` jitter at capped high refresh rates), **event-driven `1000+ Hz` lock-free mouse input**, **`4–5×` faster location/respawn loading (`BB_FAST_LOAD`: host SIMD `zlib` `11/11` detours + `64 KiB` `.dcx` chunk + `500 Hz` loading pacer + async save `fsync`)**, **frame-rate-independent FLVER foliage/flower wind sway (`HookFoliageSway`)**, in-game **`System -> Controls` tab (28 actions + rebindable `F8`/`F9`)** & Title Screen **`Exit Game`**, expanded `SelectRow` `Display` menu (100% live switching without restart), **`SMAA T2x`**, **`DLSS 4.5` / `DLAA 4 (Preset K)`**, fused zero-pass `RCAS`, **`2x/4x` UI icon super-resolution**, and NVIDIA **DXGI Flip-Model Swapchain** (`0x20D690F8 = 1`) |

</details>

<details>
<summary><b>2. ⚙️ Architecture: Refactored Dantelion Modules vs. PS4 Runtime</b></summary>
<br/>

* **Engine overview:** *Bloodborne* runs on FromSoftware's proprietary **Dantelion2** engine alongside **Havok** (physics/cloth/ragdolls), **Morpheme** (character animation), **FMOD** (audio), and **Scaleform GFx** (UI).
* **What still runs from the original PS4 binary (`eboot.bin`):**
  * Gameplay logic, AI, combat state machines, world streaming, Havok physics & cloth simulation, Morpheme animation graphs, FMOD audio mixing, and parts of the PS4 system/GNM context still execute from the relinked original binary over the `bbport` POSIX/Win32 runtime layer.
* **What was refactored and replaced (~90% of the critical graphics bottleneck):**
  * The core Dantelion graphics & post-processing modules (`01_CoreGraphics2_GNM`, `02_GX_GraphicsFramework`, `03_TPF_GNF_ShaderContainers`, `04_SceneRenderer`, `05_PostProcess_Yebis2`, `06_DeferredLightingAndAO`, `07_SFX_ParticleGpuRenderer`) were decompiled into native **C++20** (`dantelion_gx/`) and wired directly into the **DirectVk Vulkan 1.3** backend (`gpu/`).
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

| In-Game Scenario (`2560×1440` Native) | Observed Frame Rate | Pure HW GPU Busy (All 10 Passes) | CPU RHI (`Stage A` / `Stage B`) | Per-Process VRAM / RAM |
| :--- | :---: | :---: | :---: | :---: |
| **Heavy Central Yharnam Bonfire Vista (`m24`)** (`~1,100` active draws after occlusion & light-tile batching, Uncapped, `Off + RCAS`, All Post-FX On) | **`340 – 355 FPS`** (`~2.85 – 2.95 ms` Avg; **`170.0 FPS` locked** with `DLAA + RCAS`, `Jit 0.01–0.15 ms`, `p95 = 5.88 ms`) | **`2.60 – 2.64 ms`** (`~90%` GPU busy, `0.00 ms` mid-frame bubble) | `Stage A: 2.85 ms` / `Stage B: 2.30 ms` | `~3.09 GB` VRAM (`0` Faults/s) |
| **Lighter / Enclosed Areas (e.g. Hunter's Dream)** (`2560×1440` Native, Uncapped, `Off + RCAS`) | **`480 – 700+ FPS`** (`~1.41 – 2.08 ms`, `480+` All Post-FX On / `~700` Post-FX Off) | **`1.31 – 1.95 ms`** | `Stage A: 0.61 – 0.84 ms` / `Stage B: 0.52 – 0.67 ms` | `~2.59 – 3.14 GB` VRAM (`0` Faults/s) |

*(Additional side-by-side scene comparisons will be added as testing progresses).*

</details>

<details>
<summary><b>4. 🎮 PC Graphics, Anti-Aliasing, Upscalers & Controls</b></summary>
<br/>

* **Anti-Aliasing & Upscalers (`System -> Display`):**
  * **7 AA Modes:** `Off` (default), `SMAA T2x` (Morphological Ultra + 2× Temporal Resolve, `~0.24 ms`), `SMAA 1x Ultra` (`~0.16 ms`), `DLAA 4 (Transformer Preset K)`, `DLAA 3 (CNN Fast)`, `TAA`, or `FXAA`.
  * **Upscalers:** `NVIDIA DLSS 4.5 / 4`, `AMD FSR 4`, `AMD FSR 3.1` (`Quality`, `Balanced`, `Performance`, `Ultra Perf.`), or `Off (Native 100%)`.
  * **Zero-Pass `RCAS` Sharpening:** Fused directly into the final present shader (`0.00 ms` extra render pass cost).
* **Window Modes & Display:**
  * Live **3-State Window Mode** (`Fullscreen`, `Borderless Windowed`, `Windowed` + `Alt+Enter`), live Resolution (`720p`–`4K` with automatic monitor detection `output_res=auto`), Ultrawide (`21:9` / `32:9`), `16×` Anisotropic Filtering, and **`2x / 4x` GPU UI & Icon Super-Resolution**.
  * On Windows NVIDIA GPUs, adds **DXGI Flip-Model Swapchain** (`0x20D690F8 = 1`) alongside **Prefer Maximum Performance** (`0x1057EB71 = 1`) in the driver profile, preventing `dwm.exe` Desktop Window Manager composition overhead and enabling G-Sync / Independent Flip in Borderless mode.
* **Live Graphics & Remaster Toggles (`System -> Effects`, 0 restarts):**
  * Toggle Bloom/Glare, Depth of Field, Motion Blur, Chromatic Aberration, Vignette, Film Grain, Distant Blur, and Model LOD on the fly, plus 4 shader enhancements (`Multi-Scale GTAO/SSAO`, `Screen-Space Contact Micro-Shadows + 12-Tap Vogel PCF`, `Adaptive SSR + 30 Hz Cubemap Fix`, `True Black / OLED Contrast De-Haze`).
* **Mouse & Keyboard (`System -> Controls`):**
  * 28 rebindable actions (including `Debug` on `F8` and `Additional Debug` on `F9`), raw Mouse Look, Menu Mouse Cursor, Mouse Wheel bindings (`Wheel Up / Down`), secondary `+Alt Key` slots, **`Left Alt` Slow Walk (`40%` speed for stealth backstabs)**, and **`Enter / Esc / Options` Cutscene Skipping**.

</details>

<details>
<summary><b>5. 🛠️ Building from Source & Diagnostic Flags</b></summary>
<br/>

```bash
# 1. Build the C++20 Graphics Engine (dantelion_gx -> libbb_gx_engine_native.a):
cmake -S dantelion_gx -B dantelion_gx/build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C dantelion_gx/build

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
  2. **Прямой командный поток `DirectVk` (`decommission_pm4`) и конвейер внутри L3-домена (`Stage A → Stage B → VkRecorder`):** отказ от лишнего сканирования гостевого PM4/CE-потока на `Stage A`, пакеты `DrawPipe` / `CopyBatch` без аллокаций в куче, `O(1)` фильтры `PendingReads` (`1 МБ`) и записей константных буферов (`64 КБ`), мгновенный пропуск неизменяемых `.flver`-буферов (`2b. Buffers & Ring: 0.11–0.14 мс`), привязка `GXWorker` к игровому L3-кэшу и вынос оценки 3D-отсечения на свободный ресурс `Stage B` (`0.00 мс` нагрузки на `Stage A`).
  3. **Аппаратное отсечение `SceneOcclusionCuller` 2.0 и батчинг отрисовок (`LightTileBatcher`, `ObjectVelocityCuller`, `VsmShadowMemo`, `VfxUiBatcher`):** отсечение `27–41%` закрытых стенами G-Buffer мешей (включая до `45%` далёких скелетных врагов за зданиями и поддержку `7-MRT` при `DLAA/DLSS/FSR`), объединение `64×64` тайлов локального света по всем `84/84` (`100%`) шейдерам `GXLightAcc` (`-40..55%` дровов в плотно освещённых сценах), мемоизация VSM-теней, стаггеринг дальних каскадов CSM и пропуск `Motion Blur` в покое.
  4. **123 нативных шейдера GLSL 460 (`opt35`) и фиксы физики на высоком FPS:** 4 нативных вершинных шейдера FLVER-листвы (с пропуском нулевых весов костей) + uniform early-out детальных и кубических карт в двусторонних шейдерах листвы (`2b. Foliage & Cutout: 0.16–0.32 мс`), экранный PCF-шейдер теней CSM, атмосферный туман, Yebis 2 Bokeh DoF / Bloom / Glare, плавная физика плащей Havok выше 30 FPS и **корректное качание травы, кустов и цветов на любом FPS (`HookFoliageSway`)**.
  5. **Встроенное ПК-меню (`Display`, `Effects`, `Controls` и нативный пункт `Exit Game` / `Выйти из игры` в главном меню), 3 режима физики следования камеры по `F4` (`Original (PS4)` / `Balanced` с отзывчивостью и плавным инерционным отъездом в духе *Lies of P* / `Locked (1:1)`), ускорение загрузок локаций и респавна в `4–5×` (`BB_FAST_LOAD`), межпоточная защита кучи `GFX` при стриминге секторов и универсальный «Чёрный ящик» крашей (`logs/crash-*.log`),** двухступенчатый фазовый пейсинг кадров и вывода (`0.01–0.15 мс` джиттера при лимите FPS) с `1000+ Гц` событийным опросом мыши, `SMAA T2x`, `DLAA 4 (Preset K)`, `DLSS 4.5`, `FSR 4`, однопроходный `RCAS`, 3 режима окна (`Fullscreen / Borderless / Windowed`), автоматическое включение **DXGI Flip-Model Swapchain** в драйвере NVIDIA на Windows и профайлер `F8` / `F9`.

### Примеры производительности (`Ryzen 9 7950X + RTX 4090`, `2560×1440 Native`)
> **Примечание:** итоговый FPS зависит от конкретного процессора (особенно объёма L3-кэша и частоты на ядро), видеокарты, памяти и разрешения. Ниже приведены результаты замеров на тестовом ПК (`AMD Ryzen 9 7950X + NVIDIA RTX 4090, 2560×1440 Native`, все эффекты включены) для ориентира:
* **Тяжёлая панорама Центрального Ярнама у костра (`~1100` активных дровов после отсечения и батчинга света, без лимита FPS, все эффекты включены, `Off + RCAS`):** **`340–355 FPS`** (`~2.85–2.95 мс`; либо фиксированные **`170.0 FPS`** при включённом `DLAA + RCAS` с джиттером **`0.01–0.15 мс`**, `p95 = 5.88 мс`), чистое время работы GPU по всем 10 проходам — **`2.60–2.64 мс`** (`0.00 мс` пузырей между пассами), CPU RHI — **`Stage A: 2.85 мс` / `Stage B: 2.30 мс`**, `~3.09 ГБ` VRAM самого процесса (`0` Faults/s).
* **Закрытые локации (Сон Охотника, `2560×1440 Native`, без лимита FPS):** **`480–486 FPS`** со всеми ультра-эффектами (`~1.95 мс` GPU, `Stage A: 0.82 мс`) и до **`700+ FPS`** при выключенных пост-эффектах (`~1.33 мс` GPU, `Stage A: 0.62 мс`, `~2.59 ГБ` VRAM).

### Как запустить за 30 секунд:
1. Скачайте архив для **Windows** (`.zip`) или **Linux** (`.tar.gz`) из раздела **[Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest)**.
2. Распакуйте его рядом с папкой вашей игры **`CUSA03173`** (`v1.09` GOTY).
3. На **Windows** запустите **`Bloodborne.exe`** (или **`Launcher.exe`** для настроек); на **Linux** — **`./play.sh`** (или **`./launcher.sh`**).

</details>

---

## 📜 Credits & License

* **Upstream Foundation:**
  * **[`bbport` (`deadinside28/bloodborne_pc`)](https://github.com/deadinside28/bloodborne_pc)** — Host ELF/PE loader, PS4 kernel/system runtime (`src/`), base in-game `Display`/`Effects` menu & upscaler bridges (`FSR 3.1/4`, `DLSS`, `TAA`, `SMAA 1x`, `RCAS`), Havok cloth & frame-pacing hooks, and online module (`bbnet20`).
  * **[`shadPS4` (`shadps4-emu/shadPS4`)](https://github.com/shadps4-emu/shadPS4)** — Original PS4 Vulkan video core and shader recompiler foundation (`GPL-2.0-or-later`).
* **`bloodborne-pc-vulkan` Additions ([@lapothecary](https://github.com/lapothecary)):**
  * Refactored C++20 Dantelion graphics & post-FX pipeline ([`dantelion_gx/`](dantelion_gx), [`bb_golden_engine.cpp`](gpu/shim/bb_golden_engine.cpp)), DirectVk 1.3 split VRAM allocator interception ([`direct_buffers.cpp`](gpu/buffers/direct_buffers.cpp), [`direct_textures.cpp`](gpu/textures/direct_textures.cpp)), 123 clean vectorized GLSL 460 shaders ([`gpu/shader/backend/spirv/native/`](gpu/shader/backend/spirv/native)), 3D hardware `SceneOcclusionCuller`, Stage-A `LightTileBatcher` / `ObjectVelocityCuller` / `VsmShadowMemo` / `VfxUiBatcher`, high-FPS foliage wind sway fix (`HookFoliageSway`), in-game `System -> Controls` tab & sealed Scaleform GFx menu engine ([`bbport_menu_engine.h`](gpu/shim/bbport_menu_engine.h)), `SMAA T2x`, `DLSS 4.5` / `DLAA 4 Preset K`, `2x/4x` UI super-resolution, 21-subpass `Debug` / `Additional Debug` (`F8`/`F9`) profiler HUD, and Windows/Linux L3/CCD thread topology & NVAPI DXGI swapchain integration.
* **License:** **[GNU General Public License v3.0](LICENSE)** (`GPL-3.0`).
