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

## 🆕 What's New in `v1.1.6` — Full-Game Vulkan Shader Pre-Baker & Zero-Stutter Runtime

* **Full-Game + `eboot.bin` Vulkan Shader & PSO Pre-Baker (`0% → 100%`, `5,761 / 5,761` Shaders):** Added `bb-probe --bake-shaders` and a dedicated **Vulkan Shader Cache & Pre-Compilation** (`0% → 100%` progress bar + safe **Clear Cache**) card across all Windows & Linux Launchers (`Engine and Debug` / `Performance`). Scans all 7 `dvdroot_ps4/shader/*.shaderbnd.dcx` archives **plus** all embedded `Shdr\x07` containers inside `eboot.bin` (`354` Scaleform GFx UI / internal GNM / Compute shaders + the active `EXP_32_ABGR` Yebis 2 Post-FX bank: `CS=68 + PS=4,058 + VS=1,635 = 5,761` unique programs + `423` base PSOs), loading all `5,761` modules in **`~292 ms`** at startup across 12 threads.
* **Polite Invisible Background Auto-Baker (`bb:AutoBake`):** On game startup, automatically checks if all `5,761` shaders are already warm in `user/cache/` (`0%` CPU when warm). On a fresh install — even if you press **`PLAY`** right away without running the Launcher baker — silently pre-compiles missing shaders in the background on `1–3` OS idle-priority threads (`SCHED_IDLE` / `THREAD_PRIORITY_LOWEST`, `~10–15%` CPU) without blocking the Title Screen or impacting FPS.
* **Zero-Stutter / Zero-Pop-In `Async SPV + PSO` Engine (`5,000 ms -> ~2.5 ms`):** Switched runtime specialization misses and background `GetProgramAsyncWorker` threads to fast dead-branch DCE (`runtime_fast_opt` for `>= 512` words, upgrading to full `-O` `opt31_*.spv` on next startup), normalized unbound (`!sharp`) descriptor defaults (`NormalizeUnboundDefaults`), snapshotted per-frame `flat_buf` + `PtrVertexBufferTable` (`0x1028...`) synchronously on `Stage A`, and added `GraphicsPipeline::UpdateStageHash` — eliminating shader-compile stutters and multi-second invisible character/armor pop-in on first combat hits.
* **Hunter Attire & Face Blood-Splatter Fix (`GXFlver_*_Bloody_*` `cldist_attr0.zw`):** Fixed missing blood splatters on the Hunter's clothes, cape, and face after kills on strict desktop GPUs (e.g. NVIDIA RTX 40-series) by initializing unwritten `ClipDistance` attributes (`cldist_attr0.zw = 1.0f`) in `LowerUserClipPlanes` + healing existing cached `.spv` binaries at startup (`EnsureClipDistanceInitializedInSpv`, `< 3 ms`), plus added `[BB-HITCH]` (`logs/bb_hitch.log`) frame-spike diagnostics.
* **Windows Cold-Start `AutoBake` Crash Fix, Self-Updater Fix & Safe FPS Optimizations (`129` Native GLSL 460 Shaders):**
  * Fixed the Windows cold-start `Assertion Failed! (size <= Capacity / 4)` use-after-free crash in `DrainAutoBakeReadyStages()` (missing `ComputePipeline::SetStageInfo()` rebind after moving `item->info`) and pinned the 1-click SHA-256 self-updater to tag-specific URLs with cache-busting & re-verification.
  * Added **4-Bucket `Front-to-Back` G-Buffer Reordering (`Near → Mid → Far → Cutout`)** on `Stage B` (`-14%` G-Buffer GPU time, `+5.5%` descriptor savings, and `+2–8%` better hardware `SceneOcclusionCuller` hit rate via pre-populated depth), eliminated `bb:Resolve0` `100 µs` spin-wait (`O(1)` `resolve_queue`, `-30–36%` CPU on the resolve thread), balanced 5-worker `dantelion_gx` skinning + SIMD bone/frustum hoisting, expanded `MicroShadow` culling (`-65..85` micro-shadow draws/frame in corridors), and added **6 more native GLSL 460 shaders** (`123 → 129`: Bloom/Glare, Motion Blur tile dilation, SSR Hi-Z, Soft Particles, Scaleform BGRA).

### Highlights from `v1.1.5`
* **Dual-Stage Monotonic Frame & Present Pacing (`Jit: 2.50 ms -> 0.01–0.15 ms`):** Monotonic phase-locked deadline pacer (`HookPacerWait`, `0x2034964`) + post-render output `Present Pacing` on `bb:Present`, `5`-image `Mailbox`/`Immediate` swapchain buffering, zero-drift `sim_dt` stabilizer (`p50 = 5.87 ms`, `p95 = 5.88 ms`), and **event-driven `1000+ Hz` lock-free mouse input**.
* **3 Camera Follow Modes (`F4` Hotkey) & *Lies of P*-Style `Balanced` Camera:** Live switching between **`Original (PS4)`**, **`Balanced`** (default), and **`Locked (1:1)`** (`F4` OSD toast, `System -> Controls`, `Insert` overlay, and GUI Launchers). The **`Balanced`** preset brings a ***Lies of P*-inspired** physical spring-arm feel calibrated for *Bloodborne*: direct world-space `XZ` pivot inertia from `0.00 s` (no harsh camera jerk on lateral `A/D` or diagonal movement), smooth quadratic (`y ~ t^2`) sprint/jog pull-back with native Havok wall collision, a crisp `~0.55 s` stop return, stair/cobblestone vertical smoothing, and `0%` mouse rotation interference.
* **`4–5×` Faster Location/Respawn Loading (`BB_FAST_LOAD`), Streaming `GFX` Heap Fixes, 1-Click SHA-256 Auto-Updater & Expanded Menus:** Host SIMD `zlib` (`11/11` detours + `64 KiB` `.dcx` chunk, `~11–12 s -> ~2.0–2.5 s` respawn reload), thread-safe `dantelion_gx` teardown, 1-click SHA-256 Launcher self-updater, two-way sync of all 17 native `CSOptionsData` settings, expanded Scaleform `System`/`Display`/`Effects`/`Controls` (`ScrollBarV` + `0-ms` wheel), crash Black Box (`logs/crash-*.log`), and Title Screen **`Exit Game`**.
* *(See [Releases](https://github.com/lapothecary/bloodborne-pc-vulkan/releases/latest) for full `v1.1.6`, `v1.1.5` & `v1.1.4` changelogs, including `SceneOcclusionCuller` 2.0, `decommission_pm4`, and `129` native GLSL 460 shaders).*

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
| **Shader Pipeline** | Runtime GCN → SPIR-V translation | Runtime GCN → SPIR-V translation + background recipe cache | **Full-Game `5,761`-Shader Pre-Baker (`0% → 100%`, `~292 ms` 12-thread warmup) + Polite Idle `bb:AutoBake` + `~2.5 ms` `runtime_fast_opt` Async `SPV + PSO` Engine + 129 Hand-Crafted Vectorized GLSL 460 Shaders** replacing the heaviest rendering passes (G-Buffer, early-out 2-sided foliage/cutout, Deferred PBR, Atmospheric Fog, 2-phase Adaptive SSR, SSAO, Yebis 2 Bloom/DoF/LUT, CSM/VSM shadows, Scaleform UI & light-tile batching) |
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
  4. **Предкомпиляция всех `5,761` шейдеров игры (`0% → 100%` в лаунчере + тихий фоновый `bb:AutoBake` + `~2.5 мс` `Async SPV + PSO` без пропадания моделей и фикс крови на одежде `cldist_attr0`), 123 нативных шейдера GLSL 460 (`opt35`) и фиксы физики на высоком FPS:** полный пре-бейкер всех 7 архивов `.shaderbnd.dcx` и встроенных шейдеров `eboot.bin` (`~292 мс` прогрев на 12 потоках), 4 нативных вершинных шейдера FLVER-листвы (с пропуском нулевых весов костей) + uniform early-out детальных и кубических карт в двусторонних шейдерах листвы (`2b. Foliage & Cutout: 0.16–0.32 мс`), экранный PCF-шейдер теней CSM, атмосферный туман, Yebis 2 Bokeh DoF / Bloom / Glare, плавная физика плащей Havok выше 30 FPS и **корректное качание травы, кустов и цветов на любом FPS (`HookFoliageSway`)**.
  5. **Обновление порта в 1 клик с проверкой SHA-256 прямо из лаунчера (`user/savedata`, `bbport.ini` и `mods/` сохраняются), единая двусторонняя синхронизация всех 17 нативных настроек игры (`Music`/`SFX`/`Voice`, яркость, кровь, субтитры, язык, камера и сеть) между `userdata0010`, `bbport.ini` и лаунчером, расширенное Scaleform-меню (`7` вкладок `System` и `11`-строчные `Display`/`Effects`/`Controls` с готическим скроллбаром и `0-мс` колесом мыши), нативный пункт `Exit Game` / `Выйти из игры` в главном меню, 3 режима физики следования камеры по `F4` (`Original (PS4)` / `Balanced` с отзывчивостью и плавным инерционным отъездом в духе *Lies of P* / `Locked (1:1)`), ускорение загрузок локаций и респавна в `4–5×` (`BB_FAST_LOAD`), межпоточная защита кучи `GFX` при стриминге секторов и универсальный «Чёрный ящик» крашей (`logs/crash-*.log`),** двухступенчатый фазовый пейсинг кадров и вывода (`0.01–0.15 мс` джиттера при лимите FPS) с `1000+ Гц` событийным опросом мыши, `SMAA T2x`, `DLAA 4 (Preset K)`, `DLSS 4.5`, `FSR 4`, однопроходный `RCAS`, 3 режима окна (`Fullscreen / Borderless / Windowed`), автоматическое включение **DXGI Flip-Model Swapchain** в драйвере NVIDIA на Windows и профайлер `F8` / `F9`.

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
