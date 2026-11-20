// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_SECTIONS=1 (diagnostics): time the draw recording thread (and, "A:" parts, the GPU
// command thread) spends per part of a draw, inclusive (nested parts count in their parent too),
// printed with the draw pipe statistics.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <x86intrin.h>

namespace BbSections {
enum Id : std::uint32_t {
    Packet,          ///< a whole packet of the draw pipe
    Task,            ///< ordered tasks (fences, signals, DMA, flips)
    DrawRecord,
    PrepareRenderState,
    BindResources,
    BindBuffers,
    BindTextures,
    BeginRendering,
    ResolveVertexBuffers,
    ResolveIndexBuffer,
    EmitVertexBuffers,
    UpdateDynamicState,
    DispatchRecord,
    ShaderHle,
    ObtainBuffer,
    Flush,
    Motion,          ///< object motion (bone palettes, index ranges)
    PipelineBind,    ///< Pipeline::BindResources (descriptors, push constants)
    Kick,            ///< ResetBindings and the hand-over to the Vulkan recording threads
    Preupload,
    EnsureResident,
    ObtainStream,  ///< ObtainBuffer's copy of a small read-only buffer into the stream buffer
    ObtainVram,    ///< ObtainBuffer's VRAM path (SynchronizeMemory)
    LayerBind,     ///< BB_LAYER_MEMORY: BufferCache::LayerBind
    // The GPU command thread (stage A), per draw handed over.
    ADraw,         ///< Rasterizer::Draw: the pipeline selection and the hand-over
    APipeline,     ///< PipelineCache::GetGraphicsPipeline
    APrepared,     ///< its prepared-draw check (TryPreparedPipeline)
    APostDraw,     ///< packing the draw for the recording thread
    ARing,         ///< constants copied into the ring (CollectRingBindings)
    ARingChecks,   ///< its checks of each buffer (GPU-written, pending writes, VRAM copies)
    ARingCopy,     ///< its copies (memcpy and flush)
    ARingSharp,    ///< its V# reads and range clamps
    ARingNote,     ///< its pending read notes (NotePendingRead)
    ADecode,       ///< decoding a whole submission (everything of stage A)
    ACullBatch,    ///< Stage A cullers & batchers (MicroShadow/VSM/LightBatch/VelCull/VfxUi)
    ACullMicro,
    ACullVsm,
    ACullLight,
    ACullVel,
    ACullVfx,
    AFpsoSig,
    AFpsoFlat,
    AFpsoMatch,
    BOccSort,
    BDrawEmit,
    Count,
};
inline constexpr const char* Names[Count] = {
    "packet",       "tasks",       "draw",          "render state", "bind",
    "buffers",      "textures",    "begin render",  "vertex",       "index",
    "emit vertex",  "dynamic",     "dispatch",      "HLE shader",   "ObtainBuffer",
    "flush",        "motion",      "descriptors",   "kick",         "preupload",
    "EnsureResident", "stream copy", "VRAM path", "layer bind",
    "A: draw", "A: pipeline", "A: prepared", "A: post", "A: ring", "A: ring checks",
    "A: ring copy", "A: ring sharp", "A: ring note", "A: decode", "A: cull/batch",
    "A: micro", "A: vsm", "A: light", "A: vel", "A: vfx",
    "A: fpso sig", "A: fpso flat", "A: fpso match",
    "B: occ+sort", "B: draw emit",
};
// Per-thread counters on their own cache lines: stage A and stage B run on different cores (often
// different CCDs) and a scope ends ~20 times per draw on each; a line written by both bounced
// between them on every draw (~100 ns over the Infinity Fabric each time).
struct alignas(64) Counters {
    std::array<std::atomic<std::uint64_t>, Count> v{};
    std::atomic<std::uint64_t>& operator[](std::size_t i) { return v[i]; }
    const std::atomic<std::uint64_t>& operator[](std::size_t i) const { return v[i]; }
};
struct alignas(64) PaddedCounter {
    std::atomic<std::uint64_t> value{0};
    std::uint64_t load(std::memory_order order = std::memory_order_seq_cst) const {
        return value.load(order);
    }
    void store(std::uint64_t v, std::memory_order order = std::memory_order_seq_cst) {
        value.store(v, order);
    }
    std::uint64_t fetch_add(std::uint64_t v, std::memory_order order = std::memory_order_seq_cst) {
        return value.fetch_add(v, order);
    }
};
inline Counters cycles_a{}; ///< written by the GPU command thread only
inline Counters cycles_b{}; ///< written by the draw recording thread only
/// Both stages summed (read side only).
struct CyclesSum {
    struct Ref {
        std::size_t i;
        std::uint64_t load(std::memory_order order = std::memory_order_relaxed) const {
            return cycles_a[i].load(order) + cycles_b[i].load(order);
        }
    };
    Ref operator[](std::size_t i) const { return {i}; }
};
inline constexpr CyclesSum cycles{};
/// Constant buffers stage A looked at and bytes it copied into the ring (with the sections).
inline PaddedCounter ring_buffers{}, ring_copies{}, ring_bytes{};
/// DrawPipe live counters mirrored for the in-game frame breakdown HUD.
inline PaddedCounter pipe_packets{}, pipe_tasks{}, pipe_drains{}, pipe_drain_cycles{},
    pipe_busy_cycles{}, resolve_packets{}, resolve_busy_cycles{}, resolve_wakes{},
    stage_b_wait_res_cycles{}, fast_pso_hits{}, fast_pso_full{};

inline bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("BB_SECTIONS");
        return !env || env[0] != '0';
    }();
    return on;
}
/// Set on the draw recording thread (DrawPipe::Run) and the GPU command thread: only their time
/// is counted.
inline thread_local bool recording_thread = false;
inline thread_local bool command_thread = false;
inline std::uint32_t a_decode_depth = 0;
inline std::uint32_t a_decode_pause_depth = 0;
inline std::uint64_t a_decode_excluded_cycles = 0;

inline void ExcludeFromADecode(std::uint64_t cycles) noexcept {
    if (command_thread && a_decode_depth > 0 && a_decode_pause_depth == 0) {
        a_decode_excluded_cycles += cycles;
    }
}

struct PauseADecodeScope {
    std::uint64_t start = 0;
    bool active = false;
    PauseADecodeScope() noexcept {
        if (Enabled() && command_thread && a_decode_depth > 0) {
            active = true;
            if (a_decode_pause_depth++ == 0) {
                start = __rdtsc();
            }
        }
    }
    ~PauseADecodeScope() noexcept {
        if (active && --a_decode_pause_depth == 0 && start) {
            a_decode_excluded_cycles += (__rdtsc() - start);
        }
    }
};

struct Scope {
    explicit Scope(Id id_)
        : id{id_}, start{Enabled() && (recording_thread || command_thread) ? __rdtsc() : 0} {
        if (start && id == ADecode && command_thread) {
            if (a_decode_depth++ == 0) {
                a_decode_pause_depth = 0;
                a_decode_excluded_cycles = 0;
            }
        }
    }
    ~Scope() {
        if (start) {
            std::uint64_t delta = __rdtsc() - start;
            if (id == ADecode && command_thread && --a_decode_depth == 0) {
                delta = delta > a_decode_excluded_cycles ? (delta - a_decode_excluded_cycles) : 0;
                a_decode_excluded_cycles = 0;
            }
            auto& counters = recording_thread ? cycles_b : cycles_a;
            counters[id].store(counters[id].load(std::memory_order_relaxed) + delta,
                               std::memory_order_relaxed);
        }
    }
    Id id;
    std::uint64_t start;
};

/// Game-engine (`FD4TaskExecutor::RunTask` / `0x103d3b0`) per-`SprjTaskGroup` + per-task X-Ray profiler.
/// Zero static TLS; tracks Thread 0 (Main), Worker threads (T1+), and exact task function VMAs.
struct MainXRay {
    static constexpr std::size_t MaxGroups = 128;
    static constexpr std::size_t MaxTasks = 256;
    static constexpr std::size_t NameCap = 40;
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxGroups> cycles_t0{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxGroups> cycles_wk{};
    std::array<std::atomic<std::uint8_t>, MaxGroups> name_ready{};
    char names[MaxGroups][NameCap]{};
    std::atomic<std::uint32_t> group_count{0};

    alignas(64) std::array<std::atomic<std::uint64_t>, MaxTasks> task_cycles{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxTasks> task_calls{};
    std::array<std::uint64_t, MaxTasks> task_fn_off{};
    std::array<std::int16_t, MaxTasks> task_group_slot{};
    std::atomic<std::uint32_t> task_count{0};

    // Sub-timers for the 5 call buckets inside `SceneSub` (`0x1ed82d0`, 100% of `AdhocDraw`):
    // 0: GX_LightCull (0x2194340/0x2293060/0x2192d20), 1: Scene_1f5c1b0,
    // 2: Draw_1db6d70 (3x), 3: Queue_f5d780 (8x), 4: Init_Misc (0x1ede680/0x20b55b0/0xe47290/0xe47840/0x1d97bc0)
    static constexpr std::size_t AdhocSubCount = 5;
    alignas(64) std::array<std::atomic<std::uint64_t>, AdhocSubCount> adhoc_sub_cycles{};
    std::atomic<std::int32_t> movemap_last_state{-1};
    std::atomic<bool> adhoc_sub_installed{false};

    // Per-callback breakdown inside `sub_2293060` (`dantelion_gx/06_DeferredLightingAndAO`):
    static constexpr std::size_t MaxPassCallbacks = 64;
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxPassCallbacks> pass_cycles{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxPassCallbacks> pass_calls{};
    std::array<std::uint64_t, MaxPassCallbacks> pass_fn_off{};
    std::array<std::uint32_t, MaxPassCallbacks> pass_kind{}; // 1=prep(+0x20), 2=exec(+0x38), 3=sort, 4=submit
    std::atomic<std::uint32_t> pass_count{0};
    std::atomic<std::uint32_t> pass_last_items{0};

    // GXWorkerTaskPool (`dantelion_gx/04_SceneRenderer/GXRenderThreadAndWorkerTaskPool.cpp`) X-Ray:
    // 0: main_work, 1: main_spin, 2: wake_sem, 3: wait_idle, 4: flush_calls, 5: wake_calls
    alignas(64) std::array<std::atomic<std::uint64_t>, 6> gxpool_stats{};
    std::atomic<std::uint32_t> gxpool_workers{0};
    static constexpr std::size_t MaxParTasks = 64;
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxParTasks> par_tot_cycles{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxParTasks> par_t0_cycles{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxParTasks> par_calls{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxParTasks> par_items{};
    alignas(64) std::array<std::atomic<std::uint64_t>, MaxParTasks> par_max_cycles{};
    std::array<std::uint64_t, MaxParTasks> par_fn_off{};
    std::atomic<std::uint32_t> par_count{0};

    std::atomic<bool> installed{false};
    std::atomic<bool> deep_enabled{false};
};
inline MainXRay main_xray{};
} // namespace BbSections

#define BB_SECTION_JOIN2(a, b) a##b
#define BB_SECTION_JOIN(a, b) BB_SECTION_JOIN2(a, b)
#define BB_SECTION(id) BbSections::Scope BB_SECTION_JOIN(bb_section_scope_, __LINE__){BbSections::id}
