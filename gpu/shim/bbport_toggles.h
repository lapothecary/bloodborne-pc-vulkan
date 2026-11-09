// bbport: optimizations that can be switched off while the game runs (BB_TOGGLE_FILE,
// see runtime_memory.c), to find which one changes rendering without restarting.
#pragma once
#include <array>
#include <atomic>
#include <csetjmp>
#include <chrono>
#include <cstdint>
#include <cstdlib>

extern "C" std::uint64_t runtime_disabled_optimizations;
/// Temporary experiment bits: the second number in BB_TOGGLE_FILE (runtime_memory.c).
extern "C" std::uint64_t runtime_experiment_bits;
/// Recovery point for speculative guest memory reads on this thread (runtime_memory.c).
extern "C" __thread sigjmp_buf* runtime_fault_recover;

namespace BbToggle {
enum : std::uint64_t {
    RegionCache = 1,
    FetchShaderCache = 2,
    PageTrackingEarlyExit = 4,
    PendingPollLimit = 8,
    ThreadedRecording = 16,
    ImageDescCache = 32,
    LockFreeUploadCheck = 64,
    FindImageCache = 128,
    DeferredUploads = 256,
    AccessMemo = 512, ///< retired (the access tracker merges ranges itself)
    TextureBindingMemo = 1024,
    CoarseReadTracking = 2048,
    PreparedResources = 4096,
    DrawPreparation = 8192,
    DeferredStreamCopies = 16384,
    HotPages = 32768,
    FaultWindow = 65536,
    ParallelCopies = 131072,
    AsyncFences = 262144,
    PoolSmallCopies = 524288,
    RecordPrefetch = 1ull << 32,
    TextureViewMemo = 1ull << 33,
    TextureBindHelper = 1ull << 34,
    EarlyDrawInputs = 1ull << 35,
    ConstantRing = 1ull << 36,
    DrawPipeline = 1ull << 37,
    PipelinedTasks = 1ull << 38,
    PendingFenceWaits = 1ull << 39,
    PipelinedDispatch = 1ull << 40,
    RecorderFences = 1ull << 41,
    PipelinedMemoryWrites = 1ull << 42,
    MultiCopyShader = 1ull << 43,
    RenderStateMemo = 1ull << 44,
    TextureSetMemo = 1ull << 45,
    PipelinedIndirectDraws = 1ull << 46,
    SceneAttachmentsOnly = 1ull << 47,
    SampleSceneProxies = 1ull << 48,
    OrderedGuestWrites = 1ull << 49,
    SceneHalfRes = 1ull << 50, ///< live scaling also reduces the 960x540 post targets
    /// Clears of proxied scene images clear the proxy too instead of resolving it first.
    SceneClearProxies = 1ull << 20,
    UpdateImageFastPath = 1u << 30,
    /// Bit set: the texture collector frees nothing (A/B of its evictions while the game runs).
    TextureCollector = 1ull << 31,
    // TAA A/B in one run: optional techniques, off by default (no measured gain, 2026-10-02).
    TaaTonemapBlend = 1ull << 51,
    TaaClip = 1ull << 52,
    TaaVariance = 1ull << 53,
    TaaFilter = 1ull << 54,
    // On by default (bit set: off): history of a thin feature this jitter phase missed is kept
    // when nothing moves. Static-camera flicker of railings/window bars p99.9 -45% (2026-10-03).
    TaaKeepNearerHistory = 1ull << 55,
    /// FSR 4 (4.0 and 4.1.1) gets the motion vectors with y negated (motionVectorScale.y = -1); FSR 3
    /// and TAA keep them. A/B of the vertical convention FSR 4 expects.
    Fsr4MotionYFlip = 1ull << 56,
    SceneMipBias = 1ull << 57, ///< negative LOD bias of G-buffer samplers at reduced scene sizes
    /// The command stream is cut into segments recorded in parallel (BB_VK_RECORD_THREADS);
    /// bit set: no new cuts, one recording thread at a time.
    ParallelRecording = 1ull << 58,
    /// Finished GPU writes are read back on the readback queue (BufferCache::DownloadMemory);
    /// bit set: on the graphics queue after everything queued, as before.
    ReadbackQueue = 1ull << 59,
    /// Camera vectors from the inverse of the view matrix itself instead of the game's stored
    /// inverse; bit set: the other one of BB_CAMERA_INVERSE (A/B while the game runs).
    CameraOwnInverse = 1ull << 60,
    /// Scene textures use BB_ANISO (16) times anisotropic filtering instead of the game's
    /// ratio; bit set: the game's own samplers (A/B while the game runs).
    ForcedAniso = 1ull << 61,
    /// Camera motion vectors: the screen y sign derived from the G-buffer viewport, inverted
    /// (A/B of the y convention while the game runs).
    CameraYFlip = 1ull << 62,
    /// FSR 4 on the scaled presets gets the tonemapped frame decoded to linear light (its output
    /// encoded again); bit set: the encoded frame as before (A/B while the game runs).
    Fsr4EncodedInput = 1ull << 63,
    // Bits 20-29 are used as raw debug toggles by the camera/object motion and the upscaler.
};
inline bool Disabled(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_disabled_optimizations, __ATOMIC_RELAXED) & bit) != 0;
}
/// A temporary experiment switched on by bit `bit` of the second number in BB_TOGGLE_FILE.
inline bool Experiment(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_experiment_bits, __ATOMIC_RELAXED) & bit) != 0;
}
} // namespace BbToggle

namespace BbStats {
/// Guest writes caught by page protection, and pages currently left unprotected as hot.
inline std::atomic<std::uint64_t> tracker_faults{0};
/// Monotonic write-fault counter for HUD deltas (never zeroed by BB_FRAME_STATS).
inline std::atomic<std::uint64_t> tracker_faults_total{0};
/// Writes the GPU side heard of from the writers (libc copies over watched pages, no write tracking).
inline std::atomic<std::uint64_t> cpu_write_notes{0}, cpu_write_note_bytes{0};
/// Ranges the game's GPU memory allocator handed out (guest hook, BB_GUEST_IN_PLACE).
inline std::atomic<std::uint64_t> gpu_range_allocs{0}, gpu_range_alloc_bytes{0};
/// Buffer bindings (BB_GUEST_IN_PLACE): bytes bound where the game's memory is (read over PCIe on a
/// discrete GPU) and bytes bound to VRAM copies.
inline std::atomic<std::uint64_t> bound_in_place_bytes{0}, bound_vram_bytes{0};
inline std::atomic<std::uint64_t> bound_in_place_written_bytes{0};
/// Command processor writes put into the GPU's VRAM data (no readback); blocks whose GPU data went
/// back to the game's memory before moving in place; downloads of GPU data (readbacks).
inline std::atomic<std::uint64_t> gpu_data_updates{0}, copied_back_blocks{0}, readbacks{0};
inline std::atomic<std::uint64_t> late_vram_writes{0}; ///< labels and the like put into VRAM copies
/// BB_GUEST_IN_PLACE: copies of in-place data the CPU writes into VRAM for reading (ShadowCopy).
inline std::atomic<std::uint64_t> shadow_copies{0}, shadow_bytes{0};
/// BB_HONEST_LABELS: submissions sent early because the GPU had finished everything before them.
inline std::atomic<std::uint64_t> idle_flushes{0};
inline std::atomic<std::uint64_t> drain_flushes{0}; ///< submissions sent as the GPU reached the last one
/// EOP fences with data: decoded, and their labels written (a growing gap: lost fences, guest leaks).
inline std::atomic<std::uint64_t> eop_decoded{0}, eop_written{0}, gpu_labels{0};
/// True legacy PM4 EventWriteEopTask packets with data (excludes DirectVk label/query packets).
inline std::atomic<std::uint64_t> legacy_eop_decoded{0};
/// N1 Native VkBuffer direct hits vs BufferCache emulator fallback resolutions.
inline std::atomic<std::uint64_t> native_buf_hits{0}, native_buf_misses{0};
/// N2.1 Native VkImageView/VkSampler direct handle hits vs legacy T#/S# fallback resolutions.
inline std::atomic<std::uint64_t> native_tex_hits{0}, native_tex_misses{0};
/// N2.2 Incremental Push Descriptor slots deduplicated vs total graphics descriptor slots bound.
inline std::atomic<std::uint64_t> desc_push_saved{0}, desc_push_total{0};
/// G2.2 Render passes upgraded from AttachmentLoadOp::eLoad to eDontCare vs total color passes.
inline std::atomic<std::uint64_t> dont_care_passes{0}, total_render_passes{0};
/// sceGnmAreSubmitsAllowed calls, and those that answered no (submission lock held).
inline std::atomic<std::uint64_t> submits_allowed_queries{0}, submits_refused{0}, submit_done_calls{0};
/// Guest time blocked in Gnm submissions on the previous frame (submission lock or BB_SUBMIT_LOCK=frame).
inline std::atomic<std::uint64_t> gnm_frame_waits{0}, gnm_frame_wait_ns{0};
/// Submissions to the compute queues (sceGnmDingDong).
inline std::atomic<std::uint64_t> asc_submits{0};
/// The game's GPU range allocators (seen at their collector, 0x26aa860): object, live ranges, bytes.
inline std::array<std::atomic<std::uint64_t>, 4> range_allocators{}, range_live{}, range_bytes{};
inline std::atomic<std::int64_t> hot_pages{0};
/// Stall diagnostics (BB_FRAME_STATS): per-frame deltas printed for frames over 40 ms.
inline std::atomic<std::uint64_t> images_registered{0};
inline std::atomic<std::uint64_t> image_upload_bytes{0};
inline std::atomic<std::uint64_t> buffer_upload_bytes{0};
inline std::atomic<int> gpu_thread_clock{-1}; ///< clockid_t of the GPU command thread
inline std::atomic<std::uint64_t> draws{0}, dispatches{0}, submissions{0};
/// Frames the GPU command thread has started (display pass), for per-frame diagnostics.
inline std::atomic<std::uint64_t> gpu_frames{0};
/// Wall time spent in operations suspected of stalls (ns, all threads).
inline std::atomic<std::uint64_t> t_resident{0}, t_protect{0}, t_image_create{0}, t_refresh{0},
    t_staging{0}, t_host_wait{0}, t_copy{0}, copy_bytes{0}, t_read_faults{0}, read_faults{0},
    t_write_faults{0}, t_copy_cpu{0}, copy_sys_us{0}, copy_minflt{0};
/// Layer memory module (BB_LAYER_MEMORY), wall time per kind (ns, inclusive): 0 mprotect for write
/// traps, 1 promotions, 2 demotions, 3 trap processing, 4 yielding VRAM, 5 volatile refreshes,
/// 6 republishing paged entries, 7 paged uploads; and the mprotect calls.
inline std::array<std::atomic<std::uint64_t>, 12> t_layer{};
/// Page table entries written (blocks); 8-11 of t_layer: their staging request, guest view, mirrors,
/// copy command.
inline std::atomic<std::uint64_t> layer_entries_written{0};
inline std::atomic<std::uint64_t> layer_mprotects{0};
/// Diagnostics are collected only with BB_FRAME_STATS=1.
inline const bool enabled = [] {
    const char* env = std::getenv("BB_FRAME_STATS");
    return env && env[0] == '1';
}();
struct Timer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start =
        enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~Timer() {
        if (!enabled) {
            return;
        }
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
/// Wall time the GPU command thread waited for guest submissions (ns).
inline std::atomic<std::uint64_t> gpu_idle_ns{0};
/// Draws recorded into the reduced scene targets, and draws after the scene started.
inline std::atomic<std::uint64_t> reduced_draws{0}, scene_draws{0};
/// Wall time spent blocked in the scheduler (ns): waiting for the recording thread to drain,
/// for host copies before a submission or fence, and for GPU ticks.
inline std::atomic<std::uint64_t> sync_recording_ns{0}, host_copies_wait_ns{0}, tick_wait_ns{0},
    copy_threads_wait_ns{0}, host_copy_waits{0}, host_copy_waits_skipped{0};
/// Wall time the frame preparation waited for the GPU to finish an earlier frame (BB_FRAMES_AHEAD):
/// how GPU-bound the frames are (BB_FRAME_LOG).
inline std::atomic<std::uint64_t> present_wait_ns{0};
/// V7.3: Hardware GPU execution vs inter-submission idle gap nanoseconds (zero-RenderPass-break
/// submission-boundary Vulkan timestamp queries, collected 7 frames behind without waiting).
inline std::atomic<std::uint64_t> hw_gpu_busy_ns{0}, hw_gpu_gap_ns{0}, hw_gpu_frames{0};
/// Per-category hardware GPU render-pass breakdown (10 semantic pass buckets, sum == hw_gpu_busy_ns).
enum HudGpuCat : std::uint8_t {
    GpuCat_Shadows = 0,        // 1. Shadow Maps (CSM / Spot / VSM)
    GpuCat_GBuffer = 1,        // 2. G-Buffer (3D Scene & Foliage)
    GpuCat_Ssao = 2,           // 3. SSAO (Ambient Occlusion & Bilateral Blur)
    GpuCat_DeferredLight = 3,  // 4. Deferred PBR Lighting, Light Volumes & Sky/Fog
    GpuCat_Ssr = 4,            // 5. Water / Reflections (SSR Raymarch & Mip Chain)
    GpuCat_ForwardAlpha = 5,   // 6. Forward Alpha (Hair, Blood, Cloth, VFX)
    GpuCat_ComputeCopies = 6,  // 7. Compute & Copies (Particles, HLE, Transfers)
    GpuCat_PostFxYebis = 7,    // 8. Yebis 2 Post-FX (DoF, MotionBlur, Bloom, 3D LUT)
    GpuCat_AntiAliasing = 8,   // 9. Anti-Aliasing & Upscaler (SMAA / TAA / FSR / DLSS)
    GpuCat_UiPresent = 9,      // 10. Scaleform UI / HUD & Present (RCAS)
    GpuCat_Count = 10,
    GpuCat_Gap = 0xFF
};
/// Fine-grained X-Ray GPU sub-pass breakdown (21 buckets, sum per group == parent GpuCat).
enum HudGpuSubCat : std::uint8_t {
    // 1. Shadow Maps (GpuCat_Shadows)
    GpuSub_Shadow_CascadeAtlas = 0, // 1a. 4096x4096 CSM & Spot Depth
    GpuSub_Shadow_VsmAndPcf = 1,    // 1b. VSM Blur & Screen-Space PCF
    // 2. G-Buffer (GpuCat_GBuffer)
    GpuSub_GBuf_WorldAndChr = 2,    // 2a. World & Character 6-MRT PBR
    GpuSub_GBuf_FoliageAndCloth = 3,// 2b. Foliage & Two-Sided Cutout
    // 3. SSAO (GpuCat_Ssao)
    GpuSub_Ssao_GenReproj = 4,      // 3a. SSAO Raw Gen & Reproject
    GpuSub_Ssao_BilateralBlur = 5,  // 3b. SSAO Bilateral Upscale/Blur
    // 4. Deferred PBR & Sky/Fog (GpuCat_DeferredLight)
    GpuSub_Def_DirectLocalPbr = 6,  // 4a. Sun/IBL & Local Point Lights
    GpuSub_Def_SkyFogShafts = 7,    // 4b. Sky Dome, Fog & Light Shafts
    // 5. SSR Reflections & Mips (GpuCat_Ssr)
    GpuSub_Ssr_Raymarch = 8,        // 5a. Half-Res SSR Raymarching
    GpuSub_Ssr_HiZAndMips = 9,      // 5b. Hi-Z Depth & Glossy Mip Blur
    // 6. Forward Alpha (GpuCat_ForwardAlpha)
    GpuSub_Fwd_HairClothAlpha = 10, // 6a. Hair, Blood, Cloth & Water
    GpuSub_Fwd_SoftParticles = 11,  // 6b. Soft Particles & Flipbook VFX
    // 7. Compute & Copies (GpuCat_ComputeCopies)
    GpuSub_Comp_ParticlesCs = 12,   // 7a. GPU Particle Sim & Compute
    GpuSub_Comp_CopiesAndHle = 13,  // 7b. Buffer/Image Copies & HLE
    // 8. Yebis 2 Post-FX (GpuCat_PostFxYebis)
    GpuSub_Post_DoF = 14,           // 8a. Depth of Field (CoC & Bokeh)
    GpuSub_Post_MotionBlur = 15,    // 8b. Motion Blur (Velocity & Blur)
    GpuSub_Post_BloomGlare = 16,    // 8c. Bloom, Emissive Glow & Glare
    GpuSub_Post_TonemapLut = 17,    // 8d. HDR->LDR Tonemap & 3D LUT
    // 9. Anti-Aliasing & Upscaler (GpuCat_AntiAliasing)
    GpuSub_Aa_Upscaler = 18,        // 9a. Anti-Aliasing / Upscaler Pass
    // 10. Scaleform UI & Present (GpuCat_UiPresent)
    GpuSub_Ui_ScaleformHud = 19,    // 10a. Scaleform Game HUD & Menus
    GpuSub_Ui_RcasPresent = 20,     // 10b. RCAS Sharpen, Overlay & Flip
    GpuSub_Count = 21,
    GpuSub_Gap = 0xFF
};
constexpr std::uint8_t ParentGpuCat(std::uint8_t subcat) noexcept {
    switch (subcat) {
    case GpuSub_Shadow_CascadeAtlas:
    case GpuSub_Shadow_VsmAndPcf:
        return GpuCat_Shadows;
    case GpuSub_GBuf_WorldAndChr:
    case GpuSub_GBuf_FoliageAndCloth:
        return GpuCat_GBuffer;
    case GpuSub_Ssao_GenReproj:
    case GpuSub_Ssao_BilateralBlur:
        return GpuCat_Ssao;
    case GpuSub_Def_DirectLocalPbr:
    case GpuSub_Def_SkyFogShafts:
        return GpuCat_DeferredLight;
    case GpuSub_Ssr_Raymarch:
    case GpuSub_Ssr_HiZAndMips:
        return GpuCat_Ssr;
    case GpuSub_Fwd_HairClothAlpha:
    case GpuSub_Fwd_SoftParticles:
        return GpuCat_ForwardAlpha;
    case GpuSub_Comp_ParticlesCs:
    case GpuSub_Comp_CopiesAndHle:
        return GpuCat_ComputeCopies;
    case GpuSub_Post_DoF:
    case GpuSub_Post_MotionBlur:
    case GpuSub_Post_BloomGlare:
    case GpuSub_Post_TonemapLut:
        return GpuCat_PostFxYebis;
    case GpuSub_Aa_Upscaler:
        return GpuCat_AntiAliasing;
    case GpuSub_Ui_ScaleformHud:
    case GpuSub_Ui_RcasPresent:
        return GpuCat_UiPresent;
    default:
        return GpuCat_Gap;
    }
}
inline std::array<std::atomic<std::uint64_t>, GpuCat_Count> hw_gpu_cat_ns{};
inline std::array<std::atomic<std::uint64_t>, GpuSub_Count> hw_gpu_subcat_ns{};
/// Active CPU execution nanoseconds of the game's Main thread (CLOCK_THREAD_CPUTIME_ID, excluding FPS cap sleep).
inline std::atomic<std::uint64_t> main_cpu_busy_ns{0}, main_cpu_frames{0};
/// Frames shown (Presenter::PrepareFrame), and bytes the background pre-upload put into the arena.
inline std::atomic<std::uint32_t> frame_number{0};
inline std::atomic<std::uint64_t> preupload_bytes{0};
/// Device memory allocated (VMA blocks, arena residency) and freed, bytes.
inline std::atomic<std::uint64_t> device_alloc_bytes{0}, device_free_bytes{0};
/// Memory statistics: registered images (their guest size) and how many; the buffer cache's VRAM
/// for guest blocks (allocated, never returned to the driver), and the part of it unused (its
/// free list and the rest of the current 64 MiB block).
inline std::atomic<std::uint64_t> live_image_bytes{0}, live_images{0};
/// Every Vulkan image we allocated (their memory as allocated: host sizes, mips, scene targets).
inline std::atomic<std::uint64_t> vk_image_bytes{0};
inline std::atomic<std::uint64_t> residency_alloc_bytes{0}, residency_unused_bytes{0};
/// The texture cache collector: the usage it compares, the mark it starts at, images it freed.
inline std::atomic<std::uint64_t> gc_used_bytes{0}, gc_trigger_bytes{0}, gc_freed_images{0};
/// Its critical mark (images used a moment ago evicted past it): VRAM copies of the game's
/// memory stop short of it and yield first (BufferCache::VramPromotionsPaused, LayerProcessIdle).
inline std::atomic<std::uint64_t> gc_critical_bytes{0};
/// Seconds of a steady clock, updated at every guest submission (cheap ages for caches).
inline std::atomic<std::uint32_t> coarse_second{0};
/// VRAM blocks moved back in place when idle, unbound with the memory the game unmapped, and
/// residency chunks given back to the driver.
inline std::atomic<std::uint64_t> vram_idle_bytes{0}, vram_unmapped_bytes{0}, vram_chunks_freed_bytes{0};
/// Dynamic blocks the loaders wrote whole since: they may have a VRAM copy again (ProcessPendingAssets).
inline std::atomic<std::uint64_t> dynamic_reloaded_bytes{0};
/// GPU writes (binding sizes) kept in place because the CPU wrote their blocks (layer_cpu_blocks).
inline std::atomic<std::uint64_t> gpu_writes_kept_by_cpu_bytes{0};
/// Residency chunks, chunks sent away (ProcessIdleBlocks), and their blocks that had to stay.
inline std::atomic<std::uint64_t> residency_chunk_count{0}, evacuations{0}, evac_kept_gpu{0}, evac_kept_other{0};
/// Images the collector passed over (GPU-written, not evictable without memory pressure).
inline std::atomic<std::uint64_t> gc_kept_images{0};
struct WaitTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~WaitTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count(),
                        std::memory_order_relaxed);
    }
};
/// GPU thread rusage, refreshed after each graphics submission.
inline std::atomic<std::uint64_t> gpu_sys_us{0}, gpu_user_us{0}, gpu_invol_switches{0},
    gpu_vol_switches{0}, gpu_minor_faults{0};
/// Protection faults (signals) taken by the GPU thread itself.
inline std::atomic<std::uint64_t> gpu_signal_faults{0};
/// Protection changes: calls and pages, those removing write access (TLB shootdowns) apart.
inline std::atomic<std::uint64_t> protect_calls{0}, protect_pages{0}, protect_revoke_calls{0},
    protect_revoke_pages{0};
} // namespace BbStats
