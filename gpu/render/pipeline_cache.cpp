// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: program and pipeline selection (see pipeline_cache.h).

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <unistd.h>

#include "bb_golden_engine.h"
#include "bbport_sections.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "buffers/direct_buffers.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "cp/command_processor.h"
#include "game_profile.h"
#include "gcn/decoder.h"
#include "render/direct_shaders.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline_cache.h"
#include "render/pipeline_store.h"
#include "render/presenter.h"
#include "render/recipe_catalog.h"
#include "render/scheduler.h"
#include "render/shader_capture.h"
#include "render/vk_draw_prep.h"
#include "shader/backend/spirv/emit_spirv.h"
#include "shader/backend/spirv/spirv_opt.h"
#include "shader/info.h"
#include "shader/runtime_info.h"
#include "upscale/motion_history.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

// Shader/pipeline compile time on the GPU thread, reported by BB_FRAME_STATS.
std::atomic<u64> g_bb_compile_ns;
std::atomic<u32> g_bb_compiles;

namespace {

constexpr u32 SpirvVersion1_6 = 0x00010600U;

constexpr std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

void AddCompileTime(std::chrono::steady_clock::time_point start) {
    g_bb_compile_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count());
}

struct CompileTimer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~CompileTimer() {
        AddCompileTime(start);
        ++g_bb_compiles;
    }
};

/// The vertex outputs as the hardware lays them out (VS_OUT_CNTL): the misc vector, then the
/// clip/cull distance vectors that are enabled.
u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const Gcn::VsOutputControl& ctl) {
    u32 num_outputs = 0;
    if (ctl.vs_out_misc_enable) {
        auto& misc = outputs[num_outputs++];
        misc[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc[1] = ctl.use_vtx_edge_flag
                      ? Output::EdgeFlag
                      : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc[2] = ctl.use_vtx_kill_flag
                      ? Output::KillFlag
                      : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }
    const auto distance = [&](u32 index) {
        if (ctl.IsClipDistEnabled(index)) {
            return Output(u32(Output::ClipDist0) + index);
        }
        return ctl.IsCullDistEnabled(index) ? Output(u32(Output::CullDist0) + index)
                                            : Output::None;
    };
    const std::array<bool, 2> distance_vectors = {bool(ctl.vs_out_ccdist0_enable),
                                                  bool(ctl.vs_out_ccdist1_enable)};
    for (u32 vector = 0; vector < distance_vectors.size(); ++vector) {
        if (distance_vectors[vector]) {
            auto& map = outputs[num_outputs++];
            for (u32 component = 0; component < map.size(); ++component) {
                map[component] = distance(vector * 4 + component);
            }
        }
    }
    return num_outputs;
}

/// Clip distances (guest-exported, or user clip planes the vertex stage lowers) need the
/// fragment stage to discard where the host has no clip distances (NVIDIA): it then reads them
/// at input locations after the vertex outputs.
bool NeedsClipDistanceEmulation(const Gcn::Regs& regs, const Shader::Profile& profile) {
    const bool lowers_user_clip_planes =
        regs.clipper_control.user_clip_plane_enable &&
        !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
    return ((regs.vs_output_control.clip_distance_enable &&
             !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
            lowers_user_clip_planes) &&
           profile.needs_clip_distance_emulation;
}

bool IsDualSourceBlend(const Gcn::BlendControl& blend) {
    if (!blend.enable) {
        return false;
    }
    bool dual = GcnToVk::IsDualSourceBlendFactor(blend.color_dst_factor) ||
                GcnToVk::IsDualSourceBlendFactor(blend.color_src_factor);
    if (blend.separate_alpha_blend) {
        dual |= GcnToVk::IsDualSourceBlendFactor(blend.alpha_dst_factor) ||
                GcnToVk::IsDualSourceBlendFactor(blend.alpha_src_factor);
    }
    return dual;
}

Gcn::PrimitiveType GsInputPrimitive(const Gcn::Regs& regs) {
    if (regs.stage_enable.raw != Gcn::ShaderStageEnable::LsHsEsGs) {
        return regs.primitive_type;
    }
    switch (regs.tess_config.topology) {
    case Gcn::TessellationTopology::Point:
        return Gcn::PrimitiveType::PointList;
    case Gcn::TessellationTopology::Line:
        return Gcn::PrimitiveType::LineList;
    case Gcn::TessellationTopology::TriangleCw:
    case Gcn::TessellationTopology::TriangleCcw:
        return Gcn::PrimitiveType::TriangleList;
    default:
        UNREACHABLE();
    }
}

/// The render target a draw goes to: its first color buffer written, else its depth buffer,
/// else 0.
u64 DrawTarget(const Gcn::Regs& regs) {
    if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
        for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
            if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
                return regs.color_buffers[cb].Address();
            }
        }
    }
    return regs.depth_buffer.DepthValid() ? regs.depth_buffer.DepthAddress() : 0;
}

Shader::Profile MakeProfile(const Instance& instance, u32 sparse_page_shift) {
    const auto& vk12 = instance.GetVk12Properties();
    const auto driver = instance.GetDriverID();
    const bool nvidia = driver == vk::DriverId::eNvidiaProprietary;
    return Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() && nvidia,
        // Older NVIDIA (Pascal) has no barycentrics; BB_INTERP_INT_FIX=0/1 overrides.
        .needs_integer_interpolation_fix = [&] {
            if (const char* env = std::getenv("BB_INTERP_INT_FIX")) {
                return env[0] == '1';
            }
            return !instance.IsFragmentShaderBarycentricSupported() && nvidia;
        }(),
        .needs_lds_barriers = nvidia || driver == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = driver == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = nvidia,
        .supports_shader_stencil_export = instance.IsShaderStencilExportSupported(),
        // BB_LAYER_MEMORY: buffers over nearly all memory go through the page table.
        .paged_buffers = VideoCore::BufferCache::LayerPagedActive(),
        // The game's buffer copy shader, from its profile (games/).
        .buffer_copy_shader_hash = Game::BufferCopyShader(),
    };
}

} // namespace

/// BB_ASYNC_PIPELINES: worker threads creating graphics pipelines (the driver's compile,
/// vkCreateGraphicsPipelines, with its own VkPipelineCache synchronization). Jobs start in order.
class PipelineCompiler {
public:
    explicit PipelineCompiler(u32 threads) {
        for (u32 i = 0; i < threads; ++i) {
            workers.emplace_back([this] { Run(); });
        }
    }
    ~PipelineCompiler() {
        {
            std::scoped_lock lk{mutex};
            stop = true;
        }
        cv.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }
    void Submit(std::function<void()> job) {
        {
            std::scoped_lock lk{mutex};
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }

private:
    void Run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, [&] { return stop || !jobs.empty(); });
                if (jobs.empty()) {
                    return; // stopping, every job done
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
        }
    }
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> workers;
    bool stop = false;
};

struct PipelineCache::PendingPipeline {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport sdata{};
};

struct AsyncCompileCtx {
    struct StageSnap {
        bool valid = false;
        VAddr pgm_base = 0;
        u64 hash = 0;
        std::array<u32, Gcn::NUM_USER_DATA> user_data{};
        std::vector<u32> code;
        std::vector<u32> flat_buf; // Walked synchronously on Stage A at draw time (never stale!)
    };
    struct BuiltStage {
        HwStage hw_stage{};
        SwStage sw_stage{};
        u64 hash = 0;
        VAddr pgm_base = 0;
        size_t assumed_perm = 0;
        Shader::Info info;
        Shader::StageSpecialization spec;
        vk::ShaderModule module;
        std::vector<u32> raw_spv;
        std::vector<u32> opt_spv;
    };
    std::array<StageSnap, 8> stages{};
    std::vector<u32> vs_copy_code;
    u64 vs_copy_hash = 0;
    std::deque<std::vector<u32>> owned_blobs; // CUE / FetchShader copies referenced by user_data
    std::deque<Shader::Info> owned_infos;     // Stable Info clones for hits/builds during worker key refresh
    std::deque<BuiltStage> built_stages;
    bool failed = false;
};

struct PipelineCache::AsyncSpvJob {
    u64 sig = 0;
    Gcn::Regs regs_copy{};
    DrawIndirectParams params{};
    AsyncCompileCtx ctx{};
    GraphicsPipelineKey key{};
    std::array<u64, MaxShaderStages> stage_pgm_hashes{};
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport sdata{};
    bool ok = false;
};

using Gcn::RegDirty;

struct PipelineCache::FastSelect {
    struct CachedFetch {
        std::vector<u32> words;
        Shader::Gcn::FetchShaderData data;
    };
    struct SpecWordCheck {
        u16 dw_off = 0;
        u8 kind = 0; ///< 0=(flat[off]&mask)==exp, 1=flat[off]!=0, 2=(flat[off]|(flat[off+1]&0x3f))!=0, 3=non-contig T#
        u8 pad = 0;
        u32 mask = 0;
        u32 expected = 0;
    };
    struct Stage {
        Program* program = nullptr;
        HwStage hw_stage{};
        SwStage sw_stage{};
        VAddr pgm_base = 0;
        u64 shader_hash = 0;
        VAddr copy_pgm_base = 0;
        u64 copy_shader_hash = 0;
        u32 perm_idx = 0;
        const u32* last_fetch_code = nullptr;
        const CachedFetch* last_cached_fetch = nullptr;
        bool rt_verified = false;
        bool fast_check_attempted = false;
        u16 fast_check_count = 0;
        u16 fast_check_min_flat = 0;
        u32 fast_check_offset = 0;
        Shader::Backend::Bindings fast_check_start{};
        const Shader::Gcn::FetchShaderData* fast_vs_fetch = nullptr;
        u16 fast_vs_valid_mask = 0;
        u8 fast_vs_attr_count = 0;
        std::array<u32, 16> fast_vs_dw3_exp{};
        u64 last_flat_hash = 0;
        u32 last_flat_words = 0;
        std::array<u32, 4> last_ud_head{};
        Shader::Backend::Bindings last_start{};
        Shader::RuntimeInfo runtime_info{};
    };
    static constexpr u32 MaxStages = 4; ///< VS (2), EsGs (3), or LsHs tessellation (4: PS+HS+VS+LS)
    static constexpr u32 MaxSigWords = 96;
    static constexpr u32 NumSets = 512;
    static constexpr u32 NumWays = 4;
    static constexpr u32 TableSize = NumSets * NumWays;

    struct Entry {
        u64 key = 0;
        u64 last_use = 0;
        u64 shader_epoch = 0;
        const GraphicsPipeline* pipeline = nullptr;
        u16 sig_words = 0;
        u8 num_stages = 0;
        bool motion = false;
        bool motion_possible = false;
        std::array<u32, MaxSigWords> sig{};
        std::array<Stage, MaxStages> stages{};
    };

    /// Compact L1D-resident key mirror (`16 KiB`): probing all 4 ways reads 1 cache line instead
    /// of 4 scattered `Entry`-strided cache lines!
    std::array<u64, TableSize> hot_keys{};
    /// Compact L1D-resident MRU way hint (`512 B`): tries the last-winning way first.
    std::array<u8, NumSets> mru_way{};
    std::vector<SpecWordCheck> fast_checks_pool;
    std::array<Entry, TableSize> entries{};
    std::unordered_map<const u32*, CachedFetch> fetch_map;
    std::array<std::pair<const u32*, const CachedFetch*>, 64> fetch_mru{};
    u64 use_clock = 0;

    u64 last_sig_epoch = 0;
    DrawIndirectParams last_params{};
    VAddr last_positions_addr = 0;
    bool last_obj_motion = false;
    int last_upscaler = -1;
    size_t last_set_base = 0;
    u32 last_matching_ways_mask = 0;
    Entry* last_first_match = nullptr;

    std::array<Stage, MaxStages> pending{};
    u32 num_pending = 0;
    bool pending_ok = true;
    bool pending_motion_possible = false;

    const GraphicsPipeline* last_vsharps_pipeline = nullptr;
    boost::container::static_vector<Gcn::Buffer, Shader::NUM_BUFFERS> last_vsharps;

    struct VsMemo {
        const Program* program = nullptr;
        u32 perm_idx = 0;
        const Shader::Gcn::FetchShaderData* fetch = nullptr;
        u32 ud_words = 0;
        std::array<u32, 16> ud_full{};
        boost::container::static_vector<Gcn::Buffer, Shader::NUM_BUFFERS> vsharps{};
    } vs_memo{};

    u64 hits = 0, full = 0, miss_regs = 0, miss_program = 0, miss_spec = 0, miss_motion = 0;
    u64 not_eligible = 0, verify_bad = 0, calls = 0;
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};

struct PipelineCache::MtFastPsoTable {
    static constexpr u32 MaxStages = FastSelect::MaxStages;
    static constexpr u32 MaxSigWords = FastSelect::MaxSigWords;
    static constexpr u32 NumSets = FastSelect::NumSets;
    static constexpr u32 NumWays = FastSelect::NumWays;
    static constexpr u32 TableSize = NumSets * NumWays;

    struct StageSnap {
        Shader::Info info_template{};
        Shader::StageSpecialization spec{};
        Shader::RuntimeInfo runtime_info{};
        HwStage hw_stage{};
        SwStage sw_stage{};
        VAddr pgm_base = 0;
        u64 shader_hash = 0;
        VAddr copy_pgm_base = 0;
        u64 copy_shader_hash = 0;
        bool rt_verified = false;
    };

    struct Entry {
        u64 key = 0;
        u64 publish_seq = 0;
        mutable std::atomic<u64> shader_epoch{0};
        const GraphicsPipeline* pipeline = nullptr;
        u16 sig_words = 0;
        u8 num_stages = 0;
        bool motion = false;
        bool motion_possible = false;
        std::array<u32, MaxSigWords> sig{};
        std::array<StageSnap, MaxStages> stages{};
    };

    std::array<std::atomic<const Entry*>, TableSize> slots{};
    std::mutex publish_mutex;
    std::vector<std::unique_ptr<Entry>> arena;
    u64 publish_clock = 0;
};

void PipelineCache::InvalidateFastSelect() {
    if (fast) {
        fast->hot_keys.fill(0ULL);
    fast->mru_way.fill(0u);
        fast->fast_checks_pool.clear();
        for (auto& e : fast->entries) {
            e.key = 0;
            e.shader_epoch = 0;
            e.pipeline = nullptr;
            for (auto& s : e.stages) {
                s.fast_check_count = 0;
                s.fast_check_attempted = false;
            }
        }
        fast->fetch_map.clear();
        fast->fetch_mru.fill({});
        fast->use_clock = 0;
        fast->last_sig_epoch = 0;
        fast->last_first_match = nullptr;
        fast->last_matching_ways_mask = 0;
        fast->last_vsharps_pipeline = nullptr;
        fast->last_vsharps.clear();
        fast->vs_memo = {};
    }
    if (mt_fast) {
        std::scoped_lock lk{mt_fast->publish_mutex};
        for (auto& slot : mt_fast->slots) {
            slot.store(nullptr, std::memory_order_release);
        }
    }
}

bool PipelineCache::ConsumeFastVsharps(
    const GraphicsPipeline* pipeline,
    boost::container::static_vector<Gcn::Buffer, Shader::NUM_BUFFERS>& out) noexcept {
    if (!fast || !pipeline || fast->last_vsharps_pipeline != pipeline) {
        return false;
    }
    fast->last_vsharps_pipeline = nullptr;
    out = fast->last_vsharps;
    return true;
}

const boost::container::static_vector<Gcn::Buffer, Shader::NUM_BUFFERS>*
PipelineCache::PeekFastVsharps(const GraphicsPipeline* pipeline) noexcept {
    if (!fast || !pipeline || fast->last_vsharps_pipeline != pipeline) {
        return nullptr;
    }
    fast->last_vsharps_pipeline = nullptr;
    return &fast->last_vsharps;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             BbCp::CommandProcessor* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes},
      profile{MakeProfile(instance_, sparse_page_shift)} {
    fast = std::make_unique<FastSelect>();
    mt_fast = std::make_unique<MtFastPsoTable>();
    sel.regs = liverpool ? &liverpool->regs : nullptr;

    // Keep the 6 ms manifest index synchronous so `bb_golden_engine.cpp` never races with it.
    DirectShaders::InitializeManifestLibrary();

    const char* async_env = std::getenv("BB_ASYNC_PIPELINES");
    if (!(async_env && async_env[0] == '0') && instance.IsVertexInputDynamicState()) {
        const char* thr_env = std::getenv("BB_ASYNC_PIPELINE_THREADS");
        const u32 default_threads = std::clamp(std::thread::hardware_concurrency() / 3u, 2u, 8u);
        const u32 threads = std::max(
            thr_env && *thr_env ? u32(std::strtoul(thr_env, nullptr, 10)) : default_threads,
            1u);
        compiler = std::make_unique<PipelineCompiler>(threads);
        std::printf("[bb_vulkan] V4.2 Async PipelineCompiler prewarmed with %u worker threads\n",
                    threads);
    }

    const bool auto_bake = (liverpool != nullptr && EmulatorSettings.IsPipelineCacheEnabled());
    const char* sync_prewarm_env = std::getenv("BB_PREWARM_SYNC");
    const char* export_env = std::getenv("BB_EXPORT_RECIPES");
    const bool sync_prewarm =
        (sync_prewarm_env && sync_prewarm_env[0] == '1') ||
        (export_env && export_env[0] == '1');
    if (sync_prewarm) {
        WarmUp();
        prewarm_phase.store(8u, std::memory_order_release);
        prewarm_done.store(true, std::memory_order_release);
        if (auto_bake) {
            StartBackgroundAutoBake();
        }
    } else {
        // Overlap V4.2 disk/driver PSO prewarm (~0.4-0.8 s) with `eboot.bin` CPU/IO boot (~1.31 s):
        // finishes before the first Vulkan draw call while cutting boot-to-menu time!
        prewarm_thread = std::thread([this, auto_bake]() {
            Common::SetCurrentThreadName("bb:PsoPrewarm");
            prewarm_tid.store(u32(gettid()), std::memory_order_release);
            WarmUp();
            if (auto_bake && !prewarm_cancel.load(std::memory_order_acquire)) {
                prewarm_phase.store(7u, std::memory_order_release);
                prewarm_done_items.store(0u, std::memory_order_relaxed);
                prewarm_total_items.store(0u, std::memory_order_relaxed);
                StartBackgroundAutoBake();
            }
            prewarm_tid.store(0u, std::memory_order_release);
            prewarm_phase.store(8u, std::memory_order_release);
            prewarm_done.store(true, std::memory_order_release);
        });
    }
}

void PipelineCache::EnsurePrewarmDone() {
    if (prewarm_done.load(std::memory_order_acquire)) [[likely]] {
        return;
    }
    std::scoped_lock lk{prewarm_join_mutex};
    if (prewarm_thread.joinable()) {
        prewarm_thread.join();
    }
}

PipelineCache::~PipelineCache() {
    prewarm_cancel.store(true, std::memory_order_release);
    {
        std::scoped_lock lk{prewarm_join_mutex};
        if (prewarm_thread.joinable()) {
            prewarm_thread.join();
        }
    }
    StopBackgroundAutoBake();
    compiler.reset();
    {
        std::scoped_lock lk{async_spv_mutex};
        for (auto& job : completed_spv_jobs) {
            for (auto& bs : job->ctx.built_stages) {
                if (bs.module) {
                    instance.GetDevice().destroyShaderModule(bs.module);
                }
            }
        }
        completed_spv_jobs.clear();
        pending_spv_jobs.clear();
    }
    if (driver_cache_save_thread.joinable()) {
        driver_cache_save_thread.join();
    }
    if (driver_cache_dirty.load(std::memory_order_relaxed)) {
        SaveDriverPipelineCache();
    }
    RecipeCatalog::Get().Shutdown();
    const auto tel = DirectShaders::ReadTelemetry();
    if (tel.registered_binaries != 0 || tel.stage_a_direct_hits != 0 ||
        tel.worker_direct_hits != 0) {
        std::printf("[bb_vulkan] V4.1 DirectShaders telemetry: registered=%llu "
                    "(manifest=%llu/%u), StageA_hits=%llu, DrawPrep_hits=%llu\n",
                    static_cast<unsigned long long>(tel.registered_binaries),
                    static_cast<unsigned long long>(tel.manifest_binaries_bound),
                    tel.manifest_total,
                    static_cast<unsigned long long>(tel.stage_a_direct_hits),
                    static_cast<unsigned long long>(tel.worker_direct_hits));
    }
    DirectShaders::ClearCachedPrograms();
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(PipelineSelection& sel, HwStage stage,
                                                           SwStage l_stage) {
    auto& info = sel.runtime_infos[u32(l_stage)];
    const auto& regs = *sel.regs;
    const auto common = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        common(regs.ls_program);
        // The hull stage is selected first: its constant buffer has the LS output stride.
        Shader::TessellationDataConstantBuffer tess_constants{};
        sel.infos[u32(SwStage::TessellationControl)]->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        common(regs.hs_program);
        break;
    case HwStage::Export:
        common(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        common(regs.gs_program);
        auto& gs = info.hw.gs;
        gs.num_outputs = MapOutputs(gs.outputs, regs.vs_output_control);
        gs.output_vertices = regs.vgt_gs_max_vert_out;
        gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        gs.in_primitive = GsInputPrimitive(regs);
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs.out_primitive[stream_id] = regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs.mode = regs.vgt_gs_mode.mode;
        // The copy shader (the VS program) moves the GS ring's vertices to the outputs.
        DirectShaders::Snapshot copy_snap;
        const VAddr copy_base = regs.vs_program.Address<VAddr>();
        if (sel.async_ctx && !sel.async_ctx->vs_copy_code.empty()) {
            gs.vs_copy = sel.async_ctx->vs_copy_code;
            gs.vs_copy_hash = sel.async_ctx->vs_copy_hash;
        } else if (DirectShaders::TryLookup(copy_base, copy_snap, sel.worker != nullptr) &&
                   copy_snap.cpu_immutable != 0) {
            gs.vs_copy = std::span<const u32>{reinterpret_cast<const u32*>(copy_base),
                                              copy_snap.code_dwords};
            gs.vs_copy_hash = copy_snap.shader_hash;
        } else {
            const auto copy = Gcn::GetParams(regs.vs_program);
            gs.vs_copy = copy.code;
            gs.vs_copy_hash = copy.hash;
        }
        DumpShader(gs.vs_copy, gs.vs_copy_hash, HwStage::Vertex, 0, "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        common(regs.vs_program);
        auto& vs = info.hw.vs;
        vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        vs.num_outputs = MapOutputs(vs.outputs, regs.vs_output_control);
        vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW;
        vs.clip_disable = regs.IsClipDisabled();
        vs.motion_vectors = sel.motion;
        break;
    }
    case HwStage::Fragment: {
        common(regs.ps_program);
        auto& fs = info.hw.fs;
        fs.en_flags = regs.ps_input_ena;
        fs.addr_flags = regs.ps_input_addr;
        fs.num_inputs = regs.num_interp;
        fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        fs.z_export_format = regs.z_export_format;
        fs.motion_vectors = sel.motion;
        const auto& depth = regs.depth_shader_control;
        const u8 stencil_export =
            depth.stencil_op_val_export_enable | depth.stencil_test_val_export_enable;
        fs.mrtz_mask = depth.z_export_enable | (stencil_export << 1) |
                       (depth.mask_export_enable << 2) | (depth.coverage_to_mask_enable << 3);
        fs.dual_source_blending = IsDualSourceBlend(regs.blend_control[0]);
        for (u32 i = 0; i < regs.num_interp; i++) {
            const auto& input = regs.ps_inputs[i];
            fs.inputs[i] = {
                .param_index = u8(input.input_offset),
                .is_default = bool(input.use_default),
                .is_flat = bool(input.flat_shade),
                .default_value = u8(input.default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            fs.color_buffers[i] = sel.graphics_key.color_buffers[i];
        }
        fs.clip_distance_emulation = NeedsClipDistanceEmulation(regs, profile);
        break;
    }
    case HwStage::Compute: {
        const auto& cs = liverpool->GetCsRegs();
        info.props.num_user_data = cs.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs.settings.fp_round_mode64;
        info.hw.cs.workgroup_size = {cs.num_thread_x.full, cs.num_thread_y.full,
                                     cs.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs.IsTgidEnabled(0), cs.IsTgidEnabled(1), cs.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = sel.draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = sel.draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == Gcn::PrimitiveType::RectList ||
            regs.primitive_type == Gcn::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl:
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    case SwStage::TessellationEval:
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    default:
        break;
    }
    return info;
}

bool PipelineCache::PrepareGraphicsPipeline(PipelineSelection& worker_sel) {
    if (!prewarm_done.load(std::memory_order_acquire)) [[unlikely]] {
        return false;
    }
    // Tessellation stages read constant buffers from memory at selection time: not prepared.
    if (worker_sel.regs->stage_enable.hs_en) {
        return false;
    }
    // Single shared_lock + single sigsetjmp across all stages of the draw (cuts cross-CCD
    // programs_mutex atomic RMWs and sigsetjmp overhead in half on bb:DrawPrep0..1).
    std::shared_lock lk{programs_mutex};
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        runtime_fault_recover = nullptr;
        worker_sel.worker->failed = true;
        return false;
    }
    runtime_fault_recover = &recover;
    const bool ok = RefreshGraphicsKey(worker_sel) && !worker_sel.worker->failed;
    runtime_fault_recover = nullptr;
    return ok;
}

namespace {
/// The fetch shader a vertex program calls (through a pointer in its user data) decodes up to
/// its s_setpc_b64: the parse asserts on anything else. Memory the game is still filling, or
/// reused, holds garbage there (another game: "illegal instruction" while loading).
bool FetchCodeDecodes(const u32* code) {
    const u32* pc = code;
    for (u32 i = 0; i < 512; ++i) {
        const auto encoding = GcnIsa::EncodingOf(*pc);
        if (encoding == GcnIsa::InstEncoding::ILLEGAL) {
            return false;
        }
        const bool vector = encoding == GcnIsa::InstEncoding::VOP1 ||
                            encoding == GcnIsa::InstEncoding::VOP2 ||
                            encoding == GcnIsa::InstEncoding::VOPC;
        if (vector && ((*pc & 0x1FF) == 0xF9 || (*pc & 0x1FF) == 0xFA)) {
            return false; // SDWA/DPP: not in a PS4 program
        }
        if (GcnIsa::Decode(pc).opcode == GcnIsa::Opcode::S_SETPC_B64) {
            return true;
        }
    }
    return false;
}
} // namespace

/// The GPU's writes before this draw, for a program that is not whole in memory (a copy into
/// place still on the GPU): at most once per address every 100 ms (thousands of draws a frame
/// use code that is simply not there; each wait is a whole GPU round trip). False: not tried.
bool PipelineCache::SyncProgramMemory(VAddr address, u64 size) {
    if (!program_memory_sync) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (const auto tried = program_sync_tried.find(address);
        tried != program_sync_tried.end() && now - tried->second < std::chrono::milliseconds(100)) {
        return false;
    }
    if (program_sync_tried.size() >= 4096) {
        program_sync_tried.clear();
    }
    program_sync_tried[address] = now;
    program_memory_sync(address, size);
    return true;
}

/// Every program the games' compilers emit starts with "s_mov_b32 vcc_hi, length" and ends in
/// an "OrbShdr" header. Anything else at a program address is not a program yet: the game copied
/// it into place with the GPU (DMA) and the CPU's view does not have it (or only its start), or
/// it freed the memory. The header search would find the next program's header (or none: an
/// assertion) and the translator would decode garbage. On the GPU command thread (`may_sync`)
/// what the GPU wrote there is read back first; still nothing: the draw is skipped. A draw
/// preparation worker gives such a draw to the GPU command thread.
bool PipelineCache::LooksLikeProgram(const u32* code, HwStage stage, bool may_sync) {
    if (may_sync && program_write_sync) {
        program_write_sync(reinterpret_cast<VAddr>(code), 256);
    }
    // Its header where its first instruction says: a header found further on (the search) may
    // be the next program's while this one is still being copied in.
    const auto complete = [code] { return Gcn::DirectBinaryInfo(code) != nullptr; };
    if (complete()) {
        return true;
    }
    const u32 first = code[0];
    if (may_sync && SyncProgramMemory(reinterpret_cast<VAddr>(code), 0x10000)) {
        if (complete()) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_INFO(Render_Vulkan, "stage {} program at {} arrived from the GPU's writes",
                         u32(stage), fmt::ptr(code));
            }
            return true;
        }
        // Read back and still no header in place: a program laid out differently (the header
        // searched for, as before), else not a program.
        if (first == 0xBEEB03FF && code[0] == 0xBEEB03FF && Gcn::FindBinaryInfo(code)) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan,
                            "stage {} program at {}: its header is not where its first "
                            "instruction says; the next header found is used",
                            u32(stage), fmt::ptr(code));
            }
            return true;
        }
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
        LOG_WARNING(Render_Vulkan,
                    "stage {} program at {} is not a whole program ({:#010x} {:#010x} {:#010x} "
                    "{:#010x}, {:#010x} before a read back, header {}): draw skipped",
                    u32(stage), fmt::ptr(code), code[0], code[1], code[2], code[3], first,
                    Gcn::FindBinaryInfo(code) ? "found" : "missing");
    }
    return false;
}

/// A new program decodes whole before it is translated. A program the GPU copied into place
/// (DMA_DATA, CP DMA) may not be in the CPU's view of the memory yet: then what the GPU wrote is
/// read back once and the program decoded again. Still not decoding: the draw is skipped.
bool PipelineCache::ProgramDecodes(const Shader::ShaderParams& params, HwStage stage) {
    const auto first_illegal = [&]() -> s64 {
        const u32* pc = params.code.data();
        const u32* const end = pc + params.code.size();
        while (pc < end) {
            const auto encoding = GcnIsa::EncodingOf(*pc);
            if (encoding == GcnIsa::InstEncoding::ILLEGAL) {
                return pc - params.code.data();
            }
            // SDWA (0xF9) and DPP (0xFA) operands came with later GCN generations: in a PS4
            // program such a word is garbage, not an instruction.
            const bool vector = encoding == GcnIsa::InstEncoding::VOP1 ||
                                encoding == GcnIsa::InstEncoding::VOP2 ||
                                encoding == GcnIsa::InstEncoding::VOPC;
            if (vector && ((*pc & 0x1FF) == 0xF9 || (*pc & 0x1FF) == 0xFA)) {
                return pc - params.code.data();
            }
            GcnIsa::Decode(pc);
        }
        return -1;
    };
    s64 bad = first_illegal();
    if (bad < 0) {
        return true;
    }
    const VAddr address = reinterpret_cast<VAddr>(params.code.data());
    const u64 bytes = params.code.size_bytes();
    const u32 word = params.code[bad];
    const s64 at = bad;
    if (SyncProgramMemory(address, bytes)) {
        bad = first_illegal();
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "stage {} program {:#x} at {:#x} ({} bytes): word {:#010x} at dword {} does "
                    "not decode; {}",
                    u32(stage), params.hash, address, bytes, word, at,
                    bad < 0 ? "decodes after reading back what the GPU wrote"
                            : "still not after a read back: draw skipped");
    }
    return bad < 0;
}

const GraphicsPipeline* PipelineCache::TryPreparedPipeline(const PreparedDraw& prepared) {
    EnsurePrewarmDone();
    BB_SECTION(APrepared);
    if (prepared.state.load(std::memory_order_acquire) != PreparedDraw::Ready ||
        prepared.reg_checksum != liverpool->gfx_reg_checksum) {
        return nullptr;
    }
    // Same registers; the stage programs and their flattened user data (which also covers
    // everything read from guest memory for the specialization) must match as well. This is
    // the per-stage work GetProgram does on the regular path.
    const auto& regs = liverpool->regs;
    for (u32 i = 0; i < prepared.num_stages; ++i) {
        const auto& stage = prepared.stages[i];
        const auto* pgm = regs.ProgramForStage(static_cast<u32>(stage.hw_stage));
        if (!pgm) {
            return nullptr;
        }
        const VAddr base = pgm->Address<VAddr>();
        if (base != stage.pgm_base || base == 0) {
            return nullptr;
        }
        DirectShaders::Snapshot direct_snap;
        if (DirectShaders::TryLookup(base, direct_snap, /*is_worker=*/false) &&
            direct_snap.cpu_immutable != 0) {
            if (direct_snap.shader_hash != stage.hash) {
                return nullptr;
            }
        } else {
            if (program_write_sync) {
                program_write_sync(base, 256);
            }
            const auto* bin = Gcn::DirectBinaryInfo(reinterpret_cast<const u32*>(base));
            if (!bin || bin->shader_hash != stage.hash) {
                return nullptr; // not a program or changed: the regular path skips or reselects
            }
        }
        auto& info = const_cast<Program*>(stage.program)->info;
        info.pgm_base = base;
        info.user_data = pgm->user_data;
        info.RefreshFlatBuf();
        if (info.flattened_ud_buf.size() != stage.flat_size ||
            std::memcmp(info.flattened_ud_buf.data(), stage.flat,
                        stage.flat_size * sizeof(u32)) != 0) {
            return nullptr;
        }
    }
    if (last_graphics_pipeline && last_graphics_pipeline->GetGraphicsKey() == prepared.key) {
        return last_graphics_pipeline;
    }
    const auto it = prepared.key_hash ? graphics_pipelines.find(prepared.key, prepared.key_hash)
                                      : graphics_pipelines.find(prepared.key);
    if (it != graphics_pipelines.end()) {
        last_graphics_pipeline = it->second.get();
        return last_graphics_pipeline;
    }
    return nullptr;
}

namespace {
/// Whether a G-buffer vertex shader gets the motion attachment (its bone palette buffers):
/// shared by the full selection (RefreshGraphicsKey) and the fast select.
bool VsSelectsMotion(const Shader::Info& vs) {
    const u32* const vs_flat = vs.FlatUserData().data();
    const VAddr vs_base = vs.ProgramBase();
    for (const auto& resource : vs.buffers) {
        if (resource.IsSpecial()) {
            continue;
        }
        const auto buffer = resource.GetSharpDirect(vs_flat, vs_base);
        if (buffer.Valid() && buffer.GetStride() == 16 &&
            Motion::ClassifyBuffer(buffer.GetSize()) != Motion::BufferRole::Other) {
            return true;
        }
    }
    return false;
}

bool AllMotionEnv() {
    // BB_OBJECT_MOTION_ALL=1: the old broad selection, for visual A/B tests.
    static const bool all_motion = [] {
        const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
        return value && value[0] == '1';
    }();
    return all_motion;
}

bool FastSelectEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_FAST_SELECT");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

bool FastSelectVerify() {
    static const bool verify = [] {
        const char* env = std::getenv("BB_FAST_SELECT_VERIFY");
        return env && env[0] == '1';
    }();
    return verify;
}

/// Packs every register field and global setting that `RefreshGraphicsKey` + `BuildRuntimeInfo`
/// read in `VgtStages::Vs`, `VgtStages::EsGs`, or `VgtStages::LsHs` + `IsVertexInputDynamicState()`
/// mode into `out`. Returns 0 if the draw configuration is not eligible for fast selection.
u32 BuildFastSelectSig(const Gcn::Regs& regs, const DrawIndirectParams& params,
                       bool dynamic_vertex_input, bool tessellation_supported,
                       std::array<u32, 96>& out) noexcept {
    if (!dynamic_vertex_input || regs.num_interp > 32u) {
        return 0;
    }
    const bool is_vs = (regs.stage_enable.raw == Gcn::ShaderStageEnable::VgtStages::Vs);
    const bool is_esgs = (regs.stage_enable.raw == Gcn::ShaderStageEnable::VgtStages::EsGs &&
                          !regs.vgt_gs_mode.onchip && !regs.vgt_strmout_config.raw);
    const bool is_lshs =
        (regs.stage_enable.raw == Gcn::ShaderStageEnable::VgtStages::LsHs && tessellation_supported);
    if (!is_vs && !is_esgs && !is_lshs) {
        return 0;
    }
    std::memcpy(&out[0], &regs.reg_array[0x2C08], 4 * sizeof(u32)); // PS address + settings
    std::memcpy(&out[4], &regs.reg_array[0x2C48], 4 * sizeof(u32)); // VS address + settings

    const bool obj_motion = BbSettings::Get().object_motion.load(std::memory_order_relaxed);
    const int upscaler = BbSettings::EffectiveUpscaler();
    const bool has_pos = (Shader::MotionVectors::positions_address != 0);

    out[8] = u32(params.vertex_sgpr_offset) | (u32(params.instance_sgpr_offset) << 16);
    out[9] = u32(obj_motion) |
             (u32(has_pos) << 1) |
             (u32(upscaler & 0xFF) << 4) |
             (u32(regs.stage_enable.raw & 0x7u) << 9) |
             (u32(!regs.depth_render_override.disable_viewport_clamp) << 12) |
             (u32(regs.polygon_control.provoking_vtx_last) << 13) |
             (u32(regs.polygon_control.PolyMode()) << 14) |
             (u32(regs.barycentric_control.front_face_all_bits) << 18) |
             (u32(IsDualSourceBlend(regs.blend_control[0])) << 19) |
             (u32(regs.aa_config.NumSamples()) << 20);
    out[10] = u32(regs.primitive_type) | (u32(regs.stage_enable.raw) << 16);
    out[11] = regs.reg_array[0xA204]; // PA_CL_CLIP_CNTL
    out[12] = regs.reg_array[0xA207]; // PA_CL_VS_OUT_CNTL
    out[13] = u32(regs.color_control.mode) |
              (u32(regs.color_control.rop3) << 8) |
              (u32(regs.z_export_format) << 16);
    out[14] = regs.color_shader_mask.raw;
    out[15] = regs.color_target_mask.raw;
    out[16] = regs.color_export_format.raw;
    out[17] = regs.reg_array[0xA203]; // DB_SHADER_CONTROL
    out[18] = regs.reg_array[0xA1B3]; // SPI_PS_INPUT_ENA
    out[19] = regs.reg_array[0xA1B4]; // SPI_PS_INPUT_ADDR
    out[20] = regs.vgt_instance_step_rate_0;
    out[21] = regs.vgt_instance_step_rate_1;

    const auto& db = regs.depth_buffer;
    const bool db_d = db.DepthValid();
    const bool db_s = db.StencilValid();
    const bool color_disabled =
        (regs.color_control.mode == Gcn::ColorControl::OperationMode::Disable);
    u32 bound_mask = 0;
    u32 idx = 23;
    for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        if (!color_disabled && regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
            bound_mask |= (1u << cb);
            const auto& col_buf = regs.color_buffers[cb];
            out[idx++] = (col_buf.info.raw & 0x00023FFCu) |
                         (u32(col_buf.attrib.num_fragments_log2) << 24) |
                         0x80000000u;
            out[idx++] = (regs.blend_control[cb].enable && !col_buf.info.blend_bypass)
                             ? regs.reg_array[0xA1E0 + cb]
                             : 0u;
        }
    }
    const u32 n_interp = regs.num_interp;
    out[22] = (db_d ? (u32(db.z_info.format) | 0x8u) : 0u) |
              ((db_s ? (u32(db.stencil_info.format) | 0x8u) : 0u) << 4) |
              (u32((db_d || db_s) ? db.NumSamples() : 1u) << 8) |
              (bound_mask << 14) |
              (n_interp << 22) |
              (u32(bool(regs.color_buffers[Shader::MotionVectors::Output])) << 28);
    if (n_interp != 0) {
        std::memcpy(&out[idx], &regs.reg_array[0xA191], n_interp * sizeof(u32));
        idx += n_interp;
    }
    if (is_esgs) {
        std::memcpy(&out[idx], &regs.reg_array[0x2CC8], 4 * sizeof(u32)); // ES address + settings
        idx += 4;
        std::memcpy(&out[idx], &regs.reg_array[0x2C88], 4 * sizeof(u32)); // GS address + settings
        idx += 4;
        out[idx++] = regs.vgt_esgs_ring_itemsize;
        out[idx++] = regs.vgt_gs_max_vert_out;
        out[idx++] = std::bit_cast<u32>(regs.vgt_gs_instance_cnt);
        out[idx++] = std::bit_cast<u32>(regs.vgt_gs_out_prim_type);
        out[idx++] = regs.vgt_gs_vert_itemsize[0];
        out[idx++] = std::bit_cast<u32>(regs.vgt_gs_mode);
    }
    if (is_lshs) {
        std::memcpy(&out[idx], &regs.reg_array[0x2D08], 4 * sizeof(u32)); // HS address + settings
        idx += 4;
        std::memcpy(&out[idx], &regs.reg_array[0x2D48], 4 * sizeof(u32)); // LS address + settings
        idx += 4;
        out[idx++] = std::bit_cast<u32>(regs.ls_hs_config);
        out[idx++] = std::bit_cast<u32>(regs.tess_config);
    }
    return idx;
}
} // namespace


void PipelineCache::NoteFastProgram(Program* program, HwStage hw_stage, SwStage sw_stage,
                                    VAddr pgm_base, u64 shader_hash, size_t perm_idx) {
    if (!fast || sw_stage == SwStage::Compute) {
        return;
    }
    auto& f = *fast;
    if (f.num_pending >= FastSelect::MaxStages) {
        f.pending_ok = false;
        return;
    }
    auto& st = f.pending[f.num_pending++];
    st.program = program;
    st.hw_stage = hw_stage;
    st.sw_stage = sw_stage;
    st.pgm_base = pgm_base;
    st.shader_hash = shader_hash;
    st.copy_pgm_base = 0;
    st.copy_shader_hash = 0;
    st.perm_idx = static_cast<u32>(perm_idx);
    st.last_fetch_code = nullptr;
    st.last_cached_fetch = nullptr;
    st.rt_verified = true;
    st.fast_check_attempted = false;
    st.fast_check_count = 0;
    st.fast_check_min_flat = 0;
    st.fast_check_offset = 0;
    st.fast_check_start = {};
}

void PipelineCache::CommitFastSelect(const GraphicsPipeline* pipeline,
                                     const DrawIndirectParams& params) {
    auto& f = *fast;
    f.last_sig_epoch = 0;
    f.last_first_match = nullptr;
    f.last_matching_ways_mask = 0;
    const auto& regs = liverpool->regs;
    if (!pipeline || !f.pending_ok || f.num_pending == 0) {
        ++f.not_eligible;
        return;
    }
    std::array<u32, FastSelect::MaxSigWords> sig;
    const u32 sig_len =
        BuildFastSelectSig(regs, params, instance.IsVertexInputDynamicState(),
                           instance.IsTessellationSupported(), sig);
    if (sig_len == 0) {
        return;
    }
    const u64 sig_key = XXH3_64bits(sig.data(), sig_len * sizeof(u32)) | 1ULL;
    const size_t set_base =
        ((sig_key >> 1) & (FastSelect::NumSets - 1)) * FastSelect::NumWays;

    FastSelect::Entry* target = nullptr;
    size_t target_idx = set_base;
    for (u32 w = 0; w < FastSelect::NumWays; ++w) {
        if (f.hot_keys[set_base + w] != sig_key) {
            continue;
        }
        auto& cand = f.entries[set_base + w];
        if (cand.pipeline == pipeline && cand.sig_words == sig_len &&
            std::memcmp(cand.sig.data(), sig.data(), sig_len * sizeof(u32)) == 0) {
            target = &cand;
            target_idx = set_base + w;
            break;
        }
    }
    if (!target) {
        target = &f.entries[set_base];
        target_idx = set_base;
        for (u32 w = 1; w < FastSelect::NumWays; ++w) {
            auto& cand = f.entries[set_base + w];
            if (cand.key == 0) {
                target = &cand;
                target_idx = set_base + w;
                break;
            }
            if (cand.last_use < target->last_use) {
                target = &cand;
                target_idx = set_base + w;
            }
        }
    }

    f.hot_keys[target_idx] = sig_key;
    target->key = sig_key;
    target->last_use = ++f.use_clock;
    target->shader_epoch = DirectShaders::InvalidationEpoch();
    target->pipeline = pipeline;
    target->sig_words = static_cast<u16>(sig_len);
    target->num_stages = static_cast<u8>(f.num_pending);
    target->motion = sel.motion;
    target->motion_possible = f.pending_motion_possible;
    std::memcpy(target->sig.data(), sig.data(), sig_len * sizeof(u32));
    for (u32 i = 0; i < f.num_pending; ++i) {
        target->stages[i] = f.pending[i];
        target->stages[i].runtime_info =
            sel.runtime_infos[static_cast<u32>(f.pending[i].sw_stage)];
        if (target->stages[i].hw_stage == HwStage::Geometry) {
            const VAddr copy_base = regs.vs_program.Address<VAddr>();
            target->stages[i].copy_pgm_base = copy_base;
            u64 copy_hash = 0;
            if (copy_base != 0) {
                DirectShaders::Snapshot copy_snap;
                if (DirectShaders::TryLookup(copy_base, copy_snap, /*is_worker=*/false)) {
                    copy_hash = copy_snap.shader_hash;
                } else if (const auto* bin =
                               Gcn::DirectBinaryInfo(reinterpret_cast<const u32*>(copy_base))) {
                    copy_hash = bin->shader_hash;
                }
            }
            target->stages[i].copy_shader_hash = copy_hash;
        }
    }

    if (mt_fast) {
        bool snap_ok = true;
        auto entry = std::make_unique<MtFastPsoTable::Entry>();
        entry->key = sig_key;
        entry->shader_epoch.store(target->shader_epoch, std::memory_order_relaxed);
        entry->pipeline = pipeline;
        entry->sig_words = static_cast<u16>(sig_len);
        entry->num_stages = static_cast<u8>(f.num_pending);
        entry->motion = sel.motion;
        entry->motion_possible = f.pending_motion_possible;
        std::memcpy(entry->sig.data(), sig.data(), sig_len * sizeof(u32));
        for (u32 i = 0; i < f.num_pending; ++i) {
            const auto& src = target->stages[i];
            if (!src.program || !src.program->info_template ||
                src.perm_idx >= src.program->modules.size()) {
                snap_ok = false;
                break;
            }
            auto& dst = entry->stages[i];
            dst.info_template = *src.program->info_template;
            dst.spec = src.program->modules[src.perm_idx].spec;
            dst.spec.info = &dst.info_template;
            dst.runtime_info = src.runtime_info;
            dst.hw_stage = src.hw_stage;
            dst.sw_stage = src.sw_stage;
            dst.pgm_base = src.pgm_base;
            dst.shader_hash = src.shader_hash;
            dst.copy_pgm_base = src.copy_pgm_base;
            dst.copy_shader_hash = src.copy_shader_hash;
            dst.rt_verified = !dst.info_template.has_fetch_shader;
        }
        if (snap_ok) {
            std::scoped_lock lk{mt_fast->publish_mutex};
            u32 victim_way = 0;
            u64 oldest_seq = ~0ULL;
            bool duplicate = false;
            for (u32 w = 0; w < MtFastPsoTable::NumWays; ++w) {
                const auto* cur = mt_fast->slots[set_base + w].load(std::memory_order_relaxed);
                if (!cur) {
                    victim_way = w;
                    oldest_seq = 0;
                    break;
                }
                if (cur->key == sig_key && cur->pipeline == pipeline &&
                    cur->sig_words == sig_len &&
                    std::memcmp(cur->sig.data(), sig.data(), sig_len * sizeof(u32)) == 0) {
                    duplicate = true;
                    break;
                }
                if (cur->publish_seq < oldest_seq) {
                    oldest_seq = cur->publish_seq;
                    victim_way = w;
                }
            }
            if (!duplicate) {
                entry->publish_seq = ++mt_fast->publish_clock;
                const MtFastPsoTable::Entry* raw = entry.get();
                mt_fast->arena.push_back(std::move(entry));
                mt_fast->slots[set_base + victim_way].store(raw, std::memory_order_release);
            }
        }
    }
}

const GraphicsPipeline* PipelineCache::TryFastSelect(const DrawIndirectParams& params) {
    auto& f = *fast;
    f.last_vsharps_pipeline = nullptr;
    const auto& regs = liverpool->regs;
    const u64 cur_sig_epoch = liverpool->pso_sig_epoch;
    const VAddr cur_pos_addr = Shader::MotionVectors::positions_address;
    const bool cur_obj_motion = BbSettings::Get().object_motion.load(std::memory_order_relaxed);
    const int cur_upscaler = BbSettings::EffectiveUpscaler();

    size_t set_base = 0;
    FastSelect::Entry* first_match = nullptr;
    u32 matching_ways_mask = 0;

    {
        BB_SECTION(AFpsoSig);
        if (f.last_sig_epoch == cur_sig_epoch && f.last_first_match != nullptr &&
            f.last_positions_addr == cur_pos_addr && f.last_obj_motion == cur_obj_motion &&
            f.last_upscaler == cur_upscaler &&
            std::memcmp(&f.last_params, &params, sizeof(params)) == 0) {
            set_base = f.last_set_base;
            first_match = f.last_first_match;
            matching_ways_mask = f.last_matching_ways_mask;
        } else {
            std::array<u32, FastSelect::MaxSigWords> sig;
            const u32 sig_len =
                BuildFastSelectSig(regs, params, instance.IsVertexInputDynamicState(),
                                   instance.IsTessellationSupported(), sig);
            if (sig_len == 0) {
                ++f.not_eligible;
                return nullptr;
            }
            const u64 sig_key = XXH3_64bits(sig.data(), sig_len * sizeof(u32)) | 1ULL;
            set_base = ((sig_key >> 1) & (FastSelect::NumSets - 1)) * FastSelect::NumWays;

            const u64* const kptr = &f.hot_keys[set_base];
            for (u32 w = 0; w < FastSelect::NumWays; ++w) {
                if (kptr[w] == sig_key) {
                    auto& cand = f.entries[set_base + w];
                    if (cand.sig_words == sig_len &&
                        std::memcmp(cand.sig.data(), sig.data(), sig_len * sizeof(u32)) == 0) {
                        if (!first_match) {
                            first_match = &cand;
                        }
                        matching_ways_mask |= (1u << w);
                    }
                }
            }
            if (!first_match) {
                ++f.miss_regs;
                return nullptr;
            }
            f.last_sig_epoch = cur_sig_epoch;
            f.last_params = params;
            f.last_positions_addr = cur_pos_addr;
            f.last_obj_motion = cur_obj_motion;
            f.last_upscaler = cur_upscaler;
            f.last_set_base = set_base;
            f.last_first_match = first_match;
            f.last_matching_ways_mask = matching_ways_mask;
        }
    }

    const u64 cur_epoch = DirectShaders::InvalidationEpoch();
    const bool epoch_valid = (first_match->shader_epoch == cur_epoch);

    const auto verify_code_hash = [&](VAddr base, u64 expected_hash) -> bool {
        if (base == 0) {
            return false;
        }
        DirectShaders::Snapshot direct_snap;
        if (DirectShaders::TryLookup(base, direct_snap, /*is_worker=*/false) &&
            direct_snap.cpu_immutable != 0) {
            return direct_snap.shader_hash == expected_hash;
        }
        if (program_write_sync) {
            program_write_sync(base, 256);
        }
        const auto* bin = Gcn::DirectBinaryInfo(reinterpret_cast<const u32*>(base));
        return bin && bin->shader_hash == expected_hash;
    };

    const auto lookup_fetch = [&](const Shader::Info& info,
                                  FastSelect::Stage* st_slot) -> const Shader::Gcn::FetchShaderData* {
        const u32* const code =
            Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
        if (!code) {
            return nullptr;
        }
        if (st_slot && st_slot->last_cached_fetch &&
            !BbToggle::Disabled(BbToggle::FetchShaderCache)) {
            if (st_slot->last_fetch_code == code) {
                if (epoch_valid ||
                    std::memcmp(code, st_slot->last_cached_fetch->words.data(),
                                st_slot->last_cached_fetch->words.size() * sizeof(u32)) == 0) {
                    return &st_slot->last_cached_fetch->data;
                }
            } else if (!st_slot->last_cached_fetch->words.empty() &&
                       code[0] == st_slot->last_cached_fetch->words[0] &&
                       std::memcmp(code, st_slot->last_cached_fetch->words.data(),
                                   st_slot->last_cached_fetch->words.size() * sizeof(u32)) == 0) {
                st_slot->last_fetch_code = code;
                return &st_slot->last_cached_fetch->data;
            }
        }
        auto& mru = f.fetch_mru[(reinterpret_cast<uintptr_t>(code) >> 4) & 63];
        if (!BbToggle::Disabled(BbToggle::FetchShaderCache) && mru.first == code && mru.second &&
            (epoch_valid ||
             std::memcmp(code, mru.second->words.data(),
                         mru.second->words.size() * sizeof(u32)) == 0)) {
            if (st_slot) {
                st_slot->last_fetch_code = code;
                st_slot->last_cached_fetch = mru.second;
            }
            return &mru.second->data;
        }
        if (const auto it = f.fetch_map.find(code);
            !BbToggle::Disabled(BbToggle::FetchShaderCache) && it != f.fetch_map.end() &&
            (epoch_valid ||
             std::memcmp(code, it->second.words.data(),
                         it->second.words.size() * sizeof(u32)) == 0)) {
            mru = {code, &it->second};
            if (st_slot) {
                st_slot->last_fetch_code = code;
                st_slot->last_cached_fetch = &it->second;
            }
            return &it->second.data;
        }
        if (program_write_sync) {
            program_write_sync(reinterpret_cast<VAddr>(code), 256);
        }
        if (!FetchCodeDecodes(code)) {
            return nullptr;
        }
        auto parsed = Shader::Gcn::ParseFetchShader(info);
        if (!parsed || parsed->size == 0 || (parsed->size & 3u) != 0) {
            return nullptr;
        }
        if (f.fetch_map.size() >= 4096) {
            f.fetch_map.clear();
            f.fetch_mru.fill({});
            f.vs_memo.fetch = nullptr;
            for (auto& e : f.entries) {
                for (auto& s : e.stages) {
                    s.last_fetch_code = nullptr;
                    s.last_cached_fetch = nullptr;
                }
            }
        }
        auto& slot = f.fetch_map[code];
        slot.words.assign(code, code + (parsed->size / sizeof(u32)));
        slot.data = std::move(*parsed);
        mru = {code, &slot};
        if (st_slot) {
            st_slot->last_fetch_code = code;
            st_slot->last_cached_fetch = &slot;
        }
        return &slot.data;
    };

    // Refresh user data & fetch shader once per stage across all candidate ways in the set.
    std::array<const Shader::Gcn::FetchShaderData*, FastSelect::MaxStages> cur_fetch{};
    const Shader::Info* vs_info = nullptr;
    {
        BB_SECTION(AFpsoFlat);
        for (u32 i = 0; i < first_match->num_stages; ++i) {
            auto& st = first_match->stages[i];
            const auto* pgm = regs.ProgramForStage(static_cast<u32>(st.hw_stage));
            if (!pgm) {
                ++f.miss_program;
                return nullptr;
            }
            const VAddr base = pgm->Address<VAddr>();
            if (base != st.pgm_base || base == 0) {
                ++f.miss_program;
                return nullptr;
            }
            if (!epoch_valid && !verify_code_hash(base, st.shader_hash)) {
                ++f.miss_program;
                return nullptr;
            }
            if (st.hw_stage == HwStage::Geometry) {
                const VAddr copy_base = regs.vs_program.Address<VAddr>();
                if (copy_base != st.copy_pgm_base || copy_base == 0 ||
                    (!epoch_valid && !verify_code_hash(copy_base, st.copy_shader_hash))) {
                    ++f.miss_program;
                    return nullptr;
                }
            }
            auto& info = st.program->info;
            info.pgm_base = base;
            info.user_data = pgm->user_data;
            info.RefreshFlatBuf();
            if (info.has_fetch_shader) {
                cur_fetch[i] = lookup_fetch(info, &st);
                if (!cur_fetch[i]) {
                    ++f.miss_program;
                    return nullptr;
                }
            }
            if (st.sw_stage == SwStage::Vertex) {
                vs_info = &info;
            }
        }
        if (!epoch_valid) {
            first_match->shader_epoch = cur_epoch;
        }
    }

    const bool cur_motion = first_match->motion_possible &&
                            (AllMotionEnv() || (vs_info && VsSelectsMotion(*vs_info)));

    BB_SECTION(AFpsoMatch);
    bool saw_motion_mismatch = false;
    const u32 set_idx = static_cast<u32>(set_base / FastSelect::NumWays);
    const u32 mru_w = f.mru_way[set_idx] & (FastSelect::NumWays - 1u);
    for (u32 wi = 0; wi < FastSelect::NumWays; ++wi) {
        const u32 w = (mru_w + wi) & (FastSelect::NumWays - 1u);
        if (((matching_ways_mask >> w) & 1u) == 0) {
            continue;
        }
        auto& way = f.entries[set_base + w];
        if (way.num_stages != first_match->num_stages) {
            continue;
        }
        if (way.motion_possible && way.motion != cur_motion) {
            saw_motion_mismatch = true;
            continue;
        }
        bool programs_ok = true;
        std::array<const Shader::Gcn::FetchShaderData*, FastSelect::MaxStages> way_fetch =
            cur_fetch;
        for (u32 i = 0; i < way.num_stages; ++i) {
            const auto& fst = first_match->stages[i];
            auto& wst = way.stages[i];
            if (wst.hw_stage != fst.hw_stage || wst.pgm_base != fst.pgm_base ||
                wst.shader_hash != fst.shader_hash ||
                wst.copy_pgm_base != fst.copy_pgm_base ||
                wst.copy_shader_hash != fst.copy_shader_hash) {
                programs_ok = false;
                break;
            }
            if (wst.program != fst.program) {
                const auto* pgm = regs.ProgramForStage(static_cast<u32>(wst.hw_stage));
                auto& winfo = wst.program->info;
                winfo.pgm_base = wst.pgm_base;
                winfo.user_data = pgm->user_data;
                winfo.RefreshFlatBuf();
                if (winfo.has_fetch_shader) {
                    way_fetch[i] = lookup_fetch(winfo, &wst);
                    if (!way_fetch[i]) {
                        programs_ok = false;
                        break;
                    }
                } else {
                    way_fetch[i] = nullptr;
                }
            }
        }
        if (!programs_ok) {
            continue;
        }
        Shader::Backend::Bindings binding{};
        bool spec_ok = true;
        bool filled_vsharps = false;
        const Shader::Info* tcs_info = nullptr;
        for (u32 i = 0; i < way.num_stages; ++i) {
            auto& st = way.stages[i];
            const auto& modules = st.program->modules;
            auto& info = st.program->info;
            if (st.sw_stage == SwStage::TessellationControl) {
                tcs_info = &info;
            }
            if (st.hw_stage == HwStage::Local && tcs_info) {
                Shader::TessellationDataConstantBuffer tess_constants{};
                tcs_info->ReadTessConstantBuffer(tess_constants);
                if (st.runtime_info.hw.ls.ls_stride != tess_constants.ls_stride) {
                    spec_ok = false;
                    break;
                }
            }
            const bool want_vsharps =
                (st.sw_stage == SwStage::Vertex && info.has_fetch_shader && way_fetch[i] != nullptr);
            if (st.perm_idx >= modules.size()) {
                spec_ok = false;
                break;
            }
            const auto start_binding = binding;
            const auto ud_span = info.user_data;
            const u32 ud_words = static_cast<u32>(ud_span.size());
            const u32 flat_words = static_cast<u32>(info.flattened_ud_buf.size());
            const u32* const flat_ptr = info.flattened_ud_buf.data();

            // Step 2D-3: Ultra-fast compiled `SpecWordCheck` verification (~15 ns vs ~150 ns
            // `MatchesInPlace`). Every mismatch safely falls through to exact `MatchesInPlace`.
            if (st.fast_check_count != 0u && !profile.paged_buffers &&
                flat_words >= st.fast_check_min_flat && start_binding == st.fast_check_start)
                [[likely]] {
                bool fast_ok = true;
                if (want_vsharps) {
                    const auto& spec = modules[st.perm_idx].spec;
                    const auto& attrs = way_fetch[i]->attributes;
                    if (way_fetch[i] == st.fast_vs_fetch && st.fast_vs_attr_count == attrs.size() &&
                        st.fast_vs_attr_count != 0u) [[likely]] {
                        f.last_vsharps.clear();
                        const u16 vmask = st.fast_vs_valid_mask;
                        for (size_t a = 0; a < attrs.size(); ++a) {
                            const auto& desc = attrs[a];
                            const auto sharp = desc.GetSharpDirect(ud_span);
                            if (f.last_vsharps.size() < f.last_vsharps.capacity()) {
                                f.last_vsharps.push_back(sharp);
                            }
                            const bool is_valid = static_cast<bool>(sharp);
                            const bool exp_valid = ((vmask >> a) & 1u) != 0u;
                            u32 dw3 = 0u;
                            std::memcpy(&dw3, reinterpret_cast<const u8*>(&sharp) + 12u,
                                        sizeof(u32));
                            if (is_valid == exp_valid &&
                                (!is_valid || (dw3 & 0x007FFFFFu) == st.fast_vs_dw3_exp[a]))
                                [[likely]] {
                                continue;
                            }
                            // Exact fallback for equivalent `NumberClass` / `RemapSwizzle`:
                            Shader::VsAttribSpecialization aspec{};
                            if (is_valid) {
                                using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                                if (const auto step_rate = desc.GetStepRate();
                                    step_rate != InstanceIdType::None) {
                                    aspec.divisor =
                                        step_rate == InstanceIdType::OverStepRate0
                                            ? st.runtime_info.sw.vs.step_rate_0
                                            : (step_rate == InstanceIdType::OverStepRate1
                                                   ? st.runtime_info.sw.vs.step_rate_1
                                                   : 1u);
                                }
                                aspec.num_class = ::Gcn::GetNumberClass(sharp.GetNumberFmt());
                                aspec.dst_select = sharp.DstSelect();
                            }
                            if (spec.vs_attribs[a] != aspec) {
                                f.last_vsharps.clear();
                                fast_ok = false;
                                break;
                            }
                        }
                    } else if (spec.vs_attribs.size() != attrs.size()) {
                        fast_ok = false;
                    } else {
                        f.last_vsharps.clear();
                        for (size_t a = 0; a < attrs.size(); ++a) {
                            const auto& desc = attrs[a];
                            const auto sharp = desc.GetSharpDirect(ud_span);
                            if (f.last_vsharps.size() < f.last_vsharps.capacity()) {
                                f.last_vsharps.push_back(sharp);
                            }
                            Shader::VsAttribSpecialization aspec{};
                            if (sharp) {
                                using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
                                if (const auto step_rate = desc.GetStepRate();
                                    step_rate != InstanceIdType::None) {
                                    aspec.divisor =
                                        step_rate == InstanceIdType::OverStepRate0
                                            ? st.runtime_info.sw.vs.step_rate_0
                                            : (step_rate == InstanceIdType::OverStepRate1
                                                   ? st.runtime_info.sw.vs.step_rate_1
                                                   : 1u);
                                }
                                aspec.num_class = ::Gcn::GetNumberClass(sharp.GetNumberFmt());
                                aspec.dst_select = sharp.DstSelect();
                            }
                            if (spec.vs_attribs[a] != aspec) {
                                f.last_vsharps.clear();
                                fast_ok = false;
                                break;
                            }
                        }
                    }
                }
                if (fast_ok) {
                    const auto* const chk = f.fast_checks_pool.data() + st.fast_check_offset;
                    const u32 cnt = st.fast_check_count;
                    for (u32 c = 0; c < cnt; ++c) {
                        const auto& wc = chk[c];
                        if (wc.kind == 0u) [[likely]] {
                            if ((flat_ptr[wc.dw_off] & wc.mask) != wc.expected) {
                                fast_ok = false;
                                break;
                            }
                        } else if (wc.kind == 1u) {
                            if (flat_ptr[wc.dw_off] == 0u) {
                                fast_ok = false;
                                break;
                            }
                        } else if (wc.kind == 2u) {
                            if ((flat_ptr[wc.dw_off] | (flat_ptr[wc.dw_off + 1u] & 0x3Fu)) == 0u) {
                                fast_ok = false;
                                break;
                            }
                        } else {
                            if ((flat_ptr[wc.dw_off] | (flat_ptr[wc.mask] & 0x3Fu)) == 0u) {
                                fast_ok = false;
                                break;
                            }
                        }
                    }
                    if (fast_ok) [[likely]] {
                        if (want_vsharps) {
                            filled_vsharps = true;
                        }
                        info.AddBindings(binding);
                        continue;
                    }
                }
            }

            const bool can_memo_flat =
                st.hw_stage != HwStage::Local && st.sw_stage != SwStage::TessellationControl &&
                (!info.has_fetch_shader || (want_vsharps && st.rt_verified));
            u64 flat_hash = 0;
            bool head_same = false;
            if (can_memo_flat) {
                if (flat_words >= 4u && flat_words == st.last_flat_words &&
                    start_binding == st.last_start && flat_ptr[0] == st.last_ud_head[0] &&
                    flat_ptr[1] == st.last_ud_head[1] && flat_ptr[2] == st.last_ud_head[2] &&
                    flat_ptr[3] == st.last_ud_head[3]) {
                    head_same = true;
                    const bool fetch_ud_ok =
                        !want_vsharps ||
                        (f.vs_memo.program == st.program && f.vs_memo.perm_idx == st.perm_idx &&
                         f.vs_memo.fetch == way_fetch[i] && ud_words == f.vs_memo.ud_words &&
                         ud_words <= f.vs_memo.ud_full.size() &&
                         (ud_words == 0u ||
                          std::memcmp(ud_span.data(), f.vs_memo.ud_full.data(),
                                      ud_words * sizeof(u32)) == 0));
                    flat_hash = XXH3_64bits(flat_ptr, flat_words * sizeof(u32)) | 1ULL;
                    if (fetch_ud_ok && st.last_flat_hash != 0ULL &&
                        flat_hash == st.last_flat_hash) {
                        if (want_vsharps) {
                            f.last_vsharps = f.vs_memo.vsharps;
                            filled_vsharps = true;
                        }
                        info.AddBindings(binding);
                        continue;
                    }
                }
            }
            if (!modules[st.perm_idx].spec.MatchesInPlace(
                    info, st.runtime_info, profile, binding, way_fetch[i], st.rt_verified,
                    want_vsharps ? &f.last_vsharps : nullptr)) {
                spec_ok = false;
                break;
            }
            if (!st.fast_check_attempted && !profile.paged_buffers && st.rt_verified &&
                info.fmasks.empty() && st.hw_stage != HwStage::Local &&
                st.sw_stage != SwStage::TessellationControl &&
                st.sw_stage != SwStage::TessellationEval && !info.Snapshot() &&
                f.fast_checks_pool.size() < 65536u) {
                st.fast_check_attempted = true;
                const auto& spec = modules[st.perm_idx].spec;
                if (!spec.bitset.none() && spec.buffers.size() == info.buffers.size() &&
                    spec.images.size() == info.images.size() &&
                    spec.samplers.size() == info.samplers.size()) {
                    std::array<FastSelect::SpecWordCheck, 192> local_chk{};
                    u32 local_cnt = 0u;
                    u32 max_off = 0u;
                    bool ok_compile = true;
                    auto push_chk = [&](u16 off, u8 kind, u32 mask, u32 exp) noexcept {
                        if (local_cnt >= local_chk.size() || off >= flat_words) {
                            ok_compile = false;
                            return;
                        }
                        local_chk[local_cnt++] = {off, kind, 0u, mask, exp};
                        if (off > max_off) {
                            max_off = off;
                        }
                        if (kind == 2u && (u32(off) + 1u) > max_off) {
                            max_off = u32(off) + 1u;
                        } else if (kind == 3u && mask > max_off) {
                            max_off = mask;
                        }
                    };
                    auto push_masked = [&](u16 off, u32 mask) noexcept {
                        if (off >= flat_words) {
                            ok_compile = false;
                            return;
                        }
                        push_chk(off, 0u, mask, flat_ptr[off] & mask);
                    };
                    for (size_t b = 0; ok_compile && b < info.buffers.size(); ++b) {
                        const auto& desc = info.buffers[b];
                        const auto& sf = desc.sharp_fetch;
                        const auto sharp = desc.GetSharpDirect(flat_ptr, info.pgm_base);
                        if (sharp) {
                            if (sf.load_mask & 4u) {
                                push_chk(sf.offsets[2], 1u, 0u, 0u);
                            }
                            if (sf.load_mask & 2u) {
                                push_masked(sf.offsets[1], 0xBFFF0000u);
                            }
                            if (sf.load_mask & 8u) {
                                const u32 m3 = (desc.is_formatted || sharp.swizzle_enable)
                                                   ? 0xC07FFFFFu
                                                   : 0xC0000000u;
                                push_masked(sf.offsets[3], m3);
                            }
                        } else {
                            if ((sf.load_mask & 12u) == 0u) {
                                // Compile-time constant null sharp (neither dw2 nor dw3 varies).
                            } else if ((sf.load_mask & 4u) && sf.offsets[2] < flat_words &&
                                       flat_ptr[sf.offsets[2]] == 0u) {
                                push_chk(sf.offsets[2], 0u, ~0u, 0u);
                            } else if ((sf.load_mask & 8u) && sf.offsets[3] < flat_words &&
                                       (flat_ptr[sf.offsets[3]] & 0xC0000000u) != 0u) {
                                push_masked(sf.offsets[3], 0xC0000000u);
                            } else {
                                ok_compile = false;
                            }
                        }
                    }
                    for (size_t im = 0; ok_compile && im < info.images.size(); ++im) {
                        const auto& desc = info.images[im];
                        const auto& sf = desc.sharp_fetch;
                        const auto sharp = desc.GetSharpDirect(flat_ptr);
                        if (sharp) {
                            if ((sf.load_mask & 3u) == 3u) {
                                if (sf.offsets[1] == sf.offsets[0] + 1u) {
                                    push_chk(sf.offsets[0], 2u, 0u, 0u);
                                } else if (sf.offsets[1] < flat_words) {
                                    push_chk(sf.offsets[0], 3u, sf.offsets[1], 0u);
                                } else {
                                    ok_compile = false;
                                }
                            } else if ((sf.load_mask & 1u) && sf.offsets[0] < flat_words &&
                                       flat_ptr[sf.offsets[0]] != 0u) {
                                push_chk(sf.offsets[0], 1u, 0u, 0u);
                            } else if (sf.load_mask & 3u) {
                                ok_compile = false;
                            }
                            if (ok_compile && (sf.load_mask & 2u)) {
                                push_masked(sf.offsets[1], 0x3FF00000u);
                            }
                            if (ok_compile && (sf.load_mask & 8u)) {
                                u32 m3 = 0xF0000000u;
                                if (desc.mip_fallback_mode ==
                                    Shader::MipStorageFallbackMode::DynamicIndex) {
                                    m3 |= 0x000FF000u;
                                }
                                if (desc.is_written) {
                                    m3 |= 0x00000FFFu;
                                }
                                push_masked(sf.offsets[3], m3);
                            }
                        } else {
                            if ((sf.load_mask & 3u) == 3u && sf.offsets[0] < flat_words &&
                                sf.offsets[1] < flat_words && flat_ptr[sf.offsets[0]] == 0u &&
                                (flat_ptr[sf.offsets[1]] & 0x3Fu) == 0u) {
                                push_chk(sf.offsets[0], 0u, ~0u, 0u);
                                push_chk(sf.offsets[1], 0u, 0x3Fu, 0u);
                            } else {
                                ok_compile = false;
                            }
                        }
                    }
                    for (size_t sm = 0; ok_compile && sm < info.samplers.size(); ++sm) {
                        const auto& sf = info.samplers[sm].sharp_fetch;
                        if (sf.load_mask & 1u) {
                            push_masked(sf.offsets[0], 0x00108000u);
                        }
                    }
                    if (ok_compile && local_cnt > 0u && (max_off + 1u) <= 0xFFFFu) {
                        st.fast_check_offset = static_cast<u32>(f.fast_checks_pool.size());
                        st.fast_check_count = static_cast<u16>(local_cnt);
                        st.fast_check_min_flat = static_cast<u16>(max_off + 1u);
                        st.fast_check_start = start_binding;
                        f.fast_checks_pool.insert(f.fast_checks_pool.end(), local_chk.begin(),
                                                  local_chk.begin() + local_cnt);
                        if (want_vsharps && way_fetch[i] &&
                            way_fetch[i]->attributes.size() == spec.vs_attribs.size() &&
                            way_fetch[i]->attributes.size() <= st.fast_vs_dw3_exp.size() &&
                            f.last_vsharps.size() == way_fetch[i]->attributes.size()) {
                            u16 vmask = 0u;
                            for (size_t a = 0; a < f.last_vsharps.size(); ++a) {
                                const auto& sh = f.last_vsharps[a];
                                if (sh) {
                                    vmask |= static_cast<u16>(1u << a);
                                    u32 dw3 = 0u;
                                    std::memcpy(&dw3, reinterpret_cast<const u8*>(&sh) + 12u,
                                                sizeof(u32));
                                    st.fast_vs_dw3_exp[a] = dw3 & 0x007FFFFFu;
                                } else {
                                    st.fast_vs_dw3_exp[a] = 0u;
                                }
                            }
                            st.fast_vs_valid_mask = vmask;
                            st.fast_vs_attr_count = static_cast<u8>(f.last_vsharps.size());
                            st.fast_vs_fetch = way_fetch[i];
                        }
                    }
                }
            }
            if (can_memo_flat && flat_words >= 4u) {
                st.last_ud_head = {flat_ptr[0], flat_ptr[1], flat_ptr[2], flat_ptr[3]};
                st.last_flat_words = flat_words;
                st.last_start = start_binding;
                // Lazy hash: ONLY arm `last_flat_hash` + `vs_memo` when `head_same` already
                // matched (2+ consecutive draws with identical flat head). When every draw has
                // unique CB/VB addresses (~95% of draws), we skip both `XXH3_64bits` and copying
                // 520 B of `last_vsharps` completely!
                if (head_same) {
                    st.last_flat_hash = flat_hash;
                    if (want_vsharps && ud_words <= f.vs_memo.ud_full.size()) {
                        f.vs_memo.program = st.program;
                        f.vs_memo.perm_idx = st.perm_idx;
                        f.vs_memo.fetch = way_fetch[i];
                        f.vs_memo.ud_words = ud_words;
                        if (ud_words > 0u) {
                            std::memcpy(f.vs_memo.ud_full.data(), ud_span.data(),
                                        ud_words * sizeof(u32));
                        }
                        f.vs_memo.vsharps = f.last_vsharps;
                    }
                } else {
                    st.last_flat_hash = 0ULL;
                }
            }
            if (want_vsharps) {
                filled_vsharps = true;
            }
            info.AddBindings(binding);
        }
        if (spec_ok) {
            f.mru_way[set_idx] = static_cast<u8>(w);
            for (u32 i = 0; i < way.num_stages; ++i) {
                way.stages[i].program->last_used = way.stages[i].perm_idx;
            }
            way.last_use = ++f.use_clock;
            way.shader_epoch = cur_epoch;
            if (filled_vsharps) {
                f.last_vsharps_pipeline = way.pipeline;
            }
            ++f.hits;
            BbSections::fast_pso_hits.store(
                BbSections::fast_pso_hits.load(std::memory_order_relaxed) + 1,
                std::memory_order_relaxed);
            return way.pipeline;
        }
    }
    if (saw_motion_mismatch) {
        ++f.miss_motion;
    } else {
        ++f.miss_spec;
    }
    return nullptr;
}

void PipelineCache::PrintFastSelectStats() {
    auto& f = *fast;
    static const bool verbose = [] {
        const char* e = std::getenv("BB_VERBOSE_STATS");
        return e && e[0] == '1';
    }();
    if (!verbose || (++f.calls & 1023) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - f.window < std::chrono::seconds(5)) {
        return;
    }
    f.window = now;
    const u64 total = std::max<u64>(f.hits + f.full, 1);
    std::printf("Fast select (5 s): reused %llu (%.1f%%), full %llu; misses: registers %llu, "
                "program %llu, specialization %llu, motion %llu; not eligible %llu; verify "
                "mismatches %llu\n",
                (unsigned long long)f.hits, 100.0 * double(f.hits) / double(total),
                (unsigned long long)f.full, (unsigned long long)f.miss_regs,
                (unsigned long long)f.miss_program, (unsigned long long)f.miss_spec,
                (unsigned long long)f.miss_motion, (unsigned long long)f.not_eligible,
                (unsigned long long)f.verify_bad);
    f.hits = f.full = f.miss_regs = f.miss_program = f.miss_spec = f.miss_motion = 0;
    f.not_eligible = 0;
}

void PipelineCache::NoteDrawTarget(const Gcn::Regs& regs) {
    const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (frame != last_autobake_drain_frame &&
        autobake_has_ready.load(std::memory_order_relaxed)) [[unlikely]] {
        last_autobake_drain_frame = frame;
        DrainAutoBakeReadyStages();
    }
    const u64 quick_key =
        (u64(regs.color_buffers[0].base_address) << 32) ^
        (u64(regs.depth_buffer.z_read_base) << 8) ^
        u64(regs.color_target_mask.raw);
    if (frame == last_target_frame && quick_key == last_target_quick_key) {
        return;
    }
    last_target_quick_key = quick_key;
    const u64 target = DrawTarget(regs);
    if (target == 0 || (target == last_target && frame == last_target_frame)) {
        return;
    }
    last_target = target;
    last_target_frame = frame;
    auto& streak = target_streaks[target];
    if (streak.last_frame != frame) {
        streak.frames = streak.last_frame + 1 == frame ? streak.frames + 1 : 1;
        streak.last_frame = frame;
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params,
                                                           const PreparedDraw* prepared,
                                                           bool indirect) {
    EnsurePrewarmDone();
    BB_SECTION(APipeline);
    used_prepared = nullptr;
    NoteDrawTarget(liverpool->regs);
    if (prepared) {
        if (const auto* pipeline = TryPreparedPipeline(*prepared)) {
            used_prepared = prepared;
            return pipeline;
        }
    }
    const bool fast_select = FastSelectEnabled() && fast;
    const GraphicsPipeline* fast_pipeline = nullptr;
    if (fast_select) {
        PrintFastSelectStats();
        fast_pipeline = TryFastSelect(params);
        if (fast_pipeline && !FastSelectVerify()) {
            return fast_pipeline;
        }
    }
    const GraphicsPipeline* const selected = SelectGraphicsPipelineFull(params, indirect);
    if (fast_select) {
        if (fast_pipeline) {
            if (selected != fast_pipeline) {
                auto& f = *fast;
                if (f.verify_bad++ < 16) {
                    const auto* vs = selected ? selected->GetStages()[u32(SwStage::Vertex)]
                                              : nullptr;
                    std::printf("Fast select VERIFY mismatch: reused %p, full selection %p "
                                "(vs %016llx)\n",
                                static_cast<const void*>(fast_pipeline),
                                static_cast<const void*>(selected),
                                (unsigned long long)(vs ? vs->pgm_hash : 0));
                }
            }
        } else {
            ++fast->full;
            BbSections::fast_pso_full.store(
                BbSections::fast_pso_full.load(std::memory_order_relaxed) + 1,
                std::memory_order_relaxed);
            if (selected) {
                CommitFastSelect(selected, params);
            }
        }
    }
    return selected;
}

const GraphicsPipeline* PipelineCache::SelectGraphicsPipelineFull(const DrawIndirectParams params,
                                                                  bool indirect) {
    DrainCompletedAsyncSpvJobs();
    sel.draw_indirect_params = params;
    const bool skippable = AsyncSkippable(sel, indirect);
    if (skippable) {
        sel.async_probe = true;
        sel.async_spv_miss = false;
        const bool probe_ok = RefreshGraphicsKey(sel);
        sel.async_probe = false;
        if (!probe_ok) {
            if (sel.async_spv_miss && QueueAsyncSpvJob(params, indirect)) {
                return nullptr;
            }
            if (!sel.async_spv_miss || !RefreshGraphicsKey(sel)) {
                return nullptr;
            }
        }
    } else if (!RefreshGraphicsKey(sel)) {
        return nullptr;
    }
    if (const auto found = graphics_pipelines.find(sel.graphics_key);
        found != graphics_pipelines.end()) {
        return found->second.get();
    }
    if (const auto pending = pending_graphics.find(sel.graphics_key);
        pending != pending_graphics.end()) {
        const std::shared_ptr<PendingPipeline> job = pending->second;
        {
            std::unique_lock lk{job->mutex};
            if (!job->done) {
                if (skippable) {
                    ++async_skipped;
                    return nullptr; // drawn from the frame its pipeline is ready
                }
                // A pass that is not drawn every frame needs it now: the wait is compile time.
                const auto start = std::chrono::steady_clock::now();
                job->cv.wait(lk, [&] { return job->done; });
                const u64 wait_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count());
                AddCompileTime(start);
                ++async_waited;
                BbStats::hitch_pso_sync_ns.fetch_add(wait_ns, std::memory_order_relaxed);
                BbStats::hitch_pso_sync_count.fetch_add(1, std::memory_order_relaxed);
                if (wait_ns >= 1'500'000ull) {
                    const u64 vs_h = sel.infos[0] ? sel.infos[0]->pgm_hash : 0ull;
                    const u64 ps_h = sel.infos[1] ? sel.infos[1]->pgm_hash : 0ull;
                    BbStats::LogHitch(
                        "[BB-HITCH][PSO-WAIT] frame=%u %.2f ms key=%#llx vs=%#llx ps=%#llx draws=%llu\n",
                        BbStats::frame_number.load(std::memory_order_relaxed),
                        double(wait_ns) / 1e6,
                        static_cast<unsigned long long>(std::hash<GraphicsPipelineKey>{}(sel.graphics_key)),
                        static_cast<unsigned long long>(vs_h),
                        static_cast<unsigned long long>(ps_h),
                        static_cast<unsigned long long>(last_frame_draws));
                }
            }
        }
        pending_graphics.erase(sel.graphics_key);
        return FinishPipeline(sel.graphics_key, *job);
    }
    if (skippable) {
        return CompileInBackground(sel);
    }

    const auto sync_t0 = std::chrono::steady_clock::now();
    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(sel.graphics_key);
    LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
    CompileTimer timer;
    GraphicsPipeline::SerializationSupport sdata{};
    auto built = std::make_unique<GraphicsPipeline>(instance, scheduler, desc_heap, profile,
                                                    sel.graphics_key, *pipeline_cache, sel.infos,
                                                    sel.runtime_infos, sel.fetch_shader,
                                                    sel.modules, sdata, false);
    PipelineStore::SaveGraphics(sel.graphics_key, pipeline_hash, sdata);
    NoteDriverPipelineCacheDirty();
    const u64 sync_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - sync_t0)
                                .count());
    BbStats::hitch_pso_sync_ns.fetch_add(sync_ns, std::memory_order_relaxed);
    BbStats::hitch_pso_sync_count.fetch_add(1, std::memory_order_relaxed);
    if (sync_ns >= 1'500'000ull || last_frame_draws >= 250) {
        const u64 vs_h = sel.infos[0] ? sel.infos[0]->pgm_hash : 0ull;
        const u64 ps_h = sel.infos[1] ? sel.infos[1]->pgm_hash : 0ull;
        const u64 target = sel.regs ? DrawTarget(*sel.regs) : 0ull;
        const auto st_it = target_streaks.find(target);
        const u32 st_frames = (target != 0 && st_it != target_streaks.end()) ? st_it->second.frames : 0u;
        BbStats::LogHitch(
            "[BB-HITCH][PSO-SYNC] frame=%u %.2f ms key=%#llx vs=%#llx ps=%#llx depth=%u idx=%u "
            "ind=%u streak=%u target=%#llx draws=%llu\n",
            BbStats::frame_number.load(std::memory_order_relaxed), double(sync_ns) / 1e6,
            static_cast<unsigned long long>(pipeline_hash), static_cast<unsigned long long>(vs_h),
            static_cast<unsigned long long>(ps_h),
            sel.regs ? unsigned(bool(sel.regs->depth_control.depth_enable)) : 0u,
            sel.regs ? unsigned(sel.regs->num_indices) : 0u, unsigned(indirect), st_frames,
            static_cast<unsigned long long>(target),
            static_cast<unsigned long long>(last_frame_draws));
    }
    sel.fetch_shader.reset();
    const GraphicsPipeline* result = built.get();
    {
        std::unique_lock gr_lk{graphics_pipelines_mutex};
        graphics_pipelines.insert_or_assign(sel.graphics_key, std::move(built));
    }
    return result;
}

const GraphicsPipeline* PipelineCache::CompileInBackground(const PipelineSelection& selection) {
    if (!compiler) {
        const char* env = std::getenv("BB_ASYNC_PIPELINE_THREADS");
        const u32 threads =
            std::max(env && *env ? u32(std::strtoul(env, nullptr, 10))
                                 : std::clamp(std::thread::hardware_concurrency() / 3, 2u, 6u),
                     1u);
        compiler = std::make_unique<PipelineCompiler>(threads);
        std::printf("GPU: new pipelines of passes drawn every frame compile on %u threads "
                    "(BB_ASYNC_PIPELINES=0: on the GPU thread, the game waits)\n",
                    threads);
    }
    auto job = std::make_shared<PendingPipeline>();
    pending_graphics.emplace(selection.graphics_key, job);
    ++async_started;
    ++g_bb_compiles;
    BbStats::hitch_pso_async_count.fetch_add(1, std::memory_order_relaxed);
    BbStats::shader_async_submitted.fetch_add(1u, std::memory_order_relaxed);
    compiler->Submit([this, job, key = selection.graphics_key, infos = selection.infos,
                      runtime_infos = selection.runtime_infos, fetch = selection.fetch_shader,
                      modules = selection.modules]() mutable {
        GraphicsPipeline::SerializationSupport sdata{};
        auto pipeline = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, key, *pipeline_cache, infos, runtime_infos,
            fetch, modules, sdata, false);
        {
            std::scoped_lock lk{job->mutex};
            job->pipeline = std::move(pipeline);
            job->sdata = std::move(sdata);
            job->done = true;
            job->cv.notify_all();
        }
        BbStats::shader_async_completed.fetch_add(1u, std::memory_order_relaxed);
    });
    sel.fetch_shader.reset();
    return nullptr;
}

const GraphicsPipeline* PipelineCache::FinishPipeline(const GraphicsPipelineKey& key,
                                                      PendingPipeline& job) {
    PipelineStore::SaveGraphics(key, std::hash<GraphicsPipelineKey>{}(key), job.sdata);
    NoteDriverPipelineCacheDirty();
    const GraphicsPipeline* result = job.pipeline.get();
    {
        std::unique_lock gr_lk{graphics_pipelines_mutex};
        graphics_pipelines.insert_or_assign(key, std::move(job.pipeline));
    }
    return result;
}

const GraphicsPipeline* PipelineCache::SelectGraphicsPipelineMt(const Gcn::Regs& regs,
                                                                MtWorkerPsoState& worker_state,
                                                                DrawIndirectParams params,
                                                                bool indirect,
                                                                bool allow_compile_fallback,
                                                                MtPsoFailKind* out_fail_kind) {
    EnsurePrewarmDone();
    if (out_fail_kind) {
        *out_fail_kind = MtPsoFailKind::None;
    }
    std::array<u32, MtFastPsoTable::MaxSigWords> sig;
    const u32 sig_len =
        BuildFastSelectSig(regs, params, instance.IsVertexInputDynamicState(),
                           instance.IsTessellationSupported(), sig);
    u64 sig_key = 0;
    size_t set_base = 0;

    const auto lookup_worker_info =
        [&](const MtFastPsoTable::StageSnap& snap) -> Shader::Info& {
        const void* key = &snap;
        auto& mru = worker_state.stage_info_mru[(reinterpret_cast<uintptr_t>(key) >> 6) & 63];
        if (mru.first == key && mru.second != nullptr) [[likely]] {
            return *mru.second;
        }
        auto& inserted =
            worker_state.stage_info_cache.try_emplace(key, snap.info_template).first->second;
        mru = {key, &inserted};
        return inserted;
    };

    const auto lookup_worker_memo =
        [&](const MtFastPsoTable::StageSnap& snap) -> MtWorkerPsoState::StageMemo& {
        const void* key = &snap;
        auto& mru = worker_state.stage_memo_mru[(reinterpret_cast<uintptr_t>(key) >> 6) & 63];
        if (mru.first == key && mru.second != nullptr) [[likely]] {
            return *mru.second;
        }
        auto& inserted = worker_state.stage_memo_cache[key];
        mru = {key, &inserted};
        return inserted;
    };

    const auto verify_code_hash_mt = [](VAddr base, u64 expected_hash) noexcept -> bool {
        if (base == 0) {
            return false;
        }
        DirectShaders::Snapshot direct_snap;
        if (DirectShaders::TryLookup(base, direct_snap, /*is_worker=*/true) &&
            direct_snap.cpu_immutable != 0) {
            return direct_snap.shader_hash == expected_hash;
        }
        const auto* bin = Gcn::DirectBinaryInfo(reinterpret_cast<const u32*>(base));
        return bin && bin->shader_hash == expected_hash;
    };

    const auto lookup_fetch_mt =
        [&](const Shader::Info& info, bool epoch_valid) -> const Shader::Gcn::FetchShaderData* {
        const u32* const code =
            Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
        if (!code) {
            return nullptr;
        }
        auto& mru = worker_state.fetch_mru[(reinterpret_cast<uintptr_t>(code) >> 4) & 63];
        if (!BbToggle::Disabled(BbToggle::FetchShaderCache) && mru.first == code && mru.second &&
            (epoch_valid ||
             std::memcmp(code, mru.second->words.data(),
                         mru.second->words.size() * sizeof(u32)) == 0)) [[likely]] {
            return &mru.second->data;
        }
        if (const auto it = worker_state.fetch_map.find(code);
            !BbToggle::Disabled(BbToggle::FetchShaderCache) &&
            it != worker_state.fetch_map.end() &&
            (epoch_valid ||
             std::memcmp(code, it->second.words.data(),
                         it->second.words.size() * sizeof(u32)) == 0)) {
            mru = {code, &it->second};
            return &it->second.data;
        }
        if (!FetchCodeDecodes(code)) {
            return nullptr;
        }
        auto parsed = Shader::Gcn::ParseFetchShader(info);
        if (!parsed || parsed->size == 0 || (parsed->size & 3u) != 0) {
            return nullptr;
        }
        if (worker_state.fetch_map.size() >= 1024) {
            worker_state.fetch_map.clear();
            worker_state.fetch_mru.fill({});
        }
        auto& slot = worker_state.fetch_map[code];
        slot.words.assign(code, code + (parsed->size / sizeof(u32)));
        slot.data = std::move(*parsed);
        mru = {code, &slot};
        return &slot.data;
    };

    if (sig_len != 0 && mt_fast && FastSelectEnabled()) {
        sig_key = XXH3_64bits(sig.data(), sig_len * sizeof(u32)) | 1ULL;
        set_base = ((sig_key >> 1) & (MtFastPsoTable::NumSets - 1)) * MtFastPsoTable::NumWays;

        std::array<const MtFastPsoTable::Entry*, MtFastPsoTable::NumWays> ways{};
        const MtFastPsoTable::Entry* first_match = nullptr;
        u32 matching_ways_mask = 0;
        for (u32 w = 0; w < MtFastPsoTable::NumWays; ++w) {
            const auto* cand = mt_fast->slots[set_base + w].load(std::memory_order_acquire);
            ways[w] = cand;
            if (cand && cand->key == sig_key && cand->sig_words == sig_len &&
                std::memcmp(cand->sig.data(), sig.data(), sig_len * sizeof(u32)) == 0) {
                if (!first_match) {
                    first_match = cand;
                }
                matching_ways_mask |= (1u << w);
            }
        }

        if (first_match) {
            const u64 cur_epoch = DirectShaders::InvalidationEpoch();
            for (u32 w = 0; w < MtFastPsoTable::NumWays; ++w) {
                if (((matching_ways_mask >> w) & 1u) == 0) {
                    continue;
                }
                const auto& way = *ways[w];
                const bool epoch_valid =
                    (way.shader_epoch.load(std::memory_order_relaxed) == cur_epoch);

                std::array<Shader::Info*, MtFastPsoTable::MaxStages> way_infos{};
                std::array<const Shader::Gcn::FetchShaderData*, MtFastPsoTable::MaxStages>
                    way_fetch{};
                const Shader::Info* vs_info = nullptr;
                bool programs_ok = true;

                for (u32 i = 0; i < way.num_stages; ++i) {
                    const auto& st = way.stages[i];
                    const auto* pgm = regs.ProgramForStage(static_cast<u32>(st.hw_stage));
                    if (!pgm) {
                        programs_ok = false;
                        break;
                    }
                    const VAddr base = pgm->Address<VAddr>();
                    if (base != st.pgm_base || base == 0 ||
                        (!epoch_valid && !verify_code_hash_mt(base, st.shader_hash))) {
                        programs_ok = false;
                        break;
                    }
                    if (st.hw_stage == HwStage::Geometry) {
                        const VAddr copy_base = regs.vs_program.Address<VAddr>();
                        if (copy_base != st.copy_pgm_base || copy_base == 0 ||
                            (!epoch_valid &&
                             !verify_code_hash_mt(copy_base, st.copy_shader_hash))) {
                            programs_ok = false;
                            break;
                        }
                    }
                    auto& winfo = lookup_worker_info(st);
                    winfo.pgm_base = base;
                    winfo.user_data = pgm->user_data;
                    winfo.RefreshFlatBuf();
                    if (winfo.has_fetch_shader) {
                        way_fetch[i] = lookup_fetch_mt(winfo, epoch_valid);
                        if (!way_fetch[i]) {
                            programs_ok = false;
                            break;
                        }
                    }
                    way_infos[i] = &winfo;
                    if (st.sw_stage == SwStage::Vertex) {
                        vs_info = &winfo;
                    }
                }
                if (!programs_ok) {
                    continue;
                }

                if (way.motion_possible) {
                    const bool cur_motion = AllMotionEnv() || (vs_info && VsSelectsMotion(*vs_info));
                    if (way.motion != cur_motion) {
                        continue;
                    }
                }

                Shader::Backend::Bindings binding{};
                bool spec_ok = true;
                bool filled_vsharps = false;
                const Shader::Info* tcs_info = nullptr;
                for (u32 i = 0; i < way.num_stages; ++i) {
                    const auto& st = way.stages[i];
                    auto& info = *way_infos[i];
                    if (st.sw_stage == SwStage::TessellationControl) {
                        tcs_info = &info;
                    }
                    if (st.hw_stage == HwStage::Local && tcs_info) {
                        Shader::TessellationDataConstantBuffer tess_constants{};
                        tcs_info->ReadTessConstantBuffer(tess_constants);
                        if (st.runtime_info.hw.ls.ls_stride != tess_constants.ls_stride) {
                            spec_ok = false;
                            break;
                        }
                    }
                    const bool want_vsharps =
                        (st.sw_stage == SwStage::Vertex && info.has_fetch_shader &&
                         way_fetch[i] != nullptr);
                    const bool can_memo_flat =
                        !want_vsharps && !info.has_fetch_shader && st.hw_stage != HwStage::Local &&
                        st.sw_stage != SwStage::TessellationControl;
                    u64 flat_hash = 0;
                    u32 flat_words = 0;
                    bool head_same = false;
                    const auto start_binding = binding;
                    MtWorkerPsoState::StageMemo* memo = nullptr;
                    if (can_memo_flat) {
                        memo = &lookup_worker_memo(st);
                        flat_words = static_cast<u32>(info.flattened_ud_buf.size());
                        const u32* flat_ptr = info.flattened_ud_buf.data();
                        if (flat_words >= 4u && flat_words == memo->last_flat_words &&
                            start_binding == memo->last_start &&
                            flat_ptr[0] == memo->last_ud_head[0] &&
                            flat_ptr[1] == memo->last_ud_head[1] &&
                            flat_ptr[2] == memo->last_ud_head[2] &&
                            flat_ptr[3] == memo->last_ud_head[3]) {
                            head_same = true;
                            flat_hash = XXH3_64bits(flat_ptr, flat_words * sizeof(u32)) | 1ULL;
                            if (flat_hash == memo->last_flat_hash) {
                                info.AddBindings(binding);
                                continue;
                            }
                        }
                    }
                    if (!st.spec.MatchesInPlace(info, st.runtime_info, profile, binding,
                                                way_fetch[i], st.rt_verified,
                                                want_vsharps ? &worker_state.vsharps : nullptr)) {
                        spec_ok = false;
                        break;
                    }
                    if (memo && flat_words >= 4u) {
                        const u32* flat_ptr = info.flattened_ud_buf.data();
                        memo->last_ud_head = {flat_ptr[0], flat_ptr[1], flat_ptr[2], flat_ptr[3]};
                        memo->last_flat_words = flat_words;
                        memo->last_start = start_binding;
                        memo->last_flat_hash =
                            head_same ? flat_hash
                                      : (XXH3_64bits(flat_ptr, flat_words * sizeof(u32)) | 1ULL);
                    }
                    if (want_vsharps) {
                        filled_vsharps = true;
                    }
                    info.AddBindings(binding);
                }

                if (spec_ok) {
                    if (!epoch_valid) {
                        way.shader_epoch.store(cur_epoch, std::memory_order_relaxed);
                    }
                    worker_state.bound_stage_infos.fill(nullptr);
                    for (u32 i = 0; i < way.num_stages; ++i) {
                        worker_state.bound_stage_infos[static_cast<u32>(way.stages[i].sw_stage)] =
                            way_infos[i];
                    }
                    worker_state.has_vsharps = filled_vsharps;
                    return way.pipeline;
                }
            }
        }
    }

    // Cold MtFastPsoTable miss: resolve via thread-safe `PrepareGraphicsPipeline` (`programs_mutex`
    // shared lock) and look up in `graphics_pipelines` under `graphics_pipelines_mutex`.
    (void)indirect;
    worker_state.prep_worker.stages.clear();
    worker_state.prep_worker.failed = false;
    worker_state.sel.regs = &regs;
    worker_state.sel.worker = &worker_state.prep_worker;
    worker_state.sel.draw_indirect_params = params;
    if (!PrepareGraphicsPipeline(worker_state.sel)) {
        if (out_fail_kind) {
            *out_fail_kind = MtPsoFailKind::SpecMiss;
        }
        return nullptr;
    }

    const GraphicsPipeline* resolved_pso = nullptr;
    {
        std::shared_lock gr_lk{graphics_pipelines_mutex};
        if (const auto it = graphics_pipelines.find(worker_state.sel.graphics_key);
            it != graphics_pipelines.end()) {
            resolved_pso = it->second.get();
        }
    }
    if (!resolved_pso && allow_compile_fallback) {
        std::scoped_lock c_lk{compile_fallback_mutex};
        {
            std::shared_lock gr_lk{graphics_pipelines_mutex};
            if (const auto it = graphics_pipelines.find(worker_state.sel.graphics_key);
                it != graphics_pipelines.end()) {
                resolved_pso = it->second.get();
            }
        }
        if (!resolved_pso) {
            const auto pipeline_hash =
                std::hash<GraphicsPipelineKey>{}(worker_state.sel.graphics_key);
            GraphicsPipeline::SerializationSupport sdata{};
            auto built = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, worker_state.sel.graphics_key,
                *pipeline_cache, worker_state.sel.infos, worker_state.sel.runtime_infos,
                worker_state.sel.fetch_shader, worker_state.sel.modules, sdata, false);
            PipelineStore::SaveGraphics(worker_state.sel.graphics_key, pipeline_hash, sdata);
            NoteDriverPipelineCacheDirty();
            resolved_pso = built.get();
            std::unique_lock gr_lk{graphics_pipelines_mutex};
            graphics_pipelines.insert_or_assign(worker_state.sel.graphics_key, std::move(built));
        }
    }
    if (!resolved_pso) {
        if (out_fail_kind) {
            *out_fail_kind = MtPsoFailKind::SigMiss;
        }
        return nullptr;
    }

    worker_state.bound_stage_infos = worker_state.sel.infos;
    worker_state.vsharps.clear();
    worker_state.has_vsharps = false;
    if (const Shader::Info* vs_info =
            worker_state.bound_stage_infos[static_cast<u32>(SwStage::Vertex)];
        vs_info && vs_info->has_fetch_shader && worker_state.sel.fetch_shader.has_value()) {
        for (const auto& desc : worker_state.sel.fetch_shader->attributes) {
            if (worker_state.vsharps.size() < worker_state.vsharps.capacity()) {
                worker_state.vsharps.push_back(desc.GetSharpDirect(vs_info->user_data));
            }
        }
        worker_state.has_vsharps = true;
    }

    // Publish warmed PSO into `MtFastPsoTable` so subsequent draws hit lock-free O(1).
    if (sig_len != 0 && mt_fast && FastSelectEnabled() &&
        !worker_state.prep_worker.stages.empty() &&
        worker_state.prep_worker.stages.size() <= MtFastPsoTable::MaxStages) {
        if (sig_key == 0) {
            sig_key = XXH3_64bits(sig.data(), sig_len * sizeof(u32)) | 1ULL;
            set_base = ((sig_key >> 1) & (MtFastPsoTable::NumSets - 1)) * MtFastPsoTable::NumWays;
        }
        auto entry = std::make_unique<MtFastPsoTable::Entry>();
        entry->key = sig_key;
        entry->shader_epoch.store(DirectShaders::InvalidationEpoch(), std::memory_order_relaxed);
        entry->pipeline = resolved_pso;
        entry->sig_words = static_cast<u16>(sig_len);
        entry->num_stages = static_cast<u8>(worker_state.prep_worker.stages.size());
        entry->motion = worker_state.sel.motion;
        entry->motion_possible =
            BbSettings::Get().object_motion.load(std::memory_order_relaxed) &&
            Shader::MotionVectors::positions_address != 0 &&
            regs.color_buffers[0] && regs.color_buffers[1] && regs.color_buffers[2] &&
            regs.depth_buffer.DepthValid();
        std::memcpy(entry->sig.data(), sig.data(), sig_len * sizeof(u32));

        bool snap_ok = true;
        {
            std::shared_lock p_lk{programs_mutex};
            for (u32 i = 0; i < entry->num_stages; ++i) {
                const auto& src = worker_state.prep_worker.stages[i];
                if (!src.program || !src.program->info_template ||
                    src.perm_idx >= src.program->modules.size()) {
                    snap_ok = false;
                    break;
                }
                auto& dst = entry->stages[i];
                dst.info_template = *src.program->info_template;
                dst.spec = src.program->modules[src.perm_idx].spec;
                dst.spec.info = &dst.info_template;
                dst.runtime_info =
                    worker_state.sel.runtime_infos[static_cast<u32>(src.sw_stage)];
                dst.hw_stage = src.hw_stage;
                dst.sw_stage = src.sw_stage;
                dst.pgm_base = src.pgm_base;
                dst.shader_hash = src.hash;
                if (src.hw_stage == HwStage::Geometry) {
                    const VAddr copy_base = regs.vs_program.Address<VAddr>();
                    dst.copy_pgm_base = copy_base;
                    if (copy_base != 0) {
                        DirectShaders::Snapshot copy_snap;
                        if (DirectShaders::TryLookup(copy_base, copy_snap, /*is_worker=*/true)) {
                            dst.copy_shader_hash = copy_snap.shader_hash;
                        } else if (const auto* bin = Gcn::DirectBinaryInfo(
                                       reinterpret_cast<const u32*>(copy_base))) {
                            dst.copy_shader_hash = bin->shader_hash;
                        }
                    }
                }
                dst.rt_verified = !dst.info_template.has_fetch_shader;
            }
        }

        if (snap_ok) {
            std::scoped_lock lk{mt_fast->publish_mutex};
            u32 victim_way = 0;
            u64 oldest_seq = ~0ULL;
            bool duplicate = false;
            for (u32 w = 0; w < MtFastPsoTable::NumWays; ++w) {
                const auto* cur = mt_fast->slots[set_base + w].load(std::memory_order_relaxed);
                if (!cur) {
                    victim_way = w;
                    oldest_seq = 0;
                    break;
                }
                if (cur->key == sig_key && cur->pipeline == resolved_pso &&
                    cur->sig_words == sig_len &&
                    std::memcmp(cur->sig.data(), sig.data(), sig_len * sizeof(u32)) == 0) {
                    duplicate = true;
                    break;
                }
                if (cur->publish_seq < oldest_seq) {
                    oldest_seq = cur->publish_seq;
                    victim_way = w;
                }
            }
            if (!duplicate) {
                entry->publish_seq = ++mt_fast->publish_clock;
                const MtFastPsoTable::Entry* raw = entry.get();
                mt_fast->arena.push_back(std::move(entry));
                mt_fast->slots[set_base + victim_way].store(raw, std::memory_order_release);
            }
        }
    }
    return resolved_pso;
}

bool PipelineCache::AsyncSkippable(const PipelineSelection& selection, bool indirect) {
    // Only a pass drawn every frame may go without a draw for a frame or two (its render target
    // drawn to in each of the last frames, below). A one-time render compiles at once, as during
    // loading screens (a frame of fewer than 300 draws before this one). The pipeline must not
    // depend on the draw's vertex buffers (VK_EXT_vertex_input_dynamic_state). A size rule (the
    // scene's, the output's) missed most passes: many set a 16384 scissor.
    static const bool enabled = [] {
        const char* env = std::getenv("BB_ASYNC_PIPELINES");
        return !(env && env[0] == '0');
    }();
    if (!enabled || !instance.IsVertexInputDynamicState() || !selection.regs) {
        return false;
    }
    const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    last_frame_draws = g_last_completed_frame_draws.load(std::memory_order_relaxed);
    if (frame != stats_frame) {
        stats_frame = frame;
        const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
        if (async_started != 0 && second - stats_printed >= 5) {
            stats_printed = second;
            std::printf("GPU: %llu pipelines compiled in the background, %llu draws went without "
                        "theirs meanwhile, %llu waited for one\n",
                        (unsigned long long)async_started, (unsigned long long)async_skipped,
                        (unsigned long long)async_waited);
            async_started = async_skipped = async_waited = 0;
        }
    }
    if (last_frame_draws < 64) {
        return false;
    }
    // A full-screen pass (one triangle or quad without depth testing: lighting, fog, post-processing)
    // waits for its pipeline: going without one for a frame left whole frames grey. So does a
    // non-depth indirect draw (its size unknown here). 3D depth-tested geometry (including
    // first-frame world spawn, depth-only shadow/prepass DrawIndirect, 6-index blood decals and hit
    // sparks) may compile asynchronously without stalling the frame.
    constexpr u32 FullScreenIndices = 12;
    const bool has_depth =
        bool(selection.regs->depth_control.depth_enable) &&
        (selection.regs->depth_buffer.DepthValid() || selection.regs->depth_buffer.StencilValid());
    if (!has_depth && (indirect || (!selection.regs->depth_control.depth_enable &&
                                    selection.regs->num_indices <= FullScreenIndices &&
                                    !selection.regs->blend_control[0].enable))) {
        return false;
    }
    if (has_depth) {
        return true;
    }
    // A pass drawn every frame: its render target drawn to in each of the last frames. A one-time
    // or short-burst render without depth (a face baked for a character, or a 1..4-frame Blood Paint
    // RTT splatter burst onto character UV masks) compiles at once (< 1 ms with runtime_fast_opt)
    // so baked atlases and Blood Paint splatters never miss a single draw; every-frame blended
    // targets (such as HUD elements at streak >= 8) compile asynchronously.
    const u32 min_streak = 8u;
    const u64 target = DrawTarget(*selection.regs);
    const auto streak = target_streaks.find(target);
    return target != 0 && streak != target_streaks.end() && streak->second.frames >= min_streak;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    EnsurePrewarmDone();
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto sync_t0 = std::chrono::steady_clock::now();
        const u64 cs_key_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", cs_key_hash);
        CompileTimer timer;
        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *sel.infos[0],
                                                       sel.modules[0], sdata, false);
        PipelineStore::SaveCompute(compute_key, sdata);
        NoteDriverPipelineCacheDirty();
        const u64 sync_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - sync_t0)
                                    .count());
        BbStats::hitch_pso_sync_ns.fetch_add(sync_ns, std::memory_order_relaxed);
        BbStats::hitch_pso_sync_count.fetch_add(1, std::memory_order_relaxed);
        if (sync_ns >= 1'000'000ull || last_frame_draws >= 250) {
            BbStats::LogHitch("[BB-HITCH][CS-SYNC] frame=%u %.2f ms key=%#llx cs=%#llx draws=%llu\n",
                        BbStats::frame_number.load(std::memory_order_relaxed),
                        double(sync_ns) / 1e6, static_cast<unsigned long long>(cs_key_hash),
                        static_cast<unsigned long long>(sel.infos[0] ? sel.infos[0]->pgm_hash : 0ull),
                        static_cast<unsigned long long>(last_frame_draws));
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey(PipelineSelection& sel) {
    std::memset(&sel.graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = *sel.regs;
    auto& key = sel.graphics_key;
    const auto& db = regs.depth_buffer;
    const bool db_enabled = db.DepthValid() || db.StencilValid();

    key.z_format = db.DepthValid() ? db.z_info.format : Gcn::DepthBuffer::ZFormat::Invalid;
    key.stencil_format =
        db.StencilValid() ? db.stencil_info.format : Gcn::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? db.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool color_disabled =
        regs.color_control.mode == Gcn::ColorControl::OperationMode::Disable;
    const auto bound = [&](u32 cb) {
        return !color_disabled && regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb);
    };

    // The color targets the fragment stage is translated for.
    u32 num_bound = 0;
    for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        if (!bound(cb)) {
            continue;
        }
        ++num_bound;
        const auto& col_buf = regs.color_buffers[cb];
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == Gcn::BlendControl::BlendFunc::Min ||
             bc.color_func == Gcn::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == Gcn::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == Gcn::BlendControl::BlendFactor::DstColor;
    }

    // Only potentially animated G-buffer draws may use the extra motion attachment. The
    // ordinary stage variant is needed first to inspect the vertex shader's resources.
    const bool motion_possible =
        Shader::MotionVectors::positions_address != 0 &&
        BbSettings::Get().object_motion &&
        BbSettings::EffectiveUpscaler() != BbSettings::UpscalerOff &&
        num_bound >= 5 &&
        !regs.color_buffers[Shader::MotionVectors::Output] && db.DepthValid() &&
        db.NumSamples() == 1 && !regs.IsClipDisabled() &&
        regs.stage_enable.raw == Gcn::ShaderStageEnable::VgtStages::Vs;
    sel.motion = false;
    if (!sel.worker && &sel == &this->sel && fast) {
        fast->pending_motion_possible = motion_possible;
    }

    if (!RefreshGraphicsStages(sel)) {
        return false;
    }
    if (motion_possible) {
        const auto* vs = sel.infos[static_cast<u32>(SwStage::Vertex)];
        if (vs) {
            // Shaders with a bone palette (motion_history.h). Small skeletons (weapons, props)
            // also include static world pieces: the rasterizer gives those history only while
            // their constants change.
            sel.motion = VsSelectsMotion(*vs);
        }
        // BB_MOTION_SELECT_LOG=1: each G-buffer vertex shader once, with its buffer sizes and
        // the selection, to find animated models that the size rule leaves out.
        static const bool select_log = [] {
            const char* value = std::getenv("BB_MOTION_SELECT_LOG");
            return value && value[0] == '1';
        }();
        if (select_log && vs) {
            std::string sizes;
            for (const auto& resource : vs->buffers) {
                if (resource.IsSpecial()) {
                    continue;
                }
                const auto buffer = resource.GetSharp(*vs);
                sizes += fmt::format(" {}/{}", buffer.GetSize(), buffer.GetStride());
            }
            static std::mutex log_mutex;
            static std::unordered_set<u64> logged;
            std::scoped_lock lock{log_mutex};
            if (logged.insert(vs->pgm_hash ^ std::hash<std::string>{}(sizes)).second) {
                std::printf("Motion select: vs %016llx %s, buffers (size/stride):%s\n",
                            static_cast<unsigned long long>(vs->pgm_hash),
                            sel.motion ? "ON " : "off", sizes.c_str());
            }
        }
        // BB_OBJECT_MOTION_ALL=1: the old broad selection, for visual A/B tests.
        if (AllMotionEnv()) {
            sel.motion = true;
        }
        if (sel.motion && !RefreshGraphicsStages(sel)) {
            return false;
        }
    }

    // Targets the fragment stage does not write are left out; the others get their blending,
    // write mask and sample count.
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (u32 cb = 0; cb < key.num_color_attachments; ++cb) {
        if (!bound(cb)) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }
        const auto& col_buf = regs.color_buffers[cb];
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }
        key.write_masks[cb] = vk::ColorComponentFlags{
            key.color_buffers[cb].swizzle.ApplyMask(regs.color_target_mask.GetMask(cb))};
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    if (sel.motion) {
        constexpr u32 mv = Shader::MotionVectors::Output;
        key.motion_vectors = 1;
        key.mrt_mask |= 1u << mv;
        key.num_color_attachments = mv + 1;
        auto& color_buffer = key.color_buffers[mv];
        color_buffer.data_format = Gcn::DataFormat::Format32_32_32_32;
        color_buffer.num_format = Gcn::NumberFormat::Float;
        color_buffer.swizzle = Gcn::IdentityMapping;
        key.write_masks[mv] = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        key.color_samples[mv] = 1;
    }

    // Color sample counts the host cannot mix with each other or with the depth buffer's all
    // take the depth buffer's.
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if ((!all_color_samples_same && !instance.IsMixedAnySamplesSupported()) ||
            (all_color_samples_same && depth_mismatch &&
             !instance.IsMixedDepthSamplesSupported())) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }
    return true;
}

bool PipelineCache::RefreshGraphicsStages(PipelineSelection& sel) {
    bool not_a_program = false;
    const auto& regs = *sel.regs;
    auto& key = sel.graphics_key;
    if (!sel.worker && &sel == &this->sel && fast) {
        fast->num_pending = 0;
        fast->pending_ok = true;
    }
    sel.fetch_shader = std::nullopt;
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto in = static_cast<u32>(stage_in);
        const auto out = static_cast<u32>(stage_out);
        const auto* pgm = regs.stage_enable.IsStageEnabled(in) ? regs.ProgramForStage(in) : nullptr;
        // A geometry stage also runs the copy shader (the VS program).
        const auto* copy = stage_in == HwStage::Geometry && pgm ? &regs.vs_program : nullptr;
        const VAddr pgm_base = pgm ? pgm->Address<VAddr>() : 0;
        if (pgm_base == 0) {
            key.stage_hashes[out] = 0;
            sel.infos[out] = nullptr;
            return false;
        }
        Shader::ShaderParams params{.user_data = pgm->user_data};
        Program* direct_pgm = nullptr;
        DirectShaders::Snapshot direct_snap;
        if (sel.async_ctx) {
            if (in >= sel.async_ctx->stages.size() || !sel.async_ctx->stages[in].valid) {
                not_a_program = true;
                key.stage_hashes[out] = 0;
                sel.infos[out] = nullptr;
                return false;
            }
            const auto& snap = sel.async_ctx->stages[in];
            params.user_data = snap.user_data;
            params.code = snap.code;
            params.hash = snap.hash;
        } else if (DirectShaders::TryLookup(pgm_base, direct_snap, sel.worker != nullptr) &&
                   direct_snap.cpu_immutable != 0) {
            if (copy) {
                const VAddr copy_base = copy->Address<VAddr>();
                DirectShaders::Snapshot copy_snap;
                const bool copy_ok =
                    copy_base != 0 &&
                    ((DirectShaders::TryLookup(copy_base, copy_snap, sel.worker != nullptr) &&
                      copy_snap.cpu_immutable != 0) ||
                     LooksLikeProgram(reinterpret_cast<const u32*>(copy_base), HwStage::Vertex,
                                      !sel.worker));
                if (!copy_ok) {
                    not_a_program = true;
                    key.stage_hashes[out] = 0;
                    sel.infos[out] = nullptr;
                    return false;
                }
            }
            params.code = std::span<const u32>{reinterpret_cast<const u32*>(pgm_base),
                                               direct_snap.code_dwords};
            params.hash = direct_snap.shader_hash;
            direct_pgm = direct_snap.cached_program;
        } else {
            if (!LooksLikeProgram(reinterpret_cast<const u32*>(pgm_base), stage_in, !sel.worker) ||
                (copy && (!copy->Address<u32*>() ||
                          !LooksLikeProgram(copy->Address<u32*>(), HwStage::Vertex, !sel.worker)))) {
                not_a_program = true;
                key.stage_hashes[out] = 0;
                sel.infos[out] = nullptr;
                return false;
            }
            params = Gcn::GetParams(*pgm);
            if (!sel.worker && !program_cache.contains(params.hash) &&
                !ProgramDecodes(params, stage_in)) {
                not_a_program = true;
                key.stage_hashes[out] = 0;
                sel.infos[out] = nullptr;
                return false;
            }
        }
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader;
        std::tie(sel.infos[out], sel.modules[out], fetch_shader, key.stage_hashes[out]) =
            GetProgram(sel, stage_in, stage_out, params, binding, direct_pgm);
        if (!sel.infos[out]) {
            not_a_program |= !sel.worker; // its fetch shader does not decode (GetProgram)
            key.stage_hashes[out] = 0;
            return false;
        }
        if (fetch_shader) {
            sel.fetch_shader = std::move(fetch_shader);
        }
        return true;
    };

    if (!bind_stage(HwStage::Fragment, SwStage::Fragment)) {
        // No pixel shader: the clip distance discard shader may still be needed, decided from
        // this draw's registers (not a previous draw's fragment stage).
        auto& fs = sel.runtime_infos[u32(SwStage::Fragment)];
        fs.Initialize(HwStage::Fragment, SwStage::Fragment);
        fs.hw.fs.clip_distance_emulation = NeedsClipDistanceEmulation(regs, profile);
    }
    const auto* fs_info = sel.infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    // The geometry stages of each hardware configuration, in selection order (the hull stage
    // before the local one: the LS output stride is in the hull stage's constants).
    // Static arrays: an initializer_list assigned here would point at a temporary array that ends
    // with its statement (clang reuses that stack: stages read as garbage).
    using Stage = std::pair<HwStage, SwStage>;
    static constexpr Stage EsGsStages[] = {{HwStage::Export, SwStage::Vertex},
                                           {HwStage::Geometry, SwStage::Geometry}};
    static constexpr Stage LsHsStages[] = {{HwStage::Hull, SwStage::TessellationControl},
                                           {HwStage::Vertex, SwStage::TessellationEval},
                                           {HwStage::Local, SwStage::Vertex}};
    static constexpr Stage LsHsEsGsStages[] = {{HwStage::Hull, SwStage::TessellationControl},
                                               {HwStage::Export, SwStage::TessellationEval},
                                               {HwStage::Local, SwStage::Vertex},
                                               {HwStage::Geometry, SwStage::Geometry}};
    using Vgt = Gcn::ShaderStageEnable::VgtStages;
    const auto mode = regs.stage_enable.raw;
    bool tessellation = false, geometry = false;
    std::span<const Stage> stages;
    switch (mode) {
    case Vgt::Vs:
        // A missing vertex program is drawn without one.
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    case Vgt::EsGs:
        geometry = true;
        stages = EsGsStages;
        break;
    case Vgt::LsHs:
        tessellation = true;
        stages = LsHsStages;
        break;
    case Vgt::LsHsEsGs:
        tessellation = geometry = true;
        stages = LsHsEsGsStages;
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", u32(mode));
        return false;
    }
    if (tessellation && !instance.IsTessellationSupported()) {
        return false;
    }
    if (geometry) {
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
    }
    for (const auto& [hw, sw] : stages) {
        if (!bind_stage(hw, sw)) {
            return false;
        }
    }
    if (not_a_program) {
        return false;
    }

    // Without dynamic vertex input the pipeline is specialized on the attribute formats
    // (strides stay dynamic state).
    const auto* vs_info = sel.infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && sel.fetch_shader && !instance.IsVertexInputDynamicState()) {
        u32 vertex_binding = 0;
        for (const auto& attrib : sel.fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                GcnToVk::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }
    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs = liverpool->GetCsRegs();
    const VAddr cs_base = cs.Address<VAddr>();
    if (cs_base == 0) {
        return false;
    }
    Shader::ShaderParams cs_params{.user_data = cs.user_data};
    Program* direct_pgm = nullptr;
    DirectShaders::Snapshot direct_snap;
    if (DirectShaders::TryLookup(cs_base, direct_snap, /*is_worker=*/false) &&
        direct_snap.cpu_immutable != 0) {
        cs_params.code =
            std::span<const u32>{reinterpret_cast<const u32*>(cs_base), direct_snap.code_dwords};
        cs_params.hash = direct_snap.shader_hash;
        direct_pgm = direct_snap.cached_program;
    } else {
        if (!LooksLikeProgram(reinterpret_cast<const u32*>(cs_base), HwStage::Compute, true)) {
            return false;
        }
        cs_params = Gcn::GetParams(cs);
        if (!program_cache.contains(cs_params.hash) &&
            !ProgramDecodes(cs_params, HwStage::Compute)) {
            return false;
        }
    }
    std::tie(sel.infos[0], sel.modules[0], sel.fetch_shader, compute_key.value) =
        GetProgram(sel, HwStage::Compute, SwStage::Compute, cs_params, binding, direct_pgm);
    return true;
}

vk::ShaderModule PipelineCache::CompileModuleRaw(Shader::Pools& use_pools, Shader::Info& info,
                                                 Shader::RuntimeInfo& runtime_info,
                                                 std::span<const u32> code, size_t perm_idx,
                                                 Shader::Backend::Bindings& binding,
                                                 std::vector<u32>* out_raw_spv,
                                                 std::vector<u32>* out_opt_spv,
                                                 bool runtime_fast_opt) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    // BB_SHADER_CAPTURE: only on the main GPU thread (`&use_pools == &pools`).
    const bool capture = (&use_pools == &pools) && BbShaderCapture::Enabled();
    std::optional<Shader::Info> in_info;
    const auto in_runtime_info = runtime_info;
    const auto in_binding = binding;
    if (capture) {
        in_info.emplace(info);
        BbShaderCapture::Begin();
    }
    const auto ir_program = Shader::TranslateProgram(code, use_pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    if (capture) {
        BbShaderCapture::Write(*in_info, in_runtime_info, profile, in_binding, code, perm_idx, spv,
                               info, binding);
    }
    if (Shader::Backend::SPIRV::SpirvOptEnabled()) {
        if (out_raw_spv) {
            *out_raw_spv = spv;
        } else {
            PipelineStore::SaveShaderBinary(std::vector<u32>{spv}, info.pgm_hash, perm_idx);
        }
        bool deferred_heavy = false;
        Shader::Backend::SPIRV::OptimizeSPIRV(spv, info.pgm_hash, runtime_fast_opt,
                                              &deferred_heavy);
        DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");
        const vk::ShaderModule module = CompileSPV(spv, instance.GetDevice());
        if (out_opt_spv) {
            if (!deferred_heavy) {
                *out_opt_spv = std::move(spv);
            } else {
                out_opt_spv->clear();
            }
        } else if (!deferred_heavy) {
            PipelineStore::SaveOptShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
        }
        SetObjectName(instance.GetDevice(), module,
                      GetShaderName(info.hw_stage, info.pgm_hash, perm_idx));
        return module;
    }
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    const vk::ShaderModule module = CompileSPV(spv, instance.GetDevice());
    if (out_raw_spv) {
        *out_raw_spv = std::move(spv);
    } else {
        PipelineStore::SaveShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    }
    SetObjectName(instance.GetDevice(), module, GetShaderName(info.hw_stage, info.pgm_hash, perm_idx));
    return module;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              std::span<const u32> code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    const auto spv_t0 = std::chrono::steady_clock::now();
    struct SpvHitchScope {
        std::chrono::steady_clock::time_point t0;
        const Shader::Info& info;
        size_t perm;
        u64 draws_hint;
        ~SpvHitchScope() {
            const u64 ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count());
            BbStats::hitch_spv_sync_ns.fetch_add(ns, std::memory_order_relaxed);
            BbStats::hitch_spv_sync_count.fetch_add(1, std::memory_order_relaxed);
            if (ns >= 1'000'000ull || draws_hint >= 250) {
                BbStats::LogHitch(
                    "[BB-HITCH][SPV-SYNC] frame=%u %.2f ms stage=%u hash=%#llx perm=%zu draws=%llu\n",
                    BbStats::frame_number.load(std::memory_order_relaxed), double(ns) / 1e6,
                    unsigned(info.hw_stage), static_cast<unsigned long long>(info.pgm_hash), perm,
                    static_cast<unsigned long long>(draws_hint));
            }
        }
    } spv_hitch{spv_t0, info, perm_idx, last_frame_draws};
    CompileTimer timer;
    return CompileModuleRaw(pools, info, runtime_info, code, perm_idx, binding, nullptr, nullptr,
                            /*runtime_fast_opt=*/true);
}

PipelineCache::Result PipelineCache::LookUpProgram(PipelineSelection& sel,
                                                   const Shader::RuntimeInfo& runtime_info,
                                                   const Shader::ShaderParams& params,
                                                   Shader::Backend::Bindings& binding,
                                                   HwStage hw_stage,
                                                   const Program* direct_program) {
    // A draw-preparation worker only looks up, with its own copy of the program's Info.
    // Note: PrepareGraphicsPipeline already holds shared_lock<programs_mutex> and armed
    // runtime_fault_recover across the whole draw.
    auto& worker = *sel.worker;
    const Program* program = direct_program;
    if (!program) {
        const auto found = program_cache.find(params.hash);
        if (found == program_cache.end()) {
            worker.failed = true;
            return {};
        }
        program = found->second.get();
        DirectShaders::AttachProgramToSlot(params.Base(), params.hash,
                                           const_cast<Program*>(program));
    }
    if (!program->info_template) {
        worker.failed = true;
        return {};
    }
    auto& mru_slot = worker.info_mru[(reinterpret_cast<uintptr_t>(program) >> 6) & 63];
    PrepWorker::ProgramState* prog_state = nullptr;
    if (mru_slot.first == program && mru_slot.second != nullptr) [[likely]] {
        prog_state = mru_slot.second;
    } else {
        prog_state = &worker.infos.try_emplace(program, *program->info_template).first->second;
        mru_slot = {program, prog_state};
    }
    auto& info = prog_state->info;
    if (program->info_template->uses_paged_buffers && !info.uses_paged_buffers) [[unlikely]] {
        info = *program->info_template;
    }
    info.pgm_base = params.Base();
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    if (info.has_fetch_shader &&
        !FetchCodeDecodes(Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base))) {
        worker.failed = true; // the GPU thread reads back what the GPU wrote, or skips the draw
        return {};
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
    const auto& modules = program->modules;
    const auto it =
        (prog_state->last_perm_idx < modules.size() &&
         modules[prog_state->last_perm_idx].spec == spec)
            ? (modules.begin() + prog_state->last_perm_idx)
            : std::ranges::find(modules, spec, &Program::Module::spec);
    if (it == modules.end()) {
        worker.failed = true;
        return {};
    }
    info.AddBindings(binding);
    const size_t perm_idx = static_cast<size_t>(std::distance(modules.begin(), it));
    prog_state->last_perm_idx = perm_idx;
    worker.stages.push_back({program, &info, params.hash, hw_stage, info.pgm_base,
                             &info.flattened_ud_buf, info.sw_stage, static_cast<u32>(perm_idx)});
    return {&info, it->module, std::move(spec.fetch_shader_data),
            HashCombine(params.hash, perm_idx)};
}

PipelineCache::Result PipelineCache::GetProgram(PipelineSelection& sel, HwStage hw_stage,
                                                SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding,
                                                Program* direct_program) {
    EnsurePrewarmDone();
    if (sel.async_ctx) {
        return GetProgramAsyncWorker(sel, hw_stage, sw_stage, params, binding);
    }
    auto runtime_info = BuildRuntimeInfo(sel, hw_stage, sw_stage);
    if (sel.worker) {
        return LookUpProgram(sel, runtime_info, params, binding, hw_stage, direct_program);
    }

    Program* resolved = direct_program;
    if (!resolved) {
        const auto found = program_cache.find(params.hash); // this thread is the only writer
        if (found == program_cache.end()) {
            if (sel.async_probe) {
                sel.async_spv_miss = true;
                return {};
            }
            auto new_program = std::make_unique<Program>(hw_stage, sw_stage, params);
            const auto start = binding;
            const auto module =
                CompileModule(new_program->info, runtime_info, params.code, 0, binding);
            auto spec =
                Shader::StageSpecialization(new_program->info, runtime_info, profile, start);
            const u64 perm_hash = HashCombine(params.hash, 0);
            PipelineStore::SaveShaderMeta(new_program->info, spec, perm_hash, 0);
            new_program->AddPermut(module, std::move(spec));
            new_program->info_template = std::make_unique<Shader::Info>(new_program->info);
            Program* program = new_program.get();
            {
                std::unique_lock lk{programs_mutex};
                program_cache.emplace(params.hash, std::move(new_program));
            }
            DirectShaders::NoteProgramCreated(params.hash, program);
            if (&sel == &this->sel) {
                NoteFastProgram(program, hw_stage, sw_stage, params.Base(), params.hash, 0);
            }
            return {&program->info, module, program->modules[0].spec.fetch_shader_data, perm_hash};
        }
        resolved = found.value().get();
        DirectShaders::AttachProgramToSlot(params.Base(), params.hash, resolved);
    }

    Program& program = *resolved;
    if (!program.info_template) {
        // Programs preloaded from the pipeline cache get their template on first use.
        std::unique_lock lk{programs_mutex};
        program.info_template = std::make_unique<Shader::Info>(program.info);
    }
    auto& info = program.info;
    info.pgm_base = params.Base(); // the inline constant buffer addresses depend on it
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    if (info.has_fetch_shader) {
        const u32* fetch = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
        if (!sel.worker && fetch && program_write_sync) {
            program_write_sync(reinterpret_cast<VAddr>(fetch), 256);
        }
        if (!FetchCodeDecodes(fetch)) {
            SyncProgramMemory(reinterpret_cast<VAddr>(fetch), 4096);
            if (!FetchCodeDecodes(fetch)) {
                static std::atomic<u32> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_WARNING(Render_Vulkan,
                                "program {:#x}: its fetch shader at {} does not decode "
                                "({:#010x} {:#010x}): draw skipped",
                                params.hash, fmt::ptr(fetch), fetch[0], fetch[1]);
                }
                return {};
            }
        }
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    // Consecutive draws of a program almost always use the same permutation.
    auto& modules = program.modules;
    auto it = program.last_used < modules.size() && modules[program.last_used].spec == spec
                  ? modules.begin() + program.last_used
                  : std::ranges::find(modules, spec, &Program::Module::spec);
    size_t perm_idx;
    if (it != modules.end()) {
        perm_idx = std::distance(modules.begin(), it);
        program.last_used = perm_idx;
        info.AddBindings(binding);
    } else {
        if (sel.async_probe) {
            sel.async_spv_miss = true;
            return {};
        }
        perm_idx = modules.size();
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        const auto module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);
        std::unique_lock lk{programs_mutex};
        if (new_info.uses_paged_buffers && !info.uses_paged_buffers) {
            info.uses_paged_buffers = true;
            info.buffers = new_info.buffers;
            if (program.info_template) {
                retired_info_templates.push_back(std::move(program.info_template));
            }
            program.info_template = std::make_unique<Shader::Info>(info);
        }
        spec.info = &info;
        PipelineStore::SaveShaderMeta(new_info, spec, HashCombine(params.hash, perm_idx), perm_idx);
        program.AddPermut(module, std::move(spec));
    }
    if (&sel == &this->sel) {
        NoteFastProgram(&program, hw_stage, sw_stage, params.Base(), params.hash, perm_idx);
    }
    return {&info, modules[perm_idx].module, modules[perm_idx].spec.fetch_shader_data,
            HashCombine(params.hash, perm_idx)};
}

std::string PipelineCache::GetShaderName(HwStage stage, u64 hash, std::optional<size_t> perm) {
    BbGoldenEngine::ShaderManifestInfo manifest{};
    if (BbGoldenEngine::QueryShaderManifest(hash, 0, 0, &manifest) && manifest.bundle &&
        manifest.name) {
        if (perm) {
            return fmt::format("{}_{:#018x}_{}_{}_{}", stage, hash, *perm, manifest.bundle,
                               manifest.name);
        }
        return fmt::format("{}_{:#018x}_{}_{}", stage, hash, manifest.bundle, manifest.name);
    }
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }
    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

namespace {

/// Evaluates `RefreshFlatBuf()` + FetchShader decode + `StageSpecialization` on a worker-local
/// `Shader::Info` with ZERO mutexes held inside `sigsetjmp`, so a VRAM read-trap (`siglongjmp`)
/// can never leak `programs_mutex`, `walkers.mutex`, or compiler locks.
bool SafeGuestSpecEval(Shader::Info& info, const Shader::RuntimeInfo& runtime_info,
                       const Shader::Profile& profile,
                       const Shader::Backend::Bindings& start_bnd,
                       Shader::StageSpecialization& out_spec) noexcept {
    info.flattened_ud_buf.resize(info.srt_info.flattened_bufsize_dw);
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0) != 0) {
        runtime_fault_recover = nullptr;
        return false;
    }
    runtime_fault_recover = &recover;
    info.RefreshFlatBuf();
    if (info.has_fetch_shader &&
        !FetchCodeDecodes(
            Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base))) {
        runtime_fault_recover = nullptr;
        return false;
    }
    out_spec = Shader::StageSpecialization(info, runtime_info, profile, start_bnd);
    runtime_fault_recover = nullptr;
    return true;
}

} // namespace

PipelineCache::Result PipelineCache::GetProgramAsyncWorker(PipelineSelection& sel, HwStage hw_stage,
                                                           SwStage sw_stage,
                                                           const Shader::ShaderParams& params,
                                                           Shader::Backend::Bindings& binding) {
    auto& ctx = *sel.async_ctx;
    const u32 st_idx = static_cast<u32>(hw_stage);
    const bool has_snap = (st_idx < ctx.stages.size() && ctx.stages[st_idx].valid);
    const VAddr true_pgm_base = has_snap ? ctx.stages[st_idx].pgm_base : params.Base();
    struct ScopedFlatReplay {
        explicit ScopedFlatReplay(const std::vector<u32>* buf) noexcept {
            Shader::flat_buf_replay = (buf && !buf->empty()) ? buf : nullptr;
        }
        ~ScopedFlatReplay() noexcept {
            Shader::flat_buf_replay = nullptr;
        }
    } flat_replay_guard{has_snap ? &ctx.stages[st_idx].flat_buf : nullptr};
    auto runtime_info = BuildRuntimeInfo(sel, hw_stage, sw_stage);

    // 1. Check if an earlier pass inside this same worker job already built an identical spec.
    for (auto& bs : ctx.built_stages) {
        if (bs.hash == params.hash && bs.hw_stage == hw_stage && bs.sw_stage == sw_stage) {
            Shader::Info cand_info = bs.info;
            cand_info.pgm_base = true_pgm_base;
            cand_info.user_data = params.user_data;
            Shader::StageSpecialization cand_spec;
            if (!SafeGuestSpecEval(cand_info, runtime_info, profile, binding, cand_spec)) {
                ctx.failed = true;
                return {};
            }
            if (bs.spec == cand_spec) {
                cand_info.AddBindings(binding);
                auto& stored = ctx.owned_infos.emplace_back(std::move(cand_info));
                return {&stored, bs.module, std::move(cand_spec.fetch_shader_data),
                        HashCombine(params.hash, bs.assumed_perm)};
            }
        }
    }

    // 2. Check existing program_cache: copy template under shared_lock, evaluate guest memory
    //    WITHOUT holding any lock, then match `prog->modules` under shared_lock.
    size_t existing_mod_count = 0;
    std::optional<Shader::Info> cached_tmpl;
    {
        std::shared_lock lk{programs_mutex};
        if (const auto found = program_cache.find(params.hash);
            found != program_cache.end() && found->second) {
            const Program* prog = found->second.get();
            existing_mod_count = prog->modules.size();
            cached_tmpl.emplace(prog->info_template ? *prog->info_template : prog->info);
        }
    }
    if (cached_tmpl.has_value()) {
        Shader::Info& local_info = *cached_tmpl;
        local_info.pgm_base = true_pgm_base;
        local_info.user_data = params.user_data;
        Shader::StageSpecialization spec;
        if (!SafeGuestSpecEval(local_info, runtime_info, profile, binding, spec)) {
            ctx.failed = true;
            return {};
        }
        vk::ShaderModule hit_module{};
        size_t hit_perm = 0;
        bool hit = false;
        {
            std::shared_lock lk{programs_mutex};
            if (const auto found = program_cache.find(params.hash);
                found != program_cache.end() && found->second) {
                const Program* prog = found->second.get();
                existing_mod_count = prog->modules.size();
                const auto it = std::ranges::find(prog->modules, spec, &Program::Module::spec);
                if (it != prog->modules.end()) {
                    hit_perm = static_cast<size_t>(std::distance(prog->modules.begin(), it));
                    hit_module = it->module;
                    hit = true;
                }
            }
        }
        if (hit) {
            local_info.AddBindings(binding);
            auto& stored = ctx.owned_infos.emplace_back(std::move(local_info));
            return {&stored, hit_module, std::move(spec.fetch_shader_data),
                    HashCombine(params.hash, hit_perm)};
        }
    }

    // 3. Miss: compile GCN -> SPIR-V -> spirv_opt -> VkShaderModule on worker's private Pools
    //    (runtime_fault_recover is nullptr here — params.code is already in host memory).
    size_t extra_for_hash = 0;
    for (const auto& bs : ctx.built_stages) {
        if (bs.hash == params.hash) {
            ++extra_for_hash;
        }
    }
    const size_t assumed_perm = existing_mod_count + extra_for_hash;

    thread_local std::unique_ptr<Shader::Pools> tls_worker_pools;
    if (!tls_worker_pools) {
        tls_worker_pools = std::make_unique<Shader::Pools>();
    }

    Shader::Info new_info(hw_stage, sw_stage, params);
    new_info.pgm_base = true_pgm_base;
    const auto start_bnd = binding;
    std::vector<u32> raw_spv;
    std::vector<u32> opt_spv;
    const vk::ShaderModule mod = CompileModuleRaw(
        *tls_worker_pools, new_info, runtime_info, params.code, assumed_perm, binding, &raw_spv,
        Shader::Backend::SPIRV::SpirvOptEnabled() ? &opt_spv : nullptr,
        /*runtime_fast_opt=*/true);

    Shader::StageSpecialization spec;
    if (!SafeGuestSpecEval(new_info, runtime_info, profile, start_bnd, spec)) {
        if (mod) {
            instance.GetDevice().destroyShaderModule(mod);
        }
        ctx.failed = true;
        return {};
    }
    auto fetch_copy = spec.fetch_shader_data;
    auto& built = ctx.built_stages.emplace_back(AsyncCompileCtx::BuiltStage{
        .hw_stage = hw_stage,
        .sw_stage = sw_stage,
        .hash = params.hash,
        .pgm_base = true_pgm_base,
        .assumed_perm = assumed_perm,
        .info = std::move(new_info),
        .spec = std::move(spec),
        .module = mod,
        .raw_spv = std::move(raw_spv),
        .opt_spv = std::move(opt_spv),
    });
    return {&built.info, mod, std::move(fetch_copy), HashCombine(params.hash, assumed_perm)};
}

void PipelineCache::DrainCompletedAsyncSpvJobs() {
    if (!has_completed_spv.load(std::memory_order_acquire)) [[likely]] {
        return;
    }
    constexpr size_t kMaxDrainPerCall = 64;
    std::vector<std::shared_ptr<AsyncSpvJob>> done_jobs;
    {
        std::scoped_lock lk{async_spv_mutex};
        if (completed_spv_jobs.empty()) {
            has_completed_spv.store(false, std::memory_order_relaxed);
            return;
        }
        if (completed_spv_jobs.size() <= kMaxDrainPerCall) {
            done_jobs.swap(completed_spv_jobs);
            has_completed_spv.store(false, std::memory_order_relaxed);
        } else {
            done_jobs.assign(std::make_move_iterator(completed_spv_jobs.begin()),
                             std::make_move_iterator(completed_spv_jobs.begin() + kMaxDrainPerCall));
            completed_spv_jobs.erase(completed_spv_jobs.begin(),
                                     completed_spv_jobs.begin() + kMaxDrainPerCall);
            has_completed_spv.store(true, std::memory_order_release);
        }
    }
    for (auto& job : done_jobs) {
        pending_spv_jobs.erase(job->sig);
        if (!job->ok) {
            if (failed_spv_sigs.size() < 4096) {
                failed_spv_sigs.insert(job->sig);
            }
            for (auto& bs : job->ctx.built_stages) {
                if (bs.module) {
                    instance.GetDevice().destroyShaderModule(bs.module);
                }
            }
            continue;
        }
        bool added_any = false;
        for (auto& bs : job->ctx.built_stages) {
            size_t actual_perm = 0;
            auto found = program_cache.find(bs.hash);
            if (found == program_cache.end()) {
                Shader::ShaderParams dummy_params{
                    .user_data = std::span<const u32, Gcn::NUM_USER_DATA>{
                        job->ctx.stages[static_cast<u32>(bs.hw_stage)].user_data},
                    .code = {},
                    .hash = bs.hash,
                };
                auto new_prog = std::make_unique<Program>(bs.hw_stage, bs.sw_stage, dummy_params);
                new_prog->info = bs.info;
                new_prog->info.pgm_base = bs.pgm_base;
                bs.spec.info = &new_prog->info;
                actual_perm = 0;
                if (!bs.raw_spv.empty()) {
                    PipelineStore::SaveShaderBinary(std::move(bs.raw_spv), bs.hash, 0);
                }
                if (!bs.opt_spv.empty()) {
                    PipelineStore::SaveOptShaderBinary(std::move(bs.opt_spv), bs.hash, 0);
                }
                PipelineStore::SaveShaderMeta(new_prog->info, bs.spec, HashCombine(bs.hash, 0), 0);
                new_prog->AddPermut(bs.module, std::move(bs.spec));
                bs.module = nullptr;
                new_prog->info_template = std::make_unique<Shader::Info>(new_prog->info);
                Program* raw_prog = new_prog.get();
                {
                    std::unique_lock lk{programs_mutex};
                    program_cache.emplace(bs.hash, std::move(new_prog));
                }
                DirectShaders::NoteProgramCreated(bs.hash, raw_prog);
                DirectShaders::AttachProgramToSlot(bs.pgm_base, bs.hash, raw_prog);
                added_any = true;
            } else {
                Program& prog = *found.value();
                if (!prog.info_template) {
                    std::unique_lock lk{programs_mutex};
                    prog.info_template = std::make_unique<Shader::Info>(prog.info);
                }
                const auto existing_it =
                    std::ranges::find(prog.modules, bs.spec, &Program::Module::spec);
                if (existing_it != prog.modules.end()) {
                    actual_perm = static_cast<size_t>(std::distance(prog.modules.begin(), existing_it));
                    if (bs.module) {
                        instance.GetDevice().destroyShaderModule(bs.module);
                        bs.module = nullptr;
                    }
                } else {
                    actual_perm = prog.modules.size();
                    std::unique_lock lk{programs_mutex};
                    if (bs.info.uses_paged_buffers && !prog.info.uses_paged_buffers) {
                        prog.info.uses_paged_buffers = true;
                        prog.info.buffers = bs.info.buffers;
                        if (prog.info_template) {
                            retired_info_templates.push_back(std::move(prog.info_template));
                        }
                        prog.info_template = std::make_unique<Shader::Info>(prog.info);
                    }
                    bs.spec.info = &prog.info;
                    if (!bs.raw_spv.empty()) {
                        PipelineStore::SaveShaderBinary(std::move(bs.raw_spv), bs.hash, actual_perm);
                    }
                    if (!bs.opt_spv.empty()) {
                        PipelineStore::SaveOptShaderBinary(std::move(bs.opt_spv), bs.hash, actual_perm);
                    }
                    PipelineStore::SaveShaderMeta(bs.info, bs.spec,
                                                  HashCombine(bs.hash, actual_perm), actual_perm);
                    prog.AddPermut(bs.module, std::move(bs.spec));
                    bs.module = nullptr;
                    added_any = true;
                }
                DirectShaders::AttachProgramToSlot(bs.pgm_base, bs.hash, &prog);
            }
            if (actual_perm != bs.assumed_perm) {
                const u64 expected_stage_hash = HashCombine(bs.hash, bs.assumed_perm);
                const u32 sw_idx = static_cast<u32>(bs.sw_stage);
                if (sw_idx < job->key.stage_hashes.size() &&
                    job->key.stage_hashes[sw_idx] == expected_stage_hash) {
                    const u64 actual_stage_hash = HashCombine(bs.hash, actual_perm);
                    job->key.stage_hashes[sw_idx] = actual_stage_hash;
                    if (job->pipeline) {
                        job->pipeline->UpdateStageHash(sw_idx, actual_stage_hash);
                    }
                }
            }
        }
        if (job->pipeline) {
            bool all_stages_bound = true;
            for (u32 s = 0; s < MaxShaderStages; ++s) {
                const u64 ph = job->stage_pgm_hashes[s];
                if (!ph) {
                    job->pipeline->SetStageInfo(s, nullptr);
                    continue;
                }
                const auto pit = program_cache.find(ph);
                if (pit == program_cache.end() || !pit->second) {
                    all_stages_bound = false;
                    break;
                }
                job->pipeline->SetStageInfo(s, &pit->second->info);
            }
            if (all_stages_bound && !graphics_pipelines.contains(job->key)) {
                const u64 pso_hash = std::hash<GraphicsPipelineKey>{}(job->key);
                PipelineStore::SaveGraphics(job->key, pso_hash, job->sdata);
                NoteDriverPipelineCacheDirty();
                std::unique_lock gr_lk{graphics_pipelines_mutex};
                graphics_pipelines.insert_or_assign(job->key, std::move(job->pipeline));
                added_any = true;
            }
        }
        if (!added_any && failed_spv_sigs.size() < 4096) {
            // Worker produced nothing new (e.g. spec mismatch between Stage A probe and worker
            // replay): prevent infinite re-queue storms by allowing one Stage A sync compile.
            failed_spv_sigs.insert(job->sig);
        }
    }
}

bool PipelineCache::QueueAsyncSpvJob(DrawIndirectParams params, bool indirect) {
    (void)indirect;
    if (!sel.regs || sel.regs->stage_enable.hs_en) {
        return false; // Tessellation stages read constant buffers at selection time
    }
    if (!compiler) {
        const char* env = std::getenv("BB_ASYNC_PIPELINE_THREADS");
        const u32 threads = std::max(
            env && *env ? u32(std::strtoul(env, nullptr, 10))
                        : std::clamp(std::thread::hardware_concurrency() / 3u, 2u, 8u),
            1u);
        compiler = std::make_unique<PipelineCompiler>(threads);
    }

    const auto& regs = *sel.regs;

    // 1. Compute a zero-allocation deduplication signature across PSO registers, stage program
    //    hashes, and FetchShader bytecode — NEVER hashing per-draw EUD/V# CUE pointers!
    std::array<u32, FastSelect::MaxSigWords> fast_sig;
    const u32 fast_sig_len =
        BuildFastSelectSig(regs, params, instance.IsVertexInputDynamicState(),
                           instance.IsTessellationSupported(), fast_sig);
    u64 sig = fast_sig_len != 0
                  ? XXH3_64bits(fast_sig.data(), fast_sig_len * sizeof(u32))
                  : HashCombine(u64(regs.stage_enable.raw), u64(regs.primitive_type));
    if (fast_sig_len == 0) {
        sig = HashCombine(sig, (u64(std::bit_cast<u32>(regs.depth_control)) << 32) |
                                   u64(regs.color_target_mask.raw));
        sig = HashCombine(sig, u64(regs.color_export_format.raw));
        sig = HashCombine(sig, u64(std::bit_cast<u32>(regs.blend_control[0])));
        sig = HashCombine(sig, (u64(params.vertex_sgpr_offset) << 32) |
                                   u64(params.instance_sgpr_offset));
    }

    struct PreStage {
        bool enabled = false;
        VAddr pgm_base = 0;
        const u32* code_ptr = nullptr;
        size_t code_dwords = 0;
        u64 hash = 0;
        bool has_fetch = false;
        u32 fetch_sgpr = ~0u;
        const u32* fetch_ptr = nullptr;
        size_t fetch_dwords = 0;
        u32 vb_sgpr_mask = 0;
        std::array<u32, Gcn::NUM_USER_DATA> vb_max_dwords{};
        Program* cached_prog = nullptr;
    };
    std::array<PreStage, 8> pre_stages{};
    const u32* copy_code_ptr = nullptr;
    size_t copy_code_dwords = 0;
    u64 copy_code_hash = 0;

    static constexpr HwStage kSnapStages[] = {HwStage::Fragment, HwStage::Vertex, HwStage::Geometry,
                                              HwStage::Export};
    for (HwStage hw : kSnapStages) {
        const u32 in = static_cast<u32>(hw);
        if (!regs.stage_enable.IsStageEnabled(in)) {
            continue;
        }
        const auto* pgm = regs.ProgramForStage(in);
        const VAddr pgm_base = pgm ? pgm->Address<VAddr>() : 0;
        if (pgm_base == 0) {
            continue;
        }
        auto& ps = pre_stages[in];
        DirectShaders::Snapshot direct_snap;
        if (DirectShaders::TryLookup(pgm_base, direct_snap, /*is_worker=*/false) &&
            direct_snap.cpu_immutable != 0) {
            ps.code_ptr = reinterpret_cast<const u32*>(pgm_base);
            ps.code_dwords = direct_snap.code_dwords;
            ps.hash = direct_snap.shader_hash;
        } else {
            if (!LooksLikeProgram(reinterpret_cast<const u32*>(pgm_base), hw, true)) {
                return false;
            }
            const auto p = Gcn::GetParams(*pgm);
            if (!program_cache.contains(p.hash) && !ProgramDecodes(p, hw)) {
                return false;
            }
            ps.code_ptr = p.code.data();
            ps.code_dwords = p.code.size();
            ps.hash = p.hash;
        }
        ps.enabled = true;
        ps.pgm_base = pgm_base;
        sig = HashCombine(sig, ps.hash);

        if (const auto found = program_cache.find(ps.hash);
            found != program_cache.end() && found->second) {
            ps.cached_prog = found->second.get();
            if (!ps.cached_prog->info_template) {
                std::unique_lock lk{programs_mutex};
                ps.cached_prog->info_template =
                    std::make_unique<Shader::Info>(ps.cached_prog->info);
            }
            ps.has_fetch = ps.cached_prog->info.has_fetch_shader;
            ps.fetch_sgpr = ps.cached_prog->info.fetch_shader_sgpr_base;
            // Walk SRT descriptors synchronously RIGHT NOW on Stage A while the GFX Per-Frame
            // Command Heap (`0x1028...`) holds this exact draw's live T#/V#/S# descriptors!
            auto& live_info = ps.cached_prog->info;
            live_info.pgm_base = pgm_base;
            live_info.user_data = pgm->user_data;
            live_info.RefreshFlatBuf();
            // Hash ONLY address-independent StageSpecialization fields (zero GPU pointer bits!):
            // hashing raw `flattened_ud_buf[k] & 0xFFFF0000u` included upper address bits of every
            // V#/T# ring buffer (`0x1028xxxx`), breaking deduplication and causing 100k+ duplicate
            // background compiles on repeated draws of the same material.
            u64 sharp_fmt_sig = 0x9e3779b97f4a7c15ULL;
            for (const auto& desc : live_info.buffers) {
                const auto sharp = desc.GetSharp(live_info);
                u64 w = sharp ? 1ULL : 0ULL;
                if (sharp) {
                    w |= (u64(sharp.GetStride()) & 0x3FFFULL) << 1;
                    w |= u64(Shader::IsPagedBuffer(profile.paged_buffers, sharp,
                                                   live_info.pgm_hash,
                                                   profile.buffer_copy_shader_hash))
                         << 15;
                    w |= u64(desc.is_formatted) << 16;
                    w |= u64(sharp.swizzle_enable) << 17;
                    if (desc.is_formatted) {
                        const auto ds = sharp.DstSelect();
                        const u64 ds_bits = u64(static_cast<u8>(ds.r)) |
                                            (u64(static_cast<u8>(ds.g)) << 3) |
                                            (u64(static_cast<u8>(ds.b)) << 6) |
                                            (u64(static_cast<u8>(ds.a)) << 9);
                        w |= (u64(static_cast<u32>(sharp.GetDataFmt())) & 0x3FULL) << 18;
                        w |= (u64(static_cast<u32>(sharp.GetNumberFmt())) & 0xFULL) << 24;
                        w |= (ds_bits & 0xFFFULL) << 28;
                        w |= (u64(static_cast<u32>(sharp.GetNumberConversion())) & 0xFULL) << 40;
                    }
                    if (sharp.swizzle_enable) {
                        w |= (u64(sharp.index_stride) & 0x3ULL) << 44;
                        w |= (u64(sharp.element_size) & 0x3ULL) << 46;
                    }
                }
                sharp_fmt_sig = HashCombine(sharp_fmt_sig, w);
            }
            for (const auto& desc : live_info.images) {
                const auto sharp = desc.GetSharp(live_info);
                u64 w = sharp ? 1ULL : 0ULL;
                w |= (u64(static_cast<u32>(sharp.GetViewType(desc.is_array))) & 0xFULL) << 1;
                w |= u64(::Gcn::IsInteger(sharp.GetNumberFmt())) << 5;
                w |= u64(desc.is_written) << 6;
                w |= u64(sharp.IsCube()) << 7;
                if (desc.is_written) {
                    const auto ds = sharp.DstSelect();
                    const u64 ds_bits = u64(static_cast<u8>(ds.r)) |
                                        (u64(static_cast<u8>(ds.g)) << 3) |
                                        (u64(static_cast<u8>(ds.b)) << 6) |
                                        (u64(static_cast<u8>(ds.a)) << 9);
                    w |= (ds_bits & 0xFFFULL) << 8;
                } else {
                    w |= u64(sharp.GetNumberFmt() == ::Gcn::NumberFormat::Srgb) << 8;
                }
                w |= (u64(static_cast<u32>(sharp.GetNumberConversion())) & 0xFULL) << 20;
                w |= (u64(desc.NumBindings(live_info)) & 0xFFULL) << 24;
                sharp_fmt_sig = HashCombine(sharp_fmt_sig, w);
            }
            for (const auto& desc : live_info.fmasks) {
                const auto sharp = desc.GetSharp(live_info);
                u64 w = sharp ? 1ULL : 0ULL;
                w |= (u64(sharp.width) & 0xFFFFULL) << 1;
                w |= (u64(sharp.height) & 0xFFFFULL) << 17;
                sharp_fmt_sig = HashCombine(sharp_fmt_sig, w);
            }
            for (const auto& desc : live_info.samplers) {
                const auto sharp = desc.GetSharp(live_info);
                u64 w = sharp ? 1ULL : 0ULL;
                if (sharp) {
                    w |= u64(sharp.force_unnormalized) << 1;
                    w |= u64(sharp.force_degamma) << 2;
                }
                sharp_fmt_sig = HashCombine(sharp_fmt_sig, w);
            }
            sig = HashCombine(sig, sharp_fmt_sig);
        } else if (hw == HwStage::Vertex || hw == HwStage::Export) {
            const u32* pc = ps.code_ptr;
            const u32* end = pc + std::min<size_t>(ps.code_dwords, 64);
            while (pc < end) {
                const auto inst = GcnIsa::Decode(pc);
                if (inst.opcode == GcnIsa::Opcode::S_ENDPGM) {
                    break;
                }
                if (inst.opcode == GcnIsa::Opcode::S_SWAPPC_B64) {
                    ps.has_fetch = true;
                    ps.fetch_sgpr = inst.src[0].code;
                    break;
                }
            }
        }

        if (ps.has_fetch && ps.fetch_sgpr + 1 < Gcn::NUM_USER_DATA) {
            const VAddr faddr =
                (VAddr(pgm->user_data[ps.fetch_sgpr]) |
                 (VAddr(pgm->user_data[ps.fetch_sgpr + 1]) << 32)) &
                0xFFFFFFFFFFFFULL;
            if (faddr < 0x100000ULL || faddr >= (1ULL << 40) || (faddr & 3ULL) != 0) {
                return false;
            }
            const u32* fptr = reinterpret_cast<const u32*>(faddr);
            if (program_write_sync) {
                program_write_sync(faddr, 256);
            }
            if (!FetchCodeDecodes(fptr)) {
                SyncProgramMemory(faddr, 4096);
                if (!FetchCodeDecodes(fptr)) {
                    return false;
                }
            }
            const u32* pc = fptr;
            for (u32 ins = 0; ins < 256; ++ins) {
                const auto finst = GcnIsa::Decode(pc);
                if (finst.opcode == GcnIsa::Opcode::S_SETPC_B64) {
                    break;
                }
                if (finst.inst_class == GcnIsa::InstClass::ScalarMemRd) {
                    const u32 b_sgpr = finst.src[0].code * 2u;
                    if (b_sgpr + 1 < Gcn::NUM_USER_DATA) {
                        ps.vb_sgpr_mask |= (1u << b_sgpr);
                        ps.vb_max_dwords[b_sgpr] =
                            std::max<u32>(ps.vb_max_dwords[b_sgpr], finst.control.smrd.offset + 4u);
                    }
                }
            }
            ps.fetch_ptr = fptr;
            ps.fetch_dwords = static_cast<size_t>(pc - fptr);
            sig = HashCombine(sig, XXH3_64bits(fptr, ps.fetch_dwords * sizeof(u32)));
            // Also fold vertex-attribute V# format/swizzle dwords (dw3) into `sig` on Stage A!
            for (u32 b = 0; b + 1 < Gcn::NUM_USER_DATA; b += 2) {
                if (((ps.vb_sgpr_mask >> b) & 1u) == 0) {
                    continue;
                }
                const VAddr vb_addr =
                    (VAddr(pgm->user_data[b]) | (VAddr(pgm->user_data[b + 1]) << 32)) &
                    0xFFFFFFFFFFFFULL;
                if (vb_addr >= 0x100000ULL && vb_addr < (1ULL << 40) && (vb_addr & 3ULL) == 0) {
                    const u32* vb_words = reinterpret_cast<const u32*>(vb_addr);
                    const u32 ndw = std::min<u32>(ps.vb_max_dwords[b], 128u);
                    for (u32 d = 3; d < ndw; d += 4) {
                        sig = HashCombine(sig, u64(vb_words[d]));
                    }
                }
            }
        }
    }

    if (regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry))) {
        const VAddr copy_base = regs.vs_program.Address<VAddr>();
        if (copy_base == 0) {
            return false;
        }
        DirectShaders::Snapshot copy_snap;
        if (DirectShaders::TryLookup(copy_base, copy_snap, /*is_worker=*/false) &&
            copy_snap.cpu_immutable != 0) {
            copy_code_ptr = reinterpret_cast<const u32*>(copy_base);
            copy_code_dwords = copy_snap.code_dwords;
            copy_code_hash = copy_snap.shader_hash;
        } else {
            if (!LooksLikeProgram(reinterpret_cast<const u32*>(copy_base), HwStage::Vertex, true)) {
                return false;
            }
            const auto cp = Gcn::GetParams(regs.vs_program);
            copy_code_ptr = cp.code.data();
            copy_code_dwords = cp.code.size();
            copy_code_hash = cp.hash;
        }
        sig = HashCombine(sig, copy_code_hash);
    }

    // Fast deduplication BEFORE any heap allocation: 400 draws of the same material in a single
    // frame collapse to 1 background job in < 10 us on Stage A.
    if (failed_spv_sigs.contains(sig)) {
        return false; // Fall back once to synchronous Stage A compilation
    }
    if (pending_spv_jobs.contains(sig)) {
        ++async_skipped;
        return true;
    }
    constexpr size_t kMaxPendingSpvJobs = 128;
    if (pending_spv_jobs.size() >= kMaxPendingSpvJobs) {
        ++async_skipped;
        return true;
    }

    VAddr cue_base = 0, cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);

    auto job = std::make_shared<AsyncSpvJob>();
    job->sig = sig;
    job->regs_copy = regs;
    job->params = params;

    for (HwStage hw : kSnapStages) {
        const u32 in = static_cast<u32>(hw);
        const auto& ps = pre_stages[in];
        if (!ps.enabled) {
            continue;
        }
        const auto* pgm = regs.ProgramForStage(in);
        auto& snap = job->ctx.stages[in];
        snap.code.assign(ps.code_ptr, ps.code_ptr + ps.code_dwords);
        snap.hash = ps.hash;
        snap.pgm_base = ps.pgm_base;
        snap.user_data = pgm->user_data;
        if (ps.cached_prog) {
            snap.flat_buf = ps.cached_prog->info.flattened_ud_buf;
        }
        snap.valid = true;

        for (u32 i = 0; i + 1 < Gcn::NUM_USER_DATA; i += 2) {
            const VAddr raw_ptr =
                (VAddr(snap.user_data[i]) | (VAddr(snap.user_data[i + 1]) << 32)) &
                0xFFFFFFFFFFFFULL;
            if (raw_ptr == 0) {
                continue;
            }
            if (ps.has_fetch && i == ps.fetch_sgpr && ps.fetch_ptr) {
                auto& blob =
                    job->ctx.owned_blobs.emplace_back(ps.fetch_ptr, ps.fetch_ptr + ps.fetch_dwords);
                const uintptr_t host_ptr = reinterpret_cast<uintptr_t>(blob.data());
                snap.user_data[i] = static_cast<u32>(host_ptr);
                snap.user_data[i + 1] = static_cast<u32>(host_ptr >> 32);
            } else if (((ps.vb_sgpr_mask >> i) & 1u) != 0 &&
                       raw_ptr >= 0x100000ULL && raw_ptr < (1ULL << 40) && (raw_ptr & 3ULL) == 0) {
                // Exact `PtrVertexBufferTable` (e.g. `s[4:5]` in `0x1028...` GFX Command Heap)
                // referenced by FetchShader `s_load_dwordx4` — copy exact `vb_max_dwords[i]`.
                const size_t ndw = std::min<size_t>(ps.vb_max_dwords[i], 256u);
                if (ndw >= 4) {
                    const u32* src = reinterpret_cast<const u32*>(raw_ptr);
                    auto& blob = job->ctx.owned_blobs.emplace_back(src, src + ndw);
                    const uintptr_t host_ptr = reinterpret_cast<uintptr_t>(blob.data());
                    snap.user_data[i] = static_cast<u32>(host_ptr);
                    snap.user_data[i + 1] = static_cast<u32>(host_ptr >> 32);
                }
            } else if (!ps.cached_prog && cue_base != 0 && raw_ptr >= cue_base &&
                       raw_ptr < cue_end && (raw_ptr & 3ULL) == 0) {
                const size_t copy_bytes =
                    static_cast<size_t>(std::min<u64>(cue_end - raw_ptr, 1024ULL)) & ~size_t(3);
                if (copy_bytes >= 16) {
                    const u32* src = reinterpret_cast<const u32*>(raw_ptr);
                    auto& blob =
                        job->ctx.owned_blobs.emplace_back(src, src + (copy_bytes / sizeof(u32)));
                    const uintptr_t host_ptr = reinterpret_cast<uintptr_t>(blob.data());
                    snap.user_data[i] = static_cast<u32>(host_ptr);
                    snap.user_data[i + 1] = static_cast<u32>(host_ptr >> 32);
                }
            }
        }
        // Keep `snap.flat_buf[0..15]` synchronized with the relocated host pointers in
        // `snap.user_data` so `GetFetchShaderCode(info)` (which reads `info.UserData()`) never
        // dereferences stale per-frame GFX command heap pointers on the worker thread.
        if (!snap.flat_buf.empty()) {
            const size_t n = std::min<size_t>(snap.flat_buf.size(), snap.user_data.size());
            std::memcpy(snap.flat_buf.data(), snap.user_data.data(), n * sizeof(u32));
        }
    }
    if (copy_code_ptr && copy_code_dwords > 0) {
        job->ctx.vs_copy_code.assign(copy_code_ptr, copy_code_ptr + copy_code_dwords);
        job->ctx.vs_copy_hash = copy_code_hash;
    }

    pending_spv_jobs.emplace(sig, job);
    ++async_started;
    ++async_skipped;
    ++g_bb_compiles;
    BbStats::hitch_pso_async_count.fetch_add(1, std::memory_order_relaxed);
    BbStats::shader_async_submitted.fetch_add(1u, std::memory_order_relaxed);

    compiler->Submit([this, job]() mutable {
        PipelineSelection wsel{};
        wsel.regs = &job->regs_copy;
        wsel.draw_indirect_params = job->params;
        wsel.async_ctx = &job->ctx;

        const bool key_ok = RefreshGraphicsKey(wsel) && !job->ctx.failed;
        if (key_ok) {
            job->key = wsel.graphics_key;
            for (u32 s = 0; s < MaxShaderStages; ++s) {
                job->stage_pgm_hashes[s] = wsel.infos[s] ? wsel.infos[s]->pgm_hash : 0;
            }
            bool already_built = false;
            {
                std::shared_lock gr_lk{graphics_pipelines_mutex};
                already_built = graphics_pipelines.contains(wsel.graphics_key);
            }
            if (!already_built) {
                job->pipeline = std::make_unique<GraphicsPipeline>(
                    instance, scheduler, desc_heap, profile, wsel.graphics_key, *pipeline_cache,
                    wsel.infos, wsel.runtime_infos, wsel.fetch_shader, wsel.modules, job->sdata,
                    false);
            }
            job->ok = true;
        }

        {
            std::scoped_lock lk{async_spv_mutex};
            completed_spv_jobs.push_back(std::move(job));
            has_completed_spv.store(true, std::memory_order_release);
        }
        BbStats::shader_async_completed.fetch_add(1u, std::memory_order_relaxed);
    });
    return true;
}

} // namespace Vulkan
