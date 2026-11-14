// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 4 (Path 2A): Direct Multi-Threaded Vulkan 1.3 Secondary Command Buffer Recorder.
//
// Substep 4.1 provides the complete lock-free RHI recording foundation controlled by
// `BB_MT_DIRECT_VK` (=0 off default, =1 scene passes, =2 all eligible passes):
//   - Zero static TLS (`+0 B` `PT_TLS` delta): all 8 worker slots (`0` = master render thread,
//     `1..5` = `GXWorker:1..5`, `6..7` = spare) are 64-byte cache-line isolated on the heap.
//   - Per-frame (`kFrameSlices = 4`) x per-worker `vk::UniqueCommandPool` and recyclable secondary
//     `vk::CommandBuffer`s (`VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT` +
//     `vk::CommandBufferInheritanceRenderingInfo` for Vulkan 1.3 Dynamic Rendering).
//   - Per-worker lock-free host-mapped `Stream` constant ring (`4 MiB` per worker), private
//     `GfxPushStateTracker` (delta push descriptors + push constants), private `MtWorkerPsoState`
//     (lock-free `MtFastPsoTable`), and private `DirectBuffers::MtReadMru`.
#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <span>
#include <vector>

#include "buffers/buffer.h"
#include "buffers/direct_buffers.h"
#include "common/types.h"
#include "render/dynamic_state.h"
#include "render/gfx_push_state_tracker.h"
#include "render/pipeline_cache.h"
#include "render/vk_obj_vel_cull.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Rasterizer;
class Scheduler;

/// Binary-compatible C/C++ pass inheritance descriptor for secondary Dynamic Rendering buffers.
struct MtPassInheritanceDesc {
    u32 num_color_attachments{0};
    std::array<vk::Format, 8> color_formats{};
    vk::Format depth_format{vk::Format::eUndefined};
    vk::Format stencil_format{vk::Format::eUndefined};
    vk::SampleCountFlagBits rasterization_samples{vk::SampleCountFlagBits::e1};
    u32 view_mask{0};
    vk::Rect2D render_area{};
};

struct MtConstantAllocation {
    vk::Buffer buffer{};
    u64 offset{0};
    u8* mapped{nullptr};

    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<VkBuffer>(buffer) != VK_NULL_HANDLE && mapped != nullptr;
    }
};

/// Metadata published by `GXWorker` when sealing a secondary `vk::CommandBuffer` and consumed on
/// `Stage B` inside `Rasterizer::ExecuteSecondaryCmdBuf` (`+0 B` `PT_TLS`, compact ~6.8 KiB/slot).
struct alignas(64) MtCompletedPassMeta {
    struct SampledRtRef {
        VideoCore::ImageId id{};
        u64 uid{0};
        VideoCore::SubresourceRange range{};
    };
    static constexpr u32 kMaxSampledRts = 16;

    std::atomic<u64> published_seq{0};
    std::atomic<u64> consumed_seq{0};
    vk::CommandBuffer sec_cmd{};
    const GraphicsPipeline* first_pso{nullptr};
    std::array<u32, 0x0180> sh_gfx_regs{};     // 0x2C00..0x2D7F (ps/vs/gs/es/hs/ls)
    std::array<u32, 0x0400> context_regs{};    // 0xA000..0xA3FF (all context regs)
    std::array<u32, 0x0008> uconfig_so_regs{}; // 0xC038..0xC03F (cp_strmout_cntl)
    std::array<u32, 0x0018> uconfig_vgt_regs{};// 0xC240..0xC257 (primitive_type, num_indices, ...)
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent{};
    Gcn::CbDbExtent db_extent{};
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> cb_uid{};
    u16 db_uid{0};
    u16 occ_block_id{0xFFFFu};
    u32 recorded_draws{0};
    bool is_gbuffer{false};
    bool has_camera_cb{false};
    u8 num_sampled_rt_refs{0};
    u8 motion_slot{0xFFu};
    std::array<SampledRtRef, kMaxSampledRts> sampled_rt_refs{};
    std::array<float, 216> camera_cb_floats{};

    // Final register/extent/UID delta of a compiled deferred command list, synchronized to
    // Stage A (`liverpool->regs`) and `DrawPreparation` (`bb:DrawScan` / `bb:DrawPrep`) when
    // `DirectVk::kExecSecondaryFlagSyncRegs` is set on the list's final `ExecuteSecondaryCmds`.
    bool has_end_sync{false};
    bool end_reset{false};
    bool end_db_written{false};
    u8 end_cb_written_mask{0};
    u64 end_stage_b_mask[2]{};
    std::bitset<Gcn::RegDirty::NumBlocks> end_dirty_blocks{};
    std::vector<u32> end_span_headers{}; // (word_count << 16) | word_offset
    std::vector<u32> end_span_values{};
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> end_cb_extent{};
    Gcn::CbDbExtent end_db_extent{};
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> end_cb_uid{};
    u16 end_db_uid{0};

    void ApplyEndSyncToRegs(Gcn::Regs& dst, Gcn::RegDirty* dirty, bool track_all) const noexcept {
        if (!has_end_sync) {
            return;
        }
        if (end_reset) {
            dst.SetDefaults();
            if (dirty) {
                dirty->Clear();
                dirty->reset = true;
            }
        }
        size_t val_cursor = 0;
        for (const u32 hdr : end_span_headers) {
            const u32 word_off = hdr & 0xFFFFu;
            const u32 count = hdr >> 16u;
            if (count != 0u && word_off + count <= Gcn::Regs::NumRegs &&
                val_cursor + count <= end_span_values.size()) {
                std::memcpy(&dst.reg_array[word_off], &end_span_values[val_cursor],
                            count * sizeof(u32));
                if (dirty && dirty->exact_words) [[unlikely]] {
                    for (u32 w = word_off; w < word_off + count; ++w) {
                        dirty->exact_words->set(w);
                    }
                }
            }
            val_cursor += count;
        }
        if (dirty) {
            dirty->stage_b_mask[0] |= end_stage_b_mask[0];
            dirty->stage_b_mask[1] |= end_stage_b_mask[1];
            if (dirty->track_all_blocks || track_all) {
                dirty->blocks |= end_dirty_blocks;
            }
        }
    }

    void CaptureFromRegs(const Gcn::Regs& src) noexcept {
        std::memcpy(sh_gfx_regs.data(), &src.reg_array[0x2C00],
                    sh_gfx_regs.size() * sizeof(u32));
        std::memcpy(context_regs.data(), &src.reg_array[0xA000],
                    context_regs.size() * sizeof(u32));
        std::memcpy(uconfig_so_regs.data(), &src.reg_array[0xC038],
                    uconfig_so_regs.size() * sizeof(u32));
        std::memcpy(uconfig_vgt_regs.data(), &src.reg_array[0xC240],
                    uconfig_vgt_regs.size() * sizeof(u32));
    }

    void ApplyToRegs(Gcn::Regs& dst) const noexcept {
        std::memcpy(&dst.reg_array[0x2C00], sh_gfx_regs.data(),
                    sh_gfx_regs.size() * sizeof(u32));
        std::memcpy(&dst.reg_array[0xA000], context_regs.data(),
                    context_regs.size() * sizeof(u32));
        std::memcpy(&dst.reg_array[0xC038], uconfig_so_regs.data(),
                    uconfig_so_regs.size() * sizeof(u32));
        std::memcpy(&dst.reg_array[0xC240], uconfig_vgt_regs.data(),
                    uconfig_vgt_regs.size() * sizeof(u32));
    }

    void MarkConsumed(u64 seq) noexcept {
        consumed_seq.store(seq, std::memory_order_release);
    }
};

class VkMtDirectRecorder {
public:
    static constexpr u32 kMaxWorkers = 16;
    static constexpr u32 kFrameSlices = 4;
    static constexpr u32 kCompletedMetaSlots = 1024;
    static constexpr u64 kWorkerConstantRingBytes = 4ULL << 20; // 4 MiB per worker

    VkMtDirectRecorder(const Instance& instance, Scheduler& scheduler, Rasterizer& rasterizer);
    ~VkMtDirectRecorder();

    VkMtDirectRecorder(const VkMtDirectRecorder&) = delete;
    VkMtDirectRecorder& operator=(const VkMtDirectRecorder&) = delete;

    /// Runtime mode controlled by `BB_MT_DIRECT_VK` (`0` = off, `1` = scene passes, `2` = all).
    [[nodiscard]] static int ConfiguredMode() noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept {
        return mode > 0;
    }
    [[nodiscard]] int Mode() const noexcept {
        return mode;
    }

    /// Primary `vk::RenderingInfo::flags` required when executing secondary command buffers inside
    /// a Vulkan 1.3 Dynamic Rendering pass.
    [[nodiscard]] static constexpr vk::RenderingFlags SecondaryRenderingFlags() noexcept {
        return vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;
    }

    /// Lazily initializes the heap-allocated `WorkerSlot` for `worker_id` (`0..7`).
    /// Safe to call concurrently for distinct `worker_id`s.
    bool EnsureWorkerInitialized(u32 worker_id);

    /// Opens a fresh secondary `vk::CommandBuffer` on `worker_id` with
    /// `eRenderPassContinue | eOneTimeSubmit` and `VkCommandBufferInheritanceRenderingInfo`.
    bool BeginWorkerPass(u32 worker_id, const MtPassInheritanceDesc& pass_desc);

    /// Closes the active secondary `vk::CommandBuffer` on `worker_id`, flushes worker-local
    /// non-atomic stats once to `BbStats`, and returns its handle (plus optional `out_cmd_seq`).
    [[nodiscard]] vk::CommandBuffer EndWorkerPass(u32 worker_id, u64* out_cmd_seq = nullptr);

    /// Discards an in-progress secondary `vk::CommandBuffer` on `worker_id` without flushing
    /// speculative local stats (used when a chunk seamlessly falls back to Stage A/B).
    void AbortWorkerPass(u32 worker_id) noexcept;

    /// Maps the calling OS thread (`pthread_self` / `GetCurrentThreadId`) to a dedicated
    /// `worker_id` (`0..kMaxWorkers-1`) without any `thread_local` (`+0 B` `PT_TLS`).
    [[nodiscard]] u32 AcquireThreadWorkerId() noexcept;

    /// Returns the active secondary `vk::CommandBuffer` currently open on `worker_id` (or null).
    [[nodiscard]] vk::CommandBuffer ActiveCommandBuffer(u32 worker_id) const noexcept;

    /// Lock-free graphics PSO selection & `vkCmdBindPipeline` on `worker_id`.
    [[nodiscard]] const GraphicsPipeline* SelectAndBindPipeline(
        u32 worker_id, const Gcn::Regs& regs, DrawIndirectParams params = {},
        bool indirect = false, bool allow_compile_fallback = false);

    /// Accesses the worker's thread-private `MtWorkerPsoState` (`bound_stage_infos`, `vsharps`).
    [[nodiscard]] MtWorkerPsoState* WorkerPsoState(u32 worker_id) noexcept;

    /// Sub-allocates `byte_size` bytes aligned to `alignment` in `worker_id`'s private host-mapped
    /// constant ring (copying `initial_src` and flushing non-coherent memory when non-null).
    [[nodiscard]] MtConstantAllocation AllocateWorkerConstants(
        u32 worker_id, u64 byte_size, u64 alignment = 256, const void* initial_src = nullptr);

    /// Lock-free read-only buffer, SRV `VkImageView`, and `VkSampler` resolvers for `worker_id`.
    [[nodiscard]] VideoCore::DirectBuffers::MtResolvedBuffer ResolveBuffer(
        u32 worker_id, VAddr addr, u64 requested_size) noexcept;
    [[nodiscard]] vk::ImageView ResolveSrv(u32 worker_id, const Gcn::Image& raw_sharp,
                                           const Shader::ImageResource& res);
    [[nodiscard]] Rasterizer::MtResolvedSrv ResolveSceneSrv(
        u32 worker_id, const Gcn::Image& raw_sharp, const Shader::ImageResource& res,
        const Rasterizer::MtPassTargetSummary& pass_targets);
    [[nodiscard]] vk::Sampler ResolveSampler(u32 worker_id, const Gcn::Sampler& raw_ssharp,
                                             bool is_depth, float lod_bias,
                                             Gcn::BorderColorBuffer bc_base);

    /// Overload preserving Substep 4.1 signature (`worker_id = 0` when unspecified).
    [[nodiscard]] vk::ImageView ResolveSrv(const Gcn::Image& raw_sharp,
                                           const Shader::ImageResource& res) {
        return ResolveSrv(0u, raw_sharp, res);
    }
    [[nodiscard]] vk::Sampler ResolveSampler(const Gcn::Sampler& raw_ssharp, bool is_depth,
                                             float lod_bias, Gcn::BorderColorBuffer bc_base) {
        return ResolveSampler(0u, raw_ssharp, is_depth, lod_bias, bc_base);
    }

    /// Emits single-pass delta `vkCmdPushConstants` + `vkCmdPushDescriptorSetKHR` directly into
    /// `worker_id`'s active secondary `vk::CommandBuffer` using its private `GfxPushStateTracker`.
    void PushGraphicsResources(u32 worker_id, vk::PipelineLayout layout,
                               bool uses_push_descriptors, const Shader::PushData& push_data,
                               std::span<const vk::WriteDescriptorSet> set_writes);

    /// Stage B: executes finished worker secondary command buffers inside the open primary
    /// Dynamic Rendering pass (`eContentsSecondaryCommandBuffers`), stamps the exact `(worker_id,
    /// cmd_seq)` slot in the lock-free `StageBStampRing`, and invalidates primary push state.
    void ExecuteSecondaryBuffers(std::span<const vk::CommandBuffer> secondaries,
                                 u32 worker_id = ~0u, u64 cmd_seq = 0,
                                 u16 occ_query_block_id = 0xFFFFu);

    /// Stage B: stamps `(worker_id, cmd_seq)` and releases `occ_query_block_id` when all draws in
    /// a compiled worker chunk were culled (`recorded_draws == 0`) without opening a render pass.
    void StampEmptyPassOnStageB(u32 worker_id, u64 cmd_seq, u16 occ_query_block_id) noexcept;

    /// Stage B: looks up the `MtCompletedPassMeta` slot published for `(worker_id, cmd_seq)`.
    [[nodiscard]] MtCompletedPassMeta* LookupCompletedPass(u32 worker_id, u64 cmd_seq) noexcept;

    /// GXWorker (`GXGnmCommandContext_BeginCommandList` when `*(ctx_base + 0xB510) != 0`):
    /// compiles an eligible deferred `cmdListImpl` (`G-Buffer` / `CSM & Shadows` / `Object-Velocity`
    /// in Mode 1, or any eligible pass in Mode 2) directly into a secondary `vk::CommandBuffer`
    /// and rewrites `cmdListImpl` in-place to `[CopyBufferBatch...] + [Op::ExecuteSecondaryCmds] +
    /// [Timestamp/Labels]`, leaving `m_beginptr` untouched for `labelPool` recycling.
    /// Returns `false` with zero side effects on `cmdListImpl` when ineligible or falling back.
    [[nodiscard]] bool TryCompileDeferredCommandList(const void* ctx_base, void* cmd_list_impl);

    /// Guest immediate-context hooks (`EmitCopyBufferBatch` / `EmitFillBuffer` when
    /// `*(dcb_raw + 0xB500) == 0`) maintaining exact guest-timeline CPU constant shadows for
    /// `GXWorker:1..5` without racing Stage A/B's delayed packet stream.
    void NoteGuestImmediateCopyBatch(const void* raw_regions, u32 count) noexcept;
    void InvalidateGuestCopyShadow(VAddr dst_va, u64 byte_size) noexcept;
    [[nodiscard]] bool CopyFromGuestShadow(VAddr addr, u32 min_bytes, u32 size_hint, void* dst,
                                           u32 dst_cap, u32& out_copied, u64& out_gen) noexcept;
    [[nodiscard]] bool WasCoarseRangeModifiedRecently(VAddr addr, u64 byte_size,
                                                      u32 cur_guest_frame) const noexcept;

    [[nodiscard]] static VkMtDirectRecorder* ActiveInstance() noexcept {
        return active_instance.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t kStampSlots = 1024;

    struct StageBStampSlot {
        std::atomic<u64> seq{0};
        std::atomic<u64> tick{0};
    };

    struct alignas(64) StageBStampTable {
        std::array<StageBStampSlot, kStampSlots> slots{};
    };

    struct WorkerLocalStats {
        u64 native_buf_hits{0};
        u64 native_buf_misses{0};
        u64 native_tex_hits{0};
        u64 native_tex_misses{0};
        u64 desc_push_total{0};
        u64 desc_push_saved{0};
        u64 occ_candidates{0};
        u64 occ_skipped{0};
        u64 occ_probed{0};
        u64 shadow_skipped{0};
        u64 vel_seen{0};
        u64 vel_culled{0};

        void Reset() noexcept {
            *this = WorkerLocalStats{};
        }
        void FlushToGlobal(Rasterizer& rasterizer) noexcept;
    };

    struct WorkerConstantRing {
        VideoCore::Buffer buffer;
        u64 position{0};
        u64 retired{0};

        WorkerConstantRing(const Instance& instance, u32 worker_id);

        [[nodiscard]] MtConstantAllocation Allocate(u64 size, u64 alignment, const void* src);
    };

    struct FrameCmdPool {
        vk::UniqueCommandPool pool;
        std::vector<vk::CommandBuffer> secondary_buffers;
        u32 next_buffer{0};
        u64 guest_frame{~0ull};
        u64 last_cmd_seq{0};
        u64 end_ring_pos{0};
        bool dirty_since_reset{false};
    };

    struct WorkerCompileScratch;

    struct alignas(64) WorkerSlot {
        u32 worker_id{0};
        u32 cur_slice{0};
        std::atomic<bool> compile_busy{false};
        std::vector<FrameCmdPool> pools;
        std::unique_ptr<WorkerConstantRing> constant_ring;
        std::unique_ptr<StageBStampTable> stamp_table;
        std::unique_ptr<MtCompletedPassMeta[]> completed_meta;
        std::unique_ptr<Gcn::Regs> scan_regs;
        std::unique_ptr<Gcn::Regs> replay_regs;
        std::unique_ptr<WorkerCompileScratch> compile_scratch;
        DynamicState dyn_state{};
        GfxPushStateTracker push_tracker{};
        MtWorkerPsoState pso_state{};
        VideoCore::DirectBuffers::MtReadMru buffer_mru{};
        std::array<ObjectVelocityCuller::PairMemoEntry, ObjectVelocityCuller::kPairMemoSize>
            vel_pair_memo{};
        WorkerLocalStats local_stats{};

        vk::CommandBuffer active_cmdbuf{};
        u32 active_pool_idx{0};
        u64 active_ring_start{0};
        u64 cmd_seq_gen{0};
        u16 active_occ_block_id{0xFFFFu};
        bool is_recording{false};
        u32 recorded_draws{0};
        u64 desc_epoch_counter{1};
        vk::Pipeline bound_pipeline{};

        void ReclaimCompletedPools(const Instance& instance, Scheduler& scheduler);
        vk::CommandBuffer AcquireSecondaryBuffer(const Instance& instance, Scheduler& scheduler);
    };

    class MtGuestCopyShadowCache;

    const Instance& instance;
    Scheduler& scheduler;
    Rasterizer& rasterizer;
    int mode{0};

    std::unique_ptr<MtGuestCopyShadowCache> guest_copy_shadow;
    std::array<std::atomic<u64>, kMaxWorkers> slot_thread_keys{};
    std::array<std::atomic<WorkerSlot*>, kMaxWorkers> worker_ptrs{};
    std::array<std::unique_ptr<WorkerSlot>, kMaxWorkers> workers_owned{};
    std::mutex worker_init_mutex;

    static inline std::atomic<VkMtDirectRecorder*> active_instance{nullptr};
};

} // namespace Vulkan
