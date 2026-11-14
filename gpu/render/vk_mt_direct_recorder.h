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
#include <memory>
#include <span>
#include <vector>

#include "buffers/buffer.h"
#include "buffers/direct_buffers.h"
#include "common/types.h"
#include "render/gfx_push_state_tracker.h"
#include "render/pipeline_cache.h"
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

class VkMtDirectRecorder {
public:
    static constexpr u32 kMaxWorkers = 8;
    static constexpr u32 kFrameSlices = 4;
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

    /// Closes the active secondary `vk::CommandBuffer` on `worker_id` and returns its handle.
    [[nodiscard]] vk::CommandBuffer EndWorkerPass(u32 worker_id);

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
    [[nodiscard]] vk::ImageView ResolveSrv(const Gcn::Image& raw_sharp,
                                           const Shader::ImageResource& res);
    [[nodiscard]] vk::Sampler ResolveSampler(const Gcn::Sampler& raw_ssharp, bool is_depth,
                                             float lod_bias, Gcn::BorderColorBuffer bc_base);

    /// Emits single-pass delta `vkCmdPushConstants` + `vkCmdPushDescriptorSetKHR` directly into
    /// `worker_id`'s active secondary `vk::CommandBuffer` using its private `GfxPushStateTracker`.
    void PushGraphicsResources(u32 worker_id, vk::PipelineLayout layout,
                               bool uses_push_descriptors, const Shader::PushData& push_data,
                               std::span<const vk::WriteDescriptorSet> set_writes);

    /// Stage B: executes finished worker secondary command buffers inside the open primary
    /// Dynamic Rendering pass (`eContentsSecondaryCommandBuffers`), stamps worker command pools
    /// and constant rings with the current GPU submission tick, and invalidates primary push state.
    void ExecuteSecondaryBuffers(std::span<const vk::CommandBuffer> secondaries);

    [[nodiscard]] static VkMtDirectRecorder* ActiveInstance() noexcept {
        return active_instance.load(std::memory_order_acquire);
    }

private:
    struct WorkerConstantRing {
        static constexpr size_t kMaxStamps = 64;

        struct StampEntry {
            u64 tick{0};
            u64 end_pos{0};
        };

        VideoCore::Buffer buffer;
        u64 position{0};
        u64 retired{0};
        u64 stamped_pos{0};
        std::array<StampEntry, kMaxStamps> stamps{};
        size_t head{0};
        size_t tail{0};

        WorkerConstantRing(const Instance& instance, u32 worker_id);

        [[nodiscard]] MtConstantAllocation Allocate(Scheduler& scheduler, u64 size, u64 alignment,
                                                    const void* src);
        void Stamp(u64 tick) noexcept;
        void Retire(Scheduler& scheduler) noexcept;
    };

    struct FrameCmdPool {
        vk::UniqueCommandPool pool;
        std::vector<vk::CommandBuffer> secondary_buffers;
        u32 next_buffer{0};
        u64 fence_tick{0};
        bool dirty_since_reset{false};
    };

    struct alignas(64) WorkerSlot {
        u32 worker_id{0};
        u32 cur_slice{0};
        std::vector<FrameCmdPool> pools;
        std::unique_ptr<WorkerConstantRing> constant_ring;
        GfxPushStateTracker push_tracker{};
        MtWorkerPsoState pso_state{};
        VideoCore::DirectBuffers::MtReadMru buffer_mru{};

        vk::CommandBuffer active_cmdbuf{};
        u32 active_pool_idx{0};
        bool is_recording{false};
        u32 recorded_draws{0};
        u64 desc_epoch_counter{1};
        vk::Pipeline bound_pipeline{};

        vk::CommandBuffer AcquireSecondaryBuffer(const Instance& instance, Scheduler& scheduler);
    };

    const Instance& instance;
    Scheduler& scheduler;
    Rasterizer& rasterizer;
    int mode{0};

    std::array<std::atomic<WorkerSlot*>, kMaxWorkers> worker_ptrs{};
    std::array<std::unique_ptr<WorkerSlot>, kMaxWorkers> workers_owned{};
    std::mutex worker_init_mutex;

    static inline std::atomic<VkMtDirectRecorder*> active_instance{nullptr};
};

} // namespace Vulkan
