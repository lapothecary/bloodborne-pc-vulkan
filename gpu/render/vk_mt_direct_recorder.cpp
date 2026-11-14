// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_mt_direct_recorder.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fmt/format.h>

#include "common/assert.h"
#include "render/pipeline.h"
#include "render/rasterizer.h"
#include "render/scheduler.h"
#include "shim/bbport_toggles.h"
#include "vulkan/instance.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

namespace Vulkan {

namespace {
constexpr u32 kSecondaryGrowStep = 8;
} // namespace

int VkMtDirectRecorder::ConfiguredMode() noexcept {
    static const int cached_mode = []() -> int {
        const char* env = std::getenv("BB_MT_DIRECT_VK");
        if (!env || env[0] == '\0') {
            return 0;
        }
        return std::clamp<int>(static_cast<int>(std::strtol(env, nullptr, 10)), 0, 2);
    }();
    return cached_mode;
}

VkMtDirectRecorder::WorkerConstantRing::WorkerConstantRing(const Instance& instance_, u32 wid)
    : buffer{instance_, 0, kWorkerConstantRingBytes, VideoCore::MemoryType::Stream,
             fmt::format("MtDirectConstRing:W{}", wid)} {
    ASSERT(!buffer.mapped_data.empty());
}

void VkMtDirectRecorder::WorkerConstantRing::Retire(Scheduler& scheduler_) noexcept {
    while (head != tail && scheduler_.IsFree(stamps[head].tick)) {
        retired = stamps[head].end_pos;
        head = (head + 1u) % kMaxStamps;
    }
}

void VkMtDirectRecorder::WorkerConstantRing::Stamp(u64 tick) noexcept {
    if (position == stamped_pos) {
        return;
    }
    if (head != tail) {
        const size_t last = (tail + kMaxStamps - 1u) % kMaxStamps;
        if (stamps[last].tick == tick) {
            stamps[last].end_pos = position;
            stamped_pos = position;
            return;
        }
    }
    const size_t next_tail = (tail + 1u) % kMaxStamps;
    if (next_tail == head) {
        // Coalesce oldest stamp entry into next slot so ring accounting never drops progress.
        head = (head + 1u) % kMaxStamps;
    }
    stamps[tail] = StampEntry{.tick = tick, .end_pos = position};
    tail = next_tail;
    stamped_pos = position;
}

MtConstantAllocation VkMtDirectRecorder::WorkerConstantRing::Allocate(Scheduler& scheduler_,
                                                                      u64 size, u64 alignment,
                                                                      const void* src) {
    if (size == 0 || size > kWorkerConstantRingBytes) {
        return {};
    }
    alignment = std::max<u64>(16ULL, alignment);
    u64 at = (position + alignment - 1ULL) & ~(alignment - 1ULL);
    if ((at % kWorkerConstantRingBytes) + size > kWorkerConstantRingBytes) {
        at += kWorkerConstantRingBytes - (at % kWorkerConstantRingBytes);
    }
    const auto in_use = [&]() noexcept {
        return (at + size > kWorkerConstantRingBytes) &&
               (at + size - kWorkerConstantRingBytes > retired);
    };
    if (in_use()) {
        Retire(scheduler_);
        if (in_use()) {
            return {};
        }
    }
    position = at + size;
    const u64 ring_off = at % kWorkerConstantRingBytes;
    u8* dst = buffer.mapped_data.data() + ring_off;
    if (src) {
        std::memcpy(dst, src, size);
        if (!buffer.is_coherent) {
            buffer.Flush(ring_off, size);
        }
    }
    return MtConstantAllocation{
        .buffer = buffer.Handle(),
        .offset = ring_off,
        .mapped = dst,
    };
}

vk::CommandBuffer VkMtDirectRecorder::WorkerSlot::AcquireSecondaryBuffer(
    const Instance& instance_, Scheduler& scheduler_) {
    const vk::Device device = instance_.GetDevice();
    const u32 pool_count = static_cast<u32>(pools.size());

    // Search for a current or completed FrameCmdPool slice without stalling on GPU fences.
    u32 chosen = ~0u;
    for (u32 probe = 0; probe < pool_count; ++probe) {
        const u32 idx = (cur_slice + probe) % pool_count;
        auto& fp = pools[idx];
        if (!fp.dirty_since_reset) {
            chosen = idx;
            break;
        }
        if (fp.fence_tick != 0 && scheduler_.IsFree(fp.fence_tick)) {
            const vk::Result rst = device.resetCommandPool(*fp.pool);
            ASSERT_MSG(rst == vk::Result::eSuccess, "resetCommandPool failed: {}",
                       vk::to_string(rst));
            fp.next_buffer = 0;
            fp.fence_tick = 0;
            fp.dirty_since_reset = false;
            chosen = idx;
            break;
        }
        if (fp.fence_tick == 0) {
            // Still accumulating multiple passes within the same unsubmitted tick: keep appending
            // secondary command buffers from this pool without resetting earlier ones!
            chosen = idx;
            break;
        }
    }

    if (chosen == ~0u) {
        // All preallocated slices are currently in flight on the GPU: append an overflow pool so
        // the worker never blocks on `vkWaitSemaphores` mid-frame.
        FrameCmdPool overflow{};
        auto [res, created] = device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = instance_.GetGraphicsQueueFamilyIndex(),
        });
        ASSERT_MSG(res == vk::Result::eSuccess, "MtDirect CommandPool: {}", vk::to_string(res));
        overflow.pool = std::move(created);
        SetObjectName(device, *overflow.pool, "MtDirectCmdPool:W{}:S{}", worker_id, pool_count);
        pools.push_back(std::move(overflow));
        chosen = pool_count;
    }

    cur_slice = chosen;
    active_pool_idx = chosen;
    auto& fp = pools[chosen];
    if (fp.next_buffer >= fp.secondary_buffers.size()) {
        std::array<vk::CommandBuffer, kSecondaryGrowStep> fresh{};
        const vk::CommandBufferAllocateInfo alloc_info{
            .commandPool = *fp.pool,
            .level = vk::CommandBufferLevel::eSecondary,
            .commandBufferCount = kSecondaryGrowStep,
        };
        const vk::Result alloc_res = device.allocateCommandBuffers(&alloc_info, fresh.data());
        ASSERT_MSG(alloc_res == vk::Result::eSuccess, "MtDirect SecondaryCmdBuf: {}",
                   vk::to_string(alloc_res));
        const size_t base_idx = fp.secondary_buffers.size();
        fp.secondary_buffers.insert(fp.secondary_buffers.end(), fresh.begin(), fresh.end());
        for (size_t k = 0; k < kSecondaryGrowStep; ++k) {
            SetObjectName(device, fresh[k], "MtDirectSecCmd:W{}:S{}:B{}", worker_id, chosen,
                          base_idx + k);
        }
    }

    fp.dirty_since_reset = true;
    return fp.secondary_buffers[fp.next_buffer++];
}

VkMtDirectRecorder::VkMtDirectRecorder(const Instance& instance_, Scheduler& scheduler_,
                                       Rasterizer& rasterizer_)
    : instance{instance_}, scheduler{scheduler_}, rasterizer{rasterizer_},
      mode{ConfiguredMode()} {
    if (mode > 0) {
        for (u32 w = 0; w < 6u; ++w) {
            EnsureWorkerInitialized(w);
        }
        std::printf("[bb_vulkan] Phase 4.1 VkMtDirectRecorder active (BB_MT_DIRECT_VK=%d, "
                    "prewarmed 6/%u heap worker slots x %u pools, +0 B PT_TLS)\n",
                    mode, kMaxWorkers, kFrameSlices);
    }
    active_instance.store(this, std::memory_order_release);
}

VkMtDirectRecorder::~VkMtDirectRecorder() {
    VkMtDirectRecorder* expected = this;
    active_instance.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

bool VkMtDirectRecorder::EnsureWorkerInitialized(u32 worker_id) {
    if (worker_id >= kMaxWorkers) {
        return false;
    }
    if (worker_ptrs[worker_id].load(std::memory_order_acquire) != nullptr) [[likely]] {
        return true;
    }
    std::scoped_lock lk{worker_init_mutex};
    if (worker_ptrs[worker_id].load(std::memory_order_relaxed) != nullptr) {
        return true;
    }

    const vk::Device device = instance.GetDevice();
    const u32 qf = instance.GetGraphicsQueueFamilyIndex();
    auto slot = std::make_unique<WorkerSlot>();
    slot->worker_id = worker_id;
    slot->pools.resize(kFrameSlices);
    for (u32 s = 0; s < kFrameSlices; ++s) {
        auto [res, created] = device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = qf,
        });
        ASSERT_MSG(res == vk::Result::eSuccess, "MtDirect CommandPool init: {}",
                   vk::to_string(res));
        slot->pools[s].pool = std::move(created);
        SetObjectName(device, *slot->pools[s].pool, "MtDirectCmdPool:W{}:S{}", worker_id, s);
    }
    slot->constant_ring = std::make_unique<WorkerConstantRing>(instance, worker_id);

    WorkerSlot* raw = slot.get();
    workers_owned[worker_id] = std::move(slot);
    worker_ptrs[worker_id].store(raw, std::memory_order_release);
    return true;
}

bool VkMtDirectRecorder::BeginWorkerPass(u32 worker_id, const MtPassInheritanceDesc& pass_desc) {
    if (worker_id >= kMaxWorkers || !EnsureWorkerInitialized(worker_id)) {
        return false;
    }
    auto& w = *worker_ptrs[worker_id].load(std::memory_order_relaxed);
    if (w.is_recording) {
        (void)EndWorkerPass(worker_id);
    }

    const vk::CommandBuffer cmdbuf = w.AcquireSecondaryBuffer(instance, scheduler);
    const u32 color_cnt =
        std::min<u32>(pass_desc.num_color_attachments,
                      static_cast<u32>(pass_desc.color_formats.size()));

    const vk::CommandBufferInheritanceRenderingInfo rendering_inh{
        .viewMask = pass_desc.view_mask,
        .colorAttachmentCount = color_cnt,
        .pColorAttachmentFormats = color_cnt ? pass_desc.color_formats.data() : nullptr,
        .depthAttachmentFormat = pass_desc.depth_format,
        .stencilAttachmentFormat = pass_desc.stencil_format,
        .rasterizationSamples = pass_desc.rasterization_samples,
    };
    const vk::CommandBufferInheritanceInfo inh_info{
        .pNext = &rendering_inh,
    };
    const vk::CommandBufferBeginInfo begin_info{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue |
                 vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        .pInheritanceInfo = &inh_info,
    };
    const vk::Result res = cmdbuf.begin(&begin_info);
    if (res != vk::Result::eSuccess) {
        return false;
    }

    w.active_cmdbuf = cmdbuf;
    w.is_recording = true;
    w.recorded_draws = 0;
    ++w.desc_epoch_counter;
    w.bound_pipeline = vk::Pipeline{};
    w.push_tracker.InvalidateAll();
    w.pso_state.ResetForPass();

    if (pass_desc.render_area.extent.width > 0 && pass_desc.render_area.extent.height > 0) {
        const vk::Viewport vp{
            .x = static_cast<float>(pass_desc.render_area.offset.x),
            .y = static_cast<float>(pass_desc.render_area.offset.y),
            .width = static_cast<float>(pass_desc.render_area.extent.width),
            .height = static_cast<float>(pass_desc.render_area.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        cmdbuf.setViewportWithCount(vp);
        cmdbuf.setScissorWithCount(pass_desc.render_area);
    }
    return true;
}

vk::CommandBuffer VkMtDirectRecorder::EndWorkerPass(u32 worker_id) {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return {};
    }
    const vk::CommandBuffer cmdbuf = w->active_cmdbuf;
    (void)cmdbuf.end();
    w->active_cmdbuf = vk::CommandBuffer{};
    w->is_recording = false;
    w->bound_pipeline = vk::Pipeline{};
    w->push_tracker.InvalidateAll();
    return cmdbuf;
}

vk::CommandBuffer VkMtDirectRecorder::ActiveCommandBuffer(u32 worker_id) const noexcept {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    const auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    return (w && w->is_recording) ? w->active_cmdbuf : vk::CommandBuffer{};
}

const GraphicsPipeline* VkMtDirectRecorder::SelectAndBindPipeline(u32 worker_id,
                                                                  const Gcn::Regs& regs,
                                                                  DrawIndirectParams params,
                                                                  bool indirect,
                                                                  bool allow_compile_fallback) {
    if (worker_id >= kMaxWorkers) {
        return nullptr;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return nullptr;
    }
    const GraphicsPipeline* pso = rasterizer.GetPipelineCache().SelectGraphicsPipelineMt(
        regs, w->pso_state, params, indirect, allow_compile_fallback);
    if (!pso) {
        return nullptr;
    }
    const vk::Pipeline vk_pso = pso->Handle();
    if (w->bound_pipeline != vk_pso) {
        w->active_cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, vk_pso);
        w->bound_pipeline = vk_pso;
    }
    return pso;
}

MtWorkerPsoState* VkMtDirectRecorder::WorkerPsoState(u32 worker_id) noexcept {
    if (worker_id >= kMaxWorkers) {
        return nullptr;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    return w ? &w->pso_state : nullptr;
}

MtConstantAllocation VkMtDirectRecorder::AllocateWorkerConstants(u32 worker_id, u64 byte_size,
                                                                 u64 alignment,
                                                                 const void* initial_src) {
    if (worker_id >= kMaxWorkers || !EnsureWorkerInitialized(worker_id)) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_relaxed);
    if (!w || !w->constant_ring) {
        return {};
    }
    return w->constant_ring->Allocate(scheduler, byte_size, alignment, initial_src);
}

VideoCore::DirectBuffers::MtResolvedBuffer VkMtDirectRecorder::ResolveBuffer(
    u32 worker_id, VAddr addr, u64 requested_size) noexcept {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w) {
        return {};
    }
    return rasterizer.GetDirectBuffers().TryResolveReadMt(addr, requested_size, w->buffer_mru);
}

vk::ImageView VkMtDirectRecorder::ResolveSrv(const Gcn::Image& raw_sharp,
                                             const Shader::ImageResource& res) {
    return rasterizer.ResolveSrvViewMt(raw_sharp, res);
}

vk::Sampler VkMtDirectRecorder::ResolveSampler(const Gcn::Sampler& raw_ssharp, bool is_depth,
                                               float lod_bias, Gcn::BorderColorBuffer bc_base) {
    return rasterizer.ResolveSamplerMt(raw_ssharp, is_depth, lod_bias, bc_base);
}

void VkMtDirectRecorder::PushGraphicsResources(
    u32 worker_id, vk::PipelineLayout layout, bool uses_push_descriptors,
    const Shader::PushData& push_data, std::span<const vk::WriteDescriptorSet> set_writes) {
    if (worker_id >= kMaxWorkers) {
        return;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return;
    }

    auto& tracker = w->push_tracker;
    tracker.OnDraw(&scheduler, w->desc_epoch_counter, w->desc_epoch_counter, layout);

    const bool need_push_constants =
        !tracker.has_push_data ||
        (std::memcmp(&tracker.push_data, &push_data, sizeof(push_data)) != 0);
    if (need_push_constants) {
        tracker.push_data = push_data;
        tracker.has_push_data = true;
        w->active_cmdbuf.pushConstants(layout, AllGraphicsStageBits, 0u, sizeof(push_data),
                                       &push_data);
    }

    if (!uses_push_descriptors || set_writes.empty()) {
        return;
    }

    std::array<vk::WriteDescriptorSet, kMaxTrackedBindings> delta_writes;
    u32 delta_cnt = 0;
    u64 total_slots = 0;
    u64 saved_slots = 0;
    bool fallback_all = false;

    for (const auto& write : set_writes) {
        total_slots += write.descriptorCount;
        const bool single_img = (write.pImageInfo != nullptr && !write.pBufferInfo);
        const bool single_buf = (write.pBufferInfo != nullptr && !write.pImageInfo);
        if (write.descriptorCount == 1 && write.dstArrayElement == 0 &&
            !write.pTexelBufferView && write.dstBinding < kMaxTrackedBindings &&
            (single_img || single_buf)) [[likely]] {
            const u64 slot_bit = 1ULL << write.dstBinding;
            const bool slot_valid = (tracker.valid_mask & slot_bit) != 0;
            auto& slot = tracker.slots[write.dstBinding];
            if (single_img) {
                const auto& img = *write.pImageInfo;
                const VkSampler s_h = static_cast<VkSampler>(img.sampler);
                const VkImageView v_h = static_cast<VkImageView>(img.imageView);
                if (slot_valid && slot.is_image && slot.type == write.descriptorType &&
                    slot.sampler == s_h && slot.image_view == v_h &&
                    slot.image_layout == img.imageLayout) {
                    ++saved_slots;
                    continue;
                }
                tracker.valid_mask |= slot_bit;
                slot.is_image = true;
                slot.type = write.descriptorType;
                slot.sampler = s_h;
                slot.image_view = v_h;
                slot.image_layout = img.imageLayout;
            } else {
                const auto& buf = *write.pBufferInfo;
                const VkBuffer b_h = static_cast<VkBuffer>(buf.buffer);
                if (slot_valid && !slot.is_image && slot.type == write.descriptorType &&
                    slot.buffer == b_h && slot.offset == buf.offset &&
                    slot.range == buf.range) {
                    ++saved_slots;
                    continue;
                }
                tracker.valid_mask |= slot_bit;
                slot.is_image = false;
                slot.type = write.descriptorType;
                slot.buffer = b_h;
                slot.offset = buf.offset;
                slot.range = buf.range;
            }
            if (delta_cnt < delta_writes.size()) {
                delta_writes[delta_cnt++] = write;
            } else {
                fallback_all = true;
                break;
            }
        } else {
            fallback_all = true;
            break;
        }
    }

    if (fallback_all) {
        tracker.valid_mask = 0;
        w->active_cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0,
                                              set_writes);
        BbStats::desc_push_total.fetch_add(total_slots, std::memory_order_relaxed);
        return;
    }

    if (total_slots != 0) {
        BbStats::desc_push_total.fetch_add(total_slots, std::memory_order_relaxed);
    }
    if (saved_slots != 0) {
        BbStats::desc_push_saved.fetch_add(saved_slots, std::memory_order_relaxed);
    }
    if (delta_cnt != 0) {
        w->active_cmdbuf.pushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, layout, 0,
            std::span<const vk::WriteDescriptorSet>{delta_writes.data(), delta_cnt});
    }
}

void VkMtDirectRecorder::ExecuteSecondaryBuffers(
    std::span<const vk::CommandBuffer> secondaries) {
    if (secondaries.empty()) {
        return;
    }

    const auto recorded = scheduler.RecordData(secondaries);
    scheduler.Record([recorded](vk::CommandBuffer primary) {
        primary.executeCommands(static_cast<u32>(recorded.size()), recorded.data());
    });

    // Stamp all touched worker pools and constant rings with the current GPU submission tick so
    // they recycle cleanly once the timeline semaphore passes `tick`.
    const u64 tick = scheduler.CurrentTick();
    for (u32 w = 0; w < kMaxWorkers; ++w) {
        auto* slot = worker_ptrs[w].load(std::memory_order_acquire);
        if (!slot) {
            continue;
        }
        if (slot->active_pool_idx < slot->pools.size() &&
            slot->pools[slot->active_pool_idx].dirty_since_reset) {
            slot->pools[slot->active_pool_idx].fence_tick = tick;
        }
        if (slot->constant_ring) {
            slot->constant_ring->Stamp(tick);
        }
    }

    // Secondary command buffers disturb primary command buffer state per Vulkan 1.3 §6.8:
    // bump primary descriptor/push-constant epochs so subsequent Stage B draws rebind cleanly.
    scheduler.BumpDescriptorStateEpoch();
}

} // namespace Vulkan

extern "C" {

int PS4_SYSV_ABI bb_direct_vk_mt_mode() noexcept {
    return Vulkan::VkMtDirectRecorder::ConfiguredMode();
}

int PS4_SYSV_ABI bb_direct_vk_mt_worker_begin_pass(uint32_t worker_id,
                                                   const void* pass_desc_raw) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled() || !pass_desc_raw) {
        return 0;
    }
    const auto* desc = static_cast<const Vulkan::MtPassInheritanceDesc*>(pass_desc_raw);
    return rec->BeginWorkerPass(worker_id, *desc) ? 1 : 0;
}

uint64_t PS4_SYSV_ABI bb_direct_vk_mt_worker_end_pass(uint32_t worker_id) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled()) {
        return 0ULL;
    }
    const vk::CommandBuffer cmd = rec->EndWorkerPass(worker_id);
    return reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(cmd));
}

void PS4_SYSV_ABI bb_direct_vk_mt_execute_secondaries(const uint64_t* cmd_handles,
                                                      uint32_t count) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled() || !cmd_handles || count == 0u) {
        return;
    }
    std::array<vk::CommandBuffer, 16> stack_cmds;
    const u32 n = std::min<u32>(count, static_cast<u32>(stack_cmds.size()));
    u32 valid_n = 0;
    for (u32 i = 0; i < n; ++i) {
        if (cmd_handles[i] != 0ULL) {
            stack_cmds[valid_n++] =
                vk::CommandBuffer{reinterpret_cast<VkCommandBuffer>(cmd_handles[i])};
        }
    }
    if (valid_n != 0u) {
        rec->ExecuteSecondaryBuffers(std::span<const vk::CommandBuffer>{stack_cmds.data(), valid_n});
    }
}

} // extern "C"
