// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_object_motion.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vk_mem_alloc.h>
#include <xxhash.h>
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "shader/runtime_info.h"
#include "upscale/vk_temporal_upscaler.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

namespace Vulkan {

namespace {

bool CreateBuffer(const Instance& instance, vk::DeviceSize size, bool host, vk::Buffer& buffer,
                  VmaAllocation& allocation, void** mapped, u64& address) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    VmaAllocationCreateInfo alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (host) {
        alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else {
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    VkBuffer raw{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw, &allocation, &info) !=
        VK_SUCCESS) {
        return false;
    }
    buffer = raw;
    if (mapped) {
        *mapped = info.pMappedData;
    }
    address = instance.GetDevice().getBufferAddress({.buffer = buffer});
    return address != 0;
}

} // namespace

ObjectMotion::ObjectMotion(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    // Pipeline selection limits the extra attachment and vertex stores to likely
    // animated draws. A setting or environment override can still disable the path.
    if (!BbSettings::Get().object_motion ||
        BbSettings::Get().upscaler == BbSettings::UpscalerOff) {
        std::puts("Object motion: off (enable in menu or BB_OBJECT_MOTION=1)");
        return;
    }
    const auto& features = instance.GetPhysicalDevice().getFeatures();
    if (!features.vertexPipelineStoresAndAtomics) {
        std::printf("Object motion: vertexPipelineStoresAndAtomics unsupported, off\n");
        return;
    }
    void* mapped = nullptr;
    u64 params_address = 0, positions_address = 0;
    if (!CreateBuffer(instance, vk::DeviceSize(1 + FrameSlots * ParamsPerFrame) * 32, true,
                      params_buffer, params_allocation, &mapped, params_address) ||
        !CreateBuffer(instance, vk::DeviceSize(1 + 2 * PositionsPerFrame) * 16, false,
                      positions_buffer, positions_allocation, nullptr, positions_address)) {
        std::printf("Object motion: buffer creation failed, off\n");
        return;
    }
    params_mapped = static_cast<u32*>(mapped);
    std::memset(params_mapped, 0, 32);
    Shader::MotionVectors::params_address = params_address;
    Shader::MotionVectors::positions_address = positions_address;
    enabled = true;
    std::printf("Object motion: on (%u vertices per frame)\n", PositionsPerFrame);
}

ObjectMotion::~ObjectMotion() {
    scheduler.Finish();
    const auto allocator = instance.GetAllocator();
    if (params_buffer) {
        vmaDestroyBuffer(allocator, params_buffer, params_allocation);
    }
    if (positions_buffer) {
        vmaDestroyBuffer(allocator, positions_buffer, positions_allocation);
    }
}

void ObjectMotion::AdvanceCpuHistoryToFlipLocked(u32 flip_epoch) {
    if (cpu_flip_epoch == flip_epoch) {
        return;
    }
    if (BbStats::enabled && (frame % 600) == 0 && history.stats.draws) {
        const auto& s = history.stats;
        std::printf("Object motion: %llu draws, %llu stored, %llu with history, %llu unmatched, "
                    "%llu capacity skips, %llu invalid, %llu still (600 frames); last frame %u/%u vertices; "
                    "index ranges %llu reused, %llu scanned (%llu found changed), %zu cached\n",
                    (unsigned long long)s.draws, (unsigned long long)s.stored,
                    (unsigned long long)s.loaded, (unsigned long long)s.unmatched,
                    (unsigned long long)s.exhausted, (unsigned long long)s.invalid,
                    (unsigned long long)s.still,
                    history.Used(), PositionsPerFrame, (unsigned long long)index_ranges.stats.hits,
                    (unsigned long long)index_ranges.stats.scans,
                    (unsigned long long)index_ranges.stats.stale, index_ranges.Size());
        index_ranges.stats = {};
        history.stats = {};
    }
    history.NextFrame();
    ++frame;
    if (frame % Motion::IndexRangeCache::Unused == 0) {
        index_ranges.Trim(frame);
    }
    const u32 slot = u32(flip_epoch % FrameSlots);
    slot_params_used[slot] = 0;
    cpu_flip_epoch = flip_epoch;
}

void ObjectMotion::OnFrameStart() {
    if (!enabled) {
        return;
    }
    const u32 next_stage_b_flip = TemporalUpscaler::StageBFlipsRetiredForMt() + 1u;
    const u32 slot = u32(next_stage_b_flip % FrameSlots);
    u64 wait_tick = 0;
    {
        std::scoped_lock lk{mt_mutex};
        if (cpu_flip_epoch == ~0u || static_cast<s32>(next_stage_b_flip - cpu_flip_epoch) > 0) {
            wait_tick = params_ticks[slot];
            params_ticks[slot] = 0;
            AdvanceCpuHistoryToFlipLocked(next_stage_b_flip);
        }
    }
    if (wait_tick) {
        scheduler.Wait(wait_tick);
    }
    for (auto& target : targets) {
        target.written = false;
    }
    scheduler.EndRendering();
    // Last frame's stores must be visible to this frame's loads. Also finish old reads
    // before the ping-pong half they reference is reused for writes.
    scheduler.Record([](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    });
}

u32 ObjectMotion::PrepareDraw(const DrawInfo& draw) {
    if (!enabled || BbToggle::Disabled(1u << 29)) {
        return 0;
    }
    const u32 cur_stage_b_flip = TemporalUpscaler::StageBFlipsRetiredForMt();
    std::scoped_lock lk{mt_mutex};
    if (cpu_flip_epoch == ~0u) {
        AdvanceCpuHistoryToFlipLocked(cur_stage_b_flip);
    } else if (cur_stage_b_flip != cpu_flip_epoch) {
        return 0;
    }
    const u32 slot = u32(cpu_flip_epoch % FrameSlots);
    if (slot_params_used[slot] >= ParamsPerFrame) {
        return 0;
    }
    const auto allocation = history.Prepare(draw);
    u32 flags = (allocation.store ? Shader::MotionVectors::FlagStore : 0) |
                (allocation.load ? Shader::MotionVectors::FlagLoad : 0);
    if (!flags) {
        return 0;
    }
    const u32 index = 1 + slot * ParamsPerFrame + slot_params_used[slot]++;
    u32* param = params_mapped + index * 8;
    param[0] = allocation.store;
    param[1] = allocation.load;
    param[2] = allocation.vertices;
    param[3] = flags;
    param[4] = allocation.first_vertex;
    param[5] = allocation.first_instance;
    param[6] = allocation.instances;
    param[7] = 0;
    params_ticks[slot] = scheduler.CurrentTick();
    return index;
}

u32 ObjectMotion::PrepareDrawMt(u32 guest_flip_epoch, bool gated,
                                const Motion::History::GateKey& gate_key, u64 palette,
                                bool is_indexed, VAddr index_address, u32 index_count,
                                u32 index_size, s32 base_vertex, bool prim_restart,
                                u32 num_instances, u32 first_instance, const void* index_cpu_ptr,
                                u32& out_slot) noexcept {
    out_slot = ~0u;
    if (!enabled || BbToggle::Disabled(1u << 29) || index_count == 0u || num_instances == 0u) {
        return 0;
    }
    std::scoped_lock lk{mt_mutex};
    if (cpu_flip_epoch == ~0u || static_cast<s32>(guest_flip_epoch - cpu_flip_epoch) > 0) {
        AdvanceCpuHistoryToFlipLocked(guest_flip_epoch);
    } else if (guest_flip_epoch != cpu_flip_epoch) {
        return 0;
    }
    const u32 slot = u32(cpu_flip_epoch % FrameSlots);
    if (params_ticks[slot] != 0) {
        if (scheduler.IsFree(params_ticks[slot])) {
            params_ticks[slot] = 0;
        } else {
            return 0;
        }
    }
    if (slot_params_used[slot] >= ParamsPerFrame) {
        return 0;
    }
    if (gated && !history.Moving(gate_key, palette)) {
        return 0;
    }

    Motion::VertexRange range{static_cast<u32>(base_vertex), index_count};
    u64 topology = 0;
    if (is_indexed) {
        if (!index_cpu_ptr || index_address == 0) {
            return 0;
        }
        const u64 bytes = u64(index_count) * index_size;
        const auto scanned = index_ranges.Get(
            {index_address, index_count, index_size, base_vertex, prim_restart}, frame, [&] {
                const auto scan_range =
                    index_size == 2u
                        ? Motion::IndexedRange(
                              std::span(static_cast<const u16*>(index_cpu_ptr), index_count),
                              base_vertex, prim_restart)
                        : Motion::IndexedRange(
                              std::span(static_cast<const u32*>(index_cpu_ptr), index_count),
                              base_vertex, prim_restart);
                return Motion::IndexRangeCache::Result{scan_range,
                                                       XXH3_64bits(index_cpu_ptr, bytes)};
            });
        range = scanned.range;
        topology = scanned.topology;
    }

    const auto allocation = history.Prepare({
        .shader = gate_key.shader,
        .geometry = gate_key.geometry,
        .indices = index_address,
        .topology = topology,
        .index_count = index_count,
        .instances = num_instances,
        .first_instance = first_instance,
        .vertices = range,
    });
    const u32 flags = (allocation.store ? Shader::MotionVectors::FlagStore : 0) |
                      (allocation.load ? Shader::MotionVectors::FlagLoad : 0);
    if (!flags) {
        return 0;
    }
    const u32 index = 1 + slot * ParamsPerFrame + slot_params_used[slot]++;
    u32* param = params_mapped + index * 8;
    param[0] = allocation.store;
    param[1] = allocation.load;
    param[2] = allocation.vertices;
    param[3] = flags;
    param[4] = allocation.first_vertex;
    param[5] = allocation.first_instance;
    param[6] = allocation.instances;
    param[7] = 0;
    out_slot = slot;
    return index;
}

void ObjectMotion::NoteSlotSubmittedOnStageB(u32 slot) noexcept {
    if (!enabled || slot >= FrameSlots) {
        return;
    }
    std::scoped_lock lk{mt_mutex};
    params_ticks[slot] = scheduler.CurrentTick();
}

ObjectMotion::Target& ObjectMotion::GetTarget(u32 width, u32 height) {
    for (auto& target : targets) {
        if (target.image && target.width == width && target.height == height) {
            target.used_frame = frame;
            return target;
        }
    }
    // A free slot, else the image used longest ago (not the one being read).
    Target* slot = nullptr;
    for (auto& target : targets) {
        if (!target.image) {
            slot = &target;
            break;
        }
        if (&target != read_target && (!slot || target.used_frame < slot->used_frame)) {
            slot = &target;
        }
    }
    if (slot->image) {
        // Destroyed once the GPU is past the work recorded so far (it may still use it).
        scheduler.DeferOperation([image = std::move(slot->image),
                                  view = std::move(slot->view)]() mutable {
            view.reset();
            image.Destroy();
        });
    }
    static u32 reports = 0;
    if (BbStats::enabled && reports < 16) {
        ++reports;
        std::printf("Object motion: motion image %ux%u%s\n", width, height,
                    slot->width ? " (replacing one)" : "");
    }
    const auto device = instance.GetDevice();
    slot->image = VideoCore::HostImage(device, instance.GetAllocator());
    slot->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    slot->view = Check(device.createImageViewUnique({
        .image = vk::Image(slot->image),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    slot->width = width;
    slot->height = height;
    slot->written = false;
    slot->layout = vk::ImageLayout::eUndefined;
    slot->used_frame = frame;
    return *slot;
}

void ObjectMotion::Attach(RenderState& state, u32 width, u32 height) {
    constexpr u32 slot = Shader::MotionVectors::Output;
    Target& target = GetTarget(width, height);
    auto& attachment = state.color_attachments[slot];
    attachment = {};
    attachment.image_view = *target.view;
    attachment.image_layout = vk::ImageLayout::eColorAttachmentOptimal;
    if (!target.written || target.layout != vk::ImageLayout::eColorAttachmentOptimal) {
        // Clear once per frame; later reads must not erase the validity of this history.
        scheduler.EndRendering();
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eColorAttachmentRead,
            .oldLayout = target.layout,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = vk::Image(target.image),
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        // Recorded in order on the recording thread: CommandBuffer() here switched the rest of
        // the submission (most of the G-buffer) to direct recording on the GPU thread.
        scheduler.Record([barrier](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        });
        target.layout = vk::ImageLayout::eColorAttachmentOptimal;
        attachment.is_clear = !target.written;
        target.written = true;
    }
    state.num_color_attachments = std::max<u16>(state.num_color_attachments, slot + 1);
}

vk::ImageView ObjectMotion::PrepareRead(u32 width, u32 height, bool& valid) {
    // The image of this size; otherwise any image (the pass binds one, its vectors unused).
    Target* target = const_cast<Target*>(Find(width, height));
    if (!target) {
        for (auto& candidate : targets) {
            if (candidate.image && (!target || candidate.used_frame > target->used_frame)) {
                target = &candidate;
            }
        }
    }
    if (!target) {
        target = &GetTarget(1, 1);
    }
    valid = target->written && width == target->width && height == target->height;
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = target->layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(target->image),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.Record([barrier](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    });
    target->layout = vk::ImageLayout::eGeneral;
    read_target = target;
    return *target->view;
}

} // namespace Vulkan
