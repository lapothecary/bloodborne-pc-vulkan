// SPDX-License-Identifier: GPL-2.0-or-later
// Shared single-pass delta push-descriptor & push-constant state tracker.
// Used by both Stage B primary recording (`GraphicsPipeline::BindResources`) and
// Phase 4 per-worker secondary command buffer recording (`VkMtDirectRecorder`).
#pragma once

#include <array>

#include "common/types.h"
#include "shader/resource.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Scheduler;

inline constexpr u32 kMaxTrackedBindings = 64;

struct TrackedDescSlot {
    bool is_image{false};
    vk::DescriptorType type{};
    VkSampler sampler{VK_NULL_HANDLE};
    VkImageView image_view{VK_NULL_HANDLE};
    vk::ImageLayout image_layout{vk::ImageLayout::eUndefined};
    VkBuffer buffer{VK_NULL_HANDLE};
    vk::DeviceSize offset{0};
    vk::DeviceSize range{0};
};

struct GfxPushStateTracker {
    const Scheduler* scheduler{nullptr};
    u64 desc_epoch{0};
    u64 push_epoch{0};
    vk::PipelineLayout layout{};
    u64 valid_mask{0};
    bool has_push_data{false};
    Shader::PushData push_data{};
    std::array<TrackedDescSlot, kMaxTrackedBindings> slots{};

    void OnDraw(const Scheduler* sched, u64 d_ep, u64 p_ep, vk::PipelineLayout lay) noexcept {
        if (scheduler != sched) {
            scheduler = sched;
            desc_epoch = d_ep;
            push_epoch = p_ep;
            layout = lay;
            valid_mask = 0;
            has_push_data = false;
            return;
        }
        if (desc_epoch != d_ep) {
            desc_epoch = d_ep;
            layout = lay;
            valid_mask = 0;
        } else if (layout != lay) {
            layout = lay;
            valid_mask = 0;
        }
        if (push_epoch != p_ep) {
            push_epoch = p_ep;
            has_push_data = false;
        }
    }

    void InvalidateAll() noexcept {
        scheduler = nullptr;
        desc_epoch = 0;
        push_epoch = 0;
        layout = vk::PipelineLayout{};
        valid_mask = 0;
        has_push_data = false;
    }
};

} // namespace Vulkan
