// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2 (v1.2.4): Single-Pass Downsampler (SPD) in LDS for Bloodborne's SSR Glossy Mip Pyramid.
#pragma once

#include <array>

#include "common/types.h"
#include "cp/cp_packets.h"
#include "vulkan/vk_common.h"

namespace VideoCore {
class TextureCache;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;

class SsrSpdCompute {
public:
    SsrSpdCompute(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                  VideoCore::TextureCache& texture_cache);
    ~SsrSpdCompute();

    SsrSpdCompute(const SsrSpdCompute&) = delete;
    SsrSpdCompute& operator=(const SsrSpdCompute&) = delete;

    bool Execute(const BbCp::DirectVk::GenerateMipsSpdArgs& args);

private:
    bool EnsureInit();
    void EnsureMipViews(vk::Image image, u64 image_uid, vk::Format format, u32 total_levels);

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;

    bool initialized = false;
    bool init_failed = false;

    vk::UniqueSampler linear_sampler;
    vk::UniqueDescriptorSetLayout ds_layout;
    vk::UniquePipelineLayout pipe_layout;
    vk::UniquePipeline pipeline;

    vk::Image cached_image{};
    u64 cached_uid = 0u;
    vk::Format cached_format = vk::Format::eUndefined;
    u32 cached_levels = 0u;
    std::array<vk::ImageView, 6> cached_mip_views{};
};

} // namespace Vulkan
