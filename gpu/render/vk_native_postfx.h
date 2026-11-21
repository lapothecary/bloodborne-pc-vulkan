// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2: Native Vulkan 1.3 Post-Process pipeline (replaces Yebis2 black-box when enabled).
#pragma once

#include <array>
#include <memory>
#include <unordered_map>

#include "vulkan/vk_common.h"
#include "cp/cp_packets.h"
#include "textures/image.h"
#include <vk_mem_alloc.h>

namespace VideoCore {
class BufferCache;
class TextureCache;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;
class TemporalUpscaler;

class NativePostFx {
public:
    NativePostFx(const Instance& instance, Scheduler& scheduler,
                 VideoCore::TextureCache& texture_cache, VideoCore::BufferCache& buffer_cache,
                 Runtime& runtime);
    ~NativePostFx();

    NativePostFx(const NativePostFx&) = delete;
    NativePostFx& operator=(const NativePostFx&) = delete;

    /// Runs the native Vulkan post-process pipeline for a single frame (returns true if rendered).
    bool Execute(const BbCp::DirectVk::NativePostProcessArgs& args, TemporalUpscaler* upscaler);

private:
    bool EnsureInit();
    bool EnsureBloomPyramid(u32 hdr_w, u32 hdr_h);
    bool EnsureMotionBlurTargets(u32 hdr_w, u32 hdr_h);
    bool EnsureDofTargets(u32 hdr_w, u32 hdr_h);
    vk::Pipeline GetOrCreateTonemapPipeline(vk::Format color_format);

    static constexpr u32 kBloomMips = 5u;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    VideoCore::BufferCache& buffer_cache;
    Runtime& runtime;

    bool initialized = false;
    bool init_failed = false;
    bool gpu_state_cleared = false;

    vk::UniqueSampler linear_sampler;
    VideoCore::HostImage dummy_lut3d;
    vk::UniqueImageView dummy_lut3d_view;
    VideoCore::HostImage dummy_black_2d;
    vk::UniqueImageView dummy_black_2d_view;

    VideoCore::HostImage bloom_pyramid;
    std::array<vk::UniqueImageView, kBloomMips> bloom_mip_views;
    std::array<vk::Extent2D, kBloomMips> bloom_mip_extents{};
    u32 bloom_base_w = 0u;
    u32 bloom_base_h = 0u;

    VideoCore::HostImage mb_vel_img;
    vk::UniqueImageView mb_vel_view;
    vk::Extent2D mb_vel_extent{};
    VideoCore::HostImage mb_color_img;
    vk::UniqueImageView mb_color_view;
    vk::Extent2D mb_color_extent{};

    VideoCore::HostImage dof_coc_img;
    vk::UniqueImageView dof_coc_view;
    VideoCore::HostImage dof_bokeh_img;
    vk::UniqueImageView dof_bokeh_view;
    vk::Extent2D dof_half_extent{};
    VideoCore::HostImage dof_color_img;
    vk::UniqueImageView dof_color_view;
    vk::Extent2D dof_full_extent{};

    VkBuffer hist_buf = VK_NULL_HANDLE;
    VmaAllocation hist_alloc = VK_NULL_HANDLE;
    VkBuffer exp_state_buf = VK_NULL_HANDLE;
    VmaAllocation exp_state_alloc = VK_NULL_HANDLE;

    vk::UniqueDescriptorSetLayout hist_ds_layout;
    vk::UniquePipelineLayout hist_layout;
    vk::UniquePipeline hist_pipeline;

    vk::UniqueDescriptorSetLayout auto_exp_ds_layout;
    vk::UniquePipelineLayout auto_exp_layout;
    vk::UniquePipeline auto_exp_pipeline;

    vk::UniqueDescriptorSetLayout bloom_down_ds_layout;
    vk::UniquePipelineLayout bloom_down_layout;
    vk::UniquePipeline bloom_down_pipeline;

    vk::UniqueDescriptorSetLayout bloom_up_ds_layout;
    vk::UniquePipelineLayout bloom_up_layout;
    vk::UniquePipeline bloom_up_pipeline;

    vk::UniqueDescriptorSetLayout mb_vel_ds_layout;
    vk::UniquePipelineLayout mb_vel_layout;
    vk::UniquePipeline mb_vel_pipeline;

    vk::UniqueDescriptorSetLayout mb_blur_ds_layout;
    vk::UniquePipelineLayout mb_blur_layout;
    vk::UniquePipeline mb_blur_pipeline;

    vk::UniqueDescriptorSetLayout dof_coc_ds_layout;
    vk::UniquePipelineLayout dof_coc_layout;
    vk::UniquePipeline dof_coc_pipeline;

    vk::UniqueDescriptorSetLayout dof_bokeh_ds_layout;
    vk::UniquePipelineLayout dof_bokeh_layout;
    vk::UniquePipeline dof_bokeh_pipeline;

    vk::UniqueDescriptorSetLayout dof_comp_ds_layout;
    vk::UniquePipelineLayout dof_comp_layout;
    vk::UniquePipeline dof_comp_pipeline;

    vk::UniqueDescriptorSetLayout tonemap_ds_layout;
    vk::UniquePipelineLayout tonemap_layout;
    vk::UniqueShaderModule tonemap_vs;
    vk::UniqueShaderModule tonemap_fs;
    std::unordered_map<vk::Format, vk::UniquePipeline> tonemap_pipelines;
};

} // namespace Vulkan
