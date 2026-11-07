// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: SMAA 1x Ultra & SMAA T2x (third_party/smaa, MIT) for anti_aliasing=smaa / smaa_t2x
// without an upscaler: run on the game's finished LDR scene (RGBA8, gamma space) before the UI
// draws over it. The game's own AA is off then (BbSettings::SyncAntiAliasing).

#pragma once

#include <array>

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Smaa {
public:
    Smaa(const Instance& instance, Scheduler& scheduler);
    ~Smaa();

    /// Prepares the RG16F motion vector image for `width` x `height` in GENERAL layout so
    /// CameraMotion::RecordMotion can write into MotionView() before Record(..., true).
    bool PrepareMotion(u32 width, u32 height);

    [[nodiscard]] vk::ImageView MotionView() const {
        return *motion_view;
    }

    /// Records the three SMAA 1x Ultra passes (plus the SMAA T2x temporal resolve pass when
    /// `temporal` is true) over `color` (sampled in `color_layout` through `color_view`, a UNORM
    /// view) of `width` x `height`. The result is Output(), RGBA8 in the GENERAL layout, readable
    /// by any later command. False: nothing recorded.
    bool Record(vk::ImageView color_view, vk::ImageLayout color_layout, u32 width, u32 height,
                bool temporal = false, bool reset_history = false);

    [[nodiscard]] vk::Image Output() const {
        return last_output;
    }

private:
    bool EnsureImages(u32 width, u32 height, bool temporal);
    void UploadLookupTextures();

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueSampler linear_sampler;
    vk::UniqueSampler point_sampler;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline edges_pipeline;
    vk::UniquePipeline weights_pipeline;
    vk::UniquePipeline blend_pipeline;
    vk::UniquePipeline resolve_pipeline;
    VideoCore::HostImage area_image, search_image; ///< SMAA's precomputed lookup textures
    vk::UniqueImageView area_view, search_view;
    bool uploaded = false;
    VideoCore::HostImage edges_image, weights_image, output_image;
    vk::UniqueImageView edges_view, weights_view, output_view;
    VideoCore::HostImage motion_image;
    vk::UniqueImageView motion_view;
    std::array<VideoCore::HostImage, 2> history_images{};
    std::array<vk::UniqueImageView, 2> history_views{};
    u32 history_next = 0;
    u32 frame_parity = 0;
    bool history_valid = false;
    int last_reported_mode = -1;
    vk::Image last_output{};
    u32 image_width = 0, image_height = 0;
};

} // namespace Vulkan
