// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the pass between the game's display buffer and the presented frame: a
// full-screen draw into the frame image, encoding for the display (sRGB curve with the user's
// gamma, or HDR passed through).
#pragma once

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class EncodePass {
public:
    /// Push constants of present_encode.frag.
    struct Settings {
        float gamma = 1.0f;
        /// 0 = SDR sRGB, 1 = passthrough PQ, 2 = Native HDR10 ST.2084 PQ (Rec.709 -> Rec.2020 ITMO)
        u32 hdr = 0;
        u32 srgb_input = 0;
        /// RCAS strength (0: none) on the finished frame: sharpening without an upscaler.
        float sharpen = 0.0f;
        float hdr_peak_nits = 800.0f;
        float hdr_paper_white = 240.0f;
    };

    EncodePass(vk::Device device, vk::Format sdr_output_format);
    ~EncodePass();

    /// Draws `input` (shader read layout) over all of `output` (color attachment layout, left
    /// in it).
    void Render(vk::CommandBuffer cmdbuf, vk::ImageView input, vk::ImageView output,
                vk::Extent2D output_size, const Settings& settings);

private:
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline_sdr;
    vk::UniquePipeline pipeline_hdr;
};

} // namespace Vulkan
