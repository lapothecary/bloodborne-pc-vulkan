// SPDX-FileCopyrightText: Copyright 2026 IFreemz, bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bbport: NVIDIA DLSS Super Resolution through the optional bridge (gpu/dlss_bridge, MIT;
// libbbport_dlss.so / bbport_dlss.dll), loaded at run time from next to bb-probe together with
// NVIDIA's libnvidia-ngx-dlss.so.<version> / nvngx_dlss.dll. The port itself contains no NVIDIA
// code; without the libraries, on other GPUs or
// with BB_DLSS=0 nothing changes. Adapted from IFreemz/shadPS4-Bloodborne-DLSS-FSR
// (vk_dlss_ngx).

#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Dlss {
public:
    /// The process-wide instance, or null when the DLLs are absent or BB_DLSS=0.
    static Dlss* Get();
    ~Dlss();

    /// Instance and device creation: adds what NGX needs (or disables DLSS when it is missing).
    void AppendInstanceExtensions(std::vector<const char*>& enabled);
    void AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                std::vector<const char*>& enabled);
    /// After the device exists.
    void Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
    /// The device could not be created with NGX's extensions (the driver advertises them, but
    /// what they load is missing: libcuda.so.1, the UVM module): DLSS off, FSR 3.1 instead.
    void Disable(std::string_view reason);
    void Shutdown();

    [[nodiscard]] bool Available() const;
    [[nodiscard]] bool FgAvailable() const;
    [[nodiscard]] u32 FgMaxMultiFrame() const;
    /// Non-blocking boot check: true when NVIDIA RTX + NGX bridge/extensions are present
    /// (background `bb:DlssInit` may still be finalizing NGX initialization).
    [[nodiscard]] bool EligibleForBoot() const noexcept;
    /// Why DLSS cannot run on this system (empty when it can, or before device creation).
    [[nodiscard]] std::string Problem() const;
    [[nodiscard]] std::string FgProblem() const;

    struct Resource {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        vk::ImageAspectFlags aspect;
        u32 width, height;
    };
    struct FeatureDesc {
        u32 input_width, input_height, output_width, output_height;
        int quality;
        bool hdr;
        int preset = 11; ///< 11 = DLSS 4 Transformer (Preset K), 0 = DLSS 3 CNN Default
        bool operator==(const FeatureDesc&) const = default;
    };
    struct Frame {
        Resource color, depth, motion, output;
        float jitter_x, jitter_y;
        bool reset;
        float frame_ms;
        float sharpness;
    };
    struct FgFeatureDesc {
        u32 input_width, input_height, backbuffer_width, backbuffer_height;
        vk::Format backbuffer_format{vk::Format::eUndefined};
        bool dynamic_resolution{};
        bool depth_inverted{};
        bool hdr{};
        bool operator==(const FgFeatureDesc&) const = default;
    };
    struct FgFrame {
        Resource backbuffer, depth, motion, hudless{}, output;
        u64 frame_id{};
        u32 multi_frame_count{1};
        u32 multi_frame_index{1};
        bool reset{};
        bool depth_inverted{};
        bool hdr{};
        bool camera_motion_included{true};
        u32 render_subrect_width{}, render_subrect_height{};
        float jitter_x{}, jitter_y{};
        float mv_scale_x{1.0f}, mv_scale_y{1.0f};
        std::array<float, 3> camera_pos{}, camera_up{}, camera_right{}, camera_fwd{};
        float camera_near{}, camera_far{}, camera_fov{}, camera_aspect{1.0f};
        std::array<float, 16> camera_view_to_clip{}, clip_to_camera_view{}, clip_to_prev_clip{},
            prev_clip_to_clip{}, clip_to_lens_clip{};
    };

    /// DLSS quality mode for an output/render size ratio (0 DLAA .. 4 Ultra Performance).
    static int QualityForScale(float scale);
    [[nodiscard]] bool HasFeature(const FeatureDesc& desc) const;
    /// Records the feature creation. The GPU must be done with the previous one.
    bool CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc);
    bool Evaluate(vk::CommandBuffer command, const Frame& frame);
    void ReleaseFeature();

    [[nodiscard]] bool HasFgFeature(const FgFeatureDesc& desc) const;
    /// Records the DLSS-G feature creation. The GPU must be done with the previous one.
    bool CreateFgFeature(vk::CommandBuffer command, const FgFeatureDesc& desc);
    bool EvaluateFg(vk::CommandBuffer command, const FgFrame& frame);
    void ReleaseFgFeature();

private:
    Dlss();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
