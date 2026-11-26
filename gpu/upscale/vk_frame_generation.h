// SPDX-License-Identifier: GPL-2.0-or-later
// FSR 3.1 Frame Generation (portable Vulkan interface of FSR-Vulkan, in third_party/fsr-vulkan).
// An optical-flow + motion-vector interpolated frame is generated between each pair of real 3D
// frames, with clean HUDless separation captured right before the first Scaleform UI draw.
//
// Pipeline contract (recording thread):
//  - CaptureScene: called at the end of the temporal upscaler (RunPrePostHdr / RunScaled) or
//    motion-vector pass (RunUiOnly when upscaler is Off/SMAA). Copies the 3D scene depth buffer
//    before guest passes reuse it, and records camera & motion-vector inputs.
//  - CaptureHudless: called right before the first Scaleform HUD draw onto `ui_image` (strictly
//    after NativePostFX + post-Yebis Glow/AA), giving FSR 3.1 an exact UI-free scene color so
//    static HUD stays crisp without doubling 3D glow or lights.
//  - Interpolate: called at OnDisplayPassStart when `ui_image` holds the finished scene + HUD.
//    Records OpticalFlow + FrameInterpolation Prepare & Dispatch into the frame-generation output.
//  - ResetHistory: called on non-3D frames (loading screens, title menu, blank frames) or when
//    Frame Generation is toggled off so returning to gameplay never interpolates across a cut.
//  - TakeOutput: called by Presenter::PrepareFrame to encode the interpolated frame through the
//    exact same Display Gamma 1D LUT + RCAS + SDR/HDR10 PQ EncodePass as the real frame.
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/types.h"
#include "textures/image.h"
#include "upscale/vk_camera_motion.h"
#include "upscale/vk_dlss.h"
#include "vulkan/vk_common.h"

struct FfxVkPortableFrameGenerationContext;

namespace Vulkan {

class Instance;
class Scheduler;

class FrameGeneration {
public:
    static constexpr u32 MaxInterpOutputs = 3;

    FrameGeneration(const Instance& instance, Scheduler& scheduler);
    ~FrameGeneration();

    /// The setting is on and the selected GPU backend (FSR 3.1 or DLSS-G) is ready.
    bool Wanted() const;

    /// What the scene of this frame was rendered with (render size: depth and motion).
    struct Scene {
        vk::Image color;      ///< output size, the scene/UI target (General layout when copied/read)
        vk::ImageView color_view;
        vk::Format color_format;
        u32 width, height;    ///< output size
        vk::Image depth;      ///< render size
        vk::Format depth_format;
        vk::ImageLayout depth_layout = vk::ImageLayout::eGeneral;
        vk::Image motion;     ///< render size, RG16F pixels (previous - current), General
        vk::ImageView motion_view;
        u32 render_width, render_height;
        float jitter_x, jitter_y;
        float frame_ms;
        float camera_near, camera_far, vertical_fov;
        bool reset;
        bool hdr = false;
        /// The camera in the world (FSR/DLSS-G use them).
        float camera_position[3], camera_up[3], camera_right[3], camera_forward[3];
        CameraMotion::DlssgMatrices dlssg_matrices{};
    };
    /// End of the scene render/upscale pass (recording thread): copies depth and stores inputs.
    void CaptureScene(const Scene& scene);

    /// The first Scaleform UI/HUD draw is about to run over `color` (`ui_image`): snapshots the
    /// finished scene (NativePostFX + post-Yebis Glow/AA included, HUD excluded).
    void CaptureHudless(vk::Image color);

    /// The game's display pass of `display` (recording thread): `color` (`ui_image`) now holds the
    /// finished frame with its UI. Records OpticalFlow + FrameInterpolation if a scene was captured.
    void Interpolate(vk::Image color, VAddr display);

    /// Non-3D frame (loading screen, title menu, blank flip) or mode toggle off: invalidates any
    /// pending scene/output and forces a clean history reset on the next 3D frame.
    void ResetHistory();

    /// Pre-creates the FSR 3.1 Optical Flow + Frame Interpolation context (while in menu/loading)
    /// so entering 3D gameplay incurs zero pipeline-compile hitch on the first interpolated frame.
    void WarmUp(u32 render_width, u32 render_height, u32 width, u32 height,
                vk::Format format = vk::Format::eR8G8B8A8Unorm);

    /// Presenter: interpolated frames before the last real one of `display` (General layout,
    /// the scene's format and size); consumed once. Returns the number of outputs written.
    struct Output {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        u32 width, height;
        u64 frame;
    };
    u32 TakeOutputs(VAddr display, std::array<Output, MaxInterpOutputs>& out,
                    int* out_backend = nullptr);

private:
    bool UseDlssg() const;
    bool EnsureContext(u32 render_width, u32 render_height, u32 width, u32 height, vk::Format format);
    bool EnsureTargetSlot(u32 slot);
    void Destroy();

    const Instance& instance;
    Scheduler& scheduler;
    bool supported = false;
    bool failed = false;
    std::atomic<bool> dlssg_failed{false};
    int active_backend = 0; ///< 0 = None, 1 = FSR 3.1, 2 = DLSS-G
    std::optional<Dlss::FgFeatureDesc> queued_dlssg_desc;

    FfxVkPortableFrameGenerationContext* context = nullptr;
    u32 context_render_w = 0, context_render_h = 0, context_w = 0, context_h = 0;
    vk::Format context_format{};

    Scene scene{};
    bool scene_captured = false; ///< this frame's, not yet interpolated
    u64 frame_id = 0;
    bool reset_next = true;
    u64 scene_id = 0;        ///< the captured scene's frame id (its Prepare's and Dispatch's)
    bool scene_reset = true; ///< it starts over (nothing before it to interpolate from)
    /// The scene's depth, copied at the capture (the game reuses its depth buffer before the
    /// display pass, where Prepare runs).
    VideoCore::HostImage depth_copy;
    vk::UniqueImageView depth_copy_view;
    VideoCore::HostImage hudless;     ///< the frame without its UI (CaptureHudless)
    vk::UniqueImageView hudless_view;
    bool hudless_ready = false;       ///< taken for the captured scene
    vk::Format hudless_format{};
    u32 hudless_w = 0, hudless_h = 0;
    vk::Format depth_copy_format{};
    u32 depth_copy_w = 0, depth_copy_h = 0;

    struct Target {
        VideoCore::HostImage image;
        vk::UniqueImageView view;
        vk::Format format{};
        u32 width = 0, height = 0;
        u64 frame = 0;
        bool ready = false;
    };
    std::mutex targets_mutex;
    std::array<Target, MaxInterpOutputs> targets;
    u32 ready_count = 0;
    int ready_backend = 0;
};

} // namespace Vulkan
