// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: from the game's display buffers to the window. Owns the Vulkan device, the
// schedulers and the rasterizer. A flip is prepared on the command processor's thread (the
// display buffer through the display encoding into a frame image, recorded with the
// draws) and presented on the present thread (the frame letterboxed into a swapchain image with
// the port's overlay over it).
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "platform/videoout/videoout.h"
#include "render/present_passes.h"
#include "render/splash_screen.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "vulkan/swapchain.h"

namespace BbCp {
class CommandProcessor;
}
namespace Frontend {
class WindowSDL;
}
namespace VideoCore {
class TextureCache;
struct Image;
} // namespace VideoCore

namespace Vulkan {

class Rasterizer;

/// A prepared frame: an image of the window's aspect, ready once `ready_semaphore` reaches
/// `ready_tick`; free again once `present_done` signals. The blank frame has no image.
struct Frame {
    u32 width = 0;
    u32 height = 0;
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageView image_view{};
    vk::Fence present_done{};
    vk::Semaphore ready_semaphore{};
    u64 ready_tick = 0;
    u64 present_id = 0; ///< monotonic low-latency frame ID (Reflex / Anti-Lag 2)
    bool is_hdr = false;
    u8 id = 0;
    /// Frame generation: interpolated frames before this one, encoded identically (shown first).
    static constexpr u32 MaxInterpSlots = 3;
    std::array<VmaAllocation, MaxInterpSlots> interp_allocations{};
    std::array<vk::Image, MaxInterpSlots> interp_images{};
    std::array<vk::ImageView, MaxInterpSlots> interp_views{};
    u32 interp_count = 0;
    int interp_backend = 0;
};

class Presenter {
public:
    Presenter(Frontend::WindowSDL& window, BbCp::CommandProcessor* cp);
    ~Presenter();

    bool IsHDRSupported() const {
        return swapchain.HasHDR();
    }
    /// Present thread: the swapchain in HDR10 or SDR (when the surface has HDR).
    void SetHDR(bool enable);
    /// The user's display gamma (1: the standard sRGB curve).
    void SetGamma(float gamma) {
        this->gamma.store(gamma, std::memory_order_relaxed);
    }

    /// A display buffer the game registered (its image is marked as a video out surface).
    VideoCore::Image& RegisterVideoOutSurface(const BbVideoOut::BufferAttributeGroup& attribute,
                                              VAddr cpu_address);

    /// Command processor thread: the display buffer at `cpu_address` into a free frame.
    Frame* PrepareFrame(const BbVideoOut::BufferAttributeGroup& attribute, VAddr cpu_address);
    /// Frame generation's interpolated frames before `frame` (PrepareFrame): encoded into
    /// `frame->interp_images[slot]` like the real one (including PassFusion's fused Gamma 1D LUT).
    void EncodeInterpolated(Frame* frame, u32 slot, vk::Image image, vk::Format format, u32 width,
                            u32 height, vk::Format display_view_format,
                            const EncodePass::Settings& encode, bool fused_gamma = false,
                            vk::Format fused_src_format = {},
                            vk::ComponentMapping fused_src_mapping = {},
                            vk::ImageView lut1d_view = {},
                            vk::ImageLayout lut1d_layout = vk::ImageLayout::eShaderReadOnlyOptimal);
    /// A black frame (no GPU work until it is presented).
    Frame* PrepareBlankFrame(bool present_thread);
    /// Present thread: shows `frame` and frees it (unless `is_reusing_frame`).
    void Present(Frame* frame, bool is_reusing_frame = false, bool is_game_frame = true);
    /// Waits until the GPU has finished the commands that render `frame` (any thread).
    void WaitRendered(const Frame* frame);

private:
    /// Takes a free frame, waits for its last present and resizes it to the expected size.
    Frame* GetRenderFrame();
    void ReleaseFrame(Frame* frame);
    void RecreateFrame(Frame& frame, u32 width, u32 height);
    /// The frame size: the display buffer's aspect fitted into the swapchain.
    void SetExpectedSize(u32 width, u32 height);
    /// Records the swapchain image's contents: black bars and `frame` (if any) blitted in.
    void RecordPresent(vk::CommandBuffer cmdbuf, const Frame* frame, vk::Image target,
                       vk::Extent2D extent, vk::Image source = {});
    /// Shows `source` (or the frame's image) in the next swapchain image; false: none shown.
    /// `last`: the frame's final present (signals its fence).
    /// `fg_slot_period_ns`: non-zero when Frame Generation paces both interpolated and real slots
    /// on a unified half-frame monotonic grid; `wait_gpu_done` skips redundant WaitRendered on the
    /// second slot of a pair after the first slot already waited for `frame->ready_tick`.
    bool PresentImage(Frame* frame, vk::Image source, bool last, u64 fg_slot_period_ns = 0,
                      bool wait_gpu_done = true);

    Frontend::WindowSDL& window;
    Instance instance;
    BbCp::CommandProcessor* cp;
    Scheduler draw_scheduler;
    Scheduler present_scheduler;
    Swapchain swapchain;
    u32 last_swapchain_generation = 0;
    SplashScreen splash;
    Runtime runtime;
    std::unique_ptr<Rasterizer> rasterizer;
    VideoCore::TextureCache& texture_cache;
    EncodePass encode_pass;
    std::atomic<float> gamma{1.0f};
    std::atomic<bool> hdr{false};
    std::atomic<bool> guest_hdr{false};

    std::vector<Frame> frames;
    Frame blank_frame;
    std::deque<Frame*> free_frames; ///< oldest first: its present is the likeliest done
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::deque<u64> recent_frame_ticks; ///< BB_FRAMES_AHEAD bound (PrepareFrame)

    /// Written by the present thread, read by the command processor thread.
    std::atomic<float> expected_ratio{16.0f / 9.0f};
    std::atomic<u32> expected_width{1920};
    std::atomic<u32> expected_height{1080};
};

/// True per-frame draw count (2-frame sliding max) sampled once per completed frame in
/// Presenter::PrepareFrame. Used by PipelineCache::AsyncSkippable so sparse FastPSO misses
/// across async frame ticks never misread active combat as a <64-draw loading screen.
extern std::atomic<u64> g_last_completed_frame_draws;

} // namespace Vulkan
