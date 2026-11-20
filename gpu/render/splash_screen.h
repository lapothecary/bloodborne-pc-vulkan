// SPDX-License-Identifier: GPL-2.0-or-later
// Authentic Bloodborne boot splash screen (`dvdroot_ps4/menu/title.tpf.dcx` -> `MENU_Title_00001`
// 1920x1080 main-menu artwork, falling back to `sce_sys/pic1.png` / `pic0.png`, loaded directly
// from the user's game folder): shown immediately upon window creation (even during PipelineCache
// WarmUp) and held across Bloodborne's empty gray (#808080) VideoOut flips, Skip-Intro black
// frames, and fade-in until the live game framebuffer actually renders visible content (verified
// asynchronously via a 32x18 GPU luminance probe with zero stall), then frees all VRAM.
#pragma once

#include <chrono>

#include "common/types.h"
#include "vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

class Instance;
class Scheduler;
class Swapchain;

class SplashScreen {
public:
    SplashScreen(Instance& instance, Scheduler& present_scheduler, Swapchain& swapchain);
    ~SplashScreen();

    SplashScreen(const SplashScreen&) = delete;
    SplashScreen& operator=(const SplashScreen&) = delete;

    /// Returns true while the boot splash should cover the swapchain.
    /// When `game_image` is non-null (in `eGeneral` layout with `eTransferSrc`) and the game has
    /// started issuing draws, schedules a zero-stall 32x18 GPU readback probe on `cmdbuf` to detect
    /// when Bloodborne has finished its gray (#808080) / black (#000000) / fade-in boot frames and
    /// is presenting real visible content. Once ready, releases all VRAM asynchronously and returns
    /// false forever.
    bool UpdateAndShouldCover(vk::CommandBuffer cmdbuf, vk::Image game_image,
                              vk::Extent2D game_extent);

    /// Records the splash blit (or pure black if no artwork was found) into `target` (already in
    /// `eTransferDstOptimal`).
    void RecordBlit(vk::CommandBuffer cmdbuf, vk::Image target, vk::Extent2D extent) const;

    /// Releases splash and probe resources immediately (called from `~Presenter()` after GPU idle).
    void DestroyImmediate();

private:
    bool LoadAndUpload(Scheduler& present_scheduler);
    bool CreateProbeResources();
    void PresentInitialFrame(Scheduler& present_scheduler, Swapchain& swapchain);
    void RecordProbe(vk::CommandBuffer cmdbuf, vk::Image game_image, vk::Extent2D game_extent);
    bool EvaluateProbe();
    void ScheduleRelease();

    Instance& instance;
    Scheduler& scheduler;

    // 1920x1080 boot splash image
    VmaAllocation allocation{};
    vk::Image image{};
    u32 width = 0;
    u32 height = 0;

    // Tiny 32x18 async readiness probe (2.3 KiB, destroyed immediately on handoff)
    VmaAllocation probe_img_alloc{};
    vk::Image probe_image{};
    VmaAllocation probe_buf_alloc{};
    vk::Buffer probe_buffer{};
    const u8* probe_mapped = nullptr;
    bool probe_in_flight = false;
    u64 probe_tick = 0;

    bool active = true;
    bool seen_draws = false;
    std::chrono::steady_clock::time_point first_draw_time{};
    u32 prev_max_luma = 0;
    u32 stable_probes = 0;
    u32 probe_count = 0;
};

} // namespace Vulkan
