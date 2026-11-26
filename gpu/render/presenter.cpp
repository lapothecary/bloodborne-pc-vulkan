// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/presenter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <limits>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <vk_mem_alloc.h>

#include "bbport_fast_load.h"
#include "bbport_frame_rate.h"
#include "bbport_game_menu.h"
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "core/emulator_settings.h"
#include "platform/videoout/display.h"
#include "sdl_window.h"
#include "textures/image.h"
#include "textures/texture_cache.h"
#include "upscale/vk_frame_generation.h"
#include "upscale/vk_temporal_upscaler.h"
#include "render/rasterizer.h"
#include "render/shader_hle.h"
#include "render/vk_gpu_profiler.h"
#include "vulkan/low_latency.h"
#include "vulkan/platform.h"

namespace Vulkan {

std::atomic<u64> g_last_completed_frame_draws{0};

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

/// How the display buffer is read: its format as a view (the 10-bit sRGB one has no sRGB
/// Vulkan format: it is decoded in the encode pass).
/// The shown frame's aspect (bbgpu_frame_aspect): any thread.
std::atomic<float> shown_aspect{16.0f / 9.0f};

vk::Format FrameViewFormat(BbVideoOut::PixelFormat format) {
    switch (format) {
    case BbVideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case BbVideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case BbVideoOut::PixelFormat::A2R10G10B10:
    case BbVideoOut::PixelFormat::A2R10G10B10Srgb:
    case BbVideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        UNREACHABLE_MSG("display buffer format {:#x}", u32(format));
    }
}

/// `frame` fitted into `target` keeping its aspect, centered.
vk::Rect2D Letterbox(vk::Extent2D frame, vk::Extent2D target) {
    const float frame_aspect = float(frame.width) / float(frame.height);
    const float target_aspect = float(target.width) / float(target.height);
    vk::Extent2D size = target;
    if (frame_aspect > target_aspect) {
        size.height = u32(float(target.width) / frame_aspect);
    } else {
        size.width = u32(float(target.height) * frame_aspect);
    }
    return {{s32(target.width - size.width) / 2, s32(target.height - size.height) / 2}, size};
}

/// BB_FRAMES_AHEAD (default 1): how many guest frames the command processor thread may run
/// ahead of the GPU (0: unbounded).
u32 FramesAhead() {
    static const u32 value = [] {
        const char* env = std::getenv("BB_FRAMES_AHEAD");
        return env ? u32(std::max(0, std::atoi(env))) : 1u;
    }();
    return value;
}

void WaitFence(vk::Device device, vk::Fence fence) {
    while (true) {
        const vk::Result result = device.waitForFences(fence, true, std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            return;
        }
        if (result != vk::Result::eTimeout) {
            Check<"wait for a frame">(result);
        }
    }
}

} // namespace

Presenter::Presenter(Frontend::WindowSDL& window_, BbCp::CommandProcessor* cp_)
    : window{window_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      cp{cp_}, draw_scheduler{instance, true}, present_scheduler{instance},
      swapchain{instance, window}, splash{instance, present_scheduler, swapchain},
      runtime{instance, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, runtime, cp)},
      texture_cache{rasterizer->GetTextureCache()},
      encode_pass{instance.GetDevice(), swapchain.GetSurfaceFormat().format} {
    const vk::Device device = instance.GetDevice();
    const u32 count = swapchain.GetImageCount();
    const u32 frame_count = std::max(count + 2u, 5u);
    frames.resize(frame_count);
    for (u32 i = 0; i < frame_count; ++i) {
        frames[i].id = u8(i);
        frames[i].present_done = Check<"create a frame fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        free_frames.push_back(&frames[i]);
    }
    hdr.store(swapchain.GetHDR(), std::memory_order_relaxed);
    BbOverlay::Init(instance, swapchain.GetActiveFormat(), frame_count);
}

Presenter::~Presenter() {
    draw_scheduler.Finish();
    present_scheduler.Finish();
    splash.DestroyImmediate();
    const vk::Device device = instance.GetDevice();
    for (auto& frame : frames) {
        device.destroyImageView(frame.image_view);
        if (frame.image) {
            vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        }
        for (u32 slot = 0; slot < Frame::MaxInterpSlots; ++slot) {
            device.destroyImageView(frame.interp_views[slot]);
            if (frame.interp_images[slot]) {
                vmaDestroyImage(instance.GetAllocator(), frame.interp_images[slot],
                                frame.interp_allocations[slot]);
            }
            frame.interp_images[slot] = vk::Image{};
            frame.interp_views[slot] = vk::ImageView{};
            frame.interp_allocations[slot] = VmaAllocation{};
        }
        frame.interp_count = 0;
        frame.interp_backend = 0;
        device.destroyFence(frame.present_done);
    }
}

void Presenter::SetHDR(bool enable) {
    guest_hdr.store(enable, std::memory_order_relaxed);
    const bool effective =
        IsHDRSupported() &&
        (enable || BbSettings::Get().hdr_output.load(std::memory_order_relaxed));
    if (swapchain.GetHDR() != effective) {
        // Changing HDR recreates the swapchain and waits for the device: exclude queue submits.
        std::scoped_lock lock{Scheduler::submit_mutex};
        swapchain.SetHDR(effective);
        BbOverlay::SetSwapchainFormat(swapchain.GetActiveFormat());
        last_swapchain_generation =
            BbSettings::Get().swapchain_generation.load(std::memory_order_relaxed);
        std::printf("Presenter: switched output live to %s\n",
                    effective ? "HDR10 (A2B10G10R10 ST.2084 PQ)" : "SDR (sRGB)");
    }
    hdr.store(effective, std::memory_order_release);
}

VideoCore::Image& Presenter::RegisterVideoOutSurface(
    const BbVideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    auto& image = texture_cache.GetImage(image_id);
    image.usage.vo_surface = 1u;
    return image;
}

void Presenter::RecreateFrame(Frame& frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    device.destroyImageView(frame.image_view);
    if (frame.image) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    }
    const bool is_hdr = hdr.load(std::memory_order_acquire);
    const vk::Format format =
        is_hdr ? vk::Format::eA2B10G10R10UnormPack32 : swapchain.GetSurfaceFormat().format;
    const VkImageCreateInfo image_info = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };
    const VmaAllocationCreateInfo alloc_info{
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VkImage image{};
    const VkResult result = vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info,
                                           &image, &frame.allocation, nullptr);
    Check<"allocate a frame image">(vk::Result(result));
    frame.image = image;
    frame.image_view = Check<"create a frame view">(device.createImageView({
        .image = frame.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = ColorRange,
    }));
    SetObjectName(device, frame.image, "frame {}", frame.id);
    // Frame generation's frames (made on demand: most sessions never need all slots).
    for (u32 slot = 0; slot < Frame::MaxInterpSlots; ++slot) {
        device.destroyImageView(frame.interp_views[slot]);
        if (frame.interp_images[slot]) {
            vmaDestroyImage(instance.GetAllocator(), frame.interp_images[slot],
                            frame.interp_allocations[slot]);
        }
        frame.interp_images[slot] = vk::Image{};
        frame.interp_views[slot] = vk::ImageView{};
        frame.interp_allocations[slot] = VmaAllocation{};
    }
    frame.interp_count = 0;
    frame.interp_backend = 0;
    frame.width = width;
    frame.height = height;
    frame.is_hdr = is_hdr;
}

Frame* Presenter::GetRenderFrame() {
    Frame* frame;
    {
        BbSections::PauseADecodeScope pause_decode;
        BbStats::WaitTimer timer{BbStats::present_wait_ns};
        {
            std::unique_lock lock{free_mutex};
            free_cv.wait(lock, [this] { return !free_frames.empty(); });
            frame = free_frames.front();
            free_frames.pop_front();
        }
        // Its last present has finished: the image, its FSR slot and the fence are free.
        WaitFence(instance.GetDevice(), frame->present_done);
    }
    const u32 width = expected_width.load(std::memory_order_relaxed);
    const u32 height = expected_height.load(std::memory_order_relaxed);
    if (frame->width != width || frame->height != height ||
        frame->is_hdr != hdr.load(std::memory_order_acquire)) {
        RecreateFrame(*frame, width, height);
    }
    return frame;
}

void Presenter::ReleaseFrame(Frame* frame) {
    if (frame == &blank_frame) {
        return;
    }
    {
        std::scoped_lock lock{free_mutex};
        free_frames.push_back(frame);
    }
    free_cv.notify_one();
}

Frame* Presenter::PrepareFrame(const BbVideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    BbTimeline::Note(BbTimeline::PipeTask, 4, draw_scheduler.CurrentTick());
    // Pack #2: Check if Rasterizer fused Bloodborne's final 1DArray Display Gamma LUT pass
    // (`ps_38d65b32`) directly into EncodePass (eliminating a full-res 1440p VRAM round-trip).
    Rasterizer::FusedGammaDisplay fused{};
    const bool can_fuse_gamma =
        attribute.attrib.pixel_format == BbVideoOut::PixelFormat::A8R8G8B8Srgb;
    const bool fused_gamma =
        can_fuse_gamma && rasterizer && rasterizer->ConsumeFusedGammaDisplay(cpu_address, fused);

    // Scaled upscaler presets: the output-size display buffer drawn by the port.
    TemporalUpscaler::Display display{};
    auto& upscaler = rasterizer->GetUpscaler();
    const bool upscaled = !fused_gamma && upscaler.DisplayOverride(cpu_address, display);
    VideoCore::ImageId image_id{};
    if (!fused_gamma && !upscaled) {
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        image_id = texture_cache.FindImage(desc);
        texture_cache.UpdateImage(image_id);
    }

    Frame* frame = GetRenderFrame();

    if (rasterizer) {
        rasterizer->FlushDeferredVsmCopy();
        rasterizer->FlushVsmBlurCopyBacks();
    }
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*rasterizer);
    }
    // Recorded like draws, on the recording threads: CommandBuffer() would first wait for them
    // to record everything handed over before and then submit from this thread.
    draw_scheduler.EndRendering();

    VideoCore::ImageViewInfo view_info{};
    view_info.format = FrameViewFormat(attribute.attrib.pixel_format);
    // The frame is opaque: alpha would blend it with the overlay.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;
    vk::ImageView source{};
    vk::ImageLayout source_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageView lut1d_view{};
    vk::ImageLayout lut1d_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::Extent2D source_size{};
    if (fused_gamma) {
        source = fused.src_view;
        source_layout = fused.src_layout;
        source_size = fused.src_size;
        lut1d_view = fused.lut1d_view;
        lut1d_layout = fused.lut1d_layout;
    } else if (upscaled) {
        const vk::Device device = instance.GetDevice();
        source = Check(device.createImageView({
            .image = display.image,
            .viewType = vk::ImageViewType::e2D,
            .format = view_info.format,
            .components = view_info.mapping,
            .subresourceRange = ColorRange,
        }));
        draw_scheduler.DeferOperation([device, source] { device.destroyImageView(source); });
        source_size = vk::Extent2D{display.width, display.height};
        draw_scheduler.Record([image = display.image](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier2 to_read{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = image,
                .subresourceRange = ColorRange,
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        });
    } else {
        auto& image = texture_cache.GetImage(image_id);
        source = *image.FindView(view_info).image_view;
        source_size = vk::Extent2D{image.info.size.width, image.info.size.height};
        // BB_PRESENT_DUMP_TRIGGER for the game's own display buffer too.
        if (upscaler.PresentDumpDue()) {
            runtime.Transit(&image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
                            vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            upscaler.DumpPresented(image.GetImage(), source_size.width, source_size.height,
                                   view_info.format);
        }
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader |
                            vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
    }
    expected_ratio.store(float(source_size.width) / float(source_size.height),
                         std::memory_order_relaxed);
    shown_aspect.store(float(source_size.width) / float(source_size.height),
                       std::memory_order_relaxed);

    static const float hdr_peak_nits = [] {
        const char* env = std::getenv("BB_HDR_PEAK_NITS");
        const float v = env && *env ? std::strtof(env, nullptr) : 800.0f;
        return std::clamp(v, 300.0f, 4000.0f);
    }();
    static const float hdr_paper_white = [] {
        const char* env = std::getenv("BB_HDR_PAPER_WHITE");
        const float v = env && *env ? std::strtof(env, nullptr) : 240.0f;
        return std::clamp(v, 80.0f, 600.0f);
    }();
    const u32 hdr_mode =
        !frame->is_hdr
            ? 0u
            : (attribute.attrib.pixel_format == BbVideoOut::PixelFormat::A2R10G10B10Bt2020Pq ? 1u
                                                                                             : 2u);
    const EncodePass::Settings encode{
        .gamma = gamma.load(std::memory_order_relaxed),
        .hdr = hdr_mode,
        .srgb_input = fused_gamma
                          ? u32(fused.needs_srgb_decode)
                          : u32(attribute.attrib.pixel_format ==
                                BbVideoOut::PixelFormat::A2R10G10B10Srgb),
        // Sharpening without an upscaler, or with DLSS/DLAA (NGX DLSS 2.5.1+ removed internal sharpening).
        .sharpen = (BbSettings::EffectiveUpscaler() == BbSettings::UpscalerOff ||
                    BbSettings::EffectiveUpscaler() == BbSettings::UpscalerDlss) &&
                           BbSettings::Get().sharpen
                       ? std::clamp(BbSettings::Get().sharpness.load(), 0.0f, 2.0f)
                       : 0.0f,
        .hdr_peak_nits = hdr_peak_nits,
        .hdr_paper_white = hdr_paper_white,
    };
    // Numbered here, in command order; marked where the commands are recorded.
    const u32 stream = draw_scheduler.CrumbStream();
    const u32 crumb = Breadcrumbs::Enabled()
                          ? Breadcrumbs::Note(stream, {.name = "presenter encode"})
                          : 0;
    // The encode pass holds no state: frames of different submissions may be recorded on two
    // threads at once.
    GpuProfiler::HudMarkPass(&draw_scheduler, BbStats::GpuSub_Ui_RcasPresent);
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&draw_scheduler)) {
        prof->Mark(fused_gamma ? 0x9A71FE20ULL : 0x9A71FE21ULL, [frame, fused_gamma, sharpen = encode.sharpen] {
            return fmt::format("Presenter: EncodePass {}x{} [FusedGamma1D:{} RCAS:{}] [VULKAN-NATIVE]",
                               frame->width, frame->height, fused_gamma ? 1 : 0,
                               sharpen > 0.0f ? 1 : 0);
        });
    }
    draw_scheduler.Record([this, frame, source, source_layout, lut1d_view, lut1d_layout, encode,
                           stream, crumb](vk::CommandBuffer cmdbuf) {
        if (crumb) {
            Breadcrumbs::Mark(cmdbuf, stream, crumb, false);
        }
        const vk::Extent2D size{frame->width, frame->height};
        // The frame's last reader (the present blit) is done: the fence was waited for.
        const vk::ImageMemoryBarrier2 to_attachment{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = frame->image,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_attachment});
        encode_pass.Render(cmdbuf, source, frame->image_view, size, encode, source_layout,
                           lut1d_view, lut1d_layout);
        // Read by the present blit (or a dump) in the general layout, after a semaphore wait.
        const vk::ImageMemoryBarrier2 to_general{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = frame->image,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
        if (crumb) {
            Breadcrumbs::Mark(cmdbuf, stream, crumb, true);
        }
    });

    // Frame generation: up to 3 interpolated frames before this real frame, encoded identically
    // (including PassFusion's fused 1D Display Gamma LUT, RCAS sharpening, and SDR/HDR10 PQ curve).
    frame->interp_count = 0;
    frame->interp_backend = 0;
    std::array<FrameGeneration::Output, FrameGeneration::MaxInterpOutputs> interps{};
    int fg_backend = 0;
    const u32 interp_n =
        ((fused_gamma || upscaled) && upscaler.GetFrameGeneration())
            ? upscaler.GetFrameGeneration()->TakeOutputs(cpu_address, interps, &fg_backend)
            : 0u;
    for (u32 slot = 0; slot < std::min<u32>(interp_n, Frame::MaxInterpSlots); ++slot) {
        const auto& interp = interps[slot];
        if (interp.width == source_size.width && interp.height == source_size.height) {
            EncodeInterpolated(
                frame, slot, interp.image, interp.format, interp.width, interp.height,
                view_info.format, encode, fused_gamma,
                fused.has_src_view_params ? fused.src_format : vk::Format::eUndefined,
                fused.has_src_view_params ? fused.src_mapping : vk::ComponentMapping{},
                lut1d_view, lut1d_layout);
        }
    }
    if (frame->interp_count > 0) {
        frame->interp_backend = fg_backend;
    }

    BbStats::frame_number.fetch_add(1, std::memory_order_relaxed);
    {
        static u64 prev_flip_draws = 0;
        static u64 prev_frame_draws = 0;
        const u64 cur_draws = BbStats::draws.load(std::memory_order_relaxed);
        const u64 frame_draws = cur_draws >= prev_flip_draws ? (cur_draws - prev_flip_draws) : 0ull;
        prev_flip_draws = cur_draws;
        g_last_completed_frame_draws.store(std::max(prev_frame_draws, frame_draws),
                                           std::memory_order_relaxed);
        prev_frame_draws = frame_draws;
    }
    {
        static u64 prev_sb_ns = 0;
        const u64 sb_now_ns = BbFrameRate::NowMonotonicNs();
        if (prev_sb_ns != 0 && sb_now_ns >= prev_sb_ns) {
            BbStats::last_stageb_flip_dt_ns.store(sb_now_ns - prev_sb_ns, std::memory_order_relaxed);
        }
        prev_sb_ns = sb_now_ns;
    }
    frame->ready_semaphore = draw_scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    frame->present_id = LowLatency::AssignRenderPresentId();
    SubmitInfo info{};
    info.present_id = frame->present_id;
    LowLatency::MarkRenderSubmitStart(frame->present_id);
    draw_scheduler.NoteFrameEndSubmission();
    draw_scheduler.Flush(info);
    LowLatency::MarkRenderSubmitEnd(frame->present_id);
    BbTimeline::Note(BbTimeline::PipeTask, 5, frame->ready_tick);

    // The command processor thread runs at most FramesAhead() guest frames ahead of the GPU: it
    // waits here for the frame that many flips back. When the GPU is the bottleneck it finishes
    // frames at an even rate; unbounded, the thread ran ahead and then blocked wherever a
    // resource ran out, so flips (and the guest's frame timing) came in bursts.
    if (const u32 ahead = FramesAhead()) {
        draw_scheduler.WaitPendingFrame();
        recent_frame_ticks.push_back(frame->ready_tick);
        if (recent_frame_ticks.size() > ahead) {
            const u64 tick = recent_frame_ticks.front();
            recent_frame_ticks.pop_front();
            draw_scheduler.DeferFrameWait(tick);
        }
    }
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Presented as the black bars alone: nothing to render. A blank flip of the guest still
    // ends its submission, as any flip does: the guest may wait for work recorded before it.
    if (!present_thread) {
        g_last_completed_frame_draws.store(0, std::memory_order_relaxed);
        if (rasterizer && rasterizer->GetUpscaler().GetFrameGeneration()) {
            rasterizer->GetUpscaler().GetFrameGeneration()->ResetHistory();
        }
        draw_scheduler.EndRendering();
        SubmitInfo info{};
        draw_scheduler.Flush(info);
    }
    return &blank_frame;
}

void Presenter::WaitRendered(const Frame* frame) {
    if (!frame || !frame->ready_semaphore) {
        return;
    }
    const vk::SemaphoreWaitInfo info{
        .semaphoreCount = 1,
        .pSemaphores = &frame->ready_semaphore,
        .pValues = &frame->ready_tick,
    };
    // A lost device ends the wait (the presenter reports it); a timeout waits again.
    while (instance.GetDevice().waitSemaphores(&info, 1'000'000'000) == vk::Result::eTimeout) {
    }
}

void Presenter::SetExpectedSize(u32 width, u32 height) {
    if (width == 0 || height == 0) {
        return; // minimised: keep the last size (a 0x0 frame image cannot be made, issue #31)
    }
    const float ratio = expected_ratio.load(std::memory_order_relaxed);
    if (float(width) / float(height) > ratio) {
        width = u32(float(height) * ratio);
    } else {
        height = u32(float(width) / ratio);
    }
    expected_width.store(width, std::memory_order_relaxed);
    expected_height.store(height, std::memory_order_relaxed);
}

void Presenter::EncodeInterpolated(Frame* frame, u32 slot, vk::Image image, vk::Format format,
                                   u32 width, u32 height, vk::Format display_view_format,
                                   const EncodePass::Settings& encode, bool fused_gamma,
                                   vk::Format fused_src_format,
                                   vk::ComponentMapping fused_src_mapping, vk::ImageView lut1d_view,
                                   vk::ImageLayout lut1d_layout) {
    (void)width;
    (void)height;
    if (!frame || slot >= Frame::MaxInterpSlots) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    if (!frame->interp_images[slot]) {
        const vk::Format interp_fmt =
            frame->is_hdr ? vk::Format::eA2B10G10R10UnormPack32 : swapchain.GetSurfaceFormat().format;
        const VkImageCreateInfo image_info = vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = interp_fmt,
            .extent = {frame->width, frame->height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        };
        const VmaAllocationCreateInfo alloc_info{
            .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        VkImage created{};
        if (vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info, &created,
                           &frame->interp_allocations[slot], nullptr) != VK_SUCCESS) {
            return;
        }
        frame->interp_images[slot] = created;
        frame->interp_views[slot] = Check<"create a frame view">(device.createImageView({
            .image = frame->interp_images[slot],
            .viewType = vk::ImageViewType::e2D,
            .format = interp_fmt,
            .subresourceRange = ColorRange,
        }));
    }
    // Match the real frame's exact view format and channel swizzle:
    // - When PassFusion (`fused_gamma`) is active, replicate `ps_38d65b32`'s sampled UNORM view
    //   format + BGRA/RGBA swizzle and apply the exact same 1D Display Gamma LUT inside EncodePass.
    // - Otherwise, view as `display_view_format` (sRGB + BGRA swizzle) just like the display buffer.
    vk::Format view_format = format;
    vk::ComponentMapping mapping{};
    if (fused_gamma) {
        view_format = (fused_src_format != vk::Format::eUndefined) ? fused_src_format : format;
        mapping = fused_src_mapping;
    } else {
        const bool rgba8 =
            format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eB8G8R8A8Unorm;
        const bool display8 = display_view_format == vk::Format::eB8G8R8A8Srgb ||
                              display_view_format == vk::Format::eR8G8B8A8Srgb;
        view_format = rgba8 && display8 ? display_view_format : format;
    }
    mapping.a = vk::ComponentSwizzle::eOne;
    const vk::ImageView source = Check(device.createImageView({
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = view_format,
        .components = mapping,
        .subresourceRange = ColorRange,
    }));
    draw_scheduler.DeferOperation([device, source] { device.destroyImageView(source); });
    const vk::Image dst_image = frame->interp_images[slot];
    const vk::ImageView dst_view = frame->interp_views[slot];
    draw_scheduler.Record([this, frame, dst_image, dst_view, image, source, encode, lut1d_view,
                           lut1d_layout](vk::CommandBuffer cmdbuf) {
        const std::array<vk::ImageMemoryBarrier2, 2> before{{
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eCopy |
                                vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite |
                                 vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = image,
                .subresourceRange = ColorRange,
            },
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                                vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .image = dst_image,
                .subresourceRange = ColorRange,
            },
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = before.data()});
        encode_pass.Render(cmdbuf, source, dst_view, {frame->width, frame->height}, encode,
                           vk::ImageLayout::eShaderReadOnlyOptimal, lut1d_view, lut1d_layout);
        const std::array<vk::ImageMemoryBarrier2, 2> after{{
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = image,
                .subresourceRange = ColorRange,
            },
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = dst_image,
                .subresourceRange = ColorRange,
            },
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = after.data()});
    });
    frame->interp_count = std::max(frame->interp_count, slot + 1u);
}

void Presenter::RecordPresent(vk::CommandBuffer cmdbuf, const Frame* frame, vk::Image target,
                              vk::Extent2D extent, vk::Image source) {
    const vk::Image game_img = frame ? (source ? source : frame->image) : vk::Image{};
    const vk::Extent2D game_ext =
        frame ? vk::Extent2D{frame->width, frame->height} : vk::Extent2D{0, 0};
    const bool use_splash = splash.UpdateAndShouldCover(cmdbuf, game_img, game_ext);
    const vk::ImageMemoryBarrier2 to_transfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = target,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_transfer});
    if (use_splash) {
        splash.RecordBlit(cmdbuf, target, extent);
    } else {
        const vk::Rect2D rect =
            frame ? Letterbox({frame->width, frame->height}, extent) : vk::Rect2D{{0, 0}, {0, 0}};
        const bool bars = rect.extent != extent;
        if (bars) {
            const vk::ClearColorValue black{std::array{0.0f, 0.0f, 0.0f, 1.0f}};
            cmdbuf.clearColorImage(target, vk::ImageLayout::eTransferDstOptimal, black, ColorRange);
        }
        if (frame) {
            if (bars) {
                const vk::MemoryBarrier2 cleared{
                    .srcStageMask = vk::PipelineStageFlagBits2::eClear,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                };
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
            }
            const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            const vk::ImageBlit region{
                .srcSubresource = layers,
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{s32(frame->width), s32(frame->height), 1}},
                .dstSubresource = layers,
                .dstOffsets = std::array{vk::Offset3D{rect.offset.x, rect.offset.y, 0},
                                         vk::Offset3D{rect.offset.x + s32(rect.extent.width),
                                                      rect.offset.y + s32(rect.extent.height), 1}},
            };
            cmdbuf.blitImage(source ? source : frame->image, vk::ImageLayout::eGeneral, target,
                             vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
        }
    }

    // The port's menu and FPS counter over the frame, at display resolution.
    BbGameMenu::Poll();
    const bool overlay = BbOverlay::Visible();
    const vk::ImageMemoryBarrier2 after_transfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = overlay ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                                : vk::PipelineStageFlagBits2::eNone,
        .dstAccessMask = overlay ? vk::AccessFlagBits2::eColorAttachmentRead |
                                       vk::AccessFlagBits2::eColorAttachmentWrite
                                 : vk::AccessFlagBits2::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = overlay ? vk::ImageLayout::eColorAttachmentOptimal
                             : vk::ImageLayout::ePresentSrcKHR,
        .image = target,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &after_transfer});
    if (overlay) {
        BbOverlay::Render(cmdbuf, swapchain.ImageView(), extent);
        const vk::ImageMemoryBarrier2 to_present{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .image = target,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_present});
    }
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool) {
    const auto release = [&] {
        if (!is_reusing_frame) {
            ReleaseFrame(frame);
        }
    };
    const bool blank = frame == &blank_frame;
    // Track smoothed real-frame arrival interval (ns) for unified half-frame FG pacing.
    static u64 last_real_ns = 0;
    static double real_interval_ns = 16'666'667.0;
    static u32 no_interp_streak = 0;
    const u64 now_ns = BbFrameRate::NowMonotonicNs();
    if (last_real_ns != 0 && now_ns > last_real_ns) {
        const double dt_ns = double(now_ns - last_real_ns);
        if (dt_ns > 2'000'000.0 && dt_ns < 100'000'000.0) {
            real_interval_ns = real_interval_ns * 0.9 + dt_ns * 0.1;
        }
    }
    last_real_ns = now_ns;

    const u32 fg_n = (!blank) ? std::min<u32>(frame->interp_count, Frame::MaxInterpSlots) : 0u;
    const bool has_fg = fg_n > 0u && frame->interp_images[0];
    const int mult = has_fg ? int(fg_n + 1u) : 1;
    if (has_fg) {
        no_interp_streak = 0;
        BbStats::fg_active.store(true, std::memory_order_relaxed);
        BbStats::fg_multiplier.store(mult, std::memory_order_relaxed);
        BbStats::fg_backend.store(frame->interp_backend, std::memory_order_relaxed);
    } else if (BbSettings::EffectiveFrameGen() == BbSettings::FrameGenOff ||
               BbFastLoad::IsLoadingBurstActive() || ++no_interp_streak >= 4u) {
        BbStats::fg_active.store(false, std::memory_order_relaxed);
        BbStats::fg_multiplier.store(1, std::memory_order_relaxed);
        BbStats::fg_backend.store(0, std::memory_order_relaxed);
    }

    // BB_FRAME_GEN_ONLY=1 (diagnostics): only the interpolated frames are shown.
    static const bool only_interp = std::getenv("BB_FRAME_GEN_ONLY") != nullptr;
    if (only_interp && has_fg) {
        frame->interp_count = 0;
        PresentImage(frame, frame->interp_images[0], true, 0, true);
        release();
        return;
    }
    if (has_fg) {
        frame->interp_count = 0;
        static std::atomic<u64> shown{0};
        if (const u64 n = shown.fetch_add(fg_n, std::memory_order_relaxed) + 1u;
            n == 1 || (n / 1000) != ((n + fg_n - 1u) / 1000)) {
            std::printf("Frame generation: %llu interpolated frames shown (%dx)\n",
                        (unsigned long long)(n + fg_n - 1u), mult);
            std::fflush(stdout);
        }
        const u64 cap_period_ns =
            (BbFrameRate::PresentPacingEnabled() && !BbFastLoad::IsLoadingBurstActive())
                ? BbFrameRate::TargetPeriodNs()
                : 0ull;
        u64 slot_ns = std::clamp<u64>(static_cast<u64>(real_interval_ns / double(mult)),
                                      1'000'000ull, 50'000'000ull);
        if (cap_period_ns > 0) {
            // When hitting cap/mult base (within ~6%), lock strictly to the display cap period so
            // all interpolated and real frames hit the exact monotonic display grid.
            const u64 cap_group_hi = cap_period_ns * u64(mult) + (cap_period_ns / 8ull);
            slot_ns = (static_cast<u64>(real_interval_ns) <= cap_group_hi)
                          ? cap_period_ns
                          : std::max<u64>(cap_period_ns, slot_ns);
        }
        const auto pmode = swapchain.GetPresentMode();
        const bool is_fifo =
            pmode == vk::PresentModeKHR::eFifo || pmode == vk::PresentModeKHR::eFifoRelaxed;
        bool any_interp_shown = false;
        for (u32 slot = 0; slot < fg_n; ++slot) {
            const vk::Image interp_img = frame->interp_images[slot];
            if (!interp_img) {
                continue;
            }
            const bool wait_gpu = !any_interp_shown;
            const bool shown_slot =
                PresentImage(frame, interp_img, false, slot_ns, /*wait_gpu_done=*/wait_gpu);
            if (shown_slot) {
                any_interp_shown = true;
                BbStats::fg_interpolated_frames.fetch_add(1, std::memory_order_relaxed);
                if (is_fifo) {
                    const u64 after_interp_ns = BbFrameRate::NowMonotonicNs();
                    BbFrameRate::SleepUntilMonotonicNs(after_interp_ns + slot_ns, 1);
                }
            }
        }
        PresentImage(frame, {}, true, slot_ns, /*wait_gpu_done=*/!any_interp_shown);
        release();
        return;
    }
    PresentImage(frame, {}, true, 0, true);
    release();
}

bool Presenter::PresentImage(Frame* frame, vk::Image source, bool last, u64 fg_slot_period_ns,
                             bool wait_gpu_done) {
    const u32 width = u32(window.GetWidth());
    const u32 height = u32(window.GetHeight());
    // Nothing to present to while the window is minimised (a 0x0 surface, issue #31).
    if (width == 0 || height == 0) {
        return false;
    }
    // Recreate waits for the device. Recording threads must not submit to its queues meanwhile
    // (#211); the recreation after a failed present below already uses this same mutex.
    const auto recreate = [&] {
        std::scoped_lock lock{Scheduler::submit_mutex};
        swapchain.Recreate(width, height);
    };
    // Live SDR <-> HDR10 swapchain switch from System -> Effects / Overlay.
    const bool want_hdr =
        swapchain.HasHDR() &&
        (guest_hdr.load(std::memory_order_relaxed) ||
         BbSettings::Get().hdr_output.load(std::memory_order_relaxed));
    if (want_hdr != swapchain.GetHDR()) {
        SetHDR(guest_hdr.load(std::memory_order_relaxed));
        if (!swapchain.IsPresentable()) {
            return false;
        }
    }
    // A resized window, a live VSync change, or a previously minimised surface needs a new swapchain.
    const u32 sc_gen = BbSettings::Get().swapchain_generation.load(std::memory_order_relaxed);
    if (width != swapchain.GetWidth() || height != swapchain.GetHeight() ||
        sc_gen != last_swapchain_generation || !swapchain.IsPresentable()) {
        last_swapchain_generation = sc_gen;
        recreate();
        if (!swapchain.IsPresentable()) {
            return false;
        }
    }
    const bool blank = frame == &blank_frame;
    if (!blank && frame->is_hdr != swapchain.GetHDR()) {
        // Skip the single in-flight frame rendered in the old color space to avoid a 1-frame flash.
        return false;
    }
    const u64 enter_ns = BbFrameRate::NowMonotonicNs();
    static u64 pres_deadline_ns = 0;
    static u64 pres_last_period_ns = 0;
    static u64 pres_seen_epoch = 0;
    static u64 prev_rec_end_ns = 0;
    static u64 prev_pres_end_ns = 0;
    static u64 prev_flush_ns = 0;
    static u64 prev_pres_call_ns = 0;
    bool pres_paced = false;
    u64 active_period_ns = 0;
    u64 wsub_ns = 0, wren_ns = 0, pace_ns = 0, pace_over_ns = 0, cushion_before_ns = 0;
    u32 queued_before = 0;
    const bool pace_slot = !blank && (fg_slot_period_ns > 0 || (last && !source));
    if (pace_slot) {
#ifdef __linux__
        static bool pres_timerslack_set = false;
        if (!pres_timerslack_set) {
            pres_timerslack_set = true;
            prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
        }
#endif
        const u64 period_ns =
            (fg_slot_period_ns > 0)
                ? fg_slot_period_ns
                : ((BbFrameRate::PresentPacingEnabled() && !BbFastLoad::IsLoadingBurstActive())
                       ? BbFrameRate::TargetPeriodNs()
                       : 0ull);
        const auto pmode = swapchain.GetPresentMode();
        const bool is_fifo =
            pmode == vk::PresentModeKHR::eFifo || pmode == vk::PresentModeKHR::eFifoRelaxed;
        if (is_fifo && fg_slot_period_ns > 0 && wait_gpu_done) {
            // Under VSync FIFO with Frame Generation, wait for the GPU pair to finish before
            // presenting the interpolated frame so the half-frame sleep starts at true completion.
            if (frame->ready_semaphore == draw_scheduler.GetWorkSemaphore()->Handle()) {
                draw_scheduler.WaitSubmitted(frame->ready_tick);
            }
            WaitRendered(frame);
        }
        if (period_ns > 0 && !is_fifo) {
            active_period_ns = period_ns;
            pres_paced = true;
            const u64 ws_t0 = BbFrameRate::NowMonotonicNs();
            if (wait_gpu_done) {
                if (frame->ready_semaphore == draw_scheduler.GetWorkSemaphore()->Handle()) {
                    draw_scheduler.WaitSubmitted(frame->ready_tick);
                }
                const u64 wr_t0 = BbFrameRate::NowMonotonicNs();
                wsub_ns = wr_t0 > ws_t0 ? (wr_t0 - ws_t0) : 0ull;
                WaitRendered(frame);
                const u64 ready_done_ns = BbFrameRate::NowMonotonicNs();
                wren_ns = ready_done_ns > wr_t0 ? (ready_done_ns - wr_t0) : 0ull;
            }
            const u64 ready_ns = BbFrameRate::NowMonotonicNs();
            const u64 epoch = BbFrameRate::PacerEpoch();
            const u64 target_cushion_ns = std::min<u64>(period_ns / 2ull, 2'400'000ull);
            queued_before = BbVideoOut::QueuedPresentCount();
            const bool period_changed =
                pres_last_period_ns == 0 ||
                (fg_slot_period_ns == 0
                     ? (period_ns != pres_last_period_ns)
                     : (std::max(period_ns, pres_last_period_ns) -
                            std::min(period_ns, pres_last_period_ns) >
                        period_ns / 5ull));
            if (pres_deadline_ns == 0 || epoch != pres_seen_epoch || period_changed) {
                pres_seen_epoch = epoch;
                pres_last_period_ns = period_ns;
                // Prime with half target cushion so normal camera-turn GPU swings do not break lock.
                pres_deadline_ns = ready_ns + (target_cushion_ns / 2ull);
            } else {
                pres_last_period_ns = period_ns;
                // Smooth queue drain: having 1 frame queued in present_work is normal under
                // FramesAhead=1, so trim ~3% normally (0.969x) or ~5% (0.95x) when the present
                // cushion has grown past target_cushion_ns after an external compositor stall,
                // draining backlog smoothly in ~10 frames without visible cadence sawtooth.
                const bool high_cush = (pres_deadline_ns > ready_ns + target_cushion_ns);
                const u64 step_ns =
                    (queued_before >= 2u)
                        ? (period_ns - (period_ns / 10ull))
                        : ((queued_before == 1u && fg_slot_period_ns == 0)
                               ? (period_ns - (period_ns / (high_cush ? 20ull : 32ull)))
                               : period_ns);
                pres_deadline_ns += step_ns;
                if (ready_ns >= pres_deadline_ns) {
                    pres_deadline_ns = ready_ns;
                } else {
                    if (pres_deadline_ns > ready_ns + period_ns) {
                        pres_deadline_ns = ready_ns + period_ns;
                    }
                    const u64 cushion_ns = pres_deadline_ns - ready_ns;
                    cushion_before_ns = cushion_ns;
                    if (cushion_ns > target_cushion_ns && fg_slot_period_ns == 0) {
                        const u64 pull =
                            std::min<u64>((cushion_ns - target_cushion_ns) / 64ull, 15'000ull);
                        pres_deadline_ns -= pull;
                    }
                    BbFrameRate::SleepUntilMonotonicNs(pres_deadline_ns, 1);
                    const u64 after_ns = BbFrameRate::NowMonotonicNs();
                    if (after_ns > ready_ns) {
                        pace_ns = after_ns - ready_ns;
                        BbStats::present_pace_sleep_ns.fetch_add(
                            pace_ns, std::memory_order_relaxed);
                    }
                    if (after_ns > pres_deadline_ns) {
                        pace_over_ns = after_ns - pres_deadline_ns;
                    }
                }
            }
        }
    }
    const u64 acq_t0 = BbFrameRate::NowMonotonicNs();
    if (!swapchain.AcquireNextImage()) {
        recreate();
        if (!swapchain.AcquireNextImage()) {
            // Resized faster than the swapchain can follow: this frame is skipped.
            LOG_WARNING(Render_Vulkan, "Skipping a frame: no swapchain image");
            return false;
        }
    }
    const u64 acq_t1 = BbFrameRate::NowMonotonicNs();
    const u64 acq_ns = acq_t1 > acq_t0 ? (acq_t1 - acq_t0) : 0ull;
    if (acq_ns > 0) {
        BbStats::swapchain_acquire_wait_ns.fetch_add(acq_ns, std::memory_order_relaxed);
    }

    const vk::Extent2D extent = swapchain.GetExtent();
    SetExpectedSize(extent.width, extent.height);
    // Reset here, not when the frame is taken: a frame skipped above keeps its fence signaled.
    if (!blank && last) {
        Check<"reset a frame fence">(instance.GetDevice().resetFences(frame->present_done));
    }

    const auto cmdbuf = present_scheduler.CommandBuffer();
    if (!blank && last) {
        DumpFinalFrameIfDue(instance, present_scheduler, cmdbuf, frame->image, frame->width,
                            frame->height, swapchain.GetSurfaceFormat().format);
    }
    // BB_SEQUENCE_DUMP_TRIGGER=<file> (frame generation checks): the next 16 images shown, the
    // interpolated ones ("interp") and the game's ("real") in the order they are presented.
    {
        static const char* trigger = std::getenv("BB_SEQUENCE_DUMP_TRIGGER");
        static int left = 0, index = 0;
        if (trigger && left == 0 && std::remove(trigger) == 0) {
            left = 16;
            index = 0;
        }
        if (left > 0 && !blank) {
            --left;
            DumpPresented(instance, present_scheduler, cmdbuf, source ? source : frame->image,
                          frame->width, frame->height, swapchain.GetSurfaceFormat().format, index++,
                          source ? "interp" : "real");
        }
    }
    const u64 rec_t0 = BbFrameRate::NowMonotonicNs();
    RecordPresent(cmdbuf, blank ? nullptr : frame, swapchain.Image(), extent, source);
    const u64 rec_t1 = BbFrameRate::NowMonotonicNs();
    const u64 rec_ns = rec_t1 > rec_t0 ? (rec_t1 - rec_t0) : 0ull;

    // Re-anchor pres_deadline_ns if sleep overslept or acquire/record stalled so the next frame
    // never fires 0.2..0.4 ms later as an unpaced micro-frame.
    if (pres_paced) {
        static u64 prev_pres_draws = 0;
        static double pres_draws_ema = 0.0;
        static double pres_dt_ema_ns = 0.0;
        const u64 cur_draws = BbStats::draws.load(std::memory_order_relaxed);
        const u64 delta_draws = cur_draws >= prev_pres_draws ? (cur_draws - prev_pres_draws) : 0ull;
        prev_pres_draws = cur_draws;
        pres_draws_ema = pres_draws_ema * 0.9 + double(delta_draws) * 0.1;

        const u64 slip_limit_ns = std::min<u64>(1'500'000ull, active_period_ns / 4ull);
        if (rec_t1 > pres_deadline_ns + slip_limit_ns) {
            pres_deadline_ns = rec_t1;
        }
        if (prev_rec_end_ns != 0 && rec_t1 > prev_rec_end_ns && pres_draws_ema >= 80.0) {
            const u64 dt_ns = rec_t1 - prev_rec_end_ns;
            if (pres_dt_ema_ns <= 0.0) {
                pres_dt_ema_ns = double(std::clamp<u64>(dt_ns, active_period_ns, 50'000'000ull));
            }
            const u64 base_period_ns =
                std::max<u64>(active_period_ns, static_cast<u64>(pres_dt_ema_ns));
            const u64 hi_thresh_ns = std::max<u64>(9'500'000ull, (base_period_ns * 16ull) / 10ull);
            const u64 lo_thresh_ns = (active_period_ns * 6ull) / 10ull;
            if (dt_ns < 2'000'000'000ull && (dt_ns >= hi_thresh_ns || dt_ns <= lo_thresh_ns)) [[unlikely]] {
                const u64 wait_q_ns =
                    (prev_pres_end_ns != 0 && enter_ns > prev_pres_end_ns)
                        ? (enter_ns - prev_pres_end_ns)
                        : 0ull;
                const u64 nvml_ns = BbStats::last_nvml_poll_ns.load(std::memory_order_relaxed);
                const u64 nvml_dur_ns = BbStats::last_nvml_poll_dur_ns.load(std::memory_order_relaxed);
                const long long nvml_age_ms =
                    (nvml_ns != 0 && rec_t1 >= nvml_ns)
                        ? static_cast<long long>((rec_t1 - nvml_ns) / 1'000'000ull)
                        : -1ll;
                const u64 main_dt_ns = BbStats::last_main_frame_dt_ns.load(std::memory_order_relaxed);
                const u64 gnm_dt_ns = BbStats::last_gnm_flip_dt_ns.load(std::memory_order_relaxed);
                const u64 sb_dt_ns = BbStats::last_stageb_flip_dt_ns.load(std::memory_order_relaxed);
                BbStats::LogHitch(
                    "[BB-HITCH][PRESENT] frame=%u dt=%.2f ms (prev_flush=%.2f prev_pres=%.2f "
                    "wait_q=%.2f | w_sub=%.2f w_ren=%.2f pace=%.2f [over=+%.2f] acq=%.2f "
                    "rec=%.2f | q=%u cush=%.2f | pipe[main=%.2f gnm=%.2f sb=%.2f] "
                    "nvml=%lldms[%.2fms])\n",
                    BbStats::frame_number.load(std::memory_order_relaxed), double(dt_ns) / 1e6,
                    double(prev_flush_ns) / 1e6, double(prev_pres_call_ns) / 1e6,
                    double(wait_q_ns) / 1e6, double(wsub_ns) / 1e6, double(wren_ns) / 1e6,
                    double(pace_ns) / 1e6, double(pace_over_ns) / 1e6, double(acq_ns) / 1e6,
                    double(rec_ns) / 1e6, queued_before, double(cushion_before_ns) / 1e6,
                    double(main_dt_ns) / 1e6, double(gnm_dt_ns) / 1e6, double(sb_dt_ns) / 1e6,
                    nvml_age_ms, double(nvml_dur_ns) / 1e6);
            }
            if (dt_ns < 200'000'000ull) {
                const double clamped_dt =
                    double(std::clamp<u64>(dt_ns, active_period_ns, 50'000'000ull));
                pres_dt_ema_ns = pres_dt_ema_ns * 0.94 + clamped_dt * 0.06;
            }
        }
        prev_rec_end_ns = rec_t1;
    }

    // The frame's submission goes out from a recording thread (BB_ASYNC_SUBMIT); the present
    // one, waiting for it on the same queue, must not get there first.
    const u64 fl_t0 = BbFrameRate::NowMonotonicNs();
    if (!blank && frame->ready_semaphore == draw_scheduler.GetWorkSemaphore()->Handle()) {
        draw_scheduler.WaitSubmitted(frame->ready_tick);
    }
    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    if (!blank) {
        info.AddWait(frame->ready_semaphore, frame->ready_tick);
        if (last) {
            info.AddSignal(frame->present_done);
            info.present_id = frame->present_id;
        }
    }
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    present_scheduler.Flush(info);
    const u64 fl_t1 = BbFrameRate::NowMonotonicNs();
    prev_flush_ns = fl_t1 > fl_t0 ? (fl_t1 - fl_t0) : 0ull;
    const u64 ll_present_id = (!blank && last) ? frame->present_id : 0ull;
    {
        const u64 pres_t0 = fl_t1;
        std::scoped_lock lock{Scheduler::submit_mutex};
        if (ll_present_id != 0) {
            LowLatency::MarkPresentStart(ll_present_id);
        }
        const bool ok = swapchain.Present();
        if (ll_present_id != 0) {
            LowLatency::MarkPresentEnd(ll_present_id);
        }
        const u64 pres_t1 = BbFrameRate::NowMonotonicNs();
        const u64 pres_ns = pres_t1 > pres_t0 ? (pres_t1 - pres_t0) : 0ull;
        prev_pres_call_ns = pres_ns;
        prev_pres_end_ns = pres_t1;
        if (pres_ns > 0) {
            BbStats::swapchain_present_wait_ns.fetch_add(pres_ns, std::memory_order_relaxed);
        }
        if (!ok) {
            swapchain.Recreate(width, height);
            return false;
        }
    }
    if (ll_present_id != 0) {
        LowLatency::PollTimingsIfDue();
    }
    return true;
}

} // namespace Vulkan

extern "C" float bbgpu_frame_aspect(void) {
    return Vulkan::shown_aspect.load(std::memory_order_relaxed);
}
