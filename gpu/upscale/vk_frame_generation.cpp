// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_frame_generation.h"
#include "bbport_toggles.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <vk_mem_alloc.h>

#include "bbport_settings.h"
#include "bbport_system_info.h"
#include "ffx_vk_portable.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "upscale/vk_dlss.h"
#include "vulkan/instance.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

FfxVkPortableImage Describe(vk::Image image, vk::Format format, u32 width, u32 height,
                            vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
                            FfxVkPortableResourceState state) {
    FfxVkPortableImage out{};
    out.structSize = sizeof(out);
    out.image = image;
    out.format = static_cast<VkFormat>(format);
    out.extent = {width, height};
    out.mipCount = 1;
    out.arrayLayers = 1;
    out.usage = static_cast<VkImageUsageFlags>(usage);
    out.aspect = static_cast<VkImageAspectFlags>(aspect);
    out.state = state;
    return out;
}

/// The output-size images the frame generation reads and writes: the scene's image (the UI
/// image of the upscaler), the HUD-less copy and the outputs.
constexpr vk::ImageUsageFlags ColorUsage = vk::ImageUsageFlagBits::eStorage |
                                           vk::ImageUsageFlagBits::eSampled |
                                           vk::ImageUsageFlagBits::eTransferSrc |
                                           vk::ImageUsageFlagBits::eTransferDst;

vk::ImageAspectFlags DepthAspectMask(vk::Format format) noexcept {
    const bool has_stencil = format == vk::Format::eD32SfloatS8Uint ||
                             format == vk::Format::eD24UnormS8Uint;
    return vk::ImageAspectFlagBits::eDepth |
           (has_stencil ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlags{});
}

void ImageBarrier(vk::CommandBuffer cmdbuf, vk::Image image, vk::PipelineStageFlags2 src_stage,
                  vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                  vk::AccessFlags2 dst_access, vk::ImageLayout old_layout = vk::ImageLayout::eGeneral,
                  vk::ImageLayout new_layout = vk::ImageLayout::eGeneral,
                  vk::ImageSubresourceRange range = ColorRange) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

bool FiniteVec3(const float v[3]) noexcept {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool ValidCameraInputs(const FrameGeneration::Scene& s) noexcept {
    return std::isfinite(s.camera_near) && s.camera_near > 0.0f &&
           std::isfinite(s.camera_far) && s.camera_far > s.camera_near &&
           std::isfinite(s.vertical_fov) && s.vertical_fov > 0.05f && s.vertical_fov < 3.0f &&
           FiniteVec3(s.camera_position) && FiniteVec3(s.camera_up) &&
           FiniteVec3(s.camera_right) && FiniteVec3(s.camera_forward);
}

} // namespace

namespace {
std::atomic<u64> stat_captured{0}, stat_interpolated{0}, stat_ready{0}, stat_taken{0}, stat_missed{0},
    stat_hudless{0};
void Stats() {
    static std::atomic<u64> calls{0};
    if (++calls % 2000 == 0 && stat_captured.load(std::memory_order_relaxed) != 0) {
        std::printf("Frame generation stats: captured %llu interpolated %llu hudless %llu "
                    "ready %llu taken %llu missed %llu\n",
                    (unsigned long long)stat_captured.load(),
                    (unsigned long long)stat_interpolated.load(),
                    (unsigned long long)stat_hudless.load(),
                    (unsigned long long)stat_ready.load(),
                    (unsigned long long)stat_taken.load(),
                    (unsigned long long)stat_missed.load());
        std::fflush(stdout);
    }
}
} // namespace

FrameGeneration::FrameGeneration(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    FfxVkPortableDeviceCapabilities caps{};
    caps.structSize = sizeof(caps);
    auto& settings = BbSettings::Get();
    if (ffxVkPortableQueryDeviceCapabilities(instance.GetPhysicalDevice(), &caps) !=
        FFX_VK_PORTABLE_OK) {
        settings.frame_generation_problem = "the GPU's capabilities could not be read";
    } else if (!caps.fsr3FrameGenerationPrerequisites) {
        settings.frame_generation_problem = "the GPU lacks what FSR 3 frame generation needs";
    } else {
        supported = true;
    }
    settings.frame_generation_supported = supported;
    std::printf("Frame generation: FSR 3.1 %s%s%s\n", supported ? "available" : "unavailable",
                supported ? "" : ": ", supported ? "" : settings.frame_generation_problem.load());
    std::fflush(stdout);
}

FrameGeneration::~FrameGeneration() {
    Destroy();
}

bool FrameGeneration::UseDlssg() const {
    const int mode = BbSettings::EffectiveFrameGen();
    if (!BbSettings::IsDlssFrameGen(mode) || dlssg_failed.load(std::memory_order_relaxed)) {
        return false;
    }
    const Dlss* dlss = Dlss::Get();
    return dlss && dlss->FgAvailable();
}

bool FrameGeneration::Wanted() const {
    const int mode = BbSettings::EffectiveFrameGen();
    if (mode == BbSettings::FrameGenOff) {
        return false;
    }
    if (UseDlssg()) {
        return true;
    }
    return supported && !failed;
}

void FrameGeneration::Destroy() {
    bool finished = false;
    if (context) {
        scheduler.Finish();
        finished = true;
        ffxVkPortableFrameGenerationContextDestroy(context);
        context = nullptr;
    }
    if (Dlss* dlss = Dlss::Get()) {
        if (!finished) {
            scheduler.Finish();
            finished = true;
        }
        dlss->ReleaseFgFeature();
    }
    queued_dlssg_desc.reset();
    depth_copy_view.reset();
    hudless_view.reset();
    for (auto& t : targets) {
        t.view.reset();
        t.ready = false;
    }
    ready_count = 0;
    ready_backend = 0;
}

void FrameGeneration::ResetHistory() {
    scene_captured = false;
    hudless_ready = false;
    reset_next = true;
    std::scoped_lock lock{targets_mutex};
    ready_count = 0;
    ready_backend = 0;
    for (auto& t : targets) {
        t.ready = false;
    }
}

void FrameGeneration::WarmUp(u32 render_width, u32 render_height, u32 width, u32 height,
                             vk::Format format) {
    if (!Wanted() || render_width == 0 || render_height == 0 || width == 0 || height == 0) {
        return;
    }
    if (UseDlssg()) {
        Dlss* dlss = Dlss::Get();
        const Dlss::FgFeatureDesc desc{render_width, render_height, width, height, format, false,
                                       false,        false};
        if (dlss && queued_dlssg_desc != desc) {
            scheduler.EndRendering();
            scheduler.Finish();
            dlss->ReleaseFgFeature();
            queued_dlssg_desc = desc;
            scheduler.Record([this, dlss, desc](vk::CommandBuffer cmd) {
                if (!dlss->HasFgFeature(desc) && !dlss->CreateFgFeature(cmd, desc)) {
                    dlssg_failed.store(true, std::memory_order_relaxed);
                }
            });
        }
        return;
    }
    (void)EnsureContext(render_width, render_height, width, height, format);
}

bool FrameGeneration::EnsureTargetSlot(u32 slot) {
    if (slot >= MaxInterpOutputs) {
        return false;
    }
    Target& target = targets[slot];
    if (target.image && target.view && target.format == scene.color_format &&
        target.width == scene.width && target.height == scene.height) {
        return true;
    }
    scheduler.Finish(); // prior presents/encodes may still read the old image
    const auto device = instance.GetDevice();
    target.view.reset();
    target.image = VideoCore::HostImage(device, instance.GetAllocator());
    target.image.Create(vk::ImageCreateInfo{
        // The presenter may view it as sRGB or with custom swizzle (e.g. fused Display LUT1D).
        .flags = vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage,
        .imageType = vk::ImageType::e2D,
        .format = scene.color_format,
        .extent = {scene.width, scene.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = ColorUsage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    target.view = Check<"create a frame generation view">(device.createImageViewUnique({
        .image = vk::Image(target.image),
        .viewType = vk::ImageViewType::e2D,
        .format = scene.color_format,
        .subresourceRange = ColorRange,
    }));
    target.format = scene.color_format;
    target.width = scene.width;
    target.height = scene.height;
    target.ready = false;
    return true;
}

bool FrameGeneration::EnsureContext(u32 render_width, u32 render_height, u32 width, u32 height,
                                    vk::Format format) {
    if (context && context_render_w == render_width && context_render_h == render_height &&
        context_w == width && context_h == height && context_format == format) {
        return true;
    }
    Destroy();
    const auto device = instance.GetDevice();
    const auto env_on = [](const char* name) {
        const char* value = std::getenv(name);
        return value && value[0] == '1';
    };
    FfxVkPortableDeviceInfo device_info{};
    device_info.structSize = sizeof(device_info);
    device_info.instance = instance.GetInstance();
    device_info.physicalDevice = instance.GetPhysicalDevice();
    device_info.device = device;
    device_info.getDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    device_info.queue = instance.GetGraphicsQueue();
    device_info.queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex();
    device_info.shaderFloat16Enabled = instance.IsShaderFloat16Enabled() && !env_on("BB_FSR3_FP32");
    device_info.subgroupSizeControlEnabled =
        instance.IsSubgroupSizeControlEnabled() && !env_on("BB_FSR3_WAVE32");
    device_info.synchronization2Enabled = VK_TRUE;
    device_info.shaderStorageImageWriteWithoutFormatEnabled = VK_TRUE;

    FfxVkPortableFrameGenerationCreateInfo create{};
    create.structSize = sizeof(create);
    create.maxRenderSize = {render_width, render_height};
    create.displaySize = {width, height};
    create.interpolationSourceFormat = static_cast<VkFormat>(format);
    create.outputFormat = static_cast<VkFormat>(format);
    if (const u64 issues = ffxVkPortableValidateFrameGenerationCreateInfo(&create)) {
        std::printf("Frame generation: invalid context (%s)\n", ffxVkPortableValidationIssueName(issues));
        std::fflush(stdout);
        failed = true;
        return false;
    }
    if (ffxVkPortableFrameGenerationContextCreate(&device_info, &create, &context) !=
        FFX_VK_PORTABLE_OK) {
        std::printf("Frame generation: the context could not be made\n");
        std::fflush(stdout);
        context = nullptr;
        failed = true;
        return false;
    }
    context_render_w = render_width;
    context_render_h = render_height;
    context_w = width;
    context_h = height;
    context_format = format;
    reset_next = true;
    std::printf("Frame generation: FSR 3.1 context %ux%u from %ux%u scenes (%s)\n", width, height,
                render_width, render_height, vk::to_string(format).c_str());
    std::fflush(stdout);
    return true;
}

void FrameGeneration::CaptureScene(const Scene& in) {
    scene_captured = false;
    hudless_ready = false;
    const bool use_dlssg = UseDlssg();
    const int cur_backend = use_dlssg ? 2 : 1;
    if (active_backend != 0 && active_backend != cur_backend) {
        ResetHistory();
    }
    active_backend = cur_backend;

    if (!Wanted() || !in.color || !in.depth || !in.motion || in.render_width == 0 ||
        in.render_height == 0 || in.width == 0 || in.height == 0 || !ValidCameraInputs(in) ||
        (use_dlssg && (!in.color_view || !in.motion_view || !in.dlssg_matrices.valid)) ||
        (!use_dlssg &&
         !EnsureContext(in.render_width, in.render_height, in.width, in.height, in.color_format))) {
        reset_next = true;
        return;
    }
    scene = in;
    scene.frame_ms =
        std::isfinite(in.frame_ms) ? std::clamp(in.frame_ms, 1.0f, 100.0f) : 16.667f;
    scene_captured = true;
    scene_id = ++frame_id;
    hudless_ready = false;
    scene_reset = reset_next || in.reset;
    reset_next = false;
    ++stat_captured;

    // Copy scene depth now before downstream guest passes overwrite or reuse the depth buffer.
    if (!depth_copy || !depth_copy_view || depth_copy_format != in.depth_format ||
        depth_copy_w != in.render_width || depth_copy_h != in.render_height) {
        scheduler.Finish();
        depth_copy_view.reset();
        const auto device = instance.GetDevice();
        depth_copy = VideoCore::HostImage(device, instance.GetAllocator());
        depth_copy.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = in.depth_format,
            .extent = {in.render_width, in.render_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        depth_copy_view = Check<"create a frame generation depth view">(device.createImageViewUnique({
            .image = vk::Image(depth_copy),
            .viewType = vk::ImageViewType::e2D,
            .format = in.depth_format,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        }));
        depth_copy_format = in.depth_format;
        depth_copy_w = in.render_width;
        depth_copy_h = in.render_height;
    }
    scheduler.Record([depth = in.depth, depth_layout = in.depth_layout,
                      depth_fmt = in.depth_format, copy = vk::Image(depth_copy),
                      w = in.render_width, h = in.render_height](vk::CommandBuffer cmdbuf) {
        const vk::ImageAspectFlags barrier_aspect = DepthAspectMask(depth_fmt);
        const vk::ImageSubresourceRange range{barrier_aspect, 0, 1, 0, 1};
        const std::array<vk::ImageMemoryBarrier2, 2> before{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                             vk::PipelineStageFlagBits2::eLateFragmentTests |
                             vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
             .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                              vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
             .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
             .oldLayout = depth_layout,
             .newLayout = vk::ImageLayout::eTransferSrcOptimal,
             .image = depth,
             .subresourceRange = range},
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
             .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eTransferDstOptimal,
             .image = copy,
             .subresourceRange = range},
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = before.data()});
        const vk::ImageCopy region{
            .srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
            .extent = {w, h, 1},
        };
        cmdbuf.copyImage(depth, vk::ImageLayout::eTransferSrcOptimal, copy,
                         vk::ImageLayout::eTransferDstOptimal, region);
        const std::array<vk::ImageMemoryBarrier2, 2> after{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eCopy,
             .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
             .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                             vk::PipelineStageFlagBits2::eLateFragmentTests |
                             vk::PipelineStageFlagBits2::eFragmentShader |
                             vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                              vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
             .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
             .newLayout = depth_layout,
             .image = depth,
             .subresourceRange = range},
            {.srcStageMask = vk::PipelineStageFlagBits2::eCopy,
             .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
             .oldLayout = vk::ImageLayout::eTransferDstOptimal,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = copy,
             .subresourceRange = range},
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = after.data()});
    });
}

void FrameGeneration::CaptureHudless(vk::Image color) {
    if (!scene_captured || color != scene.color || hudless_ready) {
        return;
    }
    if (!hudless || !hudless_view || hudless_format != scene.color_format ||
        hudless_w != scene.width || hudless_h != scene.height) {
        scheduler.Finish();
        hudless_view.reset();
        const auto device = instance.GetDevice();
        hudless = VideoCore::HostImage(device, instance.GetAllocator());
        hudless.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = scene.color_format,
            .extent = {scene.width, scene.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = ColorUsage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        hudless_view = Check<"create a frame generation hudless view">(device.createImageViewUnique({
            .image = vk::Image(hudless),
            .viewType = vk::ImageViewType::e2D,
            .format = scene.color_format,
            .subresourceRange = ColorRange,
        }));
        hudless_format = scene.color_format;
        hudless_w = scene.width;
        hudless_h = scene.height;
    }
    hudless_ready = true;
    scheduler.EndRendering();
    scheduler.Record([color, copy = vk::Image(hudless), w = scene.width,
                      h = scene.height](vk::CommandBuffer cmdbuf) {
        const std::array<vk::ImageMemoryBarrier2, 2> before{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
             .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                              vk::AccessFlagBits2::eShaderWrite |
                              vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
             .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
             .oldLayout = vk::ImageLayout::eGeneral,
             .newLayout = vk::ImageLayout::eTransferSrcOptimal,
             .image = color,
             .subresourceRange = ColorRange},
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
             .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eTransferDstOptimal,
             .image = copy,
             .subresourceRange = ColorRange},
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = before.data()});
        const vk::ImageCopy region{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {w, h, 1},
        };
        cmdbuf.copyImage(color, vk::ImageLayout::eTransferSrcOptimal, copy,
                         vk::ImageLayout::eTransferDstOptimal, region);
        const std::array<vk::ImageMemoryBarrier2, 2> after{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eCopy,
             .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
             .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits2::eFragmentShader |
                             vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                              vk::AccessFlagBits2::eColorAttachmentWrite |
                              vk::AccessFlagBits2::eShaderRead,
             .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = color,
             .subresourceRange = ColorRange},
            {.srcStageMask = vk::PipelineStageFlagBits2::eCopy,
             .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
             .oldLayout = vk::ImageLayout::eTransferDstOptimal,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = copy,
             .subresourceRange = ColorRange},
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = after.data()});
    });
}

void FrameGeneration::Interpolate(vk::Image color, VAddr display) {
    Stats();
    const bool use_dlssg = UseDlssg();
    if (!scene_captured || color != scene.color || (!use_dlssg && !context)) {
        return;
    }
    ++stat_interpolated;
    BbStats::fg_total_dispatches.fetch_add(1, std::memory_order_relaxed);
    scene_captured = false;
    const u64 id = scene_id;
    bool reset = scene_reset;
    (void)display;

    if (use_dlssg) {
        Dlss* dlss = Dlss::Get();
        if (!dlss || !scene.color_view || !scene.motion_view || !depth_copy_view ||
            !scene.dlssg_matrices.valid) {
            ResetHistory();
            return;
        }
        const Dlss::FgFeatureDesc desc{scene.render_width,
                                       scene.render_height,
                                       scene.width,
                                       scene.height,
                                       scene.color_format,
                                       false,
                                       false,
                                       scene.hdr};
        if (queued_dlssg_desc != desc) {
            scheduler.EndRendering();
            scheduler.Finish();
            dlss->ReleaseFgFeature();
            queued_dlssg_desc = desc;
            reset = true;
        }
        const int mode = BbSettings::EffectiveFrameGen();
        const u32 max_gpu_multi = std::max(1u, dlss->FgMaxMultiFrame());
        const u32 multi_count =
            std::clamp<u32>(u32(std::max(1, BbSettings::FrameGenMultiplier(mode) - 1)), 1u,
                            std::min<u32>(MaxInterpOutputs, max_gpu_multi));
        const u32 eval_count = reset ? 1u : multi_count;

        for (u32 slot = 0; slot < eval_count; ++slot) {
            if (!EnsureTargetSlot(slot)) {
                ResetHistory();
                return;
            }
        }
        {
            std::scoped_lock lock{targets_mutex};
            ready_count = reset ? 0u : multi_count;
            ready_backend = 2;
            for (u32 i = 0; i < MaxInterpOutputs; ++i) {
                if (i < ready_count) {
                    targets[i].ready = true;
                    targets[i].frame = id;
                } else {
                    targets[i].ready = false;
                }
            }
            if (ready_count > 0) {
                stat_ready += ready_count;
            }
        }

        const bool use_hudless = hudless_ready && hudless && hudless_view &&
                                 hudless_w == scene.width && hudless_h == scene.height &&
                                 hudless_format == scene.color_format &&
                                 !BbToggle::Disabled(1u << 30);
        if (use_hudless) {
            ++stat_hudless;
            BbStats::fg_hudless_frames.fetch_add(1, std::memory_order_relaxed);
        }
        hudless_ready = false;

        Dlss::FgFrame fg_frame{};
        fg_frame.backbuffer = {color, scene.color_view, scene.color_format,
                               vk::ImageAspectFlagBits::eColor, scene.width, scene.height};
        if (use_hudless) {
            fg_frame.hudless = {vk::Image(hudless), *hudless_view, scene.color_format,
                                vk::ImageAspectFlagBits::eColor, scene.width, scene.height};
        }
        fg_frame.depth = {vk::Image(depth_copy), *depth_copy_view, scene.depth_format,
                          vk::ImageAspectFlagBits::eDepth, scene.render_width,
                          scene.render_height};
        fg_frame.motion = {scene.motion, scene.motion_view, vk::Format::eR16G16Sfloat,
                           vk::ImageAspectFlagBits::eColor, scene.render_width,
                           scene.render_height};
        fg_frame.frame_id = id;
        fg_frame.multi_frame_count = multi_count;
        fg_frame.reset = reset;
        fg_frame.depth_inverted = false;
        fg_frame.hdr = scene.hdr;
        fg_frame.camera_motion_included = true;
        fg_frame.render_subrect_width = scene.render_width;
        fg_frame.render_subrect_height = scene.render_height;
        fg_frame.jitter_x = scene.jitter_x;
        fg_frame.jitter_y = scene.jitter_y;
        fg_frame.mv_scale_x = 1.0f / float(scene.render_width);
        fg_frame.mv_scale_y = 1.0f / float(scene.render_height);
        for (int i = 0; i < 3; ++i) {
            fg_frame.camera_pos[i] = scene.camera_position[i];
            fg_frame.camera_up[i] = scene.camera_up[i];
            fg_frame.camera_right[i] = scene.camera_right[i];
            fg_frame.camera_fwd[i] = scene.camera_forward[i];
        }
        fg_frame.camera_near = scene.camera_near;
        fg_frame.camera_far = scene.camera_far;
        fg_frame.camera_fov = scene.vertical_fov;
        fg_frame.camera_aspect = float(scene.width) / float(scene.height);
        fg_frame.camera_view_to_clip = scene.dlssg_matrices.camera_view_to_clip;
        fg_frame.clip_to_camera_view = scene.dlssg_matrices.clip_to_camera_view;
        fg_frame.clip_to_prev_clip = scene.dlssg_matrices.clip_to_prev_clip;
        fg_frame.prev_clip_to_clip = scene.dlssg_matrices.prev_clip_to_clip;

        std::array<Dlss::Resource, MaxInterpOutputs> out_res{};
        for (u32 slot = 0; slot < eval_count; ++slot) {
            out_res[slot] = {vk::Image(targets[slot].image), *targets[slot].view,
                             scene.color_format, vk::ImageAspectFlagBits::eColor, scene.width,
                             scene.height};
        }
        const vk::ImageAspectFlags depth_barrier_aspect = DepthAspectMask(scene.depth_format);
        const vk::ImageSubresourceRange depth_range{depth_barrier_aspect, 0, 1, 0, 1};

        scheduler.EndRendering();
        GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Aa_Upscaler);
        if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(&scheduler)) {
            profiler->Mark(0xD155'0001ull, [use_hudless, multi_count] {
                char buf[96];
                std::snprintf(buf, sizeof(buf),
                              "DLSS-G %ux Frame Generation [%s] [VULKAN-NATIVE]",
                              multi_count + 1u, use_hudless ? "HUDless" : "FullFrame");
                return std::string{buf};
            });
        }
        const u32 out_w = scene.width;
        const u32 out_h = scene.height;
        scheduler.Record([this, dlss, desc, fg_frame, out_res, eval_count, color,
                          depth_img = vk::Image(depth_copy), depth_range,
                          motion_img = scene.motion,
                          hudless_img = use_hudless ? vk::Image(hudless) : vk::Image{},
                          out_w, out_h](vk::CommandBuffer cmdbuf) mutable {
            ImageBarrier(cmdbuf, color,
                         vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eTransferWrite,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
                         vk::ImageLayout::eShaderReadOnlyOptimal);
            if (hudless_img) {
                ImageBarrier(cmdbuf, hudless_img,
                             vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eTransferWrite |
                                 vk::AccessFlagBits2::eShaderRead,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
                             vk::ImageLayout::eShaderReadOnlyOptimal);
            }
            ImageBarrier(cmdbuf, depth_img,
                         vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderRead,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
                         vk::ImageLayout::eShaderReadOnlyOptimal, depth_range);
            ImageBarrier(cmdbuf, motion_img,
                         vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
                         vk::ImageLayout::eShaderReadOnlyOptimal);
            for (u32 slot = 0; slot < eval_count; ++slot) {
                ImageBarrier(cmdbuf, out_res[slot].image,
                             vk::PipelineStageFlagBits2::eFragmentShader |
                                 vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eNone,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderStorageWrite,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
            }

            bool ok = dlss->HasFgFeature(desc) || dlss->CreateFgFeature(cmdbuf, desc);
            if (ok) {
                for (u32 idx = 1; idx <= eval_count; ++idx) {
                    fg_frame.multi_frame_index = idx;
                    fg_frame.output = out_res[idx - 1];
                    if (!dlss->EvaluateFg(cmdbuf, fg_frame)) {
                        ok = false;
                        break;
                    }
                }
            }

            ImageBarrier(cmdbuf, color, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead,
                         vk::PipelineStageFlagBits2::eFragmentShader |
                             vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eShaderRead |
                             vk::AccessFlagBits2::eColorAttachmentRead |
                             vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eTransferRead,
                         vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
            if (hudless_img) {
                ImageBarrier(cmdbuf, hudless_img, vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderRead,
                             vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eNone,
                             vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral);
            }
            ImageBarrier(cmdbuf, depth_img, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead,
                         vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eNone, vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::ImageLayout::eGeneral, depth_range);
            ImageBarrier(cmdbuf, motion_img, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eNone, vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::ImageLayout::eGeneral);
            for (u32 slot = 0; slot < eval_count; ++slot) {
                ImageBarrier(cmdbuf, out_res[slot].image,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderStorageWrite,
                             vk::PipelineStageFlagBits2::eFragmentShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
            }

            if (!ok) {
                for (u32 slot = 0; slot < eval_count; ++slot) {
                    const vk::Image out_img = out_res[slot].image;
                    ImageBarrier(cmdbuf, color, vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::AccessFlagBits2::eShaderRead,
                                 vk::PipelineStageFlagBits2::eCopy,
                                 vk::AccessFlagBits2::eTransferRead);
                    ImageBarrier(cmdbuf, out_img, vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::AccessFlagBits2::eShaderStorageWrite,
                                 vk::PipelineStageFlagBits2::eCopy,
                                 vk::AccessFlagBits2::eTransferWrite);
                    cmdbuf.copyImage(
                        color, vk::ImageLayout::eGeneral, out_img, vk::ImageLayout::eGeneral,
                        vk::ImageCopy{.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                      .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                      .extent = {out_w, out_h, 1}});
                    ImageBarrier(cmdbuf, out_img, vk::PipelineStageFlagBits2::eCopy,
                                 vk::AccessFlagBits2::eTransferWrite,
                                 vk::PipelineStageFlagBits2::eFragmentShader |
                                     vk::PipelineStageFlagBits2::eTransfer,
                                 vk::AccessFlagBits2::eShaderRead |
                                     vk::AccessFlagBits2::eTransferRead);
                }
                dlssg_failed.store(true, std::memory_order_release);
                reset_next = true;
                const int fallback =
                    supported ? BbSettings::FrameGenFsr3 : BbSettings::FrameGenOff;
                BbSettings::Get().frame_generation = fallback;
                std::printf("FrameGen: DLSS-G failed; falling back to %s\n",
                            BbSettings::FrameGenLabel(fallback));
                std::fflush(stdout);
                BbSysInfo::LogSettingsIfChanged();
            }
        });
        return;
    }

    const int mode = BbSettings::EffectiveFrameGen();
    const u32 multi_count = std::clamp<u32>(
        u32(std::max(1, BbSettings::FrameGenMultiplier(mode) - 1)), 1u, MaxInterpOutputs);
    const u32 eval_count = reset ? 1u : multi_count;

    for (u32 slot = 0; slot < eval_count; ++slot) {
        if (!EnsureTargetSlot(slot)) {
            ResetHistory();
            return;
        }
    }
    {
        std::scoped_lock lock{targets_mutex};
        ready_count = reset ? 0u : multi_count;
        ready_backend = 1;
        for (u32 i = 0; i < MaxInterpOutputs; ++i) {
            if (i < ready_count) {
                targets[i].ready = true;
                targets[i].frame = id;
            } else {
                targets[i].ready = false;
            }
        }
        if (ready_count > 0) {
            stat_ready += ready_count;
        }
    }

    std::array<vk::Image, MaxInterpOutputs> out_imgs{};
    std::array<FfxVkPortableImage, MaxInterpOutputs> out_descs{};
    std::array<float, MaxInterpOutputs> out_times{};
    for (u32 slot = 0; slot < eval_count; ++slot) {
        out_imgs[slot] = vk::Image(targets[slot].image);
        out_descs[slot] = Describe(out_imgs[slot], scene.color_format, scene.width, scene.height,
                                   ColorUsage, vk::ImageAspectFlagBits::eColor,
                                   FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS);
        out_times[slot] = reset ? 0.5f : float(slot + 1u) / float(multi_count + 1u);
    }

    FfxVkPortableFrameGenerationDispatchInfo dispatch{};
    dispatch.structSize = sizeof(dispatch);
    dispatch.currentColor = Describe(color, scene.color_format, scene.width, scene.height,
                                     ColorUsage | vk::ImageUsageFlagBits::eColorAttachment,
                                     vk::ImageAspectFlagBits::eColor,
                                     FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    // The UI-less frame (CaptureHudless, snapped strictly after NativePostFX + post-Yebis Glow and
    // right before the first Scaleform HUD draw): FSR interpolates it and composites HUD cleanly.
    const bool use_hudless =
        hudless_ready && hudless && hudless_w == scene.width && hudless_h == scene.height &&
        hudless_format == scene.color_format && !BbToggle::Disabled(1u << 30);
    if (use_hudless) {
        ++stat_hudless;
        BbStats::fg_hudless_frames.fetch_add(1, std::memory_order_relaxed);
        dispatch.hudlessColor = Describe(vk::Image(hudless), scene.color_format, scene.width,
                                         scene.height, ColorUsage, vk::ImageAspectFlagBits::eColor,
                                         FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    } else {
        dispatch.hudlessColor.structSize = sizeof(dispatch.hudlessColor); // none: described, null
    }
    hudless_ready = false;
    dispatch.distortionField.structSize = sizeof(dispatch.distortionField); // absent: described, null
    dispatch.output = out_descs[0];
    dispatch.displaySize = {scene.width, scene.height};
    dispatch.interpolationRect = {0, 0, scene.width, scene.height};
    dispatch.frameTimeMilliseconds = scene.frame_ms;
    dispatch.cameraNear = scene.camera_near;
    dispatch.cameraFar = scene.camera_far;
    dispatch.cameraVerticalFovRadians = scene.vertical_fov;
    dispatch.viewSpaceToMeters = 1.0f;
    dispatch.minLuminance = 0.0f;
    dispatch.maxLuminance = 1.0f;
    dispatch.transferFunction = FFX_VK_PORTABLE_TRANSFER_FUNCTION_SRGB;
    dispatch.reset = reset ? VK_TRUE : VK_FALSE;
    dispatch.frameId = id;

    {
        // Checked once with a placeholder command buffer (the real one is the recording thread's).
        static bool checked = false;
        if (!checked) {
            checked = true;
            FfxVkPortableFrameGenerationCreateInfo create{};
            create.structSize = sizeof(create);
            create.maxRenderSize = {context_render_w, context_render_h};
            create.displaySize = {context_w, context_h};
            create.interpolationSourceFormat = static_cast<VkFormat>(context_format);
            create.outputFormat = static_cast<VkFormat>(context_format);
            auto d = dispatch;
            d.commandBuffer = reinterpret_cast<VkCommandBuffer>(uintptr_t{1});
            if (const u64 issues = ffxVkPortableValidateFrameGenerationDispatchInfo(&create, &d)) {
                std::printf("Frame generation: dispatch validation: %s\n",
                            ffxVkPortableValidationIssueName(issues));
                std::fflush(stdout);
            }
        }
    }
    FfxVkPortableFrameGenerationPrepareInfo prepare{};
    prepare.structSize = sizeof(prepare);
    prepare.depth = Describe(vk::Image(depth_copy), scene.depth_format, scene.render_width,
                             scene.render_height, vk::ImageUsageFlagBits::eSampled,
                             vk::ImageAspectFlagBits::eDepth, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    prepare.motionVectors = Describe(scene.motion, vk::Format::eR16G16Sfloat, scene.render_width,
                                     scene.render_height,
                                     vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                     vk::ImageAspectFlagBits::eColor,
                                     FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    prepare.renderSize = {scene.render_width, scene.render_height};
    prepare.jitterOffset = {scene.jitter_x, scene.jitter_y};
    prepare.motionVectorScale = {1.0f, 1.0f};
    prepare.frameTimeMilliseconds = scene.frame_ms;
    prepare.cameraNear = scene.camera_near;
    prepare.cameraFar = scene.camera_far;
    prepare.cameraVerticalFovRadians = scene.vertical_fov;
    prepare.viewSpaceToMeters = 1.0f;
    prepare.minLuminance = 0.0f;
    prepare.maxLuminance = 1.0f;
    prepare.transferFunction = FFX_VK_PORTABLE_TRANSFER_FUNCTION_SRGB;
    prepare.cameraPosition = {scene.camera_position[0], scene.camera_position[1], scene.camera_position[2]};
    prepare.cameraUp = {scene.camera_up[0], scene.camera_up[1], scene.camera_up[2]};
    prepare.cameraRight = {scene.camera_right[0], scene.camera_right[1], scene.camera_right[2]};
    prepare.cameraForward = {scene.camera_forward[0], scene.camera_forward[1], scene.camera_forward[2]};
    prepare.reset = reset ? VK_TRUE : VK_FALSE;
    prepare.frameId = id;
    const FfxVkPortableImage source =
        dispatch.hudlessColor.image != VK_NULL_HANDLE ? dispatch.hudlessColor : dispatch.currentColor;

    scheduler.EndRendering();
    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Aa_Upscaler);
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(&scheduler)) {
        profiler->Mark(0xF5A3'0001ull, [use_hudless, multi_count] {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "FSR 3.1 %ux Frame Generation [%s] [VULKAN-NATIVE]",
                          multi_count + 1u, use_hudless ? "HUDless" : "FullFrame");
            return std::string{buf};
        });
    }
    const u32 out_w = scene.width;
    const u32 out_h = scene.height;
    scheduler.Record([this, prepare, source, dispatch, eval_count, out_imgs, out_descs, out_times,
                      color, out_w, out_h](vk::CommandBuffer cmdbuf) mutable {
        // The UI drew into `color`; the display pass (or fused EncodePass) reads it after.
        ImageBarrier(cmdbuf, color,
                     vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                         vk::PipelineStageFlagBits2::eComputeShader |
                         vk::PipelineStageFlagBits2::eTransfer,
                     vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
                     vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
        for (u32 slot = 0; slot < eval_count; ++slot) {
            ImageBarrier(cmdbuf, out_imgs[slot],
                         vk::PipelineStageFlagBits2::eFragmentShader |
                             vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderStorageWrite, vk::ImageLayout::eUndefined);
        }
        prepare.commandBuffer = cmdbuf;
        dispatch.commandBuffer = cmdbuf;
        const auto prepared = ffxVkPortableFrameGenerationContextPrepare(context, &prepare, &source);
        const auto dispatched =
            prepared == FFX_VK_PORTABLE_OK
                ? ffxVkPortableFrameGenerationContextRecordMultiDispatch(
                      context, &dispatch, eval_count, out_descs.data(), out_times.data())
                : prepared;
        const bool ok = dispatched == FFX_VK_PORTABLE_OK;
        if (!ok) {
            static int told = 0;
            if (told++ < 4) {
                std::printf("Frame generation: the interpolation could not be recorded (%s %d)\n",
                            prepared != FFX_VK_PORTABLE_OK ? "prepare" : "dispatch", int(dispatched));
                std::fflush(stdout);
            }
        }
        ImageBarrier(cmdbuf, color, vk::PipelineStageFlagBits2::eComputeShader,
                     vk::AccessFlagBits2::eShaderRead,
                     vk::PipelineStageFlagBits2::eFragmentShader |
                         vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                         vk::PipelineStageFlagBits2::eTransfer,
                     vk::AccessFlagBits2::eShaderRead |
                         vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eTransferRead);
        for (u32 slot = 0; slot < eval_count; ++slot) {
            ImageBarrier(cmdbuf, out_imgs[slot], vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderStorageWrite,
                         vk::PipelineStageFlagBits2::eFragmentShader |
                             vk::PipelineStageFlagBits2::eTransfer,
                         vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
        }
        if (!ok) {
            // Shown anyway (it was handed out as ready): copy the real frame instead of garbage.
            ImageBarrier(cmdbuf, color, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eCopy,
                         vk::AccessFlagBits2::eTransferRead);
            for (u32 slot = 0; slot < eval_count; ++slot) {
                const vk::Image out = out_imgs[slot];
                ImageBarrier(cmdbuf, out, vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eCopy,
                             vk::AccessFlagBits2::eTransferWrite);
                cmdbuf.copyImage(color, vk::ImageLayout::eGeneral, out, vk::ImageLayout::eGeneral,
                                 vk::ImageCopy{.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                               .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                               .extent = {out_w, out_h, 1}});
                ImageBarrier(cmdbuf, out, vk::PipelineStageFlagBits2::eCopy,
                             vk::AccessFlagBits2::eTransferWrite,
                             vk::PipelineStageFlagBits2::eFragmentShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
            }
        }
    });
}

u32 FrameGeneration::TakeOutputs(VAddr display, std::array<Output, MaxInterpOutputs>& out,
                                 int* out_backend) {
    if (!Wanted()) {
        return 0;
    }
    std::scoped_lock lock{targets_mutex};
    (void)display;
    const u32 count = std::min<u32>(ready_count, MaxInterpOutputs);
    u32 written = 0;
    for (u32 i = 0; i < count; ++i) {
        Target& t = targets[i];
        if (!t.ready || !t.image || !t.view) {
            continue;
        }
        out[written++] = {vk::Image(t.image), *t.view, t.format, t.width, t.height, t.frame};
        t.ready = false;
    }
    ready_count = 0;
    if (out_backend) {
        *out_backend = ready_backend;
    }
    if (written == 0) {
        ++stat_missed;
    } else {
        stat_taken += written;
    }
    return written;
}

} // namespace Vulkan
