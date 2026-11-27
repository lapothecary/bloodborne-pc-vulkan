// SPDX-License-Identifier: GPL-2.0-or-later
// Exercises the production target/resolve path on Vulkan (Lavapipe works, no game/window).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "common/slot_vector.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_button_prompts.h"
#include "upscale/vk_scene_resolution.h"
#include <vk_mem_alloc.h>

int main() {
    using namespace Vulkan;
    Instance instance(0, false);
    // Calls emitted by this executable use a local dispatcher. libbbgpu.so initializes
    // its own default dispatcher while constructing Instance.
    static vk::detail::DynamicLoader loader;
    vk::detail::DispatchLoaderDynamic dispatch;
    dispatch.init(
        loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
    dispatch.init(instance.GetInstance());
    dispatch.init(instance.GetDevice());
    Scheduler scheduler(instance);
    Runtime runtime(instance, scheduler);
    Common::SlotVector<VideoCore::ImageView> views;
    Common::SlotVector<VideoCore::Image> images;
    VideoCore::ImageInfo ci;
    ci.size = {1920,1080,1};
    ci.resources = {1,1};
    ci.type = Gcn::ImageType::Color2D;
    ci.pixel_format = vk::Format::eR8G8B8A8Unorm;
    ci.num_bits = 32;
    const auto color_id = images.insert(instance, runtime, views, ci);
    auto di = ci;
    di.pixel_format = vk::Format::eD32SfloatS8Uint;
    di.props.is_depth = di.props.has_stencil = true;
    const auto depth_id = images.insert(instance, runtime, views, di);
    SceneTargets targets(instance, scheduler, runtime, [&](VideoCore::ImageId id, u64 uid) {
        if (!images.is_allocated(id)) return static_cast<VideoCore::Image*>(nullptr);
        return !uid || images[id].image_uid == uid ? &images[id] : nullptr;
    });
    const bool color_eligible = targets.Eligible(images[color_id]);
    const bool depth_eligible = targets.Eligible(images[depth_id]);
    std::fprintf(stderr, "Scene target eligibility: color=%d depth=%d\n",
                 color_eligible, depth_eligible);
    if (!color_eligible || !depth_eligible) {
        std::puts("Scene targets: unsupported format blit, live scaling skipped");
        return 0;
    }

    VkBuffer readback{};
    VmaAllocation allocation{};
    VmaAllocationInfo ai{};
    const VkBufferCreateInfo bi{.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=16, .usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo ac{.flags=VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, .usage=VMA_MEMORY_USAGE_AUTO,
        .requiredFlags=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    assert(vmaCreateBuffer(instance.GetAllocator(), &bi, &ac, &readback, &allocation, &ai) == VK_SUCCESS);
    const auto pixel = [&](vk::Image image, vk::ImageAspectFlagBits aspect,
                           vk::ImageLayout layout, u32 w, u32 h) {
        scheduler.EndRendering();
        const auto cmd = scheduler.CommandBuffer();
        const vk::MemoryBarrier2 b{.srcStageMask=vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask=vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask=vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask=vk::AccessFlagBits2::eTransferRead};
        cmd.pipelineBarrier2({.memoryBarrierCount=1,.pMemoryBarriers=&b}, dispatch);
        const vk::BufferImageCopy region{.imageSubresource={aspect,0,0,1},
            .imageOffset={s32(w-1),s32(h-1),0},.imageExtent={1,1,1}};
        cmd.copyImageToBuffer(image, layout, readback, region, dispatch);
        const vk::MemoryBarrier2 host{.srcStageMask=vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask=vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask=vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask=vk::AccessFlagBits2::eHostRead};
        cmd.pipelineBarrier2({.memoryBarrierCount=1,.pMemoryBarriers=&host}, dispatch);
        scheduler.Finish();
        u32 result;
        std::memcpy(&result, ai.pMappedData, 4);
        return result;
    };
    VideoCore::ImageViewInfo cv;
    cv.format = ci.pixel_format;
    auto dv = cv;
    dv.format = di.pixel_format;
    const auto clear_native = [&](u32 rgba, u32 stencil = 7) {
        vk::ClearValue clear{};
        clear.color.float32 = std::array{float(rgba & 255)/255, float((rgba>>8)&255)/255,
                                        float((rgba>>16)&255)/255, 1.0f};
        runtime.ClearImage(&images[color_id], {}, clear);
        runtime.Transit(&images[depth_id], vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        runtime.FlushBarriers();
        scheduler.CommandBuffer().clearDepthStencilImage(
            images[depth_id].GetImage(), vk::ImageLayout::eTransferDstOptimal,
            vk::ClearDepthStencilValue{0.25f, stencil},
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth |
                vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}, dispatch);
    };
    constexpr u32 stencil_values[] = {0, 255, 128, 85, 170, 7, 13, 255, 0, 170};
    u32 stencil_index = 0;
    for (int preset : {3,1,4,2,0,3,0,1,0,3}) {
        const u32 copied_stencil = stencil_values[stencil_index++];
        const u32 resolved_stencil = copied_stencil ^ 255;
        const auto output = stencil_index > 7 ? SceneResolution::Size{3840,2160}
                                             : SceneResolution::Size{};
        const auto size = SceneResolution::ForPreset(preset, output);
        targets.SetSize(size);
        assert(SceneResolution::Unpack(SceneResolution::Pack(size)) == size);
        clear_native(0xffff0000, copied_stencil); // blue
        auto c = targets.Read(color_id,cv);
        assert(pixel(c.image,vk::ImageAspectFlagBits::eColor,c.layout,size.width,size.height)==0xffff0000);
        auto d = targets.Read(depth_id,dv);
        const u32 depth_bits = pixel(d.image,vk::ImageAspectFlagBits::eDepth,d.layout,size.width,size.height);
        assert(std::bit_cast<float>(depth_bits)==0.25f);
        assert((pixel(d.image,vk::ImageAspectFlagBits::eStencil,d.layout,size.width,size.height)&255)==copied_stencil);

        c = targets.Attachment(color_id,cv);
        d = targets.Attachment(depth_id,dv);
        auto cmd = scheduler.CommandBuffer();
        const vk::RenderingAttachmentInfo color{.imageView=c.view,.imageLayout=c.layout,
            .loadOp=vk::AttachmentLoadOp::eClear,.storeOp=vk::AttachmentStoreOp::eStore,
            .clearValue=vk::ClearValue{.color={.float32=std::array{1.f,0.f,0.f,1.f}}}};
        const vk::RenderingAttachmentInfo depth{.imageView=d.view,.imageLayout=d.layout,
            .loadOp=vk::AttachmentLoadOp::eClear,.storeOp=vk::AttachmentStoreOp::eStore,
            .clearValue=vk::ClearValue{.depthStencil={0.75f,resolved_stencil}}};
        cmd.beginRendering({.renderArea={{0,0},{size.width,size.height}},.layerCount=1,
            .colorAttachmentCount=1,.pColorAttachments=&color,
            .pDepthAttachment=&depth,.pStencilAttachment=&depth}, dispatch);
        cmd.endRendering(dispatch);
        runtime.Transit(&images[color_id],vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eTransfer,vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        assert(pixel(images[color_id].GetImage(),vk::ImageAspectFlagBits::eColor,
                     vk::ImageLayout::eTransferSrcOptimal,1920,1080)==0xff0000ff);
        runtime.Transit(&images[depth_id],vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eTransfer,vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        assert(std::bit_cast<float>(pixel(images[depth_id].GetImage(),vk::ImageAspectFlagBits::eDepth,
                     vk::ImageLayout::eTransferSrcOptimal,1920,1080))==0.75f);
        assert((pixel(images[depth_id].GetImage(),vk::ImageAspectFlagBits::eStencil,
                     vk::ImageLayout::eTransferSrcOptimal,1920,1080)&255)==resolved_stencil);
        clear_native(0xff00ff00); // overwrite after reduced render: proxy must be invalidated
        c=targets.Read(color_id,cv);
        assert(pixel(c.image,vk::ImageAspectFlagBits::eColor,c.layout,size.width,size.height)==0xff00ff00);
        std::printf("Scene targets: %ux%u color/depth/stencil roundtrip PASS\n",size.width,size.height);
    }

    // Half-resolution mip chain (the game's bloom pyramid): one proxy per level, scaled by the
    // scene's factor, resolved into its own native level.
    auto hi = ci;
    hi.size = {960, 540, 1};
    hi.resources = {4, 1};
    const auto half_id = images.insert(instance, runtime, views, hi);
    const auto scene = SceneResolution::ForPreset(4); // 640x360 (x3 of 1080p)
    targets.SetSize(scene);
    auto& half = images[half_id];
    assert(targets.Eligible(half) && !targets.EligibleScene(half));
    assert((targets.ProxySize(half, 0) == SceneResolution::Size{320, 180}));
    assert((targets.ProxySize(half, 2) == SceneResolution::Size{80, 45}));
    auto lv = cv;
    lv.range.base.level = 2;
    const auto proxy = targets.Attachment(half_id, lv);
    {
        auto cmd = scheduler.CommandBuffer();
        const vk::RenderingAttachmentInfo color{.imageView=proxy.view,.imageLayout=proxy.layout,
            .loadOp=vk::AttachmentLoadOp::eClear,.storeOp=vk::AttachmentStoreOp::eStore,
            .clearValue=vk::ClearValue{.color={.float32=std::array{0.f,1.f,0.f,1.f}}}};
        cmd.beginRendering({.renderArea={{0,0},{80,45}},.layerCount=1,
            .colorAttachmentCount=1,.pColorAttachments=&color}, dispatch);
        cmd.endRendering(dispatch);
    }
    assert(targets.ProxyCurrent(half, 2, 1) && !targets.ProxyCurrent(half, 0, 4));
    // A native access resolves the level: its last texel (239, 134) is the proxy's colour.
    runtime.Transit(&half, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    {
        scheduler.EndRendering();
        const auto cmd = scheduler.CommandBuffer();
        const vk::MemoryBarrier2 b{.srcStageMask=vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask=vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask=vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask=vk::AccessFlagBits2::eTransferRead};
        cmd.pipelineBarrier2({.memoryBarrierCount=1,.pMemoryBarriers=&b}, dispatch);
        const vk::BufferImageCopy region{.imageSubresource={vk::ImageAspectFlagBits::eColor,2,0,1},
            .imageOffset={239,134,0},.imageExtent={1,1,1}};
        cmd.copyImageToBuffer(half.GetImage(), vk::ImageLayout::eTransferSrcOptimal, readback,
                              region, dispatch);
        scheduler.Finish();
        u32 result;
        std::memcpy(&result, ai.pMappedData, 4);
        assert(result == 0xff00ff00);
    }
    std::puts("Scene targets: half-resolution mip level proxy roundtrip PASS");

    // Non-multiple-of-4/8 startup scene sizes (e.g. 16:10 1680x1050 -> Gnm 1680x1052 + half 840x528,
    // 1366x768 -> Gnm 1368x768 + half 684x384, 1440x900 -> half 720x452) must match EligibleScene /
    // Eligible while rejecting unrelated sizes.
    {
        targets.SetBaseSize({1680, 1050});
        targets.SetSize(SceneResolution::ForPreset(1, {1680, 1050})); // 1120x700 Quality
        auto ci_1052 = ci;
        ci_1052.size = {1680, 1052, 1};
        const auto c1052_id = images.insert(instance, runtime, views, ci_1052);
        auto di_1052 = di;
        di_1052.size = {1680, 1052, 1};
        const auto d1052_id = images.insert(instance, runtime, views, di_1052);
        auto hi_528 = ci;
        hi_528.size = {840, 528, 1};
        const auto h528_id = images.insert(instance, runtime, views, hi_528);
        auto unrelated = ci;
        unrelated.size = {1680, 1064, 1};
        const auto unrel_id = images.insert(instance, runtime, views, unrelated);

        assert(targets.EligibleScene(images[c1052_id]));
        assert(targets.EligibleScene(images[d1052_id]));
        assert(targets.Eligible(images[h528_id]) && !targets.EligibleScene(images[h528_id]));
        assert(!targets.Eligible(images[unrel_id]));
        assert((targets.ProxySize(images[c1052_id], 0) == SceneResolution::Size{1120, 700}));
        assert((targets.ProxySize(images[h528_id], 0) == SceneResolution::Size{560, 350}));

        assert(SceneTargets::MatchesGnmSceneDim(1368, 768, 1366, 768));
        assert(SceneTargets::MatchesGnmSceneDim(684, 384, 1366 / 2, 768 / 2));
        assert(SceneTargets::MatchesGnmSceneDim(720, 452, 1440 / 2, 900 / 2));
        assert(!SceneTargets::MatchesGnmSceneDim(1920, 1088, 1920, 1080));
        if (const char* vram_env = std::getenv("BB_VRAM_LIMIT_MB")) {
            const int lim_mb = std::atoi(vram_env);
            if (lim_mb > 0 && !instance.IsIntegrated()) {
                assert(instance.GetPhysicalVramBytes() <= u64(lim_mb) * (1ULL << 20));
                assert(instance.IsLowVramDevice() == (lim_mb <= 4608));
                std::printf("Low-VRAM gate check (%d MiB -> low_vram=%d) PASS\n",
                            lim_mb, int(instance.IsLowVramDevice()));
            }
        } else if (!instance.IsIntegrated() &&
                   instance.GetPhysicalVramBytes() > (4ULL << 30) + (512ULL << 20)) {
            assert(!instance.IsLowVramDevice());
        }
        std::puts("Scene targets: 16:10 1680x1050 (Gnm 1680x1052 / 840x528) & 1366x768 alignment PASS");
    }

    // ButtonPrompts UI Super-Resolution VRAM cap, deletion sync, and toggle-off reclamation tests.
    {
        assert(ButtonPrompts::ComputeUiUpscaleFactor(2048, 2048, true) == 0u);
        assert(ButtonPrompts::ComputeUiUpscaleFactor(2048, 2048, false) == 2u);
        assert(ButtonPrompts::ComputeUiUpscaleFactor(1024, 512, true) == 2u);
        assert(ButtonPrompts::ComputeUiUpscaleFactor(1024, 512, false) == 4u);
        assert(ButtonPrompts::ComputeUiUpscaleFactor(1024, 1024, true) == 2u);
        assert(ButtonPrompts::ComputeUiUpscaleFactor(256, 256, true) == 4u);
        assert(ButtonPrompts::ComputeAtlasUpscaleFactor(2048, 2048, true, true) == 1u);
        assert(ButtonPrompts::ComputeAtlasUpscaleFactor(2048, 2048, true, false) == 2u);
        assert(ButtonPrompts::ComputeAtlasUpscaleFactor(1024, 1024, true, true) == 2u);
        assert(ButtonPrompts::ComputeAtlasUpscaleFactor(1024, 1024, false, false) == 1u);
        assert(ButtonPrompts::ComputeUiCacheBudgetBytes(true) == (64ull << 20));
        assert(ButtonPrompts::ComputeUiCacheBudgetBytes(false) == (256ull << 20));
        assert(ButtonPrompts::ComputeUiCacheMaxEntries(true) == 32u);
        assert(ButtonPrompts::ComputeUiCacheMaxEntries(false) == 128u);

        ButtonPrompts prompts(instance, scheduler);
        VideoCore::ImageInfo ui_info{};
        ui_info.size = {1024, 1024, 1};
        ui_info.resources = {1, 1};
        ui_info.type = Gcn::ImageType::Color2D;
        ui_info.pixel_format = vk::Format::eBc7UnormBlock;
        ui_info.num_bits = 8;
        ui_info.props.is_block = true;
        VideoCore::ImageViewInfo ui_view_info{};
        ui_view_info.type = Gcn::ImageType::Color2D;
        ui_view_info.format = vk::Format::eBc7UnormBlock;
        ui_view_info.range = {{0, 0}, {1, 1}};

        const auto ui_id0 = images.insert(instance, runtime, views, ui_info);
        auto& ui_img0 = images[ui_id0];
        assert(prompts.IsUiTexture(ui_img0));

        u64 vma_blocks_before = 0, vma_used_before = 0;
        Vulkan::VmaDeviceUsage(vma_blocks_before, vma_used_before);
        const vk::ImageView upscaled0 = prompts.UiView(runtime, ui_img0, ui_view_info);
        assert(upscaled0);
        scheduler.Finish();
        u64 vma_blocks_after = 0, vma_used_after = 0;
        Vulkan::VmaDeviceUsage(vma_blocks_after, vma_used_after);
        assert(prompts.UiCacheCount() == 1u);
        assert(prompts.UiCacheBytes() >= (16ull << 20));
        assert(vma_used_after >= vma_used_before + (16ull << 20));

        // Deletion hook must immediately reclaim the upscaled HostImage (0 bytes leaked).
        prompts.OnImageDeleted(ui_img0.image_uid);
        prompts.Maintenance();
        u64 vma_blocks_del = 0, vma_used_del = 0;
        Vulkan::VmaDeviceUsage(vma_blocks_del, vma_used_del);
        assert(prompts.UiCacheCount() == 0u);
        assert(prompts.UiCacheBytes() == 0u);
        assert(prompts.RetiredCount() == 0u);
        assert(vma_used_del == vma_used_before);

        // Repeatedly upscaling >64 MiB of distinct 1024x1024 UI textures (6 x 16 MiB = 96 MiB)
        // must stay strictly within the low-VRAM byte cap when BB_VRAM_LIMIT_MB<=4608, and
        // toggling ui_upscale Off must reclaim 100% of cached UI VRAM on Maintenance().
        std::vector<VideoCore::ImageId> ui_ids;
        const u64 cap_bytes = ButtonPrompts::ComputeUiCacheBudgetBytes(instance.IsLowVramDevice());
        for (int k = 0; k < 6; ++k) {
            ui_ids.push_back(images.insert(instance, runtime, views, ui_info));
        }
        u64 vma_blocks_src = 0, vma_used_src = 0;
        Vulkan::VmaDeviceUsage(vma_blocks_src, vma_used_src);
        for (const auto id : ui_ids) {
            const vk::ImageView v = prompts.UiView(runtime, images[id], ui_view_info);
            assert(v);
            // Advance enough submissions (>4 ticks) so older finished entries become LRU-evictable.
            for (int f = 0; f < 6; ++f) {
                scheduler.Finish();
            }
            assert(prompts.UiCacheBytes() <= cap_bytes);
            u64 b_now = 0, u_now = 0;
            Vulkan::VmaDeviceUsage(b_now, u_now);
            assert(u_now - vma_used_src <= cap_bytes);
        }
        BbSettings::Get().ui_upscale.store(false, std::memory_order_relaxed);
        prompts.Maintenance();
        u64 b_off = 0, u_off = 0;
        Vulkan::VmaDeviceUsage(b_off, u_off);
        assert(prompts.UiCacheCount() == 0u);
        assert(prompts.UiCacheBytes() == 0u);
        assert(prompts.RetiredCount() == 0u);
        assert(u_off == vma_used_src);
        BbSettings::Get().ui_upscale.store(true, std::memory_order_relaxed);

        // Low-VRAM GPUs must skip 2048x2048 UI sheets (64 MiB uncompressed bloat).
        if (instance.IsLowVramDevice()) {
            auto big_ui = ui_info;
            big_ui.size = {2048, 2048, 1};
            const auto big_id = images.insert(instance, runtime, views, big_ui);
            assert(!prompts.UiView(runtime, images[big_id], ui_view_info));
        }
        std::printf("ButtonPrompts UI upscale cache (low_vram=%d, cap=%llu MiB): deletion sync, "
                    "byte budget & toggle-off PASS\n",
                    int(instance.IsLowVramDevice()),
                    static_cast<unsigned long long>(cap_bytes >> 20));
    }

    scheduler.Finish();
    vmaDestroyBuffer(instance.GetAllocator(),readback,allocation);
}
