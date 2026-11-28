// SPDX-License-Identifier: GPL-2.0-or-later
// Exercises the production target/resolve path on Vulkan (Lavapipe works, no game/window).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "common/slot_vector.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "cp/command_processor.h"
#include "vulkan/instance.h"
#include "render/direct_pso.h"
#include "render/direct_states.h"
#include "render/direct_targets.h"
#include "render/gcn_to_vk.h"
#include "render/native_cmd_stream.h"
#include "render/native_stream_shadow.h"
#include "textures/direct_textures.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_button_prompts.h"
#include "render/vk_constant_ring.h"
#include "upscale/vk_scene_resolution.h"
#include "../dantelion_gx/include/gx_direct_vulkan.h"
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

    // ImageInfo::MipOf / SliceOf micro-tiled sub-mip tests (PS4 GCN 2D_THIN1 -> 1D_THIN1 mip tail,
    // e.g. character face/head Blood Paint g_DecalDiffuseTexture mipgen targets).
    {
        auto make_parent = [](u32 w, u32 h, u32 levels, vk::Format fmt, u32 bpp) {
            VideoCore::ImageInfo p{};
            p.props.is_tiled = true;
            p.tile_mode = Gcn::TileMode::Thin2DThin;
            p.array_mode = Gcn::ArrayMode::Array2DTiledThin1;
            p.pixel_format = fmt;
            p.type = Gcn::ImageType::Color2D;
            p.size = {w, h, 1};
            p.pitch = w;
            p.resources = {levels, 1};
            p.num_samples = 1;
            p.num_bits = bpp;
            p.guest_address = 0x100000000ULL;
            p.UpdateSize();
            return p;
        };
        auto make_submip = [](const VideoCore::ImageInfo& p, u32 m, Gcn::TileMode tm,
                              Gcn::ArrayMode am) {
            VideoCore::ImageInfo sub{};
            sub.props.is_tiled = (am != Gcn::ArrayMode::ArrayLinearAligned);
            sub.tile_mode = tm;
            sub.array_mode = am;
            sub.pixel_format = p.pixel_format;
            sub.type = Gcn::ImageType::Color2D;
            sub.size = {std::max(p.size.width >> m, 1u), std::max(p.size.height >> m, 1u), 1};
            sub.pitch = p.mips_layout[m].pitch;
            sub.resources = {1, 1};
            sub.num_samples = 1;
            sub.num_bits = p.num_bits;
            sub.guest_address = p.guest_address + p.mips_layout[m].offset;
            sub.guest_size = p.mips_layout[m].size;
            sub.mips_layout[0] = {sub.guest_size, sub.pitch, p.mips_layout[m].height, 0};
            return sub;
        };

        // 256x256 R8Unorm (macro-tile 256x128): mip 1 (128x128) is already micro-tiled (1D_THIN1).
        const auto p_r8 = make_parent(256, 256, 9, vk::Format::eR8Unorm, 8);
        assert((p_r8.micro_tiled_mips & (1u << 0)) == 0u);
        assert((p_r8.micro_tiled_mips & (1u << 1)) != 0u);
        const auto sub_r8_m1 =
            make_submip(p_r8, 1, Gcn::TileMode::Thin1DThin, Gcn::ArrayMode::Array1DTiledThin1);
        assert(sub_r8_m1.MipOf(p_r8) == 1);
        assert(sub_r8_m1.SliceOf(p_r8, 1) == 0);
        // Macro-tiled mip 0 must NOT match a 1D micro-tiled view, and linear must never match.
        const auto bad_r8_m0 =
            make_submip(p_r8, 0, Gcn::TileMode::Thin1DThin, Gcn::ArrayMode::Array1DTiledThin1);
        assert(bad_r8_m0.MipOf(p_r8) == -1);
        const auto bad_r8_lin = make_submip(p_r8, 1, Gcn::TileMode::DisplayLinearAligned,
                                            Gcn::ArrayMode::ArrayLinearAligned);
        assert(bad_r8_lin.MipOf(p_r8) == -1);

        // 256x256 RGBA8 (macro-tile 128x64): mip 1 stays 2D_THIN1; mips 2..6 drop to 1D_THIN1.
        const auto p_rgba = make_parent(256, 256, 9, vk::Format::eR8G8B8A8Unorm, 32);
        assert((p_rgba.micro_tiled_mips & (1u << 1)) == 0u);
        assert((p_rgba.micro_tiled_mips & (1u << 2)) != 0u);
        for (u32 m = 1; m <= 6; ++m) {
            const bool is_micro = (p_rgba.micro_tiled_mips & (1u << m)) != 0u;
            const auto sub = make_submip(
                p_rgba, m, is_micro ? Gcn::TileMode::Thin1DThin : Gcn::TileMode::Thin2DThin,
                is_micro ? Gcn::ArrayMode::Array1DTiledThin1 : Gcn::ArrayMode::Array2DTiledThin1);
            assert(sub.MipOf(p_rgba) == s32(m));
            assert(sub.SliceOf(p_rgba, s32(m)) == 0);
        }

        // 128x128 RGBA8: mip 1 (64x64) drops straight into 1D_THIN1.
        const auto p_128 = make_parent(128, 128, 8, vk::Format::eR8G8B8A8Unorm, 32);
        assert((p_128.micro_tiled_mips & (1u << 1)) != 0u);
        const auto sub_128_m1 =
            make_submip(p_128, 1, Gcn::TileMode::Thin1DThin, Gcn::ArrayMode::Array1DTiledThin1);
        assert(sub_128_m1.MipOf(p_128) == 1);
        assert(sub_128_m1.SliceOf(p_128, 1) == 0);
        std::puts("ImageInfo::MipOf/SliceOf micro-tiled sub-mip resolution PASS");
    }

    // Regression test: DirectTargets persistent Color/Depth UID lookup + stack-temp rejection
    {
        auto heap_rt = std::make_unique<std::array<u32, 11>>();
        auto heap_dt = std::make_unique<std::array<u32, 13>>();
        heap_rt->fill(0u);
        (*heap_rt)[0] = 0x123456u; // base_address
        (*heap_rt)[4] = 0x00000123u; // info
        (*heap_rt)[6] = (1080u << 16) | 1920u; // extent

        heap_dt->fill(0u);
        (*heap_dt)[0] = 0x00000001u;
        (*heap_dt)[2] = 0x234567u; // z_read_base
        (*heap_dt)[3] = 0x234567u; // z_write_base
        (*heap_dt)[12] = (1080u << 16) | 1920u;

        const u32 rt_uid = DirectTargets::RegisterColorTarget(heap_rt->data(), nullptr, 0, 0, 1);
        const u32 dt_uid = DirectTargets::RegisterDepthTarget(heap_dt->data(), nullptr, 0, 0, 1);
        assert(rt_uid > 0u && rt_uid < DirectTargets::kMaxTargets);
        assert(dt_uid > 0u && dt_uid < DirectTargets::kMaxTargets && dt_uid != rt_uid);
        assert(DirectTargets::LookupColorTargetUid(heap_rt->data()) == rt_uid);
        assert(DirectTargets::LookupDepthTargetUid(heap_dt->data()) == dt_uid);

        // Simulate 0x217a210 post-HTILE mutation + in-place Depth re-registration (must keep dt_uid).
        (*heap_dt)[0] |= 0x400u;
        (*heap_dt)[9] = 0x345678u;
        assert(DirectTargets::RegisterDepthTarget(heap_dt->data(), nullptr, 0, 0, 1) == dt_uid);
        assert(DirectTargets::LookupDepthTargetUid(heap_dt->data()) == dt_uid);

        // Stack-allocated temporary decompress/blit structs must be rejected and never wrap slots.
        const auto stats_before = DirectTargets::GetStats();
        for (u32 iter = 0; iter < 10000u; ++iter) {
            alignas(16) u32 stack_rt[11]{};
            alignas(16) u32 stack_dt[13]{};
            stack_rt[0] = 0x400000u + iter;
            stack_dt[2] = 0x500000u + iter;
            assert(DirectTargets::RegisterColorTarget(stack_rt, nullptr, 0, 0, 1) == 0u);
            assert(DirectTargets::RegisterDepthTarget(stack_dt, nullptr, 0, 0, 1) == 0u);
        }
        const auto stats_after = DirectTargets::GetStats();
        assert(stats_after.wrap_count == stats_before.wrap_count);
        assert(stats_after.allocated_slots == stats_before.allocated_slots);
        assert(stats_after.stack_skips >= stats_before.stack_skips + 20000u);
        assert(DirectTargets::LookupColorTargetUid(heap_rt->data()) == rt_uid);
        assert(DirectTargets::LookupDepthTargetUid(heap_dt->data()) == dt_uid);
        // Lazy bind-time auto-registration on an unregistered heap RTV/DSV (copied from stack by 0x21667a0)
        auto lazy_rt = std::make_unique<std::array<u32, 11>>();
        auto lazy_dt = std::make_unique<std::array<u32, 13>>();
        lazy_rt->fill(0u);
        (*lazy_rt)[0] = 0x654321u;
        (*lazy_rt)[4] = 0x00000456u;
        (*lazy_rt)[6] = (720u << 16) | 1280u;
        lazy_dt->fill(0u);
        (*lazy_dt)[2] = 0x765432u;
        (*lazy_dt)[3] = 0x765432u;
        (*lazy_dt)[12] = (720u << 16) | 1280u;
        const u32 lazy_rt_uid = DirectTargets::LookupColorTargetUid(lazy_rt->data());
        const u32 lazy_dt_uid = DirectTargets::LookupDepthTargetUid(lazy_dt->data());
        assert(lazy_rt_uid > 0u && lazy_rt_uid != rt_uid && lazy_rt_uid != dt_uid);
        assert(lazy_dt_uid > 0u && lazy_dt_uid != lazy_rt_uid);
        assert(DirectTargets::LookupColorTargetUid(lazy_rt->data()) == lazy_rt_uid);
        assert(DirectTargets::LookupDepthTargetUid(lazy_dt->data()) == lazy_dt_uid);
        std::puts("DirectTargets persistent UID + lazy auto-reg + stack-temp guard PASS");
    }

    // ConstantRing starvation/livelock regression tests (FlushIfTickAdvanced, same-end Stamp,
    // and >1024 tiny menu ticks overflowing the stamp queue prior to first wrap).
    {
        ConstantRing ring(instance, scheduler);
        constexpr u64 kChunk = 1ull << 20;
        const u64 chunks_per_ring = ConstantRing::Capacity / kChunk;

        // 1) Fill entire ring in one tick -> Allocate fails -> Finish advances tick ->
        // FlushIfTickAdvanced() must publish pending_end and unblock allocation immediately.
        for (u64 i = 0; i < chunks_per_ring; ++i) {
            assert(ring.Allocate(kChunk, 256).has_value());
        }
        assert(!ring.Allocate(kChunk, 256).has_value());
        ring.Stamp(ring.Position());
        scheduler.Finish();
        ring.FlushIfTickAdvanced();
        assert(ring.Allocate(kChunk, 256).has_value());

        // 2) Fill ring again -> Stamp(same_end) after tick advance must also publish and recover.
        while (ring.Allocate(kChunk, 256).has_value()) {
        }
        const u64 full_pos = ring.Position();
        ring.Stamp(full_pos);
        scheduler.Finish();
        ring.Stamp(full_pos);
        assert(ring.Allocate(kChunk, 256).has_value());

        // 3) Simulate 1,500 tiny menu submissions (>1,024 stamps queue) before wrapping, then
        // verify full ring capacity retires cleanly via proactive drain + latest seqlock.
        for (int f = 0; f < 1500; ++f) {
            assert(ring.Allocate(1024, 256).has_value());
            ring.Stamp(ring.Position());
            scheduler.Flush();
            ring.FlushIfTickAdvanced();
        }
        scheduler.Finish();
        ring.FlushIfTickAdvanced();
        // Starting from non-zero offset (1500 KiB), one 1 MiB wrap-tail gap (<1 MiB) is skipped,
        // so chunks_per_ring - 1 (31 MiB) must fit and the next 1 MiB chunk must be refused
        // until stamped and retired.
        for (u64 i = 0; i < chunks_per_ring - 1; ++i) {
            assert(ring.Allocate(kChunk, 256).has_value());
        }
        assert(!ring.Allocate(kChunk, 256).has_value());
        ring.Stamp(ring.Position());
        scheduler.Finish();
        ring.FlushIfTickAdvanced();
        assert(ring.Allocate(kChunk, 256).has_value());
        std::puts("ConstantRing starvation/livelock + 1500-tick stamp-overflow recovery PASS");
    }

    // Stage 2A DirectPso: exhaustive 16x64x16x64 (1,048,576-pair) sampled-image format-class
    // equivalence proof + L1 ShaderPairCache lookup/invalidation tests.
    {
        for (u32 nf0 = 0; nf0 < 16u; ++nf0) {
            for (u32 df0 = 0; df0 < 64u; ++df0) {
                const auto n0 = static_cast<Gcn::NumberFormat>(nf0);
                const auto d0 = static_cast<Gcn::DataFormat>(df0);
                if (!DirectPso::CanUseFastSampledNfmtCheck(n0, d0)) {
                    continue;
                }
                const auto ref_nf = Gcn::RemapNumberFormat(n0, d0);
                const auto ref_conv = Gcn::MapNumberConversion(n0, d0);
                const bool ref_int = Gcn::IsInteger(ref_nf);
                const bool ref_srgb = (ref_nf == Gcn::NumberFormat::Srgb);
                for (u32 nf1 = 0; nf1 < 16u; ++nf1) {
                    for (u32 df1 = 0; df1 < 64u; ++df1) {
                        const u32 dw1 = (nf1 << 26) | (df1 << 20);
                        if (!DirectPso::MatchesFastSampledNfmtDw1(dw1, nf0 << 26, nf0)) {
                            continue;
                        }
                        const auto n1 = static_cast<Gcn::NumberFormat>(nf1);
                        const auto d1 = static_cast<Gcn::DataFormat>(df1);
                        const auto got_nf = Gcn::RemapNumberFormat(n1, d1);
                        const auto got_conv = Gcn::MapNumberConversion(n1, d1);
                        assert(got_nf == ref_nf);
                        assert(got_conv == ref_conv);
                        assert(Gcn::IsInteger(got_nf) == ref_int);
                        assert((got_nf == Gcn::NumberFormat::Srgb) == ref_srgb);
                    }
                }
            }
        }
        // Spot-check BC1 <-> BC7 / RGBA8 acceptance and 32-bit Unorm / BC4-6 Srgb / Format8 Float rejection.
        auto make_dw1 = [](Gcn::NumberFormat nf, Gcn::DataFormat df) -> u32 {
            return (static_cast<u32>(nf) << 26) | (static_cast<u32>(df) << 20);
        };
        assert(DirectPso::CanUseFastSampledNfmtCheck(Gcn::NumberFormat::Unorm, Gcn::DataFormat::FormatBc1));
        assert(DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Unorm, Gcn::DataFormat::FormatBc7), 0u, 0u));
        assert(DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Unorm, Gcn::DataFormat::Format8_8_8_8), 0u, 0u));
        assert(!DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Unorm, Gcn::DataFormat::Format32), 0u, 0u));
        assert(DirectPso::CanUseFastSampledNfmtCheck(Gcn::NumberFormat::Srgb, Gcn::DataFormat::FormatBc1));
        assert(DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Srgb, Gcn::DataFormat::FormatBc7), 9u << 26, 9u));
        assert(!DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Srgb, Gcn::DataFormat::FormatBc6), 9u << 26, 9u));
        assert(!DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Float, Gcn::DataFormat::Format8), 7u << 26, 7u));
        assert(DirectPso::MatchesFastSampledNfmtDw1(
            make_dw1(Gcn::NumberFormat::Float, Gcn::DataFormat::Format16_16_16_16), 7u << 26, 7u));

        DirectPso::ShaderPairCache pc{};
        pc.Insert(0x100000, 0x200000, 0x11, 0x22, 7u, 0xABCDu, 16u, 17u, 0x3u, 42u);
        const auto* hit = pc.Lookup(0x100000, 0x200000, 0x11, 0x22, 7u, 0xABCDu);
        assert(hit && hit->set_base == 16u && hit->first_match_idx == 17u && hit->pso_id == 42u);
        assert(pc.Lookup(0x100000, 0x200000, 0x11, 0x22, 8u, 0xABCDu) == nullptr);
        assert(pc.Lookup(0x100000, 0x200000, 0x12, 0x22, 7u, 0xABCDu) == nullptr);
        assert(pc.Lookup(0x100000, 0x200000, 0x11, 0x22, 7u, 0xABCEu) == nullptr);
        pc.Clear();
        assert(pc.Lookup(0x100000, 0x200000, 0x11, 0x22, 7u, 0xABCDu) == nullptr);
        std::puts("DirectPso 1M-pair sampled-image SpecWordCheck proof + ShaderPairCache PASS");
    }

    // Stage 3A: DirectStates registry (dedup, L1 pointer cache, in-place mutation safety, stack
    // guard) + NativeCmdStream binary writer/reader round-trip.
    {
        auto rs1 = std::make_unique<DirectStates::RasterStateDesc>();
        auto rs2 = std::make_unique<DirectStates::RasterStateDesc>();
        rs1->primitive_setup = 0x10203u;
        rs1->clip_control = 0x40506u;
        rs1->msaa_enable = 1u;
        *rs2 = *rs1;
        const auto rid1 = DirectStates::LookupOrRegisterRasterState(rs1.get());
        const auto rid2 = DirectStates::LookupOrRegisterRasterState(rs2.get());
        assert(rid1 > 0u && rid1 == rid2); // distinct heap ptrs, identical bits -> deduplicated!
        assert(DirectStates::LookupOrRegisterRasterState(rs1.get()) == rid1); // L1 hit
        DirectStates::RasterStateDesc rs_read{};
        assert(DirectStates::ReadRasterState(rid1, rs_read) && rs_read == *rs1);
        // Mutate rs2 in place -> must not return stale rid1!
        rs2->primitive_setup = 0x99999u;
        const auto rid3 = DirectStates::LookupOrRegisterRasterState(rs2.get());
        assert(rid3 > 0u && rid3 != rid1);
        assert(DirectStates::ReadRasterState(rid3, rs_read) && rs_read == *rs2);
        // Stack-allocated object must still deduplicate without polluting pointer L1.
        DirectStates::RasterStateDesc stack_rs = *rs1;
        assert(DirectStates::LookupOrRegisterRasterState(&stack_rs) == rid1);

        auto bs1 = std::make_unique<DirectStates::BlendStateDesc>();
        bs1->render_target_channel_mask = 0xFFFFFFFFu;
        bs1->blend_control[0] = 0x12345678u;
        bs1->alpha_to_coverage_enable = 1u;
        const auto bid1 = DirectStates::LookupOrRegisterBlendState(bs1.get());
        assert(bid1 > 0u);
        DirectStates::BlendStateDesc bs_read{};
        assert(DirectStates::ReadBlendState(bid1, bs_read) && bs_read == *bs1);

        auto ds1 = std::make_unique<DirectStates::DepthStencilStateDesc>();
        ds1->depth_control = 0x00000013u;
        ds1->stencil_control = 0x00001122u;
        ds1->stencil_op_control = 0x00FF00AAu;
        const auto did1 = DirectStates::LookupOrRegisterDepthStencilState(ds1.get());
        assert(did1 > 0u);
        DirectStates::DepthStencilStateDesc ds_read{};
        assert(DirectStates::ReadDepthStencilState(did1, ds_read) && ds_read == *ds1);

        // NativeCmdStream round-trip across BeginPassTargets, SetPipelineState, DrawIndexed,
        // LegacyPm4Slice (Scaleform UI bridge), Noop skip, and EndFrameFlip.
        std::array<uint32_t, 128> chunk{};
        bb_native_stream::NativeStreamWriter w(chunk.data(), static_cast<uint32_t>(chunk.size()));

        bb_native_stream::BeginPassTargetsPayload pass{};
        pass.cb_uid[0] = 7u;
        pass.db_uid = 12u;
        pass.cb_mask = 0x1u;
        pass.width = 2560u;
        pass.height = 1440u;
        assert(w.Emit(bb_native_stream::CmdOp::BeginPassTargets, 0u, pass));

        bb_native_stream::SetPipelineStatePayload pipe{};
        pipe.vs_addr = 0x1100001000ULL;
        pipe.ps_addr = 0x1100002000ULL;
        pipe.raster_id = rid1;
        pipe.blend_id = bid1;
        pipe.ds_id = did1;
        pipe.pso_id = 42u;
        pipe.prim_type = 4u;
        assert(w.Emit(bb_native_stream::CmdOp::SetPipelineState, 0u, pipe));
        assert(w.Reserve(bb_native_stream::CmdOp::Noop, 0u, 3u) != nullptr);

        bb_native_stream::DrawIndexedPayload draw{3600u, 1u, 0u, 0, 0u};
        assert(w.Emit(bb_native_stream::CmdOp::DrawIndexed, 0u, draw));

        bb_native_stream::LegacyPm4SlicePayload ui_slice{0x114AA00400ULL, 128u, 0u};
        assert(w.Emit(bb_native_stream::CmdOp::LegacyPm4Slice, 0u, ui_slice));
        assert(w.EmitEmpty(bb_native_stream::CmdOp::EndFrameFlip));

        bb_native_stream::NativeStreamReader r(w.Data(), w.SizeDwords());
        bb_native_stream::CmdHeader hdr{};
        const uint32_t* pay = nullptr;
        assert(r.Next(hdr, pay) && static_cast<bb_native_stream::CmdOp>(hdr.op) ==
                                       bb_native_stream::CmdOp::BeginPassTargets);
        bb_native_stream::BeginPassTargetsPayload pass_out{};
        std::memcpy(&pass_out, pay, sizeof(pass_out));
        assert(pass_out.cb_uid[0] == 7u && pass_out.db_uid == 12u && pass_out.width == 2560u);

        assert(r.Next(hdr, pay) && static_cast<bb_native_stream::CmdOp>(hdr.op) ==
                                       bb_native_stream::CmdOp::SetPipelineState);
        bb_native_stream::SetPipelineStatePayload pipe_out{};
        std::memcpy(&pipe_out, pay, sizeof(pipe_out));
        assert(pipe_out.raster_id == rid1 && pipe_out.blend_id == bid1 && pipe_out.ds_id == did1 &&
               pipe_out.pso_id == 42u);

        // Noop must be skipped transparently by NativeStreamReader.
        assert(r.Next(hdr, pay) && static_cast<bb_native_stream::CmdOp>(hdr.op) ==
                                       bb_native_stream::CmdOp::DrawIndexed);
        bb_native_stream::DrawIndexedPayload draw_out{};
        std::memcpy(&draw_out, pay, sizeof(draw_out));
        assert(draw_out.index_count == 3600u);

        assert(r.Next(hdr, pay) && static_cast<bb_native_stream::CmdOp>(hdr.op) ==
                                       bb_native_stream::CmdOp::LegacyPm4Slice);
        bb_native_stream::LegacyPm4SlicePayload ui_out{};
        std::memcpy(&ui_out, pay, sizeof(ui_out));
        assert(ui_out.pm4_va == 0x114AA00400ULL && ui_out.num_dwords == 128u);

        assert(r.Next(hdr, pay) && static_cast<bb_native_stream::CmdOp>(hdr.op) ==
                                       bb_native_stream::CmdOp::EndFrameFlip);
        assert(!r.Next(hdr, pay) && r.Empty());
        std::puts("Stage 3A DirectStates registry + NativeCmdStream round-trip PASS");

        // Stage 3B-1: Multi-state ShaderPairCache persistence + NativeStreamShadow parity verifier.
        DirectPso::ShaderPairCache pair_cache{};
        const u64 direct_key_a = (0x123456789ABCDEF0ULL) | (1ULL << 63);
        const u64 direct_key_b = (0x0FEDCBA987654321ULL) | (1ULL << 63);
        const u32 ids_a = u32(rid1) | (u32(bid1) << 10) | (4u << 20);
        const u32 ids_b = u32(rid1) | (u32(bid1 + 1u) << 10) | (4u << 20);
        pair_cache.Insert(0x1100001000ULL, 0x1100002000ULL, 0x11u, 0x22u, direct_key_a, 0x99u, 16u,
                          17u, 0x1u, 101u, ids_a);
        pair_cache.Insert(0x1100001000ULL, 0x1100002000ULL, 0x11u, 0x22u, direct_key_b, 0x99u, 32u,
                          33u, 0x2u, 102u, ids_b);
        const auto* hit_a = pair_cache.Lookup(0x1100001000ULL, 0x1100002000ULL, 0x11u, 0x22u,
                                              direct_key_a, 0x99u, ids_a);
        const auto* hit_b = pair_cache.Lookup(0x1100001000ULL, 0x1100002000ULL, 0x11u, 0x22u,
                                              direct_key_b, 0x99u, ids_b);
        assert(hit_a && hit_a->pso_id == 101u && hit_b && hit_b->pso_id == 102u);

        auto regs_ptr = std::make_unique<Gcn::Regs>();
        auto& regs = *regs_ptr;
        regs.SetDefaults();
        regs.reg_array[0xA204] = rs_read.primitive_setup & 0x0F7FE03Fu;
        regs.reg_array[0xA205] = rs_read.clip_control & 0x00393FFFu;
        regs.reg_array[0xA292] =
            (rs_read.msaa_enable & 1u) | ((rs_read.scissor_enable << 1) & 2u);
        regs.reg_array[0xA08E] = bs_read.render_target_channel_mask;
        for (u32 s = 0; s < 8u; ++s) {
            regs.reg_array[0xA1E0 + s] = bs_read.blend_control[s] & 0xFFFF1FFFu;
        }
        regs.reg_array[0xA200] = ds_read.depth_control & 0xC07007FFu;
        regs.reg_array[0xA10B] = ds_read.stencil_control & 0x00FFFFFFu;
        regs.reg_array[0xA10C] = ds_read.stencil_op_control & 0x00FFFF00u;
        regs.reg_array[0xC24C] = 1200u; // VGT_NUM_INDICES
        regs.reg_array[0xC24D] = 1u;    // VGT_NUM_INSTANCES

        std::array<Gcn::CbDbExtent, 8> extents{};
        extents[0].width = 2560u;
        extents[0].height = 1440u;
        std::array<u16, 8> cb_uids{7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        NativeStreamShadow shadow{};
        shadow.BeginSubmission();
        shadow.RecordDirectDraw(regs, extents.data(), cb_uids.data(), 12u, rid3, bid1, did1, 11u,
                                0u, 77u);
        // Same dynamic state -> must delta-skip SetDynamicState (dyn_packets stays 1).
        shadow.RecordDirectDraw(regs, extents.data(), cb_uids.data(), 12u, rid3, bid1, did1, 13u,
                                0u, 77u);
        // Mutate scissor -> must emit 1 additional SetDynamicState (dyn_packets == 2).
        regs.reg_array[0xA094] = 0x00100020u;
        shadow.RecordDirectDraw(regs, extents.data(), cb_uids.data(), 12u, rid3, bid1, did1, 11u,
                                4u, 77u);
        const u32 dummy_pm4[4] = {0xC0022D00u, 1u, 2u, 3u};
        shadow.RecordLegacyPm4Draw(dummy_pm4, 4u);
        assert(shadow.EndSubmission(/*log_periodic=*/false));
        const auto st_ok = shadow.CumulativeStats();
        assert(st_ok.draws_3d == 3u && st_ok.draws_pm4 == 1u && st_ok.pso_id_covered == 3u &&
               st_ok.dyn_packets == 2u && st_ok.rs_mismatches == 0u && st_ok.bs_mismatches == 0u &&
               st_ok.ds_mismatches == 0u && st_ok.stream_errors == 0u);

        // Deliberate 1-bit Blend mismatch must be caught by live parity verifier.
        regs.reg_array[0xA08E] ^= 0x1u;
        shadow.BeginSubmission();
        shadow.RecordDirectDraw(regs, extents.data(), cb_uids.data(), 12u, rid3, bid1, did1, 11u,
                                0u, 77u);
        assert(shadow.EndSubmission(/*log_periodic=*/false));
        assert(shadow.CumulativeStats().bs_mismatches == 1u);
        std::puts("Stage 3B-2 DirectState PSO key + DynamicState delta + NativeStreamShadow PASS");

        // Stage 3C-1: Verify `kUdFlagSpecInvariant` preserves `stage_spec_epoch` while bumping
        // `stage_ud_epoch`, whereas non-invariant UD / Bind*Shader / raw SetShReg / ClearState
        // always invalidate `stage_spec_epoch`.
        {
            BbCp::CommandProcessor::DirectStateTrack ds_track{};
            std::array<u64, 6> ud_ep{1, 1, 1, 1, 1, 1};
            u64 flat_inv = 1u;
            u64 csum = 0u;
            u64 pso_ep = 1u, dyn_ep = 1u, non_sh_ep = 1u;
            auto make_nop_hdr = [](u32 body_dw) -> u32 {
                return 0xC0001000u | ((body_dw - 1u) << 16);
            };
            // 1) Spec-invariant PS user-data update (sh_offset = 0x0C, mask = 0x3)
            const u32 pkt_inv[6] = {
                make_nop_hdr(5u),
                BbCp::NopPayload::DirectVulkan,
                u32(BbCp::DirectVk::Op::SetStageUserData) | BbCp::DirectVk::kUdFlagSpecInvariant,
                0x0003000Cu,
                0x11223344u,
                0x55667788u,
            };
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_inv, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ud_ep[0] == 2u && ds_track.stage_spec_epoch[0] == 1u);
            assert(regs.reg_array[0x2C0Cu] == 0x11223344u &&
                   regs.reg_array[0x2C0Du] == 0x55667788u);

            // 2) Non-invariant PS user-data update (SRVs/samplers/CB spec changed)
            const u32 pkt_mut[6] = {
                make_nop_hdr(5u),
                BbCp::NopPayload::DirectVulkan,
                u32(BbCp::DirectVk::Op::SetStageUserData),
                0x0003000Cu,
                0x99AABBCCu,
                0xDDEEFF00u,
            };
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_mut, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ud_ep[0] == 3u && ds_track.stage_spec_epoch[0] == 2u);

            // 3) BindPsShader (valid = 1, 12 payload dwords) bumps PS spec epoch only
            u32 pkt_bind_ps[15]{};
            pkt_bind_ps[0] = make_nop_hdr(14u);
            pkt_bind_ps[1] = BbCp::NopPayload::DirectVulkan;
            pkt_bind_ps[2] = u32(BbCp::DirectVk::Op::BindPsShader) | (1u << 16);
            pkt_bind_ps[3] = 0xCAFE0001u;
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_bind_ps, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ds_track.stage_spec_epoch[0] == 3u && ds_track.stage_spec_epoch[1] == 1u);

            // 4) BindVsShader (7 payload dwords) bumps VS spec epoch only
            u32 pkt_bind_vs[10]{};
            pkt_bind_vs[0] = make_nop_hdr(9u);
            pkt_bind_vs[1] = BbCp::NopPayload::DirectVulkan;
            pkt_bind_vs[2] = u32(BbCp::DirectVk::Op::BindVsShader);
            pkt_bind_vs[3] = 0xBEEF0002u;
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_bind_vs, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ds_track.stage_spec_epoch[0] == 3u && ds_track.stage_spec_epoch[1] == 2u);

            // 5) Raw PM4 SetShReg on VS (sh_off = 0x4C, 1 dword) bumps VS ud + spec + flat_inv
            const u32 pkt_sh[3] = {0xC0017600u, 0x004Cu, 0x12345678u};
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_sh, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ud_ep[1] == 2u && ds_track.stage_spec_epoch[1] == 3u && flat_inv == 2u);

            // 6) ClearState bumps all 6 stage_spec_epoch slots
            const u32 pkt_clr[2] = {0xC0001200u, 0u};
            BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_clr, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track);
            assert(ds_track.stage_spec_epoch[0] == 4u && ds_track.stage_spec_epoch[1] == 4u &&
                   ds_track.stage_spec_epoch[5] == 2u);
            std::puts("Stage 3C-1 kUdFlagSpecInvariant & stage_spec_epoch tracking PASS");

            // Stage 3C-2: Verify authoritative `Op::SetDirectStateIds` applies RS/BS/DS into
            // `Gcn::Regs` without bumping `pso_misc_state_epoch`, preserves `sref` in 0xA10C/D,
            // and verify branchless `FastSampledNfmtMismatchBits` equivalence across all 16,384 triples.
            const u64 misc_before = ds_track.pso_misc_state_epoch;
            regs.reg_array[0xA10Cu] = 0x5A00005Au; // sref = 0x5A
            regs.reg_array[0xA10Du] = 0x5A00005Au;
            const u32 pkt_rs[3] = {
                make_nop_hdr(2u),
                BbCp::NopPayload::DirectVulkan,
                u32(BbCp::DirectVk::Op::SetDirectStateIds) | (0u << 16) | (u32(rid3) << 20),
            };
            const u32 pkt_bs[3] = {
                make_nop_hdr(2u),
                BbCp::NopPayload::DirectVulkan,
                u32(BbCp::DirectVk::Op::SetDirectStateIds) | (1u << 16) | (u32(bid1) << 20),
            };
            const u32 pkt_ds[3] = {
                make_nop_hdr(2u),
                BbCp::NopPayload::DirectVulkan,
                u32(BbCp::DirectVk::Op::SetDirectStateIds) | (2u << 16) | (u32(did1) << 20),
            };
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_rs, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_bs, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, pkt_ds, csum, nullptr, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                &flat_inv, &ds_track));
            assert(ds_track.pso_misc_state_epoch == misc_before);
            assert(ds_track.rs_id == rid3 && ds_track.bs_id == bid1 && ds_track.ds_id == did1);
            assert(regs.reg_array[0xA204u] == (rs_read.primitive_setup & 0x0F7FE03Fu));
            assert(regs.reg_array[0xA205u] == (rs_read.clip_control & 0x00393FFFu));
            assert(regs.reg_array[0xA292u] ==
                   ((rs_read.msaa_enable & 1u) | ((rs_read.scissor_enable << 1) & 2u)));
            assert(regs.reg_array[0xA2DFu] == rs_read.depth_bias_clamp_bits);
            assert(regs.reg_array[0xA08Eu] == bs_read.render_target_channel_mask);
            assert(regs.reg_array[0xA2DCu] == (0xAA00u | (bs_read.alpha_to_coverage_enable & 1u)));
            for (u32 s = 0u; s < 8u; ++s) {
                assert(regs.reg_array[0xA1E0u + s] == (bs_read.blend_control[s] & 0xFFFF1FFFu));
            }
            assert(regs.reg_array[0xA200u] == (ds_read.depth_control & 0xC07007FFu));
            assert(regs.reg_array[0xA10Bu] == (ds_read.stencil_control & 0x00FFFFFFu));
            const u32 exp_st_op = 0x5A00005Au | (ds_read.stencil_op_control & 0x00FFFF00u);
            assert(regs.reg_array[0xA10Cu] == exp_st_op && regs.reg_array[0xA10Du] == exp_st_op);

            for (u32 exp_nf = 0u; exp_nf < 16u; ++exp_nf) {
                for (u32 act_nf = 0u; act_nf < 16u; ++act_nf) {
                    for (u32 act_df = 0u; act_df < 64u; ++act_df) {
                        const u32 dw1 = (act_nf << 26) | (act_df << 20);
                        const bool ref =
                            DirectPso::MatchesFastSampledNfmtDw1(dw1, exp_nf << 26, exp_nf);
                        const bool fast =
                            DirectPso::FastSampledNfmtMismatchBits(dw1, exp_nf << 26, exp_nf) == 0u;
                        assert(ref == fast);
                    }
                }
            }
            std::puts("Stage 3C-2 authoritative SetDirectStateIds + branchless SpecWordCheck PASS");

            // Stage 3C-3: Verify (1) DirectStates::LookupOrRegisterPsUsage exact `0x10873a0`
            // bit parity, deduplication, and in-place reload safety; (2) Op::SetDirectStateIds
            // (kind=3) writes 0xA191.., skips on same ID, keeps dyn_ep & misc_state_epoch
            // untouched, and does NOT dirty Stage B `stage_b_mask[0]` bits 50..56 when
            // `!track_all_blocks`; (3) `RegDirty::Mark` strips dead Stage-A-only chunks 50..56
            // from `stage_b_mask[0]` while keeping `blocks` intact for `DrawPreparation`.
            {
                // Exact PS4 2-byte `VertexExportSemantic` ({semantic, outIndex}) and
                // `PixelInputSemantic` (u16: bits[0..7]=semantic, [8..9]=defaultVal,
                // [10]=isFlatShaded, [12]=isCustom) layout matching `0x10873a0`.
                u8 vs_exp_a[6] = {
                    10u, 0u,
                    20u, 1u,
                    30u, 2u,
                };
                u16 ps_inp_a[3] = {
                    static_cast<u16>(20u | (1u << 10)), // matched sem 20 -> idx=1 | flat(0x400)
                    static_cast<u16>(99u | (2u << 8)),  // unmatched -> 0x20 | defaultVal=2 (0x220)
                    static_cast<u16>(30u | (1u << 12)), // matched sem 30 + custom(bit12) -> 0x422
                };
                const u16 u_id1 = Vulkan::DirectStates::LookupOrRegisterPsUsage(
                    vs_exp_a, 3u, ps_inp_a, 3u);
                assert(u_id1 != 0u);
                const auto* u_desc1 = Vulkan::DirectStates::GetPsUsagePtr(u_id1);
                assert(u_desc1 != nullptr && u_desc1->count == 3u);
                assert(u_desc1->dwords[0] == (1u | 0x400u));
                assert(u_desc1->dwords[1] == (0x20u | (2u << 8)));
                assert(u_desc1->dwords[2] == (0x20u | 2u | 0x400u));

                // Different pointer pair with identical semantic tables must deduplicate to u_id1.
                u8 vs_exp_b[6] = {10u, 0u, 20u, 1u, 30u, 2u};
                u16 ps_inp_b[3] = {ps_inp_a[0], ps_inp_a[1], ps_inp_a[2]};
                const u16 u_id2 = Vulkan::DirectStates::LookupOrRegisterPsUsage(
                    vs_exp_b, 3u, ps_inp_b, 3u);
                assert(u_id2 == u_id1);

                // Repeat call hits L1 seqlock cache.
                assert(Vulkan::DirectStates::LookupOrRegisterPsUsage(vs_exp_a, 3u, ps_inp_a, 3u) ==
                       u_id1);

                // In-place mutation at SAME pointer addresses (area reload) must produce a NEW ID.
                ps_inp_a[0] = 10u;
                const u16 u_id3 = Vulkan::DirectStates::LookupOrRegisterPsUsage(
                    vs_exp_a, 3u, ps_inp_a, 3u);
                assert(u_id3 != 0u && u_id3 != u_id1);
                const auto* u_desc3 = Vulkan::DirectStates::GetPsUsagePtr(u_id3);
                assert(u_desc3 != nullptr && u_desc3->count == 3u && u_desc3->dwords[0] == 0u);

                // RegDirty::Mark must mask chunks 50..56 out of stage_b_mask[0] and
                // batch_stage_b_mask[0] when !track_all_blocks, while Stage B regs still mark.
                Gcn::RegDirty rd{};
                rd.Clear();
                rd.ClearBatchStageBDirty();
                rd.track_all_blocks = false;
                rd.Mark(0xA191u, 16u);
                rd.Mark(0xA1B0u, 8u);
                assert(rd.stage_b_mask[0] == 0u && rd.batch_stage_b_mask[0] == 0u);
                rd.Mark(0xA204u, 1u);
                assert(rd.stage_b_mask[1] != 0u && rd.batch_stage_b_mask[1] != 0u);

                // Op::SetDirectStateIds(kind=3, u_id1) writes exact dwords, keeps dyn_ep & misc_ep,
                // leaves Stage B dirty mask zero when !track_all_blocks, and skips on duplicate ID.
                rd.Clear();
                rd.track_all_blocks = false;
                const u64 dyn_before = dyn_ep;
                const u64 misc_before2 = ds_track.pso_misc_state_epoch;
                regs.reg_array[0xA191u] = 0xDEAD0000u;
                regs.reg_array[0xA192u] = 0xDEAD0001u;
                regs.reg_array[0xA193u] = 0xDEAD0002u;
                const u32 pkt_psu[3] = {
                    make_nop_hdr(2u),
                    BbCp::NopPayload::DirectVulkan,
                    u32(BbCp::DirectVk::Op::SetDirectStateIds) | (3u << 16) | (u32(u_id1) << 20),
                };
                assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, pkt_psu, csum, &rd, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                    &flat_inv, &ds_track));
                assert(ds_track.ps_usage_id == u_id1);
                assert(regs.reg_array[0xA191u] == u_desc1->dwords[0]);
                assert(regs.reg_array[0xA192u] == u_desc1->dwords[1]);
                assert(regs.reg_array[0xA193u] == u_desc1->dwords[2]);
                assert(dyn_ep == dyn_before && ds_track.pso_misc_state_epoch == misc_before2);
                assert(rd.stage_b_mask[0] == 0u && rd.stage_b_mask[1] == 0u);

                // Duplicate Op 38 kind=3 with same ID must be an O(1) no-op (sentinel untouched).
                regs.reg_array[0xA191u] = 0xCAFEBABEu;
                assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, pkt_psu, csum, &rd, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                    &flat_inv, &ds_track));
                assert(regs.reg_array[0xA191u] == 0xCAFEBABEu);

                // track_all_blocks == true (BB_PIPE_VERIFY=1) must mark `blocks` on ID change.
                rd.Clear();
                rd.track_all_blocks = true;
                const u32 pkt_psu3[3] = {
                    make_nop_hdr(2u),
                    BbCp::NopPayload::DirectVulkan,
                    u32(BbCp::DirectVk::Op::SetDirectStateIds) | (3u << 16) | (u32(u_id3) << 20),
                };
                assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, pkt_psu3, csum, &rd, false, &pso_ep, &dyn_ep, &non_sh_ep, ud_ep.data(),
                    &flat_inv, &ds_track));
                assert(ds_track.ps_usage_id == u_id3 && regs.reg_array[0xA191u] == 0u);
                assert(rd.blocks.any() && rd.stage_b_mask[0] != 0u);
                std::puts("Stage 3C-3 PsUsageId L1 cache + Stage-A-only chunk strip PASS");

                // Stage 4A: Verify precompiled DirectStates Vulkan dynamic state parity
                // (`CompiledRasterDyn` & `CompiledDepthStencilDyn`), `kStageBVpUnaliasedMask0`
                // exact chunk coverage, and O(1) duplicate Op 38 kind=0/1/2 skip.
                {
                    const auto* crs = Vulkan::DirectStates::GetCompiledRasterDyn(rid3);
                    const auto* cds = Vulkan::DirectStates::GetCompiledDepthStencilDyn(did1);
                    assert(crs != nullptr && cds != nullptr);
                    assert(crs->cull_mode_vk ==
                           static_cast<u32>(GcnToVk::CullMode(regs.polygon_control.CullingMode())));
                    assert(crs->front_face_vk ==
                           static_cast<u32>(GcnToVk::FrontFace(regs.polygon_control.front_face)));
                    assert(crs->depth_bias_enable == regs.polygon_control.NeedsBias());
                    assert(crs->depth_bias_const == regs.poly_offset.front_offset);
                    assert(crs->depth_bias_clamp == regs.poly_offset.depth_bias);
                    assert(crs->depth_bias_slope == regs.poly_offset.front_scale / 16.0f);
                    assert(crs->rasterizer_discard ==
                           (regs.clipper_control.dx_rasterization_kill != 0u));
                    const u8 exp_vp_flags = static_cast<u8>(
                        (regs.clipper_control.clip_disable != 0u ? 1u : 0u) |
                        (regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW ? 2u : 0u) |
                        (regs.mode_control.vport_scissor_enable != 0u ? 4u : 0u) |
                        (regs.polygon_control.enable_window_offset != 0u ? 8u : 0u));
                    assert(crs->vp_flags == exp_vp_flags);

                    const auto& dc = regs.depth_control;
                    const auto& sc = regs.stencil_control;
                    assert(cds->depth_enable == (dc.depth_enable != 0u));
                    assert(cds->depth_write_enable == (dc.depth_write_enable != 0u));
                    assert(cds->depth_bounds_enable == (dc.depth_bounds_enable != 0u));
                    assert(cds->depth_compare_op_vk ==
                           static_cast<u32>(GcnToVk::CompareOp(dc.depth_func)));
                    assert(cds->stencil_enable == (dc.stencil_enable != 0u));
                    assert(cds->two_sided == (dc.backface_enable != 0u));
                    assert(cds->front_fail_vk ==
                           static_cast<u32>(GcnToVk::StencilOp(sc.stencil_fail_front)));
                    assert(cds->front_pass_vk ==
                           static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zpass_front)));
                    assert(cds->front_zfail_vk ==
                           static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zfail_front)));
                    assert(cds->front_cmp_vk ==
                           static_cast<u32>(GcnToVk::CompareOp(dc.stencil_ref_func)));

                    // Verify kStageBVpUnaliasedMask0 covers viewport/scissor regs and never
                    // triggers on RS/BS/DS regs.
                    Gcn::RegDirty vp_rd{};
                    vp_rd.track_all_blocks = false;
                    for (const u32 r : {0xA00Cu, 0xA080u, 0xA090u, 0xA094u, 0xA115u, 0xA16Eu}) {
                        vp_rd.Clear();
                        vp_rd.Mark(r, 1u);
                        assert((vp_rd.stage_b_mask[0] & Gcn::RegDirty::kStageBVpUnaliasedMask0) !=
                               0ULL);
                    }
                    for (const u32 r : {0xA1E0u, 0xA200u, 0xA10Bu, 0xA10Cu, 0xA2DFu, 0xA08Eu}) {
                        vp_rd.Clear();
                        vp_rd.Mark(r, 1u);
                        assert((vp_rd.stage_b_mask[0] & Gcn::RegDirty::kStageBVpUnaliasedMask0) ==
                               0ULL);
                    }

                    // Duplicate Op 38 kind=0/1/2 with same ID must skip cleanly when !track_all.
                    vp_rd.Clear();
                    const u32 saved_a204 = regs.reg_array[0xA204u];
                    regs.reg_array[0xA204u] = 0x13579BDFu;
                    assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                        regs, pkt_rs, csum, &vp_rd, false, &pso_ep, &dyn_ep, &non_sh_ep,
                        ud_ep.data(), &flat_inv, &ds_track));
                    assert(regs.reg_array[0xA204u] == 0x13579BDFu && vp_rd.stage_b_mask[1] == 0u);
                    regs.reg_array[0xA204u] = saved_a204;
                    std::puts("Stage 4A precompiled DirectStates dynamic state + VP split PASS");

                    // Stage 4B: Verify Op 38 defers Stage-B chunks 82/91/92 (RS), 91 (BS 0xA2DC),
                    // and 33 (DS 0xA10B when 0xA10C/D unchanged) under `!track_all_blocks`, sets
                    // `rs_chunks_deferred`/`ds_chunks_deferred`, and still marks everything under
                    // `track_all_blocks == true` (`BB_PIPE_VERIFY=1`).
                    {
                        Gcn::RegDirty d4b{};
                        d4b.track_all_blocks = false;
                        d4b.Clear();
                        ds_track.rs_id = 0u;
                        ds_track.bs_id = 0u;
                        ds_track.ds_id = 0u;
                        regs.reg_array[0xA292u] ^= 0x3u;
                        regs.reg_array[0xA2DFu] ^= 0x1u;
                        regs.reg_array[0xA2E0u] ^= 0x1u;
                        regs.reg_array[0xA2DCu] ^= 0x1u;
                        regs.reg_array[0xA10Bu] ^= 0x1u;
                        // Keep 0xA10C/D already matching `did1` masks so chunk 33 stays clean.
                        const u32 st_bits = ds_read.stencil_op_control & 0x00FFFF00u;
                        regs.reg_array[0xA10Cu] =
                            (regs.reg_array[0xA10Cu] & 0xFF0000FFu) | st_bits;
                        regs.reg_array[0xA10Du] =
                            (regs.reg_array[0xA10Du] & 0xFF0000FFu) | st_bits;

                        assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                            regs, pkt_rs, csum, &d4b, false, &pso_ep, &dyn_ep, &non_sh_ep,
                            ud_ep.data(), &flat_inv, &ds_track));
                        assert(d4b.rs_chunks_deferred);
                        const u64 rs_defer_bits = (1ULL << (82u - 64u)) | (1ULL << (91u - 64u)) |
                                                  (1ULL << (92u - 64u));
                        assert((d4b.stage_b_mask[1] & rs_defer_bits) == 0ULL);

                        assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                            regs, pkt_bs, csum, &d4b, false, &pso_ep, &dyn_ep, &non_sh_ep,
                            ud_ep.data(), &flat_inv, &ds_track));
                        assert((d4b.stage_b_mask[1] & (1ULL << (91u - 64u))) == 0ULL);

                        assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                            regs, pkt_ds, csum, &d4b, false, &pso_ep, &dyn_ep, &non_sh_ep,
                            ud_ep.data(), &flat_inv, &ds_track));
                        assert(d4b.ds_chunks_deferred);
                        assert((d4b.stage_b_mask[0] & (1ULL << 33u)) == 0ULL);

                        // track_all_blocks == true must mark all deferred chunks.
                        d4b.track_all_blocks = true;
                        d4b.Clear();
                        ds_track.rs_id = 0u;
                        ds_track.ds_id = 0u;
                        assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                            regs, pkt_rs, csum, &d4b, false, &pso_ep, &dyn_ep, &non_sh_ep,
                            ud_ep.data(), &flat_inv, &ds_track));
                        assert(BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                            regs, pkt_ds, csum, &d4b, false, &pso_ep, &dyn_ep, &non_sh_ep,
                            ud_ep.data(), &flat_inv, &ds_track));
                        assert((d4b.stage_b_mask[1] & rs_defer_bits) == rs_defer_bits);
                        assert((d4b.stage_b_mask[0] & (1ULL << 33u)) != 0ULL);
                        std::puts("Stage 4B deferred RS/BS/DS DrawPipe chunks PASS");
                    }

                    // Stage 4C: Compile-time proof that `gx_direct_vk::kExcludedDfmtMaskByNfmt`
                    // mirrors `Vulkan::DirectPso::kExcludedDfmtMaskByNfmt` bit-for-bit, plus
                    // exhaustive `SrvSharpSpecEquivalent` (T#, V# with non-zero mtype, null) and
                    // `StageUserDataBatch::Flush` contiguous/sparse coalescing verification.
                    {
                        constexpr auto kCheckExcludedParity = []() constexpr -> bool {
                            for (u32 nf = 0u; nf < 16u; ++nf) {
                                if (gx_direct_vk::kExcludedDfmtMaskByNfmt[nf] !=
                                    DirectPso::kExcludedDfmtMaskByNfmt[nf]) {
                                    return false;
                                }
                            }
                            return true;
                        }();
                        static_assert(kCheckExcludedParity,
                                      "gx_direct_vk::kExcludedDfmtMaskByNfmt out of sync with DirectPso");

                        auto make_tsharp = [](u32 addr_lo, Gcn::NumberFormat nf, Gcn::DataFormat df,
                                              Gcn::ImageType ty) -> std::array<u32, 8> {
                            std::array<u32, 8> t{};
                            t[0] = addr_lo;
                            t[1] = (static_cast<u32>(nf) << 26) | (static_cast<u32>(df) << 20);
                            t[3] = static_cast<u32>(ty) << 28;
                            return t;
                        };
                        auto make_vsharp = [](u32 addr_lo, u32 stride, u32 num_rec, u32 dw3_low,
                                              u32 mtype) -> std::array<u32, 8> {
                            std::array<u32, 8> v{};
                            v[0] = addr_lo;
                            v[1] = (stride & 0x3FFFu) << 16;
                            v[2] = num_rec;
                            v[3] = (dw3_low & 0x007FFFFFu) | ((mtype & 7u) << 27);
                            return v;
                        };

                        const auto t_bc7_unorm = make_tsharp(0x1000u, Gcn::NumberFormat::Unorm,
                                                             Gcn::DataFormat::FormatBc7,
                                                             Gcn::ImageType::Color2D);
                        const auto t_bc1_unorm = make_tsharp(0x2000u, Gcn::NumberFormat::Unorm,
                                                             Gcn::DataFormat::FormatBc1,
                                                             Gcn::ImageType::Color2D);
                        const auto t_rgba_unorm = make_tsharp(0x3000u, Gcn::NumberFormat::Unorm,
                                                              Gcn::DataFormat::Format8_8_8_8,
                                                              Gcn::ImageType::Color2D);
                        const auto t_f32_unorm_a = make_tsharp(0x4000u, Gcn::NumberFormat::Unorm,
                                                               Gcn::DataFormat::Format32,
                                                               Gcn::ImageType::Color2D);
                        const auto t_f32_unorm_b = make_tsharp(0x5000u, Gcn::NumberFormat::Unorm,
                                                               Gcn::DataFormat::Format32,
                                                               Gcn::ImageType::Color2D);
                        const auto t_bc7_srgb = make_tsharp(0x6000u, Gcn::NumberFormat::Srgb,
                                                            Gcn::DataFormat::FormatBc7,
                                                            Gcn::ImageType::Color2D);
                        const auto t_bc1_srgb = make_tsharp(0x7000u, Gcn::NumberFormat::Srgb,
                                                            Gcn::DataFormat::FormatBc1,
                                                            Gcn::ImageType::Color2D);
                        const auto t_bc5_srgb = make_tsharp(0x8000u, Gcn::NumberFormat::Srgb,
                                                            Gcn::DataFormat::FormatBc5,
                                                            Gcn::ImageType::Color2D);
                        const auto t_bc7_cube = make_tsharp(0x9000u, Gcn::NumberFormat::Unorm,
                                                            Gcn::DataFormat::FormatBc7,
                                                            Gcn::ImageType::Cube);
                        const auto t_null_2d = make_tsharp(0u, Gcn::NumberFormat::Unorm,
                                                           Gcn::DataFormat::FormatBc7,
                                                           Gcn::ImageType::Color2D);
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_bc1_unorm.data()));
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_rgba_unorm.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_f32_unorm_a.data()));
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(t_f32_unorm_a.data(), t_f32_unorm_b.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_bc7_srgb.data()));
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_srgb.data(), t_bc1_srgb.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_srgb.data(), t_bc5_srgb.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_bc7_cube.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(t_bc7_unorm.data(), t_null_2d.data()));
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(t_null_2d.data(), t_null_2d.data()));

                        // Buffer SRV V# with non-zero mtype (bits 29:27 = 5 -> 0x28000000) must NOT
                        // be misclassified as T#, and must detect stride/swizzle/format/null diffs.
                        const auto v_a = make_vsharp(0x1000u, 16u, 64u, 0x00020FACu, 5u);
                        const auto v_b = make_vsharp(0x9000u, 16u, 128u, 0x00020FACu, 2u);
                        const auto v_stride = make_vsharp(0x1000u, 32u, 64u, 0x00020FACu, 5u);
                        const auto v_fmt = make_vsharp(0x1000u, 16u, 64u, 0x00030FACu, 5u);
                        const auto v_null = make_vsharp(0x1000u, 16u, 0u, 0x00020FACu, 5u);
                        assert(gx_direct_vk::SrvSharpSpecEquivalent(v_a.data(), v_b.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(v_a.data(), v_stride.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(v_a.data(), v_fmt.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(v_a.data(), v_null.data()));
                        assert(!gx_direct_vk::SrvSharpSpecEquivalent(v_a.data(), t_bc7_unorm.data()));

                        // Verify StageUserDataBatch contiguous (offset > 0) and sparse Flush.
                        std::array<u32, 32> dcb_buf{};
                        struct MockDcb {
                            u32* begin;
                            u32* end;
                            u32* cmd;
                        } mock_dcb{dcb_buf.data(), dcb_buf.data() + dcb_buf.size(), dcb_buf.data()};
                        gx_direct_vk::StageUserDataBatch batch{};
                        const u32 seq_vals[4] = {0x11u, 0x22u, 0x33u, 0x44u};
                        batch.SetDwords(&mock_dcb, 1u, 2u, seq_vals, 4u); // contiguous SGPR 2..5
                        batch.Flush(&mock_dcb, 1u, /*spec_invariant=*/true);
                        assert(mock_dcb.cmd == dcb_buf.data() + 8u);
                        assert((dcb_buf[2] & gx_direct_vk::kUdFlagSpecInvariant) != 0u);
                        assert((dcb_buf[3] >> 16) == 0x003Cu);
                        assert(dcb_buf[4] == 0x11u && dcb_buf[5] == 0x22u &&
                               dcb_buf[6] == 0x33u && dcb_buf[7] == 0x44u);

                        u32* const sparse_start = mock_dcb.cmd;
                        batch.SetDword(&mock_dcb, 1u, 0u, 0xAAu);
                        batch.SetDword(&mock_dcb, 1u, 5u, 0xBBu);
                        batch.Flush(&mock_dcb, 1u, /*spec_invariant=*/false);
                        assert(mock_dcb.cmd == sparse_start + 6u);
                        assert((sparse_start[2] & gx_direct_vk::kUdFlagSpecInvariant) == 0u);
                        assert((sparse_start[3] >> 16) == 0x0021u);
                        assert(sparse_start[4] == 0xAAu && sparse_start[5] == 0xBBu);
                        std::puts("Stage 4C SrvSharpSpecEquivalent + StageUserDataBatch coalescing PASS");
                    }

                    // Stage 5 / Step 1: Verify lock-free DirectTextures::ReadNativeSrvByUid &
                    // ReadNativeSamplerByUid round-trip + upgraded bb_native_stream execution
                    // payloads (`SetDynamicState` 84B, `SetVertexBuffers` 24B, `BindStageResources`
                    // with inline `ud_regs` + `NativeSrvBinding` 12B + `NativeConstBinding` 24B).
                    {
                        std::array<u32, 8> raw_tsharp{
                            0x12345600u,
                            (static_cast<u32>(Gcn::NumberFormat::Unorm) << 26) |
                                (static_cast<u32>(Gcn::DataFormat::FormatBc7) << 20),
                            0x03FF03FFu,
                            static_cast<u32>(Gcn::ImageType::Color2D) << 28,
                            0u,
                            0u,
                            0u,
                            0u,
                        };
                        const std::array<u32, 8> expected_clean_t = raw_tsharp;
                        const u16 srv_uid =
                            VideoCore::DirectTextures::RegisterNativeSrv(raw_tsharp.data(), 77u, true);
                        assert(srv_uid != 0u);
                        const u32 tagged_dw7 = raw_tsharp[7];
                        assert((tagged_dw7 >> 24) == VideoCore::DirectTextures::kSrvTagMagic);
                        const u8 srv_gen = static_cast<u8>((tagged_dw7 >> 16) & 0xFFu);
                        Gcn::Image read_t{};
                        assert(VideoCore::DirectTextures::ReadNativeSrvByUid(srv_uid, srv_gen, read_t));
                        assert(std::memcmp(&read_t, expected_clean_t.data(), sizeof(read_t)) == 0);
                        assert(!VideoCore::DirectTextures::ReadNativeSrvByUid(
                            srv_uid, static_cast<u8>(srv_gen ^ 1u), read_t));

                        std::array<u32, 4> raw_ssharp{0x00001234u, 0x00005678u, 0x00112233u, 0x0000000Au};
                        const std::array<u32, 4> expected_clean_s = raw_ssharp;
                        const u16 smp_uid =
                            VideoCore::DirectTextures::RegisterNativeSampler(raw_ssharp.data());
                        assert(smp_uid != 0u);
                        Gcn::Sampler read_s{};
                        assert(VideoCore::DirectTextures::ReadNativeSamplerByUid(smp_uid, read_s));
                        assert(std::memcmp(&read_s, expected_clean_s.data(), sizeof(read_s)) == 0);

                        std::array<u32, 128> nstream{};
                        bb_native_stream::NativeStreamWriter nw(nstream.data(), u32(nstream.size()));
                        bb_native_stream::SetDynamicStatePayload dp{};
                        dp.vp_xscale = 1280.0f;
                        dp.vp_xoffset = 1280.0f;
                        dp.vp_yscale = -720.0f;
                        dp.vp_yoffset = 720.0f;
                        dp.vp_zmin = 0.0f;
                        dp.vp_zmax = 1.0f;
                        dp.sc_left = 0;
                        dp.sc_top = 0;
                        dp.sc_right = 2560;
                        dp.sc_bottom = 1440;
                        dp.stencil_ref_front = 0x01FF015Au;
                        dp.stencil_ref_back = 0x02EE02A5u;
                        dp.line_width = 1.0f;
                        dp.ta_bc_base = 0xCAFEu;
                        assert(nw.Emit(bb_native_stream::CmdOp::SetDynamicState, 0xFFu, dp));

                        u32* vout = nw.Reserve(bb_native_stream::CmdOp::SetVertexBuffers,
                                               bb_native_stream::kVbFlagIndexed, 10u);
                        assert(vout != nullptr);
                        bb_native_stream::SetVertexBuffersHeader vh{0x1100008000ULL, 0xFFFFu, 1u,
                                                                    2u, 1u, 0u};
                        std::memcpy(vout, &vh, sizeof(vh));
                        Gcn::Buffer vb0{};
                        vb0.base_address = 0x1100009000ULL;
                        vb0.stride = 32u;
                        vb0.num_records = 100u;
                        std::memcpy(vout + 6u, &vb0, sizeof(vb0));

                        u32* bout = nw.Reserve(bb_native_stream::CmdOp::BindStageResources, 2u,
                                               1u + 2u + 3u + 1u + 6u);
                        assert(bout != nullptr);
                        bb_native_stream::BindStageResourcesHeader bh{0u, 1u, 1u, 1u};
                        std::memcpy(bout, &bh, sizeof(bh));
                        bout[1] = 0x11112222u;
                        bout[2] = 0x33334444u;
                        bb_native_stream::NativeSrvBinding sb{tagged_dw7, raw_tsharp[0], raw_tsharp[1]};
                        std::memcpy(bout + 3u, &sb, sizeof(sb));
                        bout[6] = smp_uid;
                        bb_native_stream::NativeConstBinding cb{
                            0x110000A000ULL, 256u, 512u, 0u, 0u, 1u};
                        std::memcpy(bout + 7u, &cb, sizeof(cb));

                        bb_native_stream::NativeStreamReader nr(nw.Data(), nw.SizeDwords());
                        bb_native_stream::CmdHeader h{};
                        const u32* p = nullptr;
                        assert(nr.Next(h, p) &&
                               static_cast<bb_native_stream::CmdOp>(h.op) ==
                                   bb_native_stream::CmdOp::SetDynamicState &&
                               h.flags == 0xFFu);
                        bb_native_stream::SetDynamicStatePayload dp_out{};
                        std::memcpy(&dp_out, p, sizeof(dp_out));
                        assert(dp_out.stencil_ref_front == 0x01FF015Au &&
                               dp_out.stencil_ref_back == 0x02EE02A5u &&
                               dp_out.ta_bc_base == 0xCAFEu);
                        assert(nr.Next(h, p) &&
                               static_cast<bb_native_stream::CmdOp>(h.op) ==
                                   bb_native_stream::CmdOp::SetVertexBuffers);
                        bb_native_stream::SetVertexBuffersHeader vh_out{};
                        std::memcpy(&vh_out, p, sizeof(vh_out));
                        assert(vh_out.index_va == 0x1100008000ULL && vh_out.num_vbs == 1u);
                        assert(nr.Next(h, p) &&
                               static_cast<bb_native_stream::CmdOp>(h.op) ==
                                   bb_native_stream::CmdOp::BindStageResources &&
                               h.flags == 2u);
                        assert(p[1] == 0x11112222u && p[2] == 0x33334444u);
                        bb_native_stream::NativeConstBinding cb_out{};
                        std::memcpy(&cb_out, p + 7u, sizeof(cb_out));
                        assert(cb_out.gpu_va == 0x110000A000ULL && cb_out.ring_offset == 512u &&
                               cb_out.ring_id == 0u);
                        assert(!nr.Next(h, p) && nr.Empty());
                        std::puts("Stage 5 Step 1 DirectTextures UID readers + NativeCmdStream exec PASS");

                        // Stage 5 Steps P1+P2: Lock-free `DirectStates::Intern*Desc` deduplication
                        // + `CmdOp::BeginRenderPass` (CB + DB + `kPassFlagHasCb0Hint`) round-trip.
                        {
                            DirectStates::RasterStateDesc irs{};
                            irs.primitive_setup = 0x00010203u;
                            irs.clip_control = 0x00001007u;
                            irs.msaa_enable = 1u;
                            irs.scissor_enable = 1u;
                            const u16 rid_a = DirectStates::InternRasterStateDesc(irs);
                            const u16 rid_b = DirectStates::InternRasterStateDesc(irs);
                            assert(rid_a > 0u && rid_a == rid_b);
                            const auto* cr = DirectStates::GetCompiledRasterDyn(rid_a);
                            assert(cr != nullptr && (cr->vp_flags & 4u) != 0u);
                            irs.scissor_enable = 0u;
                            const u16 rid_c = DirectStates::InternRasterStateDesc(irs);
                            assert(rid_c > 0u && rid_c != rid_a);

                            DirectStates::BlendStateDesc ibs{};
                            ibs.render_target_channel_mask = 0x0000000Fu;
                            ibs.blend_control[0] = 0x40010001u;
                            const u16 bid_a = DirectStates::InternBlendStateDesc(ibs);
                            assert(bid_a > 0u && bid_a == DirectStates::InternBlendStateDesc(ibs));

                            DirectStates::DepthStencilStateDesc ids{};
                            ids.depth_control = 0x00000017u;
                            ids.stencil_control = 0x00010203u;
                            ids.stencil_op_control = 0x00FFFF00u;
                            const u16 did_a = DirectStates::InternDepthStencilStateDesc(ids);
                            assert(did_a > 0u &&
                                   did_a == DirectStates::InternDepthStencilStateDesc(ids));
                            const auto* cd = DirectStates::GetCompiledDepthStencilDyn(did_a);
                            assert(cd != nullptr && cd->depth_enable && cd->stencil_enable);

                            // Round-trip `BeginRenderPass` (depth-only with `kPassFlagHasCb0Hint`).
                            std::array<u32, 128> rp_buf{};
                            bb_native_stream::NativeStreamWriter rpw(rp_buf.data(),
                                                                     u32(rp_buf.size()));
                            constexpr u16 rp_dw =
                                u16((sizeof(bb_native_stream::BeginRenderPassHeader) / 4u) + 16u +
                                    9u);
                            u32* rout = rpw.Reserve(
                                bb_native_stream::CmdOp::BeginRenderPass,
                                u8(bb_native_stream::kPassFlagHasDepth |
                                   bb_native_stream::kPassFlagHasCb0Hint),
                                rp_dw);
                            assert(rout != nullptr);
                            bb_native_stream::BeginRenderPassHeader ph{};
                            ph.cb_uid[0] = 9u;
                            ph.db_uid = 14u;
                            ph.active_cb_mask = 0u;
                            ph.color_control = 0x00CC0010u;
                            ph.num_indices = 6u;
                            std::memcpy(rout, &ph, sizeof(ph));
                            u32* tp = rout + (sizeof(ph) / 4u);
                            Gcn::ColorBuffer cb0{};
                            cb0.base_address = 0x112233u;
                            std::memcpy(tp, &cb0, sizeof(cb0));
                            tp[15] = 0x0437077Fu;
                            tp += 16u;
                            Gcn::DepthBuffer db0{};
                            db0.z_read_base = 0x445566u;
                            std::memcpy(tp, &db0, sizeof(db0));
                            tp[8] = 0x0437077Fu;

                            bb_native_stream::NativeStreamReader rpr(rpw.Data(), rpw.SizeDwords());
                            bb_native_stream::CmdHeader rh{};
                            const u32* rp = nullptr;
                            assert(rpr.Next(rh, rp) &&
                                   static_cast<bb_native_stream::CmdOp>(rh.op) ==
                                       bb_native_stream::CmdOp::BeginRenderPass &&
                                   rh.flags == (bb_native_stream::kPassFlagHasDepth |
                                                bb_native_stream::kPassFlagHasCb0Hint));
                            bb_native_stream::BeginRenderPassHeader ph_out{};
                            std::memcpy(&ph_out, rp, sizeof(ph_out));
                            assert(ph_out.cb_uid[0] == 9u && ph_out.db_uid == 14u &&
                                   ph_out.num_indices == 6u);
                            const u32* rtp = rp + (sizeof(ph_out) / 4u);
                            Gcn::ColorBuffer cb0_out{};
                            std::memcpy(&cb0_out, rtp, sizeof(cb0_out));
                            assert(cb0_out.base_address == 0x112233u && rtp[15] == 0x0437077Fu);
                            rtp += 16u;
                            Gcn::DepthBuffer db0_out{};
                            std::memcpy(&db0_out, rtp, sizeof(db0_out));
                            assert(db0_out.z_read_base == 0x445566u && rtp[8] == 0x0437077Fu);
                            assert(!rpr.Next(rh, rp) && rpr.Empty());
                            std::puts(
                                "Stage 5 Steps P1+P2 DirectStates::Intern*Desc + BeginRenderPass PASS");
                        }
                    }
                }
            }
        }
    }

    scheduler.Finish();
    vmaDestroyBuffer(instance.GetAllocator(),readback,allocation);
}
