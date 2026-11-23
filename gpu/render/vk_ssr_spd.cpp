// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2 (v1.2.4): Single-Pass Downsampler (SPD) in LDS for Bloodborne's SSR Glossy Mip Pyramid.
#include "render/vk_ssr_spd.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>

#include "bbgpu/shaders/ssr_glossy_spd_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "textures/direct_textures.h"
#include "textures/image.h"
#include "textures/texture_cache.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"
#include "vulkan/platform.h"

namespace Vulkan {

namespace {

struct SsrSpdPush {
    u32 mip1_w, mip1_h;
    u32 mip2_w, mip2_h;
    u32 mip3_w, mip3_h;
    u32 mip4_w, mip4_h;
    u32 mip5_w, mip5_h;
    float inv_mip1_w, inv_mip1_h;
    u32 gen_mips;
    u32 pad0;
};
static_assert(sizeof(SsrSpdPush) == 56u);

} // namespace

SsrSpdCompute::SsrSpdCompute(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                             VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_} {}

SsrSpdCompute::~SsrSpdCompute() {
    const vk::Device device = instance.GetDevice();
    for (auto& v : cached_mip_views) {
        if (v) {
            device.destroyImageView(v);
            v = VK_NULL_HANDLE;
        }
    }
}

bool SsrSpdCompute::EnsureInit() {
    if (initialized) {
        return true;
    }
    if (init_failed) {
        return false;
    }
    const vk::Device device = instance.GetDevice();
    linear_sampler = Check<"create ssr spd linear sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }));

    const std::array<vk::DescriptorSetLayoutBinding, 7> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 3,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 4,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 5,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 6,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    ds_layout = Check<"create ssr spd ds layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange pc_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(SsrSpdPush),
    };
    const vk::DescriptorSetLayout dsl = *ds_layout;
    pipe_layout = Check<"create ssr spd pipe layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    }));
    const vk::ShaderModule mod = CompileSPV(SSR_GLOSSY_SPD_COMP, device);
    pipeline = Check<"create ssr spd pipeline">(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = mod, .pName = "main"},
                .layout = *pipe_layout,
            }));
    device.destroyShaderModule(mod);
    initialized = true;
    return true;
}

void SsrSpdCompute::EnsureMipViews(vk::Image image, u64 image_uid, vk::Format format,
                                   u32 total_levels) {
    const u32 want_levels = std::min(total_levels, 6u);
    if (cached_image == image && cached_uid == image_uid && cached_format == format &&
        cached_levels >= want_levels) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    auto old_views = cached_mip_views;
    cached_mip_views.fill(VK_NULL_HANDLE);
    scheduler.DeferOperation([device, old_views]() {
        for (vk::ImageView v : old_views) {
            if (v) {
                device.destroyImageView(v);
            }
        }
    });
    for (u32 mip = 0; mip < want_levels; ++mip) {
        cached_mip_views[mip] = Check<"create ssr spd mip view">(device.createImageView({
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1},
        }));
    }
    cached_image = image;
    cached_uid = image_uid;
    cached_format = format;
    cached_levels = want_levels;
}

bool SsrSpdCompute::Execute(const BbCp::DirectVk::GenerateMipsSpdArgs& args) {
    if (!EnsureInit() || args.num_mips < 2u) {
        return false;
    }
    Gcn::Image tsharp{};
    std::memcpy(&tsharp, args.tex_tsharp, sizeof(tsharp));
    u8 gen8 = 0u;
    (void)VideoCore::DirectTextures::UnpackAndCleanSrv(tsharp, gen8);
    if (!tsharp.Valid()) {
        return false;
    }
    Shader::ImageResource res{};
    res.is_written = true;
    res.needs_native = false;
    VideoCore::TextureCache::ImageDesc desc{tsharp, res};
    const VideoCore::ImageId img_id = texture_cache.FindImage(desc);
    if (!img_id) {
        return false;
    }
    auto& img = texture_cache.GetImage(img_id);
    const u32 total_levels = img.info.resources.levels;
    if (total_levels < 2u) {
        return false;
    }
    const u32 gen_mips = std::min({args.num_mips - 1u, total_levels - 1u, 5u});
    const u32 base_w = img.info.size.width;
    const u32 base_h = img.info.size.height;
    if (base_w < 4u || base_h < 4u) {
        return false;
    }

    scheduler.EndRendering();
    if (VideoCore::True(img.flags & VideoCore::ImageFlagBits::DirectImmutable)) {
        texture_cache.DemoteDirectImmutable(img_id);
    }

    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Ssr_HiZAndMips);

    const bool can_spd =
        img.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
        bool(img.usage_flags & vk::ImageUsageFlagBits::eStorage);

    if (can_spd) {
        EnsureMipViews(img.GetImage(), img.image_uid, img.info.pixel_format, gen_mips + 1u);

        const VideoCore::SubresourceRange mip0_range{{0u, 0u}, {1u, 1u}};
        const VideoCore::SubresourceRange dst_range{{1u, 0u}, {gen_mips, 1u}};
        runtime.Transit(&img, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead, mip0_range);
        runtime.Transit(&img, vk::ImageLayout::eGeneral,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderWrite, dst_range);
        runtime.FlushBarriers();

        if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
            prof->Mark(0x9A71FE30ULL, [base_w, base_h, gen_mips] {
                return fmt::format("SsrSpdCompute: Glossy Mips 1..{} ({}x{}) (LDS) [VULKAN-NATIVE]",
                                   gen_mips, base_w, base_h);
            });
        }

        const auto mip_dim = [](u32 v, u32 m) noexcept { return std::max(v >> m, 1u); };
        const u32 m1_w = mip_dim(base_w, 1u);
        const u32 m1_h = mip_dim(base_h, 1u);
        const SsrSpdPush pc{
            .mip1_w = m1_w,
            .mip1_h = m1_h,
            .mip2_w = mip_dim(base_w, 2u),
            .mip2_h = mip_dim(base_h, 2u),
            .mip3_w = mip_dim(base_w, 3u),
            .mip3_h = mip_dim(base_h, 3u),
            .mip4_w = mip_dim(base_w, 4u),
            .mip4_h = mip_dim(base_h, 4u),
            .mip5_w = mip_dim(base_w, 5u),
            .mip5_h = mip_dim(base_h, 5u),
            .inv_mip1_w = 1.0f / float(m1_w),
            .inv_mip1_h = 1.0f / float(m1_h),
            .gen_mips = gen_mips,
            .pad0 = 0u,
        };
        const vk::ImageView v0 = cached_mip_views[0];
        std::array<vk::ImageView, 5> dst_views{};
        for (u32 i = 0; i < 5u; ++i) {
            dst_views[i] = (i < gen_mips) ? cached_mip_views[i + 1u] : cached_mip_views[1];
        }
        const vk::Sampler samp = *linear_sampler;
        const vk::Pipeline pipe = *pipeline;
        const vk::PipelineLayout layout = *pipe_layout;

        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
            const vk::DescriptorImageInfo src_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = v0,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo samp_info{
                .sampler = samp,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = vk::ImageLayout::eUndefined,
            };
            std::array<vk::DescriptorImageInfo, 5> dst_infos{};
            for (u32 i = 0; i < 5u; ++i) {
                dst_infos[i] = {
                    .sampler = VK_NULL_HANDLE,
                    .imageView = dst_views[i],
                    .imageLayout = vk::ImageLayout::eGeneral,
                };
            }
            const std::array<vk::WriteDescriptorSet, 7> writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &src_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_infos[0]},
                {.dstBinding = 3,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_infos[1]},
                {.dstBinding = 4,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_infos[2]},
                {.dstBinding = 5,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_infos[3]},
                {.dstBinding = 6,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_infos[4]},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SsrSpdPush),
                                 &pc);
            cmdbuf.dispatch((m1_w + 31u) / 32u, (m1_h + 31u) / 32u, 1u);
        });
    } else {
        // Generic non-RGBA16F fallback: sequential hardware linear blits.
        for (u32 mip = 1u; mip <= gen_mips; ++mip) {
            const VideoCore::SubresourceRange src_r{{mip - 1u, 0u}, {1u, 1u}};
            const VideoCore::SubresourceRange dst_r{{mip, 0u}, {1u, 1u}};
            runtime.Transit(&img, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eBlit,
                            vk::AccessFlagBits2::eTransferRead, src_r);
            runtime.Transit(&img, vk::ImageLayout::eTransferDstOptimal,
                            vk::PipelineStageFlagBits2::eBlit,
                            vk::AccessFlagBits2::eTransferWrite, dst_r);
            runtime.FlushBarriers();
            const s32 sw = s32(std::max(base_w >> (mip - 1u), 1u));
            const s32 sh = s32(std::max(base_h >> (mip - 1u), 1u));
            const s32 dw = s32(std::max(base_w >> mip, 1u));
            const s32 dh = s32(std::max(base_h >> mip, 1u));
            const vk::Image vk_img = img.GetImage();
            const vk::ImageBlit blit{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, mip - 1u, 0, 1},
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{sw, sh, 1}},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 1},
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{dw, dh, 1}},
            };
            scheduler.Record([vk_img, blit](vk::CommandBuffer cmdbuf) {
                cmdbuf.blitImage(vk_img, vk::ImageLayout::eTransferSrcOptimal, vk_img,
                                 vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
            });
        }
    }

    img.gpu_write_tick = scheduler.CurrentTick();
    scheduler.BumpDescriptorStateEpoch();
    return true;
}

} // namespace Vulkan
