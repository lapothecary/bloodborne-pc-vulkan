// SPDX-License-Identifier: GPL-2.0-or-later
// Execute production camera motion SPIR-V: distant geometry, sky and stale object vectors.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "vulkan/instance.h"
#include "render/scheduler.h"
#include "shim/bbport_cloth_interp.h"
#include "cloth_exp/cloth_exp_local_fp64.h"
#include "cloth_exp/cloth_exp_xpbd.h"
#include "test_vk_image.h"
#include "upscale/vk_camera_motion.h"
#include "third_party/fsr-vulkan/include/ffx_vk_portable.h"
#include "bbgpu/shaders/camera_motion_comp.h"
#include <vk_mem_alloc.h>

int main() {
    Vulkan::Instance instance(0, false);
    static vk::detail::DynamicLoader loader;
    vk::detail::DispatchLoaderDynamic d;
    d.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
    d.init(instance.GetInstance()); d.init(instance.GetDevice());
    const auto device = instance.GetDevice();
    Vulkan::Scheduler scheduler(instance);
    constexpr u32 W=9,H=7,N=W*H;
    std::array<TestImage,3> images;
    std::array<vk::UniqueImageView,3> views;
    std::array<vk::DescriptorSetLayoutBinding,3> bindings{};
    for (u32 i=0;i<3;++i) {
        const auto format=i==1 ? vk::Format::eR16G16Sfloat : vk::Format::eR32G32B32A32Sfloat;
        images[i]=TestImage(device,instance.GetAllocator());
        images[i].Create({.imageType=vk::ImageType::e2D,.format=format,.extent={W,H,1},
            .mipLevels=1,.arrayLayers=1,.samples=vk::SampleCountFlagBits::e1,
            .tiling=vk::ImageTiling::eOptimal,.usage=vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst});
        views[i]=device.createImageViewUnique({.image=vk::Image(images[i]),
            .viewType=vk::ImageViewType::e2D,.format=format,
            .subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1}},nullptr,d).value;
        bindings[i]={.binding=i,.descriptorType=i==1 ? vk::DescriptorType::eStorageImage :
            vk::DescriptorType::eSampledImage,.descriptorCount=1,
            .stageFlags=vk::ShaderStageFlagBits::eCompute};
    }
    auto descriptors=device.createDescriptorSetLayoutUnique({
        .flags=vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount=3,.pBindings=bindings.data()},nullptr,d).value;
    struct Params {
        std::array<float,12> reproject;
        std::array<float,4> proj,prev_proj;
        std::array<float,2> size,jitter,previous_jitter;
        u32 mode;
    };
    const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute,0,sizeof(Params)};
    auto layout=device.createPipelineLayoutUnique({.setLayoutCount=1,.pSetLayouts=&*descriptors,
        .pushConstantRangeCount=1,.pPushConstantRanges=&push},nullptr,d).value;
    auto module=device.createShaderModuleUnique({.codeSize=sizeof(CAMERA_MOTION_COMP),
        .pCode=CAMERA_MOTION_COMP},nullptr,d).value;
    auto pipeline=device.createComputePipelineUnique({}, {
        .stage={.stage=vk::ShaderStageFlagBits::eCompute,.module=*module,.pName="main"},
        .layout=*layout},nullptr,d).value;
    VkBuffer staging{}; VmaAllocation allocation{}; VmaAllocationInfo ai{};
    const VkBufferCreateInfo bi{.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=N*16*2,.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo ac{.flags=VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,.usage=VMA_MEMORY_USAGE_AUTO,
        .requiredFlags=VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    assert(vmaCreateBuffer(instance.GetAllocator(),&bi,&ac,&staging,&allocation,&ai)==VK_SUCCESS);
    const auto barrier=[&](vk::CommandBuffer cmd,vk::PipelineStageFlags2 src,
                          vk::AccessFlags2 src_access,vk::PipelineStageFlags2 dst,
                          vk::AccessFlags2 dst_access) {
        const vk::MemoryBarrier2 b{.srcStageMask=src,.srcAccessMask=src_access,
                                  .dstStageMask=dst,.dstAccessMask=dst_access};
        cmd.pipelineBarrier2({.memoryBarrierCount=1,.pMemoryBarriers=&b},d);
    };
    const auto run=[&](const char* label,float depth,float translation,float rotation,
                       float object_depth,u32 mode,float expected,bool jitter=false,
                       float obj_r=2.f,float obj_g=0.f,float obj_b=1.f) {
        auto* pixels=static_cast<float*>(ai.pMappedData);
        for (u32 p=0;p<N;++p) {
            pixels[p*4]=depth;
            pixels[N*4+p*4]=obj_r; pixels[N*4+p*4+1]=obj_g;
            pixels[N*4+p*4+2]=obj_b; pixels[N*4+p*4+3]=object_depth;
        }
        const auto cmd=scheduler.CommandBuffer();
        std::array<vk::ImageMemoryBarrier2,3> transitions{};
        for (u32 i=0;i<3;++i) transitions[i]={
            .srcStageMask=vk::PipelineStageFlagBits2::eAllCommands,
            .dstStageMask=vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask=vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout=vk::ImageLayout::eUndefined,.newLayout=vk::ImageLayout::eGeneral,
            .image=vk::Image(images[i]),.subresourceRange={vk::ImageAspectFlagBits::eColor,0,1,0,1}};
        cmd.pipelineBarrier2({.imageMemoryBarrierCount=3,.pImageMemoryBarriers=transitions.data()},d);
        for (u32 i : {0u,2u}) {
            const vk::BufferImageCopy region{.bufferOffset=i/2*N*16,
                .imageSubresource={vk::ImageAspectFlagBits::eColor,0,0,1},.imageExtent={W,H,1}};
            cmd.copyBufferToImage(staging,vk::Image(images[i]),vk::ImageLayout::eGeneral,region,d);
        }
        barrier(cmd,vk::PipelineStageFlagBits2::eTransfer,vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader,vk::AccessFlagBits2::eShaderRead);
        std::array<vk::DescriptorImageInfo,3> infos{};
        std::array<vk::WriteDescriptorSet,3> writes{};
        for (u32 i=0;i<3;++i) {
            infos[i]={.imageView=*views[i],.imageLayout=vk::ImageLayout::eGeneral};
            writes[i]={.dstBinding=i,.descriptorCount=1,.descriptorType=bindings[i].descriptorType,
                .pImageInfo=&infos[i]};
        }
        Params params{{1,0,rotation,translation,0,1,0,0,0,0,1,0},
            {1,1,1,-0.05f},{1,1,1,-0.05f},{W,H},
            {jitter ? 0.37f : 0.f,jitter ? -0.29f : 0.f},{0,0},mode};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*pipeline,d);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,*layout,0,writes,d);
        cmd.pushConstants(*layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(params),&params,d);
        cmd.dispatch((W+7)/8,(H+7)/8,1,d);
        barrier(cmd,vk::PipelineStageFlagBits2::eComputeShader,vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eTransfer,vk::AccessFlagBits2::eTransferRead);
        const vk::BufferImageCopy region{.imageSubresource={vk::ImageAspectFlagBits::eColor,0,0,1},
            .imageOffset={4,3,0},.imageExtent={1,1,1}};
        cmd.copyImageToBuffer(vk::Image(images[1]),vk::ImageLayout::eGeneral,staging,region,d);
        barrier(cmd,vk::PipelineStageFlagBits2::eTransfer,vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eHost,vk::AccessFlagBits2::eHostRead);
        scheduler.Finish();
        const auto* result=static_cast<u16*>(ai.pMappedData);
        const auto half=[](u16 v) {
            const int e=(v>>10)&31;
            return (v&0x8000 ? -1.f : 1.f)*(e==0 ? std::ldexp(float(v&1023),-24) :
                std::ldexp(1.f+float(v&1023)/1024.f,e-15));
        };
        assert(std::abs(half(result[0])-expected)<0.002f);
        assert(std::abs(half(result[1]))<0.001f);
        std::printf("Camera motion: %s PASS (%.5f)\n",label,half(result[0]));
    };
    constexpr float distant=0.999995f;
    run("distant geometry moves",distant,1000,0,0,0,4.5f*1000*(1-distant)/0.05f);
    run("sky rotates",1,0,0.1f,0,0,0.45f);
    run("sky ignores translation",1,1000,0,0,0,0);
    run("static camera cancels jitter",distant,0,0,0,0,0,true);
    run("matching object vector",distant,0,0,distant,1,2);
    run("stale distant object vector",distant,0,0,0.9999f,1,0);
    run("Pass 0x2e moving Hunter self-motion",0.5f,0,0,1.f,32u,0.212625f,false,0.30f,0.50f,0.f);
    run("Pass 0x2e static NPC cancels camera pan",distant,1000,0,1.f,32u,
        4.5f*1000*(1-distant)/0.05f,false,0.10f,0.50f,0.f);

    {
        const float theta = 0.25f;
        const float c = std::cos(theta), s = std::sin(theta);
        const float tx = 1.5f, ty = -0.5f, tz = 2.0f;
        const std::array<float, 12> cur_v{1,0,0,0, 0,1,0,0, 0,0,1,0};
        const std::array<float, 12> cur_inv{1,0,0,0, 0,1,0,0, 0,0,1,0};
        const std::array<float, 12> prev_v{c,0,s,tx, 0,1,0,ty, -s,0,c,tz};
        const std::array<float, 12> prev_inv{c,0,-s,-(c*tx-s*tz), 0,1,0,-ty, s,0,c,-(s*tx+c*tz)};
        const std::array<float, 4> proj{1.2f, -1.6f, 1.0001f, -0.1f};
        const auto m = Vulkan::CameraMotion::ComputeDlssgMatricesFromCameras(
            cur_v, cur_inv, proj, prev_v, prev_inv, proj);
        assert(m.valid);
        const auto mul4 = [](const std::array<float, 16>& a, const std::array<float, 16>& b) {
            std::array<float, 16> out{};
            for (int r = 0; r < 4; ++r)
                for (int col = 0; col < 4; ++col)
                    for (int k = 0; k < 4; ++k) out[r * 4 + col] += a[r * 4 + k] * b[k * 4 + col];
            return out;
        };
        const auto mul_row = [](const std::array<float, 4>& v, const std::array<float, 16>& mat) {
            std::array<float, 4> out{};
            for (int col = 0; col < 4; ++col)
                for (int k = 0; k < 4; ++k) out[col] += v[k] * mat[k * 4 + col];
            return out;
        };
        const auto p_ident = mul4(m.camera_view_to_clip, m.clip_to_camera_view);
        const auto c2p_ident = mul4(m.clip_to_prev_clip, m.prev_clip_to_clip);
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                const float e = r == col ? 1.0f : 0.0f;
                assert(std::abs(p_ident[r * 4 + col] - e) < 1e-5f);
                assert(std::abs(c2p_ident[r * 4 + col] - e) < 1e-4f);
            }
        }
        const std::array<float, 4> pt{0.8f, -0.6f, 8.0f, 1.0f};
        const auto cur_clip = mul_row(pt, m.camera_view_to_clip);
        assert(std::abs((cur_clip[1] / cur_clip[3]) - (-(pt[1] * proj[1] / pt[2]))) < 1e-5f);
        const auto repr_clip = mul_row(cur_clip, m.clip_to_prev_clip);
        const std::array<float, 4> prev_pt{c*pt[0]+s*pt[2]+tx, pt[1]+ty, -s*pt[0]+c*pt[2]+tz, 1.f};
        const auto exp_prev_clip = mul_row(prev_pt, m.camera_view_to_clip);
        for (int i = 0; i < 3; ++i) {
            assert(std::abs((repr_clip[i] / repr_clip[3]) - (exp_prev_clip[i] / exp_prev_clip[3])) < 1e-5f);
        }
        std::printf("Camera motion: DLSS-G row-major matrices round-trip PASS\n");
    }

    // Real Vulkan compute verification of FSR 3.1 multi-frame interpolation (t = 0.25, 0.50, 0.75)
    // and 12-frame descriptor/view ring stability (exceeding FFX_MAX_QUEUED_FRAMES = 8).
    {
        FfxVkPortableDeviceCapabilities caps{};
        caps.structSize = sizeof(caps);
        assert(ffxVkPortableQueryDeviceCapabilities(instance.GetPhysicalDevice(), &caps) ==
               FFX_VK_PORTABLE_OK);
        assert(caps.fsr3FrameGenerationPrerequisites);

        FfxVkPortableDeviceInfo fg_dev{};
        fg_dev.structSize = sizeof(fg_dev);
        fg_dev.instance = instance.GetInstance();
        fg_dev.physicalDevice = instance.GetPhysicalDevice();
        fg_dev.device = device;
        fg_dev.getDeviceProcAddr = d.vkGetDeviceProcAddr;
        fg_dev.queue = instance.GetGraphicsQueue();
        fg_dev.queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex();
        fg_dev.shaderFloat16Enabled = instance.IsShaderFloat16Enabled();
        fg_dev.subgroupSizeControlEnabled = instance.IsSubgroupSizeControlEnabled();
        fg_dev.synchronization2Enabled = VK_TRUE;
        fg_dev.shaderStorageImageWriteWithoutFormatEnabled = VK_TRUE;

        constexpr u32 S = 256;
        FfxVkPortableFrameGenerationCreateInfo fg_create{};
        fg_create.structSize = sizeof(fg_create);
        fg_create.maxRenderSize = {S, S};
        fg_create.displaySize = {S, S};
        fg_create.interpolationSourceFormat = VK_FORMAT_R8G8B8A8_UNORM;
        fg_create.outputFormat = VK_FORMAT_R8G8B8A8_UNORM;

        FfxVkPortableFrameGenerationContext* fg_ctx = nullptr;
        assert(ffxVkPortableFrameGenerationContextCreate(&fg_dev, &fg_create, &fg_ctx) ==
               FFX_VK_PORTABLE_OK);
        assert(fg_ctx != nullptr);

        const vk::ImageUsageFlags fg_color_usage = vk::ImageUsageFlagBits::eStorage |
                                                   vk::ImageUsageFlagBits::eSampled |
                                                   vk::ImageUsageFlagBits::eTransferSrc |
                                                   vk::ImageUsageFlagBits::eTransferDst |
                                                   vk::ImageUsageFlagBits::eColorAttachment;
        const vk::ImageUsageFlags fg_depth_usage = vk::ImageUsageFlagBits::eSampled |
                                                   vk::ImageUsageFlagBits::eTransferDst |
                                                   vk::ImageUsageFlagBits::eDepthStencilAttachment;
        const vk::ImageUsageFlags fg_motion_usage = vk::ImageUsageFlagBits::eStorage |
                                                    vk::ImageUsageFlagBits::eSampled |
                                                    vk::ImageUsageFlagBits::eTransferDst;
        const vk::ImageUsageFlags fg_out_usage = vk::ImageUsageFlagBits::eStorage |
                                                 vk::ImageUsageFlagBits::eSampled |
                                                 vk::ImageUsageFlagBits::eTransferSrc |
                                                 vk::ImageUsageFlagBits::eTransferDst;

        const auto make_img = [&](vk::Format fmt, vk::ImageUsageFlags usage) {
            TestImage img(device, instance.GetAllocator());
            img.Create({.imageType = vk::ImageType::e2D,
                        .format = fmt,
                        .extent = {S, S, 1},
                        .mipLevels = 1,
                        .arrayLayers = 1,
                        .samples = vk::SampleCountFlagBits::e1,
                        .tiling = vk::ImageTiling::eOptimal,
                        .usage = usage,
                        .initialLayout = vk::ImageLayout::eUndefined});
            return img;
        };
        TestImage fg_color = make_img(vk::Format::eR8G8B8A8Unorm, fg_color_usage);
        TestImage fg_depth = make_img(vk::Format::eD32Sfloat, fg_depth_usage);
        TestImage fg_motion = make_img(vk::Format::eR16G16Sfloat, fg_motion_usage);
        std::array<TestImage, 3> fg_outs = {
            make_img(vk::Format::eR8G8B8A8Unorm, fg_out_usage),
            make_img(vk::Format::eR8G8B8A8Unorm, fg_out_usage),
            make_img(vk::Format::eR8G8B8A8Unorm, fg_out_usage),
        };

        const auto make_desc = [&](vk::Image img, vk::Format fmt, vk::ImageUsageFlags usage,
                                   vk::ImageAspectFlags aspect, FfxVkPortableResourceState st) {
            FfxVkPortableImage out{};
            out.structSize = sizeof(out);
            out.image = img;
            out.format = static_cast<VkFormat>(fmt);
            out.extent = {S, S};
            out.mipCount = 1;
            out.arrayLayers = 1;
            out.usage = static_cast<VkImageUsageFlags>(usage);
            out.aspect = static_cast<VkImageAspectFlags>(aspect);
            out.state = st;
            return out;
        };

        const FfxVkPortableImage color_desc =
            make_desc(vk::Image(fg_color), vk::Format::eR8G8B8A8Unorm, fg_color_usage,
                      vk::ImageAspectFlagBits::eColor, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        const FfxVkPortableImage depth_desc =
            make_desc(vk::Image(fg_depth), vk::Format::eD32Sfloat, fg_depth_usage,
                      vk::ImageAspectFlagBits::eDepth, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        const FfxVkPortableImage motion_desc =
            make_desc(vk::Image(fg_motion), vk::Format::eR16G16Sfloat, fg_motion_usage,
                      vk::ImageAspectFlagBits::eColor, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        const std::array<FfxVkPortableImage, 3> out_descs = {
            make_desc(vk::Image(fg_outs[0]), vk::Format::eR8G8B8A8Unorm, fg_out_usage,
                      vk::ImageAspectFlagBits::eColor,
                      FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS),
            make_desc(vk::Image(fg_outs[1]), vk::Format::eR8G8B8A8Unorm, fg_out_usage,
                      vk::ImageAspectFlagBits::eColor,
                      FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS),
            make_desc(vk::Image(fg_outs[2]), vk::Format::eR8G8B8A8Unorm, fg_out_usage,
                      vk::ImageAspectFlagBits::eColor,
                      FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS),
        };

        constexpr VkDeviceSize kPlaneBytes = VkDeviceSize(S) * S * 4; // S*S * 4B
        VkBuffer fg_staging{};
        VmaAllocation fg_alloc{};
        VmaAllocationInfo fg_ai{};
        const VkBufferCreateInfo fg_bi{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = kPlaneBytes * 3, // color + depth + motion uploads, OR 3 output readbacks
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        assert(vmaCreateBuffer(instance.GetAllocator(), &fg_bi, &ac, &fg_staging, &fg_alloc, &fg_ai) ==
               VK_SUCCESS);

        const auto float_to_half = [](float v) -> u16 {
            u32 f = 0;
            std::memcpy(&f, &v, sizeof(f));
            const u32 sign = (f >> 16) & 0x8000u;
            const int exp = int((f >> 23) & 0xffu) - 127 + 15;
            const u32 mant = f & 0x7fffffu;
            if (exp <= 0) return u16(sign);
            if (exp >= 31) return u16(sign | 0x7c00u);
            return u16(sign | (u32(exp) << 10) | (mant >> 13));
        };

        const auto submit_fg_frame = [&](u64 fid, int cx, float mv_uv_x, bool reset, u32 n_outs,
                                         const float* times, bool readback) {
            auto* base = static_cast<u8*>(fg_ai.pMappedData);
            auto* rgba = reinterpret_cast<u32*>(base);
            auto* depth_f = reinterpret_cast<float*>(base + kPlaneBytes);
            auto* mv_h = reinterpret_cast<u16*>(base + kPlaneBytes * 2);
            const u16 half_mv = float_to_half(mv_uv_x);
            const u16 half_zero = 0;

            for (u32 y = 0; y < S; ++y) {
                for (u32 x = 0; x < S; ++x) {
                    const u32 idx = y * S + x;
                    const bool inside = (int(x) >= cx - 16 && int(x) < cx + 16 &&
                                         y >= 112 && y < 144);
                    rgba[idx] = inside ? 0xfff0f0f0u : 0xff101010u;
                    depth_f[idx] = inside ? 0.25f : 0.95f;
                    mv_h[idx * 2] = inside ? half_mv : half_zero;
                    mv_h[idx * 2 + 1] = half_zero;
                }
            }

            const auto cmd = scheduler.CommandBuffer();
            const vk::ImageAspectFlags color_asp = vk::ImageAspectFlagBits::eColor;
            const vk::ImageAspectFlags depth_asp = vk::ImageAspectFlagBits::eDepth;
            const auto img_bar = [&](vk::Image img, vk::ImageAspectFlags asp,
                                     vk::PipelineStageFlags2 src_s, vk::AccessFlags2 src_a,
                                     vk::PipelineStageFlags2 dst_s, vk::AccessFlags2 dst_a,
                                     vk::ImageLayout old_l, vk::ImageLayout new_l) {
                const vk::ImageMemoryBarrier2 b{.srcStageMask = src_s,
                                                .srcAccessMask = src_a,
                                                .dstStageMask = dst_s,
                                                .dstAccessMask = dst_a,
                                                .oldLayout = old_l,
                                                .newLayout = new_l,
                                                .image = img,
                                                .subresourceRange = {asp, 0, 1, 0, 1}};
                cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b}, d);
            };
            img_bar(vk::Image(fg_color), color_asp, vk::PipelineStageFlagBits2::eAllCommands, {},
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            img_bar(vk::Image(fg_depth), depth_asp, vk::PipelineStageFlagBits2::eAllCommands, {},
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            img_bar(vk::Image(fg_motion), color_asp, vk::PipelineStageFlagBits2::eAllCommands, {},
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            for (u32 o = 0; o < n_outs; ++o) {
                img_bar(vk::Image(fg_outs[o]), color_asp, vk::PipelineStageFlagBits2::eAllCommands, {},
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderStorageWrite, vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eGeneral);
            }

            cmd.copyBufferToImage(
                fg_staging, vk::Image(fg_color), vk::ImageLayout::eTransferDstOptimal,
                vk::BufferImageCopy{0, 0, 0, {color_asp, 0, 0, 1}, {}, {S, S, 1}}, d);
            cmd.copyBufferToImage(
                fg_staging, vk::Image(fg_depth), vk::ImageLayout::eTransferDstOptimal,
                vk::BufferImageCopy{kPlaneBytes, 0, 0, {depth_asp, 0, 0, 1}, {}, {S, S, 1}}, d);
            cmd.copyBufferToImage(
                fg_staging, vk::Image(fg_motion), vk::ImageLayout::eTransferDstOptimal,
                vk::BufferImageCopy{kPlaneBytes * 2, 0, 0, {color_asp, 0, 0, 1}, {}, {S, S, 1}}, d);

            img_bar(vk::Image(fg_color), color_asp, vk::PipelineStageFlagBits2::eTransfer,
                    vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eGeneral);
            img_bar(vk::Image(fg_depth), depth_asp, vk::PipelineStageFlagBits2::eTransfer,
                    vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eGeneral);
            img_bar(vk::Image(fg_motion), color_asp, vk::PipelineStageFlagBits2::eTransfer,
                    vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eGeneral);

            FfxVkPortableFrameGenerationPrepareInfo prep{};
            prep.structSize = sizeof(prep);
            prep.commandBuffer = cmd;
            prep.depth = depth_desc;
            prep.motionVectors = motion_desc;
            prep.renderSize = {S, S};
            prep.motionVectorScale = {1.0f, 1.0f};
            prep.frameTimeMilliseconds = 16.667f;
            prep.cameraNear = 0.1f;
            prep.cameraFar = 100.0f;
            prep.cameraVerticalFovRadians = 0.8f;
            prep.viewSpaceToMeters = 1.0f;
            prep.minLuminance = 0.0f;
            prep.maxLuminance = 1.0f;
            prep.transferFunction = FFX_VK_PORTABLE_TRANSFER_FUNCTION_SRGB;
            prep.cameraUp = {0.0f, 1.0f, 0.0f};
            prep.cameraRight = {1.0f, 0.0f, 0.0f};
            prep.cameraForward = {0.0f, 0.0f, 1.0f};
            prep.reset = reset ? VK_TRUE : VK_FALSE;
            prep.frameId = fid;

            FfxVkPortableFrameGenerationDispatchInfo disp{};
            disp.structSize = sizeof(disp);
            disp.commandBuffer = cmd;
            disp.currentColor = color_desc;
            disp.hudlessColor.structSize = sizeof(disp.hudlessColor);
            disp.distortionField.structSize = sizeof(disp.distortionField);
            disp.output = out_descs[0];
            disp.displaySize = {S, S};
            disp.interpolationRect = {0, 0, S, S};
            disp.frameTimeMilliseconds = 16.667f;
            disp.cameraNear = 0.1f;
            disp.cameraFar = 100.0f;
            disp.cameraVerticalFovRadians = 0.8f;
            disp.viewSpaceToMeters = 1.0f;
            disp.minLuminance = 0.0f;
            disp.maxLuminance = 1.0f;
            disp.transferFunction = FFX_VK_PORTABLE_TRANSFER_FUNCTION_SRGB;
            disp.reset = reset ? VK_TRUE : VK_FALSE;
            disp.frameId = fid;

            assert(ffxVkPortableFrameGenerationContextPrepare(fg_ctx, &prep, &color_desc) ==
                   FFX_VK_PORTABLE_OK);
            assert(ffxVkPortableFrameGenerationContextRecordMultiDispatch(
                       fg_ctx, &disp, n_outs, out_descs.data(), times) == FFX_VK_PORTABLE_OK);

            if (readback) {
                for (u32 o = 0; o < n_outs; ++o) {
                    img_bar(vk::Image(fg_outs[o]), color_asp,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderStorageWrite,
                            vk::PipelineStageFlagBits2::eTransfer,
                            vk::AccessFlagBits2::eTransferRead, vk::ImageLayout::eGeneral,
                            vk::ImageLayout::eTransferSrcOptimal);
                    cmd.copyImageToBuffer(
                        vk::Image(fg_outs[o]), vk::ImageLayout::eTransferSrcOptimal, fg_staging,
                        vk::BufferImageCopy{kPlaneBytes * o, 0, 0, {color_asp, 0, 0, 1}, {}, {S, S, 1}},
                        d);
                }
                barrier(cmd, vk::PipelineStageFlagBits2::eTransfer,
                        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                        vk::AccessFlagBits2::eHostRead);
            }
            scheduler.Finish();
        };

        const std::array<float, 3> times_4x{0.25f, 0.50f, 0.75f};
        submit_fg_frame(1, 80, 0.0f, true, 1, &times_4x[1], false);
        submit_fg_frame(2, 112, -32.0f, false, 3, times_4x.data(), true);

        const auto* out_pixels = static_cast<const u32*>(fg_ai.pMappedData);
        std::array<float, 3> centroids{};
        for (u32 o = 0; o < 3; ++o) {
            const u32* plane = out_pixels + o * (S * S);
            double sum_w = 0.0, sum_x = 0.0;
            for (u32 y = 112; y < 144; ++y) {
                for (u32 x = 0; x < S; ++x) {
                    const u32 r = plane[y * S + x] & 0xffu;
                    if (r > 128u) {
                        const double w = double(r);
                        sum_w += w;
                        sum_x += w * double(x);
                    }
                }
            }
            assert(sum_w > 0.0);
            centroids[o] = float(sum_x / sum_w);
        }
        std::printf("FSR3 4x centroids: t=0.25 -> %.2f, t=0.50 -> %.2f, t=0.75 -> %.2f\n",
                    centroids[0], centroids[1], centroids[2]);
        std::fflush(stdout);
        assert(std::abs(centroids[0] - 88.0f) < 3.0f);
        assert(std::abs(centroids[1] - 96.0f) < 3.0f);
        assert(std::abs(centroids[2] - 104.0f) < 3.0f);
        assert(centroids[0] + 4.0f < centroids[1] && centroids[1] + 4.0f < centroids[2]);

        for (u64 fid = 3; fid <= 12; ++fid) {
            const int cx = (fid & 1u) ? 80 : 112;
            const float mv = (fid & 1u) ? 32.0f : -32.0f;
            submit_fg_frame(fid, cx, mv, false, 3, times_4x.data(), false);
        }

        assert(ffxVkPortableFrameGenerationContextDestroy(fg_ctx) == FFX_VK_PORTABLE_OK);
        vmaDestroyBuffer(instance.GetAllocator(), fg_staging, fg_alloc);
        std::printf("Camera motion: FSR 3.1 4x multi-subframe GPU interpolation PASS "
                    "(cx=%.1f, %.1f, %.1f; 12f ring ok)\n",
                    centroids[0], centroids[1], centroids[2]);
    }

    // 11. Havok Cloth fixed-30 Hz sub-frame pose interpolation & monotonic hitch resilience
    {
        auto make_rigid_yaw_pitch = [](float yaw, float pitch, float tx, float ty, float tz,
                                       float out[16]) {
            const float cy = std::cos(yaw), sy = std::sin(yaw);
            const float cp = std::cos(pitch), sp = std::sin(pitch);
            // Ry(yaw) * Rx(pitch), column-major
            out[0] = cy;       out[1] = 0.0f; out[2] = -sy;      out[3] = 0.0f;
            out[4] = sy * sp;  out[5] = cp;   out[6] = cy * sp;  out[7] = 0.0f;
            out[8] = sy * cp;  out[9] = -sp;  out[10] = cy * cp; out[11] = 0.0f;
            out[12] = tx;      out[13] = ty;  out[14] = tz;      out[15] = 1.0f;
        };
        auto xform_pt = [](const float m[16], const float p[3], float out[3]) {
            out[0] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
            out[1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
            out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
        };

        float m0[16]{}, m1[16]{};
        make_rigid_yaw_pitch(0.15f, -0.10f, 1.2f, 0.3f, -0.8f, m0);
        make_rigid_yaw_pitch(0.15f + 1.5707963f, 0.45f, 1.5f, 0.35f, -0.4f, m1);

        const std::array<float, 6> betas{0.0f, 0.25f, 0.4167f, 0.5f, 0.8333f, 1.0f};
        const float probe[3]{-0.37f, 1.42f, 0.91f};
        for (const float beta : betas) {
            float l2w[16]{}, w2l[16]{};
            BbClothInterp::LerpTransformMatrix4x4(m0, m1, beta, l2w);
            assert(BbClothInterp::InvertRigidMatrix4x4(l2w, w2l));

            for (int c0 = 0; c0 < 3; ++c0) {
                for (int c1 = 0; c1 < 3; ++c1) {
                    const float dot = l2w[c0 * 4 + 0] * l2w[c1 * 4 + 0] +
                                      l2w[c0 * 4 + 1] * l2w[c1 * 4 + 1] +
                                      l2w[c0 * 4 + 2] * l2w[c1 * 4 + 2];
                    const float want = (c0 == c1) ? 1.0f : 0.0f;
                    assert(std::abs(dot - want) < 1e-5f);
                }
            }
            const float cx = l2w[5] * l2w[10] - l2w[6] * l2w[9];
            const float cy = l2w[6] * l2w[8] - l2w[4] * l2w[10];
            const float cz = l2w[4] * l2w[9] - l2w[5] * l2w[8];
            const float det = l2w[0] * cx + l2w[1] * cy + l2w[2] * cz;
            assert(std::abs(det - 1.0f) < 1e-5f);

            float loc[3]{}, back[3]{};
            xform_pt(w2l, probe, loc);
            xform_pt(l2w, loc, back);
            assert(std::abs(back[0] - probe[0]) < 1e-5f);
            assert(std::abs(back[1] - probe[1]) < 1e-5f);
            assert(std::abs(back[2] - probe[2]) < 1e-5f);
        }

        float m_far[16]{}, m_snap[16]{};
        make_rigid_yaw_pitch(0.8f, 0.2f, 3.0f, 0.3f, -0.8f, m_far);
        BbClothInterp::LerpTransformMatrix4x4(m0, m_far, 0.5f, m_snap);
        for (int k = 0; k < 16; ++k) {
            assert(std::abs(m_snap[k] - m_far[k]) < 1e-6f);
        }

        auto run_cloth_fps_check = [](float fps, bool inject_hitch_42p5) {
            constexpr float kFrame30 = 1.0f / 30.0f;
            constexpr float kVChar = 5.2f;
            const float base_dt = 1.0f / fps;
            double accum = 0.0;
            std::uint32_t streak = 0;
            bool initialized = false;
            float raw_curr = 0.0f, raw_prev = 0.0f, prev_tm = 0.0f;
            float local_prev = 0.0f, local_curr = 0.0f;
            float t_elapsed = 0.0f;
            float prev_anchor = 0.0f;
            float prev_world_render = -1e9f;
            bool have_prev_world = false;

            float min_loc = 1e9f, max_loc = -1e9f;
            const int total_frames = static_cast<int>(fps * 3.0f);
            for (int f = 1; f <= total_frames; ++f) {
                const float dt = (inject_hitch_42p5 && f == int(fps * 1.5f)) ? kFrame30 : base_dt;
                t_elapsed += dt;
                const float curr_anchor = kVChar * t_elapsed;
                const auto tick = BbClothInterp::ComputeClothFixed30Tick(accum, streak, dt, kFrame30);
                if (tick.do_step) {
                    const float step_anchor =
                        prev_anchor + tick.step_beta * (curr_anchor - prev_anchor);
                    if (!initialized) {
                        prev_tm = step_anchor;
                        raw_curr = step_anchor;
                        raw_prev = step_anchor;
                        local_curr = 0.0f;
                        local_prev = 0.0f;
                        initialized = true;
                    } else {
                        local_prev = local_curr;
                        const float delta_tm = step_anchor - prev_tm;
                        const float meas_speed = std::abs(delta_tm) / tick.step_dt;
                        assert(std::abs(meas_speed - kVChar) < 1e-3f);
                        const float blend =
                            std::clamp((meas_speed - 4.0f) / (6.5f - 4.0f), 0.0f, 1.0f) * 0.75f;
                        raw_curr += blend * delta_tm;
                        raw_prev += blend * delta_tm;
                        prev_tm = step_anchor;

                        const float sdt = tick.step_dt * 0.25f;
                        float cur = raw_curr;
                        float prv = raw_prev;
                        for (int s = 0; s < 4; ++s) {
                            const float vel = (cur - prv) * 0.96f;
                            float nxt = cur + vel - 0.8f * sdt;
                            const float err = (nxt - prev_tm) - (-0.35f);
                            nxt -= 0.45f * err;
                            prv = cur;
                            cur = nxt;
                        }
                        raw_curr = cur;
                        raw_prev = prv;
                        local_curr = raw_curr - step_anchor;
                    }
                }
                const float loc_render = local_prev + (local_curr - local_prev) * tick.alpha;
                const float world_render = curr_anchor + loc_render;
                if (t_elapsed >= 1.0f) {
                    min_loc = std::min(min_loc, loc_render);
                    max_loc = std::max(max_loc, loc_render);
                    if (have_prev_world) {
                        assert(world_render > prev_world_render - 1e-5f);
                    }
                    prev_world_render = world_render;
                    have_prev_world = true;
                }
                prev_anchor = curr_anchor;
            }
            const float ripple_mm = (max_loc - min_loc) * 1000.0f;
            assert(ripple_mm < 0.05f);
        };

        for (const float fps : {30.0f, 42.5f, 170.0f / 3.0f, 85.0f, 170.0f}) {
            run_cloth_fps_check(fps, false);
        }
        run_cloth_fps_check(42.5f, true);
        std::printf("Camera motion: Havok cloth fixed30 sub-frame pose interpolation PASS "
                    "(30/42.5/56.7/85/170 FPS + 33.3ms hitch monotonic)\n");
    }

    // 12. Pure native arbitrary-FPS Havok Cloth constraint laws & 2D cape pendulum invariance
    {
        using BbClothInterp::ComputeClothElasticStiffnessScale;
        using BbClothInterp::ComputeClothExponentialRelaxationFactor;
        using BbClothInterp::ComputeNativeClothConstraintFactor;
        using BbClothInterp::HclConstraintSetType;

        constexpr float kHavokExp = -1.725f;
        constexpr int kSubsteps = 2;
        const float game_m2 = std::exp2f(kHavokExp * std::log2f(float(kSubsteps)));

        const std::array<float, 10> test_fps{
            30.0f, 42.5f, 170.0f / 3.0f, 60.0f, 85.0f, 120.0f, 170.0f, 240.0f, 360.0f, 500.0f};
        for (const float fps : test_fps) {
            const float frames = std::clamp(30.0f / fps, 1.0f / 300.0f, 1.0f);
            const float elastic = ComputeClothElasticStiffnessScale(frames, -kHavokExp);

            // (a) hclStretchLinkConstraintSet (type 2 LRA 1.0f tether) stays unscaled across all FPS
            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kStretchLink), 1.0f, 1, 0,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic) -
                            1.0f) < 1e-6f);
            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kStretchLink), 1.0f, 2, 0,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic) -
                            game_m2) < 1e-6f);

            // (b) hclLocalRangeConstraintSet (type 5): hard 1.0f cages stay unscaled;
            //     soft 0.50f cages preserve exact continuous-time exponential retention
            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kLocalRange), 1.0f, 1, 0,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic) -
                            1.0f) < 1e-6f);
            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kLocalRange), 1.0f, 2, 0,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic)) < 1e-6f);
            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kLocalRange), 1.0f, 2, 1,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic) -
                            1.0f) < 1e-6f);

            const float f_t5_m1 = ComputeNativeClothConstraintFactor(
                static_cast<u32>(HclConstraintSetType::kLocalRange), 0.50f, 1, 0, kSubsteps, 1.0f,
                1.0f, kHavokExp, frames, elastic);
            const double ret_m1 = std::pow(1.0 - double(0.50f * f_t5_m1), double(kSubsteps) / double(frames));
            assert(std::abs(ret_m1 - 0.25) < 1e-5);

            assert(std::abs(ComputeNativeClothConstraintFactor(
                                static_cast<u32>(HclConstraintSetType::kLocalRange), 0.50f, 2, 0,
                                kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic)) < 1e-6f);
            const float f_t5_m2 = ComputeNativeClothConstraintFactor(
                static_cast<u32>(HclConstraintSetType::kLocalRange), 0.50f, 2, 1, kSubsteps, 1.0f,
                1.0f, kHavokExp, frames, elastic);
            const double ret_m2 = std::pow(1.0 - double(0.50f * f_t5_m2), 1.0 / double(frames));
            assert(std::abs(ret_m2 - 0.50) < 1e-5);

            // (c) Elastic weave & bend sets (types 1, 16, 20) follow frames^1.725
            for (const HclConstraintSetType t :
                 {HclConstraintSetType::kStandardLink, HclConstraintSetType::kBendStiffnessMx,
                  HclConstraintSetType::kCompressibleLink}) {
                const float f_el = ComputeNativeClothConstraintFactor(
                    static_cast<u32>(t), 1.0f, 2, 0, kSubsteps, 1.0f, 1.0f, kHavokExp, frames,
                    elastic);
                assert(std::abs(f_el - game_m2 * elastic) < 1e-6f);
            }
        }

        // Full 9-particle 2D Havok Cloth cape simulation (preStep 0x76fee6 + Verlet 0x774119 +
        // StandardLink type=1 + StretchLink LRA type=2 + LocalRange type=5 minNormalDist floor)
        struct CapeResult {
            float x_swing = 0.0f;
            float y_final = 0.0f;
            float max_x_pen = -1e9f;
        };
        auto run_native_cape = [&](float fps) -> CapeResult {
            constexpr int Np = 9;
            constexpr float kSeg = 0.08f; // 0.64 m cape
            constexpr float kMinNormalDist = -0.0023f; // BD_M_1300 Hunter Garb back clearance
            const float dt = 1.0f / fps;
            const float sdt = dt / float(kSubsteps);
            const float frames = std::clamp(30.0f / fps, 1.0f / 300.0f, 1.0f);
            const float elastic = ComputeClothElasticStiffnessScale(frames, -kHavokExp);

            float cx[Np]{}, cy[Np]{}, px[Np]{}, py[Np]{};
            float refx[Np]{}, refy[Np]{};
            for (int i = 0; i < Np; ++i) {
                cx[i] = px[i] = -float(i) * kSeg * 0.5f;
                cy[i] = py[i] = -float(i) * kSeg * 0.8660254f;
                refx[i] = 0.0f;
                refy[i] = -float(i) * kSeg;
            }
            const float eff_damp = std::pow(1.0f - 0.70f, sdt);
            const float g_dt2 = -9.81f * sdt * sdt;
            CapeResult res{};
            const int total_frames = static_cast<int>(std::lround(2.0f * fps));
            const int swing_frame = static_cast<int>(std::lround(0.25f * fps));

            for (int f = 1; f <= total_frames; ++f) {
                for (int s = 0; s < kSubsteps; ++s) {
                    const float f1 = ComputeNativeClothConstraintFactor(
                        static_cast<u32>(HclConstraintSetType::kStandardLink), 0.65f, 2, s,
                        kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic);
                    const float f2 = ComputeNativeClothConstraintFactor(
                        static_cast<u32>(HclConstraintSetType::kStretchLink), 1.0f, 2, s,
                        kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic);
                    const float f5 = ComputeNativeClothConstraintFactor(
                        static_cast<u32>(HclConstraintSetType::kLocalRange), 0.50f, 2, s,
                        kSubsteps, 1.0f, 1.0f, kHavokExp, frames, elastic);

                    for (int i = 1; i < Np; ++i) {
                        const float pre_x = cx[i];
                        const float pre_y = cy[i];
                        cx[i] += (pre_x - px[i]) * eff_damp;
                        cy[i] += (pre_y - py[i]) * eff_damp + g_dt2;
                        px[i] = pre_x;
                        py[i] = pre_y;
                    }

                    // Solve type 1 StandardLink springs (both directions)
                    const float eff_k1 = std::min(1.0f, 0.65f * f1);
                    for (int i = 0; i < Np - 1; ++i) {
                        const float dx = cx[i + 1] - cx[i];
                        const float dy = cy[i + 1] - cy[i];
                        const float d = std::sqrt(dx * dx + dy * dy);
                        if (d > 1e-6f) {
                            const float diff = eff_k1 * (d - kSeg) / d;
                            if (i == 0) {
                                cx[1] -= dx * diff;
                                cy[1] -= dy * diff;
                            } else {
                                cx[i] += 0.5f * dx * diff;
                                cy[i] += 0.5f * dy * diff;
                                cx[i + 1] -= 0.5f * dx * diff;
                                cy[i + 1] -= 0.5f * dy * diff;
                            }
                        }
                    }

                    // Solve type 2 StretchLink LRA 1-hop max-length tether (unscaled across FPS)
                    const float eff_k2 = std::min(1.0f, 1.0f * f2);
                    for (int i = 1; i < Np; ++i) {
                        const float max_lra = 1.08f * float(i) * kSeg;
                        const float dx = cx[i] - cx[0];
                        const float dy = cy[i] - cy[0];
                        const float d = std::sqrt(dx * dx + dy * dy);
                        if (d > max_lra && d > 1e-6f) {
                            const float pull = eff_k2 * (d - max_lra) / d;
                            cx[i] -= dx * pull;
                            cy[i] -= dy * pull;
                        }
                    }

                    // Solve type 5 LocalRange: unscaled back surface normal pushout + soft maxDistance
                    if (f5 > 0.0f) {
                        const float eff_k5 = std::min(1.0f, 0.50f * f5);
                        const float max_cone = 0.45f;
                        for (int i = 1; i < Np; ++i) {
                            const float dx = cx[i] - refx[i];
                            const float dy = cy[i] - refy[i];
                            const float d = std::sqrt(dx * dx + dy * dy);
                            if (d > max_cone && d > 1e-6f) {
                                const float pull = eff_k5 * (d - max_cone) / d;
                                cx[i] -= dx * pull;
                                cy[i] -= dy * pull;
                            }
                            // Body surface along x >= 0 with outward normal -X (allowed: x <= -kMinNormalDist)
                            if (cx[i] > -kMinNormalDist) {
                                cx[i] = -kMinNormalDist;
                            }
                        }
                    }
                }
                for (int i = 1; i < Np; ++i) {
                    res.max_x_pen = std::max(res.max_x_pen, cx[i]);
                }
                if (f == swing_frame) {
                    res.x_swing = cx[Np - 1];
                }
                if (f == total_frames) {
                    res.y_final = cy[Np - 1];
                }
            }
            return res;
        };

        const CapeResult ref30 = run_native_cape(30.0f);
        for (const float fps : {30.0f, 42.5f, 170.0f / 3.0f, 85.0f, 170.0f, 360.0f}) {
            const CapeResult cur = run_native_cape(fps);
            assert(cur.max_x_pen <= 0.002301f); // 0.000 mm torso clipping below -kMinNormalDist
            assert(std::abs(cur.y_final - ref30.y_final) < 0.018f);
            assert(std::abs(cur.x_swing - ref30.x_swing) < 0.030f);
        }
        std::printf("Camera motion: native arbitrary-FPS Havok cloth invariance PASS "
                    "(30..360 FPS LRA+LocalRange+Verlet, y_ref=%.3fm x_swing=%.3fm)\n",
                    ref30.y_final, ref30.x_swing);
    }

    // 13. Continuous-time Havok Cloth Wind PRNG, native SprjDirWind parity, and high-FPS idle calm
    {
        using BbClothInterp::ClothXorShift128;
        using BbClothInterp::ComputeClothElasticStiffnessScale;
        using BbClothInterp::BlendClothPointWindSpeedAndDir;
        using BbClothInterp::BlendClothWindVectors3D;
        using BbClothInterp::ClothXorShift128;
        using BbClothInterp::ComputeClothExponentialRelaxationFactor;
        using BbClothInterp::ComputeClothWindSmoothState;
        using BbClothInterp::ComputeHighFpsClothDampingNegRate;
        using BbClothInterp::ComputeNativeClothConstraintFactor;
        using BbClothInterp::ComputeNativeClothEffectiveDamping;
        using BbClothInterp::HclConstraintSetType;
        using BbClothInterp::HclSimClothDataOffsets;
        using BbClothInterp::HclSimClothInstanceOffsets;
        using BbClothInterp::SprjWindActionOffsets;
        using BbClothInterp::SprjDirWindApplyNative;

        // (a) Bit-exact XorShift128 sequence matching PS4 eboot.bin+0x20f81c0 / 0x1053f14
        {
            ClothXorShift128 rng{};
            rng.Seed(10000u);
            constexpr float expected6[6] = {
                0.088540f, 0.430971f, 0.366490f, 0.832743f, 0.222345f, 0.159638f};
            for (int k = 0; k < 6; ++k) {
                const float v = rng.NextFloat01();
                assert(std::abs(v - expected6[k]) < 1e-5f);
            }
        }

        // (b) Smooth 30 Hz output wind vector blending continuity & speed preservation
        {
            auto eval_discrete_wind = [](std::uint32_t seed, float out[4]) {
                ClothXorShift128 rng{};
                rng.Seed(seed);
                const float g = rng.NextFloat01();
                const float spd = 0.8f + 0.6f * g;
                const float yaw = 0.45f * (g - 0.5f);
                out[0] = spd * std::cos(yaw);
                out[1] = 0.0f;
                out[2] = spd * std::sin(yaw);
                out[3] = 0.0f;
            };
            float max_step_delta = 0.0f;
            float prev_v[4]{};
            {
                const auto ws0 = ComputeClothWindSmoothState(0.0);
                float v0[4]{}, v1[4]{};
                eval_discrete_wind(ws0.seed0, v0);
                eval_discrete_wind(ws0.seed1, v1);
                BlendClothWindVectors3D(v0, v1, ws0.smooth, prev_v);
            }
            for (int ms = 1; ms <= 2000; ++ms) {
                const auto ws = ComputeClothWindSmoothState(double(ms) * 0.001);
                assert(ws.smooth >= 0.0f && ws.smooth <= 1.0f);
                float v0[4]{}, v1[4]{}, cur_v[4]{};
                eval_discrete_wind(ws.seed0, v0);
                eval_discrete_wind(ws.seed1, v1);
                BlendClothWindVectors3D(v0, v1, ws.smooth, cur_v);
                const float d = std::sqrt((cur_v[0] - prev_v[0]) * (cur_v[0] - prev_v[0]) +
                                          (cur_v[2] - prev_v[2]) * (cur_v[2] - prev_v[2]));
                max_step_delta = std::max(max_step_delta, d);
                std::memcpy(prev_v, cur_v, 16);
            }
            assert(max_step_delta < 0.05f);

            // Exact integer 30 Hz tick matches unblended discrete wind at seed0
            const auto ws_tick = ComputeClothWindSmoothState(1.0 / 30.0);
            assert(ws_tick.seed0 == 10000u && ws_tick.smooth < 1e-5f);
            float v_ref[4]{}, v_next[4]{}, v_out[4]{};
            eval_discrete_wind(ws_tick.seed0, v_ref);
            eval_discrete_wind(ws_tick.seed1, v_next);
            BlendClothWindVectors3D(v_ref, v_next, ws_tick.smooth, v_out);
            assert(std::abs(v_out[0] - v_ref[0]) < 1e-6f);
            assert(std::abs(v_out[2] - v_ref[2]) < 1e-6f);

            // Wide-angle & 180-deg opposite gusts never collapse wind speed at smooth = 0.5
            const float opp0[4] = {4.0f, 0.0f, 0.0f, 0.0f};
            const float opp1[4] = {-4.0f, 0.0f, 0.0f, 0.0f};
            float opp_mid[4]{};
            BlendClothWindVectors3D(opp0, opp1, 0.5f, opp_mid);
            const float opp_spd =
                std::sqrt(opp_mid[0] * opp_mid[0] + opp_mid[1] * opp_mid[1] + opp_mid[2] * opp_mid[2]);
            assert(std::abs(opp_spd - 4.0f) < 1e-4f);
        }

        // (c) Native SprjDirWind parity (30 FPS reference, cull_step reuse, prev_sdt velocity interval)
        {
            constexpr int Np = 9;
            struct alignas(16) Vec4 {
                float x, y, z, w;
            };
            std::array<std::uint8_t, 0x100> action{};
            std::array<std::uint8_t, 0x200> sim{};
            std::array<std::uint8_t, 0x180> sim_data{};
            std::array<Vec4, Np> cur{}, prev{}, norm{}, pdata{}, forces{}, forces_ref{};

            for (int i = 0; i < Np; ++i) {
                cur[i] = {0.10f * float(i), -0.08f * float(i), 0.02f * float(i), 1.0f};
                prev[i] = {cur[i].x - 0.001f * float(i + 1), cur[i].y + 0.0005f, cur[i].z, 1.0f};
                norm[i] = {0.0f, 0.0f, 1.0f, 0.0f};
                pdata[i] = {0.25f + 0.05f * float(i), 0.0f, 0.0f, 0.0f};
            }
            float total_mass = 0.0f;
            for (const auto& pd : pdata) total_mass += pd.x;

            const float drag_coeff = 3.5f;
            const float cull_step = 0.45f;
            const std::uint8_t cull_en = 1;
            std::memcpy(action.data() + SprjWindActionOffsets::kDragCoeff, &drag_coeff, 4);
            std::memcpy(action.data() + SprjWindActionOffsets::kDirCullStep, &cull_step, 4);
            action[SprjWindActionOffsets::kDirCullEnabled] = cull_en;

            const Vec4* pdata_ptr = pdata.data();
            const Vec4* cur_ptr = cur.data();
            const Vec4* prev_ptr = prev.data();
            const Vec4* norm_ptr = norm.data();
            const std::uint8_t* sdata_ptr = sim_data.data();
            std::memcpy(sim_data.data() + HclSimClothDataOffsets::kParticleDatasData, &pdata_ptr, 8);
            std::memcpy(sim_data.data() + HclSimClothDataOffsets::kTotalMass, &total_mass, 4);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kSimClothData, &sdata_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kCurrentPositions, &cur_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPreviousPositions, &prev_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kNumParticles, &Np, 4);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kParticleNormalsData, &norm_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kParticleNormalsSize, &Np, 4);
            float prev_sdt = 1.0f / 60.0f;
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPrevSubstepDt, &prev_sdt, 4);

            const float dir_vec[4] = {0.8f, 0.1f, 0.6f, 0.0f};
            const float params[4] = {0.75f, 1.25f, 120.0f, 12.0f};
            const bool ok = SprjDirWindApplyNative(
                action.data(), sim.data(), &forces[0].x, dir_vec, params, 1.0f / 60.0f, 10000u,
                20000u, 0.0f, 1.0f, true);
            assert(ok);

            // Independent exact check of particle 0 & particle 1 + cull_step reuse on particle 2
            ClothXorShift128 ref_rng{};
            ref_rng.Seed(10000u);
            const float ws = std::sqrt(dir_vec[0] * dir_vec[0] + dir_vec[1] * dir_vec[1] +
                                       dir_vec[2] * dir_vec[2]);
            const float dn[3] = {dir_vec[0] / ws, dir_vec[1] / ws, dir_vec[2] / ws};
            for (int i = 0; i < Np; ++i) {
                float intp = 0.0f;
                const bool eval =
                    (i == 0 || i == Np - 1 || (1.0f > cull_step + std::modf(cull_step * float(i), &intp)));
                if (eval) {
                    float u[6]{};
                    for (int k = 0; k < 6; ++k) u[k] = ref_rng.NextFloat01();
                    const float dx = cur[i].x - prev[i].x;
                    const float dy = cur[i].y - prev[i].y;
                    const float dz = cur[i].z - prev[i].z;
                    const float tx = (dx + (u[0] * 0.2f + 0.05f)) * (u[1] * 0.6f - 0.1f);
                    const float ty = (dy + (u[2] * 0.2f + 0.05f)) * (u[3] * 0.6f - 0.1f);
                    const float tz = (dz + (u[4] * 0.2f + 0.05f)) * (u[5] * 0.6f - 0.1f);
                    const float dist =
                        std::sqrt(cur[i].x * cur[i].x + cur[i].y * cur[i].y + cur[i].z * cur[i].z);
                    const float ang = (params[2] + params[3] * dist) / 180.0f - 3.14159274f;
                    const float wv = std::clamp(0.5f * std::sin(ang) + 0.5f, 0.0f, 1.0f);
                    const float spd = ((1.0f - wv) * params[0] + wv * params[1]) * ws;
                    const float nfac =
                        std::fabs(dn[0] * norm[i].x + dn[1] * norm[i].y + dn[2] * norm[i].z);
                    const float sc = (drag_coeff * nfac) * (pdata[i].x / total_mass);
                    forces_ref[i].x = sc * (dn[0] * spd + tx - dx * 60.0f);
                    forces_ref[i].y = sc * (dn[1] * spd + ty - dy * 60.0f);
                    forces_ref[i].z = sc * (dn[2] * spd + tz - dz * 60.0f);
                } else {
                    forces_ref[i] = forces_ref[i - 1];
                }
                assert(std::abs(forces[i].x - forces_ref[i].x) < 1e-5f);
                assert(std::abs(forces[i].y - forces_ref[i].y) < 1e-5f);
                assert(std::abs(forces[i].z - forces_ref[i].z) < 1e-5f);
            }
            // Particle 2 should have been skipped (cull_step 0.45 + frac 0.90 >= 1.0)
            assert(std::abs(forces[2].x - forces[1].x) < 1e-7f);

            // Verify prev_sdt != sdt uses [sim + 0x108] rather than raw sdt (`~0.052 mN` delta on particle 1)
            std::array<Vec4, Np> forces_prev340{};
            std::array<Vec4, Np> forces_prev170{};
            prev_sdt = 1.0f / 340.0f;
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPrevSubstepDt, &prev_sdt, 4);
            SprjDirWindApplyNative(action.data(), sim.data(), &forces_prev340[0].x, dir_vec, params,
                                   1.0f / 170.0f, 10000u, 20000u, 0.0f, 30.0f / 170.0f, true);
            prev_sdt = 1.0f / 170.0f;
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPrevSubstepDt, &prev_sdt, 4);
            SprjDirWindApplyNative(action.data(), sim.data(), &forces_prev170[0].x, dir_vec, params,
                                   1.0f / 170.0f, 10000u, 20000u, 0.0f, 30.0f / 170.0f, true);
            assert(std::abs(forces_prev340[1].x - forces_prev170[1].x) > 0.04f);
            assert(std::abs(forces_prev340[1].x - forces_ref[1].x) > 0.07f);
        }

        // (d) Standing-still high-FPS cape tremor suppression (30..428 FPS vs legacy 1000 Hz wind)
        struct IdleMetrics {
            float rms_acc = 0.0f;
            float rms_jerk = 0.0f;
        };
        auto measure_idle_metrics = [&](float fps, bool smooth_fix) -> IdleMetrics {
            constexpr int Np = 9;
            constexpr int kSubsteps = 2;
            constexpr float kSeg = 0.08f;
            struct alignas(16) Vec4 {
                float x, y, z, w;
            };
            std::array<std::uint8_t, 0x100> action{};
            std::array<std::uint8_t, 0x200> sim{};
            std::array<std::uint8_t, 0x180> sim_data{};
            std::array<Vec4, Np> cur{}, prev{}, pdata{}, forces{};
            float total_mass = 0.0f;
            for (int i = 0; i < Np; ++i) {
                cur[i] = prev[i] = {0.0f, -float(i) * kSeg, 0.0f, 1.0f};
                pdata[i] = {0.25f, 0.0f, 0.0f, 0.0f};
                total_mass += pdata[i].x;
            }
            const float drag_coeff = 2.0f;
            std::memcpy(action.data() + SprjWindActionOffsets::kDragCoeff, &drag_coeff, 4);
            const Vec4* pdata_ptr = pdata.data();
            const Vec4* cur_ptr = cur.data();
            const Vec4* prev_ptr = prev.data();
            const std::uint8_t* sdata_ptr = sim_data.data();
            std::memcpy(sim_data.data() + HclSimClothDataOffsets::kParticleDatasData, &pdata_ptr, 8);
            std::memcpy(sim_data.data() + HclSimClothDataOffsets::kTotalMass, &total_mass, 4);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kSimClothData, &sdata_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kCurrentPositions, &cur_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPreviousPositions, &prev_ptr, 8);
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kNumParticles, &Np, 4);

            const float dt = 1.0f / fps;
            const float sdt = dt / float(kSubsteps);
            float prev_sdt = sdt;
            std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPrevSubstepDt, &prev_sdt, 4);

            const float frames = std::clamp(dt * 30.0f, 1.0f / 300.0f, 1.0f);
            const float elastic = ComputeClothElasticStiffnessScale(frames, 1.725f);
            const float neg_rate = smooth_fix ? ComputeHighFpsClothDampingNegRate(frames, 0.94f) : 0.0f;
            const float eff_damp = ComputeNativeClothEffectiveDamping(0.12f, sdt, neg_rate);

            const int total_frames = static_cast<int>(std::lround(2.0f * fps));
            const int warmup_frames = static_cast<int>(std::lround(0.5f * fps));
            double sum_acc2 = 0.0;
            double sum_jerk2 = 0.0;
            int samples = 0;
            float prev_vx = 0.0f, prev_ax = 0.0f;
            double sim_t = 0.0;

            auto eval_world_dir = [](std::uint32_t seed, float out[4]) {
                ClothXorShift128 rng{};
                rng.Seed(seed);
                const float g = rng.NextFloat01();
                const float wind_spd = 0.8f + 0.6f * g;
                const float yaw = 0.45f * (g - 0.5f);
                out[0] = wind_spd * std::cos(yaw);
                out[1] = 0.0f;
                out[2] = wind_spd * std::sin(yaw);
                out[3] = 0.0f;
            };

            for (int f = 1; f <= total_frames; ++f) {
                sim_t += double(dt);
                const auto ws = ComputeClothWindSmoothState(sim_t);
                float dir_vec[4]{};
                if (smooth_fix) {
                    float v0[4]{}, v1[4]{};
                    eval_world_dir(ws.seed0, v0);
                    eval_world_dir(ws.seed1, v1);
                    BlendClothWindVectors3D(v0, v1, ws.smooth, dir_vec);
                } else {
                    const std::uint32_t raw_seed =
                        static_cast<std::uint32_t>(std::lround(std::fmod(sim_t * 30.0, 1000.0) * 1000.0));
                    eval_world_dir(raw_seed, dir_vec);
                }
                const float phase_deg = float(sim_t * 180.0);
                const float params[4] = {0.75f, 1.25f, phase_deg, 10.0f};

                for (int s = 0; s < kSubsteps; ++s) {
                    std::memset(forces.data(), 0, sizeof(forces));
                    SprjDirWindApplyNative(action.data(), sim.data(), &forces[0].x, dir_vec, params,
                                           sdt, ws.seed0, ws.seed1, smooth_fix ? ws.smooth : 0.0f,
                                           frames, smooth_fix);
                    // Simulate Havok preStep velocity rescale + damping & gravity Verlet integration
                    const float ratio = (sdt / prev_sdt) * eff_damp;
                    prev_sdt = sdt;
                    std::memcpy(sim.data() + HclSimClothInstanceOffsets::kPrevSubstepDt, &prev_sdt, 4);
                    for (int i = 1; i < Np; ++i) {
                        const float pre_x = cur[i].x;
                        const float pre_y = cur[i].y;
                        const float pre_z = cur[i].z;
                        const float ax = forces[i].x / pdata[i].x;
                        const float ay = (forces[i].y / pdata[i].x) - 9.81f;
                        const float az = forces[i].z / pdata[i].x;
                        cur[i].x += (pre_x - prev[i].x) * ratio + ax * sdt * sdt;
                        cur[i].y += (pre_y - prev[i].y) * ratio + ay * sdt * sdt;
                        cur[i].z += (pre_z - prev[i].z) * ratio + az * sdt * sdt;
                        prev[i].x = pre_x;
                        prev[i].y = pre_y;
                        prev[i].z = pre_z;
                    }

                    const float f1 = ComputeNativeClothConstraintFactor(
                        static_cast<u32>(HclConstraintSetType::kStandardLink), 0.65f, 2, s,
                        kSubsteps, 1.0f, 1.0f, -1.725f, frames, elastic);
                    const float eff_k1 = std::min(1.0f, 0.65f * f1);
                    for (int i = 0; i < Np - 1; ++i) {
                        const float dx = cur[i + 1].x - cur[i].x;
                        const float dy = cur[i + 1].y - cur[i].y;
                        const float dz = cur[i + 1].z - cur[i].z;
                        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (d > 1e-6f) {
                            const float diff = eff_k1 * (d - kSeg) / d;
                            if (i == 0) {
                                cur[1].x -= dx * diff;
                                cur[1].y -= dy * diff;
                                cur[1].z -= dz * diff;
                            } else {
                                cur[i].x += 0.5f * dx * diff;
                                cur[i].y += 0.5f * dy * diff;
                                cur[i].z += 0.5f * dz * diff;
                                cur[i + 1].x -= 0.5f * dx * diff;
                                cur[i + 1].y -= 0.5f * dy * diff;
                                cur[i + 1].z -= 0.5f * dz * diff;
                            }
                        }
                    }
                }

                const float tip_vx = (cur[Np - 1].x - prev[Np - 1].x) / sdt;
                const float tip_ax = (tip_vx - prev_vx) / dt;
                const float tip_jx = (tip_ax - prev_ax) / dt;
                if (f > warmup_frames + 2) {
                    sum_acc2 += double(tip_ax) * double(tip_ax);
                    sum_jerk2 += double(tip_jx) * double(tip_jx);
                    ++samples;
                }
                prev_vx = tip_vx;
                prev_ax = tip_ax;
            }
            return samples > 0
                       ? IdleMetrics{static_cast<float>(std::sqrt(sum_acc2 / double(samples))),
                                     static_cast<float>(std::sqrt(sum_jerk2 / double(samples)))}
                       : IdleMetrics{};
        };

        const IdleMetrics m30 = measure_idle_metrics(30.0f, true);
        const IdleMetrics m428_broken = measure_idle_metrics(428.5f, false);
        const IdleMetrics m85 = measure_idle_metrics(85.0f, true);
        const IdleMetrics m170 = measure_idle_metrics(170.0f, true);
        const IdleMetrics m428 = measure_idle_metrics(428.5f, true);
        assert(m428.rms_jerk < m428_broken.rms_jerk * 0.25f);
        assert(m428.rms_acc < m428_broken.rms_acc * 0.75f);
        assert(m85.rms_acc <= m30.rms_acc * 1.05f);
        assert(m170.rms_acc <= m30.rms_acc * 1.05f);
        assert(m428.rms_acc <= m30.rms_acc * 1.05f);
        assert(m85.rms_jerk <= m30.rms_jerk * 1.25f);
        assert(m170.rms_jerk <= m30.rms_jerk * 1.25f);
        assert(m428.rms_jerk <= m30.rms_jerk * 1.25f);
        std::printf("Camera motion: smooth SprjDirWind + idle cloth calm PASS "
                    "(acc 30=%.3f -> 85=%.3f 170=%.3f 428=%.3f vs broken=%.3f m/s^2; "
                    "jerk 30=%.2f -> 428=%.2f vs broken=%.2f m/s^3)\n",
                    m30.rms_acc, m85.rms_acc, m170.rms_acc, m428.rms_acc, m428_broken.rms_acc,
                    m30.rms_jerk, m428.rms_jerk, m428_broken.rms_jerk);
    }

    // 14. 500 FPS Perlin wind non-speedhack regression + universal LocalRange r8d=1 & preStep damping
    {
        using BbClothInterp::BlendClothPointWindSpeedAndDir;
        using BbClothInterp::BlendClothWindVectors3D;
        using BbClothInterp::ClothXorShift128;
        using BbClothInterp::ComputeClothExponentialRelaxationFactor;
        using BbClothInterp::ComputeClothWindSmoothState;
        using BbClothInterp::ComputeHighFpsClothDampingNegRate;
        using BbClothInterp::ComputeNativeClothEffectiveDamping;

        // 4-octave Perlin fBm matching sub_10522f0 / SprjPointWind (0x2685e30: y = 6*p + 400, mul = 9)
        auto fade5 = [](float t) noexcept -> float {
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        };
        auto grad1 = [](int i) noexcept -> float {
            std::uint32_t x = (static_cast<std::uint32_t>(i) * 374761393u) ^ 668265263u;
            x = ((x ^ (x >> 13)) * 1274126177u);
            return (float(x ^ (x >> 16)) / 2147483648.0f) - 1.0f;
        };
        auto perlin_fbm4 = [&](float p, float offset) noexcept -> float {
            const float y = 6.0f * p + offset;
            float sum = 0.0f, freq = 1.0f, amp = 0.5f;
            for (int oct = 0; oct < 4; ++oct) {
                const float x = y * freq;
                const int xi = static_cast<int>(std::floor(x));
                const float xf = x - float(xi);
                const float g0 = grad1(xi) * xf;
                const float g1 = grad1(xi + 1) * (xf - 1.0f);
                sum += (g0 + fade5(xf) * (g1 - g0)) * amp;
                freq *= 2.0f;
                amp *= 0.5f;
            }
            return std::min(1.0f, std::fabs(sum * 9.0f));
        };
        auto eval_yharnam_wind = [&](float gust01, float out[4]) noexcept {
            // Central Yharnam Wind.param: coord [6.0, 15.0], speed [0.5, 4.0], yaw [-30, +30] deg
            const float p = 6.0f + 9.0f * gust01;
            const float spd = 0.5f + 3.5f * perlin_fbm4(p, 400.0f);
            const float yaw = (-0.523599f) + 1.047198f * perlin_fbm4(p, 150.0f);
            out[0] = spd * std::cos(yaw);
            out[1] = 0.0f;
            out[2] = spd * std::sin(yaw);
            out[3] = 0.0f;
        };

        struct WindRateMetrics {
            float travel_per_s = 0.0f;
            float reversals_hz = 0.0f;
        };
        auto measure_perlin_wind = [&](float fps, bool broken_input_sweep) -> WindRateMetrics {
            const float dt = 1.0f / fps;
            const int n_frames = static_cast<int>(std::lround(4.0f * fps));
            double travel = 0.0;
            int revs = 0;
            float prev_v[4]{};
            float prev_spd = 0.0f;
            float prev_dspd = 0.0f;
            for (int f = 0; f < n_frames; ++f) {
                const double t = double(f) * double(dt);
                const auto ws = ComputeClothWindSmoothState(t);
                ClothXorShift128 r0{}, r1{};
                r0.Seed(ws.seed0);
                r1.Seed(ws.seed1);
                const float g0 = r0.NextFloat01();
                const float g1 = r1.NextFloat01();
                float cur_v[4]{};
                if (broken_input_sweep) {
                    eval_yharnam_wind(g0 + ws.smooth * (g1 - g0), cur_v);
                } else {
                    float v0[4]{}, v1[4]{};
                    eval_yharnam_wind(g0, v0);
                    eval_yharnam_wind(g1, v1);
                    BlendClothWindVectors3D(v0, v1, ws.smooth, cur_v);
                }
                const float cur_spd = std::sqrt(cur_v[0] * cur_v[0] + cur_v[2] * cur_v[2]);
                if (f > 0) {
                    const float dx = cur_v[0] - prev_v[0];
                    const float dz = cur_v[2] - prev_v[2];
                    travel += std::sqrt(double(dx * dx + dz * dz));
                    const float dspd = cur_spd - prev_spd;
                    if (std::fabs(dspd) > 1e-4f) {
                        if (prev_dspd * dspd < 0.0f) ++revs;
                        prev_dspd = dspd;
                    }
                }
                prev_spd = cur_spd;
                std::memcpy(prev_v, cur_v, 16);
            }
            return {static_cast<float>(travel / 4.0), float(revs) / 4.0f};
        };

        const WindRateMetrics w30 = measure_perlin_wind(30.0f, false);
        const WindRateMetrics w500_broken = measure_perlin_wind(500.0f, true);
        const WindRateMetrics w85_fixed = measure_perlin_wind(85.0f, false);
        const WindRateMetrics w170_fixed = measure_perlin_wind(170.0f, false);
        const WindRateMetrics w500_fixed = measure_perlin_wind(500.0f, false);

        // Broken Perlin input sweep caused >10x wind speedhack; output vector blend matches 30 FPS 1.0x!
        assert(w500_broken.travel_per_s > w30.travel_per_s * 10.0f);
        assert(w500_broken.reversals_hz > w30.reversals_hz * 10.0f);
        assert(w85_fixed.travel_per_s >= w30.travel_per_s * 0.90f &&
               w85_fixed.travel_per_s <= w30.travel_per_s * 1.15f);
        assert(w170_fixed.travel_per_s >= w30.travel_per_s * 0.90f &&
               w170_fixed.travel_per_s <= w30.travel_per_s * 1.15f);
        assert(w500_fixed.travel_per_s >= w30.travel_per_s * 0.90f &&
               w500_fixed.travel_per_s <= w30.travel_per_s * 1.15f);
        assert(std::fabs(w500_fixed.reversals_hz - w30.reversals_hz) <= 1.5f);

        // SprjPointWind speed+dir helper parity
        {
            const float d0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
            const float d1[4] = {0.0f, 0.0f, 1.0f, 0.0f};
            float spd_out = 0.0f;
            float dir_out[4]{};
            BlendClothPointWindSpeedAndDir(2.0f, d0, 6.0f, d1, 0.5f, &spd_out, dir_out);
            assert(std::fabs(spd_out - 4.0f) < 1e-5f);
            const float dlen = std::sqrt(dir_out[0] * dir_out[0] + dir_out[2] * dir_out[2]);
            assert(std::fabs(dlen - 1.0f) < 1e-5f);
        }

        // Universal LocalRange (r8d=1 on adaptStiffness=0, subSteps=1 & 2) + preStep damping parity
        for (const float fps : {30.0f, 85.0f, 170.0f, 500.0f}) {
            const float dt = 1.0f / fps;
            const float frames = dt * 30.0f;
            const float s_lr = 0.50f;
            const float fac = ComputeClothExponentialRelaxationFactor(s_lr, frames);
            for (const int substeps : {1, 2}) {
                const double rem_fps = std::pow(1.0 - double(s_lr * fac), double(substeps) / double(frames));
                const double rem_30 = std::pow(1.0 - double(s_lr), double(substeps));
                assert(std::abs(rem_fps - rem_30) < 1e-4);

                const float sdt = dt / float(substeps);
                const float neg_rate = ComputeHighFpsClothDampingNegRate(frames, 0.94f);
                const float d0 = ComputeNativeClothEffectiveDamping(0.0f, sdt, neg_rate);
                const float d95 = ComputeNativeClothEffectiveDamping(0.95f, sdt, neg_rate);
                const float d100 = ComputeNativeClothEffectiveDamping(1.0f, sdt, neg_rate);
                assert(d0 > 0.85f && d0 <= 1.0f);
                assert(d95 > 0.0f && d95 < d0);
                assert(d100 == 0.0f);
            }
        }
        std::printf("Camera motion: 500 FPS Perlin wind non-speedhack + r8d=1 LocalRange PASS "
                    "(travel 30=%.2f -> 500_fixed=%.2f vs 500_broken=%.2f m/s^2; "
                    "rev 30=%.1fHz -> 500_fixed=%.1fHz vs 500_broken=%.1fHz)\n",
                    w30.travel_per_s, w500_fixed.travel_per_s, w500_broken.travel_per_s,
                    w30.reversals_hz, w500_fixed.reversals_hz, w500_broken.reversals_hz);
    }

    // 15. Fixed-30 + per-render-frame live-body LocalRange & collider pushout (60..500 FPS sprint anti-clipping)
    {
        using BbClothInterp::ComputeClothFixed30Tick;
        using BbClothInterp::MulAffineTransform4x4;

        // Verify MulAffineTransform4x4 matches column-major hkTransform composition (0x76e533..0x76e603).
        {
            const float parent[16] = {
                0.0f, 0.0f, 1.0f, 0.0f,  // col 0: +Z
                0.0f, 1.0f, 0.0f, 0.0f,  // col 1: +Y
                -1.0f, 0.0f, 0.0f, 0.0f, // col 2: -X
                10.0f, 2.0f, -5.0f, 1.0f // col 3: translation
            };
            const float local[16] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.2f, 0.5f, -0.3f, 1.0f
            };
            float world[16]{};
            MulAffineTransform4x4(parent, local, world);
            // parent * (0.2, 0.5, -0.3, 1) = (10 + 0.3, 2 + 0.5, -5 + 0.2, 1) = (10.3, 2.5, -4.8, 1)
            assert(std::fabs(world[12] - 10.3f) < 1e-5f);
            assert(std::fabs(world[13] - 2.5f) < 1e-5f);
            assert(std::fabs(world[14] - (-4.8f)) < 1e-5f);
            assert(std::fabs(world[0] - 0.0f) < 1e-5f && std::fabs(world[2] - 1.0f) < 1e-5f);
        }

        // Simulate a 9-particle cape behind a Hunter transitioning into a forward sprint + spine lean +
        // swinging thigh sphere collider. Without live-body post-projection, the 33.3 ms fixed-30 LERP
        // buffer penetrates 35-60 mm into the live t_f back/thigh at 60 FPS; with post-projection,
        // penetration is 0.000 mm across 30..500 FPS while standing idle ripple stays 0.000 mm.
        struct LivePushoutMetrics {
            float max_back_pen_mm = 0.0f;
            float max_thigh_pen_mm = 0.0f;
            float idle_ripple_mm = 0.0f;
        };
        auto run_fixed30_sprint_test = [&](float fps, bool post_collide) -> LivePushoutMetrics {
            constexpr int Np = 9;
            constexpr float kSeg = 0.08f;
            constexpr float kFrame30 = 1.0f / 30.0f;
            constexpr float kMinNormalDist = 0.0023f; // 2.3 mm outward clearance along -X back normal
            constexpr float kThighRadius = 0.11f;

            // Character root moves along +X; spine leans backward/upward dynamically relative to root;
            // thigh swings backward (-X) during sprint stride.
            auto eval_pose = [](float t, float& root_x, float& spine_tilt_x_per_y, float& thigh_x, float& thigh_y) {
                if (t < 1.0f) {
                    // First 1.0 s: standing completely still (tests zero idle tremor).
                    root_x = 0.0f;
                    spine_tilt_x_per_y = 0.0f;
                    thigh_x = 0.14f;
                    thigh_y = -0.45f;
                } else {
                    const float ts = t - 1.0f;
                    root_x = 5.0f * ts;
                    // Spine/back leans and rocks relative to root pelvis at stride cadence:
                    spine_tilt_x_per_y = 0.32f * std::sin(14.0f * ts);
                    // Thigh swings backward into cape zone during stride:
                    thigh_x = root_x + 0.04f - 0.16f * std::max(0.0f, std::sin(14.0f * ts));
                    thigh_y = -0.45f;
                }
            };
            // Back surface X at vertical local height y <= 0 (cloth must stay at x <= back_x - kMinNormalDist):
            auto back_surface_x = [](float root_x, float spine_tilt, float y) {
                return root_x + spine_tilt * y;
            };
            auto project_live_body = [&](std::array<float, Np>& wx, std::array<float, Np>& wy,
                                         float root_x, float spine_tilt, float thigh_x, float thigh_y) {
                for (int i = 1; i < Np; ++i) {
                    // 1. LocalRange back-surface minNormalDist pushout (outward normal = -X)
                    const float max_allowed_x = back_surface_x(root_x, spine_tilt, wy[i]) - kMinNormalDist;
                    if (wx[i] > max_allowed_x) {
                        wx[i] = max_allowed_x;
                    }
                    // 2. Live thigh collider pushout
                    const float dx = wx[i] - thigh_x;
                    const float dy = wy[i] - thigh_y;
                    const float d2 = dx * dx + dy * dy;
                    if (d2 < kThighRadius * kThighRadius && d2 > 1e-12f) {
                        const float d = std::sqrt(d2);
                        const float push = (kThighRadius - d) / d;
                        wx[i] += dx * push;
                        wy[i] += dy * push;
                        const float recheck_x = back_surface_x(root_x, spine_tilt, wy[i]) - kMinNormalDist;
                        if (wx[i] > recheck_x) wx[i] = recheck_x;
                    }
                }
            };

            std::array<float, Np> raw_x{}, raw_y{}, prev_x{}, prev_y{};
            std::array<float, Np> loc_prev_x{}, loc_prev_y{}, loc_curr_x{}, loc_curr_y{};
            for (int i = 0; i < Np; ++i) {
                raw_x[i] = prev_x[i] = -kMinNormalDist;
                raw_y[i] = prev_y[i] = -float(i) * kSeg;
                loc_prev_x[i] = loc_curr_x[i] = raw_x[i];
                loc_prev_y[i] = loc_curr_y[i] = raw_y[i];
            }

            double accum = 0.0;
            u32 streak = 0;
            const float dt = 1.0f / fps;
            const int total_frames = static_cast<int>(fps * 2.5f);
            LivePushoutMetrics m{};
            float idle_min_x = 1e9f, idle_max_x = -1e9f;

            for (int f = 1; f <= total_frames; ++f) {
                const float t = float(f) * dt;
                float root_x = 0.0f, spine_tilt = 0.0f, thigh_x = 0.0f, thigh_y = 0.0f;
                eval_pose(t, root_x, spine_tilt, thigh_x, thigh_y);

                const auto tick = ComputeClothFixed30Tick(accum, streak, dt, kFrame30);
                if (tick.do_step) {
                    loc_prev_x = loc_curr_x;
                    loc_prev_y = loc_curr_y;
                    raw_x[0] = prev_x[0] = root_x - kMinNormalDist;
                    raw_y[0] = prev_y[0] = 0.0f;
                    const float sdt = tick.step_dt * 0.5f;
                    for (int s = 0; s < 2; ++s) {
                        for (int i = 1; i < Np; ++i) {
                            const float vx = (raw_x[i] - prev_x[i]) * 0.96f;
                            const float vy = (raw_y[i] - prev_y[i]) * 0.96f;
                            prev_x[i] = raw_x[i];
                            prev_y[i] = raw_y[i];
                            raw_x[i] += vx;
                            raw_y[i] += vy - 9.81f * sdt * sdt;
                        }
                        for (int i = 1; i < Np; ++i) {
                            const float dx = raw_x[i] - raw_x[i - 1];
                            const float dy = raw_y[i] - raw_y[i - 1];
                            const float d = std::sqrt(dx * dx + dy * dy);
                            if (d > 1e-6f) {
                                const float err = (d - kSeg) / d;
                                const float w = (i == 1) ? 1.0f : 0.5f;
                                raw_x[i] -= w * err * dx;
                                raw_y[i] -= w * err * dy;
                                if (i > 1) {
                                    raw_x[i - 1] += 0.5f * err * dx;
                                    raw_y[i - 1] += 0.5f * err * dy;
                                }
                            }
                        }
                        project_live_body(raw_x, raw_y, root_x, spine_tilt, thigh_x, thigh_y);
                    }
                    for (int i = 0; i < Np; ++i) {
                        loc_curr_x[i] = raw_x[i] - root_x;
                        loc_curr_y[i] = raw_y[i];
                    }
                }

                std::array<float, Np> disp_x = raw_x;
                std::array<float, Np> disp_y = raw_y;
                if (tick.alpha < 0.999f) {
                    disp_x[0] = root_x - kMinNormalDist;
                    disp_y[0] = 0.0f;
                    for (int i = 1; i < Np; ++i) {
                        disp_x[i] = root_x + loc_prev_x[i] + (loc_curr_x[i] - loc_prev_x[i]) * tick.alpha;
                        disp_y[i] = loc_prev_y[i] + (loc_curr_y[i] - loc_prev_y[i]) * tick.alpha;
                    }
                    if (post_collide) {
                        project_live_body(disp_x, disp_y, root_x, spine_tilt, thigh_x, thigh_y);
                    }
                }

                if (t >= 0.4f && t <= 0.95f) {
                    idle_min_x = std::min(idle_min_x, disp_x[Np - 1] - root_x);
                    idle_max_x = std::max(idle_max_x, disp_x[Np - 1] - root_x);
                }
                if (t >= 1.05f) {
                    for (int i = 1; i < Np; ++i) {
                        const float max_allowed_x = back_surface_x(root_x, spine_tilt, disp_y[i]) - kMinNormalDist;
                        const float back_pen = (disp_x[i] - max_allowed_x) * 1000.0f;
                        m.max_back_pen_mm = std::max(m.max_back_pen_mm, back_pen);

                        const float dx = disp_x[i] - thigh_x;
                        const float dy = disp_y[i] - thigh_y;
                        const float thigh_pen = (kThighRadius - std::sqrt(dx * dx + dy * dy)) * 1000.0f;
                        m.max_thigh_pen_mm = std::max(m.max_thigh_pen_mm, thigh_pen);
                    }
                }
            }
            m.idle_ripple_mm = (idle_max_x - idle_min_x) * 1000.0f;
            return m;
        };

        const auto broken_60 = run_fixed30_sprint_test(60.0f, false);
        assert(broken_60.max_back_pen_mm > 25.0f || broken_60.max_thigh_pen_mm > 25.0f);

        for (const float fps : {30.0f, 60.0f, 120.0f, 170.0f, 500.0f}) {
            const auto fixed = run_fixed30_sprint_test(fps, true);
            assert(fixed.max_back_pen_mm <= 0.01f);
            assert(fixed.max_thigh_pen_mm <= 0.01f);
            assert(fixed.idle_ripple_mm < 0.05f);
        }
        const auto fixed_60 = run_fixed30_sprint_test(60.0f, true);
        const auto fixed_500 = run_fixed30_sprint_test(500.0f, true);
        std::printf("Camera motion: 60..500 FPS fixed30 live-body back+capsule pushout PASS "
                    "(60_unprojected_pen=%.1f/%.1fmm -> 60_fixed=%.2f/%.2fmm, "
                    "500_fixed=%.2f/%.2fmm, idle_ripple=%.3fmm)\n",
                    broken_60.max_back_pen_mm, broken_60.max_thigh_pen_mm,
                    fixed_60.max_back_pen_mm, fixed_60.max_thigh_pen_mm,
                    fixed_500.max_back_pen_mm, fixed_500.max_thigh_pen_mm,
                    fixed_500.idle_ripple_mm);
    }

    // 16. Mode 1.1 (Character-Local FP64 + Explicit Velocity + Every-Substep Inelastic
    //     LocalRange/StretchLink Projection) at Central Yharnam (380, -45, -520) up to 500 FPS:
    //     - Reproduces exact BD_M_1300_c.hkx failure mode of world-space FP32 Havok at 500 FPS:
    //       (a) LocalRange skipped on substeps 0..1 (`0x1167a80`) + no prev_pos update (`0x2553210`)
    //           against an 8 mm tighter StretchLink pulls particles inside the cage and slams them
    //           outward at >10 m/s fake Verlet bounce velocity;
    //       (b) 500 m world-space FP32 coordinates quantize motion to 0.061 mm steps (`0.087 m/s` at
    //           0.7 ms substep_dt).
    //     - Verifies Mode 1.1 eliminates both completely (<0.005 m/s idle speed, 0.00 mm cage breach,
    //       100% preserved tangential pendulum velocity).
    {
        // Direct unit check on ApplyInelasticConstraintProjection3D:
        {
            const float sdt = 0.0007f; // ~476 FPS, 3 substeps
            // Particle moving inward (-X = -1.0 m/s) and tangentially (-Y = -2.5 m/s), clamped +8 mm along +X:
            const float pre[3] = {-0.008f, -0.300f, 0.0f};
            const float post[3] = {0.000f, -0.300f, 0.0f};
            float prev[3] = {pre[0] - (-1.0f) * sdt, pre[1] - (-2.5f) * sdt, 0.0f};
            const float raw_vx = (post[0] - prev[0]) / sdt;
            assert(raw_vx > 10.0f); // >10 m/s fake bounce before inelastic projection!
            BbClothExp::ApplyInelasticConstraintProjection3D(pre, post, prev);
            const float proj_vx = (post[0] - prev[0]) / sdt;
            const float proj_vy = (post[1] - prev[1]) / sdt;
            assert(std::fabs(proj_vx) < 1e-4f);
            assert(std::fabs(proj_vy - (-2.5f)) < 1e-3f);
        }

        struct Mode11Metrics {
            double max_idle_speed = 0.0;
            double max_cage_pen_mm = 0.0;
            double breath_step_err_um = 0.0;
        };

        auto run_yharnam_500fps = [](bool mode_1_1, float fps) -> Mode11Metrics {
            constexpr double kAnchorBase[3] = {380.0, -45.0, -520.0};
            constexpr int kPs4Substeps = 3;
            constexpr double kPs4Sdt = (1.0 / 30.0) / double(kPs4Substeps);
            constexpr float kStretchTargetX = -0.008f; // 8 mm tighter than LocalRange cage (x >= 0.0)
            constexpr float kCageMinX = 0.000f;
            const double dt = 1.0 / double(fps);
            const int total_frames = static_cast<int>(std::ceil(1.2 * fps));

            // Particle starts settled on the cage surface with subtle 0.2 mm breathing on anchor.
            double x_loc[3] = {0.0, -0.35, 0.0};
            double v_world[3] = {0.0, 0.0, 0.0};
            float wx_cur = static_cast<float>(kAnchorBase[0] + x_loc[0]);
            float wy_cur = static_cast<float>(kAnchorBase[1] + x_loc[1]);
            float wz_cur = static_cast<float>(kAnchorBase[2] + x_loc[2]);
            float wx_prev = wx_cur;
            float wy_prev = wy_cur;
            float wz_prev = wz_cur;

            Mode11Metrics m{};
            double prev_breath_rel_x = 0.0;

            for (int f = 0; f < total_frames; ++f) {
                const double t = double(f + 1) * dt;
                const double breath_x = 0.0002 * std::sin(2.2 * t);
                const double anc[3] = {kAnchorBase[0] + breath_x, kAnchorBase[1], kAnchorBase[2]};

                if (!mode_1_1) {
                    // Legacy world-space FP32 Havok (3 substeps, LocalRange only on last substep,
                    // no inelastic prev_pos update):
                    const float sdt = static_cast<float>(dt / double(kPs4Substeps));
                    const float ratio = static_cast<float>(double(sdt) / kPs4Sdt);
                    const float stretch_stiff = std::pow(ratio, 1.725f) * 0.9f;
                    const float cage_wx = static_cast<float>(anc[0]) + kCageMinX;
                    const float target_wx = static_cast<float>(anc[0]) + kStretchTargetX;

                    for (int s = 0; s < kPs4Substeps; ++s) {
                        const float vx = (wx_cur - wx_prev) / sdt;
                        const float vy = (wy_cur - wy_prev) / sdt;
                        wx_prev = wx_cur;
                        wy_prev = wy_cur;
                        wx_cur += vx * sdt;
                        wy_cur += (vy - 9.81f * sdt) * sdt;
                        // Vertical pin tether at -0.35 m:
                        const float target_wy = static_cast<float>(anc[1] - 0.35);
                        wy_cur = target_wy;
                        wy_prev = target_wy;
                        // StretchLink pulls toward target_wx (-8 mm inside cage):
                        wx_cur += (target_wx - wx_cur) * stretch_stiff;
                        // Original 0x1167a80 runs LocalRange ONLY on s == kPs4Substeps - 1 without prev_pos fix:
                        if (s == kPs4Substeps - 1 && wx_cur < cage_wx) {
                            wx_cur = cage_wx;
                        }
                        if (f > int(0.3 * fps)) {
                            const double pen_mm = double(cage_wx - wx_cur) * 1000.0;
                            m.max_cage_pen_mm = std::max(m.max_cage_pen_mm, pen_mm);
                        }
                    }
                    const double end_speed = std::fabs(double(wx_cur - wx_prev) / double(sdt));
                    if (f > int(0.3 * fps)) {
                        m.max_idle_speed = std::max(m.max_idle_speed, end_speed);
                        const double rel_x = double(wx_cur) - anc[0];
                        if (f > int(0.35 * fps)) {
                            m.breath_step_err_um =
                                std::max(m.breath_step_err_um, std::fabs(rel_x - prev_breath_rel_x) * 1e6);
                        }
                        prev_breath_rel_x = rel_x;
                    }
                } else {
                    // Mode 1.1: Local-Space FP64 + adaptive substeps + every-substep inelastic projection:
                    const int eff_sub = std::clamp(int(std::ceil(dt / kPs4Sdt - 0.05)), 1, kPs4Substeps);
                    const double sdt = dt / double(eff_sub);
                    const float sdt_f = static_cast<float>(sdt);
                    const float stretch_stiff =
                        static_cast<float>(std::pow(sdt / kPs4Sdt, 1.725) * 0.9);
                    const double damp = std::pow(0.96, sdt * 30.0);

                    // Shift local coordinate frame to new anchor_w:
                    const double prev_anc_x = kAnchorBase[0] + 0.0002 * std::sin(2.2 * (t - dt));
                    x_loc[0] -= (anc[0] - prev_anc_x);

                    for (int s = 0; s < eff_sub; ++s) {
                        v_world[0] *= damp;
                        v_world[1] = (v_world[1] - 9.81 * sdt) * damp;
                        const double x_old[3] = {x_loc[0], x_loc[1], x_loc[2]};
                        x_loc[0] = x_old[0] + v_world[0] * sdt;
                        x_loc[1] = -0.35; // vertical tether

                        float cur_l[3] = {float(x_loc[0]), float(x_loc[1]), float(x_loc[2])};
                        float prev_l[3] = {float(x_old[0]), float(-0.35), float(x_old[2])};

                        // StretchLink with inelastic projection:
                        {
                            const float pre_c[3] = {cur_l[0], cur_l[1], cur_l[2]};
                            cur_l[0] += (kStretchTargetX - cur_l[0]) * stretch_stiff;
                            BbClothExp::ApplyInelasticConstraintProjection3D(pre_c, cur_l, prev_l);
                        }
                        // LocalRange on EVERY substep with inelastic projection:
                        {
                            const float pre_c[3] = {cur_l[0], cur_l[1], cur_l[2]};
                            if (cur_l[0] < kCageMinX) {
                                cur_l[0] = kCageMinX;
                            }
                            BbClothExp::ApplyInelasticConstraintProjection3D(pre_c, cur_l, prev_l);
                        }

                        x_loc[0] = double(cur_l[0]);
                        x_loc[1] = double(cur_l[1]);
                        v_world[0] = (x_loc[0] - double(prev_l[0])) / sdt;
                        v_world[1] = 0.0;
                        (void)sdt_f;

                        if (f > int(0.3 * fps)) {
                            m.max_cage_pen_mm =
                                std::max(m.max_cage_pen_mm, double(kCageMinX - cur_l[0]) * 1000.0);
                        }
                    }
                    if (f > int(0.3 * fps)) {
                        m.max_idle_speed = std::max(m.max_idle_speed, std::fabs(v_world[0]));
                        if (f > int(0.35 * fps)) {
                            m.breath_step_err_um =
                                std::max(m.breath_step_err_um, std::fabs(x_loc[0] - prev_breath_rel_x) * 1e6);
                        }
                        prev_breath_rel_x = x_loc[0];
                    }
                }
                (void)wz_cur;
                (void)wz_prev;
            }
            return m;
        };

        const auto world_fp32_500 = run_yharnam_500fps(false, 500.0f);
        assert(world_fp32_500.max_idle_speed > 0.5 || world_fp32_500.max_cage_pen_mm > 0.05);

        for (const float fps : {60.0f, 120.0f, 240.0f, 500.0f}) {
            const auto mode11 = run_yharnam_500fps(true, fps);
            assert(mode11.max_cage_pen_mm <= 1e-4);
            assert(mode11.max_idle_speed < 0.005);
            assert(mode11.breath_step_err_um < 1.0);
        }
        const auto mode11_500 = run_yharnam_500fps(true, 500.0f);
        std::printf("Camera motion: Mode 1.1 Local-Space FP64 + inelastic cage/tether 60..500 FPS PASS "
                    "(world_fp32_500: vel=%.3fm/s pen=%.3fmm step=%.1fum -> "
                    "mode1.1_500: vel=%.5fm/s pen=%.4fmm step=%.2fum)\n",
                    world_fp32_500.max_idle_speed, world_fp32_500.max_cage_pen_mm,
                    world_fp32_500.breath_step_err_um, mode11_500.max_idle_speed,
                    mode11_500.max_cage_pen_mm, mode11_500.breath_step_err_um);
    }

    // 17. Mode 1.2 (Small-Steps FP64 XPBD: Exact Havok->Compliance Calibration + Lagrange
    //     Multiplier Accumulation Across 30..500 FPS):
    //     - Verifies inextensible structural fibers (`k = 1.0`, `alpha = 0`) maintain <0.01 mm
    //       error across 30..500 FPS.
    //     - Verifies compliant silk/leather springs (`k = 0.25`, `alpha = w*h0^2*(1-k)/k`) settle
    //       to the exact physical Hookean equilibrium `y_eq = -L0 - m*g*alpha` identically across
    //       30, 60, 170, and 500 FPS (<0.25 mm spread across a 16.7x FPS range!).
    {
        constexpr double kMass = 0.05; // 50 g cloth vertex
        constexpr double kInvMass = 1.0 / kMass;
        constexpr double kGrav = -9.81;
        constexpr double kRestLen = 0.50;
        constexpr double kPs4Sdt = (1.0 / 30.0) / 3.0;

        auto simulate_xpbd_hang = [&](double k_hk, float fps) -> double {
            const double alpha =
                BbClothExp::ComputeXpbdComplianceFromHavok(k_hk, kInvMass, kPs4Sdt, 1.0);
            const double dt = 1.0 / double(fps);
            const int substeps = std::clamp(int(std::ceil(dt / (1.0 / 240.0))), 2, 8);
            const double sdt = dt / double(substeps);
            const int frames = static_cast<int>(std::ceil(2.5 * fps));

            double xa[3] = {0.0, 0.0, 0.0};
            double xb[3] = {0.0, -kRestLen, 0.0};
            double vb[3] = {0.0, 0.0, 0.0};

            for (int f = 0; f < frames; ++f) {
                for (int s = 0; s < substeps; ++s) {
                    const double xa_old[3] = {xa[0], xa[1], xa[2]};
                    const double xb_old[3] = {xb[0], xb[1], xb[2]};
                    const double damp = std::exp(-3.5 * sdt);
                    vb[1] = (vb[1] + kGrav * sdt) * damp;
                    xb[1] += vb[1] * sdt;

                    double lambda = 0.0;
                    for (int gs = 0; gs < 2; ++gs) {
                        BbClothExp::SolveXpbdDistanceLinkFp64(xa, xb, xa_old, xb_old, 0.0, kInvMass,
                                                              kRestLen, alpha, 0.02, sdt, false,
                                                              false, lambda);
                    }
                    vb[0] = (xb[0] - xb_old[0]) / sdt;
                    vb[1] = (xb[1] - xb_old[1]) / sdt;
                    vb[2] = (xb[2] - xb_old[2]) / sdt;
                }
            }
            return xb[1];
        };

        for (const float fps : {30.0f, 60.0f, 170.0f, 500.0f}) {
            const double y_stiff = simulate_xpbd_hang(1.0, fps);
            assert(std::fabs(y_stiff - (-kRestLen)) < 1e-5);
        }

        const double alpha_silk =
            BbClothExp::ComputeXpbdComplianceFromHavok(0.25, kInvMass, kPs4Sdt, 1.0);
        const double expected_y = -kRestLen - (kMass * std::fabs(kGrav) * alpha_silk);
        const double y_30 = simulate_xpbd_hang(0.25, 30.0f);
        const double y_60 = simulate_xpbd_hang(0.25, 60.0f);
        const double y_170 = simulate_xpbd_hang(0.25, 170.0f);
        const double y_500 = simulate_xpbd_hang(0.25, 500.0f);

        for (const double y : {y_30, y_60, y_170, y_500}) {
            assert(std::fabs(y - expected_y) < 0.00025); // within 0.25 mm of exact Hookean sag!
        }
        const double max_spread_mm =
            (std::max({y_30, y_60, y_170, y_500}) - std::min({y_30, y_60, y_170, y_500})) * 1000.0;
        assert(max_spread_mm < 0.25);

        std::printf("Camera motion: Mode 1.2 Small-Steps FP64 XPBD 30..500 FPS invariance PASS "
                    "(exact_eq=%.5fm: 30=%.5fm 60=%.5fm 170=%.5fm 500=%.5fm, spread=%.4fmm)\n",
                    expected_y, y_30, y_60, y_170, y_500, max_spread_mm);
    }

    // 18. Mode 1.2 (`bd_m_1300` Series Stencil Compliance Calibration — Anti-Membrane-Locking):
    //     - Verifies `ComputeXpbdMeshCalibratedCompliance` gives non-zero series compliance on
    //       `k_raw >= 1.0` weave (`0.54`) and over-relaxed bend quads (`1.67 -> 1.00`) while
    //       preserving exact `30..500 FPS` equilibrium invariance (`< 0.15 mm` spread).
    {
        constexpr double kInvMass = 50.0; // 20 g Hunter Garb particle
        constexpr double kPs4Sdt = 1.0 / 30.0;
        const double alpha_weave =
            BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.0, 2.0 * kInvMass, kPs4Sdt, 0.54);
        const double alpha_bend_over =
            BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.67, 4.0 * kInvMass, kPs4Sdt, 1.00);
        const double alpha_tether =
            BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.0, kInvMass, kPs4Sdt, 0.002);
        assert(alpha_weave > 0.05 && alpha_bend_over > 0.20 && alpha_tether < 0.0005);

        auto sim_capelet_strip = [&](float fps) -> double {
            const double dt = 1.0 / double(fps);
            const int sub = std::clamp(int(std::ceil(dt / (1.0 / 180.0))), 2, 6);
            const double sdt = dt / double(sub);
            const double damp = std::pow(1.0 - 0.95, sdt);
            double x0[3] = {0.0, 0.0, 0.0}, x1[3] = {0.0, -0.15, 0.0}, x2[3] = {0.0, -0.30, 0.0};
            double v1[3]{}, v2[3]{};
            for (int f = 0; f < int(std::ceil(2.0 * fps)); ++f) {
                for (int s = 0; s < sub; ++s) {
                    const double o0[3] = {x0[0], x0[1], x0[2]};
                    const double o1[3] = {x1[0], x1[1], x1[2]};
                    const double o2[3] = {x2[0], x2[1], x2[2]};
                    v1[1] = v1[1] * damp - 17.0 * sdt;
                    v2[1] = v2[1] * damp - 17.0 * sdt;
                    x1[1] += v1[1] * sdt;
                    x2[1] += v2[1] * sdt;
                    double l01 = 0.0, l12 = 0.0, lt = 0.0;
                    for (int gs = 0; gs < 2; ++gs) {
                        BbClothExp::SolveXpbdDistanceLinkFp64(
                            x0, x1, o0, o1, 0.0, kInvMass, 0.15,
                            BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.0, kInvMass, kPs4Sdt, 0.54),
                            0.003, sdt, false, false, l01);
                        BbClothExp::SolveXpbdDistanceLinkFp64(
                            x1, x2, o1, o2, kInvMass, kInvMass, 0.15, alpha_weave,
                            0.003, sdt, false, false, l12);
                        BbClothExp::SolveXpbdDistanceLinkFp64(
                            x0, x2, o0, o2, 0.0, kInvMass, 0.305, alpha_tether,
                            0.006, sdt, true, false, lt);
                    }
                    v1[1] = (x1[1] - o1[1]) / sdt;
                    v2[1] = (x2[1] - o2[1]) / sdt;
                }
            }
            return x2[1];
        };

        const double c30 = sim_capelet_strip(30.0f);
        const double c170 = sim_capelet_strip(170.0f);
        const double c500 = sim_capelet_strip(500.0f);
        const double spread_mm =
            (std::max({c30, c170, c500}) - std::min({c30, c170, c500})) * 1000.0;
        assert(c170 < -0.303 && c170 > -0.306 && spread_mm < 0.15);
        std::printf("Camera motion: Mode 1.2 bd_m_1300 series-calibrated drape PASS "
                    "(30=%.5fm 170=%.5fm 500=%.5fm, spread=%.4fmm)\n",
                    c30, c170, c500, spread_mm);
    }

    // 19. Mode 1.2 (`bd_m_1300` `L0` Belt-Strap Anti-Chatter: Inelastic `StretchLink` + Capsule
    //     Normal Clamping + `+/-15%` Substep Hysteresis):
    //     - Verifies `SelectXpbdSubstepsWithHysteresis` eliminates `2 <-> 3` substep toggling on
    //       `+/-8%` frame-time variance around `90 FPS` (`11.11 ms`, FG 2x base rate).
    //     - Verifies `ClampXpbdInelasticPushFp64` stops lateral ping-pong oscillation when a waist
    //       belt particle is pulled into a curved cylindrical thigh capsule by `StretchLink` while
    //       preserving 100% Galilean-invariant forward drag during a `5 m/s` sprint.
    {
        int raw_flips = 0;
        int hyst_flips = 0;
        int prev_raw = 0;
        int active_sub = 0;
        double smooth_dt = 1.0 / 90.0;
        for (int f = 0; f < 180; ++f) {
            const double dt = (1.0 / 90.0) * (1.0 + ((f & 1) ? 0.06 : -0.06));
            const int r = std::clamp(static_cast<int>(std::ceil(dt / (1.0 / 180.0))), 2, 6);
            if (f > 0 && r != prev_raw) {
                ++raw_flips;
            }
            prev_raw = r;

            const double ema_a = std::clamp(dt / (1.0 / 30.0), 0.05, 1.0);
            smooth_dt += (dt - smooth_dt) * ema_a;
            const int next_sub = BbClothExp::SelectXpbdSubstepsWithHysteresis(smooth_dt, active_sub);
            if (f > 0 && next_sub != active_sub) {
                ++hyst_flips;
            }
            active_sub = next_sub;
        }
        assert(raw_flips > 100 && hyst_flips == 0);

        auto sim_belt_against_thigh = [&](bool inelastic_fix, double sprint_vz) -> std::pair<double, double> {
            constexpr double kInvMass = 50.0;
            constexpr double kPs4Sdt = 1.0 / 30.0;
            const double alpha_tether =
                BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.0, kInvMass, kPs4Sdt, 0.002);
            // Exact `bd_m_1300` `L0` twin thigh capsules (`L_Thigh` +0.088, `R_Thigh` -0.088, r_tot=0.1035,
            // cleft z = -0.0545 m while waist pin at z = 0.0 tethers belt to L_max = 0.223 m -> z = -0.047 m):
            double xa[3] = {0.0, 0.0, 0.0};
            double xb[3] = {+0.012, -0.218, -0.055};
            double vb[3] = {0.0, 0.0, sprint_vz};
            double col_l[3] = {+0.088, -0.220, 0.0};
            double col_r[3] = {-0.088, -0.220, 0.0};
            constexpr double kColRad = 0.1035;

            int sub_state = 0;
            double sm_dt = 1.0 / 170.0;
            double min_x = +1e9, max_x = -1e9;

            for (int f = 0; f < 340; ++f) {
                const double dt = (1.0 / 170.0) * (1.0 + 0.05 * std::sin(0.73 * double(f)));
                int sub = std::clamp(int(std::ceil(dt / (1.0 / 180.0))), 2, 6);
                if (inelastic_fix) {
                    sm_dt += (dt - sm_dt) * std::clamp(dt / (1.0 / 30.0), 0.05, 1.0);
                    sub_state = BbClothExp::SelectXpbdSubstepsWithHysteresis(sm_dt, sub_state);
                    sub = sub_state;
                }
                const double sdt = dt / double(sub);
                const double hf_damp = inelastic_fix ? std::pow(0.95, sdt * 30.0) : 1.0;
                const double damp = std::exp(-0.85 * sdt) * hf_damp;

                for (int s = 0; s < sub; ++s) {
                    const double oa[3] = {xa[0], xa[1], xa[2]};
                    xa[2] += sprint_vz * sdt;
                    col_l[2] += sprint_vz * sdt;
                    col_r[2] += sprint_vz * sdt;

                    double ob[3] = {xb[0], xb[1], xb[2]};
                    vb[0] *= damp;
                    vb[1] = (vb[1] - 9.81 * sdt) * damp;
                    vb[2] = sprint_vz + (vb[2] - sprint_vz) * damp;

                    xb[0] += vb[0] * sdt;
                    xb[1] += vb[1] * sdt;
                    xb[2] += vb[2] * sdt;

                    double lt = 0.0;
                    for (int gs = 0; gs < 2; ++gs) {
                        const double pre_b[3] = {xb[0], xb[1], xb[2]};
                        BbClothExp::SolveXpbdDistanceLinkFp64(xa, xb, oa, ob, 0.0, kInvMass, 0.223,
                                                              alpha_tether, 0.006, sdt, true, false,
                                                              lt);
                        if (inelastic_fix) {
                            const double cdx = xb[0] - pre_b[0];
                            const double cdy = xb[1] - pre_b[1];
                            const double cdz = xb[2] - pre_b[2];
                            const double cd2 = cdx * cdx + cdy * cdy + cdz * cdz;
                            if (cd2 > 1e-24) {
                                const double inv_cd = 1.0 / std::sqrt(cd2);
                                const double anc_step = std::max(
                                    0.0, ((xa[0] - oa[0]) * cdx + (xa[1] - oa[1]) * cdy +
                                          (xa[2] - oa[2]) * cdz) *
                                             inv_cd);
                                BbClothExp::ClampXpbdInelasticPushFp64(pre_b, xb, ob, ob, anc_step);
                            }
                        }
                    }

                    // Twin vertical thigh cylinder pushouts in XZ (`L_Thigh` + `R_Thigh`):
                    const double pre_c[3] = {xb[0], xb[1], xb[2]};
                    for (const double* col_c : {col_l, col_r}) {
                        const double rx = xb[0] - col_c[0];
                        const double rz = xb[2] - col_c[2];
                        const double r2 = rx * rx + rz * rz;
                        if (r2 < kColRad * kColRad && r2 > 1e-16) {
                            const double r = std::sqrt(r2);
                            const double push = kColRad - r;
                            xb[0] += (rx / r) * push;
                            xb[2] += (rz / r) * push;
                        }
                    }
                    if (xb[0] != pre_c[0] || xb[2] != pre_c[2]) {
                        xb[0] = double(float(xb[0]));
                        xb[2] = double(float(xb[2]));
                        if (inelastic_fix) {
                            const double cdx = xb[0] - pre_c[0];
                            const double cdz = xb[2] - pre_c[2];
                            const double cd = std::sqrt(cdx * cdx + cdz * cdz);
                            const double uz = (cd > 1e-12) ? (cdz / cd) : 0.0;
                            const double col_step =
                                std::max(0.0, sprint_vz * uz - 0.04) * sdt;
                            BbClothExp::ClampXpbdInelasticPushFp64(pre_c, xb, ob, ob, col_step);
                        }
                    }

                    vb[0] = (xb[0] - ob[0]) / sdt;
                    vb[1] = (xb[1] - ob[1]) / sdt;
                    vb[2] = (xb[2] - ob[2]) / sdt;
                }

                if (f >= 170) {
                    min_x = std::min(min_x, xb[0] - xa[0]);
                    max_x = std::max(max_x, xb[0] - xa[0]);
                }
            }
            return {(max_x - min_x) * 1000.0, vb[2]};
        };

        const auto [raw_sway_mm, raw_vz] = sim_belt_against_thigh(false, 0.0);
        const auto [fix_sway_mm, fix_vz] = sim_belt_against_thigh(true, 0.0);
        const auto [sprint_sway_mm, sprint_vz] = sim_belt_against_thigh(true, 5.0);
        (void)raw_vz;
        (void)fix_vz;
        std::printf("Camera motion: Mode 1.2 L0 belt-strap inelastic contact + hysteresis PASS "
                    "(flips %d->%d, lateral_sway %.3fmm->%.4fmm, sprint_sway=%.4fmm, sprint_vz=%.2fm/s)\n",
                    raw_flips, hyst_flips, raw_sway_mm, fix_sway_mm, sprint_sway_mm, sprint_vz);
        std::fflush(stdout);
        assert(raw_sway_mm > 3.0);
        assert(fix_sway_mm < 0.35 && fix_sway_mm < raw_sway_mm * 0.08);
        assert(sprint_sway_mm < 0.35 && sprint_vz > 4.2);
    }

    // 20. Mode 1.2 (`1.00x` Quadrature Bend Stiffness & Mass-Invariant Area Aerodynamics `0..3x`):
    //     - Verifies `ComputeXpbdMeshCalibratedCompliance(..., 0.36)` preserves >= 2.8x stiffer
    //       leather coat bending at `k_raw = 0.88` vs legacy `intrinsic + 1.00` while flooring
    //       `k_raw >= 1.0` capelets cleanly.
    //     - Verifies area-normalized aerodynamic lift produces identical cape billow angles across
    //       `bd_m_1300` layers (`m = 0.048 kg` vs `m = 0.750 kg`, 15.6x mass spread!) and scales
    //       visibly from `0.00x` (`0.0 deg`) -> `1.00x` (`~28 deg`) -> `3.00x` (`~58 deg`)!
    {
        constexpr double kPs4Sdt = 1.0 / 30.0;
        constexpr double kWq = 80.0;
        const double alpha_new_k88 =
            BbClothExp::ComputeXpbdMeshCalibratedCompliance(0.88, kWq, kPs4Sdt, 0.36, 1.0);
        const double old_series_k88 = kWq * (kPs4Sdt * kPs4Sdt) * (((1.0 - 0.88) / 0.88) + 1.00);
        const double stiff_gain = old_series_k88 / alpha_new_k88;
        assert(stiff_gain > 2.8);

        auto eval_sprint_billow_deg = [&](double vtx_mass_kg, double aero_scale) -> double {
            // 5.0 m/s sprint (`rv = (0, 0, -5)`), `grav_y = -15 m/s^2`, `1.65 kg/m^2` oiled leather/wool:
            constexpr double kAirOverFabric = 1.225 / 1.65;
            constexpr double kGravY = -15.0;
            const double grav_scale = std::clamp(std::fabs(kGravY) / 9.81, 0.80, 2.00);
            const double v_mag = 5.0;
            const double q_acc = 0.5 * kAirOverFabric * (v_mag * v_mag) * grav_scale * aero_scale;
            // Equilibrium trail angle `phi` from vertical under gravity + normal/lift/scoop pressure:
            double phi = 0.0;
            for (int it = 0; it < 40; ++it) {
                // Cape tangent `(0, -cos(phi), -sin(phi))`, inner back normal `(0, +sin(phi), -cos(phi))`,
                // airflow `u = (0, 0, -1)` -> `cos_theta = cos(phi)`:
                const double cos_t = std::cos(phi);
                const double sin_t = std::sin(phi);
                const double a_norm = q_acc * 1.15 * cos_t * std::fabs(cos_t);
                const double a_lift = q_acc * 0.90 * (2.0 * std::fabs(cos_t) * sin_t);
                const double a_scoop_y = q_acc * 0.28 * std::fabs(cos_t);
                const double az = -(a_norm + a_lift) * cos_t;
                const double ay = kGravY + (a_norm + a_lift) * sin_t + a_scoop_y;
                (void)vtx_mass_kg; // Area-normalized acceleration is strictly mass-invariant!
                phi = 0.5 * phi + 0.5 * std::atan2(-az, std::max(0.5, -ay));
            }
            return phi * (180.0 / 3.14159265358979323846);
        };

        const double deg_0x = eval_sprint_billow_deg(0.1667, 0.0);
        const double deg_1x_coat = eval_sprint_billow_deg(0.0482, 1.0);
        const double deg_1x_belt = eval_sprint_billow_deg(0.7500, 1.0);
        const double deg_3x = eval_sprint_billow_deg(0.1667, 3.0);
        assert(std::fabs(deg_0x) < 1e-6);
        assert(std::fabs(deg_1x_coat - deg_1x_belt) < 1e-9);
        assert(deg_1x_coat > 22.0 && deg_1x_coat < 48.0);
        assert(deg_3x > deg_1x_coat + 18.0);

        std::printf("Camera motion: Mode 1.2 1.00x quadrature bend (%.2fx stiffer) + 0..3x aero PASS "
                    "(billow 0x=%.1fdeg, 1x=%.1fdeg, 3x=%.1fdeg)\n",
                    stiff_gain, deg_0x, deg_1x_coat, deg_3x);
    }

    // 21. Mode 1.2 Trailer Hunter Belt (`bd_m_2200` `BD_M_2200_belt` `Sim[3]` 1-Triangle `MeshDeform`
    //     Proxy Without `StretchLink` + `scaleNormalBehaviour = 1`) Zero-Inflation & Rubber-Band Test:
    //     - Verifies `IsXpbdStrapOrProxyTopology` classifies narrow straps & 1-triangle `MeshDeform`
    //       proxies (`Sim[3]`, `Sim[4]`, `BD_M_1300_ac`) while keeping wide coat/cape sheets (`L3`,
    //       `L4`) on anti-locking `0.52` weave.
    //     - Verifies that under `Compliance = 3.50x` + walking/landing impulse, decoupling bending
    //       softness from `StandardLink` stretching + `0.004` leather strap weave reduces 3D belt
    //       `MeshDeform` height inflation & rubber bounce by >25x (`< 0.5 mm` vs `> 18 mm`).
    {
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(3, 1, 3, 3, false) == true);         // BD_M_2200_belt Sim[3]
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(23, 20, 22, 40, true) == true);      // BD_M_2200_belt Sim[4]
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(24, 16, 24, 36, true) == true);      // BD_M_1300_ac
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(7624, 248, 70, 230, false) == true); // c5500 boss mesh without StretchLink
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(197, 289, 105, 486, true) == false); // BD_M_1300_mantle
        assert(BbClothExp::IsXpbdStrapOrProxyTopology(234, 349, 117, 582, true) == false); // BD_M_1300_cape
        {
            const BbClothExp::XpbdTuningParams def_xpbd{};
            assert(std::abs(def_xpbd.compliance_scale - 2.50f) < 1e-4f);
            assert(std::abs(def_xpbd.aero_lift_scale - 0.25f) < 1e-4f);
            assert(std::abs(def_xpbd.damping_scale - 1.50f) < 1e-4f);
        }

        auto sim_trailer_belt_proxy = [&](bool fixed_strap_law, double comp_scale) {
            // Exact `bd_m_2200.partsbnd.dcx` `Sim[3]` (`n = 3`, `P[1]` mass = `8.0 kg`, `0` StretchLinks!):
            const double p0[3] = {+0.0857, +1.1179, +0.0989};
            const double p1_init[3] = {-0.0268, +1.0439, +0.1476};
            const double p2[3] = {-0.1153, +1.1302, +0.0983};
            const double rest10 = std::sqrt((p1_init[0] - p0[0]) * (p1_init[0] - p0[0]) +
                                            (p1_init[1] - p0[1]) * (p1_init[1] - p0[1]) +
                                            (p1_init[2] - p0[2]) * (p1_init[2] - p0[2]));
            const double rest21 = std::sqrt((p1_init[0] - p2[0]) * (p1_init[0] - p2[0]) +
                                            (p1_init[1] - p2[1]) * (p1_init[1] - p2[1]) +
                                            (p1_init[2] - p2[2]) * (p1_init[2] - p2[2]));
            constexpr double kW1 = 0.125; // 8.0 kg vertex!
            constexpr double kPs4Sdt = 1.0 / 30.0;
            constexpr double kSdt = (1.0 / 90.0) / 2.0;

            const double stencil_c = fixed_strap_law ? 0.0005 : 0.52;
            const double beta = fixed_strap_law ? 0.060 : 0.003;
            const double eff_scale =
                fixed_strap_law ? BbClothExp::ComputeXpbdEffectiveWeaveScale(comp_scale, true)
                                : comp_scale;
            const double alpha =
                BbClothExp::ComputeXpbdMeshCalibratedCompliance(1.0, kW1, kPs4Sdt, stencil_c, 1.0) *
                eff_scale;

            double x0[3] = {p0[0], p0[1], p0[2]};
            double x1[3] = {p1_init[0], p1_init[1], p1_init[2]};
            double x2[3] = {p2[0], p2[1], p2[2]};
            double v1[3] = {0.0, -1.8, 0.0}; // downward landing impulse

            double max_sag_mm = 0.0;
            double eq_sag_mm = 0.0;
            double min_post_mm = 1e9;
            double max_post_mm = -1e9;
            for (int step = 0; step < 180; ++step) {
                const double old1[3] = {x1[0], x1[1], x1[2]};
                const double damp = std::exp(-0.35 * kSdt);
                v1[0] *= damp;
                v1[1] = v1[1] * damp - 9.81 * kSdt;
                v1[2] *= damp;
                x1[0] += v1[0] * kSdt;
                x1[1] += v1[1] * kSdt;
                x1[2] += v1[2] * kSdt;
                if (x1[2] < p1_init[2]) {
                    x1[2] = p1_init[2]; // waist minNormalDist floor
                }
                double lam0 = 0.0, lam2 = 0.0;
                for (int gs = 0; gs < 2; ++gs) {
                    BbClothExp::SolveXpbdDistanceLinkFp64(x0, x1, x0, old1, 0.0, kW1, rest10, alpha,
                                                          beta, kSdt, false, false, lam0);
                    BbClothExp::SolveXpbdDistanceLinkFp64(x2, x1, x2, old1, 0.0, kW1, rest21, alpha,
                                                          beta, kSdt, false, false, lam2);
                }
                double old1_clamped[3] = {old1[0], old1[1], old1[2]};
                if (x1[2] < p1_init[2]) {
                    x1[2] = p1_init[2];
                    if (old1_clamped[2] < p1_init[2]) {
                        old1_clamped[2] = p1_init[2];
                    }
                }
                v1[0] = (x1[0] - old1_clamped[0]) / kSdt;
                v1[1] = (x1[1] - old1_clamped[1]) / kSdt;
                v1[2] = (x1[2] - old1_clamped[2]) / kSdt;

                const double sag_mm = (p1_init[1] - x1[1]) * 1000.0;
                max_sag_mm = std::max(max_sag_mm, sag_mm);
                eq_sag_mm = sag_mm;
                if (step >= 30) {
                    min_post_mm = std::min(min_post_mm, sag_mm);
                    max_post_mm = std::max(max_post_mm, sag_mm);
                }
            }
            // `Op[5] MeshDeform_2` extrapolates belt root height by ~3.16x of `P[1]` sag!
            struct BeltMetrics {
                double eq_mm;
                double peak_mm;
                double bounce_mm;
            };
            return BeltMetrics{eq_sag_mm * 3.16, max_sag_mm * 3.16,
                               (max_post_mm - min_post_mm) * 3.16};
        };

        const auto old_m = sim_trailer_belt_proxy(false, 3.5);
        const auto new_m = sim_trailer_belt_proxy(true, 3.5);
        std::printf("Camera motion: Mode 1.2 Trailer belt MeshDeform zero-inflation PASS "
                    "(3.5x eq %.1fmm->%.3fmm, peak %.1fmm->%.2fmm, rubber_bounce %.1fmm->%.3fmm)\n",
                    old_m.eq_mm, new_m.eq_mm, old_m.peak_mm, new_m.peak_mm, old_m.bounce_mm,
                    new_m.bounce_mm);
        std::fflush(stdout);
        assert(old_m.peak_mm > 200.0 && old_m.bounce_mm > 200.0);
        assert(std::fabs(new_m.eq_mm) < 0.50 && new_m.peak_mm < 9.0 && new_m.bounce_mm < 0.25);
    }

    vmaDestroyBuffer(instance.GetAllocator(),staging,allocation);
}
