// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_vsm_blur_batcher.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <fmt/format.h>
#include <vk_mem_alloc.h>

#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "shim/bbport_toggles.h"
#include "textures/image.h"
#include "textures/image_view.h"
#include "textures/texture_cache.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/vsm_blur_to_scratch_comp.h"

namespace Vulkan {

VsmBlurBatcher::VsmBlurBatcher(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                               VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_} {}

VsmBlurBatcher::~VsmBlurBatcher() {
    if (params_buf && params_alloc) {
        vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(params_buf), params_alloc);
        params_buf = VK_NULL_HANDLE;
        params_alloc = nullptr;
    }
}

bool VsmBlurBatcher::Enabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_VSM_COMPUTE_BLUR");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

bool VsmBlurBatcher::ConflictsWithCopyBack(VideoCore::ImageId id, u32 mip,
                                           u32 layer) const noexcept {
    for (const auto& b : pending_blurs) {
        if (b.face_id == id && b.mip == mip && b.layer == layer) {
            return true;
        }
    }
    return false;
}

void VsmBlurBatcher::EnsurePipelines() {
    if (pipelines_ready) {
        return;
    }
    const vk::Device device = instance.GetDevice();

    linear_clamp_sampler = Check<"vsm blur linear sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }));

    const vk::PushConstantRange pc_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    }};
    scratch_set_layout = Check<"vsm blur scratch set layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::DescriptorSetLayout set_l = *scratch_set_layout;
    scratch_pipe_layout = Check<"vsm blur scratch pipe layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_l,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    }));
    const vk::ShaderModule cs = CompileSPV(VSM_BLUR_TO_SCRATCH_COMP, device);
    ASSERT_MSG(cs, "vsm_blur_to_scratch.comp failed to compile");
    scratch_pipeline = Check<"vsm blur scratch pipeline">(device.createComputePipelineUnique(
        {}, {
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = cs,
                          .pName = "main"},
                .layout = *scratch_pipe_layout,
            }));
    SetObjectName(device, *scratch_pipeline, "vsm blur single wave");
    device.destroyShaderModule(cs);

    pipelines_ready = true;
}

void VsmBlurBatcher::EnsureScratchPool() {
    if (scratch_ready) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    scratch_pool = VideoCore::HostImage(device, instance.GetAllocator());
    scratch_pool.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16Sfloat,
        .extent = {kScratchDim, kScratchDim, 1u},
        .mipLevels = 1u,
        .arrayLayers = kMaxScratchSlices,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    SetObjectName(device, vk::Image(scratch_pool), "vsm blur scratch pool 256x256x64");

    for (u32 i = 0; i < kMaxScratchSlices; ++i) {
        scratch_slice_views[i] = Check<"vsm blur scratch slice view">(device.createImageViewUnique({
            .image = vk::Image(scratch_pool),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0u, 1u, i, 1u},
        }));
    }

    const vk::ImageMemoryBarrier2 init_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(scratch_pool),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, kMaxScratchSlices},
    };
    scheduler.EndRendering();
    scheduler.Record([init_barrier](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &init_barrier});
    });

    scratch_ready = true;
}

void VsmBlurBatcher::EnsureParamsBuffer() {
    if (params_ready) {
        return;
    }
    const vk::BufferCreateInfo bci{
        .size = u64(kMaxScratchSlices) * kParamSlotStride,
        .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkBuffer raw_buf = VK_NULL_HANDLE;
    const VkResult res = vmaCreateBuffer(instance.GetAllocator(),
                                         reinterpret_cast<const VkBufferCreateInfo*>(&bci), &aci,
                                         &raw_buf, &params_alloc, nullptr);
    ASSERT_MSG(res == VK_SUCCESS, "vsm blur params buffer creation failed");
    params_buf = vk::Buffer(raw_buf);
    SetObjectName(instance.GetDevice(), params_buf, "vsm blur params snapshot 32KiB");
    params_ready = true;
}

bool VsmBlurBatcher::TryConsumeDeferredAndBlur(VideoCore::ImageId cb0_id, u32 cb0_mip,
                                               u32 cb0_layer, VAddr cb0_addr, VAddr fs_img0_addr,
                                               const vk::DescriptorBufferInfo& ssbo1,
                                               const vk::DescriptorBufferInfo& ssbo2, u32 d0_dw,
                                               u32 d1_dw, bool ssbo1_immutable,
                                               bool ssbo2_immutable) {
    if (!deferred_copy.valid || !cb0_id || !ssbo1.buffer || !ssbo2.buffer ||
        !IsSupportedDim(deferred_copy.dim)) {
        return false;
    }
    auto& src_copy_img = texture_cache.GetImage(deferred_copy.src_id);
    auto& dst_copy_img = texture_cache.GetImage(deferred_copy.dst_id);
    auto& face_img = texture_cache.GetImage(cb0_id);

    const bool dst_matches =
        fs_img0_addr != 0 &&
        (dst_copy_img.info.guest_address == fs_img0_addr ||
         deferred_copy.args.dst_va == fs_img0_addr ||
         deferred_copy.dst_tsharp_addr == fs_img0_addr);
    const bool src_matches =
        (deferred_copy.src_id == cb0_id && deferred_copy.src_mip == cb0_mip &&
         deferred_copy.src_layer == cb0_layer) ||
        (src_copy_img.GetImage() == face_img.GetImage() && deferred_copy.src_mip == cb0_mip &&
         deferred_copy.src_layer == cb0_layer) ||
        (cb0_addr != 0 &&
         (deferred_copy.src_tsharp_addr == cb0_addr || deferred_copy.args.src_va == cb0_addr) &&
         deferred_copy.src_mip == cb0_mip && deferred_copy.src_layer == cb0_layer);

    if (!src_matches || !dst_matches) {
        return false;
    }
    if (face_img.info.pixel_format != vk::Format::eR16G16Sfloat ||
        face_img.info.num_samples != 1u || face_img.info.props.is_depth ||
        face_img.info.props.is_block) {
        return false;
    }

    const u32 dim = deferred_copy.dim;
    // Cancel the deferred `Copy(F[slice] -> S)` completely!
    deferred_copy = {};

    if (pending_blurs.size() >= kMaxScratchSlices ||
        ConflictsWithCopyBack(cb0_id, cb0_mip, cb0_layer)) {
        FlushCopyBacks();
    }

    const u32 slot = static_cast<u32>(pending_blurs.size());
    vk::DescriptorBufferInfo use_ssbo1 = ssbo1;
    vk::DescriptorBufferInfo use_ssbo2 = ssbo2;
    u32 use_d0 = d0_dw;
    u32 use_d1 = d1_dw;

    if (!ssbo1_immutable || !ssbo2_immutable) {
        EnsureParamsBuffer();
        scheduler.EndRendering();
        const u64 slot_base = u64(slot) * kParamSlotStride;
        const vk::Buffer dst_p = params_buf;
        if (!ssbo1_immutable) {
            const vk::BufferCopy c1{
                .srcOffset = ssbo1.offset + u64(d0_dw) * 4u,
                .dstOffset = slot_base,
                .size = 208u, // 13 * vec4
            };
            scheduler.Record([src = ssbo1.buffer, dst_p, c1](vk::CommandBuffer cmdbuf) {
                cmdbuf.copyBuffer(src, dst_p, 1u, &c1);
            });
            use_ssbo1 = vk::DescriptorBufferInfo{dst_p, slot_base, 208u};
            use_d0 = 0u;
        }
        if (!ssbo2_immutable) {
            const vk::BufferCopy c2{
                .srcOffset = ssbo2.offset + u64(d1_dw) * 4u,
                .dstOffset = slot_base + 256u,
                .size = 32u, // scale at dwords 4..5
            };
            scheduler.Record([src = ssbo2.buffer, dst_p, c2](vk::CommandBuffer cmdbuf) {
                cmdbuf.copyBuffer(src, dst_p, 1u, &c2);
            });
            use_ssbo2 = vk::DescriptorBufferInfo{dst_p, slot_base + 256u, 32u};
            use_d1 = 0u;
        }
        used_params_this_wave = true;
    }

    pending_blurs.push_back({
        .face_id = cb0_id,
        .mip = cb0_mip,
        .layer = cb0_layer,
        .dim = dim,
        .scratch_slot = slot,
        .ssbo1 = use_ssbo1,
        .ssbo2 = use_ssbo2,
        .d0_dw = use_d0,
        .d1_dw = use_d1,
    });

    if (!logged_first_use) {
        logged_first_use = true;
        std::printf("VsmBlurBatcher: Pack #4 v2 Single-Wave Compute active (imm1=%d imm2=%d, "
                    "BB_VSM_COMPUTE_BLUR=0 to disable)\n",
                    ssbo1_immutable ? 1 : 0, ssbo2_immutable ? 1 : 0);
    }
    return true;
}

void VsmBlurBatcher::FlushCopyBacks() {
    if (pending_blurs.empty()) {
        return;
    }
    const u32 num_faces = static_cast<u32>(pending_blurs.size());
    EnsurePipelines();
    EnsureScratchPool();
    scheduler.EndRendering();

    // 1. Transition ALL queued cubemap slices ColorAttachment -> ShaderReadOnly in ONE batch flush.
    struct RecordedDispatch {
        vk::DescriptorBufferInfo ssbo1{};
        vk::DescriptorBufferInfo ssbo2{};
        vk::ImageView src_view{};
        vk::ImageView dst_view{};
        PushConstants pc{};
        u32 groups = 1u;
    };
    boost::container::small_vector<RecordedDispatch, kMaxScratchSlices> dispatches;
    dispatches.reserve(num_faces);

    for (const auto& b : pending_blurs) {
        auto& face_img = texture_cache.GetImage(b.face_id);
        const VideoCore::SubresourceRange range{
            .base = {b.mip, b.layer},
            .extent = {1u, 1u},
        };
        runtime.Transit(&face_img, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead, range);

        VideoCore::ImageViewInfo sampled_view_info{};
        sampled_view_info.type = Gcn::ImageType::Color2D;
        sampled_view_info.format = vk::Format::eR16G16Sfloat;
        sampled_view_info.range = range;
        sampled_view_info.is_storage = false;
        const vk::ImageView src_view = *face_img.FindView(sampled_view_info).image_view;
        dispatches.push_back({
            .ssbo1 = b.ssbo1,
            .ssbo2 = b.ssbo2,
            .src_view = src_view,
            .dst_view = *scratch_slice_views[b.scratch_slot],
            .pc = {b.dim, b.d0_dw, b.d1_dw, 0u},
            .groups = (b.dim + 15u) / 16u,
        });
    }
    runtime.FlushBarriers();

    if (used_params_this_wave) {
        const vk::BufferMemoryBarrier2 param_bar{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer = params_buf,
            .offset = 0,
            .size = u64(num_faces) * kParamSlotStride,
        };
        scheduler.Record([param_bar](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2({.bufferMemoryBarrierCount = 1,
                                     .pBufferMemoryBarriers = &param_bar});
        });
    }

    // 2. ONE single Compute wave for ALL queued faces (1 Graphics->Compute switch, 1 pipeline bind,
    //    0 barriers between faces — all 32..256 faces run concurrently across all SMs!).
    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Shadow_VsmAndPcf);
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        prof->Mark(0x9A71FE35ULL, [] {
            return std::string{"VsmBlurCompute: SingleWaveAllFaces [VULKAN-NATIVE]"};
        });
    }
    const vk::Pipeline pipe = *scratch_pipeline;
    const vk::PipelineLayout layout = *scratch_pipe_layout;
    const vk::Sampler samp = *linear_clamp_sampler;
    const auto rec_dispatches =
        scheduler.RecordData(std::span<const RecordedDispatch>{dispatches});
    scheduler.Record([pipe, layout, samp, rec_dispatches](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
        for (const auto& d : rec_dispatches) {
            const vk::DescriptorImageInfo src_info{
                .sampler = samp,
                .imageView = d.src_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dst_info{
                .imageView = d.dst_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 4> writes{{
                {
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &d.ssbo1,
                },
                {
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &d.ssbo2,
                },
                {
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &src_info,
                },
                {
                    .dstBinding = 3,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &dst_info,
                },
            }};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants),
                                 &d.pc);
            cmdbuf.dispatch(d.groups, d.groups, 1u);
        }
    });

    // 3. Sort by (face_id, mip, layer) so all 6 faces of each cubemap copy back in 1 vkCmdCopyImage.
    std::sort(pending_blurs.begin(), pending_blurs.end(),
              [](const PendingBlur& a, const PendingBlur& b) {
                  if (a.face_id.index != b.face_id.index) {
                      return a.face_id.index < b.face_id.index;
                  }
                  if (a.mip != b.mip) {
                      return a.mip < b.mip;
                  }
                  return a.layer < b.layer;
              });

    for (const auto& b : pending_blurs) {
        auto& dst_img = texture_cache.GetImage(b.face_id);
        const VideoCore::SubresourceRange range{
            .base = {b.mip, b.layer},
            .extent = {1u, 1u},
        };
        runtime.Transit(&dst_img, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                        range);
    }
    runtime.FlushBarriers();

    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Shadow_VsmAndPcf);
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        prof->Mark(0x9A71FE34ULL, [] {
            return std::string{"VsmBlurCompute: GroupedCopyBack [VULKAN-NATIVE]"};
        });
    }

    const vk::Image scratch_vk = vk::Image(scratch_pool);
    const vk::ImageMemoryBarrier2 pre_copy{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = scratch_vk,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, num_faces},
    };
    scheduler.Record([pre_copy](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &pre_copy});
    });

    size_t i = 0;
    while (i < pending_blurs.size()) {
        const VideoCore::ImageId cur_id = pending_blurs[i].face_id;
        auto& dst_img = texture_cache.GetImage(cur_id);
        const vk::Image dst_vk = dst_img.GetImage();
        boost::container::small_vector<vk::ImageCopy, 12> regions;
        while (i < pending_blurs.size() && pending_blurs[i].face_id == cur_id) {
            const auto& b = pending_blurs[i];
            regions.push_back(vk::ImageCopy{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0u, b.scratch_slot, 1u},
                .srcOffset = {0, 0, 0},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, b.mip, b.layer, 1u},
                .dstOffset = {0, 0, 0},
                .extent = {b.dim, b.dim, 1u},
            });
            ++i;
        }
        const auto recorded_regions = scheduler.RecordData(std::span<const vk::ImageCopy>{regions});
        scheduler.Record([scratch_vk, dst_vk, recorded_regions](vk::CommandBuffer cmdbuf) {
            cmdbuf.copyImage(scratch_vk, vk::ImageLayout::eGeneral, dst_vk,
                             vk::ImageLayout::eTransferDstOptimal, u32(recorded_regions.size()),
                             recorded_regions.data());
        });
        dst_img.flags |= VideoCore::ImageFlagBits::GpuModified;
        dst_img.gpu_write_tick = scheduler.CurrentTick();
    }

    const vk::ImageMemoryBarrier2 post_copy{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = scratch_vk,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, num_faces},
    };
    scheduler.Record([post_copy](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &post_copy});
    });

    if (used_params_this_wave) {
        const vk::BufferMemoryBarrier2 param_post{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = params_buf,
            .offset = 0,
            .size = u64(num_faces) * kParamSlotStride,
        };
        scheduler.Record([param_post](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2({.bufferMemoryBarrierCount = 1,
                                     .pBufferMemoryBarriers = &param_post});
        });
        used_params_this_wave = false;
    }

    pending_blurs.clear();
}

} // namespace Vulkan
