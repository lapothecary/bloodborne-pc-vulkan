// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/present_passes.h"

#include <array>

#include "vulkan/platform.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/fullscreen_vert.h"
#include "bbgpu/shaders/present_encode_frag.h"
#include "bbgpu/shaders/present_encode_lut1d_frag.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

vk::UniqueSampler LinearClampSampler(vk::Device device) {
    return Check<"create present sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = -1000.0f,
        .maxLod = 1000.0f,
    }));
}

} // namespace

EncodePass::EncodePass(vk::Device device, vk::Format sdr_output_format)
    : sampler{LinearClampSampler(device)} {
    const vk::DescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };
    set_layout = Check<"create encode set layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 1,
        .pBindings = &binding,
    }));
    const vk::PushConstantRange constants{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .size = sizeof(Settings),
    };
    const vk::DescriptorSetLayout set = *set_layout;
    layout = Check<"create encode layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constants,
    }));

    const std::array<vk::DescriptorSetLayoutBinding, 2> fused_bindings{{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    }};
    set_layout_fused =
        Check<"create encode fused set layout">(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(fused_bindings.size()),
            .pBindings = fused_bindings.data(),
        }));
    const vk::DescriptorSetLayout set_fused = *set_layout_fused;
    layout_fused = Check<"create encode fused layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_fused,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constants,
    }));

    const vk::ShaderModule vertex = CompileSPV(FULLSCREEN_VERT, device);
    const vk::ShaderModule fragment = CompileSPV(PRESENT_ENCODE_FRAG, device);
    const vk::ShaderModule fragment_fused = CompileSPV(PRESENT_ENCODE_LUT1D_FRAG, device);
    ASSERT_MSG(vertex && fragment && fragment_fused, "encode pass shaders did not compile");
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertex,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragment,
            .pName = "main",
        },
    };
    const std::array stages_fused{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertex,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragment_fused,
            .pName = "main",
        },
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo raster{.lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const vk::PipelineColorBlendAttachmentState write_all{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend{
        .attachmentCount = 1,
        .pAttachments = &write_all,
    };
    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = u32(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    auto build_pipeline = [&](vk::Format fmt, const char* name,
                              const std::array<vk::PipelineShaderStageCreateInfo, 2>& stg,
                              vk::PipelineLayout pipe_layout) {
        const vk::PipelineRenderingCreateInfo rendering{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &fmt,
        };
        auto pipe = Check<"create encode pipeline">(device.createGraphicsPipelineUnique(
            {}, vk::GraphicsPipelineCreateInfo{
                    .pNext = &rendering,
                    .stageCount = u32(stg.size()),
                    .pStages = stg.data(),
                    .pVertexInputState = &vertex_input,
                    .pInputAssemblyState = &assembly,
                    .pViewportState = &viewport,
                    .pRasterizationState = &raster,
                    .pMultisampleState = &multisample,
                    .pColorBlendState = &blend,
                    .pDynamicState = &dynamic,
                    .layout = pipe_layout,
                }));
        SetObjectName(device, *pipe, name);
        return pipe;
    };
    pipeline_sdr = build_pipeline(sdr_output_format, "present encode sdr", stages, *layout);
    pipeline_hdr = build_pipeline(vk::Format::eA2B10G10R10UnormPack32, "present encode hdr10",
                                  stages, *layout);
    pipeline_sdr_fused = build_pipeline(sdr_output_format, "present encode sdr lut1d",
                                        stages_fused, *layout_fused);
    pipeline_hdr_fused = build_pipeline(vk::Format::eA2B10G10R10UnormPack32,
                                        "present encode hdr10 lut1d", stages_fused, *layout_fused);
    device.destroyShaderModule(vertex);
    device.destroyShaderModule(fragment);
    device.destroyShaderModule(fragment_fused);
}

EncodePass::~EncodePass() = default;

void EncodePass::Render(vk::CommandBuffer cmdbuf, vk::ImageView input, vk::ImageView output,
                        vk::Extent2D output_size, const Settings& settings,
                        vk::ImageLayout input_layout, vk::ImageView lut1d_view,
                        vk::ImageLayout lut1d_layout) {
    const vk::RenderingAttachmentInfo attachment{
        .imageView = output,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::DescriptorImageInfo source{
        .sampler = *sampler,
        .imageView = input,
        .imageLayout = input_layout,
    };
    const bool fused = bool(lut1d_view);
    const vk::PipelineLayout active_layout = fused ? *layout_fused : *layout;
    const vk::Pipeline active_pipe =
        fused ? (settings.hdr != 0u ? *pipeline_hdr_fused : *pipeline_sdr_fused)
              : (settings.hdr != 0u ? *pipeline_hdr : *pipeline_sdr);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, active_pipe);
    cmdbuf.setViewport(0, vk::Viewport{0.0f, 0.0f, float(output_size.width),
                                       float(output_size.height), 0.0f, 1.0f});
    cmdbuf.setScissor(0, vk::Rect2D{{0, 0}, output_size});
    if (fused) {
        const vk::DescriptorImageInfo lut1d{
            .sampler = *sampler,
            .imageView = lut1d_view,
            .imageLayout = lut1d_layout,
        };
        const std::array<vk::WriteDescriptorSet, 2> writes{{
            {
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &source,
            },
            {
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &lut1d,
            },
        }};
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, active_layout, 0, writes);
    } else {
        const vk::WriteDescriptorSet write{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &source,
        };
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, active_layout, 0, write);
    }
    cmdbuf.pushConstants(active_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(Settings),
                         &settings);
    cmdbuf.beginRendering({
        .renderArea = {{0, 0}, output_size},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    cmdbuf.draw(3, 1, 0, 0);
    cmdbuf.endRendering();
}

} // namespace Vulkan
