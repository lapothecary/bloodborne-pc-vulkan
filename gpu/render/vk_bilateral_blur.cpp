// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_bilateral_blur.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

#include "bbgpu/shaders/bilateral_blur_2d_comp.h"
#include "bbgpu/shaders/bilateral_blur_h_comp.h"
#include "bbgpu/shaders/bilateral_blur_v_comp.h"

namespace Vulkan {

BilateralBlurCompute::BilateralBlurCompute(const Instance& instance_, Scheduler& scheduler_,
                                           Runtime& runtime_,
                                           VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_} {}

BilateralBlurCompute::~BilateralBlurCompute() = default;

bool BilateralBlurCompute::Enabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_BILATERAL_COMPUTE");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

void BilateralBlurCompute::EnsurePipelines() {
    if (pipeline_h && pipeline_v && pipeline_2d) {
        return;
    }
    const vk::Device dev = instance.GetDevice();

    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
    }};
    desc_layout = Check<"bilateral blur desc layout">(dev.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));

    const vk::PushConstantRange pc_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PushConstants1D),
    };
    const vk::DescriptorSetLayout dsl = *desc_layout;
    pipe_layout = Check<"bilateral blur pipe layout">(dev.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    }));

    auto make_pipe = [&](std::span<const u32> spv, const char* name) -> vk::UniquePipeline {
        const vk::ShaderModule cs = CompileSPV(spv, dev);
        ASSERT_MSG(cs, "{} failed to compile", name);
        const vk::ComputePipelineCreateInfo pci{
            .stage =
                {
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = cs,
                    .pName = "main",
                },
            .layout = *pipe_layout,
        };
        auto pipe = Check<"bilateral blur pipeline">(dev.createComputePipelineUnique({}, pci));
        SetObjectName(dev, *pipe, name);
        dev.destroyShaderModule(cs);
        return pipe;
    };

    pipeline_h = make_pipe(BILATERAL_BLUR_H_COMP, "bilateral_blur_h.comp");
    pipeline_v = make_pipe(BILATERAL_BLUR_V_COMP, "bilateral_blur_v.comp");
    pipeline_2d = make_pipe(BILATERAL_BLUR_2D_COMP, "bilateral_blur_2d.comp");
}

bool BilateralBlurCompute::TryDispatch(VideoCore::ImageId src_id, u32 src_mip, u32 src_layer,
                                       VideoCore::ImageId dst_id, u32 dst_mip, u32 dst_layer,
                                       const u8* cb_ptr, u32 cb_size_bytes, bool is_nearest) {
    if (!Enabled() || !src_id || !dst_id || src_id == dst_id || !cb_ptr || cb_size_bytes < 208u) {
        if (Enabled() && !logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: reject basic (src=%u dst=%u cb=%p sz=%u)\n",
                        src_id.index, dst_id.index, static_cast<const void*>(cb_ptr),
                        cb_size_bytes);
        }
        return false;
    }

    auto& src_img = texture_cache.GetImage(src_id);
    auto& dst_img = texture_cache.GetImage(dst_id);
    src_mip = std::min(src_mip, src_img.info.resources.levels - 1u);
    src_layer = std::min(src_layer, src_img.info.resources.layers - 1u);
    dst_mip = std::min(dst_mip, dst_img.info.resources.levels - 1u);
    dst_layer = std::min(dst_layer, dst_img.info.resources.layers - 1u);
    if (src_img.GetImage() == dst_img.GetImage() && src_mip == dst_mip && src_layer == dst_layer) {
        if (!logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: reject same vk::Image (src=%u dst=%u)\n",
                        src_id.index, dst_id.index);
        }
        return false;
    }
    if (src_img.info.pixel_format != vk::Format::eR8G8B8A8Unorm ||
        dst_img.info.pixel_format != vk::Format::eR8G8B8A8Unorm ||
        src_img.info.num_samples != 1u || dst_img.info.num_samples != 1u ||
        src_img.info.props.is_depth || dst_img.info.props.is_depth ||
        src_img.info.props.is_block || dst_img.info.props.is_block) {
        if (!logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: reject fmt/props (src_fmt=%d dst_fmt=%d ms=%u/%u)\n",
                        int(src_img.info.pixel_format), int(dst_img.info.pixel_format),
                        src_img.info.num_samples, dst_img.info.num_samples);
        }
        return false;
    }

    const u32 w = std::max(1u, src_img.info.size.width >> src_mip);
    const u32 h = std::max(1u, src_img.info.size.height >> src_mip);
    const u32 dw = std::max(1u, dst_img.info.size.width >> dst_mip);
    const u32 dh = std::max(1u, dst_img.info.size.height >> dst_mip);
    if (w != dw || h != dh || w < 16u || h < 16u) {
        if (!logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: reject dims (%ux%u vs %ux%u)\n", w, h, dw, dh);
        }
        return false;
    }

    // Parse 52 floats (208 B) from `cbBlurParam`:
    //   [0] = z_a
    //   [4 + k*4 + 0..2] = (off_uv.x, off_uv.y, kw) for k = 0..11
    f32 cb[52];
    std::memcpy(cb, cb_ptr, sizeof(cb));
    if (!std::isfinite(cb[0])) {
        return false;
    }

    s32 dx[12]{};
    s32 dy[12]{};
    f32 kws[12]{};
    u32 max_abs_x = 0u;
    u32 max_abs_y = 0u;
    bool all_linear_exact = true;

    const f32 fw = static_cast<f32>(w);
    const f32 fh = static_cast<f32>(h);
    for (u32 k = 0u; k < 12u; ++k) {
        const f32 ox = cb[4u + k * 4u + 0u];
        const f32 oy = cb[4u + k * 4u + 1u];
        const f32 kw = cb[4u + k * 4u + 2u];
        if (!std::isfinite(ox) || !std::isfinite(oy) || !std::isfinite(kw)) {
            return false;
        }
        kws[k] = kw;

        const f32 tx = ox * fw;
        const f32 ty = oy * fh;
        // For VK_FILTER_NEAREST, `floor((gx + 0.5) + tx) == gx + floor(0.5 + tx)` identically
        // for EVERY integer pixel `gx`, even when `tx` is fractional (e.g. 1080p -> 1440p scale!).
        const s32 ix = static_cast<s32>(std::floor(0.5f + tx));
        const s32 iy = static_cast<s32>(std::floor(0.5f + ty));
        if (std::fabs(tx - std::round(tx)) > 0.04f || std::fabs(ty - std::round(ty)) > 0.04f) {
            all_linear_exact = false;
        }
        dx[k] = ix;
        dy[k] = iy;
        max_abs_x = std::max(max_abs_x, static_cast<u32>(std::abs(ix)));
        max_abs_y = std::max(max_abs_y, static_cast<u32>(std::abs(iy)));
    }

    if (!is_nearest && !all_linear_exact) {
        if (!logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: linear sampler with fractional taps on %ux%u "
                        "(tap0=%.2f,%.2f tap1=%.2f,%.2f), falling back to fragment\n",
                        w, h, cb[4] * fw, cb[5] * fh, cb[8] * fw, cb[9] * fh);
        }
        return false;
    }

    enum class Mode : u8 { None, H, V, TwoD };
    Mode mode = Mode::None;
    if (max_abs_y == 0u && max_abs_x > 0u && max_abs_x <= kMaxHalo1D) {
        mode = Mode::H;
    } else if (max_abs_x == 0u && max_abs_y > 0u && max_abs_y <= kMaxHalo1D) {
        mode = Mode::V;
    } else if (max_abs_x <= kMaxHalo2D && max_abs_y <= kMaxHalo2D &&
               (max_abs_x > 0u || max_abs_y > 0u)) {
        mode = Mode::TwoD;
    }

    if (mode == Mode::None) {
        if (!logged_fallback) {
            logged_fallback = true;
            std::printf("BilateralBlurCompute: unsupported tap radius on %ux%u "
                        "(max=%u,%u nearest=%d tap0=%.2f,%.2f tap1=%.2f,%.2f), falling back\n",
                        w, h, max_abs_x, max_abs_y, is_nearest ? 1 : 0, cb[4] * fw, cb[5] * fh,
                        cb[8] * fw, cb[9] * fh);
        }
        return false;
    }

    EnsurePipelines();
    scheduler.EndRendering();

    const VideoCore::SubresourceRange src_range{
        .base = {src_mip, src_layer},
        .extent = {1u, 1u},
    };
    const VideoCore::SubresourceRange dst_range{
        .base = {dst_mip, dst_layer},
        .extent = {1u, 1u},
    };
    runtime.Transit(&src_img, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                    src_range);
    runtime.Transit(&dst_img, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                    dst_range);
    runtime.FlushBarriers();

    VideoCore::ImageViewInfo src_view_info{};
    src_view_info.type = Gcn::ImageType::Color2D;
    src_view_info.format = vk::Format::eR8G8B8A8Unorm;
    src_view_info.range = src_range;
    src_view_info.is_storage = false;
    const vk::ImageView src_view = *src_img.FindView(src_view_info).image_view;

    VideoCore::ImageViewInfo dst_view_info{};
    dst_view_info.type = Gcn::ImageType::Color2D;
    dst_view_info.format = vk::Format::eR8G8B8A8Unorm;
    dst_view_info.range = dst_range;
    dst_view_info.is_storage = true;
    const vk::ImageView dst_view = *dst_img.FindView(dst_view_info).image_view;

    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Shadow_VsmAndPcf);
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        if (mode == Mode::H) {
            prof->Mark(0x9A71FE41ULL, [] {
                return std::string{"BilateralBlurCompute: H (LDS) [VULKAN-NATIVE]"};
            });
        } else if (mode == Mode::V) {
            prof->Mark(0x9A71FE42ULL, [] {
                return std::string{"BilateralBlurCompute: V (LDS) [VULKAN-NATIVE]"};
            });
        } else {
            prof->Mark(0x9A71FE43ULL, [] {
                return std::string{"BilateralBlurCompute: 2D (LDS) [VULKAN-NATIVE]"};
            });
        }
    }

    const vk::PipelineLayout layout = *pipe_layout;

    if (mode == Mode::H || mode == Mode::V) {
        const bool is_h = (mode == Mode::H);
        PushConstants1D pc{};
        pc.width = w;
        pc.height = h;
        pc.z_a = cb[0];
        pc.halo = std::clamp(is_h ? max_abs_x : max_abs_y, 1u, kMaxHalo1D);
        std::memcpy(pc.delta, is_h ? dx : dy, sizeof(pc.delta));
        std::memcpy(pc.kw, kws, sizeof(pc.kw));

        const vk::Pipeline pipe = is_h ? *pipeline_h : *pipeline_v;
        const u32 gx = is_h ? ((w + 127u) / 128u) : ((w + 15u) / 16u);
        const u32 gy = is_h ? h : ((h + 31u) / 32u);

        scheduler.Record([pipe, layout, src_view, dst_view, pc, gx, gy](vk::CommandBuffer cmdbuf) {
            const vk::DescriptorImageInfo src_info{
                .imageView = src_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dst_info{
                .imageView = dst_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &src_info,
                },
                {
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &dst_info,
                },
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstants1D), &pc);
            cmdbuf.dispatch(gx, gy, 1u);
        });

        if (is_h && !logged_h) {
            logged_h = true;
            std::printf("BilateralBlurCompute: Pack #3 LDS H active (%ux%u, halo=%u, nearest=%d, "
                        "BB_BILATERAL_COMPUTE=0 to disable)\n",
                        w, h, pc.halo, is_nearest ? 1 : 0);
        } else if (!is_h && !logged_v) {
            logged_v = true;
            std::printf("BilateralBlurCompute: Pack #3 LDS V active (%ux%u, halo=%u, nearest=%d)\n",
                        w, h, pc.halo, is_nearest ? 1 : 0);
        }
    } else {
        PushConstants2D pc{};
        pc.width = w;
        pc.height = h;
        pc.z_a = cb[0];
        pc.halo = std::clamp(std::max(max_abs_x, max_abs_y), 1u, kMaxHalo2D);
        for (u32 i = 0u; i < 6u; ++i) {
            const u32 p0 = (static_cast<u32>(dx[i * 2u]) & 0xFFu) |
                           ((static_cast<u32>(dy[i * 2u]) & 0xFFu) << 8u);
            const u32 p1 = (static_cast<u32>(dx[i * 2u + 1u]) & 0xFFu) |
                           ((static_cast<u32>(dy[i * 2u + 1u]) & 0xFFu) << 8u);
            pc.dxy[i] = p0 | (p1 << 16u);
        }
        std::memcpy(pc.kw, kws, sizeof(pc.kw));

        const vk::Pipeline pipe = *pipeline_2d;
        const u32 gx = (w + 31u) / 32u;
        const u32 gy = (h + 15u) / 16u;

        scheduler.Record([pipe, layout, src_view, dst_view, pc, gx, gy](vk::CommandBuffer cmdbuf) {
            const vk::DescriptorImageInfo src_info{
                .imageView = src_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dst_info{
                .imageView = dst_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &src_info,
                },
                {
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage,
                    .pImageInfo = &dst_info,
                },
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstants2D), &pc);
            cmdbuf.dispatch(gx, gy, 1u);
        });

        static u32 logged_halo_mask = 0u;
        const u32 hbit = 1u << std::min(pc.halo, 31u);
        if ((logged_halo_mask & hbit) == 0u) {
            logged_halo_mask |= hbit;
            std::printf("BilateralBlurCompute: Pack #3 v2 LDS 2D active (%ux%u, halo=%u, nearest=%d, "
                        "taps=[(%d,%d),(%d,%d),(%d,%d)...], BB_BILATERAL_COMPUTE=0 to disable)\n",
                        w, h, pc.halo, is_nearest ? 1 : 0, dx[0], dy[0], dx[1], dy[1], dx[2],
                        dy[2]);
        }
    }

    dst_img.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_img.gpu_write_tick = scheduler.CurrentTick();
    return true;
}

} // namespace Vulkan
