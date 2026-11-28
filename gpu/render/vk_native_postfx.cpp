// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2: Native Vulkan 1.3 Post-Process pipeline — Step 1:
// Unified Tonemap + Original 16x16x16 3D LUT + True Black & Vibrance +
// Resolution-Independent Chromatic Aberration + Optical Vignette + True-dt Auto-Exposure.

#include "render/vk_native_postfx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "bbgpu/shaders/fullscreen_vert.h"
#include "bbgpu/shaders/native_postfx_auto_exp_comp.h"
#include "bbgpu/shaders/native_postfx_bloom_down_comp.h"
#include "bbgpu/shaders/native_postfx_bloom_up_comp.h"
#include "bbgpu/shaders/native_postfx_dof_bokeh_comp.h"
#include "bbgpu/shaders/native_postfx_dof_coc_comp.h"
#include "bbgpu/shaders/native_postfx_dof_comp_comp.h"
#include "bbgpu/shaders/native_postfx_luma_hist_comp.h"
#include "bbgpu/shaders/native_postfx_mb_blur_comp.h"
#include "bbgpu/shaders/native_postfx_mb_vel_comp.h"
#include "bbgpu/shaders/native_postfx_ssgi_prefilter_comp.h"
#include "bbgpu/shaders/native_postfx_ssgi_trace_comp.h"
#include "bbgpu/shaders/native_postfx_ssgi_upsample_frag.h"
#include "bbgpu/shaders/native_postfx_tonemap_frag.h"
#include "buffers/buffer_cache.h"
#include "gcn/regs_targets.h"
#include "gcn/sharps.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "shader/info.h"
#include "shim/bbport_frame_rate.h"
#include "shim/bbport_settings.h"
#include "shim/bbport_toggles.h"
#include "textures/direct_textures.h"
#include "textures/texture_cache.h"
#include "upscale/vk_temporal_upscaler.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

namespace {

struct HistPush {
    u32 width;
    u32 height;
    float min_log_lum;
    float inv_log_lum_range;
    float meter_min_x;
    float meter_min_y;
    float meter_max_x;
    float meter_max_y;
    float scene_lum_scale;
    u32 stride;
};
static_assert(sizeof(HistPush) == 40);

struct AutoExpPush {
    float min_log_lum;
    float log_lum_range;
    float middle_gray;
    float min_exposure;
    float max_exposure;
    float manual_exposure;
    float dt_sec;
    float bright_adapt_rate;
    float dark_adapt_rate;
    float bright_speed_limit;
    float dark_speed_limit;
    u32 flags;
};
static_assert(sizeof(AutoExpPush) == 48);

struct BloomDownPush {
    u32 dst_w;
    u32 dst_h;
    float inv_src_w;
    float inv_src_h;
    float manual_exposure;
    float inv_scene_lum_scale;
    float threshold;
    float knee;
    float luma_gamma;
    float remap_factor;
    u32 flags;
    float min_log_lum;
    float inv_log_lum_range;
    float meter_min_x;
    float meter_min_y;
    float meter_max_x;
    float meter_max_y;
};
static_assert(sizeof(BloomDownPush) == 68);

struct BloomUpPush {
    u32 dst_w;
    u32 dst_h;
    float radius_x;
    float radius_y;
    float up_weight;
    float cur_weight;
};
static_assert(sizeof(BloomUpPush) == 24);

struct MbVelUbo {
    float clip_to_prev[16];
    float params0[4];  // x=cam_scale, y=obj_scale, z=cam_max_uv, w=obj_max_uv
    float params1[4];  // x=fps_q4, y=inv_enc_scale, z=self_deadzone_ndc, w=0
    u32 dims_flags[4]; // x=dst_w, y=dst_h, z=flags, w=reserved
};
static_assert(sizeof(MbVelUbo) == 112);

struct MbBlurPush {
    u32 width;
    u32 height;
    float inv_vel_w;
    float inv_vel_h;
    u32 frame_index;
    u32 flags;
};
static_assert(sizeof(MbBlurPush) == 24);

struct DofUbo {
    float clip_phys[4];    // x=nearZ, y=farZ, z=focusDist, w=physCocFactor
    float near_params[4];  // x=nearStart, y=nearEnd, z=nearCocUv, w=negCocScale
    float far_params[4];   // x=farStart,  y=farEnd,  z=farCocUv,  w=posCocScale
    float misc_params[4];  // x=farCutoff, y=maxCocUv, z=aspect, w=dofScale
    u32 dims_flags[4];     // x=half_w, y=half_h, z=full_w, w=full_h
    u32 mode_flags[4];     // x=flags (bit0=custom, bit1=nearCust, bit2=farCust), y=frame_index, z=0, w=0
    float depth_jitter[4]; // xy=render-pixel jitter of depth_tex, zw=0
};
static_assert(sizeof(DofUbo) == 112);

struct DofBokehPush {
    u32 width;
    u32 height;
    float inv_aspect;
    float max_coc_uv;
    u32 frame_index;
    u32 reserved;
};
static_assert(sizeof(DofBokehPush) == 24);

struct SsgiPrefilterPush {
    u32 dst_w;
    u32 dst_h;
    u32 src_w;
    u32 src_h;
    u32 is_mip1;
};
static_assert(sizeof(SsgiPrefilterPush) == 20);

struct SsgiTracePush {
    u32 dst_w;
    u32 dst_h;
    u32 d1_offset;
    float aspect;
};
static_assert(sizeof(SsgiTracePush) == 16);

struct SsgiUpsamplePush {
    float inv_half_w;
    float inv_half_h;
};
static_assert(sizeof(SsgiUpsamplePush) == 8);

struct TonemapUbo {
    float tone_params[4];  // x=exposure, y=inv_gamma, z=tone_factor, w=filmic_contrast
    float scene_vig[4];    // x=inv_scene_lum_scale, y=vig_power, z=vig_tan_half_fov, w=aspect
    float color_corr[4];   // x=hue_rad, y=saturation, z=value_scale, w=contrast
    float ca_params[4];    // xy=lat_disp, zw=uni_disp
    float bloom_params[4]; // x=bloom_intensity, yzw=reserved
    u32 flags_misc[4];     // x=flags, y=tone_type, z=ca_samples, w=frame_index
};
static_assert(sizeof(TonemapUbo) == 96);

constexpr u32 kShaderFlagLut3d = 1u << 0;
constexpr u32 kShaderFlagAutoExp = 1u << 1;
constexpr u32 kShaderFlagCa = 1u << 2;
constexpr u32 kShaderFlagTrueBlack = 1u << 3;
constexpr u32 kShaderFlagBgraExport = 1u << 4;
constexpr u32 kShaderFlagBloom = 1u << 5;

constexpr float kMinLogLum = -10.0f;
constexpr float kMaxLogLum = 6.0f;
constexpr float kLogLumRange = kMaxLogLum - kMinLogLum;

constexpr VkDeviceSize kHistBufBytes = 256u * sizeof(u32);
constexpr VkDeviceSize kExpStateBufBytes = 64u;

// Exact 21-entry Yebis2 Sensitometric S-Curve contrast lookup table extracted from eboot.bin
// (file offset 0x2be3fa0, sampled at 0xc31e07..0xc31e94 for c832 / c833 in ps_5e67d866.frag).
constexpr std::array<float, 21> kYebisSensitometricTable = {
    0.000000f, 0.048482f, 0.090533f, 0.127458f, 0.160230f, 0.189600f, 0.217000f,
    0.246641f, 0.279107f, 0.314625f, 0.353440f, 0.395867f, 0.442200f, 0.492722f,
    0.547864f, 0.608000f, 0.673580f, 0.745071f, 0.823000f, 0.907950f, 1.000000f,
};

inline void ComputeYebisSensitometricParams(float filmic_contrast, float& out_c832,
                                            float& out_c833) noexcept {
    const float c = std::clamp(filmic_contrast, 0.0f, 1.0f);
    const float t = (c > 0.0f) ? std::pow(c, 1.0f / 3.0f) : 0.0f;
    out_c832 = 1.0f + 2.0891273f * t;
    const float scaled = std::clamp(t, 0.0f, 1.0f) / 0.05f;
    const int i0 = std::clamp(static_cast<int>(std::floor(scaled)), 0, 20);
    const int i1 = std::min(i0 + 1, 20);
    const float frac = scaled - static_cast<float>(i0);
    out_c833 = kYebisSensitometricTable[i0] +
               (kYebisSensitometricTable[i1] - kYebisSensitometricTable[i0]) * frac;
}

} // namespace

NativePostFx::NativePostFx(const Instance& instance_, Scheduler& scheduler_,
                           VideoCore::TextureCache& texture_cache_,
                           VideoCore::BufferCache& buffer_cache_, Runtime& runtime_)
    : instance(instance_), scheduler(scheduler_), texture_cache(texture_cache_),
      buffer_cache(buffer_cache_), runtime(runtime_) {}

NativePostFx::~NativePostFx() {
    if (hist_buf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(instance.GetAllocator(), hist_buf, hist_alloc);
        hist_buf = VK_NULL_HANDLE;
        hist_alloc = VK_NULL_HANDLE;
    }
    if (exp_state_buf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(instance.GetAllocator(), exp_state_buf, exp_state_alloc);
        exp_state_buf = VK_NULL_HANDLE;
        exp_state_alloc = VK_NULL_HANDLE;
    }
}

void NativePostFx::ReleaseDisabledTargets() {
    const auto& settings = BbSettings::Get();
    const bool native_on = settings.native_postfx.load(std::memory_order_relaxed);
    const bool bloom_enabled = native_on && settings.effects[7].load(std::memory_order_relaxed);
    const bool haze_enabled =
        bloom_enabled && settings.fire_heat_haze.load(std::memory_order_relaxed);
    const bool ssgi_enabled = settings.ssgi_enabled.load(std::memory_order_relaxed);
    const bool mb_debug = settings.native_mb_debug_vis.load(std::memory_order_relaxed);
    const bool mb_enabled =
        native_on &&
        ((settings.effects[2].load(std::memory_order_relaxed) &&
          (settings.motion_blur_camera.load(std::memory_order_relaxed) > 0 ||
           settings.motion_blur_object.load(std::memory_order_relaxed) > 0)) ||
         mb_debug);
    const bool dof_enabled =
        native_on && settings.effects[1].load(std::memory_order_relaxed) && !mb_debug;

    const bool drop_bloom = !bloom_enabled && bool(bloom_pyramid);
    const bool drop_haze = !haze_enabled && bool(opaque_hdr_half);
    const bool drop_ssgi = !ssgi_enabled &&
                           (bool(ssgi_rad_mip0) || bool(ssgi_rad_mip1) || bool(ssgi_raw_img) ||
                            bool(ssgi_denoise_img));
    const bool drop_mb = !mb_enabled && (bool(mb_vel_img) || bool(mb_tile_max_img) || bool(mb_color_img));
    const bool drop_dof = !dof_enabled && (bool(dof_coc_img) || bool(dof_bokeh_img) || bool(dof_color_img));
    if (!drop_bloom && !drop_haze && !drop_ssgi && !drop_mb && !drop_dof) {
        return;
    }

    scheduler.Finish();
    if (drop_bloom) {
        for (auto& view : bloom_mip_views) {
            view.reset();
        }
        bloom_pyramid = VideoCore::HostImage{};
        bloom_mip_extents = {};
        bloom_base_w = bloom_base_h = 0u;
    }
    if (drop_haze) {
        opaque_hdr_view.reset();
        opaque_hdr_half = VideoCore::HostImage{};
        opaque_hdr_extent = vk::Extent2D{0u, 0u};
        opaque_hdr_src_vk = VK_NULL_HANDLE;
        opaque_hdr_valid = false;
    }
    if (drop_ssgi) {
        ssgi_rad_mip0_view.reset();
        ssgi_rad_mip1_view.reset();
        ssgi_raw_view.reset();
        ssgi_denoise_view.reset();
        ssgi_rad_mip0 = VideoCore::HostImage{};
        ssgi_rad_mip1 = VideoCore::HostImage{};
        ssgi_raw_img = VideoCore::HostImage{};
        ssgi_denoise_img = VideoCore::HostImage{};
        ssgi_half_extent = vk::Extent2D{0u, 0u};
        ssgi_quarter_extent = vk::Extent2D{0u, 0u};
    }
    if (drop_mb) {
        mb_vel_view.reset();
        mb_tile_max_view.reset();
        mb_color_view.reset();
        mb_vel_img = VideoCore::HostImage{};
        mb_tile_max_img = VideoCore::HostImage{};
        mb_color_img = VideoCore::HostImage{};
        mb_vel_extent = vk::Extent2D{0u, 0u};
        mb_tile_extent = vk::Extent2D{0u, 0u};
        mb_color_extent = vk::Extent2D{0u, 0u};
    }
    if (drop_dof) {
        dof_coc_view.reset();
        dof_bokeh_view.reset();
        dof_color_view.reset();
        dof_coc_img = VideoCore::HostImage{};
        dof_bokeh_img = VideoCore::HostImage{};
        dof_color_img = VideoCore::HostImage{};
        dof_half_extent = vk::Extent2D{0u, 0u};
        dof_full_extent = vk::Extent2D{0u, 0u};
    }
}

bool NativePostFx::EnsureInit() {
    if (initialized) {
        return true;
    }
    if (init_failed) {
        return false;
    }

    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();

    // 1. Linear clamp-to-edge sampler shared across histogram, scene, and 3D LUT.
    linear_sampler = Check<"create native postfx sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxLod = 0.0f,
    }));

    // 2. 1x1x1 dummy 3D image for binding 1 when a scene has no active 3D LUT.
    dummy_lut3d = VideoCore::HostImage(device, allocator);
    dummy_lut3d.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e3D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    dummy_lut3d_view = Check<"create dummy 3d lut view">(device.createImageViewUnique({
        .image = vk::Image(dummy_lut3d),
        .viewType = vk::ImageViewType::e3D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));

    // 2b. 1x1 dummy 2D black image for binding 5 when Bloom is disabled.
    dummy_black_2d = VideoCore::HostImage(device, allocator);
    dummy_black_2d.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    dummy_black_2d_view = Check<"create dummy black 2d view">(device.createImageViewUnique({
        .image = vk::Image(dummy_black_2d),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));

    // 3. Device-local SSBOs for 256-bin luminance histogram and persistent adapted exposure.
    auto make_ssbo = [&](VkDeviceSize bytes, VkBuffer& out_buf, VmaAllocation& out_alloc) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        return vmaCreateBuffer(allocator, &bci, &aci, &out_buf, &out_alloc, nullptr) == VK_SUCCESS;
    };
    if (!make_ssbo(kHistBufBytes, hist_buf, hist_alloc) ||
        !make_ssbo(kExpStateBufBytes, exp_state_buf, exp_state_alloc)) {
        init_failed = true;
        return false;
    }

    // 4. Compute pipeline 1: 256-bin log2 luminance histogram.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        hist_ds_layout = Check<"create postfx hist ds layout">(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(HistPush),
        };
        const vk::DescriptorSetLayout dsl = *hist_ds_layout;
        hist_layout = Check<"create postfx hist layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_LUMA_HIST_COMP, device);
        hist_pipeline = Check<"create postfx hist pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *hist_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5. Compute pipeline 2: true-dt Auto-Exposure reduction & EV adaptation.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        auto_exp_ds_layout =
            Check<"create postfx auto_exp ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(AutoExpPush),
        };
        const vk::DescriptorSetLayout dsl = *auto_exp_ds_layout;
        auto_exp_layout = Check<"create postfx auto_exp layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_AUTO_EXP_COMP, device);
        auto_exp_pipeline = Check<"create postfx auto_exp pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *auto_exp_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5b. Compute pipeline 3: FP16 13-tap Jimenez Bloom Downsample (prefilter + Karis + Fire Heat Haze + Fused AutoExp Hist).
    {
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
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 6,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        bloom_down_ds_layout =
            Check<"create postfx bloom_down ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(BloomDownPush),
        };
        const vk::DescriptorSetLayout dsl = *bloom_down_ds_layout;
        bloom_down_layout = Check<"create postfx bloom_down layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_BLOOM_DOWN_COMP, device);
        bloom_down_pipeline = Check<"create postfx bloom_down pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *bloom_down_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5c. Compute pipeline 4: FP16 9-tap Tent Bloom Upsample & Progressive Combine.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
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
        }};
        bloom_up_ds_layout =
            Check<"create postfx bloom_up ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(BloomUpPush),
        };
        const vk::DescriptorSetLayout dsl = *bloom_up_ds_layout;
        bloom_up_layout = Check<"create postfx bloom_up layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_BLOOM_UP_COMP, device);
        bloom_up_pipeline = Check<"create postfx bloom_up pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *bloom_up_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5d. Compute pipeline 5: Half-res Combined Camera + Object Velocity Map + Tile-Max Reduction.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 6> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        mb_vel_ds_layout =
            Check<"create postfx mb_vel ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::DescriptorSetLayout dsl = *mb_vel_ds_layout;
        mb_vel_layout = Check<"create postfx mb_vel layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_MB_VEL_COMP, device);
        mb_vel_pipeline = Check<"create postfx mb_vel pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *mb_vel_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5e. Compute pipeline 6: Full-res 16-tap Bilateral IGN Motion Blur (with 3x3 Tile Early-Out).
    {
        const std::array<vk::DescriptorSetLayoutBinding, 5> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        mb_blur_ds_layout =
            Check<"create postfx mb_blur ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(MbBlurPush),
        };
        const vk::DescriptorSetLayout dsl = *mb_blur_ds_layout;
        mb_blur_layout = Check<"create postfx mb_blur layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_MB_BLUR_COMP, device);
        mb_blur_pipeline = Check<"create postfx mb_blur pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *mb_blur_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5c. Compute pipeline: Step 4 DoF Half-Res CoC + Karis Pre-Filter.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 5> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        dof_coc_ds_layout =
            Check<"create postfx dof_coc ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::DescriptorSetLayout dsl = *dof_coc_ds_layout;
        dof_coc_layout = Check<"create postfx dof_coc layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_DOF_COC_COMP, device);
        dof_coc_pipeline = Check<"create postfx dof_coc pipeline">(device.createComputePipelineUnique(
            {}, {
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = cs,
                              .pName = "main"},
                    .layout = *dof_coc_layout,
                }));
        device.destroyShaderModule(cs);
    }

    // 5d. Compute pipeline: Step 4 DoF Half-Res 24-Tap Vogel Disk Bokeh Gather.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
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
        }};
        dof_bokeh_ds_layout =
            Check<"create postfx dof_bokeh ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(DofBokehPush),
        };
        const vk::DescriptorSetLayout dsl = *dof_bokeh_ds_layout;
        dof_bokeh_layout =
            Check<"create postfx dof_bokeh layout">(device.createPipelineLayoutUnique({
                .setLayoutCount = 1,
                .pSetLayouts = &dsl,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pc_range,
            }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_DOF_BOKEH_COMP, device);
        dof_bokeh_pipeline =
            Check<"create postfx dof_bokeh pipeline">(device.createComputePipelineUnique(
                {}, {
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = cs,
                                  .pName = "main"},
                        .layout = *dof_bokeh_layout,
                    }));
        device.destroyShaderModule(cs);
    }

    // 5e. Compute pipeline: Step 4 DoF Full-Res Bilateral Composite.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 7> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 6,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        dof_comp_ds_layout =
            Check<"create postfx dof_comp ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::DescriptorSetLayout dsl = *dof_comp_ds_layout;
        dof_comp_layout = Check<"create postfx dof_comp layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
        }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_DOF_COMP_COMP, device);
        dof_comp_pipeline =
            Check<"create postfx dof_comp pipeline">(device.createComputePipelineUnique(
                {}, {
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = cs,
                                  .pName = "main"},
                        .layout = *dof_comp_layout,
                    }));
        device.destroyShaderModule(cs);
    }

    // 6. Graphics pipeline layout & shader modules for Tonemap + Bloom + 3D LUT + CA + Vignette + TrueBlack.
    {
        const std::array<vk::DescriptorSetLayoutBinding, 6> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
        }};
        tonemap_ds_layout =
            Check<"create postfx tonemap ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::DescriptorSetLayout dsl = *tonemap_ds_layout;
        tonemap_layout = Check<"create postfx tonemap layout">(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &dsl,
        }));
        tonemap_vs = vk::UniqueShaderModule(CompileSPV(FULLSCREEN_VERT, device), device);
        tonemap_fs = vk::UniqueShaderModule(CompileSPV(NATIVE_POSTFX_TONEMAP_FRAG, device), device);
    }

    // 7. Half-Res L2-Resident SSGI pipelines (Prefilter + Trace + Bilateral Upsample).
    {
        const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        ssgi_prefilter_ds_layout =
            Check<"create postfx ssgi_prefilter ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(SsgiPrefilterPush),
        };
        const vk::DescriptorSetLayout dsl = *ssgi_prefilter_ds_layout;
        ssgi_prefilter_layout =
            Check<"create postfx ssgi_prefilter layout">(device.createPipelineLayoutUnique({
                .setLayoutCount = 1,
                .pSetLayouts = &dsl,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pc_range,
            }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_SSGI_PREFILTER_COMP, device);
        ssgi_prefilter_pipeline =
            Check<"create postfx ssgi_prefilter pipeline">(device.createComputePipelineUnique(
                {}, {
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = cs,
                                  .pName = "main"},
                        .layout = *ssgi_prefilter_layout,
                    }));
        device.destroyShaderModule(cs);
    }
    {
        const std::array<vk::DescriptorSetLayoutBinding, 6> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        ssgi_trace_ds_layout =
            Check<"create postfx ssgi_trace ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(SsgiTracePush),
        };
        const vk::DescriptorSetLayout dsl = *ssgi_trace_ds_layout;
        ssgi_trace_layout =
            Check<"create postfx ssgi_trace layout">(device.createPipelineLayoutUnique({
                .setLayoutCount = 1,
                .pSetLayouts = &dsl,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pc_range,
            }));
        const vk::ShaderModule cs = CompileSPV(NATIVE_POSTFX_SSGI_TRACE_COMP, device);
        ssgi_trace_pipeline =
            Check<"create postfx ssgi_trace pipeline">(device.createComputePipelineUnique(
                {}, {
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = cs,
                                  .pName = "main"},
                        .layout = *ssgi_trace_layout,
                    }));
        device.destroyShaderModule(cs);
    }
    {
        const std::array<vk::DescriptorSetLayoutBinding, 6> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 3,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 4,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
            {.binding = 5,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eFragment},
        }};
        ssgi_upsample_ds_layout =
            Check<"create postfx ssgi_upsample ds layout">(device.createDescriptorSetLayoutUnique({
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = u32(bindings.size()),
                .pBindings = bindings.data(),
            }));
        const vk::PushConstantRange pc_range{
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(SsgiUpsamplePush),
        };
        const vk::DescriptorSetLayout dsl = *ssgi_upsample_ds_layout;
        ssgi_upsample_layout =
            Check<"create postfx ssgi_upsample layout">(device.createPipelineLayoutUnique({
                .setLayoutCount = 1,
                .pSetLayouts = &dsl,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pc_range,
            }));
        ssgi_upsample_fs =
            vk::UniqueShaderModule(CompileSPV(NATIVE_POSTFX_SSGI_UPSAMPLE_FRAG, device), device);
    }

    initialized = true;
    return true;
}

bool NativePostFx::EnsureBloomPyramid(u32 hdr_w, u32 hdr_h) {
    // Quarter-res base at >=1080p (480x270 @ 1080p matches PS4 Yebis2, 640x360 @ 1440p, 960x540 @ 4K):
    // 13-tap bilinear Jimenez downsample cleanly covers all 16 source pixels per 4x4 block with zero
    // aliasing while keeping mip 0 L2-resident (~1.8 MB @ 1440p) and 4x lighter on mid-range GPUs.
    const u32 div = (hdr_w > 1280u || hdr_h > 720u) ? 4u : 2u;
    const u32 base_w = std::max(hdr_w / div, 16u);
    const u32 base_h = std::max(hdr_h / div, 16u);
    if (bloom_pyramid && bloom_base_w == base_w && bloom_base_h == base_h) {
        return true;
    }

    scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();
    for (auto& v : bloom_mip_views) {
        v.reset();
    }
    bloom_pyramid = VideoCore::HostImage(device, allocator);
    bloom_pyramid.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {base_w, base_h, 1},
        .mipLevels = kBloomMips,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    u32 mw = base_w;
    u32 mh = base_h;
    for (u32 i = 0; i < kBloomMips; ++i) {
        bloom_mip_extents[i] = vk::Extent2D{mw, mh};
        bloom_mip_views[i] = Check<"create bloom mip view">(device.createImageViewUnique({
            .image = vk::Image(bloom_pyramid),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, i, 1, 0, 1},
        }));
        mw = std::max(mw / 2u, 1u);
        mh = std::max(mh / 2u, 1u);
    }
    bloom_base_w = base_w;
    bloom_base_h = base_h;
    return true;
}

bool NativePostFx::EnsureOpaqueHdrTarget(u32 hdr_w, u32 hdr_h) {
    const u32 half_w = std::max(hdr_w / 2u, 16u);
    const u32 half_h = std::max(hdr_h / 2u, 16u);
    if (opaque_hdr_half && opaque_hdr_view && opaque_hdr_extent.width == half_w &&
        opaque_hdr_extent.height == half_h) {
        return true;
    }

    scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();
    opaque_hdr_view.reset();
    opaque_hdr_half = VideoCore::HostImage(device, allocator);
    opaque_hdr_half.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {half_w, half_h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    opaque_hdr_view = Check<"create opaque hdr half view">(device.createImageViewUnique({
        .image = vk::Image(opaque_hdr_half),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    opaque_hdr_extent = vk::Extent2D{half_w, half_h};
    return true;
}

void NativePostFx::CaptureOpaqueHdr(VideoCore::Image& hdr_image, vk::Image proxy_img,
                                    u32 proxy_w, u32 proxy_h) {
    if (!EnsureInit()) {
        return;
    }
    const bool use_proxy = (proxy_img != VK_NULL_HANDLE && proxy_w >= 64u && proxy_h >= 64u);
    const u32 w = use_proxy ? proxy_w : hdr_image.info.size.width;
    const u32 h = use_proxy ? proxy_h : hdr_image.info.size.height;
    const vk::Image src_img = use_proxy ? proxy_img : hdr_image.GetImage();
    if (w < 64u || h < 64u || !src_img) {
        return;
    }
    if (!EnsureOpaqueHdrTarget(w, h)) {
        return;
    }

    scheduler.EndRendering();
    if (!use_proxy) {
        runtime.Transit(&hdr_image, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
    }

    const vk::Image dst_img = vk::Image(opaque_hdr_half);
    const vk::Extent2D dst_ext = opaque_hdr_extent;
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        prof->Mark(0x9A71FE01ULL, [w, h, dst_ext] {
            return fmt::format("NativePostFX: CaptureOpaqueHdr {}x{} -> {}x{} [VULKAN-NATIVE]",
                               w, h, dst_ext.width, dst_ext.height);
        });
    }
    scheduler.Record([src_img, dst_img, w, h, dst_ext, use_proxy](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange one_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        if (use_proxy) {
            const vk::ImageMemoryBarrier2 src_to_transfer{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .image = src_img,
                .subresourceRange = one_range,
            };
            const vk::ImageMemoryBarrier2 to_dst{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = dst_img,
                .subresourceRange = one_range,
            };
            const std::array<vk::ImageMemoryBarrier2, 2> pre{{src_to_transfer, to_dst}};
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(pre.size()), .pImageMemoryBarriers = pre.data()});
        } else {
            const vk::ImageMemoryBarrier2 to_dst{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = dst_img,
                .subresourceRange = one_range,
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst});
        }

        const vk::ImageBlit region{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .srcOffsets = std::array<vk::Offset3D, 2>{{{0, 0, 0}, {s32(w), s32(h), 1}}},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstOffsets =
                std::array<vk::Offset3D, 2>{{{0, 0, 0}, {s32(dst_ext.width), s32(dst_ext.height), 1}}},
        };
        cmdbuf.blitImage(src_img, vk::ImageLayout::eTransferSrcOptimal, dst_img,
                         vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);

        const vk::ImageMemoryBarrier2 to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = dst_img,
            .subresourceRange = one_range,
        };
        if (use_proxy) {
            // Restore reduced proxy back to ColorAttachmentOptimal so SceneTargets' tracked layout
            // stays exact for subsequent forward/GXFfx draws into the same proxy.
            const vk::ImageMemoryBarrier2 src_back{
                .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
                .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                                 vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .image = src_img,
                .subresourceRange = one_range,
            };
            const std::array<vk::ImageMemoryBarrier2, 2> post{{to_read, src_back}};
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(post.size()), .pImageMemoryBarriers = post.data()});
        } else {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        }
    });

    // Always tag with the guest hdr_image handle so Execute() matches in both native and reduced
    // upscaler modes.
    opaque_hdr_src_vk = hdr_image.GetImage();
    opaque_hdr_valid = true;
}

bool NativePostFx::EnsureSsgiTargets(u32 hdr_w, u32 hdr_h) {
    const u32 half_w = std::max(hdr_w / 2u, 16u);
    const u32 half_h = std::max(hdr_h / 2u, 16u);
    const u32 qtr_w = std::max(half_w / 2u, 8u);
    const u32 qtr_h = std::max(half_h / 2u, 8u);
    if (ssgi_rad_mip0 && ssgi_rad_mip1 && ssgi_raw_img && ssgi_denoise_img &&
        ssgi_half_extent.width == half_w && ssgi_half_extent.height == half_h &&
        ssgi_quarter_extent.width == qtr_w && ssgi_quarter_extent.height == qtr_h) {
        return true;
    }

    scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();
    auto make_img = [&](VideoCore::HostImage& img, vk::UniqueImageView& view, u32 w, u32 h) {
        view.reset();
        img = VideoCore::HostImage(device, allocator);
        img.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view = Check<"create ssgi target view">(device.createImageViewUnique({
            .image = vk::Image(img),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    make_img(ssgi_rad_mip0, ssgi_rad_mip0_view, half_w, half_h);
    make_img(ssgi_rad_mip1, ssgi_rad_mip1_view, qtr_w, qtr_h);
    make_img(ssgi_raw_img, ssgi_raw_view, half_w, half_h);
    make_img(ssgi_denoise_img, ssgi_denoise_view, half_w, half_h);
    ssgi_half_extent = vk::Extent2D{half_w, half_h};
    ssgi_quarter_extent = vk::Extent2D{qtr_w, qtr_h};
    return true;
}

vk::Pipeline NativePostFx::GetOrCreateSsgiUpsamplePipeline(vk::Format color_format) {
    if (const auto it = ssgi_upsample_pipelines.find(color_format);
        it != ssgi_upsample_pipelines.end()) {
        return *it->second;
    }
    const vk::Device device = instance.GetDevice();
    const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{{
        {.stage = vk::ShaderStageFlagBits::eVertex, .module = *tonemap_vs, .pName = "main"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = *ssgi_upsample_fs,
         .pName = "main"},
    }};
    const vk::PipelineVertexInputStateCreateInfo vi{};
    const vk::PipelineInputAssemblyStateCreateInfo ia{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo vp{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rs{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo ms{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    // Hardware blend: dst_rgb = src_rgb * 1 + dst_rgb * src_a (preserve dst_a untouched).
    const vk::PipelineColorBlendAttachmentState cb_att{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB,
    };
    const vk::PipelineColorBlendStateCreateInfo cb{
        .attachmentCount = 1,
        .pAttachments = &cb_att,
    };
    const std::array<vk::DynamicState, 2> dyn_states{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    const vk::PipelineDynamicStateCreateInfo dyn{
        .dynamicStateCount = u32(dyn_states.size()),
        .pDynamicStates = dyn_states.data(),
    };
    const vk::PipelineRenderingCreateInfo rend_info{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    vk::GraphicsPipelineCreateInfo pci{
        .pNext = &rend_info,
        .stageCount = u32(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pColorBlendState = &cb,
        .pDynamicState = &dyn,
        .layout = *ssgi_upsample_layout,
    };
    auto pipe = Check<"create ssgi upsample pipeline">(device.createGraphicsPipelineUnique({}, pci));
    const vk::Pipeline handle = *pipe;
    ssgi_upsample_pipelines.emplace(color_format, std::move(pipe));
    return handle;
}

void NativePostFx::ExecuteHalfResSsgi(VideoCore::Image& hdr_image, vk::ImageView hdr_view,
                                      vk::ImageView g0_view, vk::ImageView depth_view,
                                      vk::ImageView g2_view, vk::ImageView diff_view,
                                      vk::ImageView amb_view,
                                      const vk::DescriptorBufferInfo& ssbo2_info,
                                      u32 d1_dword_offset, vk::Image proxy_img, u32 proxy_w,
                                      u32 proxy_h) {
    if (!EnsureInit()) {
        return;
    }
    const bool use_proxy = (proxy_img != VK_NULL_HANDLE && proxy_w >= 64u && proxy_h >= 64u);
    const u32 w = use_proxy ? proxy_w : hdr_image.info.size.width;
    const u32 h = use_proxy ? proxy_h : hdr_image.info.size.height;
    if (w < 64u || h < 64u || !hdr_view || !g0_view || !depth_view || !g2_view || !diff_view ||
        !amb_view || !ssbo2_info.buffer) {
        return;
    }
    if (!EnsureSsgiTargets(w, h)) {
        return;
    }
    const vk::Pipeline up_pipe = GetOrCreateSsgiUpsamplePipeline(hdr_image.info.pixel_format);
    if (!up_pipe) {
        return;
    }

    scheduler.EndRendering();
    if (!use_proxy) {
        runtime.Transit(&hdr_image, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentRead |
                            vk::AccessFlagBits2::eColorAttachmentWrite);
        runtime.FlushBarriers();
    }

    const vk::Image m0_img = vk::Image(ssgi_rad_mip0);
    const vk::ImageView m0_view = *ssgi_rad_mip0_view;
    const vk::Image m1_img = vk::Image(ssgi_rad_mip1);
    const vk::ImageView m1_view = *ssgi_rad_mip1_view;
    const vk::Image raw_img = vk::Image(ssgi_raw_img);
    const vk::ImageView raw_view = *ssgi_raw_view;
    const vk::Image denoise_img = vk::Image(ssgi_denoise_img);
    const vk::ImageView denoise_view = *ssgi_denoise_view;
    const vk::Extent2D half_ext = ssgi_half_extent;
    const vk::Extent2D qtr_ext = ssgi_quarter_extent;
    const vk::Sampler samp = *linear_sampler;
    const vk::Pipeline pf_pipe = *ssgi_prefilter_pipeline;
    const vk::PipelineLayout pf_layout = *ssgi_prefilter_layout;
    const vk::Pipeline tr_pipe = *ssgi_trace_pipeline;
    const vk::PipelineLayout tr_layout = *ssgi_trace_layout;
    const vk::PipelineLayout up_layout = *ssgi_upsample_layout;

    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        prof->Mark(0x9A71FE02ULL, [half_ext, qtr_ext] {
            return fmt::format("NativePostFX: HalfResSSGI {}x{} + {}x{} [VULKAN-NATIVE]",
                               half_ext.width, half_ext.height, qtr_ext.width, qtr_ext.height);
        });
    }
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange one_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        const std::array<vk::ImageMemoryBarrier2, 4> to_gen{{
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eFragmentShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = m0_img,
             .subresourceRange = one_range},
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = m1_img,
             .subresourceRange = one_range},
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eFragmentShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = raw_img,
             .subresourceRange = one_range},
            {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                             vk::PipelineStageFlagBits2::eFragmentShader,
             .srcAccessMask = vk::AccessFlagBits2::eNone,
             .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
             .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
             .oldLayout = vk::ImageLayout::eUndefined,
             .newLayout = vk::ImageLayout::eGeneral,
             .image = denoise_img,
             .subresourceRange = one_range},
        }};
        cmdbuf.pipelineBarrier2(
            {.imageMemoryBarrierCount = u32(to_gen.size()), .pImageMemoryBarriers = to_gen.data()});

        // 1. Prefilter Mip 0 (Half-Res packed firefly-clamped Radiance+Depth)
        const vk::DescriptorImageInfo d_in{VK_NULL_HANDLE, depth_view,
                                           vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo diff_in{VK_NULL_HANDLE, diff_view,
                                              vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo amb_in{VK_NULL_HANDLE, amb_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo m0_out{VK_NULL_HANDLE, m0_view, vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 4> pf0_writes{{
            {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d_in},
            {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &diff_in},
            {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &amb_in},
            {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &m0_out},
        }};
        const SsgiPrefilterPush pf0_pc{half_ext.width, half_ext.height, w, h, 0u};
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pf_pipe);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pf_layout, 0, pf0_writes);
        cmdbuf.pushConstants(pf_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pf0_pc), &pf0_pc);
        cmdbuf.dispatch((half_ext.width + 15u) / 16u, (half_ext.height + 15u) / 16u, 1u);

        const vk::ImageMemoryBarrier2 m0_ready{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = m0_img,
            .subresourceRange = one_range,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &m0_ready});

        // 2. Prefilter Mip 1 (Quarter-Res tent-filtered Radiance + clean inner-2x2 Depth)
        const vk::DescriptorImageInfo m0_in{VK_NULL_HANDLE, m0_view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo m1_out{VK_NULL_HANDLE, m1_view, vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 4> pf1_writes{{
            {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
            {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
            {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
            {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &m1_out},
        }};
        const SsgiPrefilterPush pf1_pc{qtr_ext.width, qtr_ext.height, half_ext.width, half_ext.height, 1u};
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pf_layout, 0, pf1_writes);
        cmdbuf.pushConstants(pf_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pf1_pc), &pf1_pc);
        cmdbuf.dispatch((qtr_ext.width + 15u) / 16u, (qtr_ext.height + 15u) / 16u, 1u);

        const vk::ImageMemoryBarrier2 m1_ready{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = m1_img,
            .subresourceRange = one_range,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &m1_ready});

        // 3. Half-Res 32-tap point-sampled HBIL SSGI Trace (L2-resident, zero bilinear depth bleed)
        const vk::DescriptorImageInfo m1_in{VK_NULL_HANDLE, m1_view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo samp_in{samp, VK_NULL_HANDLE, vk::ImageLayout::eUndefined};
        const vk::DescriptorImageInfo raw_out{VK_NULL_HANDLE, raw_view, vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 6> tr_writes{{
            {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
            {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m1_in},
            {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samp_in},
            {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &raw_out},
            {.dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &ssbo2_info},
            {.dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d_in},
        }};
        const SsgiTracePush tr_pc{half_ext.width, half_ext.height, d1_dword_offset,
                                  float(w) / float(std::max(h, 1u))};
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, tr_pipe);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, tr_layout, 0, tr_writes);
        cmdbuf.pushConstants(tr_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(tr_pc), &tr_pc);
        cmdbuf.dispatch((half_ext.width + 15u) / 16u, (half_ext.height + 15u) / 16u, 1u);

        // 3b. 3-Pass Edge-Preserving A-Trous Wavelet Denoiser (strides 1 -> 2 -> 4 in Half-Res)
        const vk::ImageMemoryBarrier2 raw_to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = raw_img,
            .subresourceRange = one_range,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &raw_to_read});

        const vk::DescriptorImageInfo raw_in_ro{VK_NULL_HANDLE, raw_view,
                                                vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo den_out_rw{VK_NULL_HANDLE, denoise_view,
                                                 vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo den_in_ro{VK_NULL_HANDLE, denoise_view,
                                                vk::ImageLayout::eShaderReadOnlyOptimal};
        const u32 half_gx = (half_ext.width + 15u) / 16u;
        const u32 half_gy = (half_ext.height + 15u) / 16u;
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pf_pipe);

        // Pass 1 (mode=2, stride=1 + firefly variance clamp): raw -> denoise
        {
            const std::array<vk::WriteDescriptorSet, 4> dn1_writes{{
                {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &raw_in_ro},
                {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &den_out_rw},
            }};
            const SsgiPrefilterPush dn1_pc{half_ext.width, half_ext.height, half_ext.width, half_ext.height, 2u};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pf_layout, 0, dn1_writes);
            cmdbuf.pushConstants(pf_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(dn1_pc), &dn1_pc);
            cmdbuf.dispatch(half_gx, half_gy, 1u);
        }
        {
            const std::array<vk::ImageMemoryBarrier2, 2> swap1{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = denoise_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = raw_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(swap1.size()), .pImageMemoryBarriers = swap1.data()});
        }

        // Pass 2 (mode=3, stride=2): denoise -> raw
        {
            const std::array<vk::WriteDescriptorSet, 4> dn2_writes{{
                {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &den_in_ro},
                {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &raw_out},
            }};
            const SsgiPrefilterPush dn2_pc{half_ext.width, half_ext.height, half_ext.width, half_ext.height, 3u};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pf_layout, 0, dn2_writes);
            cmdbuf.pushConstants(pf_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(dn2_pc), &dn2_pc);
            cmdbuf.dispatch(half_gx, half_gy, 1u);
        }
        {
            const std::array<vk::ImageMemoryBarrier2, 2> swap2{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = raw_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = denoise_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(swap2.size()), .pImageMemoryBarriers = swap2.data()});
        }

        // Pass 3 (mode=4, stride=4): raw -> denoise
        {
            const std::array<vk::WriteDescriptorSet, 4> dn3_writes{{
                {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &raw_in_ro},
                {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
                {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &den_out_rw},
            }};
            const SsgiPrefilterPush dn3_pc{half_ext.width, half_ext.height, half_ext.width, half_ext.height, 4u};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pf_layout, 0, dn3_writes);
            cmdbuf.pushConstants(pf_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(dn3_pc), &dn3_pc);
            cmdbuf.dispatch(half_gx, half_gy, 1u);
        }

        const vk::ImageMemoryBarrier2 den_ready{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = denoise_img,
            .subresourceRange = one_range,
        };
        if (use_proxy) {
            const vk::ImageMemoryBarrier2 hdr_sync{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                                 vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .image = proxy_img,
                .subresourceRange = one_range,
            };
            const std::array<vk::ImageMemoryBarrier2, 2> up_barriers{{den_ready, hdr_sync}};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(up_barriers.size()),
                                     .pImageMemoryBarriers = up_barriers.data()});
        } else {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &den_ready});
        }

        // 4. Full-Res 4-tap exact joint-bilateral upsample & blend directly onto hdr_image
        const vk::DescriptorImageInfo g0_in{VK_NULL_HANDLE, g0_view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
        const vk::DescriptorImageInfo g2_in{VK_NULL_HANDLE, g2_view,
                                            vk::ImageLayout::eShaderReadOnlyOptimal};
        const std::array<vk::WriteDescriptorSet, 6> up_writes{{
            {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &g0_in},
            {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d_in},
            {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &g2_in},
            {.dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &den_in_ro},
            {.dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m0_in},
            {.dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samp_in},
        }};
        const SsgiUpsamplePush up_pc{1.0f / float(half_ext.width), 1.0f / float(half_ext.height)};
        const vk::RenderingAttachmentInfo color_att{
            .imageView = hdr_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, up_pipe);
        cmdbuf.setViewport(0, vk::Viewport{0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f});
        cmdbuf.setScissor(0, vk::Rect2D{{0, 0}, {w, h}});
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, up_layout, 0, up_writes);
        cmdbuf.pushConstants(up_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(up_pc), &up_pc);
        cmdbuf.beginRendering({
            .renderArea = {{0, 0}, {w, h}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_att,
        });
        cmdbuf.draw(3u, 1u, 0u, 0u);
        cmdbuf.endRendering();
    });

    scheduler.BumpDescriptorStateEpoch();
    scheduler.BumpPushConstantsEpoch();
}

bool NativePostFx::EnsureMotionBlurTargets(u32 hdr_w, u32 hdr_h) {
    const u32 vel_w = std::max(hdr_w / 2u, 16u);
    const u32 vel_h = std::max(hdr_h / 2u, 16u);
    const u32 tile_w = (vel_w + 15u) / 16u;
    const u32 tile_h = (vel_h + 15u) / 16u;
    if (mb_vel_img && mb_tile_max_img && mb_color_img && mb_vel_extent.width == vel_w &&
        mb_vel_extent.height == vel_h && mb_tile_extent.width == tile_w &&
        mb_tile_extent.height == tile_h && mb_color_extent.width == hdr_w &&
        mb_color_extent.height == hdr_h) {
        return true;
    }

    scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();

    mb_vel_view.reset();
    mb_vel_img = VideoCore::HostImage(device, allocator);
    mb_vel_img.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {vel_w, vel_h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    mb_vel_view = Check<"create mb vel view">(device.createImageViewUnique({
        .image = vk::Image(mb_vel_img),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    mb_vel_extent = vk::Extent2D{vel_w, vel_h};

    mb_tile_max_view.reset();
    mb_tile_max_img = VideoCore::HostImage(device, allocator);
    mb_tile_max_img.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32Sfloat,
        .extent = {tile_w, tile_h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    mb_tile_max_view = Check<"create mb tile_max view">(device.createImageViewUnique({
        .image = vk::Image(mb_tile_max_img),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    mb_tile_extent = vk::Extent2D{tile_w, tile_h};

    mb_color_view.reset();
    mb_color_img = VideoCore::HostImage(device, allocator);
    mb_color_img.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {hdr_w, hdr_h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    mb_color_view = Check<"create mb color view">(device.createImageViewUnique({
        .image = vk::Image(mb_color_img),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    mb_color_extent = vk::Extent2D{hdr_w, hdr_h};
    return true;
}

bool NativePostFx::EnsureDofTargets(u32 hdr_w, u32 hdr_h) {
    const u32 half_w = std::max(hdr_w / 2u, 16u);
    const u32 half_h = std::max(hdr_h / 2u, 16u);
    if (dof_coc_img && dof_bokeh_img && dof_color_img && dof_half_extent.width == half_w &&
        dof_half_extent.height == half_h && dof_full_extent.width == hdr_w &&
        dof_full_extent.height == hdr_h) {
        return true;
    }

    scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    const VmaAllocator allocator = instance.GetAllocator();

    auto create_rgba16f = [&](VideoCore::HostImage& img, vk::UniqueImageView& view, u32 w, u32 h) {
        view.reset();
        img = VideoCore::HostImage(device, allocator);
        img.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                     vk::ImageUsageFlagBits::eTransferSrc,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view = Check<"create dof target view">(device.createImageViewUnique({
            .image = vk::Image(img),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };

    create_rgba16f(dof_coc_img, dof_coc_view, half_w, half_h);
    create_rgba16f(dof_bokeh_img, dof_bokeh_view, half_w, half_h);
    create_rgba16f(dof_color_img, dof_color_view, hdr_w, hdr_h);
    dof_half_extent = vk::Extent2D{half_w, half_h};
    dof_full_extent = vk::Extent2D{hdr_w, hdr_h};
    return true;
}

vk::Pipeline NativePostFx::GetOrCreateTonemapPipeline(vk::Format color_format) {
    if (auto it = tonemap_pipelines.find(color_format); it != tonemap_pipelines.end()) {
        return *it->second;
    }

    const vk::Device device = instance.GetDevice();
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *tonemap_vs,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = *tonemap_fs,
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
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    auto pipe = Check<"create native postfx tonemap pipeline">(device.createGraphicsPipelineUnique(
        {}, vk::GraphicsPipelineCreateInfo{
                .pNext = &rendering,
                .stageCount = u32(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &assembly,
                .pViewportState = &viewport,
                .pRasterizationState = &raster,
                .pMultisampleState = &multisample,
                .pColorBlendState = &blend,
                .pDynamicState = &dynamic,
                .layout = *tonemap_layout,
            }));
    const vk::Pipeline handle = *pipe;
    tonemap_pipelines.emplace(color_format, std::move(pipe));
    return handle;
}

bool NativePostFx::Execute(const BbCp::DirectVk::NativePostProcessArgs& args,
                           TemporalUpscaler* upscaler) {
    if (!EnsureInit()) {
        return false;
    }

    // 1. Resolve HDR input scene texture from T#.
    Gcn::Image hdr_sharp{};
    std::memcpy(&hdr_sharp, args.in_color_tsharp, sizeof(hdr_sharp));
    u8 hdr_gen8 = 0u;
    (void)VideoCore::DirectTextures::UnpackAndCleanSrv(hdr_sharp, hdr_gen8);
    if (!hdr_sharp.Valid()) {
        return false;
    }
    Shader::ImageResource hdr_res{};
    hdr_res.is_written = false;
    hdr_res.needs_native = false;
    VideoCore::TextureCache::ImageDesc hdr_desc{hdr_sharp, hdr_res};
    const VideoCore::ImageId hdr_id = texture_cache.FindImage(hdr_desc);
    if (!hdr_id) {
        return false;
    }
    const auto& hdr_view_obj = texture_cache.FindTexture(hdr_id, hdr_desc);
    auto& hdr_image = texture_cache.GetImage(hdr_id);

    // 2. Resolve destination LDR render target from Gcn::ColorBuffer (11 dwords).
    Gcn::ColorBuffer col_buf{};
    std::memcpy(&col_buf, args.out_cb_regs, sizeof(args.out_cb_regs));
    if (!col_buf) {
        return false;
    }
    const Gcn::CbDbExtent hint = std::bit_cast<Gcn::CbDbExtent>(args.out_cb_regs[6]);
    VideoCore::TextureCache::ImageDesc out_desc{col_buf, hint};
    const VideoCore::ImageId out_id = texture_cache.FindImage(out_desc);
    if (!out_id) {
        return false;
    }
    const auto& out_view_obj = texture_cache.FindRenderTarget(out_id, out_desc);
    auto& out_image = texture_cache.GetImage(out_id);

    // 3. Optionally resolve biome 16x16x16 3D LUT texture from T#.
    bool lut3d_active = false;
    VideoCore::ImageId lut_id{};
    vk::ImageView lut_view = *dummy_lut3d_view;
    if ((args.flags & BbCp::DirectVk::kNativePostFlagLut3d) != 0u) {
        Gcn::Image lut_sharp{};
        std::memcpy(&lut_sharp, args.lut3d_tsharp, sizeof(lut_sharp));
        u8 lut_gen8 = 0u;
        (void)VideoCore::DirectTextures::UnpackAndCleanSrv(lut_sharp, lut_gen8);
        if (lut_sharp.Valid()) {
            Shader::ImageResource lut_res{};
            lut_res.is_written = false;
            lut_res.needs_native = false;
            VideoCore::TextureCache::ImageDesc lut_desc{lut_sharp, lut_res};
            lut_id = texture_cache.FindImage(lut_desc);
            if (lut_id) {
                const auto& lut_view_obj = texture_cache.FindTexture(lut_id, lut_desc);
                if (lut_view_obj.info.type == Gcn::ImageType::Color3D) {
                    lut_view = *lut_view_obj.image_view;
                    lut3d_active = true;
                }
            }
        }
    }

    const vk::Extent2D out_extent{
        out_image.info.size.width,
        out_image.info.size.height,
    };
    const vk::Extent2D hdr_extent{
        hdr_image.info.size.width,
        hdr_image.info.size.height,
    };
    if (out_extent.width == 0u || out_extent.height == 0u || hdr_extent.width == 0u ||
        hdr_extent.height == 0u) {
        return false;
    }

    const vk::Pipeline tonemap_pipe = GetOrCreateTonemapPipeline(out_image.info.pixel_format);
    if (!tonemap_pipe) {
        return false;
    }

    // 4. Close any active guest dynamic render pass & resolve Linear HDR Pre-Post AA/Upscaling.
    scheduler.EndRendering();

    const bool auto_exp_on = (args.flags & BbCp::DirectVk::kNativePostFlagAutoExposure) != 0u;
    const bool ca_on = (args.flags & BbCp::DirectVk::kNativePostFlagCa) != 0u;
    const bool true_black_on = (args.flags & BbCp::DirectVk::kNativePostFlagTrueBlack) != 0u;
    const bool reset_history = (args.flags & BbCp::DirectVk::kNativePostFlagResetHistory) != 0u;
    const bool bloom_flag_on =
        ((args.flags & BbCp::DirectVk::kNativePostFlagBloom) != 0u) && (args.bloom_luminance > 1e-4f);

    // Step 3 & 4: Resolve Depth & Object Velocity (velWrap) when Motion Blur or DoF is enabled.
    const auto& settings = BbSettings::Get();
    const int mb_cam_lvl = std::clamp(settings.motion_blur_camera.load(std::memory_order_relaxed), 0, 3);
    const int mb_obj_lvl = std::clamp(settings.motion_blur_object.load(std::memory_order_relaxed), 0, 3);
    const bool mb_debug_vis = settings.native_mb_debug_vis.load(std::memory_order_relaxed);
    const bool mb_gparam_on = (args.flags & BbCp::DirectVk::kNativePostFlagMotionBlur) != 0u;
    const bool mb_wanted = (mb_gparam_on && (mb_cam_lvl > 0 || mb_obj_lvl > 0)) || mb_debug_vis;
    const bool dof_wanted =
        ((args.flags & BbCp::DirectVk::kNativePostFlagDof) != 0u) && !mb_debug_vis;

    bool has_depth = false;
    VideoCore::ImageId depth_id{};
    vk::ImageView depth_view = *dummy_black_2d_view;
    if ((mb_wanted || dof_wanted) && (args.flags & BbCp::DirectVk::kNativePostFlagHasDepth) != 0u) {
        Gcn::Image depth_sharp{};
        std::memcpy(&depth_sharp, args.in_depth_tsharp, sizeof(depth_sharp));
        u8 depth_gen8 = 0u;
        (void)VideoCore::DirectTextures::UnpackAndCleanSrv(depth_sharp, depth_gen8);
        if (depth_sharp.Valid()) {
            Shader::ImageResource depth_res{};
            depth_res.is_written = false;
            depth_res.needs_native = false;
            VideoCore::TextureCache::ImageDesc depth_desc{depth_sharp, depth_res};
            depth_id = texture_cache.FindImage(depth_desc);
            if (depth_id) {
                const auto& depth_view_obj = texture_cache.FindTexture(depth_id, depth_desc);
                depth_view = *depth_view_obj.image_view;
                has_depth = true;
            }
        }
    }

    const bool wants_prepost_hdr =
        upscaler && upscaler->WantsPrePostHdr() &&
        (hdr_image.info.pixel_format == vk::Format::eR16G16B16A16Sfloat);
    const bool fg_wanted_now = upscaler && upscaler->FrameGenWanted();
    const bool obj_moved = (args.flags & BbCp::DirectVk::kNativePostFlagObjMoved) != 0u;
    bool has_vel = false;
    VideoCore::ImageId vel_id{};
    vk::ImageView vel_view = *dummy_black_2d_view;
    u32 vel_fmt_dbg = 0u;
    u32 vel_w_dbg = 0u;
    u32 vel_h_dbg = 0u;
    if ((mb_wanted || wants_prepost_hdr || fg_wanted_now) &&
        (args.flags & BbCp::DirectVk::kNativePostFlagHasVel) != 0u) {
        Gcn::Image vel_sharp{};
        std::memcpy(&vel_sharp, args.vel_tsharp, sizeof(vel_sharp));
        u8 vel_gen8 = 0u;
        (void)VideoCore::DirectTextures::UnpackAndCleanSrv(vel_sharp, vel_gen8);
        if (vel_sharp.Valid()) {
            Shader::ImageResource vel_res{};
            vel_res.is_written = false;
            vel_res.needs_native = false;
            VideoCore::TextureCache::ImageDesc vel_desc{vel_sharp, vel_res};
            vel_id = texture_cache.FindImage(vel_desc);
            if (vel_id) {
                const auto& vel_img_dbg = texture_cache.GetImage(vel_id);
                vel_fmt_dbg = u32(vel_img_dbg.info.pixel_format);
                vel_w_dbg = vel_img_dbg.info.size.width;
                vel_h_dbg = vel_img_dbg.info.size.height;
                const auto& vel_view_obj = texture_cache.FindTexture(vel_id, vel_desc);
                vel_view = *vel_view_obj.image_view;
                has_vel = true;
            }
        }
    }

    // Phase 1 (v1.2.4): Linear HDR temporal AA / upscaling (DLAA/DLSS/FSR 3.1/FSR 4/TAA).
    // - Case A (render == output, e.g. DLAA/NativeAA/TAA): runs FIRST on raw Linear HDR before
    //   MotionBlur, DoF, Bloom, Fire Heat Haze, Tonemap & CA -> zero conflict between AA & PostFX.
    // - Case B (render < output, e.g. DLSS/FSR Quality): MotionBlur & DoF run at render-res with
    //   pixel-exact depth/vel, then upscaler reconstructs output-res Linear HDR before Bloom/Tonemap/CA.
    vk::Image src_hdr_img = hdr_image.GetImage();
    vk::ImageView src_hdr_view = *hdr_view_obj.image_view;
    vk::ImageLayout src_hdr_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageUsageFlags src_hdr_usage = hdr_image.usage_flags;
    vk::Extent2D render_extent = hdr_extent;
    vk::Extent2D post_aa_extent = hdr_extent;
    vk::Extent2D final_out_extent = out_extent;
    vk::ImageLayout depth_read_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    bool reduced_proxy = false;
    if (wants_prepost_hdr) {
        const auto prepost_in =
            upscaler->ResolvePrePostInput(hdr_id, src_hdr_view, depth_id, depth_view);
        if (prepost_in.color_image && prepost_in.color_view && prepost_in.render_extent.width > 0u &&
            prepost_in.render_extent.height > 0u) {
            src_hdr_img = prepost_in.color_image;
            src_hdr_view = prepost_in.color_view;
            src_hdr_layout = prepost_in.color_layout;
            src_hdr_usage = prepost_in.color_usage;
            render_extent = prepost_in.render_extent;
            post_aa_extent = prepost_in.output_extent;
            final_out_extent = prepost_in.output_extent;
            reduced_proxy = prepost_in.reduced_proxy;
            if (reduced_proxy && prepost_in.depth_view) {
                depth_view = prepost_in.depth_view;
                depth_read_layout = prepost_in.depth_layout;
            }
        }
    }

    // Unified AAA pipeline order (NVIDIA DLSS / AMD FSR Programming Guide):
    // ALWAYS run temporal AA / upscaling (DLAA/DLSS/FSR 3.1/FSR 4/TAA) FIRST on raw, sharp,
    // jittered Linear HDR BEFORE MotionBlur, DoF, Bloom, Heat Haze, Tonemap & CA.
    // Running DoF before DLSS feeds stochastic IGN bokeh noise into the upscaler and ruins
    // fine foliage reconstruction at low input resolutions (e.g. DLSS 4.5 Ultra Performance).
    const float frames30_now = std::clamp(BbFrameRate::CurrentFramesPer30(), 0.05f, 1.5f);
    const float fps_scale_now = std::clamp(1.0f / frames30_now, 1.0f, 12.0f);
    const int fps_q4_int_now =
        std::clamp(static_cast<int>(std::lround(fps_scale_now * 16.0f)), 16, 255);
    bool vel_transited_early = false;
    if (upscaler && has_vel && vel_id && obj_moved && (wants_prepost_hdr || fg_wanted_now)) {
        auto& vel_image = texture_cache.GetImage(vel_id);
        runtime.Transit(&vel_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        vel_transited_early = true;
        upscaler->SetNativeObjectVelocity(vel_view, vk::ImageLayout::eShaderReadOnlyOptimal,
                                          u32(fps_q4_int_now));
    }

    const std::array<float, 2> frame_depth_jitter =
        (upscaler && wants_prepost_hdr) ? upscaler->Jitter() : std::array<float, 2>{0.0f, 0.0f};
    TemporalUpscaler::PrePostHdrResult aa_res{};
    if (wants_prepost_hdr) {
        aa_res = upscaler->RunPrePostHdr(hdr_id, src_hdr_img, src_hdr_view, src_hdr_layout,
                                         render_extent, src_hdr_usage, /*src_is_guest_hdr=*/true,
                                         out_id, out_image.info.pixel_format, out_extent);
        if (aa_res.executed) {
            src_hdr_img = aa_res.image;
            src_hdr_view = aa_res.view;
            src_hdr_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
            render_extent = aa_res.extent;
            post_aa_extent = aa_res.extent;
            final_out_extent = aa_res.extent;
        } else {
            post_aa_extent = hdr_extent;
            final_out_extent = out_extent;
        }
    }

    const bool cam_moved = (args.flags & BbCp::DirectVk::kNativePostFlagCamMoved) != 0u;
    const bool mb_cam_active = mb_wanted && (mb_cam_lvl > 0) && has_depth && cam_moved;
    const bool mb_obj_active = mb_wanted && (mb_obj_lvl > 0) && has_vel && obj_moved;
    const bool mb_active = (mb_cam_active || mb_obj_active || mb_debug_vis) &&
                           EnsureMotionBlurTargets(render_extent.width, render_extent.height);
    // Pack #1 (#6 + #10): After >=2 consecutive static frames with neither camera nor object
    // motion, let Rasterizer skip the redundant 2-pass ps_abf92450 velocity dilation (which is
    // a mathematical no-op when v==0). Immediately cleared on any motion frame.
    {
        static u32 s_static_mb_streak = 0u;
        if (mb_wanted && !cam_moved && !obj_moved && !mb_debug_vis) {
            if (s_static_mb_streak < 2u) {
                ++s_static_mb_streak;
            }
            BbStats::mb_idle_skip_dilate.store(s_static_mb_streak >= 2u, std::memory_order_relaxed);
        } else {
            s_static_mb_streak = 0u;
            BbStats::mb_idle_skip_dilate.store(false, std::memory_order_relaxed);
        }
    }
    const bool dof_active =
        dof_wanted && has_depth && EnsureDofTargets(render_extent.width, render_extent.height);

    if (!reduced_proxy && !(wants_prepost_hdr && aa_res.executed)) {
        runtime.Transit(&hdr_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    if ((mb_active || dof_active) && has_depth && depth_id && !reduced_proxy) {
        auto& depth_image = texture_cache.GetImage(depth_id);
        runtime.Transit(&depth_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    if (mb_active && has_vel && vel_id && !vel_transited_early) {
        auto& vel_image = texture_cache.GetImage(vel_id);
        runtime.Transit(&vel_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    if (lut3d_active && lut_id) {
        auto& lut_image = texture_cache.GetImage(lut_id);
        runtime.Transit(&lut_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    runtime.FlushBarriers();

    // Determine exact RGBA vs BGRA export from GNM ColorBuffer swizzle & Vulkan format.
    const auto swz = col_buf.Swizzle();
    const bool bgra_export =
        (swz.r == Gcn::CompSwizzle::Blue && swz.b == Gcn::CompSwizzle::Red) &&
        (out_image.info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
         out_image.info.pixel_format == vk::Format::eR8G8B8A8Srgb);

    // Prepare shared exposure/optics scalars.
    const float safe_gamma = (args.gamma > 0.1f && args.gamma < 10.0f) ? args.gamma : 2.2f;
    const float safe_lum_scale =
        (args.scene_lum_scale > 0.01f && args.scene_lum_scale < 100.0f) ? args.scene_lum_scale
                                                                        : 1.0f;
    // Yebis2 (eboot.bin 0xbe8f60: vdivss 0x6b0(%r12), %xmm0, %xmm0) DIVIDES by SceneLuminanceScale!
    const float inv_lum_scale = 1.0f / safe_lum_scale;
    const float safe_fov = std::clamp(args.fov_y_rad, 0.2f, 2.5f);
    const float fov_dep = std::clamp(args.vig_fov_dep, 0.0f, 1.0f);
    const float tan_half_fov = std::tan(0.5f * safe_fov);
    const float vig_tan = 0.57735027f + (tan_half_fov - 0.57735027f) * fov_dep;
    const float aspect = (args.aspect > 0.2f && args.aspect < 6.0f)
                             ? args.aspect
                             : (float(final_out_extent.width) / float(final_out_extent.height));

    // Opt-in diagnostic log (`BB_POSTFX_LOG=1`) of live GPARAM values for 1:1 Yebis2 parity tuning.
    static const bool s_postfx_log =
        (std::getenv("BB_POSTFX_LOG") != nullptr && std::getenv("BB_POSTFX_LOG")[0] == '1');
    if (s_postfx_log) {
        static u32 s_last_flags = 0xFFFFFFFFu;
        static int s_last_tone_type = -999;
        static float s_last_exp = -1.0f;
        static float s_last_tf = -1.0f;
        static float s_last_lum = -1.0f;
        const bool changed = (args.flags != s_last_flags) || (args.tone_type != s_last_tone_type) ||
                             (std::abs(args.exposure - s_last_exp) > 0.02f) ||
                             (std::abs(args.tone_factor - s_last_tf) > 0.02f) ||
                             (std::abs(args.scene_lum_scale - s_last_lum) > 0.02f);
        if (changed) {
            s_last_flags = args.flags;
            s_last_tone_type = args.tone_type;
            s_last_exp = args.exposure;
            s_last_tf = args.tone_factor;
            s_last_lum = args.scene_lum_scale;
            std::printf(
                "[NativePostFX] flags=0x%x toneType=%d exp=%.4f gamma=%.4f toneFac=%.4f "
                "filmic=%.4f lumScale=%.4f midGray=%.4f vig=(%.3f,%.3f) "
                "bloom=(on=%d lum=%.3f thr=%.3f rad=%.3f) "
                "mb=(on=%d cam=%d obj=%d timeR=%.3f maxL=%.3f vScale=%.3f hasD=%d hasV=%d vel=%u:%ux%u) "
                "dof=(on=%d cust=%d/%d/%d foc=%.2f f=%.2f near=%.1f..%.1f:%.2f*%.2f far=%.1f..%.1f:%.2f*%.2f cut=%.1f z=%.2f..%.0f) "
                "hdrFmt=%u outFmt=%u bgra=%d lut3d=%d preAA=%d (%ux%u->%ux%u)\n",
                args.flags, args.tone_type, args.exposure, args.gamma, args.tone_factor,
                args.filmic_contrast, args.scene_lum_scale, args.middle_gray, args.vig_power,
                args.vig_fov_dep, bloom_flag_on ? 1 : 0, args.bloom_luminance, args.bloom_threshold,
                args.bloom_radius_scale, mb_active ? 1 : 0, mb_cam_lvl, mb_obj_lvl,
                args.mb_time_ratio, args.mb_max_len, args.mb_vel_scale, has_depth ? 1 : 0,
                has_vel ? 1 : 0, vel_fmt_dbg, vel_w_dbg, vel_h_dbg, dof_active ? 1 : 0,
                (args.flags & BbCp::DirectVk::kNativePostFlagDofCustom) ? 1 : 0,
                (args.flags & BbCp::DirectVk::kNativePostFlagDofNearCust) ? 1 : 0,
                (args.flags & BbCp::DirectVk::kNativePostFlagDofFarCust) ? 1 : 0,
                args.dof_focus_dist, args.dof_fstop, args.dof_near_start, args.dof_near_end,
                args.dof_near_coc_size, args.dof_near_coc_scale, args.dof_far_start,
                args.dof_far_end, args.dof_far_coc_size, args.dof_far_coc_scale,
                args.dof_far_cutoff, args.dof_view_near_z, args.dof_view_far_z,
                u32(hdr_image.info.pixel_format), u32(out_image.info.pixel_format),
                bgra_export ? 1 : 0, lut3d_active ? 1 : 0, wants_prepost_hdr ? 1 : 0,
                render_extent.width, render_extent.height, final_out_extent.width,
                final_out_extent.height);
        }
    }

    auto& stream = buffer_cache.GetStreamBuffer();
    const vk::Buffer ubo_buffer = stream.Handle();

    // Prepare Step 3 Motion Blur UBO (independent FPS-normalized Camera & Object scales).
    u64 mb_ubo_offset = 0u;
    if (mb_active) {
        static constexpr float kMbMult[4] = {0.0f, 0.45f, 1.00f, 1.75f};
        static constexpr float kLenMult[4] = {0.0f, 0.65f, 1.00f, 1.60f};
        const float frames30 = std::clamp(BbFrameRate::CurrentFramesPer30(), 0.05f, 1.5f);
        const float fps_scale_raw = std::clamp(1.0f / frames30, 1.0f, 12.0f);
        // Match exact Q4.4 quantization used in Pass 0x2e (`buf_offsets[37]`) so
        // `obj_ndc_30 - cam_ndc_30` cancels to 0.000 on static NPCs at every FPS!
        const int fps_q4_int =
            std::clamp(static_cast<int>(std::lround(fps_scale_raw * 16.0f)), 16, 255);
        const float fps_q4 = float(fps_q4_int) * 0.0625f;
        const float time_ratio =
            (args.mb_time_ratio > 0.01f && args.mb_time_ratio <= 4.0f) ? args.mb_time_ratio : 0.5f;
        const float max_len_ndc =
            (args.mb_max_len > 0.005f && args.mb_max_len <= 1.0f) ? args.mb_max_len : 0.06f;
        const float max_len_uv = max_len_ndc * 0.5f;
        const float vel_scale =
            (args.mb_vel_scale > 0.05f && args.mb_vel_scale <= 5.0f) ? args.mb_vel_scale : 1.0f;
        const float obj_boost =
            std::clamp(settings.native_mb_obj_boost.load(std::memory_order_relaxed), 0.25f, 4.0f);
        // Exact PS4 30-FPS Object Motion Blur shutter ratio: BlurTimeRatio * 1.45 (~0.138 in
        // Hunter's Dream, matching PS4 original rather than over-amplifying by raw MotionBlurScale=4.0).
        const float obj_time_scale = std::clamp(time_ratio * 1.45f, 0.10f, 0.28f);

        // When Frame Generation is active, attenuate pre-interpolation stochastic IGN Motion Blur
        // shutter so Optical Flow and Frame Interpolation track crisp limb/weapon silhouettes
        // instead of warping pre-blurred streaks into double contours.
        const int fg_mult = fg_wanted_now
                                ? BbSettings::FrameGenMultiplier(BbSettings::EffectiveFrameGen())
                                : 1;
        const float fg_scale = (fg_mult >= 2) ? (2.0f / float(fg_mult)) : 1.0f;
        const float fg_cam_damp = (fg_mult >= 2) ? (0.45f * fg_scale) : 1.0f;
        const float fg_obj_damp = (fg_mult >= 2) ? (0.25f * fg_scale) : 1.0f;
        MbVelUbo mb_ubo{};
        std::memcpy(mb_ubo.clip_to_prev, args.clip_to_prev_clip, sizeof(mb_ubo.clip_to_prev));
        mb_ubo.params0[0] = time_ratio * fps_q4 * kMbMult[mb_cam_lvl] * fg_cam_damp;
        mb_ubo.params0[1] = obj_time_scale * kMbMult[mb_obj_lvl] * obj_boost * fg_obj_damp;
        mb_ubo.params0[2] = std::clamp(max_len_uv * kLenMult[mb_cam_lvl] * fg_cam_damp, 0.004f, 0.09f);
        mb_ubo.params0[3] = std::clamp(max_len_uv * 1.25f * kLenMult[mb_obj_lvl] * fg_obj_damp, 0.004f, 0.11f);
        mb_ubo.params1[0] = fps_q4;
        mb_ubo.params1[1] = 0.25f;   // Exact inverse of Pass 0x2e Native PostFX enc_scale = 4.0
        mb_ubo.params1[2] = 0.0065f; // NDC deadzone (~3.3 LSB) crushes 8-bit static NPC residual noise
        mb_ubo.params1[3] = 0.0f;
        mb_ubo.dims_flags[0] = mb_vel_extent.width;
        mb_ubo.dims_flags[1] = mb_vel_extent.height;
        mb_ubo.dims_flags[2] = (mb_cam_active ? 1u : 0u) | (mb_obj_active ? 2u : 0u) |
                               (has_depth ? 4u : 0u) | (has_vel ? 8u : 0u);
        mb_ubo.dims_flags[3] = 0u;
        mb_ubo_offset = stream.Copy(&mb_ubo, sizeof(mb_ubo), 256);
    }

    // Prepare Step 4 Depth of Field UBO (exact Bloodborne thin-lens + CustomDof from live GPARAM).
    u64 dof_ubo_offset = 0u;
    float dof_max_coc_uv = 0.018f;
    if (dof_active) {
        const float ccd_diag_m =
            std::clamp(args.dof_ccd_diag_mm > 1.0f ? args.dof_ccd_diag_mm : 43.267f, 5.0f, 120.0f) *
            0.001f;
        const float sensor_h_m = ccd_diag_m / std::sqrt(aspect * aspect + 1.0f);
        const float focal_m = 0.5f * sensor_h_m / std::max(tan_half_fov, 0.05f);
        const float fstop = std::clamp(args.dof_fstop > 0.5f ? args.dof_fstop : 5.6f, 0.7f, 64.0f);
        const float focus_dist =
            std::clamp(args.dof_focus_dist > 0.05f ? args.dof_focus_dist : 10.0f, 0.1f, 5000.0f);
        const float aperture_m = focal_m / fstop;
        const float dof_str =
            std::clamp(settings.native_dof_strength.load(std::memory_order_relaxed), 0.25f, 3.0f);
        // Signed CoC in screen-height UV radius (0.5 * diameter / sensor_h_m -> radius).
        const float phys_coc_factor =
            0.5f * dof_str * (aperture_m * focal_m) /
            (std::max(focus_dist - focal_m, 0.05f) * std::max(sensor_h_m, 1e-4f));

        // Verified from live Hunter's Dream GPARAM (`far=10..50: 0.30 * 0.02 = 0.0060`) & pixel-gradient
        // A/B measurement vs Yebis2 (`Tree 12-18m`, `Far pillars 50m+`):
        // CustomDof `CocSize * CocScale` is CoC DIAMETER in screen-height UV; scaling by 0.38
        // compensates for half-res 2x2 pre-filter + optical disk FWHM -> exact 1:1 Yebis2 blur width
        // (~3.3 px radius @ 1440p on 50m+ pillars) while keeping mid-ground trees razor-sharp.
        const float near_start = std::max(args.dof_near_start, 0.0f);
        const float near_end = std::max(args.dof_near_end, near_start + 1e-3f);
        const float near_coc_uv = 0.38f * std::max(args.dof_near_coc_size, 0.0f) * dof_str;
        const float near_scale = std::clamp(args.dof_near_coc_scale, 0.0f, 2.0f);

        const float far_start = std::max(args.dof_far_start, 0.0f);
        const float far_end = std::max(args.dof_far_end, far_start + 1e-3f);
        const float far_coc_uv = 0.38f * std::max(args.dof_far_coc_size, 0.0f) * dof_str;
        const float far_scale = std::clamp(args.dof_far_coc_scale, 0.0f, 2.0f);

        const float dof_scale = std::clamp(args.dof_scale > 0.01f ? args.dof_scale : 0.5f, 0.05f, 0.5f);
        dof_max_coc_uv = std::clamp(0.026f * (dof_scale / 0.5f) * std::max(dof_str, 1.0f), 0.008f, 0.042f);

        u32 dof_mode = 0u;
        if ((args.flags & BbCp::DirectVk::kNativePostFlagDofCustom) != 0u) dof_mode |= 1u;
        if ((args.flags & BbCp::DirectVk::kNativePostFlagDofNearCust) != 0u) dof_mode |= 2u;
        if ((args.flags & BbCp::DirectVk::kNativePostFlagDofFarCust) != 0u) dof_mode |= 4u;

        DofUbo dof_ubo{};
        dof_ubo.clip_phys[0] = std::max(args.dof_view_near_z, 0.01f);
        dof_ubo.clip_phys[1] = std::max(args.dof_view_far_z, dof_ubo.clip_phys[0] + 1.0f);
        dof_ubo.clip_phys[2] = focus_dist;
        dof_ubo.clip_phys[3] = phys_coc_factor;
        dof_ubo.near_params[0] = near_start;
        dof_ubo.near_params[1] = near_end;
        dof_ubo.near_params[2] = near_coc_uv;
        dof_ubo.near_params[3] = near_scale;
        dof_ubo.far_params[0] = far_start;
        dof_ubo.far_params[1] = far_end;
        dof_ubo.far_params[2] = far_coc_uv;
        dof_ubo.far_params[3] = far_scale;
        dof_ubo.misc_params[0] = std::max(args.dof_far_cutoff, 0.0f);
        dof_ubo.misc_params[1] = dof_max_coc_uv;
        dof_ubo.misc_params[2] = aspect;
        dof_ubo.misc_params[3] = dof_scale;
        dof_ubo.dims_flags[0] = dof_half_extent.width;
        dof_ubo.dims_flags[1] = dof_half_extent.height;
        dof_ubo.dims_flags[2] = render_extent.width;
        dof_ubo.dims_flags[3] = render_extent.height;
        dof_ubo.mode_flags[0] = dof_mode;
        dof_ubo.mode_flags[1] = args.frame_index;
        dof_ubo.mode_flags[2] = 0u;
        dof_ubo.mode_flags[3] = 0u;
        dof_ubo.depth_jitter[0] = aa_res.executed ? frame_depth_jitter[0] : 0.0f;
        dof_ubo.depth_jitter[1] = aa_res.executed ? frame_depth_jitter[1] : 0.0f;
        dof_ubo.depth_jitter[2] = 0.0f;
        dof_ubo.depth_jitter[3] = 0.0f;
        dof_ubo_offset = stream.Copy(&dof_ubo, sizeof(dof_ubo), 256);
    }

    const bool need_clear = !gpu_state_cleared;
    gpu_state_cleared = true;

    const vk::Sampler samp = *linear_sampler;
    const vk::Image dummy_img = vk::Image(dummy_lut3d);
    const vk::Image dummy_blk_img = vk::Image(dummy_black_2d);
    const vk::Buffer h_buf = hist_buf;
    const vk::Buffer e_buf = exp_state_buf;
    const vk::Pipeline h_pipe = *hist_pipeline;
    const vk::PipelineLayout h_layout = *hist_layout;
    const vk::Pipeline ae_pipe = *auto_exp_pipeline;
    const vk::PipelineLayout ae_layout = *auto_exp_layout;
    const vk::Pipeline bd_pipe = *bloom_down_pipeline;
    const vk::PipelineLayout bd_layout = *bloom_down_layout;
    const vk::Pipeline bu_pipe = *bloom_up_pipeline;
    const vk::PipelineLayout bu_layout = *bloom_up_layout;
    const vk::Pipeline mv_pipe = *mb_vel_pipeline;
    const vk::PipelineLayout mv_layout = *mb_vel_layout;
    const vk::Pipeline mb_pipe = *mb_blur_pipeline;
    const vk::PipelineLayout mb_layout = *mb_blur_layout;
    const vk::Pipeline dc_pipe = *dof_coc_pipeline;
    const vk::PipelineLayout dc_layout = *dof_coc_layout;
    const vk::Pipeline db_pipe = *dof_bokeh_pipeline;
    const vk::PipelineLayout db_layout = *dof_bokeh_layout;
    const vk::Pipeline dm_pipe = *dof_comp_pipeline;
    const vk::PipelineLayout dm_layout = *dof_comp_layout;
    const vk::PipelineLayout tm_layout = *tonemap_layout;

    const vk::Image mb_v_img = mb_active ? vk::Image(mb_vel_img) : vk::Image{};
    const vk::ImageView mb_v_view = mb_active ? *mb_vel_view : *dummy_black_2d_view;
    const vk::Extent2D mb_v_ext = mb_vel_extent;
    const vk::Image mb_tm_img = mb_active ? vk::Image(mb_tile_max_img) : vk::Image{};
    const vk::ImageView mb_tm_view = mb_active ? *mb_tile_max_view : *dummy_black_2d_view;
    const vk::Image mb_c_img = mb_active ? vk::Image(mb_color_img) : vk::Image{};
    const vk::ImageView mb_c_view = mb_active ? *mb_color_view : src_hdr_view;
    const vk::ImageView after_mb_view = mb_active ? mb_c_view : src_hdr_view;
    const vk::ImageLayout after_mb_layout =
        mb_active ? vk::ImageLayout::eShaderReadOnlyOptimal : src_hdr_layout;

    const vk::Image d_coc_img = dof_active ? vk::Image(dof_coc_img) : vk::Image{};
    const vk::ImageView d_coc_view = dof_active ? *dof_coc_view : *dummy_black_2d_view;
    const vk::Image d_bok_img = dof_active ? vk::Image(dof_bokeh_img) : vk::Image{};
    const vk::ImageView d_bok_view = dof_active ? *dof_bokeh_view : *dummy_black_2d_view;
    const vk::Extent2D d_half_ext = dof_half_extent;
    const vk::Image d_col_img = dof_active ? vk::Image(dof_color_img) : vk::Image{};
    const vk::ImageView d_col_view = dof_active ? *dof_color_view : after_mb_view;
    vk::ImageView post_scene_view = dof_active ? d_col_view : after_mb_view;
    vk::ImageLayout post_scene_layout =
        dof_active ? vk::ImageLayout::eShaderReadOnlyOptimal : after_mb_layout;
    const DofBokehPush dof_bokeh_pc{
        .width = d_half_ext.width,
        .height = d_half_ext.height,
        .inv_aspect = (aspect > 0.1f) ? (1.0f / aspect) : (9.0f / 16.0f),
        .max_coc_uv = dof_max_coc_uv,
        .frame_index = args.frame_index,
        .reserved = 0u,
    };

    const MbBlurPush mb_blur_pc{
        .width = render_extent.width,
        .height = render_extent.height,
        .inv_vel_w = (mb_v_ext.width > 0u) ? (1.0f / float(mb_v_ext.width)) : 0.0f,
        .inv_vel_h = (mb_v_ext.height > 0u) ? (1.0f / float(mb_v_ext.height)) : 0.0f,
        .frame_index = args.frame_index,
        .flags = (mb_cam_active ? 1u : 0u) | (mb_obj_active ? 2u : 0u) | (mb_debug_vis ? 4u : 0u),
    };

    if (need_clear) {
        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
            cmdbuf.fillBuffer(h_buf, 0, kHistBufBytes, 0u);
            cmdbuf.fillBuffer(e_buf, 0, kExpStateBufBytes, 0u);
            const vk::ImageMemoryBarrier2 blk_to_clear{
                .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eClear,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = dummy_blk_img,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &blk_to_clear,
            });
            const vk::ClearColorValue zero_col{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};
            const vk::ImageSubresourceRange one_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            cmdbuf.clearColorImage(dummy_blk_img, vk::ImageLayout::eTransferDstOptimal, &zero_col,
                                   1, &one_range);

            const std::array<vk::ImageMemoryBarrier2, 2> dummy_bars{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eNone,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = dummy_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eClear,
                 .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = dummy_blk_img,
                 .subresourceRange = one_range},
            }};
            const vk::MemoryBarrier2 buf_bar{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                 vk::AccessFlagBits2::eShaderWrite,
            };
            cmdbuf.pipelineBarrier2({
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &buf_bar,
                .imageMemoryBarrierCount = u32(dummy_bars.size()),
                .pImageMemoryBarriers = dummy_bars.data(),
            });
        });
    }

    // Pass 0 (Compute): Native 16-tap Bilateral IGN Motion Blur (Camera Depth + Object velWrap).
    if (mb_active) {
        GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Post_MotionBlur);
        if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
            prof->Mark(0x9A71FE10ULL, [render_extent] {
                return fmt::format("NativePostFX: MotionBlur {}x{} [VULKAN-NATIVE]",
                                   render_extent.width, render_extent.height);
            });
        }
        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange one_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            const std::array<vk::ImageMemoryBarrier2, 2> vel_to_write{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = mb_v_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = mb_tm_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(vel_to_write.size()),
                                     .pImageMemoryBarriers = vel_to_write.data()});

            const vk::DescriptorImageInfo d_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = depth_view,
                .imageLayout = depth_read_layout,
            };
            const vk::DescriptorImageInfo v_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = vel_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo samp_info{
                .sampler = samp,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = vk::ImageLayout::eUndefined,
            };
            const vk::DescriptorImageInfo out_v_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = mb_v_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const vk::DescriptorImageInfo out_tm_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = mb_tm_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const vk::DescriptorBufferInfo mb_ubo_info{
                .buffer = ubo_buffer,
                .offset = mb_ubo_offset,
                .range = sizeof(MbVelUbo),
            };
            const std::array<vk::WriteDescriptorSet, 6> mv_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &d_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &v_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 3,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &out_v_info},
                {.dstBinding = 4,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eUniformBuffer,
                 .pBufferInfo = &mb_ubo_info},
                {.dstBinding = 5,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &out_tm_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, mv_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, mv_layout, 0, mv_writes);
            cmdbuf.dispatch((mb_v_ext.width + 15u) / 16u, (mb_v_ext.height + 15u) / 16u, 1u);

            const std::array<vk::ImageMemoryBarrier2, 3> vel_done_bars{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = mb_v_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = mb_tm_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = mb_c_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(vel_done_bars.size()),
                                     .pImageMemoryBarriers = vel_done_bars.data()});

            const vk::DescriptorImageInfo hdr_in_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = src_hdr_view,
                .imageLayout = src_hdr_layout,
            };
            const vk::DescriptorImageInfo vel_read_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = mb_v_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo tm_read_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = mb_tm_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo mb_out_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = mb_c_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 5> mb_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &hdr_in_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &vel_read_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 3,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &mb_out_info},
                {.dstBinding = 4,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &tm_read_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, mb_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, mb_layout, 0, mb_writes);
            cmdbuf.pushConstants(mb_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(MbBlurPush),
                                 &mb_blur_pc);
            cmdbuf.dispatch((render_extent.width + 15u) / 16u, (render_extent.height + 15u) / 16u, 1u);

            const vk::ImageMemoryBarrier2 mb_to_sampled{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = mb_c_img,
                .subresourceRange = one_range,
            };
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &mb_to_sampled});
        });
    }

    // Pass 0b (Compute): Native Resolution-Independent FP16 Bokeh Depth of Field (3 passes).
    if (dof_active) {
        GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Post_DoF);
        if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
            prof->Mark(0x9A71FE11ULL, [render_extent] {
                return fmt::format("NativePostFX: DoF {}x{} [VULKAN-NATIVE]",
                                   render_extent.width, render_extent.height);
            });
        }
        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange one_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            const vk::DescriptorImageInfo samp_info{
                .sampler = samp,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = vk::ImageLayout::eUndefined,
            };
            const vk::DescriptorBufferInfo dof_ubo_info{
                .buffer = ubo_buffer,
                .offset = dof_ubo_offset,
                .range = sizeof(DofUbo),
            };
            const vk::DescriptorImageInfo scene_in_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = after_mb_view,
                .imageLayout = after_mb_layout,
            };
            const vk::DescriptorImageInfo depth_in_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = depth_view,
                .imageLayout = depth_read_layout,
            };

            // 0b-1: Half-res CoC + Karis-weighted HDR pre-filter.
            const vk::ImageMemoryBarrier2 coc_to_write{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = d_coc_img,
                .subresourceRange = one_range,
            };
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &coc_to_write});

            const vk::DescriptorImageInfo coc_out_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = d_coc_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 5> dc_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &scene_in_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &depth_in_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 3,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &coc_out_info},
                {.dstBinding = 4,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eUniformBuffer,
                 .pBufferInfo = &dof_ubo_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, dc_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, dc_layout, 0, dc_writes);
            cmdbuf.dispatch((d_half_ext.width + 15u) / 16u, (d_half_ext.height + 15u) / 16u, 1u);

            // 0b-2: Half-res 24-tap Golden-Angle Vogel Disk Bokeh gather.
            const std::array<vk::ImageMemoryBarrier2, 2> coc_done_bars{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = d_coc_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = d_bok_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(coc_done_bars.size()),
                                     .pImageMemoryBarriers = coc_done_bars.data()});

            const vk::DescriptorImageInfo coc_read_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = d_coc_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo bok_out_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = d_bok_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 3> db_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &coc_read_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &bok_out_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, db_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, db_layout, 0, db_writes);
            cmdbuf.pushConstants(db_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(DofBokehPush), &dof_bokeh_pc);
            cmdbuf.dispatch((d_half_ext.width + 15u) / 16u, (d_half_ext.height + 15u) / 16u, 1u);

            // 0b-3: Full-res bilateral composite (preserves native 1440p/4K sharp silhouettes).
            const std::array<vk::ImageMemoryBarrier2, 2> bok_done_bars{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                 .oldLayout = vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                 .image = d_bok_img,
                 .subresourceRange = one_range},
                {.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eFragmentShader,
                 .srcAccessMask = vk::AccessFlagBits2::eNone,
                 .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = d_col_img,
                 .subresourceRange = one_range},
            }};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(bok_done_bars.size()),
                                     .pImageMemoryBarriers = bok_done_bars.data()});

            const vk::DescriptorImageInfo bok_read_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = d_bok_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dof_out_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = d_col_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 7> dm_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &scene_in_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &depth_in_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &bok_read_info},
                {.dstBinding = 3,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 4,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dof_out_info},
                {.dstBinding = 5,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eUniformBuffer,
                 .pBufferInfo = &dof_ubo_info},
                {.dstBinding = 6,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &coc_read_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, dm_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, dm_layout, 0, dm_writes);
            cmdbuf.dispatch((render_extent.width + 15u) / 16u, (render_extent.height + 15u) / 16u,
                            1u);

            const vk::ImageMemoryBarrier2 dof_to_sampled{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = d_col_img,
                .subresourceRange = one_range,
            };
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &dof_to_sampled});
        });
    }

    const vk::ImageView raw_hdr_view = src_hdr_view;
    const vk::ImageLayout raw_hdr_layout = src_hdr_layout;

    // Prepare final output target (direct ui_image on upscaled/FG frames, or guest out_image).
    const bool use_ui_target = aa_res.executed && aa_res.use_ui_target && aa_res.ui_image &&
                               aa_res.ui_view;
    vk::ImageView final_out_view = *out_view_obj.image_view;
    vk::Image final_out_ui_img{};
    if (use_ui_target) {
        final_out_view = aa_res.ui_view;
        final_out_ui_img = aa_res.ui_image;
    } else {
        if (VideoCore::True(out_image.flags & VideoCore::ImageFlagBits::DirectImmutable)) {
            texture_cache.DemoteDirectImmutable(out_id);
        }
        runtime.Transit(&out_image, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::AccessFlagBits2::eColorAttachmentWrite);
        runtime.FlushBarriers();
    }

    const bool bloom_on =
        bloom_flag_on && EnsureBloomPyramid(post_aa_extent.width, post_aa_extent.height);
    const bool heat_haze_on =
        bloom_on && settings.fire_heat_haze.load(std::memory_order_relaxed);
    const float heat_haze_str =
        heat_haze_on
            ? std::clamp(settings.fire_heat_haze_strength.load(std::memory_order_relaxed), 0.0f, 2.0f)
            : 0.0f;

    u32 shader_flags = 0u;
    if (lut3d_active) shader_flags |= kShaderFlagLut3d;
    if (auto_exp_on) shader_flags |= kShaderFlagAutoExp;
    if (ca_on) shader_flags |= kShaderFlagCa;
    if (true_black_on) shader_flags |= kShaderFlagTrueBlack;
    if (bgra_export) shader_flags |= kShaderFlagBgraExport;
    if (bloom_on) shader_flags |= kShaderFlagBloom;

    float c832 = 1.0f;
    float c833 = 0.0f;
    ComputeYebisSensitometricParams(args.filmic_contrast, c832, c833);

    const float bloom_int_scale =
        std::clamp(settings.native_bloom_intensity.load(std::memory_order_relaxed), 0.0f, 0.40f);
    const float bloom_intensity =
        bloom_on ? std::clamp(args.bloom_luminance * bloom_int_scale, 0.0f, 2.25f) : 0.0f;

    TonemapUbo ubo{};
    ubo.tone_params[0] = (args.exposure > 1e-4f) ? args.exposure : 1.0f;
    ubo.tone_params[1] = 1.0f / safe_gamma;
    ubo.tone_params[2] = c832;
    ubo.tone_params[3] = c833;
    ubo.scene_vig[0] = inv_lum_scale;
    ubo.scene_vig[1] = (args.vig_power > 0.0f) ? args.vig_power : 0.0f;
    ubo.scene_vig[2] = vig_tan;
    ubo.scene_vig[3] = aspect;
    ubo.color_corr[0] = args.hue_rad;
    ubo.color_corr[1] = (args.saturation >= 0.0f) ? args.saturation : 1.0f;
    ubo.color_corr[2] = (args.value_scale >= 0.0f) ? args.value_scale : 1.0f;
    ubo.color_corr[3] = (args.contrast >= 0.0f) ? args.contrast : 1.0f;
    ubo.ca_params[0] = args.ca_lat_disp[0];
    ubo.ca_params[1] = args.ca_lat_disp[1];
    ubo.ca_params[2] = args.ca_uni_disp[0];
    ubo.ca_params[3] = args.ca_uni_disp[1];
    static const auto s_postfx_t0 = std::chrono::steady_clock::now();
    const double now_sec_hh =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - s_postfx_t0).count();
    ubo.bloom_params[0] = bloom_intensity;
    ubo.bloom_params[1] = heat_haze_str;
    ubo.bloom_params[2] = static_cast<float>(std::fmod(now_sec_hh, 4096.0));
    ubo.bloom_params[3] = 1.0f / static_cast<float>(std::max(final_out_extent.height, 1u));
    ubo.flags_misc[0] = shader_flags;
    ubo.flags_misc[1] = static_cast<u32>(std::max(args.tone_type, 0));
    ubo.flags_misc[2] = static_cast<u32>(std::clamp(args.ca_wave_samples, 3, 7));
    ubo.flags_misc[3] = args.frame_index;
    const u64 ubo_offset = stream.Copy(&ubo, sizeof(ubo), 256);

    // Metering rect from GPARAM (width, height, offX, offY).
    const float mw = (args.meter_rect[0] > 0.05f && args.meter_rect[0] <= 1.0f) ? args.meter_rect[0]
                                                                                 : 1.0f;
    const float mh = (args.meter_rect[1] > 0.05f && args.meter_rect[1] <= 1.0f) ? args.meter_rect[1]
                                                                                 : 1.0f;
    const float mcx = 0.5f + std::clamp(args.meter_rect[2], -0.45f, 0.45f);
    const float mcy = 0.5f + std::clamp(args.meter_rect[3], -0.45f, 0.45f);
    const u32 hist_stride = 4u;
    const HistPush hist_pc{
        .width = post_aa_extent.width,
        .height = post_aa_extent.height,
        .min_log_lum = kMinLogLum,
        .inv_log_lum_range = 1.0f / kLogLumRange,
        .meter_min_x = std::clamp(mcx - 0.5f * mw, 0.0f, 1.0f),
        .meter_min_y = std::clamp(mcy - 0.5f * mh, 0.0f, 1.0f),
        .meter_max_x = std::clamp(mcx + 0.5f * mw, 0.0f, 1.0f),
        .meter_max_y = std::clamp(mcy + 0.5f * mh, 0.0f, 1.0f),
        .scene_lum_scale = inv_lum_scale,
        .stride = hist_stride,
    };

    const float bright_rate = std::clamp(
        (args.adapt_scale > 1e-4f ? args.adapt_scale : 0.1f) * 18.0f, 0.5f, 12.0f);
    const float dark_raw = (args.dark_scale > 1e-4f) ? args.dark_scale : args.adapt_scale;
    const float dark_rate = std::clamp((dark_raw > 1e-4f ? dark_raw : 0.05f) * 12.0f, 0.25f, 10.0f);
    const float bright_lim = (args.adapt_speed_limit > 1e-4f) ? args.adapt_speed_limit : 4.0f;
    const float dark_lim =
        (args.dark_speed_limit > 1e-4f) ? args.dark_speed_limit : bright_lim;

    const AutoExpPush auto_exp_pc{
        .min_log_lum = kMinLogLum,
        .log_lum_range = kLogLumRange,
        .middle_gray = (args.middle_gray > 1e-3f && args.middle_gray <= 2.0f) ? args.middle_gray
                                                                               : 0.18f,
        .min_exposure = (args.min_exposure > 1e-3f) ? args.min_exposure : 0.05f,
        .max_exposure = (args.max_exposure > 1e-3f) ? args.max_exposure : 16.0f,
        .manual_exposure = ubo.tone_params[0],
        .dt_sec = std::clamp(args.delta_time_sec, 1e-4f, 0.25f),
        .bright_adapt_rate = bright_rate,
        .dark_adapt_rate = dark_rate,
        .bright_speed_limit = bright_lim,
        .dark_speed_limit = dark_lim,
        .flags = (auto_exp_on ? 1u : 0u) | (reset_history ? 2u : 0u),
    };

    const vk::ImageView bloom_final_view =
        bloom_on ? *bloom_mip_views[0] : *dummy_black_2d_view;
    const vk::Image bloom_img = bloom_on ? vk::Image(bloom_pyramid) : vk::Image{};
    std::array<vk::ImageView, kBloomMips> b_views{};
    std::array<vk::Extent2D, kBloomMips> b_extents{};
    if (bloom_on) {
        for (u32 i = 0; i < kBloomMips; ++i) {
            b_views[i] = *bloom_mip_views[i];
            b_extents[i] = bloom_mip_extents[i];
        }
    }

    const float bloom_thr_mult =
        std::clamp(settings.native_bloom_threshold.load(std::memory_order_relaxed), 0.50f, 5.00f);
    const float bloom_thr = std::clamp(args.bloom_threshold * bloom_thr_mult, 0.20f, 4.00f);
    const float bloom_knee = std::max(bloom_thr * 0.30f, 0.08f);
    const float bloom_luma_gam =
        (args.bloom_luma_gamma > 0.1f && args.bloom_luma_gamma < 10.0f) ? args.bloom_luma_gamma
                                                                        : 1.0f;
    const float bloom_remap = std::clamp(args.bloom_remap, 0.0f, 5.0f);
    const float bloom_rad_scale =
        std::clamp(args.bloom_radius_scale > 0.05f ? args.bloom_radius_scale : 1.0f, 0.35f, 4.0f);
    const float bloom_disp =
        std::clamp(args.bloom_dispersion > 0.05f ? args.bloom_dispersion : 1.0f, 0.25f, 2.5f);
    const float bloom_up_w = 0.52f * std::pow(bloom_disp, 0.25f);
    const float bloom_cur_w = 0.62f;
    const float manual_exp = ubo.tone_params[0];
    const bool use_opaque_diff =
        heat_haze_on && opaque_hdr_valid && opaque_hdr_view &&
        (opaque_hdr_src_vk == hdr_image.GetImage());
    opaque_hdr_valid = false;
    const vk::ImageView opaque_v = use_opaque_diff ? *opaque_hdr_view : *dummy_black_2d_view;

    // Pass 1 & 2 (Compute): True-dt Auto-Exposure + 5-level FP16 Dual-Kawase Bloom Pyramid.
    if (auto_exp_on || bloom_on) {
        GpuProfiler::HudMarkPass(&scheduler, bloom_on ? BbStats::GpuSub_Post_BloomGlare
                                                      : BbStats::GpuSub_Post_TonemapLut);
        if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
            prof->Mark(bloom_on ? 0x9A71FE12ULL : 0x9A71FE14ULL,
                       [post_aa_extent, bloom_on, heat_haze_on] {
                           return bloom_on
                                      ? fmt::format(
                                            "NativePostFX: Bloom+AutoExp {}x{} [Haze:{}] [VULKAN-NATIVE]",
                                            post_aa_extent.width, post_aa_extent.height,
                                            heat_haze_on ? 1 : 0)
                                      : fmt::format(
                                            "NativePostFX: AutoExposure {}x{} [VULKAN-NATIVE]",
                                            post_aa_extent.width, post_aa_extent.height);
                       });
        }
        scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const bool fused_hist = bloom_on && auto_exp_on;
        const vk::DescriptorBufferInfo hist_info{
            .buffer = h_buf,
            .offset = 0,
            .range = kHistBufBytes,
        };
        const vk::DescriptorBufferInfo exp_info{
            .buffer = e_buf,
            .offset = 0,
            .range = kExpStateBufBytes,
        };
        const std::array<vk::WriteDescriptorSet, 2> ae_writes{{
            {.dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &hist_info},
            {.dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &exp_info},
        }};

        // Standalone Auto-Exposure path (only when Bloom is OFF; when Bloom is ON, histogram is
        // fused directly into bloom_down mip 0 with zero extra HDR texture reads).
        if (auto_exp_on && !fused_hist) {
            const vk::DescriptorImageInfo hdr_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = raw_hdr_view,
                .imageLayout = raw_hdr_layout,
            };
            const vk::DescriptorImageInfo samp_info{
                .sampler = samp,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = vk::ImageLayout::eUndefined,
            };
            const std::array<vk::WriteDescriptorSet, 3> hist_writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &hdr_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampler,
                 .pImageInfo = &samp_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &hist_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, h_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, h_layout, 0, hist_writes);
            cmdbuf.pushConstants(h_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(HistPush),
                                 &hist_pc);
            const u32 sample_w = (post_aa_extent.width + hist_stride - 1u) / hist_stride;
            const u32 sample_h = (post_aa_extent.height + hist_stride - 1u) / hist_stride;
            cmdbuf.dispatch((sample_w + 15u) / 16u, (sample_h + 15u) / 16u, 1u);

            const vk::MemoryBarrier2 hist_to_ae{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                 vk::AccessFlagBits2::eShaderWrite,
            };
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &hist_to_ae});

            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, ae_pipe);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, ae_layout, 0, ae_writes);
            cmdbuf.pushConstants(ae_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(AutoExpPush), &auto_exp_pc);
            cmdbuf.dispatch(1u, 1u, 1u);

            const vk::MemoryBarrier2 ae_to_next{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            };
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &ae_to_next});
        }

        // Pass 2b (Compute): 5-level FP16 Dual-Kawase / Jimenez Bloom Pyramid (down + up),
        // with Phase 3 Step 3.1 Fused Auto-Exposure Histogram in mip 0 & batched mip transitions.
        if (bloom_on) {
            const vk::DescriptorImageInfo samp_info{
                .sampler = samp,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = vk::ImageLayout::eUndefined,
            };

            // Transition all 5 bloom mips to eGeneral in ONE single batched barrier.
            const vk::ImageMemoryBarrier2 all_mips_to_storage{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = bloom_img,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, kBloomMips, 0, 1},
            };
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &all_mips_to_storage});

            // Downsample chain: mip 0 (prefilter + Karis + fused AutoExp hist) -> mip 1 -> 2 -> 3 -> 4.
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, bd_pipe);
            for (u32 mip = 0; mip < kBloomMips; ++mip) {
                const vk::ImageView src_v = (mip == 0u) ? post_scene_view : b_views[mip - 1u];
                const vk::ImageLayout src_l =
                    (mip == 0u) ? post_scene_layout : vk::ImageLayout::eShaderReadOnlyOptimal;
                const vk::Extent2D src_ext = (mip == 0u) ? post_aa_extent : b_extents[mip - 1u];
                const vk::Extent2D dst_ext = b_extents[mip];

                const vk::DescriptorImageInfo src_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = src_v,
                    .imageLayout = src_l,
                };
                const vk::DescriptorImageInfo dst_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = b_views[mip],
                    .imageLayout = vk::ImageLayout::eGeneral,
                };
                const vk::DescriptorImageInfo op_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = opaque_v,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                };
                const vk::DescriptorImageInfo raw_hdr_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = raw_hdr_view,
                    .imageLayout = raw_hdr_layout,
                };
                const std::array<vk::WriteDescriptorSet, 7> bd_writes{{
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
                     .pImageInfo = &dst_info},
                    {.dstBinding = 3,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &exp_info},
                    {.dstBinding = 4,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage,
                     .pImageInfo = &op_info},
                    {.dstBinding = 5,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage,
                     .pImageInfo = &raw_hdr_info},
                    {.dstBinding = 6,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &hist_info},
                }};
                const BloomDownPush bd_pc{
                    .dst_w = dst_ext.width,
                    .dst_h = dst_ext.height,
                    .inv_src_w = 1.0f / float(src_ext.width),
                    .inv_src_h = 1.0f / float(src_ext.height),
                    .manual_exposure = manual_exp,
                    .inv_scene_lum_scale = inv_lum_scale,
                    .threshold = bloom_thr,
                    .knee = bloom_knee,
                    .luma_gamma = bloom_luma_gam,
                    .remap_factor = bloom_remap,
                    .flags = ((mip == 0u) ? 1u : 0u) | (auto_exp_on ? 2u : 0u) |
                             ((mip == 0u && use_opaque_diff) ? 4u : 0u) |
                             ((mip == 0u && fused_hist) ? 8u : 0u) |
                             ((mip == 0u && fused_hist &&
                               (post_aa_extent.width / std::max(dst_ext.width, 1u)) >= 3u)
                                  ? 16u
                                  : 0u) |
                             ((mip == 0u && post_scene_view == raw_hdr_view) ? 32u : 0u),
                    .min_log_lum = hist_pc.min_log_lum,
                    .inv_log_lum_range = hist_pc.inv_log_lum_range,
                    .meter_min_x = hist_pc.meter_min_x,
                    .meter_min_y = hist_pc.meter_min_y,
                    .meter_max_x = hist_pc.meter_max_x,
                    .meter_max_y = hist_pc.meter_max_y,
                };
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, bd_layout, 0, bd_writes);
                cmdbuf.pushConstants(bd_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(BloomDownPush), &bd_pc);
                cmdbuf.dispatch((dst_ext.width + 15u) / 16u, (dst_ext.height + 15u) / 16u, 1u);

                const vk::ImageMemoryBarrier2 to_sampled{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                    vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .image = bloom_img,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1},
                };
                if (mip == 0u && fused_hist) {
                    const vk::MemoryBarrier2 hist_to_ae{
                        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                         vk::AccessFlagBits2::eShaderWrite,
                    };
                    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1,
                                             .pMemoryBarriers = &hist_to_ae,
                                             .imageMemoryBarrierCount = 1,
                                             .pImageMemoryBarriers = &to_sampled});
                    // Dispatch 1x1x1 Auto-Exposure reduction concurrently with bloom_down mips 1..4!
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, ae_pipe);
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, ae_layout, 0,
                                                ae_writes);
                    cmdbuf.pushConstants(ae_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                         sizeof(AutoExpPush), &auto_exp_pc);
                    cmdbuf.dispatch(1u, 1u, 1u);
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, bd_pipe);
                } else {
                    cmdbuf.pipelineBarrier2(
                        {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_sampled});
                }
            }

            // Upsample chain: mip 4 -> 3 -> 2 -> 1 -> 0 (9-tap tent progressive combine).
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, bu_pipe);
            for (int mip = int(kBloomMips) - 2; mip >= 0; --mip) {
                const u32 umip = u32(mip);
                const vk::ImageMemoryBarrier2 to_rw{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                     vk::AccessFlagBits2::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .image = bloom_img,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, umip, 1, 0, 1},
                };
                cmdbuf.pipelineBarrier2(
                    {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_rw});

                const vk::Extent2D dst_ext = b_extents[umip];
                const vk::DescriptorImageInfo lower_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = b_views[umip + 1u],
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                };
                const vk::DescriptorImageInfo dst_info{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = b_views[umip],
                    .imageLayout = vk::ImageLayout::eGeneral,
                };
                const std::array<vk::WriteDescriptorSet, 3> bu_writes{{
                    {.dstBinding = 0,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage,
                     .pImageInfo = &lower_info},
                    {.dstBinding = 1,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampler,
                     .pImageInfo = &samp_info},
                    {.dstBinding = 2,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageImage,
                     .pImageInfo = &dst_info},
                }};
                const float lower_inv_h = 1.0f / float(std::max(b_extents[umip + 1u].height, 1u));
                const float rad_y = lower_inv_h * 0.85f * bloom_rad_scale;
                const float rad_x = rad_y / aspect;
                const BloomUpPush bu_pc{
                    .dst_w = dst_ext.width,
                    .dst_h = dst_ext.height,
                    .radius_x = rad_x,
                    .radius_y = rad_y,
                    .up_weight = bloom_up_w,
                    .cur_weight = bloom_cur_w,
                };
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, bu_layout, 0, bu_writes);
                cmdbuf.pushConstants(bu_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(BloomUpPush), &bu_pc);
                cmdbuf.dispatch((dst_ext.width + 15u) / 16u, (dst_ext.height + 15u) / 16u, 1u);

                const vk::ImageMemoryBarrier2 to_sampled{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                    vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .image = bloom_img,
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, umip, 1, 0, 1},
                };
                if (umip == 0u && fused_hist) {
                    const vk::MemoryBarrier2 ae_to_tonemap{
                        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                        vk::PipelineStageFlagBits2::eFragmentShader,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    };
                    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1,
                                             .pMemoryBarriers = &ae_to_tonemap,
                                             .imageMemoryBarrierCount = 1,
                                             .pImageMemoryBarriers = &to_sampled});
                } else {
                    cmdbuf.pipelineBarrier2(
                        {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_sampled});
                }
            }
        }
        });
    }

    // Pass 3 (Graphics): Unified Tonemap + Bloom + Original 3D LUT + CA + Vignette + True Black.
    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Post_TonemapLut);
    if (auto* prof = GpuProfiler::Get(); prof && prof->Records(&scheduler)) {
        prof->Mark(0x9A71FE13ULL, [final_out_extent, bloom_on, heat_haze_on] {
            return fmt::format("NativePostFX: Tonemap+LUT {}x{} [Bloom:{} Haze:{}] [VULKAN-NATIVE]",
                               final_out_extent.width, final_out_extent.height, bloom_on ? 1 : 0,
                               heat_haze_on ? 1 : 0);
        });
    }
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const vk::DescriptorImageInfo hdr_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = post_scene_view,
            .imageLayout = post_scene_layout,
        };
        const vk::DescriptorImageInfo lut_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = lut_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        const vk::DescriptorImageInfo samp_info{
            .sampler = samp,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = vk::ImageLayout::eUndefined,
        };
        const vk::DescriptorBufferInfo exp_info{
            .buffer = e_buf,
            .offset = 0,
            .range = kExpStateBufBytes,
        };
        const vk::DescriptorBufferInfo ubo_info{
            .buffer = ubo_buffer,
            .offset = ubo_offset,
            .range = sizeof(TonemapUbo),
        };
        const vk::DescriptorImageInfo bloom_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = bloom_final_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        const std::array<vk::WriteDescriptorSet, 6> tm_writes{{
            {.dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &hdr_info},
            {.dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &lut_info},
            {.dstBinding = 2,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampler,
             .pImageInfo = &samp_info},
            {.dstBinding = 3,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer,
             .pBufferInfo = &exp_info},
            {.dstBinding = 4,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eUniformBuffer,
             .pBufferInfo = &ubo_info},
            {.dstBinding = 5,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &bloom_info},
        }};
        const vk::RenderingAttachmentInfo color_att{
            .imageView = final_out_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, tonemap_pipe);
        cmdbuf.setViewport(0, vk::Viewport{0.0f, 0.0f, float(final_out_extent.width),
                                           float(final_out_extent.height), 0.0f, 1.0f});
        cmdbuf.setScissor(0, vk::Rect2D{{0, 0}, final_out_extent});
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, tm_layout, 0, tm_writes);
        cmdbuf.beginRendering({
            .renderArea = {{0, 0}, final_out_extent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_att,
        });
        cmdbuf.draw(3u, 1u, 0u, 0u);
        cmdbuf.endRendering();
        if (use_ui_target && final_out_ui_img) {
            const vk::ImageMemoryBarrier2 ui_ready{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                                vk::PipelineStageFlagBits2::eFragmentShader |
                                vk::PipelineStageFlagBits2::eComputeShader |
                                vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                                 vk::AccessFlagBits2::eColorAttachmentWrite |
                                 vk::AccessFlagBits2::eShaderRead |
                                 vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = final_out_ui_img,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ui_ready});
        }
    });

    out_image.gpu_write_tick = scheduler.CurrentTick();
    if (upscaler && !use_ui_target) {
        upscaler->OnColorTarget(out_id);
    }
    scheduler.BumpDescriptorStateEpoch();
    return true;
}

} // namespace Vulkan
