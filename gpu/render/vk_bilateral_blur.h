// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "common/types.h"
#include "textures/texture_cache.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;

/// Pack #3: Native Vulkan Compute + LDS replacement for 12-tap depth-aware bilateral
/// shadow blur (`vs 0x061fbbf6 + ps 0x11ffdfb5`, `GXScreenSpaceShadow_Blur.ppo`).
///
/// Validates the 208-byte `cbBlurParam` constant buffer on the CPU and dispatches bank-conflict-free
/// LDS compute shaders (`bilateral_blur_h.comp`, `bilateral_blur_v.comp`, or `bilateral_blur_2d.comp`)
/// with push-constant kernel parameters (0 SSBO reads, precomputed reciprocal depth denominator,
/// warp-level sky early-out). Uses exact `floor(0.5 + offset_px)` when the sampler is point/nearest
/// (guaranteed for 24-bit RGB-packed depth). Falls back transparently to `ps_11ffdfb5.frag` on any
/// unsupported layout. Controlled by `BB_BILATERAL_COMPUTE` (default `1`; set `0` for A/B testing).
class BilateralBlurCompute {
public:
    static constexpr u32 kMaxHalo1D = 16u;
    static constexpr u32 kMaxHalo2D = 6u;

    struct PushConstants1D {
        u32 width = 0u;
        u32 height = 0u;
        f32 z_a = 1.0f;
        u32 halo = 6u;
        s32 delta[12]{};
        f32 kw[12]{};
    };
    static_assert(sizeof(PushConstants1D) == 112u);

    struct PushConstants2D {
        u32 width = 0u;
        u32 height = 0u;
        f32 z_a = 1.0f;
        u32 halo = 4u;
        u32 dxy[6]{};
        u32 pad[2]{};
        f32 kw[12]{};
    };
    static_assert(sizeof(PushConstants2D) == 96u);

    BilateralBlurCompute(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                         VideoCore::TextureCache& texture_cache);
    ~BilateralBlurCompute();

    BilateralBlurCompute(const BilateralBlurCompute&) = delete;
    BilateralBlurCompute& operator=(const BilateralBlurCompute&) = delete;

    static bool Enabled() noexcept;

    bool TryDispatch(VideoCore::ImageId src_id, u32 src_mip, u32 src_layer,
                     VideoCore::ImageId dst_id, u32 dst_mip, u32 dst_layer,
                     const u8* cb_ptr, u32 cb_size_bytes, bool is_nearest);

private:
    void EnsurePipelines();

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;

    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipe_layout;
    vk::UniquePipeline pipeline_h;
    vk::UniquePipeline pipeline_v;
    vk::UniquePipeline pipeline_2d;

    bool logged_h = false;
    bool logged_v = false;
    bool logged_2d = false;
    bool logged_fallback = false;
};

} // namespace Vulkan
