// SPDX-License-Identifier: GPL-2.0-or-later
// Pack #4 (v2): Deferred Single-Wave Native Vulkan Compute VSM Blur Batcher (`vs 0x0b0acf50 + ps 0x0f61fd27`).
//
// Intercepts Bloodborne's serialized cubemap VSM blur sequence:
//   `CopyTextureRegion(F[slice] -> S)` + `Draw(ps_0f61fd27, S -> F[slice])`
// Cancels ALL `Copy(F -> S)` transfers and ALL `ps_0f61fd27` graphics render passes, queuing
// every face across all active VSM cubemaps (`32/64/128/256`) while the GPU stays 100% in Graphics
// mode rendering shadow geometry. Right before non-VSM passes read the cubemaps, executes ONE
// single Graphics->Compute wave (1 pipeline bind, all faces concurrent across all SMs with HW
// bilinear filtering into an L2-resident 64-slice scratch pool) + ONE grouped copy-back per cubemap.
#pragma once

#include <array>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "render/rasterizer.h"
#include "textures/image.h"
#include "textures/types.h"
#include "vulkan/vk_common.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;

class VsmBlurBatcher {
public:
    static constexpr u32 kScratchDim = 256u;
    static constexpr u32 kMaxScratchSlices = 64u;

    struct DeferredCopy {
        bool valid = false;
        Rasterizer::DirectCopyTextureRegionArgs args{};
        VideoCore::ImageId src_id{};
        u32 src_mip = 0;
        u32 src_layer = 0;
        VAddr src_tsharp_addr = 0;
        VideoCore::ImageId dst_id{};
        u32 dst_mip = 0;
        u32 dst_layer = 0;
        VAddr dst_tsharp_addr = 0;
        u32 dim = 0;
    };

    VsmBlurBatcher(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                   VideoCore::TextureCache& texture_cache);
    ~VsmBlurBatcher();

    VsmBlurBatcher(const VsmBlurBatcher&) = delete;
    VsmBlurBatcher& operator=(const VsmBlurBatcher&) = delete;

    [[nodiscard]] static bool Enabled() noexcept;
    [[nodiscard]] static constexpr bool IsSupportedDim(u32 dim) noexcept {
        return dim == 32u || dim == 64u || dim == 128u || dim == 256u;
    }

    [[nodiscard]] bool HasDeferredCopy() const noexcept {
        return deferred_copy.valid;
    }
    void SetDeferredCopy(const DeferredCopy& copy) noexcept {
        deferred_copy = copy;
    }
    [[nodiscard]] bool TakeDeferredCopy(DeferredCopy& out) noexcept {
        if (!deferred_copy.valid) {
            return false;
        }
        out = deferred_copy;
        deferred_copy = {};
        return true;
    }

    [[nodiscard]] bool HasPendingCopyBacks() const noexcept {
        return !pending_blurs.empty();
    }
    [[nodiscard]] bool ConflictsWithCopyBack(VideoCore::ImageId id, u32 mip,
                                             u32 layer) const noexcept;

    /// Consumes `deferred_copy` and queues the face's 13-tap Gaussian blur for the single-wave
    /// Compute flush. `ssbo1_immutable` / `ssbo2_immutable` indicate whether each buffer already
    /// resides in immutable per-frame ring/stream memory; mutable CBs are snapshotted (208 B / 32 B)
    /// into a 32 KiB GPU ring so ALL faces still execute in ONE single Compute wave.
    [[nodiscard]] bool TryConsumeDeferredAndBlur(VideoCore::ImageId cb0_id, u32 cb0_mip,
                                                 u32 cb0_layer, VAddr cb0_addr, VAddr fs_img0_addr,
                                                 const vk::DescriptorBufferInfo& ssbo1,
                                                 const vk::DescriptorBufferInfo& ssbo2,
                                                 u32 d0_dw, u32 d1_dw, bool ssbo1_immutable,
                                                 bool ssbo2_immutable);

    /// Executes ALL queued VSM blurs in ONE concurrent Compute wave + ONE grouped copy-back.
    void FlushCopyBacks();

private:
    struct PendingBlur {
        VideoCore::ImageId face_id{};
        u32 mip = 0;
        u32 layer = 0;
        u32 dim = 0;
        u32 scratch_slot = 0;
        vk::DescriptorBufferInfo ssbo1{};
        vk::DescriptorBufferInfo ssbo2{};
        u32 d0_dw = 0;
        u32 d1_dw = 0;
    };

    struct PushConstants {
        u32 dim = 0;
        u32 d0_dw = 0;
        u32 d1_dw = 0;
        u32 pad = 0;
    };
    static_assert(sizeof(PushConstants) == 16);

    static constexpr u32 kParamSlotStride = 512u; // 256-B aligned halves: +0 (ssbo1 208B), +256 (ssbo2 32B)

    void EnsurePipelines();
    void EnsureScratchPool();
    void EnsureParamsBuffer();

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;

    bool pipelines_ready = false;
    bool scratch_ready = false;
    bool params_ready = false;
    bool used_params_this_wave = false;
    bool logged_first_use = false;

    vk::UniqueSampler linear_clamp_sampler;

    vk::UniqueDescriptorSetLayout scratch_set_layout;
    vk::UniquePipelineLayout scratch_pipe_layout;
    vk::UniquePipeline scratch_pipeline;

    VideoCore::HostImage scratch_pool;
    std::array<vk::UniqueImageView, kMaxScratchSlices> scratch_slice_views{};

    vk::Buffer params_buf{};
    VmaAllocation params_alloc{};

    DeferredCopy deferred_copy{};
    boost::container::small_vector<PendingBlur, kMaxScratchSlices> pending_blurs;
};

} // namespace Vulkan
