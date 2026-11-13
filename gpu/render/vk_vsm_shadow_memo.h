// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 9 (Target A): Stage-A Memoization for Local VSM Point/Spot Shadow Maps (`R16G16Sfloat`
// `32x32..512x512` cubemap faces & 2D spot maps) and their separable 13-tap Gaussian blurs
// (`Filter\Sample13Weight.ppo`, `vs 0x0b0acf50 + ps 0x0f61fd27`).
//
// Eliminates redundant 170 Hz re-rendering, subresource copies, clears, and blur passes for
// unchanged local shadow faces while re-rendering with 0-frame latency whenever any caster
// (character, enemy, bone palette, cloth dynamic VB, breakable prop, or light frustum) changes.
#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"
#include "gcn/regs.h"
#include "gcn/sharps.h"
#include "render/rasterizer.h"

namespace Vulkan {

class VsmShadowMemo {
public:
    static constexpr size_t kMaxFaces = 64u;
    static constexpr size_t kMaxScratchLinks = 16u;
    static constexpr size_t kMaxDrawsPerFace = 48u;
    static constexpr u64 kMaxPeriodicRefreshFrames = 120u;
    static constexpr size_t kStageBWords =
        Gcn::RegDirty::StageBNumChunks * Gcn::RegDirty::StageBChunkWords;

    using StageBChunks = std::array<u32, kStageBWords>;

    VsmShadowMemo();
    ~VsmShadowMemo() = default;

    VsmShadowMemo(const VsmShadowMemo&) = delete;
    VsmShadowMemo& operator=(const VsmShadowMemo&) = delete;

    [[nodiscard]] static bool Enabled() noexcept;

    /// Lock-free global hooks called from any thread (allocator / DirectBuffers / DirectTextures)
    /// and from Stage A whenever GPU buffer or texture ranges are written or retired.
    static void NoteBufferWriteGlobal(VAddr addr, u64 byte_size) noexcept;
    static void NoteRetiredTextureRangeGlobal(VAddr addr, u64 byte_size) noexcept;
    [[nodiscard]] static u64 QueryPage64kWriteDigest(VAddr addr, u64 byte_size) noexcept;

    [[nodiscard]] bool IsReplaying() const noexcept {
        return replaying;
    }
    [[nodiscard]] bool HasPendingWork() const noexcept {
        return !replaying &&
               (pending_face.active || has_pending_depth_clear || num_deferred_clears != 0u);
    }

    /// Stage-A hooks for DirectVk clear / copy / draw commands:
    [[nodiscard]] bool TryInterceptClearColor(Rasterizer& rasterizer,
                                              const Rasterizer::DirectClearColorArgs& args);
    [[nodiscard]] bool TryInterceptClearDepth(Rasterizer& rasterizer,
                                              const Rasterizer::DirectClearDepthStencilArgs& args);
    [[nodiscard]] bool TryInterceptCopyTextureRegion(
        Rasterizer& rasterizer, const Rasterizer::DirectCopyTextureRegionArgs& args);
    void NotifyBeforeCopyBatch(Rasterizer& rasterizer, const u32* raw_regions, u32 count);
    void InvalidateTargetRangeOnStageA(Rasterizer& rasterizer, VAddr dst_va, u64 byte_size);

    [[nodiscard]] bool TryHandleDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                     const PreparedDraw* used_prepared, bool is_indexed,
                                     u32 index_offset);

    /// Flushes any open pending face, depth clear, and deferred color clears in exact stream order.
    void FlushAll(Rasterizer& rasterizer);

private:
    struct VsmTargetKey {
        bool valid = false;
        VAddr base_va = 0;
        u32 slice = 0;
        VAddr slice_va = 0;
        u32 slice_bytes = 0;
        u32 dim = 0;
    };

    struct FaceEntry {
        bool allocated = false;
        bool valid = false;
        bool is_known_vsm = false;
        bool was_blurred = false;
        bool saw_blur_after_face = false;
        bool skip_blur_chain = false;
        bool has_deferred_clear = false;
        VAddr base_va = 0;
        u32 slice = 0;
        VAddr slice_va = 0;
        u32 slice_bytes = 0;
        u32 dim = 0;
        u32 num_last_draws = 0;
        u64 geom_digest = 0;
        u64 last_render_frame = 0;
        u64 last_seen_frame = 0;
        Rasterizer::DirectClearColorArgs deferred_clear_args{};
    };

    struct ScratchLink {
        bool valid = false;
        bool skip = false;
        bool from_blur = false;
        u32 face_idx = 0;
        VAddr base_va = 0;
        u32 slice = 0;
        VAddr slice_va = 0;
        u32 dim = 0;
        u64 last_frame = 0;
    };

    struct BufferedStage {
        VAddr pgm_base = 0;
        std::vector<u32> user_data;
        std::vector<u32> flattened_ud_buf;
    };

    struct BufferedDraw {
        const GraphicsPipeline* pipeline = nullptr;
        bool is_indexed = false;
        u32 index_offset = 0;
        u32 num_stages = 0;
        std::array<BufferedStage, Shader::MaxStageTypes> stages{};
        std::vector<Gcn::Buffer> vsharps;
        StageBChunks regs_chunks{};
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent{};
        Gcn::CbDbExtent db_extent{};
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> cb_uid{};
        u16 db_uid = 0;
    };

    struct BoundRange {
        VAddr addr = 0;
        u64 size = 0;
    };

    struct PendingFace {
        bool active = false;
        bool unmemoizable = false;
        bool has_depth_clear = false;
        u32 face_idx = 0;
        VsmTargetKey target{};
        VAddr db_addr = 0;
        u32 db_slice = 0;
        u64 running_digest = 0;
        Rasterizer::DirectClearDepthStencilArgs depth_clear_args{};
        Gcn::RegDirty saved_initial_dirty{};
        StageBChunks saved_initial_chunks{};
        u32 num_draws = 0;
        std::array<BufferedDraw, kMaxDrawsPerFace> draws{};
        std::vector<BoundRange> bound_ranges;
    };

    void DrainRetiredRanges() noexcept;
    void InvalidateOverlappingMemory(VAddr addr, u64 byte_size) noexcept;
    void MarkKnownVsmBase(VAddr base_va) noexcept;

    [[nodiscard]] s32 FindFaceIndex(const VsmTargetKey& key) const noexcept;
    [[nodiscard]] u32 GetOrAllocateFaceIndex(const VsmTargetKey& key, u64 cur_frame) noexcept;
    void EnrichFaceKey(FaceEntry& f, const VsmTargetKey& key) noexcept;

    void SetScratchLink(const VsmTargetKey& scratch_key, u32 face_idx, bool skip,
                        bool from_blur, u64 cur_frame) noexcept;
    [[nodiscard]] const ScratchLink* FindScratchLink(const VsmTargetKey& key) const noexcept;

    void EmitClearColorNow(Rasterizer& rasterizer,
                           const Rasterizer::DirectClearColorArgs& args);
    void EmitClearDepthNow(Rasterizer& rasterizer,
                           const Rasterizer::DirectClearDepthStencilArgs& args);
    void FlushPendingDepthClear(Rasterizer& rasterizer);
    void FinalizeDeferredClear(Rasterizer& rasterizer, u32 face_idx);
    void FlushAllDeferredClears(Rasterizer& rasterizer);

    void FinishPendingFace(Rasterizer& rasterizer, bool force_miss = false);
    void MaybeLogTelemetry() noexcept;

    std::array<FaceEntry, kMaxFaces> faces{};
    std::array<ScratchLink, kMaxScratchLinks> scratch_links{};
    std::array<VAddr, 32> known_vsm_bases{};
    u32 known_vsm_cursor = 0;

    PendingFace pending_face{};
    VsmTargetKey passthrough_target{};
    bool has_pending_depth_clear = false;
    Rasterizer::DirectClearDepthStencilArgs pending_depth_clear_args{};
    u32 num_deferred_clears = 0;
    bool replaying = false;
    u64 last_periodic_refresh_frame = ~0ULL;
    u64 last_seen_retire_seq = 0;

    // Diagnostics & telemetry:
    bool first_active_logged = false;
    bool summary_logged = false;
    u64 stat_faces_evaluated = 0;
    u64 stat_face_hits = 0;
    u64 stat_face_misses = 0;
    u64 stat_empty_hits = 0;
    u64 stat_unmemoizable = 0;
    u64 stat_geom_draws_saved = 0;
    u64 stat_blur_draws_saved = 0;
    u64 stat_copies_saved = 0;
    u64 stat_clears_saved = 0;
};

} // namespace Vulkan
