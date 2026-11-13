// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 8: Deferred local-light tile quad batching (`Filter\GXLightAcc.vpo`, `0x000000008747a367`)
// on Stage A. Coalesces the ~722 per-frame 4-vertex scissor-clipped light tile draws into ~8..16
// merged `TriangleStrip` draws grouped by `(GraphicsPipeline*, ssbo_1, PS T#/S#)` with zero visual
// difference (100% exact per-tile `[x0, x1) x [y0, y1)` pixel coverage, clip-space `z_min`,
// per-tile `depth_bounds = [z_min, z_max]`, and per-tile `cbLightAccLightParam` in `ssbo_2`).
#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"
#include "gcn/regs.h"
#include "gcn/sharps.h"
#include "render/rasterizer.h"

namespace Vulkan {

struct BatchedLightStageOverride {
    VAddr pgm_base = 0;
    std::span<const u32> user_data{};
    std::span<const u32> flattened_ud_buf{};
    std::span<const Rasterizer::RingBinding> ring_bindings{};
    bool collect_ring_bindings = false;
};

struct BatchedLightDrawOverride {
    std::span<const BatchedLightStageOverride> stages{};
    std::span<const Gcn::Buffer> vsharps{};
    bool note_guest_vb_ib_reads = false;
};

class LightTileBatcher {
public:
    static constexpr u32 kMaxShadowBytes = 2048u;
    static constexpr size_t kShadowSets = 256u;
    static constexpr size_t kShadowWays = 16u;
    static constexpr size_t kShadowSlots = kShadowSets * kShadowWays;
    static constexpr size_t kMaxGroupsPerPass = 32u;
    static constexpr u32 kMaxGroupSsbo2Dwords = 60000u * 4u; // <= 960 KiB, fits in exact f32 tile_v4_idx
    static constexpr size_t kStageBWords =
        Gcn::RegDirty::StageBNumChunks * Gcn::RegDirty::StageBChunkWords;

    using StageBChunks = std::array<u32, kStageBWords>;

    struct alignas(8) TileVertex {
        float x;
        float y;
        float z;
        float w;
        float u;
        float v;
    };
    static_assert(sizeof(TileVertex) == 24);

    LightTileBatcher();
    ~LightTileBatcher() = default;

    LightTileBatcher(const LightTileBatcher&) = delete;
    LightTileBatcher& operator=(const LightTileBatcher&) = delete;

    [[nodiscard]] static bool Enabled() noexcept;
    [[nodiscard]] static bool IsBatchedLightAccPsHash(u64 pgm_hash) noexcept;

    [[nodiscard]] bool HasPendingTiles() const noexcept {
        return pending_tiles != 0 && !flushing;
    }
    [[nodiscard]] bool IsFlushing() const noexcept {
        return flushing;
    }

    /// Records CPU-source constant buffer updates from `DirectVk::Op::CopyBufferBatch` on Stage A.
    void NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept;

    /// Invalidates any Stage-A CPU shadow slot overlapping `[addr, addr + size)`.
    void InvalidateCpuShadow(VAddr addr, u64 size) noexcept;

    /// Looks up a CPU-source constant buffer shadow (`<= 2048 B`) recorded on Stage A.
    [[nodiscard]] const u8* LookupCpuShadow(VAddr addr, u32 size,
                                            u32* out_entry_remaining = nullptr) noexcept;

    /// Returns true if `task` is a small (`<= 2048` B) CPU-source `DirectVkCopyBatchTask` or a
    /// staging-pool release label batch (`DirectVkRelease*LabelsBatchTask`) that does not require
    /// flushing open light-tile batches.
    [[nodiscard]] static bool IsBatchSafeCpuConstantCopyTask(Rasterizer::OrderedTask task,
                                                             const void* data,
                                                             u32 size) noexcept;

    void NoteTaskFlush() noexcept { ++stat_task_flushes; }

    /// Attempts to absorb a `vs_8747a367` local-light tile draw on Stage A.
    [[nodiscard]] bool TryBatchDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                    const PreparedDraw* used_prepared, bool is_indexed,
                                    u32 index_offset);

    /// Emits all accumulated `BatchGroup`s in strict group-creation order to `draw_pipe`.
    void Flush(Rasterizer& rasterizer);

private:
    static constexpr u32 kCoarseShift = 18u; // 256 KiB coarse invalidation granules
    static constexpr size_t kCoarseBuckets = 262144u; // covers 64 GiB directly (1 MiB on heap)

    struct ShadowMeta {
        VAddr dst_va = 0;
        u32 byte_size = 0;
        u32 last_use = 0;
        u32 coarse_e0 = 0;
        u32 coarse_e1 = 0;
    };
    static_assert(sizeof(ShadowMeta) == 24);

    struct StageSnapshot {
        VAddr pgm_base = 0;
        u32 non_special_buf_indices[2]{0, 0};
        u32 num_non_special_bufs = 0;
        std::vector<u32> user_data;
        std::vector<u32> flattened_ud_buf;
    };

    struct BatchGroup {
        const GraphicsPipeline* pipeline = nullptr;
        u64 resource_hash = 0;
        u32 ssbo1_bytes = 0;
        std::array<u8, kMaxShadowBytes> ssbo1{};
        std::vector<Gcn::Image> ps_images;
        std::vector<Gcn::Sampler> ps_samplers;
        std::array<Gcn::Buffer, 2> orig_vsharps{};
        std::array<u32, 2> attr_semantics{0, 1};
        u32 num_stages = 0;
        std::array<StageSnapshot, Shader::MaxStageTypes> stages{};
        StageBChunks base_chunks{};
        Gcn::Scissor union_screen_scissor{};
        Gcn::ViewportScissor union_vport_scissor0{};
        bool has_vport_scissor = false;
        u32 tile_count = 0;
        TileVertex last_t3{};
        std::vector<TileVertex> vertices;
        std::vector<u32> ssbo2_dwords;

        void Reset() noexcept {
            pipeline = nullptr;
            resource_hash = 0;
            ssbo1_bytes = 0;
            ps_images.clear();
            ps_samplers.clear();
            num_stages = 0;
            has_vport_scissor = false;
            tile_count = 0;
            vertices.clear();
            ssbo2_dwords.clear();
        }
    };

    [[nodiscard]] u32 CoarseEpochAt(VAddr addr) const noexcept {
        return coarse_epoch_256k[(addr >> kCoarseShift) & (kCoarseBuckets - 1u)] +
               coarse_wipe_bias;
    }

    std::vector<ShadowMeta> shadow_meta;
    std::vector<u8> shadow_bytes;
    std::vector<u32> coarse_epoch_256k;
    VAddr shadow_min_va = ~VAddr{0};
    VAddr shadow_max_va = 0;
    u32 shadow_clock = 0;
    u32 coarse_clock = 0;
    u32 coarse_wipe_bias = 0;

    bool pass_active = false;
    bool pass_commutative_blend = false;
    bool flushing = false;
    bool first_active_logged = false;
    bool summary_logged = false;
    u32 pending_tiles = 0;
    u32 active_groups = 0;
    StageBChunks pass_norm_chunks{};
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> pass_cb_extent{};
    Gcn::CbDbExtent pass_db_extent{};
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> pass_cb_uid{};
    u16 pass_db_uid = 0;
    std::array<BatchGroup, kMaxGroupsPerPass> groups{};

    // Diagnostic counters (1-line summary after 2,000 tiles + optional `BB_LIGHT_TILE_BATCH_LOG=1`)
    u64 stat_seen = 0;
    u64 stat_tiles_batched = 0;
    u64 stat_tiles_empty = 0;
    u64 stat_batches_emitted = 0;
    u64 stat_task_flushes = 0;
    u64 stat_fallbacks = 0;
    u64 fb_regs = 0;
    u64 fb_layout = 0;
    u64 fb_cb0 = 0;
    u64 fb_ssbo1 = 0;
    u64 fb_ssbo2 = 0;
    u64 fb_vb = 0;
    u64 fb_quad = 0;
};

} // namespace Vulkan
