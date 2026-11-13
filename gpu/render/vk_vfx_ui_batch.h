// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 11 (Target C): Soft-Particle VFX, Scaleform GFx 2D HUD & Small-Draw Run Batcher on Stage A.
//
// Coalesces consecutive compatible small draws (`<= 256` indices, non-instanced) in strict
// painter's order with zero visual difference and zero Stage-B mutex contention:
//   1. Scaleform GFx 2D TexGen + CxForm (`VS 0x24042a9b`): packs each quad's 24-dword `ssbo_1`
//      (`[cx_mul, cx_add, pos_row0, pos_row1, tex_row0, tex_row1]`) sequentially into `ConstantRing`
//      and encodes `-float(quad_idx + 1)` in `vs_in_attr1.w` (`vs_24042a9b.vert` native dual-path).
//   2. Scaleform GFx 2D UV Sprite (`VS 0xe94b5b06`) & Stencil Mask (`VS 0xa400228b`): pre-transforms
//      per-quad 2x4 matrix + CxForm on the CPU and binds canonical identity `ssbo_1` / `ssbo_2`,
//      preserving shader-side `push_data.motion_param` ultrawide UI edge stretch.
//   3. FFX Soft-Particle Tessellation (`LS 0x4cd21a0f` + `HS` + `DS`) & Quad (`VS 0x41ae600f`)
//      paired with `PS 0x71c96598` / `0x5e224cb6` / `0xd729c469` / `0x20739372`: bakes per-emitter
//      `PSCommonBuffer` `tint` (`dwords[36..39]`) into `vs_in_attr1` (`Format32_32_32_32 Float`)
//      with canonical `tint = (1, 1, 1, 1)` and compares only shader-read CB dwords across emitters
//      (ignoring unused per-emitter matrices in `PSCommonBuffer[0..15]` and `DSCommonBuffer` gaps).
//   4. Generic consecutive small-draw runs (`<= 256` indices) sharing identical `GraphicsPipeline*`,
//      active render-target/blend/depth/stencil/raster/scissor registers, T#/S# descriptors, and CB
//      bytes: expands indexed/non-indexed vertices into `ConstantRing` while preserving original
//      `vsharp` attribute formats, swizzles, and winding parity.
#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"
#include "gcn/regs.h"
#include "gcn/sharps.h"
#include "render/rasterizer.h"
#include "render/vk_light_tile_batch.h"

namespace Vulkan {

class VfxUiBatcher {
public:
    static constexpr u32 kMaxSmallDrawIndices = 256u;
    static constexpr u32 kMaxMergedVertices = 16384u;
    static constexpr u32 kMaxScaleform2404Quads = 2048u;
    static constexpr size_t kMaxAttrs = 16u;
    static constexpr size_t kMaxNonSpecialBufs = 8u;
    static constexpr size_t kSmallStagingMaxBytes = 4096u;
    static constexpr size_t kCpuStagingSets = 128u;
    static constexpr size_t kCpuStagingWays = 4u;
    static constexpr size_t kCpuStagingSlots = kCpuStagingSets * kCpuStagingWays;
    static constexpr size_t kLargeStagingSlots = 32u;
    static constexpr u32 kCoarseShift = 18u;          // 256 KiB coarse invalidation granule
    static constexpr size_t kCoarseBuckets = 262144u; // covers 64 GiB directly (`1 MiB` table)
    static constexpr size_t kStageBWords =
        Gcn::RegDirty::StageBNumChunks * Gcn::RegDirty::StageBChunkWords;

    using StageBChunks = std::array<u32, kStageBWords>;

    VfxUiBatcher();
    ~VfxUiBatcher() = default;

    VfxUiBatcher(const VfxUiBatcher&) = delete;
    VfxUiBatcher& operator=(const VfxUiBatcher&) = delete;

    [[nodiscard]] static bool Enabled() noexcept;

    [[nodiscard]] bool HasPendingDraws() const noexcept {
        return open.active && !flushing;
    }
    [[nodiscard]] const GraphicsPipeline* CurrentPipeline() const noexcept {
        return open.active ? open.pipeline : nullptr;
    }
    [[nodiscard]] bool IsFlushing() const noexcept {
        return flushing;
    }

    /// Records `kCopyFlagKnownCpuSource` buffer copies of ANY size on Stage A (redirecting
    /// native-owned dynamic VB/IB/CB `dst_va` reads directly to their live `auxCpuRing`/`cueHeap`
    /// `src_va` staging bytes with zero Stage-B locks).
    void NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept;

    /// Invalidates any Stage-A staging redirect overlapping `[addr, addr + size)`.
    void InvalidateCpuShadow(VAddr addr, u64 size) noexcept;

    /// Returns true if `task` is a `DirectVkRelease*LabelsBatchTask` or a `DirectVkCopyBatchTask`
    /// whose regions are all `kCopyFlagKnownCpuSource` (safe to queue without breaking an open
    /// `VfxUiBatcher` run because absorbed draws already snapshot their vertices & CBs on Stage A).
    [[nodiscard]] static bool IsBatchSafeTask(Rasterizer::OrderedTask task, const void* data,
                                              u32 size) noexcept;

    void NoteTaskFlush() noexcept {
        ++stat_task_flushes;
    }

    /// Attempts to absorb a draw into the open sequential batch (or flush the open batch and start
    /// a new one). Returns `true` if absorbed; returns `false` (after flushing any open batch) if
    /// the caller should execute the draw normally.
    [[nodiscard]] bool TryBatchDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                    const PreparedDraw* used_prepared, bool is_indexed,
                                    u32 index_offset);

    /// Emits the currently open sequential batch (if any) into `ConstantRing` + `PostDraw`.
    void Flush(Rasterizer& rasterizer);

private:
    enum class Mode : u8 {
        None = 0,
        Scaleform2404 = 1,
        ScaleformE94B = 2,
        ScaleformA400 = 3,
        FfxTintBaked = 4,
    };

    struct CpuStagingSlot {
        VAddr dst_va = 0;
        VAddr src_va = 0;
        u32 byte_size = 0;
        u32 last_use = 0;
        u32 coarse_e0 = 0;
        u32 coarse_e1 = 0;
    };

    struct StageBufCapture {
        u32 sharp_idx = 0;
        std::vector<u8> bytes;
    };

    struct StageCapture {
        const Shader::Info* info = nullptr;
        u64 pgm_hash = 0;
        VAddr pgm_base = 0;
        std::vector<u32> user_data;
        std::vector<u32> flattened_ud_buf;
        u32 num_bufs = 0;
        std::array<StageBufCapture, kMaxNonSpecialBufs> bufs{};
        std::vector<Gcn::Image> images;
        std::vector<Gcn::Sampler> samplers;
    };

    struct AttrStream {
        Gcn::Buffer out_vsharp{};
        u32 stride = 0;
        std::vector<u8> bytes;
        std::array<u8, 64> last_vtx{};
    };

    struct OpenBatch {
        bool active = false;
        Mode mode = Mode::None;
        const GraphicsPipeline* pipeline = nullptr;
        Gcn::PrimitiveType prim_type = Gcn::PrimitiveType::TriangleList;
        u32 patch_cp_count = 1u;
        u32 num_draws = 0u;
        u32 num_verts = 0u;
        u32 sf2404_quads = 0u;
        u32 num_attrs = 0u;
        u32 num_stages = 0u;
        u32 vs_slot = 0u;
        u32 ffx_ps_slot = 0u;
        StageBChunks base_chunks{};
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent{};
        Gcn::CbDbExtent db_extent{};
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> cb_uid{};
        u16 db_uid = 0;
        std::array<AttrStream, kMaxAttrs> attrs{};
        std::array<StageCapture, Shader::MaxStageTypes> stages{};
        std::vector<u32> sf2404_cb_dwords;

        void Reset() noexcept {
            active = false;
            mode = Mode::None;
            pipeline = nullptr;
            num_draws = 0u;
            num_verts = 0u;
            sf2404_quads = 0u;
            num_attrs = 0u;
            num_stages = 0u;
            vs_slot = 0u;
            ffx_ps_slot = 0u;
            sf2404_cb_dwords.clear();
            for (auto& a : attrs) {
                a.bytes.clear();
                a.stride = 0u;
            }
            for (auto& st : stages) {
                st.info = nullptr;
                st.user_data.clear();
                st.flattened_ud_buf.clear();
                for (u32 b = 0; b < st.num_bufs; ++b) {
                    st.bufs[b].bytes.clear();
                }
                st.num_bufs = 0u;
                st.images.clear();
                st.samplers.clear();
            }
        }
    };

    [[nodiscard]] static size_t SmallStagingSet(VAddr page4k) noexcept {
        u64 x = page4k * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 32;
        return static_cast<size_t>(x & (kCpuStagingSets - 1u));
    }

    [[nodiscard]] u32 CoarseEpochAt(VAddr va) const noexcept {
        return coarse_epoch_256k[static_cast<size_t>((va >> kCoarseShift) & (kCoarseBuckets - 1u))] +
               coarse_wipe_bias;
    }

    void RecomputeLargeStagingBounds() noexcept;

    [[nodiscard]] const u8* ResolveCpuReadableRange(Rasterizer& rasterizer, VAddr addr,
                                                    u64 byte_size) noexcept;

    std::vector<CpuStagingSlot> small_staging_map;
    std::array<CpuStagingSlot, kLargeStagingSlots> large_staging{};
    std::vector<u32> coarse_epoch_256k;
    VAddr large_min_va = ~VAddr{0};
    VAddr large_max_va = 0;
    u32 large_active_count = 0;
    u32 staging_clock = 0;
    u32 coarse_clock = 0;
    u32 coarse_wipe_bias = 0;

    OpenBatch open{};
    bool flushing = false;

    u64 stat_candidates_seen = 0;
    u64 stat_ffx_draws_in = 0;
    u64 stat_ffx_draws_saved = 0;
    u64 stat_ui_draws_in = 0;
    u64 stat_ui_draws_saved = 0;
    u64 stat_eq_draws_in = 0;
    u64 stat_eq_draws_saved = 0;
    u64 stat_batches_emitted = 0;
    u64 stat_single_passthrough = 0;
    u64 stat_task_flushes = 0;
    u64 fb_regs = 0;
    u64 fb_layout = 0;
    u64 fb_cb = 0;
    u64 fb_vb = 0;
    u64 fb_ib = 0;
    bool first_active_logged = false;
    bool summary_logged = false;
};

} // namespace Vulkan
