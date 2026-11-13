// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 10 (Target B): Stage-A Stationary Object-Velocity Culling (`Pass 0x2e`, `1280x720`
// `R8G8B8A8Unorm` `velWrap`) for Bloodborne (`CUSA03173`).
//
// Mathematical invariant (`sub_22a0a30` + `GXYebisPostProcessDirectorAndLutCompute.cpp:2546`):
//   1. `velWrap` (`viewBytes + 0xce0` / `*(viewBytes + 0xcf8)`) is cleared to `0x007f7fff`
//      (`vel.xy = 0.5` = zero object velocity, `Alpha = 0.0`).
//   2. `Pass 0x2e` scratch depth (`viewBytes + 0xd20`, `1280x720 D32SfloatS8Uint`) is pre-filled
//      by downsampling the full `2560x1440` G-Buffer depth (`sub_e645d0`), which already contains
//      the exact depth of all stationary characters, corpses, crucified beasts, and FLVER props.
//   3. Where `velWrap.a == 0.0`, Yebis 2 reconstructs exact camera motion vectors from scene depth
//      and `(curViewProj, prevViewProj)`, while moving enemies behind a skipped stationary corpse
//      still fail the `1280x720` depth test against the downsampled G-Buffer depth.
//   4. On rare frames (`< 0.1%`) where feedback/radial/silhouette blur (`viewBytes[0x2970]`,
//      `viewBytes[0x2971]`, `viewBytes + 0x2960`) reads `velWrap` as a silhouette mask,
//      `NoteSilhouetteMaskActiveGlobal()` holds culling off for 150 ms across the pipeline.
#pragma once

#include <array>
#include <vector>

#include "common/types.h"
#include "render/rasterizer.h"

namespace Vulkan {

class ObjectVelocityCuller {
public:
    static constexpr u32 kMaxTinyPivot = 464u;   // `<= 464 B`: `model_cb` (`416 B`), `rigid_pair` (`96 B`), 1..9-bone CBs
    static constexpr u32 kTinySlotStride = 512u; // power-of-two slot stride
    static constexpr size_t kTinySets = 64u;
    static constexpr size_t kTinyWays = 16u;
    static constexpr size_t kTinySlots = kTinySets * kTinyWays; // 1024 slots = 512 KiB

    static constexpr u32 kMaxSmallBytes = 2048u; // `465..2048 B`: `0x540` (`1344 B`) character/corpse palettes (immune to tiny CB churn!)
    static constexpr size_t kSmallSets = 128u;
    static constexpr size_t kSmallWays = 16u;
    static constexpr size_t kSmallSlots = kSmallSets * kSmallWays; // 2048 slots = 4 MiB

    static constexpr u32 kMaxLargeBytes = 32768u; // `2049..32768 B`: unified character/boss skeletons up to 682 bones
    static constexpr size_t kLargeSets = 32u;
    static constexpr size_t kLargeWays = 16u;
    static constexpr size_t kLargeSlots = kLargeSets * kLargeWays; // 512 slots = 16 MiB

    static constexpr size_t kPairMemoSize = 512u; // power-of-two direct/2-way pair memo
    static constexpr u32 kCoarseShift = 18u;      // 256 KiB coarse invalidation granule (`> 32 KiB` writes)
    static constexpr size_t kCoarseBuckets = 262144u; // covers 64 GiB directly (`1 MiB` table)

    ObjectVelocityCuller();
    ~ObjectVelocityCuller() = default;

    ObjectVelocityCuller(const ObjectVelocityCuller&) = delete;
    ObjectVelocityCuller& operator=(const ObjectVelocityCuller&) = delete;

    [[nodiscard]] static bool Enabled() noexcept;
    [[nodiscard]] static float EpsilonThreshold() noexcept;

    /// Called from guest engine (`sub_229e990`) when feedback/radial/silhouette blur is active.
    static void NoteSilhouetteMaskActiveGlobal() noexcept;
    [[nodiscard]] static bool IsSilhouetteMaskActive() noexcept;

    /// Records CPU-source constant/palette buffer copies (`16..24576 B`) on Stage A.
    void NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept;

    /// Invalidates any cached CPU shadow overlapping `[addr, addr + size)` on Stage A.
    void InvalidateCpuShadow(VAddr addr, u64 size) noexcept;

    /// Stage-A draw hook called from `Rasterizer::Draw` before `PostDraw` / `DrawRecord`.
    /// Returns `true` if the draw is a `Pass 0x2e` object-velocity draw whose object transform /
    /// bone palette delta vs the previous frame is `<= EpsilonThreshold()` (`0` pixels of object
    /// motion blur in `R8G8B8A8Unorm`), having already decremented `BbStats::draws`.
    [[nodiscard]] bool TryCullDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                   bool is_indexed);

private:
    struct ShadowMeta {
        VAddr dst_va = 0;
        u32 byte_size = 0;
        u32 last_use = 0;
        u64 upload_gen = 0;
        u32 coarse_e0 = 0;
        u32 coarse_e1 = 0;
    };
    static_assert(sizeof(ShadowMeta) == 32);

    struct PairMemoEntry {
        VAddr cur_addr = 0;
        VAddr prev_addr = 0;
        u64 cur_gen = 0;
        u64 prev_gen = 0;
        u32 byte_off = 0;
        u32 cmp_bytes = 0;
        u8 stationary = 0;
        u8 valid = 0;
        u16 pad = 0;
        u32 pad2 = 0;
    };
    static_assert(sizeof(PairMemoEntry) == 48);

    [[nodiscard]] static size_t TinyPageSet(VAddr page512) noexcept {
        u64 x = page512 * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 32;
        return static_cast<size_t>(x & (kTinySets - 1u));
    }

    [[nodiscard]] static size_t SmallPageSet(VAddr page2k) noexcept {
        u64 x = page2k * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 32;
        return static_cast<size_t>(x & (kSmallSets - 1u));
    }

    [[nodiscard]] static size_t LargePageSet(VAddr page32k) noexcept {
        u64 x = page32k * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 32;
        return static_cast<size_t>(x & (kLargeSets - 1u));
    }

    [[nodiscard]] const u8* LookupCpuShadow(VAddr addr, u32 min_bytes, u32 size_hint, u32* out_rem,
                                            u64* out_gen) noexcept;

    [[nodiscard]] u32 CoarseEpochAt(VAddr va) const noexcept {
        return coarse_epoch_256k[static_cast<size_t>((va >> kCoarseShift) & (kCoarseBuckets - 1u))] +
               coarse_wipe_bias;
    }

    std::vector<ShadowMeta> tiny_meta;
    std::vector<u8> tiny_bytes;
    std::vector<ShadowMeta> small_meta;
    std::vector<u8> small_bytes;
    std::vector<ShadowMeta> large_meta;
    std::vector<u8> large_bytes;
    std::vector<u32> coarse_epoch_256k;
    std::array<PairMemoEntry, kPairMemoSize> pair_memo{};

    u32 shadow_clock = 0;
    u32 coarse_clock = 0;
    u32 coarse_wipe_bias = 0;
    u64 shadow_upload_gen = 0;

    // Telemetry counters:
    u64 stat_seen = 0;
    u64 stat_culled = 0;
    u64 stat_moving = 0;
    u64 stat_cloth_kept = 0;
    u64 stat_pair_hits = 0;
    u64 stat_fallbacks = 0;
    u64 fb_silhouette = 0;
    u64 fb_regs = 0;
    u64 fb_layout = 0;
    u64 fb_cb_cur = 0;
    u64 fb_cb_prev = 0;
    u64 fb_inst = 0;
    u64 last_unk_vs = 0;
    u32 last_reg_bits = 0;
    bool first_summary_logged = false;
};

} // namespace Vulkan
