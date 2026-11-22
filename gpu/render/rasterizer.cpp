// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, the draw and dispatch entry points: pipeline selection, then recording here or on the
// draw recording thread (raster_pipe.cpp), and the per-frame bookkeeping.

#include <mutex>
#include <set>
#include <string>
#include <filesystem>
#include "render/raster_internal.h"
#include "render/shader_hle.h"
#include "render/vk_light_tile_batch.h"
#include "render/vk_mt_direct_recorder.h"
#include "render/vk_vsm_shadow_memo.h"
#include "render/vk_obj_vel_cull.h"
#include "render/vk_vfx_ui_batch.h"
#include "render/vk_native_postfx.h"
#include "shader/backend/spirv/native_shaders.h"
#include "shim/bbport_frame_rate.h"
#include "shim/bbport_settings.h"
#include "textures/direct_textures.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"
#include <vk_mem_alloc.h>

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end);

namespace Vulkan {

namespace {
std::atomic<u32> g_draws_in_frame{0};
std::atomic<bool> g_mid_frame_flushed{false};
std::atomic<u64> g_frame_first_draw_ns{0};
u64 tls_avg_decode_ns = 0;
u64 tls_avg_gpu_busy_ns = 0;
u64 tls_prev_hw_busy_ns = 0;
u64 tls_prev_hw_frames = 0;

inline u64 SteadyNowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
} // namespace

/// Block 10 + Phase 4.2: Hardware Temporal Scene Occlusion Culler (`Stage B` + `GXWorker:0..5`
/// lock-free MT `VkQueryPool` blocks, `+0 B` `PT_TLS`).
struct SceneOcclusionCuller {
    static constexpr u32 NumSlices = 8;
    static constexpr u32 SliceQueries = 1024;
    static constexpr u32 MtQueryBlocks = 128;
    static constexpr u32 MtQueriesPerBlock = 256;
    static constexpr u32 TableBuckets = 1024;
    static constexpr u32 BucketWays = 8;
    static constexpr u32 TotalEntries = TableBuckets * BucketWays;
    static constexpr u32 SkelProbeBuckets = 4;
    static constexpr float kMinSkelCullDistSq = 36.0f;   // >= 6.0m from camera (never player/melee/near decals)
    static constexpr float kSkelTrackDistSq = 1.00f;     // 1.0m per-frame 3D anchor tracking radius
    static constexpr float kSkelSameSpotDistSq = 0.04f;  // 0.2m same-frame duplicate threshold
    static constexpr u16 kMinZeroStreak = 5u;            // >= 5 consecutive occluded probes before skip
    static constexpr u32 kMinCullIndices = 72u;          // Never cull 36-index decal cubes / tiny floor quads

    [[nodiscard]] static bool IsSafeOpaqueGBufferDraw(const Gcn::Regs& regs, u32 mrt_mask,
                                                      u32 expected_mrt_mask) noexcept {
        if (mrt_mask != expected_mrt_mask || !regs.depth_control.depth_enable ||
            !regs.depth_control.depth_write_enable ||
            regs.depth_render_control.depth_clear_enable ||
            regs.depth_render_control.stencil_clear_enable) {
            return false;
        }
        // Never cull or reorder depth-biased decals / coplanar floor overlays (blood puddles, grime).
        if (regs.polygon_control.NeedsBias()) {
            return false;
        }
        // Reject if ANY of the 6 G-Buffer MRTs has blending enabled (not just CB0).
        for (u32 cb = 0; cb < 6u; ++cb) {
            if (regs.blend_control[cb].enable) {
                return false;
            }
        }
        // Reject non-standard depth compare ops (e.g. Equal/Always coplanar decal passes).
        const auto df = regs.depth_control.depth_func;
        if (df != Gcn::CompareFunc::Less && df != Gcn::CompareFunc::LessEqual &&
            df != Gcn::CompareFunc::Greater && df != Gcn::CompareFunc::GreaterEqual) {
            return false;
        }
        // Reject stencil-tested draws (decal / mask-gated passes).
        if (regs.depth_control.stencil_enable &&
            (regs.depth_control.stencil_ref_func != Gcn::CompareFunc::Always ||
             (regs.depth_control.backface_enable &&
              regs.depth_control.stencil_bf_func != Gcn::CompareFunc::Always))) {
            return false;
        }
        return true;
    }

    [[nodiscard]] static bool IsNearCameraWorldObject(const u8* palette_ptr, u32 palette_size,
                                                      u32 mat_byte_off, bool is_inst_gbuf_vs,
                                                      u32 num_instances, const u8* inst_cb_ptr,
                                                      u32 inst_cb_size, float cam_x, float cam_y,
                                                      float cam_z) noexcept {
        if (!palette_ptr || u64(mat_byte_off) + 48ull > u64(palette_size)) {
            return false;
        }
        const float* mf = reinterpret_cast<const float*>(palette_ptr + mat_byte_off);
        const float wx = mf[3];
        const float wy = mf[7];
        const float wz = mf[11];
        if ((wx != 0.0f || wy != 0.0f || wz != 0.0f) && std::isfinite(wx) && std::isfinite(wy) &&
            std::isfinite(wz)) {
            const float dx = wx - cam_x;
            const float dy = wy - cam_y;
            const float dz = wz - cam_z;
            if (dx * dx + dy * dy + dz * dz < kMinSkelCullDistSq) {
                return true;
            }
        }
        if (is_inst_gbuf_vs && num_instances > 1u && inst_cb_ptr && inst_cb_size > 0u) {
            const u32 check_n = std::min<u32>(num_instances, 4u);
            for (u32 k = 1u; k < check_n; ++k) {
                const u32 ioff = k * 208u + 80u;
                if (ioff + sizeof(u32) <= inst_cb_size) {
                    u32 ibone = 0u;
                    std::memcpy(&ibone, inst_cb_ptr + ioff, sizeof(u32));
                    if (u64(ibone) * 48ull + 48ull <= u64(palette_size)) {
                        const float* imf =
                            reinterpret_cast<const float*>(palette_ptr + ibone * 48u);
                        const float ix = imf[3];
                        const float iy = imf[7];
                        const float iz = imf[11];
                        if ((ix != 0.0f || iy != 0.0f || iz != 0.0f) && std::isfinite(ix) &&
                            std::isfinite(iy) && std::isfinite(iz)) {
                            const float dx = ix - cam_x;
                            const float dy = iy - cam_y;
                            const float dz = iz - cam_z;
                            if (dx * dx + dy * dy + dz * dz < kMinSkelCullDistSq) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    static constexpr size_t kMeshFilterSlots = 8192; // 16 KiB L1d active-occluded mesh refcounts

    struct Entry {
        std::atomic<u64> key{0};
        std::atomic<u32> last_seen_frame{0};
        std::atomic<u32> skip_until_frame{0};
        std::atomic<float> pos_x{0.0f};
        std::atomic<float> pos_y{0.0f};
        std::atomic<float> pos_z{0.0f};
        std::atomic<u16> zero_streak{0};
        std::atomic<u16> mesh_slot{0};
        std::atomic<u8> probe_in_flight{0};
        std::atomic<u8> multi_occ{0};
        std::atomic<u8> is_skel{0};
        std::atomic<u8> occ_active{0};
    };

    enum class MtBlockState : u8 {
        Free = 0,
        Recording = 1,
        ReadyToExecute = 2,
        Submitted = 3,
    };

    struct MtQueryBlock {
        std::atomic<u8> state{static_cast<u8>(MtBlockState::Free)};
        u32 used{0};
        u32 frame_tag{0};
        u64 submit_tick{0};
        std::array<u16, MtQueriesPerBlock> entry_idx{};
    };

    vk::UniqueQueryPool pool;
    vk::UniqueQueryPool mt_pool;
    bool initialized = false;
    bool enabled = true;
    bool stage_a_occ_cull = false;
    bool skel_occ_cull = true;
    bool mt_occ_cull = true;
    bool micro_shadow_cull = true;
    std::atomic<u32> cur_frame{16};
    std::atomic<u32> stage_a_frame{16};
    u32 cur_slice = 0;
    u32 gbuf_draws_in_frame = 0;
    u32 stage_a_gbuf_draws = 0;
    u32 log_frame_counter = 0;
    std::atomic<u64> mt_gbuf_frame_and_count{0};
    std::atomic<u32> mt_block_hint{0};
    u32 mt_last_collect_frame = 0;
    std::atomic<bool> camera_moving{false};
    std::atomic<bool> fast_camera_turn{false};
    std::atomic<u32> cam_updated_frame{0};
    std::array<std::atomic<float>, 3> pub_cam_pos{};
    std::array<std::atomic<float>, 3> pub_cam_fwd{};
    std::atomic<bool> pub_cam_valid{false};
    std::mutex cam_mutex;
    bool has_prev_cam = false;
    std::array<float, 3> prev_cam_pos{};
    std::array<float, 3> prev_cam_fwd{};

    std::array<u32, NumSlices> slice_used{};
    std::array<u64, NumSlices> slice_last_tick{};
    std::array<bool, NumSlices> slice_sealed{};
    std::array<std::array<u16, SliceQueries>, NumSlices> slice_entry_idx{};

    std::array<MtQueryBlock, MtQueryBlocks> mt_blocks{};
    std::array<Entry, TotalEntries> entries{};
    std::array<std::atomic<u16>, kMeshFilterSlots> active_mesh_occ{};

    std::atomic<u64> stat_gbuf_candidates{0};
    std::atomic<u64> stat_gbuf_skipped{0};
    std::atomic<u64> stat_stage_a_skipped{0};
    std::atomic<u64> stat_stage_b_fallbacks{0};
    std::atomic<u64> stat_vis_offload{0};
    std::atomic<u64> stat_gbuf_probed{0};
    std::atomic<u64> stat_skel_candidates{0};
    std::atomic<u64> stat_skel_skipped{0};
    std::atomic<u64> stat_multi_occ{0};
    std::atomic<u64> stat_shadow_skipped{0};
    std::atomic<u64> stat_light_candidates{0};
    std::atomic<u64> stat_light_skipped{0};

    // Phase 5.2c: Active-occluded L1 pre-filter + exact non-self-wiping Stage-A copy redirect +
    // 256-byte fine write-frame settled `DirectBuffers::shadow_dir` MRU.
    static constexpr size_t kCoarseBuckets = 65536;     // 256 KiB granules across 40-bit VA space
    static constexpr size_t kFineWriteBuckets = 65536;  // 256 B fine write-frame slots (256 KiB)
    static constexpr u32 kTinyRedirMax = 512u;
    static constexpr u32 kTinyRedirShift = 9u;
    static constexpr size_t kTinyRedirSets = 512;
    static constexpr size_t kTinyRedirWays = 4;
    static constexpr u32 kSmallRedirMax = 4096u;
    static constexpr u32 kSmallRedirShift = 12u;
    static constexpr size_t kSmallRedirSets = 256;
    static constexpr size_t kSmallRedirWays = 4;
    static constexpr u32 kLargeRedirMax = 65536u;
    static constexpr u32 kLargeRedirShift = 16u;
    static constexpr size_t kLargeRedirSets = 128;
    static constexpr size_t kLargeRedirWays = 4;
    static constexpr size_t kSettledMruSlots = 2048;

    struct RedirSlot {
        VAddr dst_va{0};
        VAddr src_va{0};
        u32 byte_size{0};
        u32 dst_e0{0};
        u32 dst_e1{0};
        u32 recorded_frame{0};
        u32 last_use{0};
        u8 src_is_ring{0};
    };

    struct SettledMruSlot {
        VAddr base_va{0};
        u64 raw_size{0};
        const u8* mirror_base{nullptr};
        u64 erase_gen{0};
    };

    std::vector<u32> coarse_epoch_256k;
    std::vector<u32> last_write_frame_256k;
    std::vector<u32> last_write_frame_256b;
    std::vector<RedirSlot> tiny_redir;
    std::vector<RedirSlot> small_redir;
    std::vector<RedirSlot> large_redir;
    std::vector<SettledMruSlot> settled_mru;
    SettledMruSlot last_settled_hit{};
    u32 redir_clock = 0;

    [[nodiscard]] static constexpr u16 ComputeMeshFilterSlot(VAddr index_address, u64 vs_hash,
                                                             u32 num_indices) noexcept {
        u64 h = index_address ^ (vs_hash * 0x9E3779B97F4A7C15ULL) ^
                (u64(num_indices) * 0xBF58476D1CE4E5B9ULL);
        h ^= h >> 32;
        h ^= h >> 16;
        return static_cast<u16>(h & (kMeshFilterSlots - 1u));
    }

    void SetEntryOccActive(Entry& e, bool active) noexcept {
        if (!stage_a_occ_cull) [[likely]] {
            return;
        }
        if (active) {
            if (e.occ_active.exchange(1u, std::memory_order_relaxed) == 0u) {
                const u16 slot =
                    e.mesh_slot.load(std::memory_order_relaxed) & u16(kMeshFilterSlots - 1u);
                active_mesh_occ[slot].fetch_add(1u, std::memory_order_relaxed);
            }
        } else {
            if (e.occ_active.exchange(0u, std::memory_order_relaxed) != 0u) {
                const u16 slot =
                    e.mesh_slot.load(std::memory_order_relaxed) & u16(kMeshFilterSlots - 1u);
                active_mesh_occ[slot].fetch_sub(1u, std::memory_order_relaxed);
            }
        }
    }

    [[nodiscard]] static constexpr size_t CoarseBucket(VAddr addr) noexcept {
        return static_cast<size_t>(((addr >> 18) ^ (addr >> 34)) & (kCoarseBuckets - 1));
    }
    [[nodiscard]] static constexpr size_t FineWriteBucket256B(u64 gran256) noexcept {
        u64 h = gran256 * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
        return static_cast<size_t>(h & (kFineWriteBuckets - 1));
    }
    [[nodiscard]] static constexpr size_t PageSetIdx(VAddr page, size_t num_sets) noexcept {
        u64 h = page * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
        return static_cast<size_t>(h & (num_sets - 1));
    }
    [[nodiscard]] static constexpr size_t SettledMruHash(VAddr addr) noexcept {
        return static_cast<size_t>(((addr >> 8) ^ (addr >> 17) ^ (addr >> 26)) &
                                   (kSettledMruSlots - 1));
    }

    [[nodiscard]] static bool IsReadableCpuSource(VAddr src_va, u32 byte_size,
                                                  bool& out_is_ring) noexcept {
        out_is_ring = false;
        if (!src_va || byte_size == 0u) {
            return false;
        }
        if (VideoCore::DirectBuffers::IsTransientCueHeap(src_va, byte_size) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(src_va, byte_size)) {
            out_is_ring = true;
            return true;
        }
        if (VideoCore::DirectBuffers::OverlapsNativeOwnedRange(src_va, byte_size)) {
            return false;
        }
        uintptr_t mapped_end = 0;
        return BbLookupMappedEnd(src_va, mapped_end) && byte_size <= (mapped_end - src_va);
    }

    void StampWriteFrames(VAddr addr, u64 size, u32 cur_f) noexcept {
        if (!addr || size == 0 || last_write_frame_256b.empty()) {
            return;
        }
        if (size <= 4096ULL) {
            const u64 g_lo = addr >> 8;
            const u64 g_hi = (addr + size - 1ULL) >> 8;
            for (u64 g = g_lo; g <= g_hi; ++g) {
                last_write_frame_256b[FineWriteBucket256B(g)] = cur_f;
            }
            return;
        }
        if (size > (16ULL << 20)) {
            std::fill(last_write_frame_256k.begin(), last_write_frame_256k.end(), cur_f);
            return;
        }
        const u64 first_p = addr >> 18;
        const u64 last_p = (addr + size - 1ULL) >> 18;
        for (u64 p = first_p; p <= last_p; ++p) {
            const size_t b = static_cast<size_t>((p ^ (p >> 16)) & (kCoarseBuckets - 1));
            last_write_frame_256k[b] = cur_f;
        }
    }

    void InvalidateRedirectRange(VAddr addr, u64 size) noexcept {
        if (!addr || size == 0 || tiny_redir.empty()) {
            return;
        }
        const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
        const u64 span = end - addr;
        if (span > kLargeRedirMax) {
            if (span > (16ULL << 20)) {
                for (u32& e : coarse_epoch_256k) {
                    ++e;
                }
            } else {
                const u64 first_p = addr >> 18;
                const u64 last_p = (end - 1ULL) >> 18;
                for (u64 p = first_p; p <= last_p; ++p) {
                    ++coarse_epoch_256k[static_cast<size_t>((p ^ (p >> 16)) &
                                                            (kCoarseBuckets - 1))];
                }
            }
            return;
        }
        const auto inv_tier = [&](RedirSlot* table, u32 max_bytes, u32 shift, size_t num_sets,
                                  size_t ways) noexcept {
            const VAddr p_lo = (addr >= max_bytes) ? ((addr - max_bytes) >> shift) : 0ULL;
            const VAddr p_hi = (end - 1ULL) >> shift;
            for (VAddr p = p_lo; p <= p_hi; ++p) {
                const size_t set_base = PageSetIdx(p, num_sets) * ways;
                for (size_t w = 0; w < ways; ++w) {
                    auto& s = table[set_base + w];
                    if (s.byte_size != 0u && s.dst_va < end && addr < (s.dst_va + s.byte_size)) {
                        s.byte_size = 0u;
                    }
                }
            }
        };
        inv_tier(tiny_redir.data(), kTinyRedirMax, kTinyRedirShift, kTinyRedirSets, kTinyRedirWays);
        inv_tier(small_redir.data(), kSmallRedirMax, kSmallRedirShift, kSmallRedirSets,
                 kSmallRedirWays);
        inv_tier(large_redir.data(), kLargeRedirMax, kLargeRedirShift, kLargeRedirSets,
                 kLargeRedirWays);
    }

    void NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept {
        if (!enabled || !stage_a_occ_cull || !raw_regions || count == 0u || count > 32u ||
            tiny_redir.empty()) {
            return;
        }
        const u32 cur_f = stage_a_frame.load(std::memory_order_relaxed);
        for (u32 i = 0; i < count; ++i) {
            Rasterizer::DirectCopyRegion reg{};
            std::memcpy(&reg, raw_regions + i * 6u, sizeof(reg));
            if (reg.dst_va == 0 || reg.byte_size == 0 ||
                !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(reg.dst_va, reg.byte_size)) {
                continue;
            }
            StampWriteFrames(reg.dst_va, reg.byte_size, cur_f);

            bool src_is_ring = false;
            if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource || reg.byte_size < 48u ||
                reg.byte_size > kLargeRedirMax ||
                !IsReadableCpuSource(reg.src_va, reg.byte_size, src_is_ring)) {
                InvalidateRedirectRange(reg.dst_va, reg.byte_size);
                continue;
            }

            RedirSlot* table = nullptr;
            size_t set_base = 0;
            size_t ways = 0;
            if (reg.byte_size <= kTinyRedirMax) {
                table = tiny_redir.data();
                set_base = PageSetIdx(reg.dst_va >> kTinyRedirShift, kTinyRedirSets) * kTinyRedirWays;
                ways = kTinyRedirWays;
            } else if (reg.byte_size <= kSmallRedirMax) {
                table = small_redir.data();
                set_base =
                    PageSetIdx(reg.dst_va >> kSmallRedirShift, kSmallRedirSets) * kSmallRedirWays;
                ways = kSmallRedirWays;
            } else {
                table = large_redir.data();
                set_base =
                    PageSetIdx(reg.dst_va >> kLargeRedirShift, kLargeRedirSets) * kLargeRedirWays;
                ways = kLargeRedirWays;
            }

            const u32 cur_e0 = coarse_epoch_256k[CoarseBucket(reg.dst_va)];
            const u32 cur_e1 = coarse_epoch_256k[CoarseBucket(reg.dst_va + reg.byte_size - 1u)];
            bool updated_in_place = false;
            for (size_t w = 0; w < ways; ++w) {
                auto& s = table[set_base + w];
                if (s.dst_va == reg.dst_va && s.byte_size == reg.byte_size && s.dst_e0 == cur_e0 &&
                    s.dst_e1 == cur_e1) {
                    s.src_va = reg.src_va;
                    s.recorded_frame = cur_f;
                    s.last_use = ++redir_clock;
                    s.src_is_ring = src_is_ring ? 1u : 0u;
                    updated_in_place = true;
                    break;
                }
            }
            if (updated_in_place) {
                continue;
            }

            InvalidateRedirectRange(reg.dst_va, reg.byte_size);
            size_t victim = 0;
            u32 oldest = ~0u;
            for (size_t w = 0; w < ways; ++w) {
                const auto& s = table[set_base + w];
                if (s.byte_size == 0u ||
                    s.dst_e0 != coarse_epoch_256k[CoarseBucket(s.dst_va)] ||
                    s.dst_e1 != coarse_epoch_256k[CoarseBucket(s.dst_va + s.byte_size - 1u)]) {
                    victim = w;
                    break;
                }
                if (s.last_use < oldest) {
                    oldest = s.last_use;
                    victim = w;
                }
            }
            auto& dst_s = table[set_base + victim];
            dst_s.dst_va = reg.dst_va;
            dst_s.src_va = reg.src_va;
            dst_s.byte_size = reg.byte_size;
            dst_s.dst_e0 = cur_e0;
            dst_s.dst_e1 = cur_e1;
            dst_s.recorded_frame = cur_f;
            dst_s.last_use = ++redir_clock;
            dst_s.src_is_ring = src_is_ring ? 1u : 0u;
        }
    }

    void InvalidateStageACpuShadow(VAddr addr, u64 size) noexcept {
        if (!enabled || !stage_a_occ_cull || !addr || size == 0 || last_write_frame_256k.empty()) {
            return;
        }
        StampWriteFrames(addr, size, stage_a_frame.load(std::memory_order_relaxed));
        InvalidateRedirectRange(addr, size);
    }

    [[nodiscard]] const u8* LookupStageARedirect(VAddr addr, u32 cb_size, u32 frame_tag) noexcept {
        if (cb_size == 0 || cb_size > kLargeRedirMax || tiny_redir.empty()) {
            return nullptr;
        }
        const auto probe_tier = [&](RedirSlot* table, u32 shift, size_t num_sets,
                                    size_t ways) noexcept -> const u8* {
            const VAddr page = addr >> shift;
            for (u32 step = 0; step < 2u; ++step) {
                if (step == 1u && page == 0u) {
                    break;
                }
                const size_t set_base = PageSetIdx(page - step, num_sets) * ways;
                for (size_t w = 0; w < ways; ++w) {
                    auto& s = table[set_base + w];
                    if (s.byte_size == 0u || addr < s.dst_va) {
                        continue;
                    }
                    const u64 sub_off = addr - s.dst_va;
                    if (sub_off >= s.byte_size || (s.byte_size - sub_off) < cb_size) {
                        continue;
                    }
                    if (s.dst_e0 != coarse_epoch_256k[CoarseBucket(s.dst_va)] ||
                        s.dst_e1 != coarse_epoch_256k[CoarseBucket(s.dst_va + s.byte_size - 1u)])
                        [[unlikely]] {
                        s.byte_size = 0u;
                        continue;
                    }
                    const u32 age = frame_tag - s.recorded_frame;
                    if (s.src_is_ring != 0u ? (age > 4u) : (age != 0u)) {
                        s.byte_size = 0u;
                        continue;
                    }
                    s.last_use = ++redir_clock;
                    return reinterpret_cast<const u8*>(s.src_va + sub_off);
                }
            }
            return nullptr;
        };

        if (cb_size <= kTinyRedirMax) {
            if (const u8* p =
                    probe_tier(tiny_redir.data(), kTinyRedirShift, kTinyRedirSets, kTinyRedirWays)) {
                return p;
            }
        }
        if (cb_size <= kSmallRedirMax) {
            if (const u8* p = probe_tier(small_redir.data(), kSmallRedirShift, kSmallRedirSets,
                                         kSmallRedirWays)) {
                return p;
            }
        }
        return probe_tier(large_redir.data(), kLargeRedirShift, kLargeRedirSets, kLargeRedirWays);
    }

    [[nodiscard]] const u8* LookupSettledMirror(const VideoCore::DirectBuffers::Manager& direct_bufs,
                                                VAddr addr, u32 cb_size, u32 frame_tag) noexcept {
        if (settled_mru.empty() || last_write_frame_256b.empty()) {
            return nullptr;
        }
        const u32 c_wf0 = last_write_frame_256k[CoarseBucket(addr)];
        const u32 c_wf1 = last_write_frame_256k[CoarseBucket(addr + cb_size - 1u)];
        if ((frame_tag - c_wf0) < 4u || (frame_tag - c_wf1) < 4u) {
            return nullptr;
        }
        const u32 check_span = std::min<u32>(cb_size, 1024u);
        const u64 g_lo = addr >> 8;
        const u64 g_hi = (addr + check_span - 1u) >> 8;
        for (u64 g = g_lo; g <= g_hi; ++g) {
            if ((frame_tag - last_write_frame_256b[FineWriteBucket256B(g)]) < 4u) {
                return nullptr;
            }
        }
        const u64 cur_gen = direct_bufs.ShadowDirEraseGen();
        if (last_settled_hit.erase_gen == cur_gen && last_settled_hit.mirror_base &&
            addr >= last_settled_hit.base_va) {
            const u64 sub_off = addr - last_settled_hit.base_va;
            if (sub_off < last_settled_hit.raw_size &&
                (last_settled_hit.raw_size - sub_off) >= cb_size) {
                return last_settled_hit.mirror_base + sub_off;
            }
        }
        const size_t h0 = SettledMruHash(addr);
        const size_t h1 = (h0 + 1u) & (kSettledMruSlots - 1);
        for (const size_t idx : {h0, h1}) {
            const auto& slot = settled_mru[idx];
            if (slot.erase_gen == cur_gen && slot.mirror_base && addr >= slot.base_va) {
                const u64 sub_off = addr - slot.base_va;
                if (sub_off < slot.raw_size && (slot.raw_size - sub_off) >= cb_size) {
                    last_settled_hit = slot;
                    return slot.mirror_base + sub_off;
                }
            }
        }
        VAddr base_va = 0;
        u64 raw_size = 0;
        const u8* mirror_base = nullptr;
        u64 out_gen = 0;
        if (!direct_bufs.TryLookupCpuShadowStageA(addr, cb_size, base_va, raw_size, mirror_base,
                                                  out_gen)) {
            return nullptr;
        }
        const SettledMruSlot fresh{base_va, raw_size, mirror_base, out_gen};
        if (settled_mru[h0].base_va != base_va || settled_mru[h0].erase_gen != out_gen) {
            settled_mru[h1] = settled_mru[h0];
        }
        settled_mru[h0] = fresh;
        last_settled_hit = fresh;
        return mirror_base + (addr - base_va);
    }

    void EnsureInit(const Instance& instance) {
        if (initialized) {
            return;
        }
        initialized = true;
        if (const char* e = std::getenv("BB_HW_OCCLUSION_CULL")) {
            enabled = (e[0] != '0');
        }
        if (const char* e = std::getenv("BB_STAGE_A_OCC_CULL")) {
            stage_a_occ_cull = (e[0] != '0');
        }
        if (const char* e = std::getenv("BB_SKEL_OCCLUSION_CULL")) {
            skel_occ_cull = (e[0] != '0');
        }
        if (const char* e = std::getenv("BB_MT_OCC_CULL")) {
            mt_occ_cull = (e[0] != '0');
        }
        if (const char* e = std::getenv("BB_MICRO_SHADOW_CULL")) {
            micro_shadow_cull = (e[0] != '0');
        }
        if (!enabled) {
            return;
        }
        const auto device = instance.GetDevice();
        const vk::QueryPoolCreateInfo info{
            .queryType = vk::QueryType::eOcclusion,
            .queryCount = NumSlices * SliceQueries,
        };
        auto res = device.createQueryPoolUnique(info);
        if (res.result == vk::Result::eSuccess) {
            pool = std::move(res.value);
            device.resetQueryPool(*pool, 0, NumSlices * SliceQueries);
        } else {
            enabled = false;
            return;
        }
        if (VkMtDirectRecorder::ConfiguredMode() > 0) {
            const vk::QueryPoolCreateInfo mt_info{
                .queryType = vk::QueryType::eOcclusion,
                .queryCount = MtQueryBlocks * MtQueriesPerBlock,
            };
            auto mt_res = device.createQueryPoolUnique(mt_info);
            if (mt_res.result == vk::Result::eSuccess) {
                mt_pool = std::move(mt_res.value);
                device.resetQueryPool(*mt_pool, 0, MtQueryBlocks * MtQueriesPerBlock);
            }
        }
        if (stage_a_occ_cull) {
            coarse_epoch_256k.assign(kCoarseBuckets, 1u);
            last_write_frame_256k.assign(kCoarseBuckets, 0u);
            last_write_frame_256b.assign(kFineWriteBuckets, 0u);
            tiny_redir.resize(kTinyRedirSets * kTinyRedirWays);
            small_redir.resize(kSmallRedirSets * kSmallRedirWays);
            large_redir.resize(kLargeRedirSets * kLargeRedirWays);
            settled_mru.resize(kSettledMruSlots);
        }
    }

    [[nodiscard]] u32 ActiveFrameTag() const noexcept {
        const u32 gf = BbStats::guest_frame_number.load(std::memory_order_relaxed);
        return (gf != 0u) ? gf : cur_frame.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u32 NextMtGBufOrdinal(u32 frame_tag) noexcept {
        u64 cur = mt_gbuf_frame_and_count.load(std::memory_order_relaxed);
        for (;;) {
            const auto cur_f = static_cast<u32>(cur >> 32u);
            const u32 next_c = (cur_f == frame_tag) ? (static_cast<u32>(cur) + 1u) : 1u;
            const u64 desired = (static_cast<u64>(frame_tag) << 32u) | next_c;
            if (mt_gbuf_frame_and_count.compare_exchange_weak(
                    cur, desired, std::memory_order_relaxed, std::memory_order_relaxed)) {
                return next_c;
            }
        }
    }

    void ApplyQuerySamplesToEntries(const u16* idx_arr, const u64* samples, u32 count,
                                    u32 ref_frame) noexcept {
        const bool fast_turn = fast_camera_turn.load(std::memory_order_relaxed);
        const bool cam_mov = camera_moving.load(std::memory_order_relaxed);
        for (u32 i = 0; i < count; ++i) {
            const u16 idx = idx_arr[i];
            Entry& e = entries[idx];
            if (e.multi_occ.load(std::memory_order_relaxed) != 0u) {
                e.zero_streak.store(0, std::memory_order_relaxed);
                e.skip_until_frame.store(0, std::memory_order_relaxed);
                SetEntryOccActive(e, false);
            } else if (samples[i] == 0) {
                u16 zs = e.zero_streak.load(std::memory_order_relaxed);
                if (zs < 32u) {
                    ++zs;
                    e.zero_streak.store(zs, std::memory_order_relaxed);
                }
                if (zs >= kMinZeroStreak && !fast_turn) {
                    const u32 stagger = static_cast<u32>(idx & 1u);
                    const bool skel = (e.is_skel.load(std::memory_order_relaxed) != 0u);
                    const u32 skip_len =
                        skel ? (cam_mov ? (1u + stagger)
                                        : (zs >= 8u ? (3u + stagger) : (2u + stagger)))
                             : (cam_mov ? (zs >= 8u ? (2u + stagger) : (1u + stagger))
                                        : (zs >= 10u  ? (6u + static_cast<u32>(idx & 3u))
                                           : zs >= 7u ? (4u + stagger)
                                                      : (2u + stagger)));
                    e.skip_until_frame.store(ref_frame + skip_len, std::memory_order_relaxed);
                    SetEntryOccActive(e, true);
                }
            } else {
                e.zero_streak.store(0, std::memory_order_relaxed);
                e.skip_until_frame.store(0, std::memory_order_relaxed);
                SetEntryOccActive(e, false);
            }
            e.probe_in_flight.store(0u, std::memory_order_release);
        }
    }

    void SealCurrentSlice(u64 tick) noexcept {
        if (!pool) {
            return;
        }
        const u32 s = cur_slice;
        if (slice_used[s] > 0 && !slice_sealed[s]) {
            slice_last_tick[s] = tick;
            slice_sealed[s] = true;
            cur_slice = (s + 1) % NumSlices;
        }
    }

    void NoteStageAFrameStart() noexcept {
        stage_a_gbuf_draws = 0;
        const u32 prev_f = stage_a_frame.load(std::memory_order_relaxed);
        const u32 sb_f = cur_frame.load(std::memory_order_relaxed);
        const u32 next_f =
            (VkMtDirectRecorder::ConfiguredMode() > 0)
                ? std::max(prev_f + 1u, BbStats::guest_frame_number.load(std::memory_order_relaxed))
                : std::clamp(prev_f + 1u, sb_f, sb_f + 1u);
        stage_a_frame.store(next_f, std::memory_order_relaxed);
    }

    void CollectRetiredSlices(vk::Device device, Scheduler& scheduler) noexcept {
        if (!pool) {
            return;
        }
        const u32 stage_b_frame = cur_frame.load(std::memory_order_relaxed);
        const u32 ref_frame =
            stage_a_occ_cull
                ? std::max(stage_b_frame, stage_a_frame.load(std::memory_order_relaxed))
                : stage_b_frame;
        std::array<u64, SliceQueries> samples{};
        for (u32 s = 0; s < NumSlices; ++s) {
            if (!slice_sealed[s] || !scheduler.IsFree(slice_last_tick[s])) {
                continue;
            }
            const u32 count = slice_used[s];
            if (count > 0) {
                const auto res = device.getQueryPoolResults(
                    *pool, s * SliceQueries, count, count * sizeof(u64), samples.data(),
                    sizeof(u64), vk::QueryResultFlagBits::e64);
                if (res == vk::Result::eSuccess) {
                    ApplyQuerySamplesToEntries(slice_entry_idx[s].data(), samples.data(), count,
                                               ref_frame);
                } else {
                    for (u32 i = 0; i < count; ++i) {
                        entries[slice_entry_idx[s][i]].probe_in_flight.store(
                            0u, std::memory_order_release);
                    }
                }
                device.resetQueryPool(*pool, s * SliceQueries, count);
            }
            slice_used[s] = 0;
            slice_sealed[s] = false;
        }

        if (!mt_pool) {
            return;
        }
        const u32 mt_ref_frame = ActiveFrameTag();
        for (u32 b = 0; b < MtQueryBlocks; ++b) {
            auto& blk = mt_blocks[b];
            if (blk.state.load(std::memory_order_acquire) !=
                static_cast<u8>(MtBlockState::Submitted)) {
                continue;
            }
            if (!scheduler.IsFree(blk.submit_tick)) {
                continue;
            }
            const u32 count = blk.used;
            if (count > 0) {
                const auto res = device.getQueryPoolResults(
                    *mt_pool, b * MtQueriesPerBlock, count, count * sizeof(u64), samples.data(),
                    sizeof(u64), vk::QueryResultFlagBits::e64);
                const u32 blk_ref = std::max(mt_ref_frame, blk.frame_tag);
                if (res == vk::Result::eSuccess) {
                    ApplyQuerySamplesToEntries(blk.entry_idx.data(), samples.data(), count,
                                               blk_ref);
                } else {
                    for (u32 i = 0; i < count; ++i) {
                        entries[blk.entry_idx[i]].probe_in_flight.store(0u,
                                                                        std::memory_order_release);
                    }
                }
                device.resetQueryPool(*mt_pool, b * MtQueriesPerBlock, count);
            }
            blk.used = 0;
            blk.submit_tick = 0;
            blk.state.store(static_cast<u8>(MtBlockState::Free), std::memory_order_release);
        }
    }

    void ApplyCameraPoseLocked(const float pos[3], const float fwd[3], u32 frame_tag) noexcept {
        bool mov = false;
        bool fast_turn = false;
        if (has_prev_cam) {
            const float cos_ang =
                fwd[0] * prev_cam_fwd[0] + fwd[1] * prev_cam_fwd[1] + fwd[2] * prev_cam_fwd[2];
            const float dx = pos[0] - prev_cam_pos[0];
            const float dy = pos[1] - prev_cam_pos[1];
            const float dz = pos[2] - prev_cam_pos[2];
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            if (cos_ang < 0.9990f || dist_sq > 0.0100f) {
                fast_turn = true;
                mov = true;
                for (auto& e : entries) {
                    e.skip_until_frame.store(0, std::memory_order_relaxed);
                    if (e.zero_streak.load(std::memory_order_relaxed) > 2u) {
                        e.zero_streak.store(2u, std::memory_order_relaxed);
                    }
                    SetEntryOccActive(e, false);
                }
            } else if (cos_ang < 0.99995f || dist_sq > 0.0002f) {
                mov = true;
            }
        }
        prev_cam_pos = {pos[0], pos[1], pos[2]};
        prev_cam_fwd = {fwd[0], fwd[1], fwd[2]};
        has_prev_cam = true;
        pub_cam_pos[0].store(pos[0], std::memory_order_relaxed);
        pub_cam_pos[1].store(pos[1], std::memory_order_relaxed);
        pub_cam_pos[2].store(pos[2], std::memory_order_relaxed);
        pub_cam_fwd[0].store(fwd[0], std::memory_order_relaxed);
        pub_cam_fwd[1].store(fwd[1], std::memory_order_relaxed);
        pub_cam_fwd[2].store(fwd[2], std::memory_order_relaxed);
        pub_cam_valid.store(true, std::memory_order_release);
        camera_moving.store(mov, std::memory_order_relaxed);
        fast_camera_turn.store(fast_turn, std::memory_order_relaxed);
        cam_updated_frame.store(frame_tag, std::memory_order_release);
    }

    void UpdateCameraFromSceneConstants(const float* data, u32 frame_tag) noexcept {
        const u32 cur_cam_f = cam_updated_frame.load(std::memory_order_acquire);
        if (!data || frame_tag == 0u ||
            (cur_cam_f != 0u && static_cast<s32>(frame_tag - cur_cam_f) <= 0)) {
            return;
        }
        if (data[0] != 3000.0f || data[4] < 64.0f || data[5] < 64.0f ||
            std::abs(data[1] * data[0] - 1.0f) > 1e-3f) {
            return;
        }
        const float pos[3] = {data[183], data[187], data[191]};
        const float fwd[3] = {data[182], data[186], data[190]};
        if (!std::isfinite(pos[0]) || !std::isfinite(pos[1]) || !std::isfinite(pos[2]) ||
            !std::isfinite(fwd[0]) || !std::isfinite(fwd[1]) || !std::isfinite(fwd[2])) {
            return;
        }
        std::scoped_lock lk{cam_mutex};
        const u32 locked_cam_f = cam_updated_frame.load(std::memory_order_relaxed);
        if (locked_cam_f != 0u && static_cast<s32>(frame_tag - locked_cam_f) <= 0) {
            return;
        }
        ApplyCameraPoseLocked(pos, fwd, frame_tag);
    }

    void OnFrameStart(const Instance& instance, Scheduler& scheduler,
                      const CameraMotion* cam) noexcept {
        EnsureInit(instance);
        gbuf_draws_in_frame = 0;
        if (!pool) {
            return;
        }
        SealCurrentSlice(scheduler.CurrentTick());
        const u32 prev_f = cur_frame.load(std::memory_order_relaxed);
        const u32 next_f =
            (VkMtDirectRecorder::ConfiguredMode() > 0)
                ? std::max(prev_f + 1u, BbStats::guest_frame_number.load(std::memory_order_relaxed))
                : (prev_f + 1u);
        cur_frame.store(next_f, std::memory_order_relaxed);
        if (stage_a_occ_cull) [[unlikely]] {
            u32 sa_f = stage_a_frame.load(std::memory_order_relaxed);
            while (sa_f < next_f &&
                   !stage_a_frame.compare_exchange_weak(sa_f, next_f, std::memory_order_relaxed)) {
            }
        }
        const u32 last_cam_f = cam_updated_frame.load(std::memory_order_acquire);
        if (next_f > last_cam_f + 1u) {
            float pos[3]{}, fwd[3]{};
            if (cam && cam->CurrentInvViewPosAndZ(pos, fwd)) {
                std::scoped_lock lk{cam_mutex};
                if (next_f > cam_updated_frame.load(std::memory_order_relaxed) + 1u) {
                    ApplyCameraPoseLocked(pos, fwd, prev_f != 0u ? prev_f : next_f);
                }
            } else {
                camera_moving.store(false, std::memory_order_relaxed);
                fast_camera_turn.store(false, std::memory_order_relaxed);
            }
        }

        CollectRetiredSlices(instance.GetDevice(), scheduler);

        if ((++log_frame_counter % 600u) == 0u) {
            const u64 cand = stat_gbuf_candidates.exchange(0, std::memory_order_relaxed);
            const u64 skip = stat_gbuf_skipped.exchange(0, std::memory_order_relaxed);
            const u64 sa_skip = stat_stage_a_skipped.exchange(0, std::memory_order_relaxed);
            const u64 sb_fb = stat_stage_b_fallbacks.exchange(0, std::memory_order_relaxed);
            const u64 vis_off = stat_vis_offload.exchange(0, std::memory_order_relaxed);
            const u64 prob = stat_gbuf_probed.exchange(0, std::memory_order_relaxed);
            const u64 skel_cand = stat_skel_candidates.exchange(0, std::memory_order_relaxed);
            const u64 skel_skip = stat_skel_skipped.exchange(0, std::memory_order_relaxed);
            const u64 multi_hits = stat_multi_occ.exchange(0, std::memory_order_relaxed);
            const u64 sh_skip = stat_shadow_skipped.exchange(0, std::memory_order_relaxed);
            const u64 lt_cand = stat_light_candidates.exchange(0, std::memory_order_relaxed);
            const u64 lt_skip = stat_light_skipped.exchange(0, std::memory_order_relaxed);
            std::array<u64, Scheduler::kNumReorderBuckets> rb_draws{};
            u64 rb_flushes = 0;
            scheduler.ConsumeReorderStats(rb_draws, rb_flushes);
            if (cand > 0 || sh_skip > 0 || lt_cand > 0 || rb_flushes > 0) {
                std::printf(
                    "[BB-CULL] 600f: GBuf occluded skipped %llu/%llu (%.1f%%, stageA %llu, "
                    "vis_off %llu, sb_fb %llu, skel %llu/%llu, probed %llu, multi %llu), Lights "
                    "skipped %llu/%llu, MicroShadow skipped %llu, GBufSort(B0=%llu B1=%llu "
                    "B2=%llu B3=%llu flushes=%llu)\n",
                    static_cast<unsigned long long>(skip), static_cast<unsigned long long>(cand),
                    cand ? (100.0 * double(skip) / double(cand)) : 0.0,
                    static_cast<unsigned long long>(sa_skip),
                    static_cast<unsigned long long>(vis_off),
                    static_cast<unsigned long long>(sb_fb),
                    static_cast<unsigned long long>(skel_skip),
                    static_cast<unsigned long long>(skel_cand),
                    static_cast<unsigned long long>(prob),
                    static_cast<unsigned long long>(multi_hits),
                    static_cast<unsigned long long>(lt_skip),
                    static_cast<unsigned long long>(lt_cand),
                    static_cast<unsigned long long>(sh_skip),
                    static_cast<unsigned long long>(rb_draws[0]),
                    static_cast<unsigned long long>(rb_draws[1]),
                    static_cast<unsigned long long>(rb_draws[2]),
                    static_cast<unsigned long long>(rb_draws[3]),
                    static_cast<unsigned long long>(rb_flushes));
            }
        }
    }

    u16 FindOrInsertEntry(u64 key, u16 mesh_slot, u32 frame_tag, bool& out_inserted) noexcept {
        out_inserted = false;
        const u32 bucket = static_cast<u32>((key ^ (key >> 33)) & (TableBuckets - 1));
        const u16 base = static_cast<u16>(bucket * BucketWays);
        for (u16 w = 0; w < BucketWays; ++w) {
            if (entries[base + w].key.load(std::memory_order_relaxed) == key &&
                entries[base + w].is_skel.load(std::memory_order_relaxed) == 0u) {
                return base + w;
            }
        }
        for (u16 w = 0; w < BucketWays; ++w) {
            Entry& cand = entries[base + w];
            if (cand.probe_in_flight.load(std::memory_order_acquire) != 0u) {
                continue;
            }
            u64 old_key = cand.key.load(std::memory_order_relaxed);
            const u32 old_seen = cand.last_seen_frame.load(std::memory_order_relaxed);
            if (old_key == 0 || (old_seen != 0u && static_cast<s32>(frame_tag - old_seen) > 4)) {
                if (cand.key.compare_exchange_strong(old_key, key, std::memory_order_acq_rel,
                                                     std::memory_order_relaxed)) {
                    SetEntryOccActive(cand, false);
                    cand.mesh_slot.store(mesh_slot, std::memory_order_relaxed);
                    cand.is_skel.store(0u, std::memory_order_relaxed);
                    cand.skip_until_frame.store(0, std::memory_order_relaxed);
                    cand.zero_streak.store(0, std::memory_order_relaxed);
                    cand.multi_occ.store(0, std::memory_order_relaxed);
                    cand.last_seen_frame.store(frame_tag, std::memory_order_release);
                    out_inserted = true;
                    return base + w;
                }
                if (old_key == key && cand.is_skel.load(std::memory_order_relaxed) == 0u) {
                    return base + w;
                }
            }
        }
        return 0xFFFFu;
    }

    u16 FindOrInsertSkelEntry(u64 mesh_key, u16 mesh_slot, float wx, float wy, float wz,
                              u32 frame_tag, bool& out_inserted) noexcept {
        out_inserted = false;
        const u32 base_bucket = static_cast<u32>((mesh_key ^ (mesh_key >> 33)) & (TableBuckets - 1));
        u16 best_idx = 0xFFFFu;
        float best_dist_sq = kSkelTrackDistSq;

        for (u32 p = 0; p < SkelProbeBuckets; ++p) {
            const u32 b = (base_bucket + p) & (TableBuckets - 1u);
            const u16 base = static_cast<u16>(b * BucketWays);
            for (u16 w = 0; w < BucketWays; ++w) {
                const u16 idx = base + w;
                const Entry& cand = entries[idx];
                if (cand.key.load(std::memory_order_relaxed) != mesh_key ||
                    cand.is_skel.load(std::memory_order_relaxed) == 0u) {
                    continue;
                }
                const float dx = wx - cand.pos_x.load(std::memory_order_relaxed);
                const float dy = wy - cand.pos_y.load(std::memory_order_relaxed);
                const float dz = wz - cand.pos_z.load(std::memory_order_relaxed);
                const float dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq > best_dist_sq) {
                    continue;
                }
                const u32 seen = cand.last_seen_frame.load(std::memory_order_relaxed);
                // Separate enemies standing 0.2m..1.0m apart get distinct ways in the same frame;
                // only exact co-located duplicate draws (< 0.2m) hit the same entry.
                if (seen == frame_tag && dist_sq > kSkelSameSpotDistSq) {
                    continue;
                }
                best_dist_sq = dist_sq;
                best_idx = idx;
            }
        }
        if (best_idx != 0xFFFFu) {
            return best_idx;
        }

        for (u32 p = 0; p < SkelProbeBuckets; ++p) {
            const u32 b = (base_bucket + p) & (TableBuckets - 1u);
            const u16 base = static_cast<u16>(b * BucketWays);
            for (u16 w = 0; w < BucketWays; ++w) {
                const u16 idx = base + w;
                Entry& cand = entries[idx];
                if (cand.probe_in_flight.load(std::memory_order_acquire) != 0u) {
                    continue;
                }
                u64 old_key = cand.key.load(std::memory_order_relaxed);
                const u32 old_seen = cand.last_seen_frame.load(std::memory_order_relaxed);
                if (old_key == 0 ||
                    (old_seen != 0u && static_cast<s32>(frame_tag - old_seen) > 4)) {
                    if (cand.key.compare_exchange_strong(old_key, mesh_key,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_relaxed)) {
                        SetEntryOccActive(cand, false);
                        cand.mesh_slot.store(mesh_slot, std::memory_order_relaxed);
                        cand.pos_x.store(wx, std::memory_order_relaxed);
                        cand.pos_y.store(wy, std::memory_order_relaxed);
                        cand.pos_z.store(wz, std::memory_order_relaxed);
                        cand.is_skel.store(1u, std::memory_order_relaxed);
                        cand.skip_until_frame.store(0, std::memory_order_relaxed);
                        cand.zero_streak.store(0, std::memory_order_relaxed);
                        cand.multi_occ.store(0, std::memory_order_relaxed);
                        cand.last_seen_frame.store(frame_tag, std::memory_order_release);
                        out_inserted = true;
                        return idx;
                    }
                }
            }
        }
        return 0xFFFFu;
    }

    s32 AllocateQuery(u16 entry_idx) noexcept {
        const u32 s = cur_slice;
        if (slice_sealed[s] || slice_used[s] >= SliceQueries) {
            return -1;
        }
        u8 expected = 0u;
        if (!entries[entry_idx].probe_in_flight.compare_exchange_strong(
                expected, 1u, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return -1;
        }
        const u32 q = slice_used[s]++;
        slice_entry_idx[s][q] = entry_idx;
        stat_gbuf_probed.fetch_add(1, std::memory_order_relaxed);
        return static_cast<s32>(s * SliceQueries + q);
    }

    [[nodiscard]] u16 AcquireMtQueryBlock(u32 frame_tag) noexcept {
        if (!enabled || !mt_pool) {
            return 0xFFFFu;
        }
        const u32 start = mt_block_hint.fetch_add(1u, std::memory_order_relaxed);
        for (u32 i = 0; i < MtQueryBlocks; ++i) {
            const u16 b = static_cast<u16>((start + i) & (MtQueryBlocks - 1u));
            u8 exp = static_cast<u8>(MtBlockState::Free);
            if (mt_blocks[b].state.compare_exchange_strong(
                    exp, static_cast<u8>(MtBlockState::Recording), std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                mt_blocks[b].used = 0;
                mt_blocks[b].frame_tag = frame_tag;
                mt_blocks[b].submit_tick = 0;
                return b;
            }
        }
        return 0xFFFFu;
    }

    [[nodiscard]] s32 AllocateMtQuery(u16 block_id, u16 entry_idx) noexcept {
        if (block_id >= MtQueryBlocks || !mt_pool) {
            return -1;
        }
        auto& blk = mt_blocks[block_id];
        if (blk.used >= MtQueriesPerBlock) {
            return -1;
        }
        u8 expected = 0u;
        if (!entries[entry_idx].probe_in_flight.compare_exchange_strong(
                expected, 1u, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return -1;
        }
        const u32 q = blk.used++;
        blk.entry_idx[q] = entry_idx;
        return static_cast<s32>(u32(block_id) * MtQueriesPerBlock + q);
    }

    void SealMtQueryBlockOnWorkerEnd(u16& io_block_id) noexcept {
        if (io_block_id >= MtQueryBlocks) {
            io_block_id = 0xFFFFu;
            return;
        }
        auto& blk = mt_blocks[io_block_id];
        if (blk.used == 0u) {
            blk.state.store(static_cast<u8>(MtBlockState::Free), std::memory_order_release);
            io_block_id = 0xFFFFu;
        } else {
            blk.state.store(static_cast<u8>(MtBlockState::ReadyToExecute),
                            std::memory_order_release);
        }
    }

    void AbortMtQueryBlock(u16 block_id) noexcept {
        if (block_id >= MtQueryBlocks) {
            return;
        }
        auto& blk = mt_blocks[block_id];
        for (u32 i = 0; i < blk.used; ++i) {
            entries[blk.entry_idx[i]].probe_in_flight.store(0u, std::memory_order_release);
        }
        blk.used = 0;
        blk.submit_tick = 0;
        blk.state.store(static_cast<u8>(MtBlockState::Free), std::memory_order_release);
    }

    void NoteMtQueryBlockExecuted(u16 block_id, u64 submit_tick) noexcept {
        if (block_id >= MtQueryBlocks) {
            return;
        }
        auto& blk = mt_blocks[block_id];
        blk.submit_tick = submit_tick;
        blk.state.store(static_cast<u8>(MtBlockState::Submitted), std::memory_order_release);
    }
};

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       BbCp::CommandProcessor* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_watch{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_watch},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_watch},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    // Before the rasterizer is bound: Liverpool enqueues buffers only once it sees it.
    native_srv_cache =
        std::make_unique<ImageDescCacheEntry[]>(VideoCore::DirectTextures::kMaxNativeSrvs);
    native_srv_fast_cache =
        std::make_unique<NativeSrvFastEntry[]>(VideoCore::DirectTextures::kMaxNativeSrvs);
    native_sampler_cache =
        std::make_unique<NativeSamplerStageBEntry[]>(VideoCore::DirectTextures::kMaxNativeSamplers);
    mt_srv_publish =
        std::make_unique<MtSrvPublishEntry[]>(VideoCore::DirectTextures::kMaxNativeSrvs);
    mt_sampler_publish =
        std::make_unique<MtSamplerPublishEntry[]>(VideoCore::DirectTextures::kMaxNativeSamplers);
    scene_occlusion_culler = std::make_unique<SceneOcclusionCuller>();
    if (!EmulatorSettings.IsNullGPU()) {
        object_vel_culler = std::make_unique<ObjectVelocityCuller>();
        scene_occlusion_culler->EnsureInit(instance);
    }
    mt_direct_recorder = std::make_unique<VkMtDirectRecorder>(instance, scheduler, *this);
    draw_prep = std::make_unique<DrawPreparation>(pipeline_cache);
    // A program (or fetch shader) that does not look like one yet: the game may have copied it
    // into place with the GPU (DMA) in work recorded but not run yet (another game: all-zero
    // programs, the load hung). Called on the GPU command thread, inside a draw: the work
    // handed to the recording thread and recorded before it runs, so a
    // program the GPU copies into place (in place in guest memory) is there. No read back of
    // GPU-modified ranges here: downloading 512 KiB windows at draws overwrote what the CPU had
    // written there with older GPU data (another game: shader code zeroed, characters missing,
    // 5 FPS), and asking this very thread for one waited forever.
    pipeline_cache.SetProgramWriteSync(
        [this](VAddr address, u64 size) { SyncProgramWrites(address, size); });
    pipeline_cache.SetProgramMemorySync([this](VAddr, u64) {
        DrainDrawPipe(); // a copy into place may still be a packet for the recording thread
        Finish();
    });
    scene_targets = std::make_unique<SceneTargets>(instance, scheduler, runtime, texture_cache);
    // Object motion first: it fixes the buffer addresses the motion shader variants embed.
    object_motion = std::make_unique<ObjectMotion>(instance, scheduler);
    camera_motion = std::make_unique<CameraMotion>(instance, scheduler, texture_cache, runtime);
    camera_motion->SetObjectMotion(object_motion.get());
    upscaler = std::make_unique<TemporalUpscaler>(instance, scheduler, texture_cache, runtime,
                                                  *camera_motion, *scene_targets);
    prompts = std::make_unique<ButtonPrompts>(instance, scheduler, buffer_cache);
    native_postfx = std::make_unique<NativePostFx>(instance, scheduler, texture_cache,
                                                   buffer_cache, runtime);
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    BbGuestMemory::Install(instance); // bbport: before the runtime maps direct memory
    memory->SetRasterizer(this);
    buffer_cache.SetDirectBuffersManager(&direct_buffers);
    VideoCore::DirectBuffers::SetActiveCaches(&buffer_cache, &texture_cache);

    GpuProfiler::Init(instance, scheduler);
    if (DrawPipeWanted()) {
        constant_ring = std::make_unique<ConstantRing>(instance, scheduler);
        light_tile_batcher = std::make_unique<LightTileBatcher>();
        vsm_shadow_memo = std::make_unique<VsmShadowMemo>();
        vfx_ui_batcher = std::make_unique<VfxUiBatcher>();
        draw_pipe = std::make_unique<DrawPipe>(&RunDrawPacket, this);
        // Keep 1 constant resolve worker by default (`spin_us = 0` in `vk_draw_pipe.h` so it
        // never burns idle spin CPU): offloading `XXH3` + Write-Combined ReBAR `memcpy` from
        // Stage A keeps `ConstRing + Drain` at `0.30 ms/frame` instead of `1.14 ms/frame` on the
        // serial Stage A critical path (`20261102_085454.log`).
        static const u32 resolvers = [] {
            if (const char* env = std::getenv("BB_RESOLVE_WORKERS")) {
                return u32(std::clamp(std::atoi(env), 0, 4));
            }
            return BbThreads::Available() >= 8 ? 1u : 0u;
        }();
        if (resolvers != 0) {
            for (u32 i = 0; i < resolvers; ++i) {
                worker_rings.push_back(std::make_unique<ConstantRing>(instance, scheduler));
            }
            draw_pipe->StartResolvers(&ResolvePacket, resolvers);
            std::printf("GPU: %u constant resolve workers (BB_RESOLVE_WORKERS)\n", resolvers);
        }
    }
    // bbport: this thread joins the texture binding helper before it changes image state.
    runtime.SetImageAccessHook(&JoinBindHelper, this);
    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        if (buffer_cache.HasPendingInPassShadows()) {
            buffer_cache.FlushInPassShadows();
        }
        direct_buffers.FlushPendingWritebacks();
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
}

Rasterizer::~Rasterizer() {
    mt_direct_recorder.reset();
    buffer_cache.SetDirectBuffersManager(nullptr);
    VideoCore::DirectBuffers::SetActiveCaches(nullptr, nullptr);
}

namespace {
/// BB_PASS_DUMP=addr,addr (hex guest addresses of render targets) with BB_PASS_DUMP_TRIGGER=<file>
/// (diagnostics, the flicker): while the file exists, every 8th row of those images is saved at
/// each display pass (BB_DUMP_DIR/pNNNNN_<addr>_<w>x<h>_<bits>.raw), little enough not to change
/// the frame's timing much.
std::vector<VAddr> PassDumpAddresses() {
    std::vector<VAddr> out;
    if (const char* env = std::getenv("BB_PASS_DUMP")) {
        for (const char* at = env; *at;) {
            char* end = nullptr;
            out.push_back(std::strtoull(at, &end, 16));
            at = *end ? end + 1 : end;
        }
    }
    return out;
}
} // namespace

void Rasterizer::DumpTracedInputs() {
    if (traced_inputs.empty()) {
        return;
    }
    static const std::string dir = [] {
        const char* env = std::getenv("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    VkDeviceSize total = 0;
    for (const auto& r : traced_inputs) total += r.size;
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = total,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        traced_inputs.clear();
        return;
    }
    scheduler.EndRendering();
    auto ranges = traced_inputs;
    scheduler.Record([ranges, buffer](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
        VkDeviceSize at = 0;
        for (const auto& r : ranges) {
            cmd.copyBuffer(r.buffer, buffer, vk::BufferCopy{r.offset, at, r.size});
            at += r.size;
        }
    });
    std::string header;
    for (const auto& r : traced_inputs) {
        header += fmt::format("{:x}+{:x}:{} ", (u64)(VkBuffer)r.buffer, r.offset, r.size);
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/g%06llu.bin", dir.c_str(),
                  (unsigned long long)traced_inputs_frame);
    scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation, info,
                                      total, file = std::string{path}, header] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            const u32 n = u32(header.size());
            std::fwrite(&n, 4, 1, f);
            std::fwrite(header.data(), 1, n, f);
            std::fwrite(info.pMappedData, 1, total, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(allocator, buffer, allocation);
    });
    traced_inputs.clear();
}

void Rasterizer::DumpPassImages(char tag) {
    static const std::vector<VAddr> addresses = PassDumpAddresses();
    static const char* trigger = std::getenv("BB_PASS_DUMP_TRIGGER");
    if (addresses.empty() || !trigger || !std::filesystem::exists(trigger)) {
        return;
    }
    static const std::string dir = [] {
        const char* env = std::getenv("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    static u32 indices[2] = {};
    const u32 n = indices[tag == 'p' ? 0 : 1]++;
    std::printf("Pass dump %c %u: GPU frame %llu\n", tag, n,
                (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed));
    scheduler.EndRendering();
    for (const VAddr address : addresses) {
        VideoCore::Image* found = nullptr;
        texture_cache.ForEachImageInRegion(address, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (image.info.guest_address == address &&
                (!found || image.info.size.width > found->info.size.width)) {
                found = &image;
            }
        });
        if (!found || found->info.num_bits == 0) {
            continue;
        }
        const u32 width = found->info.size.width, height = found->info.size.height;
        const u32 bytes = found->info.num_bits / 8;
        static const u32 step = [] {
            const char* env = std::getenv("BB_PASS_DUMP_STEP");
            return env ? std::max(1, std::atoi(env)) : 8;
        }();
        const u32 rows = (height + step - 1) / step;
        const VkDeviceSize size = VkDeviceSize(width) * rows * bytes;
        const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                           .size = size,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
        VkBuffer buffer{};
        VmaAllocation allocation{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                            &info) != VK_SUCCESS) {
            continue;
        }
        runtime.Transit(found, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        const vk::Image image = found->GetImage();
        const vk::ImageAspectFlags aspect = found->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
        scheduler.Record([=](vk::CommandBuffer cmd) {
            std::vector<vk::BufferImageCopy> regions;
            regions.reserve(rows);
            for (u32 r = 0; r < rows; ++r) {
                regions.push_back({
                    .bufferOffset = VkDeviceSize(r) * width * bytes,
                    .imageSubresource = {aspect, 0, 0, 1},
                    .imageOffset = {0, s32(r * step), 0},
                    .imageExtent = {width, 1, 1},
                });
            }
            cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, regions);
        });
        char path[512];
        std::snprintf(path, sizeof(path), "%s/%c%05u_%llx_%ux%u_%u.raw", dir.c_str(), tag, n,
                      (unsigned long long)address, width, rows, found->info.num_bits);
        scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation,
                                          info, size, file = std::string{path}] {
            vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
            if (FILE* f = std::fopen(file.c_str(), "wb")) {
                std::fwrite(info.pMappedData, 1, size, f);
                std::fclose(f);
            }
            vmaDestroyBuffer(allocator, buffer, allocation);
        });
    }
}


namespace {
// bbport: the pipeline's own type (optional<const ...>): the other one made each draw copy it,
// its attribute list included (a malloc per draw).
std::pair<u32, u32> GetDrawOffsets(
    const Gcn::Regs& regs, const Shader::Info& info,
    const std::optional<const Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

std::pair<u32, u32> GetDrawOffsetsFromFlatUd(
    const Gcn::Regs& regs, const u32* vs_flat_ud,
    const std::optional<const Shader::Gcn::FetchShaderData>& fetch_shader) noexcept {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader && vs_flat_ud) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = vs_flat_ud[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = vs_flat_ud[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}
// Classifies micro shadow-map draws (depth-only or 2-channel VSM moment maps <= 128x128)
// without touching fullscreen blur quads (`0x0b0acf50` / `<= 6` indices) or CSM cascades.
static u32 ClassifyMicroShadowLimit(const Gcn::Regs& regs, const GraphicsPipeline* pipeline,
                                    bool is_indexed) noexcept {
    if (!is_indexed || !pipeline || regs.num_indices <= 6u) {
        return 0u;
    }
    const auto& vs_stage = pipeline->GetStage(Shader::SwStage::Vertex);
    if (vs_stage.pgm_hash == 0x0b0acf50ull) {
        return 0u; // Never cull fullscreen VSM blur / copy quads
    }
    const auto& key = pipeline->GetGraphicsKey();
    const bool is_legacy_shadow_vs = (vs_stage.pgm_hash == 0x167daba6ull);
    const bool is_depth_only_shadow =
        (key.mrt_mask == 0u) && regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    const bool is_vsm_2ch_shadow =
        (key.mrt_mask == 1u) && regs.depth_control.depth_enable && regs.color_buffers[0] &&
        (regs.color_buffers[0].GetDataFmt() == Gcn::DataFormat::Format16_16 ||
         regs.color_buffers[0].GetDataFmt() == Gcn::DataFormat::Format32_32);
    if (!is_legacy_shadow_vs && !is_depth_only_shadow && !is_vsm_2ch_shadow) {
        return 0u;
    }
    const float vp_w = std::abs(regs.viewports[0].xscale) * 2.0f;
    const float vp_h = std::abs(regs.viewports[0].yscale) * 2.0f;
    const s32 sc_w =
        s32(regs.screen_scissor.bottom_right_x) - s32(regs.screen_scissor.top_left_x);
    const s32 sc_h =
        s32(regs.screen_scissor.bottom_right_y) - s32(regs.screen_scissor.top_left_y);
    const bool is_32 = (vp_w > 0.0f && vp_w <= 33.0f && vp_h > 0.0f && vp_h <= 33.0f) ||
                       (sc_w > 0 && sc_w <= 32 && sc_h > 0 && sc_h <= 32);
    if (is_32) {
        return 1u;
    }
    const bool is_64 = (vp_w > 0.0f && vp_w <= 65.0f && vp_h > 0.0f && vp_h <= 65.0f) ||
                       (sc_w > 0 && sc_w <= 64 && sc_h > 0 && sc_h <= 64);
    if (is_64) {
        return 2u;
    }
    const bool is_128 = (vp_w > 0.0f && vp_w <= 129.0f && vp_h > 0.0f && vp_h <= 129.0f) ||
                        (sc_w > 0 && sc_w <= 128 && sc_h > 0 && sc_h <= 128);
    if (is_128 && regs.num_indices <= 180u) {
        return 4u;
    }
    return 0u;
}
} // namespace

bool Rasterizer::TryCullMicroShadowDrawMt(const Gcn::Regs& regs, const GraphicsPipeline* pipeline,
                                          bool is_indexed, MtShadowCullState& state,
                                          u64* out_skipped) noexcept {
    static const bool enabled = [] {
        const char* e = std::getenv("BB_MICRO_SHADOW_CULL");
        return !e || e[0] != '0';
    }();
    if (!enabled) {
        return false;
    }
    const u32 max_keep = ClassifyMicroShadowLimit(regs, pipeline, is_indexed);
    if (max_keep == 0u) {
        return false;
    }
    const VAddr cb_addr =
        regs.color_buffers[0]
            ? (regs.color_buffers[0].Address() ^
               (u64(regs.color_buffers[0].view.slice_start) << 40))
            : 0;
    const VAddr db_addr = regs.depth_buffer.DepthAddress();
    if (regs.depth_render_control.depth_clear_enable || cb_addr != state.cb_addr ||
        db_addr != state.db_addr) {
        state.cb_addr = cb_addr;
        state.db_addr = db_addr;
        state.face_draw_idx = 0;
        return false;
    }
    ++state.face_draw_idx;
    if (state.face_draw_idx >= max_keep) {
        if (out_skipped) {
            ++(*out_skipped);
        }
        return true;
    }
    return false;
}

u16 Rasterizer::AcquireMtOcclusionQueryBlock(u32 frame_tag) noexcept {
    if (!scene_occlusion_culler || !scene_occlusion_culler->mt_occ_cull ||
        guest_occlusion_seen.load(std::memory_order_relaxed)) {
        return 0xFFFFu;
    }
    const u32 eff_tag =
        std::max(frame_tag, scene_occlusion_culler->cur_frame.load(std::memory_order_relaxed));
    return scene_occlusion_culler->AcquireMtQueryBlock(eff_tag);
}

void Rasterizer::SealMtOcclusionQueryBlock(u16& io_block_id) noexcept {
    if (scene_occlusion_culler) {
        scene_occlusion_culler->SealMtQueryBlockOnWorkerEnd(io_block_id);
    } else {
        io_block_id = 0xFFFFu;
    }
}

void Rasterizer::AbortMtOcclusionQueryBlock(u16 block_id) noexcept {
    if (scene_occlusion_culler) {
        scene_occlusion_culler->AbortMtQueryBlock(block_id);
    }
}

void Rasterizer::NoteMtOcclusionQueryBlockExecuted(u16 block_id, u64 submit_tick) noexcept {
    if (scene_occlusion_culler) {
        scene_occlusion_culler->NoteMtQueryBlockExecuted(block_id, submit_tick);
    }
}

void Rasterizer::FlushWorkerCullerStats(u64 occ_cand, u64 occ_skip, u64 occ_probe,
                                        u64 shadow_skip, u64 vel_seen, u64 vel_culled) noexcept {
    if (scene_occlusion_culler) {
        if (occ_cand != 0) {
            scene_occlusion_culler->stat_gbuf_candidates.fetch_add(occ_cand,
                                                                   std::memory_order_relaxed);
        }
        if (occ_skip != 0) {
            scene_occlusion_culler->stat_gbuf_skipped.fetch_add(occ_skip,
                                                                std::memory_order_relaxed);
        }
        if (occ_probe != 0) {
            scene_occlusion_culler->stat_gbuf_probed.fetch_add(occ_probe,
                                                               std::memory_order_relaxed);
        }
        if (shadow_skip != 0) {
            scene_occlusion_culler->stat_shadow_skipped.fetch_add(shadow_skip,
                                                                  std::memory_order_relaxed);
        }
    }
    if (object_vel_culler && (vel_seen != 0 || vel_culled != 0)) {
        object_vel_culler->FlushWorkerStats(vel_seen, vel_culled);
    }
}

Rasterizer::MtOcclusionDecision Rasterizer::EvaluateGBufferOcclusionMt(
    const Gcn::Regs& regs, const GraphicsPipeline* pipeline, const u32* vs_flat_ud,
    VAddr vs_pgm_base, bool is_indexed, u32 index_offset, std::span<const Gcn::Buffer> vsharps,
    u16 mt_query_block_id, MtResolveCbBytesFn resolve_cb, void* resolve_ctx, u64* out_candidates,
    u64* out_skipped, u64* out_probed) noexcept {
    MtOcclusionDecision decision{};
    if (!scene_occlusion_culler || !pipeline || !vs_flat_ud || !resolve_cb) {
        return decision;
    }
    auto& culler = *scene_occlusion_culler;
    const u32 expected_mrt_mask = pipeline->GetGraphicsKey().motion_vectors ? 0x7fu : 0x3fu;
    if (!culler.enabled || !culler.mt_occ_cull || !culler.mt_pool ||
        guest_occlusion_seen.load(std::memory_order_relaxed) || !is_indexed ||
        !SceneOcclusionCuller::IsSafeOpaqueGBufferDraw(regs, pipeline->GetGraphicsKey().mrt_mask,
                                                       expected_mrt_mask) ||
        regs.num_indices < SceneOcclusionCuller::kMinCullIndices) {
        return decision;
    }

    const u32 frame_tag = culler.ActiveFrameTag();
    const u32 gbuf_ord = culler.NextMtGBufOrdinal(frame_tag);

    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
    bool has_transient_vb = false;
    u64 vtx_buf_hash = 0;
    for (const auto& vb : vsharps) {
        if (vb.base_address >= cue_base && vb.base_address < cue_end) {
            has_transient_vb = true;
            break;
        }
        vtx_buf_hash ^=
            (vb.base_address * 0x9E3779B97F4A7C15ull) ^ (vb.GetSize() * 0xBF58476D1CE4E5B9ull);
    }
    if (has_transient_vb) {
        return decision;
    }

    const auto& vs_stage = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps_stage = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const u64 vs_hash = vs_stage.pgm_hash;
    const bool is_inst_gbuf_vs = (vs_hash == 0x6a833138ull || vs_hash == 0xecbef4c7ull);
    bool is_skeleton = false;
    bool has_model_cb = false;
    u32 model_base_matrix_idx = 0;
    const u8* inst_cb_ptr = nullptr;
    u32 inst_cb_size = 0;
    const u8* palette_ptr = nullptr;
    u32 palette_size = 0;
    bool worker_has_cam = false;
    float worker_cam_pos[3]{};
    float worker_cam_fwd[3]{};

    for (u32 b_idx = 0; b_idx < vs_stage.buffers.size(); ++b_idx) {
        const auto& res = vs_stage.buffers[b_idx];
        if (res.IsSpecial()) {
            continue;
        }
        const Gcn::Buffer sharp = res.GetSharpDirect(vs_flat_ud, vs_pgm_base);
        if (!sharp.Valid() || sharp.GetStride() != 16u) {
            continue;
        }
        const u32 req_sz = sharp.GetSize();
        if (req_sz != 864u && req_sz != 416u &&
            !(is_inst_gbuf_vs && req_sz >= 208u && req_sz <= 32768u && (req_sz % 208u) == 0u) &&
            !(req_sz >= 48u && req_sz <= 16384u && (req_sz % 48u) == 0u)) {
            continue;
        }
        const u8* cb_ptr = nullptr;
        u32 cb_size = 0;
        if (!resolve_cb(resolve_ctx, sharp, cb_ptr, cb_size) || !cb_ptr) {
            continue;
        }
        if (req_sz == 864u && cb_size >= 864u && sharp.GetStride() == 16u) {
            const float* cf = reinterpret_cast<const float*>(cb_ptr);
            if (cf[0] == 3000.0f && cf[4] >= 64.0f && cf[5] >= 64.0f &&
                std::abs(cf[1] * cf[0] - 1.0f) <= 1e-3f && std::isfinite(cf[183]) &&
                std::isfinite(cf[187]) && std::isfinite(cf[191]) && std::isfinite(cf[182]) &&
                std::isfinite(cf[186]) && std::isfinite(cf[190])) {
                worker_cam_pos[0] = cf[183];
                worker_cam_pos[1] = cf[187];
                worker_cam_pos[2] = cf[191];
                worker_cam_fwd[0] = cf[182];
                worker_cam_fwd[1] = cf[186];
                worker_cam_fwd[2] = cf[190];
                worker_has_cam = true;
            }
            if (cf[0] == 3000.0f) {
                continue;
            }
        }
        if (req_sz == 416u && !is_inst_gbuf_vs && cb_size >= 84u) {
            std::memcpy(&model_base_matrix_idx, cb_ptr + 80u, sizeof(u32));
            has_model_cb = true;
        } else if (is_inst_gbuf_vs && req_sz >= 208u && req_sz <= 32768u &&
                   (req_sz % 208u) == 0u && !inst_cb_ptr && cb_size >= 84u) {
            inst_cb_ptr = cb_ptr;
            inst_cb_size = std::min<u32>(cb_size, req_sz);
            std::memcpy(&model_base_matrix_idx, cb_ptr + 80u, sizeof(u32));
            has_model_cb = true;
        } else if (req_sz >= 48u && req_sz <= 16384u && (req_sz % 48u) == 0u && !palette_ptr &&
                   cb_size >= 48u) {
            palette_ptr = cb_ptr;
            palette_size = std::min<u32>(cb_size, req_sz);
        }
    }

    if ((!has_model_cb && palette_size >= 640u) ||
        (has_model_cb && !is_inst_gbuf_vs && model_base_matrix_idx == 0u && palette_size >= 640u)) {
        is_skeleton = true;
    }
    if (!palette_ptr || palette_size < 48u || (is_skeleton && !culler.skel_occ_cull) ||
        (regs.num_instances.NumInstances() > 1u && !is_inst_gbuf_vs)) {
        return decision;
    }

    const u32 mat_byte_off =
        (has_model_cb && u64(model_base_matrix_idx) * 48ull + 48ull <= u64(palette_size))
            ? (model_base_matrix_idx * 48u)
            : 0u;
    const u32 index_size =
        regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const auto [vtx_off, inst_off] =
        GetDrawOffsetsFromFlatUd(regs, vs_flat_ud, pipeline->GetFetchShader());
    const u64 ps_hash = ps_stage ? ps_stage->pgm_hash : 0ull;
    const u32 num_indices = regs.num_indices;
    const u32 num_instances = regs.num_instances.NumInstances();
    const u64 off_mix =
        (u64(vtx_off) * 0x9E3779B97F4A7C15ull) ^ (u64(inst_off) * 0xBF58476D1CE4E5B9ull);
    const u64 base_mesh_key = index_address ^ (vs_hash * 0x9E3779B97F4A7C15ull) ^
                              (ps_hash * 0xD6E8FEB86659FD93ull) ^
                              (u64(num_indices) * 0xBF58476D1CE4E5B9ull) ^
                              (u64(num_instances) * 0x94D049BB133111EBull) ^ off_mix ^
                              (vtx_buf_hash * 0xC2B2AE3D27D4EB4Full);

    const u16 mesh_slot =
        SceneOcclusionCuller::ComputeMeshFilterSlot(index_address, vs_hash, num_indices);
    bool inserted = false;
    u16 e_idx = 0xFFFFu;
    float skel_wx = 0.0f;
    float skel_wy = 0.0f;
    float skel_wz = 0.0f;
    if (is_skeleton) {
        const float* mf = reinterpret_cast<const float*>(palette_ptr + mat_byte_off);
        skel_wx = mf[3];
        skel_wy = mf[7];
        skel_wz = mf[11];
        if (!std::isfinite(skel_wx) || !std::isfinite(skel_wy) || !std::isfinite(skel_wz)) {
            return decision;
        }
        float cam_x = worker_cam_pos[0];
        float cam_y = worker_cam_pos[1];
        float cam_z = worker_cam_pos[2];
        if (!worker_has_cam) {
            if (!culler.pub_cam_valid.load(std::memory_order_acquire)) {
                return decision;
            }
            cam_x = culler.pub_cam_pos[0].load(std::memory_order_relaxed);
            cam_y = culler.pub_cam_pos[1].load(std::memory_order_relaxed);
            cam_z = culler.pub_cam_pos[2].load(std::memory_order_relaxed);
        }
        const float cdx = skel_wx - cam_x;
        const float cdy = skel_wy - cam_y;
        const float cdz = skel_wz - cam_z;
        if (cdx * cdx + cdy * cdy + cdz * cdz < SceneOcclusionCuller::kMinSkelCullDistSq) {
            return decision;
        }
        u64 skel_key = base_mesh_key ^ (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^
                       (u64(palette_size) * 0x94D049BB133111EBull);
        if (skel_key == 0ull) {
            skel_key = 1ull;
        }
        if (out_candidates) {
            ++(*out_candidates);
        }
        culler.stat_skel_candidates.fetch_add(1, std::memory_order_relaxed);
        e_idx = culler.FindOrInsertSkelEntry(skel_key, mesh_slot, skel_wx, skel_wy, skel_wz,
                                             frame_tag, inserted);
    } else {
        float cam_x = worker_cam_pos[0];
        float cam_y = worker_cam_pos[1];
        float cam_z = worker_cam_pos[2];
        bool have_cam = worker_has_cam;
        if (!have_cam && culler.pub_cam_valid.load(std::memory_order_acquire)) {
            cam_x = culler.pub_cam_pos[0].load(std::memory_order_relaxed);
            cam_y = culler.pub_cam_pos[1].load(std::memory_order_relaxed);
            cam_z = culler.pub_cam_pos[2].load(std::memory_order_relaxed);
            have_cam = true;
        }
        if (have_cam && SceneOcclusionCuller::IsNearCameraWorldObject(
                            palette_ptr, palette_size, mat_byte_off, is_inst_gbuf_vs,
                            num_instances, inst_cb_ptr, inst_cb_size, cam_x, cam_y, cam_z)) {
            return decision;
        }
        if (out_candidates) {
            ++(*out_candidates);
        }
        const u32 hash_bytes =
            std::min<u32>(palette_size - mat_byte_off, is_inst_gbuf_vs ? 384u : 192u);
        u64 world_palette_hash = XXH3_64bits_withSeed(
            palette_ptr + mat_byte_off, hash_bytes,
            (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^ u64(palette_size));
        if (inst_cb_ptr && inst_cb_size > 0u) {
            world_palette_hash ^= XXH3_64bits_withSeed(
                inst_cb_ptr, std::min<u32>(inst_cb_size, 1040u), 0xD6E8FEB86659FD93ull);
            if (is_inst_gbuf_vs && num_instances > 1u) {
                const u32 extra_inst = std::min<u32>(num_instances, 4u);
                for (u32 k = 1u; k < extra_inst; ++k) {
                    const u32 ioff = k * 208u + 80u;
                    if (ioff + sizeof(u32) <= inst_cb_size) {
                        u32 ibone = 0u;
                        std::memcpy(&ibone, inst_cb_ptr + ioff, sizeof(u32));
                        if (u64(ibone) * 48ull + 48ull <= u64(palette_size)) {
                            world_palette_hash ^= XXH3_64bits_withSeed(
                                palette_ptr + ibone * 48u, 48u,
                                0x9E3779B97F4A7C15ull * u64(k + 1u));
                        }
                    }
                }
            }
        }
        u64 draw_key = base_mesh_key ^ world_palette_hash;
        if (draw_key == 0ull) {
            draw_key = 1ull;
        }
        e_idx = culler.FindOrInsertEntry(draw_key, mesh_slot, frame_tag, inserted);
    }
    if (e_idx == 0xFFFFu) {
        return decision;
    }

    auto& e = culler.entries[e_idx];
    if (!inserted) {
        const u32 prev_seen = e.last_seen_frame.exchange(frame_tag, std::memory_order_acq_rel);
        if (prev_seen == frame_tag) {
            e.multi_occ.store(8u, std::memory_order_relaxed);
            e.zero_streak.store(0u, std::memory_order_relaxed);
            e.skip_until_frame.store(0u, std::memory_order_relaxed);
            culler.SetEntryOccActive(e, false);
            culler.stat_multi_occ.fetch_add(1, std::memory_order_relaxed);
            return decision;
        }
        if (is_skeleton) {
            e.pos_x.store(skel_wx, std::memory_order_relaxed);
            e.pos_y.store(skel_wy, std::memory_order_relaxed);
            e.pos_z.store(skel_wz, std::memory_order_relaxed);
        }
        if (prev_seen != 0u && static_cast<s32>(frame_tag - prev_seen) > 16) {
            e.zero_streak.store(0u, std::memory_order_relaxed);
            e.skip_until_frame.store(0u, std::memory_order_relaxed);
            e.multi_occ.store(0u, std::memory_order_relaxed);
            culler.SetEntryOccActive(e, false);
        } else if (const u8 m = e.multi_occ.load(std::memory_order_relaxed); m > 0u) {
            e.multi_occ.store(static_cast<u8>(m - 1u), std::memory_order_relaxed);
        }
    }

    bool local_fast_turn = culler.fast_camera_turn.load(std::memory_order_relaxed);
    bool local_cam_moving = culler.camera_moving.load(std::memory_order_relaxed);
    if (worker_has_cam && culler.pub_cam_valid.load(std::memory_order_acquire)) {
        const float p0 = culler.pub_cam_pos[0].load(std::memory_order_relaxed);
        const float p1 = culler.pub_cam_pos[1].load(std::memory_order_relaxed);
        const float p2 = culler.pub_cam_pos[2].load(std::memory_order_relaxed);
        const float f0 = culler.pub_cam_fwd[0].load(std::memory_order_relaxed);
        const float f1 = culler.pub_cam_fwd[1].load(std::memory_order_relaxed);
        const float f2 = culler.pub_cam_fwd[2].load(std::memory_order_relaxed);
        const float cos_ang =
            worker_cam_fwd[0] * f0 + worker_cam_fwd[1] * f1 + worker_cam_fwd[2] * f2;
        const float dx = worker_cam_pos[0] - p0;
        const float dy = worker_cam_pos[1] - p1;
        const float dz = worker_cam_pos[2] - p2;
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        if (cos_ang < 0.9990f || dist_sq > 0.0100f) {
            local_fast_turn = true;
            local_cam_moving = true;
        } else if (cos_ang < 0.99995f || dist_sq > 0.0002f) {
            local_cam_moving = true;
        }
    }

    if (gbuf_ord > 12u && e.multi_occ.load(std::memory_order_relaxed) == 0u && !local_fast_turn) {
        const u16 zs = e.zero_streak.load(std::memory_order_relaxed);
        const u8 p_if = e.probe_in_flight.load(std::memory_order_acquire);
        if (zs >= SceneOcclusionCuller::kMinZeroStreak && p_if == 0u &&
            frame_tag < e.skip_until_frame.load(std::memory_order_relaxed)) {
            decision.skip_draw = true;
            if (out_skipped) {
                ++(*out_skipped);
            }
            if (is_skeleton) {
                culler.stat_skel_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            return decision;
        }
        const u32 probe_mask = (is_skeleton || local_cam_moving) ? 1u : 3u;
        if (p_if == 0u && !Breadcrumbs::Enabled() && !DynCheck::Enabled() &&
            (zs > 0u || ((frame_tag + u32(e_idx)) & probe_mask) == 0u)) {
            const s32 q_slot = culler.AllocateMtQuery(mt_query_block_id, e_idx);
            if (q_slot >= 0) {
                decision.query_slot = q_slot;
                decision.pool = *culler.mt_pool;
                if (out_probed) {
                    ++(*out_probed);
                }
            }
        }
    }
    return decision;
}

bool Rasterizer::IsRegionGpuModifiedForMt(VAddr addr, u64 size) noexcept {
    if (size == 0) {
        return false;
    }
    if (buffer_cache.IsConstRamRange(addr, size)) {
        return true;
    }
    const u64 retired = gpu_retired_packet.load(std::memory_order_relaxed);
    return pending_write_table.Newest(addr, size, retired + 1) > retired;
}

void Rasterizer::ExecuteSecondaryCmdBuf(u32 worker_id, u64 cmd_seq, u32 recorded_draws,
                                        u16 occ_block_id) {
    if (!mt_direct_recorder) {
        return;
    }
    MtCompletedPassMeta* meta = mt_direct_recorder->LookupCompletedPass(worker_id, cmd_seq);
    if (!meta || !meta->first_pso) {
        mt_direct_recorder->StampEmptyPassOnStageB(worker_id, cmd_seq, occ_block_id);
        if (meta) {
            meta->MarkConsumed(cmd_seq);
        }
        return;
    }

    if ((meta->is_gbuffer || occ_block_id != 0xFFFFu) && scene_occlusion_culler &&
        scene_occlusion_culler->initialized) {
        auto& culler = *scene_occlusion_culler;
        const u32 ftag = culler.ActiveFrameTag();
        if (culler.mt_last_collect_frame != ftag) {
            culler.mt_last_collect_frame = ftag;
            culler.CollectRetiredSlices(instance.GetDevice(), scheduler);
        }
    }

    if (recorded_draws == 0) {
        mt_direct_recorder->StampEmptyPassOnStageB(worker_id, cmd_seq, occ_block_id);
        meta->MarkConsumed(cmd_seq);
        return;
    }

    if (!mt_stage_b_regs) [[unlikely]] {
        mt_stage_b_regs = std::make_unique<Gcn::Regs>();
        BbCp::CommandProcessor::InitGnmDefaultRegs(*mt_stage_b_regs, 3u);
    }
    meta->ApplyToRegs(*mt_stage_b_regs);

    if (!pending_buf_copies.empty()) [[unlikely]] {
        FlushPendingBufferCopies();
    }
    if (direct_buffers.HasPendingWritebacks()) {
        direct_buffers.FlushPendingWritebacks();
        needs_barrier = true;
    }
    if (runtime.HasPendingBarriers()) {
        needs_barrier = true;
    }

    struct ScopedMtPipeRegs {
        Rasterizer& self;
        const Gcn::Regs* prev_stage_regs;
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> prev_cb_ext;
        Gcn::CbDbExtent prev_db_ext;
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> prev_cb_uid;
        u16 prev_db_uid;

        ScopedMtPipeRegs(Rasterizer& r, const MtCompletedPassMeta& m) noexcept
            : self(r), prev_stage_regs(stage_regs), prev_cb_ext(r.pipe_cb_extent),
              prev_db_ext(r.pipe_db_extent), prev_cb_uid(r.pipe_cb_uid),
              prev_db_uid(r.pipe_db_uid) {
            stage_regs = r.mt_stage_b_regs.get();
            r.pipe_cb_extent = m.cb_extent;
            r.pipe_db_extent = m.db_extent;
            r.pipe_cb_uid = m.cb_uid;
            r.pipe_db_uid = m.db_uid;
        }
        ~ScopedMtPipeRegs() {
            stage_regs = prev_stage_regs;
            self.pipe_cb_extent = prev_cb_ext;
            self.pipe_db_extent = prev_db_ext;
            self.pipe_cb_uid = prev_cb_uid;
            self.pipe_db_uid = prev_db_uid;
        }
    } reg_scope{*this, *meta};

    const GraphicsPipeline* pipeline = meta->first_pso;
    for (u32 i = 0; i < meta->num_sampled_rt_refs; ++i) {
        const auto& ref = meta->sampled_rt_refs[i];
        if (auto* img = texture_cache.TryGetImage(ref.id, ref.uid);
            img && img->backing && !img->binding.needs_rebind) {
            texture_cache.MarkFound(ref.id);
            texture_cache.TouchImage(*img);
            constexpr auto ShaderStages =
                vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
            const auto read_access =
                img->info.props.is_depth
                    ? (vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                       vk::AccessFlagBits2::eShaderRead)
                    : vk::AccessFlagBits2::eShaderRead;
            const auto layout = img->info.props.is_depth
                                    ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                    : vk::ImageLayout::eShaderReadOnlyOptimal;
            needs_barrier |= runtime.Transit(img, layout, ShaderStages, read_access, ref.range);
        }
    }
    std::array<std::tuple<const VideoCore::Buffer*, u64, u64>,
               MtCompletedPassMeta::kMaxUavIndirectRefs>
        resolved_uav_indirects{};
    u32 num_resolved_uav_indirects = 0u;
    for (u32 i = 0; i < meta->num_uav_indirect_refs; ++i) {
        const auto& iref = meta->uav_indirect_refs[i];
        u64 req_bytes = iref.byte_size;
        auto [ibuf, ibase] = direct_buffers.TryResolveReadClamped(iref.addr, req_bytes);
        if (ibuf && req_bytes > 0) {
            needs_barrier |= runtime.IsBufferAccessed(ibuf, ibase, req_bytes);
            resolved_uav_indirects[num_resolved_uav_indirects++] = {ibuf, ibase, req_bytes};
        }
    }
    if (needs_barrier) {
        runtime.FlushBarriers();
        needs_barrier = false;
    }

    PrepareRenderState(pipeline);
    NoteUpscalerDraw(pipeline);
    if (gbuffer_draw && meta->has_camera_cb) {
        if (camera_motion && !camera_motion->PassHasCamera()) {
            camera_motion->OnConstants(meta->camera_cb_floats.data());
        }
        if (scene_occlusion_culler) {
            const u32 cam_ftag = meta->guest_frame != 0u ? meta->guest_frame
                                                         : scene_occlusion_culler->ActiveFrameTag();
            scene_occlusion_culler->UpdateCameraFromSceneConstants(
                meta->camera_cb_floats.data(), cam_ftag);
        }
    }

    const u8 sec_mode = 2u;
    const auto state = BeginRendering(pipeline, sec_mode);
    const bool will_begin_pass = !last_begin_memo_hit && scheduler.WillBeginRendering(state);
    if (HasPendingHleCopies() && will_begin_pass) {
        FlushPendingHleCopies(*this);
    }

    const u32 prev_draws_in_frame =
        g_draws_in_frame.fetch_add(recorded_draws, std::memory_order_relaxed);
    if (prev_draws_in_frame == 0u) {
        g_frame_first_draw_ns.store(SteadyNowNs(), std::memory_order_relaxed);
    }

    if (needs_barrier || (will_begin_pass && runtime.HasPendingBarriers())) {
        runtime.FlushBarriers();
        needs_barrier = false;
    }

    MarkPass(pipeline, state, will_begin_pass);
    if (occlusion) {
        occlusion->SetPassScale(target_scale[0] * target_scale[1]);
    }
    if (will_begin_pass || !scheduler.IsRendering()) {
        scheduler.BeginRendering(state, 0);
        if (begin_memo.valid) {
            begin_memo.cmd_epoch = scheduler.DescriptorStateEpoch();
        }
    }

    bound_graphics_pipeline = vk::Pipeline{};
    scheduler.GetDynamicState().Invalidate();

    const vk::CommandBuffer sec_cmd = meta->sec_cmd;
    mt_direct_recorder->ExecuteSecondaryBuffers({&sec_cmd, 1}, worker_id, cmd_seq, occ_block_id);
    for (u32 i = 0; i < num_resolved_uav_indirects; ++i) {
        const auto [ibuf, ibase, isz] = resolved_uav_indirects[i];
        runtime.AccessBuffer(ibuf, ibase, isz, vk::PipelineStageFlagBits2::eDrawIndirect,
                             vk::AccessFlagBits2::eIndirectCommandRead);
    }
    if (meta->motion_slot < 4u && object_motion) {
        object_motion->NoteSlotSubmittedOnStageB(meta->motion_slot);
    }
    meta->MarkConsumed(cmd_seq);

    BB_SECTION(Kick);
    ResetBindings(false);
    scheduler.KickRecording();
}

bool Rasterizer::TryCullMicroShadowDraw(const GraphicsPipeline* pipeline,
                                        bool is_indexed) noexcept {
    static const bool enabled = [] {
        const char* e = std::getenv("BB_MICRO_SHADOW_CULL");
        return !e || e[0] != '0';
    }();
    if (!enabled) {
        return false;
    }
    const auto& regs = liverpool->regs;
    const u32 max_keep = ClassifyMicroShadowLimit(regs, pipeline, is_indexed);
    if (max_keep == 0u) {
        return false;
    }
    const u32 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (stage_a_shadow_frame != cur_frame) {
        stage_a_shadow_frame = cur_frame;
        stage_a_shadow_cb_addr = 0;
        stage_a_shadow_db_addr = 0;
        stage_a_shadow_face_draw_idx = 0;
    }
    const VAddr cb_addr =
        regs.color_buffers[0]
            ? (regs.color_buffers[0].Address() ^
               (u64(regs.color_buffers[0].view.slice_start) << 40))
            : 0;
    const VAddr db_addr = regs.depth_buffer.DepthAddress();
    if (regs.depth_render_control.depth_clear_enable || cb_addr != stage_a_shadow_cb_addr ||
        db_addr != stage_a_shadow_db_addr) {
        stage_a_shadow_cb_addr = cb_addr;
        stage_a_shadow_db_addr = db_addr;
        stage_a_shadow_face_draw_idx = 0;
        return false;
    }
    ++stage_a_shadow_face_draw_idx;
    if (stage_a_shadow_face_draw_idx >= max_keep) {
        if (u64 cur = BbStats::draws.load(std::memory_order_relaxed); cur > 0) {
        BbStats::draws.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed);
    }
        if (scene_occlusion_culler) {
            scene_occlusion_culler->stat_shadow_skipped.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }
    return false;
}

void Rasterizer::NoteOccCullStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept {
    if (scene_occlusion_culler) {
        scene_occlusion_culler->NoteStageACpuCopyBatch(raw_regions, count);
    }
}

void Rasterizer::NoteLightBatchCullStats(u64 seen_tiles, u64 saved_draws) noexcept {
    if (scene_occlusion_culler) {
        if (seen_tiles > 0u) {
            scene_occlusion_culler->stat_light_candidates.fetch_add(seen_tiles,
                                                                    std::memory_order_relaxed);
        }
        if (saved_draws > 0u) {
            scene_occlusion_culler->stat_light_skipped.fetch_add(saved_draws,
                                                                 std::memory_order_relaxed);
        }
    }
}

void Rasterizer::InvalidateOccCullStageACpuShadow(VAddr addr, u64 size) noexcept {
    if (scene_occlusion_culler) {
        scene_occlusion_culler->InvalidateStageACpuShadow(addr, size);
    }
}

bool Rasterizer::TryCullSceneOcclusionStageA(const GraphicsPipeline* pipeline,
                                             const PreparedDraw* used_prepared, bool is_indexed,
                                             u32 index_offset, u16& out_handoff,
                                             u8& out_gbuf_bucket) {
    out_handoff = kOccNoProbe;
    out_gbuf_bucket = 0xFFu;
    if (!scene_occlusion_culler || !pipeline) {
        return false;
    }
    auto& culler = *scene_occlusion_culler;
    if (!culler.enabled || !culler.pool ||
        guest_occlusion_seen.load(std::memory_order_relaxed) ||
        (occlusion && occlusion->Events() != 0)) {
        return false;
    }
    if (!culler.stage_a_occ_cull || liverpool->pipe_dirty.reset || FrameCapture::Active() ||
        BbCeStats::Enabled()) [[unlikely]] {
        out_handoff = kOccStageBFallback;
        return false;
    }

    const auto& regs = liverpool->regs;
    const bool has_motion_mrt = pipeline->GetGraphicsKey().motion_vectors;
    const u32 expected_mrt_mask = has_motion_mrt ? 0x7fu : 0x3fu;
    const u32 mrt_mask = pipeline->GetGraphicsKey().mrt_mask;
    const u32 num_indices = regs.num_indices;
    if (!is_indexed || num_indices < SceneOcclusionCuller::kMinCullIndices ||
        regs.color_control.mode == Gcn::ColorControl::OperationMode::Disable ||
        !regs.depth_buffer.DepthValid() ||
        !SceneOcclusionCuller::IsSafeOpaqueGBufferDraw(regs, mrt_mask, expected_mrt_mask)) {
        return false;
    }
    for (u32 cb = 0; cb < 6u; ++cb) {
        if (!regs.color_buffers[cb] || regs.color_target_mask.GetMask(cb) == 0u) {
            return false;
        }
    }

    const u32 gbuf_ord = ++culler.stage_a_gbuf_draws;
    if (gbuf_ord <= 12u) {
        culler.stat_vis_offload.fetch_add(1, std::memory_order_relaxed);
        out_handoff = kOccStageBFallback;
        return false;
    }
    const u32 frame_tag = culler.stage_a_frame.load(std::memory_order_relaxed);
    const auto& vs_stage = pipeline->GetStage(Shader::SwStage::Vertex);
    const u64 vs_hash = vs_stage.pgm_hash;
    const u32 index_size =
        regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const u16 mesh_slot =
        SceneOcclusionCuller::ComputeMeshFilterSlot(index_address, vs_hash, num_indices);
    if (culler.active_mesh_occ[mesh_slot].load(std::memory_order_relaxed) == 0u) [[likely]] {
        // Mesh currently has zero occluded instances in the scene: offload periodic probe
        // scheduling to Stage B (`bb:DrawRec` idle headroom) at ~1 ns cost on Stage A.
        culler.stat_vis_offload.fetch_add(1, std::memory_order_relaxed);
        out_handoff = kOccStageBFallback;
        return false;
    }
    const auto* ps_stage = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const u32 num_instances = regs.num_instances.NumInstances();

    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
    const bool has_cue = cue_end > cue_base;

    bool has_transient_vb = false;
    u64 vtx_buf_hash = 0;
    if (used_prepared && used_prepared->vertex.valid) {
        vtx_buf_hash = used_prepared->vertex.buffers_hash;
        for (u32 i = 0; i < used_prepared->vertex.count; ++i) {
            const VAddr vb_addr = used_prepared->vertex.buffers[i].base_address;
            if (has_cue && vb_addr >= cue_base && vb_addr < cue_end) {
                has_transient_vb = true;
                break;
            }
        }
    } else if (const auto& fetch = pipeline->GetFetchShader();
               fetch && fetch->attributes.size() <= Shader::NUM_BUFFERS) {
        const auto* fast_vs = pipeline_cache.PeekFastVsharps(pipeline);
        if (fast_vs && fast_vs->size() == fetch->attributes.size()) {
            for (const auto& vb : *fast_vs) {
                if (has_cue && vb.base_address >= cue_base && vb.base_address < cue_end) {
                    has_transient_vb = true;
                    break;
                }
            }
            if (!has_transient_vb && !fast_vs->empty()) {
                vtx_buf_hash =
                    XXH3_64bits(fast_vs->data(), fast_vs->size() * sizeof(Gcn::Buffer));
            }
        } else {
            std::array<Gcn::Buffer, Shader::NUM_BUFFERS> local_vsharps{};
            const auto vs_ud = vs_stage.UserData();
            const u32 attr_cnt = static_cast<u32>(fetch->attributes.size());
            for (u32 a = 0; a < attr_cnt; ++a) {
                const Gcn::Buffer vb = fetch->attributes[a].GetSharpDirect(vs_ud);
                if (has_cue && vb.base_address >= cue_base && vb.base_address < cue_end) {
                    has_transient_vb = true;
                    break;
                }
                local_vsharps[a] = vb;
            }
            if (!has_transient_vb && attr_cnt != 0u) {
                vtx_buf_hash = XXH3_64bits(local_vsharps.data(), attr_cnt * sizeof(Gcn::Buffer));
            }
        }
    }
    if (has_transient_vb) {
        return false;
    }

    const bool is_inst_gbuf_vs = (vs_hash == 0x6a833138ull || vs_hash == 0xecbef4c7ull);
    bool is_skeleton = false;
    bool has_model_cb = false;
    u32 model_base_matrix_idx = 0;
    const u8* inst_cb_ptr = nullptr;
    u32 inst_cb_size = 0;
    const u8* palette_ptr = nullptr;
    u32 palette_size = 0;

    const PreparedStage* prep_vs = nullptr;
    if (used_prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
        for (u32 i = 0; i < used_prepared->num_stages; ++i) {
            const auto& cand = used_prepared->stages[i];
            if (&cand.program->info == &vs_stage && cand.num_buffers == vs_stage.buffers.size()) {
                prep_vs = &cand;
                break;
            }
        }
    }
    const u32* const vs_flat = vs_stage.FlatUserData().data();
    const VAddr vs_pgm_base = vs_stage.ProgramBase();

    enum class OccCbRole : u8 { None = 0, SceneCam, ModelCb, InstCb, Palette };
    for (u32 b_idx = 0; b_idx < vs_stage.buffers.size(); ++b_idx) {
        const auto& res = vs_stage.buffers[b_idx];
        if (res.IsSpecial()) {
            continue;
        }
        const Gcn::Buffer sharp =
            prep_vs ? prep_vs->buffer_sharps[b_idx] : res.GetSharpDirect(vs_flat, vs_pgm_base);
        if (!sharp.Valid() || sharp.GetStride() != 16) {
            continue;
        }
        const u32 cb_size = sharp.GetSize();
        const VAddr cb_va = sharp.base_address;
        if (cb_va == 0 || cb_size == 0) {
            continue;
        }

        OccCbRole role = OccCbRole::None;
        if (cb_size == 864u) {
            if (culler.cam_updated_frame.load(std::memory_order_relaxed) != frame_tag) {
                role = OccCbRole::SceneCam;
            }
        } else if (cb_size == 416u && !is_inst_gbuf_vs) {
            role = OccCbRole::ModelCb;
        } else if (is_inst_gbuf_vs && cb_size >= 208u && cb_size <= 32768u &&
                   (cb_size % 208u) == 0u) {
            if (!inst_cb_ptr) {
                role = OccCbRole::InstCb;
            }
        } else if (cb_size >= 48u && cb_size <= 16384u && (cb_size % 48u) == 0u) {
            if (!palette_ptr) {
                role = OccCbRole::Palette;
            }
        }
        if (role == OccCbRole::None) {
            continue;
        }

        const u8* cb_ptr = nullptr;
        if ((has_cue && cb_va >= cue_base && cb_size <= (cue_end - cb_va)) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(cb_va, cb_size) ||
            VideoCore::DirectBuffers::IsCpuReadableImmutableRange(cb_va, cb_size)) {
            cb_ptr = reinterpret_cast<const u8*>(cb_va);
        } else {
            cb_ptr = culler.LookupStageARedirect(cb_va, cb_size, frame_tag);
            if (!cb_ptr && VideoCore::DirectBuffers::OverlapsNativeOwnedRange(cb_va, cb_size)) {
                cb_ptr = culler.LookupSettledMirror(direct_buffers, cb_va, cb_size, frame_tag);
                if (!cb_ptr) {
                    if (role != OccCbRole::SceneCam) {
                        culler.stat_stage_b_fallbacks.fetch_add(1, std::memory_order_relaxed);
                        out_handoff = kOccStageBFallback;
                        return false;
                    }
                    continue;
                }
            }
        }
        if (!cb_ptr) {
            uintptr_t mapped_end = 0;
            if (BbLookupMappedEnd(cb_va, mapped_end) && cb_size <= (mapped_end - cb_va) &&
                !PendingWriteOverlaps(cb_va, cb_size) &&
                !buffer_cache.IsRegionGpuModified(cb_va, cb_size)) {
                cb_ptr = reinterpret_cast<const u8*>(cb_va);
            } else if (role != OccCbRole::SceneCam) {
                culler.stat_stage_b_fallbacks.fetch_add(1, std::memory_order_relaxed);
                out_handoff = kOccStageBFallback;
                return false;
            }
        }
        if (!cb_ptr) {
            continue;
        }

        switch (role) {
        case OccCbRole::SceneCam:
            if (reinterpret_cast<const float*>(cb_ptr)[0] == 3000.0f) {
                culler.UpdateCameraFromSceneConstants(reinterpret_cast<const float*>(cb_ptr),
                                                      frame_tag);
            }
            break;
        case OccCbRole::ModelCb:
            std::memcpy(&model_base_matrix_idx, cb_ptr + 80, sizeof(u32));
            has_model_cb = true;
            break;
        case OccCbRole::InstCb:
            inst_cb_ptr = cb_ptr;
            inst_cb_size = cb_size;
            std::memcpy(&model_base_matrix_idx, cb_ptr + 80, sizeof(u32));
            has_model_cb = true;
            break;
        case OccCbRole::Palette:
            palette_ptr = cb_ptr;
            palette_size = cb_size;
            break;
        case OccCbRole::None:
            break;
        }
    }

    if (!palette_ptr || palette_size < 48u) {
        return false;
    }
    const u32 mat_byte_off =
        (has_model_cb && u64(model_base_matrix_idx) * 48ull + 48ull <= u64(palette_size))
            ? (model_base_matrix_idx * 48u)
            : 0u;
    const u64 ps_hash = ps_stage ? ps_stage->pgm_hash : 0ull;
    // Precompute the 4-bucket Hi-Z G-Buffer reorder bucket right here while `palette_ptr` and
    // `inst_cb_ptr` are hot in L1 so Stage B never re-walks `vs_info.buffers` (~0.15-0.20 ms saved).
    {
        const bool is_cutout_ps =
            (ps_stage && ps_stage->has_discard) || ps_hash == 0x1c3953c6ull ||
            ps_hash == 0x70bb42afull || ps_hash == 0xdbeccd9cull || ps_hash == 0x2140da2bull ||
            ps_hash == 0x998b035eull || ps_hash == 0xa89e01e9ull || ps_hash == 0x0f54e5f9ull;
        if (is_cutout_ps) {
            out_gbuf_bucket = 3u;
        } else {
            out_gbuf_bucket = 1u;
            if (culler.pub_cam_valid.load(std::memory_order_relaxed) &&
                u64(mat_byte_off) + 48ull <= u64(palette_size)) {
                const float cx = culler.pub_cam_pos[0].load(std::memory_order_relaxed);
                const float cy = culler.pub_cam_pos[1].load(std::memory_order_relaxed);
                const float cz = culler.pub_cam_pos[2].load(std::memory_order_relaxed);
                const float* mf = reinterpret_cast<const float*>(palette_ptr + mat_byte_off);
                const float wx = mf[3];
                const float wy = mf[7];
                const float wz = mf[11];
                if ((wx != 0.0f || wy != 0.0f || wz != 0.0f) && std::isfinite(wx) &&
                    std::isfinite(wy) && std::isfinite(wz)) {
                    const float dx = wx - cx;
                    const float dy = wy - cy;
                    const float dz = wz - cz;
                    float min_dist_sq = dx * dx + dy * dy + dz * dz;
                    if (is_inst_gbuf_vs && num_instances > 1u && inst_cb_ptr) {
                        const u32 check_n = std::min<u32>(num_instances, 4u);
                        for (u32 k = 1u; k < check_n; ++k) {
                            const u32 ioff = k * 208u + 80u;
                            if (ioff + sizeof(u32) <= inst_cb_size) {
                                u32 ibone = 0u;
                                std::memcpy(&ibone, inst_cb_ptr + ioff, sizeof(u32));
                                if (u64(ibone) * 48ull + 48ull <= u64(palette_size)) {
                                    const float* imf =
                                        reinterpret_cast<const float*>(palette_ptr + ibone * 48u);
                                    if ((imf[3] != 0.0f || imf[7] != 0.0f || imf[11] != 0.0f) &&
                                        std::isfinite(imf[3]) && std::isfinite(imf[7]) &&
                                        std::isfinite(imf[11])) {
                                        const float ix = imf[3] - cx;
                                        const float iy = imf[7] - cy;
                                        const float iz = imf[11] - cz;
                                        const float d2 = ix * ix + iy * iy + iz * iz;
                                        if (d2 < min_dist_sq) {
                                            min_dist_sq = d2;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (min_dist_sq < 20.0f * 20.0f) {
                        out_gbuf_bucket = 0u;
                    } else if (min_dist_sq < 50.0f * 50.0f) {
                        out_gbuf_bucket = 1u;
                    } else {
                        out_gbuf_bucket = 2u;
                    }
                }
            }
        }
    }
    if (!has_model_cb && palette_size >= 640u) {
        is_skeleton = true;
    } else if (has_model_cb && !is_inst_gbuf_vs && model_base_matrix_idx == 0u &&
               palette_size >= 640u) {
        is_skeleton = true;
    }
    if (is_skeleton && !culler.skel_occ_cull) {
        return false;
    }

    const auto [vtx_off, inst_off] = GetDrawOffsets(regs, vs_stage, pipeline->GetFetchShader());
    const u64 off_mix =
        (u64(vtx_off) * 0x9E3779B97F4A7C15ull) ^ (u64(inst_off) * 0xBF58476D1CE4E5B9ull);
    const u64 base_mesh_key =
        index_address ^ (vs_hash * 0x9E3779B97F4A7C15ull) ^ (ps_hash * 0xD6E8FEB86659FD93ull) ^
        (u64(num_indices) * 0xBF58476D1CE4E5B9ull) ^
        (u64(num_instances) * 0x94D049BB133111EBull) ^ off_mix ^
        (vtx_buf_hash * 0xC2B2AE3D27D4EB4Full);

    bool inserted = false;
    u16 e_idx = 0xFFFFu;
    float skel_wx = 0.0f;
    float skel_wy = 0.0f;
    float skel_wz = 0.0f;
    if (is_skeleton) {
        const float* mf = reinterpret_cast<const float*>(palette_ptr + mat_byte_off);
        skel_wx = mf[3];
        skel_wy = mf[7];
        skel_wz = mf[11];
        if (!std::isfinite(skel_wx) || !std::isfinite(skel_wy) || !std::isfinite(skel_wz) ||
            !culler.pub_cam_valid.load(std::memory_order_acquire)) {
            return false;
        }
        const float cdx = skel_wx - culler.pub_cam_pos[0].load(std::memory_order_relaxed);
        const float cdy = skel_wy - culler.pub_cam_pos[1].load(std::memory_order_relaxed);
        const float cdz = skel_wz - culler.pub_cam_pos[2].load(std::memory_order_relaxed);
        if (cdx * cdx + cdy * cdy + cdz * cdz < SceneOcclusionCuller::kMinSkelCullDistSq) {
            return false;
        }
        u64 skel_key = base_mesh_key ^ (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^
                       (u64(palette_size) * 0x94D049BB133111EBull);
        if (skel_key == 0) {
            skel_key = 1;
        }
        culler.stat_gbuf_candidates.fetch_add(1, std::memory_order_relaxed);
        culler.stat_skel_candidates.fetch_add(1, std::memory_order_relaxed);
        e_idx = culler.FindOrInsertSkelEntry(skel_key, mesh_slot, skel_wx, skel_wy, skel_wz,
                                             frame_tag, inserted);
    } else {
        if (num_instances > 1u && !is_inst_gbuf_vs) {
            return false;
        }
        if (culler.pub_cam_valid.load(std::memory_order_acquire) &&
            SceneOcclusionCuller::IsNearCameraWorldObject(
                palette_ptr, palette_size, mat_byte_off, is_inst_gbuf_vs, num_instances,
                inst_cb_ptr, inst_cb_size,
                culler.pub_cam_pos[0].load(std::memory_order_relaxed),
                culler.pub_cam_pos[1].load(std::memory_order_relaxed),
                culler.pub_cam_pos[2].load(std::memory_order_relaxed))) {
            return false;
        }
        culler.stat_gbuf_candidates.fetch_add(1, std::memory_order_relaxed);
        const u32 hash_bytes =
            std::min<u32>(palette_size - mat_byte_off, is_inst_gbuf_vs ? 384u : 192u);
        u64 world_palette_hash = XXH3_64bits_withSeed(
            palette_ptr + mat_byte_off, hash_bytes,
            (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^ u64(palette_size));
        if (inst_cb_ptr && inst_cb_size > 0u) {
            world_palette_hash ^= XXH3_64bits_withSeed(
                inst_cb_ptr, std::min<u32>(inst_cb_size, 1040u), 0xD6E8FEB86659FD93ull);
            if (is_inst_gbuf_vs && num_instances > 1u) {
                const u32 extra_inst = std::min<u32>(num_instances, 4u);
                for (u32 k = 1u; k < extra_inst; ++k) {
                    const u32 ioff = k * 208u + 80u;
                    if (ioff + sizeof(u32) <= inst_cb_size) {
                        u32 ibone = 0u;
                        std::memcpy(&ibone, inst_cb_ptr + ioff, sizeof(u32));
                        if (u64(ibone) * 48ull + 48ull <= u64(palette_size)) {
                            world_palette_hash ^=
                                XXH3_64bits_withSeed(palette_ptr + ibone * 48u, 48u,
                                                     0x9E3779B97F4A7C15ull * u64(k + 1u));
                        }
                    }
                }
            }
        }
        u64 draw_key = base_mesh_key ^ world_palette_hash;
        if (draw_key == 0) {
            draw_key = 1;
        }
        e_idx = culler.FindOrInsertEntry(draw_key, mesh_slot, frame_tag, inserted);
    }

    if (e_idx == 0xFFFFu) {
        return false;
    }

    auto& e = culler.entries[e_idx];
    if (!inserted) {
        const u32 prev_seen = e.last_seen_frame.exchange(frame_tag, std::memory_order_acq_rel);
        if (prev_seen == frame_tag) {
            e.multi_occ.store(8u, std::memory_order_relaxed);
            e.zero_streak.store(0u, std::memory_order_relaxed);
            e.skip_until_frame.store(0u, std::memory_order_relaxed);
            culler.SetEntryOccActive(e, false);
            culler.stat_multi_occ.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (is_skeleton) {
            e.pos_x.store(skel_wx, std::memory_order_relaxed);
            e.pos_y.store(skel_wy, std::memory_order_relaxed);
            e.pos_z.store(skel_wz, std::memory_order_relaxed);
        }
        if (prev_seen != 0u && static_cast<s32>(frame_tag - prev_seen) > 16) {
            e.zero_streak.store(0u, std::memory_order_relaxed);
            e.skip_until_frame.store(0u, std::memory_order_relaxed);
            e.multi_occ.store(0u, std::memory_order_relaxed);
            culler.SetEntryOccActive(e, false);
        } else if (const u8 m = e.multi_occ.load(std::memory_order_relaxed); m > 0u) {
            e.multi_occ.store(static_cast<u8>(m - 1u), std::memory_order_relaxed);
        }
    }

    if (gbuf_ord > 12u && e.multi_occ.load(std::memory_order_relaxed) == 0u &&
        !culler.fast_camera_turn.load(std::memory_order_relaxed)) {
        const u16 zs = e.zero_streak.load(std::memory_order_relaxed);
        const u8 p_if = e.probe_in_flight.load(std::memory_order_acquire);
        if (zs >= SceneOcclusionCuller::kMinZeroStreak && p_if == 0u &&
            frame_tag < e.skip_until_frame.load(std::memory_order_relaxed)) {
            culler.stat_gbuf_skipped.fetch_add(1, std::memory_order_relaxed);
            culler.stat_stage_a_skipped.fetch_add(1, std::memory_order_relaxed);
            if (is_skeleton) {
                culler.stat_skel_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            if (u64 cur = BbStats::draws.load(std::memory_order_relaxed); cur > 0) {
        BbStats::draws.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed);
    }
            return true;
        }
        const u32 probe_mask =
            (is_skeleton || culler.camera_moving.load(std::memory_order_relaxed)) ? 1u : 3u;
        if (p_if == 0u && !Breadcrumbs::Enabled() && !DynCheck::Enabled() &&
            (zs > 0u || ((frame_tag + u32(e_idx)) & probe_mask) == 0u)) {
            out_handoff = e_idx;
        }
    }
    return false;
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const PreparedDraw* prepared) {
    BB_SECTION(ADraw);
    BbStats::draws.fetch_add(1, std::memory_order_relaxed);

    // bbport: with the draw pipeline this thread only selects the pipeline and hands the draw
    // to the recording thread (DrawRecord there); draws FilterDraw handles itself run here.
    const bool pipelined = UseDrawPipe();
    if (pipelined && !FilterDrawPasses() &&
        !BbToggle::Disabled(BbToggle::PipelinedMemoryWrites)) {
        PostDraw(nullptr, nullptr, false, 0);
        return;
    }
    if (!pipelined || !FilterDrawPasses()) {
        DrainDrawPipe(DrawPipe::ReasonDraw);
        BB_SECTION(Task);
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }
    const u32 prep_state =
        prepared ? prepared->state.load(std::memory_order_relaxed) : PreparedDraw::Pending;
    const PreparedDraw* ready_prepared =
        (prep_state == PreparedDraw::Ready) ? prepared : nullptr;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline({}, ready_prepared);
    const PreparedDraw* used_prepared = pipeline_cache.UsedPrepared();
    if (prepared) {
        draw_prep->Count(used_prepared != nullptr,
                         prep_state == PreparedDraw::Pending       ? DrawPreparation::NotReady
                         : prep_state == PreparedDraw::Unavailable ? DrawPreparation::Unavailable
                         : prepared->reg_checksum != liverpool->gfx_reg_checksum
                             ? DrawPreparation::OtherRegisters
                             : DrawPreparation::OtherPrograms);
    }
    if (!pipeline) {
        return;
    }
    if (pipelined && FilterDrawPasses()) {
        const bool run_stage_a_occ =
            scene_occlusion_culler && scene_occlusion_culler->stage_a_occ_cull && OnStageA();
        u16 occ_handoff = run_stage_a_occ ? kOccNoProbe : kOccStageBFallback;
        u8 gbuf_bucket = 0xFFu;
        {
            BB_SECTION(ACullBatch);
            if (vfx_ui_batcher && vfx_ui_batcher->HasPendingDraws() &&
                vfx_ui_batcher->CurrentPipeline() != pipeline) {
                BB_SECTION(ACullVfx);
                vfx_ui_batcher->Flush(*this);
            }
            {
                BB_SECTION(ACullMicro);
                if (OnStageA() && TryCullMicroShadowDraw(pipeline, is_indexed)) {
                    return;
                }
            }
            if (vsm_shadow_memo) {
                BB_SECTION(ACullVsm);
                if (vsm_shadow_memo->TryHandleDraw(*this, pipeline, used_prepared, is_indexed,
                                                   index_offset)) {
                    return;
                }
            }
            if (light_tile_batcher) {
                BB_SECTION(ACullLight);
                if (light_tile_batcher->TryBatchDraw(*this, pipeline, used_prepared, is_indexed,
                                                     index_offset)) {
                    return;
                }
            }
            if (object_vel_culler) {
                BB_SECTION(ACullVel);
                if (object_vel_culler->TryCullDraw(*this, pipeline, is_indexed)) {
                    return;
                }
            }
            if (run_stage_a_occ &&
                TryCullSceneOcclusionStageA(pipeline, used_prepared, is_indexed, index_offset,
                                            occ_handoff, gbuf_bucket)) {
                return;
            }
            if (vfx_ui_batcher) {
                BB_SECTION(ACullVfx);
                if (vfx_ui_batcher->TryBatchDraw(*this, pipeline, used_prepared, is_indexed,
                                                 index_offset)) {
                    return;
                }
            }
        }
        const bool is_display_pass_draw =
            run_stage_a_occ && camera_motion && camera_motion->Enabled() &&
            liverpool->regs.color_buffers[0] &&
            FrameCapture::IsDisplayBuffer(liverpool->regs.color_buffers[0].Address());
        PostDraw(pipeline, used_prepared, is_indexed, index_offset, nullptr, nullptr, nullptr,
                 occ_handoff, gbuf_bucket);
        if (is_display_pass_draw) [[unlikely]] {
            scene_occlusion_culler->NoteStageAFrameStart();
        }
        return;
    }
    if (OnStageA() && FilterDrawPasses()) {
        if (TryCullMicroShadowDraw(pipeline, is_indexed)) {
            return;
        }
        if (object_vel_culler && object_vel_culler->TryCullDraw(*this, pipeline, is_indexed)) {
            return;
        }
    }
    const auto* fast_vs = pipeline_cache.PeekFastVsharps(pipeline);
    packet_vsharps = fast_vs ? std::span<const Gcn::Buffer>{fast_vs->data(), fast_vs->size()}
                             : std::span<const Gcn::Buffer>{};
    DrawRecord(pipeline, used_prepared, is_indexed, index_offset);
    packet_vsharps = {};
}

/// The guest memory a draw reads, in its breadcrumb: a GPU stuck in the draw is reported with it
/// (a stalled read of the game's memory on NVIDIA, whose hangs show no fault).
static void NoteCrumbReads(Breadcrumbs::Crumb& crumb, const GraphicsPipeline* pipeline,
                           const auto& regs, bool is_indexed, u32 index_offset) {
    u32 n = 0;
    const auto add = [&](char kind, u64 address, u32 size) {
        if (address && n < Breadcrumbs::Crumb::MaxReads) {
            crumb.read_kinds[n] = kind;
            crumb.reads[n] = address;
            crumb.read_sizes[n++] = size;
        }
    };
    if (is_indexed) {
        const u32 index_size = regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
        add('i', regs.index_base_address.template Address<VAddr>() + u64(index_offset) * index_size,
            regs.num_indices * index_size);
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    if (const auto& fetch = pipeline->GetFetchShader()) {
        for (const auto& attribute : fetch->attributes) {
            const auto sharp = attribute.GetSharp(vs);
            add('v', sharp.base_address, sharp.GetSize());
        }
    }
    for (const auto* stage : {&vs, pipeline->GetStages()[u32(Shader::SwStage::Fragment)]}) {
        if (!stage) {
            continue;
        }
        for (const auto& buffer : stage->buffers) {
            if (!buffer.IsSpecial()) {
                const auto sharp = buffer.GetSharp(*stage);
                add('b', sharp.base_address, sharp.GetSize());
            }
        }
        for (const auto& image : stage->images) {
            add('t', image.GetSharp(*stage).Address(), 0);
        }
    }
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, const PreparedDraw* used_prepared,
                            bool is_indexed, u32 index_offset, u16 occ_handoff,
                            u8 stage_a_gbuf_bucket) {
    BB_SECTION(DrawRecord);
    packet_vsharps_hash = 0;
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    // bbport (BB_PREUPLOAD): once a frame, guest GPU memory the CPU wrote and then left alone is
    // uploaded ahead of its use, BB_PREUPLOAD_MB (16) at most per frame.
    if (const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
        frame != preupload_frame) {
        preupload_frame = frame;
        BB_SECTION(Preupload);
        static const u64 budget = [] {
            const char* env = std::getenv("BB_PREUPLOAD_MB");
            return (env ? std::strtoull(env, nullptr, 10) : 16ull) << 20;
        }();
        buffer_cache.Preupload(budget);
        prompts->Update();
    }
    const auto& regs = Regs();
    // bbport: the pass copying a finished frame to a display buffer; the previous draw's
    // target is that frame.
    if (camera_motion->Enabled() && regs.color_buffers[0] &&
        FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address())) {
        scene_started = false;
        ui_pass_started = false;
        DumpTracedInputs();
        DumpPassImages('p');
        upscaler->OnDisplayPassStart(regs.color_buffers[0].Address());
        camera_motion->OnDisplayPass(cb_descs[0].first);
        if (upscaler->OnFrameStart()) {
            object_motion->InvalidateHistory();
            camera_motion->InvalidateHistory();
        }
        camera_motion->SetJitter(upscaler->Jitter());
        object_motion->OnFrameStart();
        NoteFrameStart();
        static const char* scene_debug = std::getenv("BB_SCENE_DEBUG");
        scene_debug_frame = scene_debug && std::remove(scene_debug) == 0;
        scene_targets->debug = scene_debug_frame;
    }
    // BB_SKIP_PS=hash[,hash...] (diagnostics): draws with these pixel shaders are left out.
    static const std::vector<u64> skipped_ps = [] {
        std::vector<u64> hashes;
        if (const char* env = std::getenv("BB_SKIP_PS")) {
            for (const char* at = env; *at;) {
                char* end = nullptr;
                hashes.push_back(std::strtoull(at, &end, 16));
                at = *end ? end + 1 : end;
            }
        }
        return hashes;
    }();
    if (!skipped_ps.empty()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        if (ps && std::ranges::find(skipped_ps, ps->pgm_hash) != skipped_ps.end()) {
            return;
        }
    }
    const auto is_attachmentless_noop = [&] {
        if (pipeline->GetStages()[u32(Shader::SwStage::Fragment)] != nullptr) {
            return false;
        }
        const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
        if (vs.pgm_hash != 0x89cda9e4ull) {
            return false;
        }
        const auto& key = pipeline->GetGraphicsKey();
        if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
            for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
                if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb) &&
                    (key.mrt_mask & (1 << cb)) != 0) {
                    return false;
                }
            }
        }
        if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
            (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
            return false;
        }
        return !vs.uses_dma &&
               !std::ranges::any_of(vs.buffers, [](const auto& b) { return b.is_written; }) &&
               !std::ranges::any_of(vs.images, [](const auto& i) { return i.is_written; });
    }();
    if (is_attachmentless_noop) {
        return;
    }

    // Block 10: Smart Unified Scene Occlusion Culling (Micro-Shadow Culling now runs on Stage A).
    s32 occ_probe_entry_idx = -1;
    const u8* gbuf_sort_palette_ptr = nullptr;
    u32 gbuf_sort_palette_size = 0;
    u32 gbuf_sort_mat_byte_off = 0;
    const u8* gbuf_sort_inst_cb_ptr = nullptr;
    u32 gbuf_sort_inst_cb_size = 0;
    bool gbuf_sort_is_inst_vs = false;
    if (scene_occlusion_culler) {
        BB_SECTION(BOccSort);
        auto& culler = *scene_occlusion_culler;
        culler.EnsureInit(instance);
        const u32 frame_tag = culler.cur_frame.load(std::memory_order_relaxed);
        const auto& vs_stage = pipeline->GetStage(Shader::SwStage::Vertex);
        const u64 vs_hash = vs_stage.pgm_hash;
        const u32 num_indices = regs.num_indices;
        const u32 num_instances = regs.num_instances.NumInstances();

        // Hardware Temporal Occlusion Culling for All 6-MRT / 7-MRT (MotionVectors) G-Buffer World,
        // Cutout & Distant (>= 6m) Skinned Character Batches.
        //    Note: 2D screen-space deferred light tiles (vs 0x8747a367) are culled per-pixel on the
        //    GPU via early_fragment_tests + 3D sphere demote in the fragment shaders (0-frame
        //    latency, 100% artifact-free), NOT by skipping 2D tile draws on the CPU.
        const bool has_motion_mrt = pipeline->GetGraphicsKey().motion_vectors;
        const u32 expected_mrt_mask = has_motion_mrt ? 0x7fu : 0x3fu;
        const u32 expected_num_cb = has_motion_mrt ? 7u : 6u;
        const u32 mrt_mask = pipeline->GetGraphicsKey().mrt_mask;
        const auto* ps_stage = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        const bool is_gbuf_cull_cand =
            is_indexed && begin_memo.state.num_color_attachments == expected_num_cb &&
            SceneOcclusionCuller::IsSafeOpaqueGBufferDraw(regs, mrt_mask, expected_mrt_mask) &&
            num_indices >= SceneOcclusionCuller::kMinCullIndices;
        if (culler.enabled && culler.pool && begin_memo.valid &&
            !guest_occlusion_seen.load(std::memory_order_relaxed) &&
            (!occlusion || occlusion->Events() == 0) && is_gbuf_cull_cand) {
            const u32 gbuf_ord = ++culler.gbuf_draws_in_frame;
            if (gbuf_ord == 1) {
                culler.CollectRetiredSlices(instance.GetDevice(), scheduler);
            }
            if (occ_handoff < SceneOcclusionCuller::TotalEntries) {
                if (!DynCheck::Enabled()) [[likely]] {
                    occ_probe_entry_idx = static_cast<s32>(occ_handoff);
                }
            } else if (occ_handoff == kOccStageBFallback) {
                VAddr cue_base = 0;
                VAddr cue_end = 0;
                VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
                bool has_transient_vb = false;
                u64 vtx_buf_hash = 0;
                if (used_prepared && used_prepared->vertex.valid) {
                    vtx_buf_hash = used_prepared->vertex.buffers_hash;
                    for (u32 i = 0; i < used_prepared->vertex.count; ++i) {
                        const VAddr vb_addr = used_prepared->vertex.buffers[i].base_address;
                        if (vb_addr >= cue_base && vb_addr < cue_end) {
                            has_transient_vb = true;
                            break;
                        }
                    }
                } else if (!packet_vsharps.empty()) {
                    for (const auto& vb : packet_vsharps) {
                        if (vb.base_address >= cue_base && vb.base_address < cue_end) {
                            has_transient_vb = true;
                            break;
                        }
                    }
                    if (!has_transient_vb) {
                        vtx_buf_hash = XXH3_64bits(packet_vsharps.data(),
                                                   packet_vsharps.size() * sizeof(Gcn::Buffer));
                        packet_vsharps_hash = vtx_buf_hash;
                    }
                }

            const bool is_inst_gbuf_vs =
                (vs_hash == 0x6a833138ull || vs_hash == 0xecbef4c7ull);
            bool is_skeleton = false;
            bool has_model_cb = false;
            u32 model_base_matrix_idx = 0;
            const u8* inst_cb_ptr = nullptr;
            u32 inst_cb_size = 0;
            const u8* palette_ptr = nullptr;
            u32 palette_size = 0;
            if (!has_transient_vb) {
                const PreparedStage* prep_vs = nullptr;
                if (used_prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
                    for (u32 i = 0; i < used_prepared->num_stages; ++i) {
                        const auto& cand = used_prepared->stages[i];
                        if (&cand.program->info == &vs_stage &&
                            cand.num_buffers == vs_stage.buffers.size()) {
                            prep_vs = &cand;
                            break;
                        }
                    }
                }
                const u32* const vs_flat = vs_stage.FlatUserData().data();
                const VAddr vs_pgm_base = vs_stage.ProgramBase();
                bool need_cam =
                    (culler.cam_updated_frame.load(std::memory_order_relaxed) != frame_tag);
                enum class OccBStageRole : u8 { None = 0, SceneCam, ModelCb, InstCb, Palette };
                for (u32 b_idx = 0; b_idx < vs_stage.buffers.size(); ++b_idx) {
                    if (!need_cam && has_model_cb && palette_ptr) {
                        break;
                    }
                    const auto& res = vs_stage.buffers[b_idx];
                    if (res.IsSpecial()) {
                        continue;
                    }
                    u32 cb_size = 0;
                    const RingBinding* ring =
                        (num_ring_stages != 0) ? FindRingBinding(vs_stage, b_idx) : nullptr;
                    Gcn::Buffer sharp{};
                    if (ring) {
                        cb_size = ring->size;
                    } else {
                        sharp = prep_vs ? prep_vs->buffer_sharps[b_idx]
                                        : res.GetSharpDirect(vs_flat, vs_pgm_base);
                        if (!sharp.Valid() || sharp.GetStride() != 16) {
                            continue;
                        }
                        cb_size = sharp.GetSize();
                    }
                    // Classify by size BEFORE touching `TryGetCpuMirrorPtr` / `OverlapsNativeOwnedRange`
                    // / `ClampRangeSize` so irrelevant material/lighting CBs cost ~1 ns!
                    OccBStageRole role = OccBStageRole::None;
                    if (cb_size == 864u) {
                        if (need_cam) {
                            role = OccBStageRole::SceneCam;
                        }
                    } else if (cb_size == 416u && !is_inst_gbuf_vs) {
                        if (!has_model_cb) {
                            role = OccBStageRole::ModelCb;
                        }
                    } else if (is_inst_gbuf_vs && cb_size >= 208u && cb_size <= 32768u &&
                               (cb_size % 208u) == 0u) {
                        if (!inst_cb_ptr) {
                            role = OccBStageRole::InstCb;
                        }
                    } else if (cb_size >= 48u && cb_size <= 16384u && (cb_size % 48u) == 0u) {
                        if (!palette_ptr) {
                            role = OccBStageRole::Palette;
                        }
                    }
                    if (role == OccBStageRole::None) {
                        continue;
                    }
                    const u8* cb_ptr = nullptr;
                    if (ring) {
                        // NEVER read `constant_ring->Data(ring->offset)` (uncacheable WC/PCIe BAR
                        // memory = ~5 ms/f!) when a cached CPU RAM pointer (`cueHeap`, `AuxCpuRing`,
                        // `CpuMirrorPtr`, or mapped guest RAM) is available.
                        if (ring->address != 0) {
                            if ((ring->address >= cue_base &&
                                 ring->address + cb_size <= cue_end) ||
                                VideoCore::DirectBuffers::IsTransientAuxCpuRing(ring->address,
                                                                                cb_size) ||
                                VideoCore::DirectBuffers::IsCpuReadableImmutableRange(ring->address,
                                                                                      cb_size)) {
                                cb_ptr = reinterpret_cast<const u8*>(ring->address);
                            } else if (const u8* mirror = direct_buffers.TryGetCpuMirrorPtr(
                                           ring->address, cb_size)) {
                                cb_ptr = mirror;
                            } else if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(
                                           ring->address, cb_size) &&
                                       !buffer_cache.IsConstRamRange(ring->address, cb_size) &&
                                       memory->ClampRangeSize(ring->address, cb_size) == cb_size) {
                                cb_ptr = reinterpret_cast<const u8*>(ring->address);
                            }
                        }
                        if (!cb_ptr) {
                            cb_ptr = (ring->ring == 0)
                                         ? constant_ring->Data(ring->offset)
                                         : worker_rings[ring->ring - 1]->Data(ring->offset);
                        }
                    } else {
                        const VAddr cb_va = sharp.base_address;
                        if (cb_va != 0 && cb_size != 0) {
                            if (cb_va >= cue_base && cb_va + cb_size <= cue_end) {
                                cb_ptr = reinterpret_cast<const u8*>(cb_va);
                            } else if (const u8* mirror =
                                           direct_buffers.TryGetCpuMirrorPtr(cb_va, cb_size)) {
                                cb_ptr = mirror;
                            } else if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(
                                           cb_va, cb_size) &&
                                       memory->ClampRangeSize(cb_va, cb_size) == cb_size) {
                                cb_ptr = reinterpret_cast<const u8*>(cb_va);
                            }
                        }
                    }
                    if (!cb_ptr) {
                        continue;
                    }
                    switch (role) {
                    case OccBStageRole::SceneCam:
                        if (reinterpret_cast<const float*>(cb_ptr)[0] == 3000.0f) {
                            culler.UpdateCameraFromSceneConstants(
                                reinterpret_cast<const float*>(cb_ptr), frame_tag);
                            need_cam = false;
                        }
                        break;
                    case OccBStageRole::ModelCb:
                        std::memcpy(&model_base_matrix_idx, cb_ptr + 80, sizeof(u32));
                        has_model_cb = true;
                        break;
                    case OccBStageRole::InstCb:
                        inst_cb_ptr = cb_ptr;
                        inst_cb_size = cb_size;
                        std::memcpy(&model_base_matrix_idx, cb_ptr + 80, sizeof(u32));
                        has_model_cb = true;
                        break;
                    case OccBStageRole::Palette:
                        palette_ptr = cb_ptr;
                        palette_size = cb_size;
                        break;
                    case OccBStageRole::None:
                        break;
                    }
                }
                if (!has_model_cb && palette_size >= 640) {
                    is_skeleton = true;
                } else if (has_model_cb && !is_inst_gbuf_vs && model_base_matrix_idx == 0 &&
                           palette_size >= 640) {
                    is_skeleton = true;
                }
                if (palette_ptr && palette_size >= 48u) {
                    gbuf_sort_palette_ptr = palette_ptr;
                    gbuf_sort_palette_size = palette_size;
                    gbuf_sort_mat_byte_off =
                        (has_model_cb &&
                         u64(model_base_matrix_idx) * 48ull + 48ull <= u64(palette_size))
                            ? (model_base_matrix_idx * 48u)
                            : 0u;
                    gbuf_sort_inst_cb_ptr = inst_cb_ptr;
                    gbuf_sort_inst_cb_size = inst_cb_size;
                    gbuf_sort_is_inst_vs = is_inst_gbuf_vs;
                }
            }

            if (!has_transient_vb && palette_ptr && palette_size >= 48 &&
                (!is_skeleton || culler.skel_occ_cull)) {
                const u32 mat_byte_off =
                    (has_model_cb &&
                     u64(model_base_matrix_idx) * 48ull + 48ull <= u64(palette_size))
                        ? (model_base_matrix_idx * 48u)
                        : 0u;
                const u32 index_size =
                    regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
                const VAddr index_address =
                    regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
                const auto [vtx_off, inst_off] =
                    GetDrawOffsets(regs, vs_stage, pipeline->GetFetchShader());
                const u64 ps_hash = ps_stage ? ps_stage->pgm_hash : 0ull;
                const u64 off_mix = (u64(vtx_off) * 0x9E3779B97F4A7C15ull) ^
                                    (u64(inst_off) * 0xBF58476D1CE4E5B9ull);
                const u64 base_mesh_key =
                    index_address ^ (vs_hash * 0x9E3779B97F4A7C15ull) ^
                    (ps_hash * 0xD6E8FEB86659FD93ull) ^
                    (u64(num_indices) * 0xBF58476D1CE4E5B9ull) ^
                    (u64(num_instances) * 0x94D049BB133111EBull) ^ off_mix ^
                    (vtx_buf_hash * 0xC2B2AE3D27D4EB4Full);

                const u16 mesh_slot = SceneOcclusionCuller::ComputeMeshFilterSlot(
                    index_address, vs_hash, num_indices);
                bool inserted = false;
                u16 e_idx = 0xFFFFu;
                float skel_wx = 0.0f;
                float skel_wy = 0.0f;
                float skel_wz = 0.0f;
                if (is_skeleton) {
                    const float* mf = reinterpret_cast<const float*>(palette_ptr + mat_byte_off);
                    skel_wx = mf[3];
                    skel_wy = mf[7];
                    skel_wz = mf[11];
                    if (std::isfinite(skel_wx) && std::isfinite(skel_wy) &&
                        std::isfinite(skel_wz) &&
                        culler.pub_cam_valid.load(std::memory_order_acquire)) {
                        const float cdx =
                            skel_wx - culler.pub_cam_pos[0].load(std::memory_order_relaxed);
                        const float cdy =
                            skel_wy - culler.pub_cam_pos[1].load(std::memory_order_relaxed);
                        const float cdz =
                            skel_wz - culler.pub_cam_pos[2].load(std::memory_order_relaxed);
                        if (cdx * cdx + cdy * cdy + cdz * cdz >=
                            SceneOcclusionCuller::kMinSkelCullDistSq) {
                            u64 skel_key =
                                base_mesh_key ^
                                (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^
                                (u64(palette_size) * 0x94D049BB133111EBull);
                            if (skel_key == 0) {
                                skel_key = 1;
                            }
                            culler.stat_gbuf_candidates.fetch_add(1, std::memory_order_relaxed);
                            culler.stat_skel_candidates.fetch_add(1, std::memory_order_relaxed);
                            e_idx = culler.FindOrInsertSkelEntry(skel_key, mesh_slot, skel_wx,
                                                                 skel_wy, skel_wz, frame_tag,
                                                                 inserted);
                        }
                    }
                } else if ((num_instances <= 1u || is_inst_gbuf_vs) &&
                           !(culler.pub_cam_valid.load(std::memory_order_acquire) &&
                             SceneOcclusionCuller::IsNearCameraWorldObject(
                                 palette_ptr, palette_size, mat_byte_off, is_inst_gbuf_vs,
                                 num_instances, inst_cb_ptr, inst_cb_size,
                                 culler.pub_cam_pos[0].load(std::memory_order_relaxed),
                                 culler.pub_cam_pos[1].load(std::memory_order_relaxed),
                                 culler.pub_cam_pos[2].load(std::memory_order_relaxed)))) {
                    culler.stat_gbuf_candidates.fetch_add(1, std::memory_order_relaxed);
                    const u32 hash_bytes =
                        std::min<u32>(palette_size - mat_byte_off, is_inst_gbuf_vs ? 384u : 192u);
                    u64 world_palette_hash = XXH3_64bits_withSeed(
                        palette_ptr + mat_byte_off, hash_bytes,
                        (u64(model_base_matrix_idx) * 0x9E3779B97F4A7C15ull) ^ u64(palette_size));
                    if (inst_cb_ptr && inst_cb_size > 0) {
                        world_palette_hash ^= XXH3_64bits_withSeed(
                            inst_cb_ptr, std::min<u32>(inst_cb_size, 1040u),
                            0xD6E8FEB86659FD93ull);
                        if (is_inst_gbuf_vs && num_instances > 1) {
                            const u32 extra_inst = std::min<u32>(num_instances, 4u);
                            for (u32 k = 1u; k < extra_inst; ++k) {
                                const u32 ioff = k * 208u + 80u;
                                if (ioff + sizeof(u32) <= inst_cb_size) {
                                    u32 ibone = 0u;
                                    std::memcpy(&ibone, inst_cb_ptr + ioff, sizeof(u32));
                                    if (u64(ibone) * 48ull + 48ull <= u64(palette_size)) {
                                        world_palette_hash ^= XXH3_64bits_withSeed(
                                            palette_ptr + ibone * 48u, 48u,
                                            0x9E3779B97F4A7C15ull * u64(k + 1u));
                                    }
                                }
                            }
                        }
                    }
                    u64 draw_key = base_mesh_key ^ world_palette_hash;
                    if (draw_key == 0) {
                        draw_key = 1;
                    }
                    e_idx = culler.FindOrInsertEntry(draw_key, mesh_slot, frame_tag, inserted);
                }

                if (e_idx != 0xFFFFu) {
                    auto& e = culler.entries[e_idx];
                    bool is_dup_in_frame = false;
                    if (!inserted) {
                        const u32 prev_seen =
                            e.last_seen_frame.exchange(frame_tag, std::memory_order_acq_rel);
                        if (prev_seen == frame_tag) {
                            e.multi_occ.store(8u, std::memory_order_relaxed);
                            e.zero_streak.store(0u, std::memory_order_relaxed);
                            e.skip_until_frame.store(0u, std::memory_order_relaxed);
                            culler.SetEntryOccActive(e, false);
                            culler.stat_multi_occ.fetch_add(1, std::memory_order_relaxed);
                            is_dup_in_frame = true;
                        } else {
                            if (is_skeleton) {
                                e.pos_x.store(skel_wx, std::memory_order_relaxed);
                                e.pos_y.store(skel_wy, std::memory_order_relaxed);
                                e.pos_z.store(skel_wz, std::memory_order_relaxed);
                            }
                            if (prev_seen != 0 &&
                                static_cast<s32>(frame_tag - prev_seen) > 16) {
                                e.zero_streak.store(0u, std::memory_order_relaxed);
                                e.skip_until_frame.store(0u, std::memory_order_relaxed);
                                e.multi_occ.store(0u, std::memory_order_relaxed);
                                culler.SetEntryOccActive(e, false);
                            } else if (const u8 m = e.multi_occ.load(std::memory_order_relaxed);
                                       m > 0u) {
                                e.multi_occ.store(static_cast<u8>(m - 1u),
                                                  std::memory_order_relaxed);
                            }
                        }
                    }
                    if (!is_dup_in_frame) {
                        if (gbuf_ord > 12 && e.multi_occ.load(std::memory_order_relaxed) == 0u &&
                            !culler.fast_camera_turn.load(std::memory_order_relaxed)) {
                            const u16 zs = e.zero_streak.load(std::memory_order_relaxed);
                            const u8 p_if = e.probe_in_flight.load(std::memory_order_acquire);
                            if (zs >= SceneOcclusionCuller::kMinZeroStreak && p_if == 0u &&
                                frame_tag < e.skip_until_frame.load(std::memory_order_relaxed)) {
                                culler.stat_gbuf_skipped.fetch_add(1, std::memory_order_relaxed);
                                if (is_skeleton) {
                                    culler.stat_skel_skipped.fetch_add(1,
                                                                       std::memory_order_relaxed);
                                }
                                if (u64 cur = BbStats::draws.load(std::memory_order_relaxed); cur > 0) {
        BbStats::draws.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed);
    }
                                return;
                            }
                            const u32 probe_mask =
                                (is_skeleton ||
                                 culler.camera_moving.load(std::memory_order_relaxed))
                                    ? 1u
                                    : 3u;
                            if (p_if == 0u && !DynCheck::Enabled() &&
                                (zs > 0u || ((frame_tag + u32(e_idx)) & probe_mask) == 0u)) {
                                occ_probe_entry_idx = static_cast<s32>(e_idx);
                            }
                        }
                    }
                }
            }
            }
        }
    }

    bind_prepared = used_prepared;
    motion_draw = pipeline->GetGraphicsKey().motion_vectors;
    motion_geometry = 0;

    // BB_EXP_FINISH_BEFORE_TARGET=addr (experiment, the flicker): the GPU runs everything before
    // a draw into that color target before the draw is recorded.
    static const VAddr finish_target = [] {
        const char* env = std::getenv("BB_EXP_FINISH_BEFORE_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (finish_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == finish_target) {
        scheduler.Finish();
    }
    // BB_PASS_DUMP_AT=addr (diagnostics): BB_PASS_DUMP's images also before the first draw of a
    // pass into that color target ('q' files).
    static const VAddr dump_at = [] {
        const char* env = std::getenv("BB_PASS_DUMP_AT");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (dump_at && Regs().color_buffers[0] && Regs().color_buffers[0].Address() == dump_at &&
        !scheduler.IsRendering()) {
        const auto* q_ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        std::printf("Pass dump q stages vs %08x ps %08x\n",
                    u32(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash), u32(q_ps ? q_ps->pgm_hash : 0));
        DumpPassImages('q');
    }
    // BB_EXP_TRANSIT_TARGET=addr (experiment): the target image goes through TransferSrcOptimal
    // before the first draw of a pass into it (what BB_PASS_DUMP_AT does, without the copy).
    static const VAddr transit_target = [] {
        const char* env = std::getenv("BB_EXP_TRANSIT_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (transit_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == transit_target && !scheduler.IsRendering()) {
        VideoCore::Image* found = nullptr;
        texture_cache.ForEachImageInRegion(transit_target, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (image.info.guest_address == transit_target) found = &image;
        });
        if (found) {
            runtime.Transit(found, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
        }
    }
    // BB_EXP_BARRIER_TARGET=addr (experiment): a full memory barrier before each draw into that
    // color target (its render pass is ended first).
    static const VAddr barrier_target = [] {
        const char* env = std::getenv("BB_EXP_BARRIER_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (barrier_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == barrier_target) {
        scheduler.EndRendering();
        scheduler.Record([](vk::CommandBuffer cmd) {
            const vk::MemoryBarrier2 all{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
            };
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &all});
        });
    }
    PrepareRenderState(pipeline);
    NoteUpscalerDraw(pipeline);
    const PreparedDraw* draw_prepared = bind_prepared;
    // bbport: vertex and index buffers are resolved while the helper binds textures (their
    // commands are recorded after BeginRendering, as before).
    draw_inputs = {pipeline, draw_prepared, index_offset, is_indexed, true, false};
    const bool bound = BindResources(pipeline);
    bind_prepared = nullptr; // indirect draws and dispatches bind without prepared sharps
    const bool inputs_resolved = draw_inputs.resolved;
    draw_inputs.pending = false;
    if (!bound) {
        draw_inputs.pipeline = nullptr;
        return;
    }
    const auto state = BeginRendering(pipeline);

    if (!inputs_resolved) {
        ResolveVertexBuffers(pipeline, draw_prepared);
        if (is_indexed) {
            ResolveIndexBuffer(index_offset);
        }
    }
    draw_inputs.pipeline = nullptr;
    const bool will_begin_pass = !last_begin_memo_hit && scheduler.WillBeginRendering(state);
    if (HasPendingHleCopies() && will_begin_pass) {
        FlushPendingHleCopies(*this);
    }
    const u32 cur_draw_in_frame = g_draws_in_frame.fetch_add(1u, std::memory_order_relaxed) + 1u;
    if (cur_draw_in_frame == 1u) {
        g_frame_first_draw_ns.store(SteadyNowNs(), std::memory_order_relaxed);
    }
    static const bool mid_flush_env_forced = std::getenv("BB_MID_FLUSH_DRAWS") != nullptr;
    static const u32 mid_flush_draws = [] {
        if (const char* env = std::getenv("BB_MID_FLUSH_DRAWS")) {
            return static_cast<u32>(std::max(0, std::atoi(env)));
        }
        return 1350u;
    }();
    // Skip the mid-frame submission split when Stage A/B active decode/record is strictly the
    // bottleneck (`tls_avg_decode_ns > tls_avg_gpu_busy_ns + 0.40 ms`), or when Stage B (`DrawPipe`)
    // is already streaming `48 KiB` command buffer chunks continuously to `bb:VkRecorder` on its
    // dedicated core: an extra mid-frame vkQueueSubmit + descriptor epoch reset only adds ~0.25 ms.
    const bool cpu_decode_bound =
        !mid_flush_env_forced &&
        (DrawPipe::OnStageB() ||
         (tls_avg_gpu_busy_ns > 0 && tls_avg_decode_ns > tls_avg_gpu_busy_ns + 400'000ULL));
    if (!HonestLabels() && mid_flush_draws != 0 &&
        !g_mid_frame_flushed.load(std::memory_order_relaxed) && !cpu_decode_bound &&
        cur_draw_in_frame >= mid_flush_draws && will_begin_pass) {
        g_mid_frame_flushed.store(true, std::memory_order_relaxed);
        if (scene_occlusion_culler) {
            scene_occlusion_culler->SealCurrentSlice(scheduler.CurrentTick());
        }
        SubmitInfo info{};
        scheduler.Flush(info);
    }
    if (needs_barrier || (will_begin_pass && runtime.HasPendingBarriers())) {
        if (scheduler.HasPendingReorderBuckets()) {
            scheduler.FlushReorderBuckets();
        }
        runtime.FlushBarriers();
    }

    ScaleToTarget(pipeline);
    if (motion_draw && motion_geometry) {
        push_data.motion_param = 0;
        if (upscaler->Enabled() || camera_motion->DebugOverlay()) {
            BB_SECTION(Motion);
            const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
            const auto [base_vertex, first_instance] =
                GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
            const u32 index_size =
                regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
            const VAddr index_address = is_indexed
                ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size : 0;
            // Without a character-size skeleton the draw is gated on its bone palette changing
            // since last frame (model constants follow the camera, the palette does not). The
            // check comes first: most gated draws are static world pieces with long index lists.
            static const bool all_motion = [] {
                const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
                return value && value[0] == '1';
            }();
            bool gated = !all_motion;
            u64 palette = 0;
            const PreparedStage* prep_vs = nullptr;
            if (draw_prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
                for (u32 i = 0; i < draw_prepared->num_stages; ++i) {
                    const auto& candidate = draw_prepared->stages[i];
                    if (&candidate.program->info == &vs &&
                        candidate.num_buffers == vs.buffers.size()) {
                        prep_vs = &candidate;
                        break;
                    }
                }
            }
            for (u32 b_idx = 0; b_idx < vs.buffers.size(); ++b_idx) {
                const auto& resource = vs.buffers[b_idx];
                if (resource.IsSpecial()) continue;
                const auto buffer = prep_vs ? prep_vs->buffer_sharps[b_idx] : resource.GetSharp(vs);
                const u32 size = buffer.GetSize();
                if (!buffer.Valid() || buffer.GetStride() != 16) continue;
                const auto role = Motion::ClassifyBuffer(size);
                if (role == Motion::BufferRole::Skeleton) {
                    gated = false;
                    break;
                }
                const VAddr address = buffer.base_address;
                if (role == Motion::BufferRole::SmallSkeleton && address != 0) {
                    const u8* pal_ptr = direct_buffers.TryGetCpuMirrorPtr(address, size);
                    if (!pal_ptr &&
                        !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(address, size) &&
                        memory->ClampRangeSize(address, size) == size) {
                        pal_ptr = reinterpret_cast<const u8*>(address);
                    }
                    if (pal_ptr) {
                        palette = XXH3_64bits_withSeed(pal_ptr, size, palette);
                    }
                }
            }
            if (!gated || object_motion->Moving({.shader = vs.pgm_hash,
                                                 .geometry = motion_geometry,
                                                 .indices = index_address,
                                                 .index_count = regs.num_indices,
                                                 .instances = regs.num_instances.NumInstances(),
                                                 .first_instance = first_instance},
                                                palette)) {
                Motion::VertexRange range{base_vertex, regs.num_indices};
                u64 topology = 0;
                if (is_indexed) {
                    range = {};
                    const u64 bytes = u64(regs.num_indices) * index_size;
                    const u8* ib_data =
                        index_address ? direct_buffers.TryGetCpuMirrorPtr(index_address, bytes)
                                      : nullptr;
                    if (!ib_data && index_address &&
                        !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(index_address, bytes) &&
                        memory->ClampRangeSize(index_address, bytes) == bytes) {
                        ib_data = reinterpret_cast<const u8*>(index_address);
                    }
                    if (ib_data) {
                        const bool restart = regs.enable_primitive_restart != 0;
                        const u32 count = regs.num_indices;
                        const auto scanned = object_motion->IndexRange(
                            {index_address, count, index_size, s32(base_vertex), restart}, [&] {
                                const void* data = ib_data;
                                const auto scan_range =
                                    index_size == 2
                                        ? Motion::IndexedRange(
                                              std::span(static_cast<const u16*>(data), count),
                                              s32(base_vertex), restart)
                                        : Motion::IndexedRange(
                                              std::span(static_cast<const u32*>(data), count),
                                              s32(base_vertex), restart);
                                return Motion::IndexRangeCache::Result{scan_range,
                                                                       XXH3_64bits(data, bytes)};
                            });
                        range = scanned.range;
                        topology = scanned.topology;
                    }
                }
                push_data.motion_param = object_motion->PrepareDraw({
                    .shader = vs.pgm_hash,
                    .geometry = motion_geometry,
                    .indices = index_address,
                    .topology = topology,
                    .index_count = regs.num_indices,
                    .instances = regs.num_instances.NumInstances(),
                    .first_instance = first_instance,
                    .vertices = range,
                });
            }
        }
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark before the traced vertex shader's draws too.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = std::getenv("BB_TRACE_SHADER");
        const u64 vs_hash = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash;
        if (traced && std::strtoull(traced, nullptr, 16) == vs_hash) {
            profiler->Mark(vs_hash ^ 0xBEF0'0000ull, [&] {
                return fmt::format("the draw of vs {:016x} (from before its pass begins)", vs_hash);
            });
        }
    }
    // Geometry drawn with the scene depth is jittered, full-screen passes are not (a shifted
    // full-screen quad leaves an edge column unwritten).
    // BB_TRACE_GPU_INPUTS=vs hash (diagnostics, the flicker): the buffer ranges this draw reads
    // (descriptors and vertex streams), copied by the GPU at the next display pass (DumpPassImages).
    if (static const u64 trace_inputs = [] {
            const char* env = std::getenv("BB_TRACE_GPU_INPUTS");
            return env ? std::strtoull(env, nullptr, 16) : 0ull;
        }();
        trace_inputs && pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash == trace_inputs) {
        traced_inputs.clear();
        for (const auto& info : buffer_infos) {
            if (info.buffer && info.range != VK_WHOLE_SIZE && info.range <= 65536) {
                traced_inputs.push_back({info.buffer, info.offset, info.range});
            }
        }
        for (u32 i = 0; i < vertex_binds.num_buffers; ++i) {
            if (vertex_binds.host_buffers[i] && vertex_binds.host_sizes[i] <= 65536) {
                traced_inputs.push_back({vertex_binds.host_buffers[i], vertex_binds.host_offsets[i],
                                         vertex_binds.host_sizes[i]});
            }
        }
        traced_inputs_frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    }
    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps_info_sort = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const bool jitter_eligible =
        vs_info.pgm_hash != 0x000000008747a367ULL &&
        (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1);

    // Step 2.2: Coarse 4-Bucket Front-to-Back G-Buffer Reordering (`Near -> Mid -> Far -> Cutout`).
    // Opt-in only (BB_GBUF_SORT=1): reordering opaque draws across B0 (<20m) and B1 (origin-baked
    // MapPieces) breaks Bloodborne's coplanar GreaterEqual / stencil-ref ground blood decals under
    // shallow camera angles, while SceneOcclusionCuller already skips ~48% of occluded G-Buffer draws.
    static const bool gbuf_sort_enabled = [] {
        const char* env = std::getenv("BB_GBUF_SORT");
        return env && env[0] == '1';
    }();
    u32 gbuf_reorder_bucket = Scheduler::kNoReorderBucket;
    if (gbuf_sort_enabled && is_indexed && !will_begin_pass && !needs_barrier &&
        !runtime.HasPendingBarriers() && scheduler.IsReorderEligible() &&
        !FrameCapture::Active()) {
        BB_SECTION(BOccSort);
        const auto& gkey = pipeline->GetGraphicsKey();
        const u32 expected_mrt_mask = gkey.motion_vectors ? 0x7fu : 0x3fu;
        const u32 expected_num_cb = gkey.motion_vectors ? 7u : 6u;
        if (state.num_color_attachments == expected_num_cb &&
            SceneOcclusionCuller::IsSafeOpaqueGBufferDraw(regs, gkey.mrt_mask, expected_mrt_mask)) {
            const u64 ps_h = ps_info_sort ? ps_info_sort->pgm_hash : 0ull;
            const bool is_cutout_ps =
                (ps_info_sort && ps_info_sort->has_discard) || ps_h == 0x1c3953c6ull ||
                ps_h == 0x70bb42afull || ps_h == 0xdbeccd9cull || ps_h == 0x2140da2bull ||
                ps_h == 0x998b035eull || ps_h == 0xa89e01e9ull || ps_h == 0x0f54e5f9ull;
            if (stage_a_gbuf_bucket < Scheduler::kNumReorderBuckets) [[likely]] {
                gbuf_reorder_bucket = stage_a_gbuf_bucket;
            } else if (is_cutout_ps) {
                gbuf_reorder_bucket = 3u; // Cutout / alpha-tested foliage runs after opaque Hi-Z
            } else {
                // Opaque 6/7-MRT geometry (`layout(early_fragment_tests) in;`): bucket by camera distance.
                if (!gbuf_sort_palette_ptr && scene_occlusion_culler &&
                    scene_occlusion_culler->pub_cam_valid.load(std::memory_order_relaxed)) {
                    const u64 vs_h = vs_info.pgm_hash;
                    const bool is_inst = (vs_h == 0x6a833138ull || vs_h == 0xecbef4c7ull);
                    VAddr cue_base = 0;
                    VAddr cue_end = 0;
                    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
                    const PreparedStage* prep_vs = nullptr;
                    if (draw_prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
                        for (u32 i = 0; i < draw_prepared->num_stages; ++i) {
                            const auto& cand = draw_prepared->stages[i];
                            if (&cand.program->info == &vs_info &&
                                cand.num_buffers == vs_info.buffers.size()) {
                                prep_vs = &cand;
                                break;
                            }
                        }
                    }
                    const u32* const vs_flat = vs_info.FlatUserData().data();
                    const VAddr vs_pgm_base = vs_info.ProgramBase();
                    bool has_mcb = false;
                    u32 m_idx = 0;
                    enum class SortCbRole : u8 { None = 0, ModelCb, InstCb, Palette };
                    for (u32 b_idx = 0; b_idx < vs_info.buffers.size(); ++b_idx) {
                        if (has_mcb && gbuf_sort_palette_ptr) {
                            break;
                        }
                        const auto& res = vs_info.buffers[b_idx];
                        if (res.IsSpecial()) {
                            continue;
                        }
                        u32 cb_sz = 0;
                        const RingBinding* ring =
                            (num_ring_stages != 0) ? FindRingBinding(vs_info, b_idx) : nullptr;
                        Gcn::Buffer sharp{};
                        if (ring) {
                            cb_sz = ring->size;
                        } else {
                            sharp = prep_vs ? prep_vs->buffer_sharps[b_idx]
                                            : res.GetSharpDirect(vs_flat, vs_pgm_base);
                            if (!sharp.Valid() || sharp.GetStride() != 16) {
                                continue;
                            }
                            cb_sz = sharp.GetSize();
                        }
                        SortCbRole role = SortCbRole::None;
                        if (cb_sz == 416u && !is_inst) {
                            if (!has_mcb) {
                                role = SortCbRole::ModelCb;
                            }
                        } else if (is_inst && cb_sz >= 208u && cb_sz <= 32768u &&
                                   (cb_sz % 208u) == 0u) {
                            if (!gbuf_sort_inst_cb_ptr) {
                                role = SortCbRole::InstCb;
                            }
                        } else if (cb_sz >= 48u && cb_sz <= 16384u && (cb_sz % 48u) == 0u) {
                            if (!gbuf_sort_palette_ptr) {
                                role = SortCbRole::Palette;
                            }
                        }
                        if (role == SortCbRole::None) {
                            continue;
                        }
                        const u8* cb_p = nullptr;
                        if (ring) {
                            if (ring->address != 0) {
                                if ((ring->address >= cue_base &&
                                     ring->address + cb_sz <= cue_end) ||
                                    VideoCore::DirectBuffers::IsTransientAuxCpuRing(ring->address,
                                                                                    cb_sz)) {
                                    cb_p = reinterpret_cast<const u8*>(ring->address);
                                } else if (const u8* m = direct_buffers.TryGetCpuMirrorPtr(
                                               ring->address, cb_sz)) {
                                    cb_p = m;
                                } else if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(
                                               ring->address, cb_sz) &&
                                           !buffer_cache.IsConstRamRange(ring->address, cb_sz) &&
                                           memory->ClampRangeSize(ring->address, cb_sz) == cb_sz) {
                                    cb_p = reinterpret_cast<const u8*>(ring->address);
                                }
                            }
                            if (!cb_p) {
                                cb_p = (ring->ring == 0)
                                           ? constant_ring->Data(ring->offset)
                                           : worker_rings[ring->ring - 1]->Data(ring->offset);
                            }
                        } else {
                            const VAddr va = sharp.base_address;
                            if (va != 0 && cb_sz != 0) {
                                if (va >= cue_base && va + cb_sz <= cue_end) {
                                    cb_p = reinterpret_cast<const u8*>(va);
                                } else if (const u8* m =
                                               direct_buffers.TryGetCpuMirrorPtr(va, cb_sz)) {
                                    cb_p = m;
                                } else if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(
                                               va, cb_sz) &&
                                           memory->ClampRangeSize(va, cb_sz) == cb_sz) {
                                    cb_p = reinterpret_cast<const u8*>(va);
                                }
                            }
                        }
                        if (!cb_p) {
                            continue;
                        }
                        if (role == SortCbRole::ModelCb) {
                            std::memcpy(&m_idx, cb_p + 80u, sizeof(u32));
                            has_mcb = true;
                        } else if (role == SortCbRole::InstCb) {
                            gbuf_sort_inst_cb_ptr = cb_p;
                            gbuf_sort_inst_cb_size = cb_sz;
                            std::memcpy(&m_idx, cb_p + 80u, sizeof(u32));
                            has_mcb = true;
                        } else if (role == SortCbRole::Palette) {
                            gbuf_sort_palette_ptr = cb_p;
                            gbuf_sort_palette_size = cb_sz;
                        }
                    }
                    if (gbuf_sort_palette_ptr && gbuf_sort_palette_size >= 48u) {
                        gbuf_sort_mat_byte_off =
                            (has_mcb && u64(m_idx) * 48ull + 48ull <= u64(gbuf_sort_palette_size))
                                ? (m_idx * 48u)
                                : 0u;
                        gbuf_sort_is_inst_vs = is_inst;
                    }
                }
                gbuf_reorder_bucket = 1u; // Default: MidOpaque (including origin-baked MapPieces)
                if (scene_occlusion_culler &&
                    scene_occlusion_culler->pub_cam_valid.load(std::memory_order_relaxed) &&
                    gbuf_sort_palette_ptr &&
                    u64(gbuf_sort_mat_byte_off) + 48ull <= u64(gbuf_sort_palette_size)) {
                    const float cx =
                        scene_occlusion_culler->pub_cam_pos[0].load(std::memory_order_relaxed);
                    const float cy =
                        scene_occlusion_culler->pub_cam_pos[1].load(std::memory_order_relaxed);
                    const float cz =
                        scene_occlusion_culler->pub_cam_pos[2].load(std::memory_order_relaxed);
                    const float* mf =
                        reinterpret_cast<const float*>(gbuf_sort_palette_ptr + gbuf_sort_mat_byte_off);
                    const float wx = mf[3];
                    const float wy = mf[7];
                    const float wz = mf[11];
                    if ((wx != 0.0f || wy != 0.0f || wz != 0.0f) && std::isfinite(wx) &&
                        std::isfinite(wy) && std::isfinite(wz)) {
                        float dx = wx - cx;
                        float dy = wy - cy;
                        float dz = wz - cz;
                        float min_dist_sq = dx * dx + dy * dy + dz * dz;
                        const u32 n_inst = regs.num_instances.NumInstances();
                        if (gbuf_sort_is_inst_vs && n_inst > 1u && gbuf_sort_inst_cb_ptr) {
                            const u32 check_n = std::min<u32>(n_inst, 4u);
                            for (u32 k = 1u; k < check_n; ++k) {
                                const u32 ioff = k * 208u + 80u;
                                if (ioff + sizeof(u32) <= gbuf_sort_inst_cb_size) {
                                    u32 ibone = 0u;
                                    std::memcpy(&ibone, gbuf_sort_inst_cb_ptr + ioff, sizeof(u32));
                                    if (u64(ibone) * 48ull + 48ull <= u64(gbuf_sort_palette_size)) {
                                        const float* imf = reinterpret_cast<const float*>(
                                            gbuf_sort_palette_ptr + ibone * 48u);
                                        if ((imf[3] != 0.0f || imf[7] != 0.0f || imf[11] != 0.0f) &&
                                            std::isfinite(imf[3]) && std::isfinite(imf[7]) &&
                                            std::isfinite(imf[11])) {
                                            const float idx_x = imf[3] - cx;
                                            const float idx_y = imf[7] - cy;
                                            const float idx_z = imf[11] - cz;
                                            const float d2 =
                                                idx_x * idx_x + idx_y * idx_y + idx_z * idx_z;
                                            if (d2 < min_dist_sq) {
                                                min_dist_sq = d2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (min_dist_sq < 20.0f * 20.0f) {
                            gbuf_reorder_bucket = 0u; // NearOpaque (< 20m)
                        } else if (min_dist_sq < 50.0f * 50.0f) {
                            gbuf_reorder_bucket = 1u; // MidOpaque (20..50m)
                        } else {
                            gbuf_reorder_bucket = 2u; // FarOpaque (>= 50m)
                        }
                    }
                }
            }
        }
    }

    if (gbuf_reorder_bucket < Scheduler::kNumReorderBuckets) {
        scheduler.EnterReorderBucket(gbuf_reorder_bucket);
    } else if (scheduler.HasPendingReorderBuckets()) {
        scheduler.FlushReorderBuckets();
    }

    EmitVertexBuffers();
    if (is_indexed) {
        BB_SECTION(BDrawEmit);
        EmitIndexBuffer();
    }
    BeginDraw(pipeline, state, is_indexed, jitter_eligible, draw_prepared, index_offset, true);

    const u32 num_indices = regs.num_indices;
    const u32 num_instances = regs.num_instances.NumInstances();
    {
        BB_SECTION(BDrawEmit);
        const auto& fetch_shader = pipeline->GetFetchShader();
        const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

        const vk::Pipeline handle = pipeline->Handle();
        // BB_TRACE_PIPELINE=vs hash (diagnostics, the flicker): its pipeline handle and push data,
        // printed when they differ from its previous draw.
        if (static const u64 traced_pipeline = [] {
                const char* env = std::getenv("BB_TRACE_PIPELINE");
                return env ? std::strtoull(env, nullptr, 16) : 0ull;
            }();
            traced_pipeline && vs_info.pgm_hash == traced_pipeline) {
            static VkPipeline last_handle{};
            static std::array<float, 4> last_push{};
            const std::array<float, 4> push{push_data.xscale, push_data.xoffset, push_data.yscale,
                                            push_data.yoffset};
            static u32 prints = 0;
            if ((VkPipeline(handle) != last_handle || push != last_push) && prints++ < 2000) {
                std::printf("Pipeline trace: gpu frame %llu vs %016llx handle %p push %g %g %g %g "
                            "clipoff %d vte %#x\n",
                            (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                            (unsigned long long)vs_info.pgm_hash, (void*)VkPipeline(handle),
                            push[0], push[1], push[2], push[3], int(regs.IsClipDisabled()),
                            std::bit_cast<u32>(regs.viewport_control));
                last_handle = handle;
                last_push = push;
            }
        }
        const u32 first_vertex = vertex_offset;
        const u32 first_instance = instance_offset;
        if (!Breadcrumbs::Enabled() && !DynCheck::Enabled()) [[likely]] {
            const u64 cur_epoch = scheduler.DescriptorStateEpoch();
            vk::Pipeline& tracked_pipe =
                (gbuf_reorder_bucket < Scheduler::kNumReorderBuckets)
                    ? bound_bucket_pipeline[gbuf_reorder_bucket]
                    : bound_graphics_pipeline;
            u64& tracked_epoch =
                (gbuf_reorder_bucket < Scheduler::kNumReorderBuckets)
                    ? bound_bucket_epoch[gbuf_reorder_bucket]
                    : bound_graphics_epoch;
            const bool need_bind_pipeline =
                (tracked_epoch != cur_epoch) || (tracked_pipe != handle);
            tracked_pipe = handle;
            tracked_epoch = cur_epoch;
            const s32 hw_occ_query_slot =
                (is_indexed && occ_probe_entry_idx >= 0 && scene_occlusion_culler)
                    ? scene_occlusion_culler->AllocateQuery(static_cast<u16>(occ_probe_entry_idx))
                    : -1;
            if (is_indexed) {
                if (hw_occ_query_slot >= 0) {
                    const vk::QueryPool occ_pool = *scene_occlusion_culler->pool;
                    const u32 q_slot = static_cast<u32>(hw_occ_query_slot);
                    scheduler.Record([handle, need_bind_pipeline, occ_pool, q_slot, num_indices,
                                      num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        if (need_bind_pipeline) {
                            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                        }
                        cmdbuf.beginQuery(occ_pool, q_slot, {});
                        cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                           first_instance);
                        cmdbuf.endQuery(occ_pool, q_slot);
                    });
                } else if (need_bind_pipeline) {
                    scheduler.Record([handle, num_indices, num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                        cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                           first_instance);
                    });
                } else {
                    scheduler.Record([num_indices, num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                           first_instance);
                    });
                }
            } else {
                if (hw_occ_query_slot >= 0) {
                    const vk::QueryPool occ_pool = *scene_occlusion_culler->pool;
                    const u32 q_slot = static_cast<u32>(hw_occ_query_slot);
                    scheduler.Record([handle, need_bind_pipeline, occ_pool, q_slot, num_indices,
                                      num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        if (need_bind_pipeline) {
                            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                        }
                        cmdbuf.beginQuery(occ_pool, q_slot, {});
                        cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
                        cmdbuf.endQuery(occ_pool, q_slot);
                    });
                } else if (need_bind_pipeline) {
                    scheduler.Record([handle, num_indices, num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                        cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
                    });
                } else {
                    scheduler.Record([num_indices, num_instances, first_vertex,
                                      first_instance](vk::CommandBuffer cmdbuf) {
                        cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
                    });
                }
            }
        } else {
        bound_graphics_pipeline = handle;
        bound_graphics_epoch = scheduler.DescriptorStateEpoch();
        const auto* ps_info = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        Breadcrumbs::Crumb crumb{
            .kind = is_indexed ? Breadcrumbs::Kind::DrawIndexed : Breadcrumbs::Kind::Draw,
            .hash = {vs_info.pgm_hash, ps_info ? ps_info->pgm_hash : 0},
            .program = {vs_info.ProgramBase(), ps_info ? ps_info->ProgramBase() : 0},
            .count = {num_indices, num_instances, 0},
        };
        static const bool crumb_reads = [&] {
            const char* env = std::getenv("BB_BREADCRUMB_READS");
            return env ? env[0] == '1' : instance.GetVendorID() == 0x10de;
        }();
        if (crumb_reads && Breadcrumbs::Enabled()) {
            NoteCrumbReads(crumb, pipeline, regs, is_indexed, index_offset);
        }
        const Scissors check_scissors =
            DynCheck::Enabled() ? scheduler.GetDynamicState().CurrentScissors() : Scissors{};
        const Viewports check_viewports =
            DynCheck::Enabled() ? scheduler.GetDynamicState().CurrentViewports() : Viewports{};
        const u64 check_vs = vs_info.pgm_hash, check_ps = ps_info ? ps_info->pgm_hash : 0;
        scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            if (DynCheck::Enabled()) {
                DynCheck::Draw(cmdbuf, check_viewports, check_scissors, check_vs, check_ps);
            }
            if (is_indexed) {
                cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                   first_instance);
            } else {
                cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
            }
        });
    }
    if (gbuf_reorder_bucket < Scheduler::kNumReorderBuckets) {
        scheduler.LeaveReorderBucket();
    }
    if (ps_info_sort && ps_info_sort->pgm_hash == 0x0000000032e47788ULL && native_postfx &&
        state.num_color_attachments > 0u && cb_descs[0].first) {
        const auto& s = BbSettings::Get();
        if (s.ssgi_enabled.load(std::memory_order_relaxed) &&
            state.color_attachments[0].image_view && buffer_infos.size() >= 2u) {
            vk::ImageView ps_imgs[6]{};
            u32 img_count = 0u;
            for (const auto& w : set_writes) {
                if (w.descriptorType == vk::DescriptorType::eSampledImage && w.pImageInfo) {
                    if (img_count < 6u) {
                        ps_imgs[img_count++] = w.pImageInfo->imageView;
                    } else {
                        for (u32 k = 0u; k < 5u; ++k) ps_imgs[k] = ps_imgs[k + 1u];
                        ps_imgs[5] = w.pImageInfo->imageView;
                    }
                }
            }
            if (img_count == 6u) {
                const u32 ssbo2_slot = static_cast<u32>(buffer_infos.size() - 1u);
                const u32 d1_dw = static_cast<u32>(push_data.buf_offsets[ssbo2_slot]) >> 2u;
                native_postfx->ExecuteHalfResSsgi(
                    texture_cache.GetImage(cb_descs[0].first),
                    state.color_attachments[0].image_view,
                    ps_imgs[0], ps_imgs[1], ps_imgs[2], ps_imgs[3], ps_imgs[5],
                    buffer_infos[ssbo2_slot], d1_dw);
            }
        }
        if (s.native_postfx.load(std::memory_order_relaxed) &&
            s.fire_heat_haze.load(std::memory_order_relaxed)) {
            native_postfx->CaptureOpaqueHdr(texture_cache.GetImage(cb_descs[0].first));
        }
    }
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark right after the traced vertex shader's draws,
    // so the profile splits their pass into the draw and what follows it unlabeled.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = std::getenv("BB_TRACE_SHADER");
        if (traced && std::strtoull(traced, nullptr, 16) == vs_info.pgm_hash) {
            profiler->Mark(vs_info.pgm_hash ^ 0xAF7E'0000ull, [&] {
                return fmt::format("after the draw of vs {:016x}", vs_info.pgm_hash);
            });
        }
    }
    if (FrameCapture::Active()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        FrameCapture::Draw(vs_info.pgm_hash, ps ? ps->pgm_hash : 0, num_indices, num_instances);
        if (const auto& bc = regs.blend_control[0]; bc.enable && regs.color_buffers[0]) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  blend ps %08x idx %u: src %u dst %u func %u z %u%u",
                          u32(ps ? ps->pgm_hash : 0), num_indices, u32(bc.color_src_factor),
                          u32(bc.color_dst_factor), u32(bc.color_func),
                          u32(regs.depth_control.depth_enable),
                          u32(regs.depth_control.depth_write_enable));
            FrameCapture::Note(note);
        }
    }

    static const bool watch = [] {
        const char* env = std::getenv("BB_DRAW_WATCH");
        return env && env[0] == '1';
    }();
    if (watch) {
        WatchDraw(pipeline, state);
    }

    BB_SECTION(Kick);
    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::NoteUpscalerDraw(const GraphicsPipeline* pipeline) {
    // Single-target passes: the upscaler looks for the scene's and the UI's (UiComposition).
    if (std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1) {
        const u64 vs_hash = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash;
        if (UiComposition::MovieShader(vs_hash)) {
            ui_pass_started = true;
        }
        if (upscaler->Enabled()) {
            const auto& viewport = Regs().viewports[0];
            upscaler->OnDraw(vs_hash, cb_descs[0].first, db_desc.first,
                             UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
            if (upscaler->InUiPhase()) {
                ui_pass_started = true;
            }
        }
    } else {
        ui_pass_started = false;
    }
}

void Rasterizer::ScaleToTarget(const GraphicsPipeline* pipeline) {
    UiEdgeStretch(pipeline);
    // Screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
    if (!frame_fx_mask_valid) [[unlikely]] {
        const auto& settings = BbSettings::Get();
        u8 fx_mask = 0;
        if (settings.ssao_quality.load(std::memory_order_relaxed))    fx_mask |= 0x01u;
        if (settings.contact_shadows.load(std::memory_order_relaxed)) fx_mask |= 0x02u;
        if (settings.ssr_quality.load(std::memory_order_relaxed))     fx_mask |= 0x04u;
        if (settings.true_black_hdr.load(std::memory_order_relaxed))  fx_mask |= 0x08u;
        if (settings.pcss_shadows.load(std::memory_order_relaxed))    fx_mask |= 0x10u;
        if (settings.ssgi_enabled.load(std::memory_order_relaxed))    fx_mask |= 0x20u;
        frame_fx_mask = fx_mask;
        const u8 mb_cam = static_cast<u8>(
            std::clamp(settings.motion_blur_camera.load(std::memory_order_relaxed), 0, 3));
        const u8 mb_obj = static_cast<u8>(
            std::clamp(settings.motion_blur_object.load(std::memory_order_relaxed), 0, 3));
        const u8 mb_native_bit =
            settings.native_postfx.load(std::memory_order_relaxed) ? 0x10u : 0x00u;
        frame_mb_mask = static_cast<u8>((mb_cam & 0x3u) | ((mb_obj & 0x3u) << 2u) | mb_native_bit);
        const float mb_frames = std::clamp(BbFrameRate::CurrentFramesPer30(), 0.05f, 1.5f);
        const float mb_fps_scale = std::clamp(1.0f / mb_frames, 1.0f, 12.0f);
        frame_mb_fps_q4 = static_cast<u8>(
            std::clamp(static_cast<int>(std::lround(mb_fps_scale * 16.0f)), 16, 255));
        frame_fx_mask_valid = true;
    }
    push_data.buf_offsets[37] = frame_mb_fps_q4;
    push_data.buf_offsets[38] = frame_mb_mask;
    push_data.buf_offsets[39] = frame_fx_mask;
}

namespace {

constexpr u32 RequiredFormatWriteMask(vk::Format fmt) noexcept {
    switch (fmt) {
    case vk::Format::eR8Unorm:
    case vk::Format::eR8Snorm:
    case vk::Format::eR8Uint:
    case vk::Format::eR8Sint:
    case vk::Format::eR8Srgb:
    case vk::Format::eR16Unorm:
    case vk::Format::eR16Snorm:
    case vk::Format::eR16Uint:
    case vk::Format::eR16Sint:
    case vk::Format::eR16Sfloat:
    case vk::Format::eR32Uint:
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
        return 0x1u;
    case vk::Format::eR8G8Unorm:
    case vk::Format::eR8G8Snorm:
    case vk::Format::eR8G8Uint:
    case vk::Format::eR8G8Sint:
    case vk::Format::eR8G8Srgb:
    case vk::Format::eR16G16Unorm:
    case vk::Format::eR16G16Snorm:
    case vk::Format::eR16G16Uint:
    case vk::Format::eR16G16Sint:
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR32G32Uint:
    case vk::Format::eR32G32Sint:
    case vk::Format::eR32G32Sfloat:
        return 0x3u;
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR5G6B5UnormPack16:
    case vk::Format::eB5G6R5UnormPack16:
        return 0x7u;
    default:
        return 0xFu;
    }
}

float HalfToFloat(u16 h) noexcept {
    const u32 sign = u32(h & 0x8000u) << 16;
    const u32 exp = (h >> 10) & 0x1Fu;
    const u32 mant = h & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);
        }
        const float mag = float(mant) * (1.0f / float(1u << 24));
        return (sign != 0) ? -mag : mag;
    }
    if (exp == 31) {
        return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
    }
    return std::bit_cast<float>(sign | ((exp + 112u) << 23) | (mant << 13));
}

bool DecodeFloatVertexAttr(const Gcn::Buffer& vb, const u8* ptr, u64 avail_bytes,
                           std::array<float, 4>& out_xyzw) noexcept {
    const auto dfmt = vb.GetDataFmt();
    const auto nfmt = vb.GetNumberFmt();
    std::array<float, 4> raw{0.0f, 0.0f, 0.0f, 1.0f};
    if (nfmt == Gcn::NumberFormat::Float) {
        switch (dfmt) {
        case Gcn::DataFormat::Format32_32_32_32:
            if (avail_bytes < 16) return false;
            std::memcpy(raw.data(), ptr, 16);
            break;
        case Gcn::DataFormat::Format32_32_32:
            if (avail_bytes < 12) return false;
            std::memcpy(raw.data(), ptr, 12);
            break;
        case Gcn::DataFormat::Format32_32:
            if (avail_bytes < 8) return false;
            std::memcpy(raw.data(), ptr, 8);
            break;
        case Gcn::DataFormat::Format32:
            if (avail_bytes < 4) return false;
            std::memcpy(raw.data(), ptr, 4);
            break;
        case Gcn::DataFormat::Format16_16_16_16: {
            if (avail_bytes < 8) return false;
            u16 h[4];
            std::memcpy(h, ptr, 8);
            for (u32 i = 0; i < 4; ++i) raw[i] = HalfToFloat(h[i]);
            break;
        }
        case Gcn::DataFormat::Format16_16: {
            if (avail_bytes < 4) return false;
            u16 h[2];
            std::memcpy(h, ptr, 4);
            raw[0] = HalfToFloat(h[0]);
            raw[1] = HalfToFloat(h[1]);
            break;
        }
        default:
            return false;
        }
    } else if (nfmt == Gcn::NumberFormat::Snorm && dfmt == Gcn::DataFormat::Format16_16_16_16) {
        if (avail_bytes < 8) return false;
        s16 s[4];
        std::memcpy(s, ptr, 8);
        for (u32 i = 0; i < 4; ++i) {
            raw[i] = std::max(-1.0f, float(s[i]) * (1.0f / 32767.0f));
        }
    } else if (nfmt == Gcn::NumberFormat::Snorm && dfmt == Gcn::DataFormat::Format16_16) {
        if (avail_bytes < 4) return false;
        s16 s[2];
        std::memcpy(s, ptr, 4);
        raw[0] = std::max(-1.0f, float(s[0]) * (1.0f / 32767.0f));
        raw[1] = std::max(-1.0f, float(s[1]) * (1.0f / 32767.0f));
    } else {
        return false;
    }
    out_xyzw = vb.DstSelect().Apply(raw);
    return true;
}

struct WinPt {
    float x = 0.0f;
    float y = 0.0f;
};

float Cross2D(WinPt a, WinPt b, WinPt p) noexcept {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

bool SameWinPt(WinPt a, WinPt b) noexcept {
    return std::abs(a.x - b.x) <= 2e-2f && std::abs(a.y - b.y) <= 2e-2f;
}

bool ConvexPolyContainsRect(std::span<const WinPt> poly, float orient_sign, float w,
                            float h) noexcept {
    const std::array<WinPt, 4> corners{{{0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h}}};
    const size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        const WinPt a = poly[i];
        const WinPt b = poly[(i + 1) % n];
        const float edge_l1 = std::abs(b.x - a.x) + std::abs(b.y - a.y);
        if (edge_l1 <= 1e-3f) {
            return false;
        }
        const float tol = 2e-2f * edge_l1;
        for (const WinPt c : corners) {
            if (orient_sign * Cross2D(a, b, c) < -tol) {
                return false;
            }
        }
    }
    return true;
}

struct WinTri {
    std::array<WinPt, 3> v{};
    float det = 0.0f;
};

bool TriangleCoversRect(const WinTri& t, float w, float h) noexcept {
    return ConvexPolyContainsRect(t.v, t.det > 0.0f ? 1.0f : -1.0f, w, h);
}

bool TwoTrianglesCoverRect(const WinTri& t0, const WinTri& t1, float w, float h) noexcept {
    if (TriangleCoversRect(t0, w, h) || TriangleCoversRect(t1, w, h)) {
        return true;
    }
    int m0_a = -1, m0_b = -1, m1_a = -1, m1_b = -1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (SameWinPt(t0.v[i], t1.v[j])) {
                if (m0_a == -1) {
                    m0_a = i;
                    m1_a = j;
                } else if (i != m0_a && j != m1_a) {
                    m0_b = i;
                    m1_b = j;
                }
            }
        }
    }
    if (m0_a == -1 || m0_b == -1) {
        return false;
    }
    const int p_idx = 3 - m0_a - m0_b;
    const int q_idx = 3 - m1_a - m1_b;
    const WinPt u = t0.v[m0_a];
    const WinPt v = t0.v[m0_b];
    const WinPt p = t0.v[p_idx];
    const WinPt q = t1.v[q_idx];
    const float cp_u_v_p = Cross2D(u, v, p);
    const float cp_u_v_q = Cross2D(u, v, q);
    // P and Q must lie strictly on opposite sides of shared diagonal UV.
    if (!((cp_u_v_p > 1e-2f && cp_u_v_q < -1e-2f) || (cp_u_v_p < -1e-2f && cp_u_v_q > 1e-2f))) {
        return false;
    }
    const float cp_p_q_u = Cross2D(p, q, u);
    const float cp_p_q_v = Cross2D(p, q, v);
    // U and V must lie strictly on opposite sides of PQ so U -> P -> V -> Q is strictly convex.
    if (!((cp_p_q_u > 1e-2f && cp_p_q_v < -1e-2f) || (cp_p_q_u < -1e-2f && cp_p_q_v > 1e-2f))) {
        return false;
    }
    const std::array<WinPt, 4> quad{u, p, v, q};
    // Along U -> P -> V, point V lies on the side sign(Cross2D(u, p, v)) = sign(-cp_u_v_p).
    const float orient_sign = (-cp_u_v_p > 0.0f) ? 1.0f : -1.0f;
    return ConvexPolyContainsRect(quad, orient_sign, w, h);
}

} // namespace

u8 Rasterizer::ComputeFullscreenOverwriteMask(const GraphicsPipeline* pipeline,
                                              const RenderState& state,
                                              const PreparedDraw* draw_prepared, bool is_indexed,
                                              u32 index_offset) {
    static const bool dont_care_off = [] {
        const char* env = std::getenv("BB_DONT_CARE_OFF");
        return env && env[0] == '1';
    }();
    if (dont_care_off || !pipeline) {
        return 0;
    }
    if (state.num_color_attachments == 0 || state.num_color_attachments > Gcn::NUM_COLOR_BUFFERS ||
        state.width == 0 || state.height == 0 || state.num_layers != 1) {
        return 0;
    }
    const auto& regs = Regs();
    const auto& key = pipeline->GetGraphicsKey();
    if (key.num_samples != 1 || attachment_feedback_loop ||
        ui_placement.edge_x != 0.0f || ui_placement.edge_y != 0.0f) {
        return 0;
    }
    if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Normal ||
        key.logic_op != Gcn::ColorControl::LogicOp::Copy ||
        regs.clipper_control.dx_rasterization_kill ||
        regs.clipper_control.user_clip_plane_enable != 0 ||
        regs.vs_output_control.clip_distance_enable != 0 ||
        regs.vs_output_control.cull_distance_enable != 0 ||
        regs.vs_output_control.use_vtx_kill_flag ||
        regs.stage_enable.raw != Gcn::ShaderStageEnable::VgtStages::Vs ||
        key.polygon_mode != Gcn::PolygonMode::Fill) {
        return 0;
    }
    if (regs.depth_control.depth_enable || regs.depth_control.stencil_enable ||
        regs.depth_control.depth_bounds_enable) {
        return 0;
    }
    const u32 count = regs.num_indices;
    if (regs.num_instances.NumInstances() == 0 || count < 3 || count > 6) {
        return 0;
    }
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    if (!ps || ps->has_discard) {
        return 0;
    }

    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    enum class VsKind : u8 {
        None,
        Procedural89cda9e4,
        PassThroughXYZW,
        PassThroughXYZ1,
        PassThroughXYCb0Z_W,
        YebisCb20_21ScaleXY,
    };
    VsKind vs_kind = VsKind::None;
    switch (vs.pgm_hash) {
    case 0x89cda9e4ull:
        vs_kind = VsKind::Procedural89cda9e4;
        break;
    case 0x0b0acf50ull:
    case 0x061fbbf6ull:
        vs_kind = VsKind::PassThroughXYZW;
        break;
    case 0x637500e0ull:
        vs_kind = VsKind::PassThroughXYZ1;
        break;
    case 0x8747a367ull:
        vs_kind = VsKind::PassThroughXYCb0Z_W;
        break;
    case 0x7d668276ull:
    case 0x1ea5a07eull:
    case 0xd3ca03f3ull:
        vs_kind = VsKind::YebisCb20_21ScaleXY;
        break;
    default:
        return 0;
    }

    u8 candidate_mask = 0;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        const auto& att = state.color_attachments[cb];
        if (att.is_clear || !att.image_view || (key.mrt_mask & (1u << cb)) == 0 ||
            !cb_descs[cb].first) {
            continue;
        }
        if (key.blend_controls[cb].enable) {
            continue;
        }
        const auto& img = texture_cache.GetImage(cb_descs[cb].first);
        const u32 req_mask = RequiredFormatWriteMask(img.info.pixel_format);
        if ((u32(key.write_masks[cb]) & req_mask) != req_mask) {
            continue;
        }
        candidate_mask |= u8(1u << cb);
    }
    if (candidate_mask == 0) {
        return 0;
    }

    const auto& ds = scheduler.GetDynamicState();
    const auto& scs = ds.CurrentScissors();
    const auto& vps = ds.CurrentViewports();
    if (scs.empty() || vps.empty()) {
        return 0;
    }
    if (scs[0].offset.x > 0 || scs[0].offset.y > 0 ||
        s64(scs[0].offset.x) + s64(scs[0].extent.width) < s64(state.width) ||
        s64(scs[0].offset.y) + s64(scs[0].extent.height) < s64(state.height)) {
        return 0;
    }
    const float w_f = float(state.width);
    const float h_f = float(state.height);
    const bool clip_disabled = regs.IsClipDisabled();
    const vk::Viewport& vp = vps[0];
    if (!clip_disabled) {
        const float vp_x0 = std::min(vp.x, vp.x + vp.width);
        const float vp_x1 = std::max(vp.x, vp.x + vp.width);
        const float vp_y0 = std::min(vp.y, vp.y + vp.height);
        const float vp_y1 = std::max(vp.y, vp.y + vp.height);
        if (vp_x0 > 1e-2f || vp_x1 < w_f - 1e-2f || vp_y0 > 1e-2f || vp_y1 < h_f - 1e-2f) {
            return 0;
        }
    } else if (vp.x > 1e-2f || vp.y > 1e-2f || vp.width < w_f - 1e-2f || vp.height < h_f - 1e-2f) {
        return 0;
    }

    const auto resolve_cpu_ptr = [&](VAddr addr, u64 bytes) -> const u8* {
        if (addr == 0 || bytes == 0) {
            return nullptr;
        }
        if (HasPendingHleCopies() && PendingHleCopiesConflict(addr, bytes, false)) {
            return nullptr;
        }
        if (const u8* mirror = direct_buffers.TryGetCpuMirrorPtr(addr, bytes)) {
            return mirror;
        }
        if (VideoCore::DirectBuffers::OverlapsNativeOwnedRange(addr, bytes)) {
            return nullptr;
        }
        const bool gpu_clean =
            VideoCore::DirectBuffers::IsTransientCueHeap(addr, bytes)
                ? (!VideoCore::DirectBuffers::IsCueHeapEverGpuModified() ||
                   !buffer_cache.IsRegionGpuModified(addr, bytes))
                : !buffer_cache.IsRegionGpuModified(addr, bytes);
        if (!gpu_clean || memory->ClampRangeSize(addr, bytes) != bytes) {
            return nullptr;
        }
        return reinterpret_cast<const u8*>(addr);
    };

    const auto& fetch = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    (void)instance_offset;

    std::array<u32, 6> vtx_indices{};
    if (!is_indexed) {
        for (u32 i = 0; i < count; ++i) {
            vtx_indices[i] = vertex_offset + i;
        }
    } else {
        const bool index16 = regs.index_buffer_type.index_type == Gcn::IndexType::Index16;
        const u32 idx_size = index16 ? sizeof(u16) : sizeof(u32);
        const VAddr ib_addr =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * idx_size;
        const u64 ib_bytes = u64(count) * idx_size;
        const u8* ib_ptr = resolve_cpu_ptr(ib_addr, ib_bytes);
        if (!ib_ptr) {
            return 0;
        }
        const bool prim_restart = (regs.enable_primitive_restart & 1) != 0;
        const s64 v_off = static_cast<s64>(static_cast<s32>(vertex_offset));
        for (u32 i = 0; i < count; ++i) {
            u32 raw_idx = 0;
            if (index16) {
                u16 v16 = 0;
                std::memcpy(&v16, ib_ptr + i * sizeof(u16), sizeof(u16));
                if (prim_restart && v16 == u16(regs.primitive_restart_index & 0xFFFFu)) {
                    return 0;
                }
                raw_idx = v16;
            } else {
                std::memcpy(&raw_idx, ib_ptr + i * sizeof(u32), sizeof(u32));
                if (prim_restart && raw_idx == regs.primitive_restart_index) {
                    return 0;
                }
            }
            const s64 eff = v_off + static_cast<s64>(raw_idx);
            if (eff < 0 || eff > 0x7FFFFFFFLL) {
                return 0;
            }
            vtx_indices[i] = static_cast<u32>(eff);
        }
    }

    float cb_scale_x = 1.0f;
    float cb_scale_y = 1.0f;
    float cb0_z = 0.0f;
    if (vs_kind == VsKind::PassThroughXYCb0Z_W || vs_kind == VsKind::YebisCb20_21ScaleXY) {
        u32 cb_idx = ~0u;
        for (u32 i = 0; i < vs.buffers.size(); ++i) {
            if (!vs.buffers[i].IsSpecial()) {
                if (cb_idx != ~0u) {
                    return 0;
                }
                cb_idx = i;
            }
        }
        if (cb_idx == ~0u) {
            return 0;
        }
        const u64 needed_cb_bytes = (vs_kind == VsKind::YebisCb20_21ScaleXY) ? 88ULL : 4ULL;
        const u8* cb_ptr = nullptr;
        if (const auto* ring = FindRingBinding(vs, cb_idx)) {
            if (ring->size >= needed_cb_bytes) {
                if (ring->address != 0 &&
                    !buffer_cache.IsConstRamRange(ring->address, needed_cb_bytes) &&
                    memory->ClampRangeSize(ring->address, needed_cb_bytes) == needed_cb_bytes) {
                    cb_ptr = reinterpret_cast<const u8*>(ring->address);
                } else {
                    cb_ptr = (ring->ring == 0) ? constant_ring->Data(ring->offset)
                                               : worker_rings[ring->ring - 1]->Data(ring->offset);
                }
            }
        } else {
            const PreparedStage* prep_vs = nullptr;
            if (draw_prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
                for (u32 i = 0; i < draw_prepared->num_stages; ++i) {
                    const auto& cand = draw_prepared->stages[i];
                    if (&cand.program->info == &vs && cand.num_buffers == vs.buffers.size()) {
                        prep_vs = &cand;
                        break;
                    }
                }
            }
            const Gcn::Buffer cb_sharp =
                prep_vs ? prep_vs->buffer_sharps[cb_idx] : vs.buffers[cb_idx].GetSharp(vs);
            if (!cb_sharp.Valid() || cb_sharp.GetSize() < needed_cb_bytes) {
                return 0;
            }
            cb_ptr = resolve_cpu_ptr(cb_sharp.base_address, needed_cb_bytes);
        }
        if (!cb_ptr) {
            return 0;
        }
        if (vs_kind == VsKind::YebisCb20_21ScaleXY) {
            std::memcpy(&cb_scale_x, cb_ptr + 20 * sizeof(u32), sizeof(float));
            std::memcpy(&cb_scale_y, cb_ptr + 21 * sizeof(u32), sizeof(float));
        } else {
            std::memcpy(&cb0_z, cb_ptr, sizeof(float));
        }
    }

    Gcn::Buffer vb_attr0{};
    if (vs_kind != VsKind::Procedural89cda9e4) {
        if (!fetch) {
            return 0;
        }
        bool found_attr0 = false;
        for (u32 i = 0; i < fetch->attributes.size(); ++i) {
            const auto& attrib = fetch->attributes[i];
            if (attrib.semantic != 0) {
                continue;
            }
            if (attrib.GetStepRate() != Shader::Gcn::VertexAttribute::InstanceIdType::None) {
                return 0;
            }
            if (draw_prepared && draw_prepared->vertex.valid && i < draw_prepared->vertex.count &&
                !BbToggle::Disabled(BbToggle::PreparedResources)) {
                vb_attr0 = draw_prepared->vertex.buffers[i];
            } else if (packet_vsharps.size() == fetch->attributes.size()) {
                vb_attr0 = packet_vsharps[i];
            } else {
                vb_attr0 = attrib.GetSharp(vs);
            }
            found_attr0 = true;
            break;
        }
        if (!found_attr0 || !vb_attr0.Valid() || vb_attr0.base_address == 0) {
            return 0;
        }
    }

    std::array<WinPt, 6> win_pts{};
    const u64 vb_size = (vs_kind != VsKind::Procedural89cda9e4) ? vb_attr0.GetSize() : 0ULL;
    const u64 vb_stride = (vs_kind != VsKind::Procedural89cda9e4) ? vb_attr0.GetStride() : 0ULL;
    for (u32 i = 0; i < count; ++i) {
        float px = 0.0f, py = 0.0f, pz = 0.0f, pw = 1.0f;
        const u32 vid = vtx_indices[i];
        if (vs_kind == VsKind::Procedural89cda9e4) {
            px = float(static_cast<s32>(((vid & 1u) << 1) - 1u));
            py = float(static_cast<s32>((vid & ~1u) - 1u));
            pz = 0.0f;
            pw = 1.0f;
        } else {
            if (vb_attr0.num_records != 0 && vb_stride > 0 && vid >= vb_attr0.num_records) {
                return 0;
            }
            const u64 elem_off = u64(vid) * vb_stride;
            if (elem_off >= vb_size) {
                return 0;
            }
            const u64 avail = std::min<u64>(vb_size - elem_off, 16ULL);
            const u8* elem_ptr = resolve_cpu_ptr(vb_attr0.base_address + elem_off, avail);
            if (!elem_ptr) {
                return 0;
            }
            std::array<float, 4> attr0{};
            if (!DecodeFloatVertexAttr(vb_attr0, elem_ptr, avail, attr0)) {
                return 0;
            }
            switch (vs_kind) {
            case VsKind::PassThroughXYZW:
                px = attr0[0];
                py = attr0[1];
                pz = attr0[2];
                pw = attr0[3];
                break;
            case VsKind::PassThroughXYZ1:
                px = attr0[0];
                py = attr0[1];
                pz = attr0[2];
                pw = 1.0f;
                break;
            case VsKind::PassThroughXYCb0Z_W:
                px = attr0[0];
                py = attr0[1];
                pz = cb0_z;
                pw = attr0[3];
                break;
            case VsKind::YebisCb20_21ScaleXY:
                px = cb_scale_x * attr0[0];
                py = cb_scale_y * attr0[1];
                pz = 0.1f;
                pw = 1.0f;
                break;
            default:
                return 0;
            }
        }
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) || !std::isfinite(pw) ||
            pw <= 0.0f) {
            return 0;
        }
        if (!clip_disabled) {
            if (key.depth_clip_enable) {
                const float z_min =
                    (key.clip_space == Gcn::ClipSpace::MinusWToW) ? -pw - 1e-4f : -1e-4f;
                if (pz < z_min || pz > pw + 1e-4f) {
                    return 0;
                }
            }
            const float nx = px / pw;
            const float ny = py / pw;
            win_pts[i].x = (vp.x + 0.5f * vp.width) + (0.5f * vp.width) * nx;
            win_pts[i].y = (vp.y + 0.5f * vp.height) + (0.5f * vp.height) * ny;
        } else {
            if (std::abs(pw - 1.0f) > 1e-4f) {
                return 0;
            }
            win_pts[i].x = px * push_data.xscale + push_data.xoffset;
            win_pts[i].y = py * push_data.yscale + push_data.yoffset;
        }
    }

    const auto prim = regs.primitive_type;
    if (prim == Gcn::PrimitiveType::RectList) {
        // GCN RectList: 3 corners of an axis-aligned screen rectangle (never face-culled).
        const WinPt p0 = win_pts[0], p1 = win_pts[1], p2 = win_pts[2];
        const bool ortho0 = (std::abs(p1.x - p0.x) <= 2e-2f && std::abs(p2.y - p0.y) <= 2e-2f) ||
                            (std::abs(p1.y - p0.y) <= 2e-2f && std::abs(p2.x - p0.x) <= 2e-2f);
        const bool ortho1 = (std::abs(p0.x - p1.x) <= 2e-2f && std::abs(p2.y - p1.y) <= 2e-2f) ||
                            (std::abs(p0.y - p1.y) <= 2e-2f && std::abs(p2.x - p1.x) <= 2e-2f);
        const bool ortho2 = (std::abs(p0.x - p2.x) <= 2e-2f && std::abs(p1.y - p2.y) <= 2e-2f) ||
                            (std::abs(p0.y - p2.y) <= 2e-2f && std::abs(p1.x - p2.x) <= 2e-2f);
        if (!ortho0 && !ortho1 && !ortho2) {
            return 0;
        }
        const float rx0 = std::min({p0.x, p1.x, p2.x});
        const float rx1 = std::max({p0.x, p1.x, p2.x});
        const float ry0 = std::min({p0.y, p1.y, p2.y});
        const float ry1 = std::max({p0.y, p1.y, p2.y});
        return (rx0 <= 2e-2f && rx1 >= w_f - 2e-2f && ry0 <= 2e-2f && ry1 >= h_f - 2e-2f)
                   ? candidate_mask
                   : 0;
    }

    std::array<WinTri, 2> tris{};
    u32 num_tris = 0;
    const vk::CullModeFlags cull_mode =
        GcnToVk::IsPrimitiveCulled(prim) ? GcnToVk::CullMode(regs.polygon_control.CullingMode())
                                         : vk::CullModeFlagBits::eNone;
    const vk::FrontFace front_face = GcnToVk::FrontFace(regs.polygon_control.front_face);
    if (cull_mode == vk::CullModeFlagBits::eFrontAndBack) {
        return 0;
    }
    const auto push_tri = [&](WinPt a, WinPt b, WinPt c) {
        const float det = Cross2D(a, b, c);
        if (std::abs(det) <= 1e-2f) {
            return;
        }
        if (cull_mode != vk::CullModeFlagBits::eNone) {
            const bool is_front =
                (front_face == vk::FrontFace::eCounterClockwise) ? (det < 0.0f) : (det > 0.0f);
            if ((cull_mode == vk::CullModeFlagBits::eFront && is_front) ||
                (cull_mode == vk::CullModeFlagBits::eBack && !is_front)) {
                return;
            }
        }
        if (num_tris < tris.size()) {
            tris[num_tris++] = WinTri{{a, b, c}, det};
        }
    };

    switch (prim) {
    case Gcn::PrimitiveType::TriangleList:
        if (count == 3) {
            push_tri(win_pts[0], win_pts[1], win_pts[2]);
        } else if (count == 6) {
            push_tri(win_pts[0], win_pts[1], win_pts[2]);
            push_tri(win_pts[3], win_pts[4], win_pts[5]);
        } else {
            return 0;
        }
        break;
    case Gcn::PrimitiveType::TriangleStrip:
        if (count == 3) {
            push_tri(win_pts[0], win_pts[1], win_pts[2]);
        } else if (count == 4) {
            push_tri(win_pts[0], win_pts[1], win_pts[2]);
            push_tri(win_pts[2], win_pts[1], win_pts[3]);
        } else {
            return 0;
        }
        break;
    case Gcn::PrimitiveType::TriangleFan:
    case Gcn::PrimitiveType::QuadList:
        if (count == 4) {
            push_tri(win_pts[0], win_pts[1], win_pts[2]);
            push_tri(win_pts[0], win_pts[2], win_pts[3]);
        } else {
            return 0;
        }
        break;
    default:
        return 0;
    }

    bool covered = false;
    if (num_tris == 1) {
        covered = TriangleCoversRect(tris[0], w_f, h_f);
    } else if (num_tris == 2) {
        covered = TwoTrianglesCoverRect(tris[0], tris[1], w_f, h_f);
    }
    return covered ? candidate_mask : 0;
}

void Rasterizer::BeginDraw(const GraphicsPipeline* pipeline, const RenderState& state,
                           bool is_indexed, bool jitter_eligible,
                           const PreparedDraw* draw_prepared, u32 index_offset,
                           bool allow_dont_care) {
    // The upscaler's sub-pixel jitter, for geometry drawn with the scene depth.
    draw_jitter = {};
    if (jitter_eligible && upscaler->Enabled() && db_desc.first &&
        db_desc.first == camera_motion->Depth()) {
        draw_jitter = upscaler->Jitter();
    }
    // A new render pass sets its viewports and scissors again, even when the state tracker has
    // them already: single frames were drawn with the 960x540 viewport of the passes before (the
    // final picture in the top-left quarter, black around it; docs/Flicker_video_2026-10-17.ru.md).
    // BB_DYNAMIC_STATE_ALWAYS=1 (diagnostics): all dynamic state again at every draw.
    static const bool dynamic_always = [] {
        const char* env = std::getenv("BB_DYNAMIC_STATE_ALWAYS");
        return env && env[0] == '1';
    }();
    const bool will_begin = !last_begin_memo_hit && scheduler.WillBeginRendering(state);
    if (dynamic_always) {
        scheduler.GetDynamicState().Invalidate();
    } else if (will_begin) {
        scheduler.GetDynamicState().InvalidateViewports();
    }
    UpdateDynamicState(pipeline, is_indexed);
    // BB_DISPLAY_WATCH=1 (diagnostics): every draw into a display buffer, with its sizes and state.
    static const bool display_watch = std::getenv("BB_DISPLAY_WATCH") != nullptr;
    // BB_DISPLAY_WATCH_TARGET=addr: also every draw into that color target.
    static const VAddr watch_target = [] {
        const char* env = std::getenv("BB_DISPLAY_WATCH_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (display_watch && Regs().color_buffers[0] &&
        (FrameCapture::IsDisplayBuffer(Regs().color_buffers[0].Address()) ||
         Regs().color_buffers[0].Address() == watch_target)) {
        const auto& rvp = Regs().viewports[0];
        std::printf("Display watch regs: vp0 %g %g %g %g vte %#x jitter %g,%g new pass %d\n", rvp.xscale,
                    rvp.xoffset, rvp.yscale, rvp.yoffset, std::bit_cast<u32>(Regs().viewport_control),
                    draw_jitter[0], draw_jitter[1], int(will_begin));
        const auto& ds = scheduler.GetDynamicState();
        const auto& vps = ds.CurrentViewports();
        const auto& scs = ds.CurrentScissors();
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        const auto& image = texture_cache.GetImage(cb_descs[0].first).info;
        std::printf("Display watch: gpu frame %llu flips %llu vs %08x ps %08x pass %ux%u image %ux%u scale %g,%g "
                    "vp %g,%g %gx%g sc %d,%d %ux%u clipoff %d push %g,%g,%g,%g\n",
                    (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                    (unsigned long long)FrameCapture::Flips(),
                    u32(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash), u32(ps ? ps->pgm_hash : 0),
                    state.width, state.height, image.size.width, image.size.height, target_scale[0],
                    target_scale[1], vps.empty() ? 0.f : vps[0].x, vps.empty() ? 0.f : vps[0].y,
                    vps.empty() ? 0.f : vps[0].width, vps.empty() ? 0.f : vps[0].height,
                    scs.empty() ? 0 : scs[0].offset.x, scs.empty() ? 0 : scs[0].offset.y,
                    scs.empty() ? 0u : scs[0].extent.width, scs.empty() ? 0u : scs[0].extent.height,
                    int(Regs().IsClipDisabled()), push_data.xscale, push_data.xoffset,
                    push_data.yscale, push_data.yoffset);
    }
    MarkPass(pipeline, state, will_begin);
    if (occlusion) {
        occlusion->SetPassScale(target_scale[0] * target_scale[1]); // samples at the game's size
    }
    if (will_begin) {
        const u8 color_dont_care_mask =
            allow_dont_care ? ComputeFullscreenOverwriteMask(pipeline, state, draw_prepared,
                                                             is_indexed, index_offset)
                            : 0;
        scheduler.BeginRendering(state, color_dont_care_mask);
        if (begin_memo.valid) {
            begin_memo.cmd_epoch = scheduler.DescriptorStateEpoch();
        }
    }
    // Push constants after the pass begins: beginning it may record the occlusion results' compute
    // dispatch (OcclusionQueries::Suspend), and the driver keeps one set of push constants for
    // compute and graphics. Its 8 bytes over xoffset/yoffset drew single frames shifted by half the
    // screen (docs/Flicker_2026-10-18.ru.md).
    {
        BB_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()}, true);
    }
}

static u64 CopyRangeMask(VAddr begin, VAddr end) {
    constexpr u64 Shift = 16;
    const u64 first = begin >> Shift;
    const u64 last = (end - 1) >> Shift;
    if (last - first >= 63) {
        return ~0ull;
    }
    u64 mask = 0;
    for (u64 p = first; p <= last; ++p) {
        mask |= 1ull << ((p * 0x9E3779B97F4A7C15ull) >> 58);
    }
    return mask;
}

void Rasterizer::NoteCopyIntoMemory(VAddr address, u64 size) {
    if (size == 0 || !OnStageA()) {
        return;
    }
    if (VideoCore::DirectBuffers::IsTransientCueHeap(address, size) ||
        VideoCore::DirectBuffers::IsNativeOwnedAddress(address)) {
        return;
    }
    copies_into_memory[num_copies_into_memory++ % copies_into_memory.size()] = {address,
                                                                               address + size};
    if (num_copies_into_memory % copies_into_memory.size() == 0) {
        // Once around the ring: the bounds of the copies kept (the ones pushed out no longer).
        copies_begin = ~VAddr{0};
        copies_end = 0;
        copies_mask = 0;
        for (const auto& copy : copies_into_memory) {
            copies_begin = std::min(copies_begin, copy.begin);
            copies_end = std::max(copies_end, copy.end);
            copies_mask |= CopyRangeMask(copy.begin, copy.end);
        }
        return;
    }
    copies_begin = std::min(copies_begin, address);
    copies_end = std::max(copies_end, address + size);
    copies_mask |= CopyRangeMask(address, address + size);
}

// Program code the GPU copies into place (DMA in the stream before the draw): this thread decodes
// the draw ahead of the copy and read the code that was there before (another program: a
// character's vertices stretched into streaks in another game; or zeros). Code under a copy noted
// since the GPU last ran everything: everything before it runs first. Rare (shader uploads); the
// last 64 copies are kept (a game copies far more a frame: a wait for each program check after
// the ring filled cost a quarter of this thread).
void Rasterizer::SyncProgramWrites(VAddr address, u64 size) {
    if (address + size <= copies_begin || address >= copies_end ||
        (copies_mask & CopyRangeMask(address, address + size)) == 0 || !OnStageA()) {
        return;
    }
    bool overlaps = false;
    const u32 kept = std::min<u32>(num_copies_into_memory, u32(copies_into_memory.size()));
    for (u32 i = 0; !overlaps && i < kept; ++i) {
        const auto& copy = copies_into_memory[i];
        overlaps = copy.begin < address + size && address < copy.end;
    }
    if (!overlaps) {
        return;
    }
    DrainDrawPipe();
    Finish();
    num_copies_into_memory = 0;
    copies_begin = ~VAddr{0};
    copies_end = 0;
    copies_mask = 0;
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    // bbport: like direct draws, handed to the recording thread after the pipeline selection
    // (else the GPU thread waited here for a whole frame of queued draws).
    const bool pipelined = UseDrawPipe() && FilterDrawPasses() &&
                           !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params, nullptr, true);
    if (!pipeline) {
        return;
    }
    const IndirectDraw indirect = {arg_address + offset, count_address, stride, max_count};
    if (pipelined) {
        PostDraw(pipeline, nullptr, is_indexed, 0, nullptr, &indirect);
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, indirect);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    const VAddr count_address = indirect.count;
    const u32 stride = indirect.stride;
    const u32 max_count = indirect.max_count;

    // Indirect arguments may be GPU-written: leave these draws on camera fallback.
    motion_draw = false;
    motion_geometry = 0;
    {
        const auto& regs = Regs();
        const auto& key = pipeline->GetGraphicsKey();
        bool has_color = false;
        if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
            for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
                if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb) &&
                    (key.mrt_mask & (1 << cb)) != 0) {
                    has_color = true;
                    break;
                }
            }
        }
        const bool has_depth_stencil =
            (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
            (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
        if (!has_color && !has_depth_stencil &&
            pipeline->GetStages()[u32(Shader::SwStage::Fragment)] == nullptr) {
            const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
            if (vs.pgm_hash == 0x89cda9e4ull && !vs.uses_dma &&
                !std::ranges::any_of(vs.buffers, [](const auto& b) { return b.is_written; }) &&
                !std::ranges::any_of(vs.images, [](const auto& i) { return i.is_written; })) {
                return;
            }
        }
    }
    PrepareRenderState(pipeline);
    NoteUpscalerDraw(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    if (HasPendingHleCopies() && PendingHleCopiesConflict(indirect.args, stride * max_count)) {
        FlushPendingHleCopies(*this);
    }
    if (direct_buffers.NeedsProcessEvents(scheduler.IsRendering())) [[unlikely]] {
        direct_buffers.ProcessEvents(memory);
    }
    const VideoCore::Buffer* buffer = nullptr;
    u64 base = 0;
    bool is_native_indirect_buf = false;
    {
        u64 req_bytes = static_cast<u64>(stride) * max_count;
        std::tie(buffer, base) = direct_buffers.TryResolveReadClamped(indirect.args, req_bytes);
        if (buffer && req_bytes >= static_cast<u64>(stride) * max_count) {
            is_native_indirect_buf = true;
        } else {
            std::tie(buffer, base) =
                buffer_cache.ObtainBuffer(indirect.args, stride * max_count, false);
        }
    }
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);
    static const bool force_indirect_guard = [] {
        const char* env = std::getenv("BB_FORCE_INDIRECT_GUARD");
        return env && env[0] == '1';
    }();
    // The arguments checked on the GPU first (vk_indirect_guard.h): a dispatch, outside the
    // render pass. Native DirectBuffers are zero-initialized in VRAM and never carry garbage RAM.
    vk::Buffer checked_args = buffer->Handle();
    u64 checked_offset = base;
    if ((!is_native_indirect_buf || force_indirect_guard) && IndirectGuard::DrawEnabled()) {
        if (!indirect_guard) {
            indirect_guard = std::make_unique<IndirectGuard>(instance, scheduler);
        }
        if (needs_barrier) {
            runtime.FlushBarriers();
            needs_barrier = false;
        }
        scheduler.EndRendering();
        std::tie(checked_args, checked_offset) =
            indirect_guard->CheckDrawArgs(*buffer, base, stride, max_count, is_indexed);
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const VideoCore::Buffer* count_buffer = nullptr;
    u64 count_offset = 0;
    if (count_address != 0) {
        if (HasPendingHleCopies() && PendingHleCopiesConflict(count_address, 4)) {
            FlushPendingHleCopies(*this);
        }
        u64 cnt_bytes = 4;
        std::tie(count_buffer, count_offset) =
            direct_buffers.TryResolveReadClamped(count_address, cnt_bytes);
        if (!count_buffer || cnt_bytes < 4) {
            std::tie(count_buffer, count_offset) =
                buffer_cache.ObtainBuffer(count_address, 4, false);
        }
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
    }
    if (HasPendingHleCopies() && scheduler.WillBeginRendering(state)) {
        FlushPendingHleCopies(*this);
    }

    if (needs_barrier || (scheduler.WillBeginRendering(state) && runtime.HasPendingBarriers())) {
        runtime.FlushBarriers();
    }

    ScaleToTarget(pipeline);
    BeginDraw(pipeline, state, is_indexed, true);

    ASSERT(stride == (is_indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                 : sizeof(VkDrawIndirectCommand)));
    const vk::Pipeline handle = pipeline->Handle();
    bound_graphics_pipeline = handle;
    bound_graphics_epoch = scheduler.DescriptorStateEpoch();
    const vk::Buffer args = checked_args;
    const u64 args_offset = checked_offset;
    const vk::Buffer counts = count_address != 0 ? count_buffer->Handle() : vk::Buffer{};
    const u64 counts_offset = count_address != 0 ? count_offset : 0;
    if (!Breadcrumbs::Enabled()) [[likely]] {
        scheduler.Record([handle, is_indexed, counts, args, args_offset, counts_offset, max_count,
                          stride](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            if (is_indexed) {
                if (counts) {
                    cmdbuf.drawIndexedIndirectCount(args, args_offset, counts, counts_offset,
                                                    max_count, stride);
                } else {
                    cmdbuf.drawIndexedIndirect(args, args_offset, max_count, stride);
                }
            } else if (counts) {
                cmdbuf.drawIndirectCount(args, args_offset, counts, counts_offset, max_count,
                                         stride);
            } else {
                cmdbuf.drawIndirect(args, args_offset, max_count, stride);
            }
        });
    } else {
        const auto* vs_crumb = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
        const auto* ps_crumb = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        const Breadcrumbs::Crumb crumb{
            .kind = Breadcrumbs::Kind::DrawIndirect,
            .hash = {vs_crumb ? vs_crumb->pgm_hash : 0, ps_crumb ? ps_crumb->pgm_hash : 0},
            .program = {vs_crumb ? vs_crumb->ProgramBase() : 0,
                        ps_crumb ? ps_crumb->ProgramBase() : 0},
            .address = indirect.args,
            .count = {max_count, stride, 0},
        };
        scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            if (is_indexed) {
                if (counts) {
                    cmdbuf.drawIndexedIndirectCount(args, args_offset, counts, counts_offset,
                                                    max_count, stride);
                } else {
                    cmdbuf.drawIndexedIndirect(args, args_offset, max_count, stride);
                }
            } else if (counts) {
                cmdbuf.drawIndirectCount(args, args_offset, counts, counts_offset, max_count,
                                         stride);
            } else {
                cmdbuf.drawIndirect(args, args_offset, max_count, stride);
            }
        });
    }

    ResetBindings(false);
    runtime.AccessBuffer(buffer, base, static_cast<u64>(stride) * max_count,
                         vk::PipelineStageFlagBits2::eDrawIndirect,
                         vk::AccessFlagBits2::eIndirectCommandRead);
    if (count_buffer) {
        runtime.AccessBuffer(count_buffer, count_offset, 4,
                             vk::PipelineStageFlagBits2::eDrawIndirect,
                             vk::AccessFlagBits2::eIndirectCommandRead);
    }
    scheduler.KickRecording();
}

void Rasterizer::DispatchDirect() {
    BbStats::dispatches.fetch_add(1, std::memory_order_relaxed);
    // bbport: like draws, handed to the draw recording thread after the pipeline selection.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedDispatch);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    // BB_DISPATCH_START_LOG=1 (diagnostics, the flicker): dispatches with a nonzero thread
    // group start (COMPUTE_START_X/Y/Z), which the translation does not apply.
    if (static const bool start_log = std::getenv("BB_DISPATCH_START_LOG") != nullptr; start_log) {
        const auto& cs = CsRegs();
        static std::atomic<u32> printed{0};
        if ((cs.start_x | cs.start_y | cs.start_z) != 0 && printed.fetch_add(1) < 5000) {
            std::printf("Dispatch start: gpu frame %llu cs %016llx start %u,%u,%u dims %u,%u,%u\n",
                        (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                        (unsigned long long)pipeline->GetStage(Shader::SwStage::Compute).pgm_hash,
                        cs.start_x, cs.start_y, cs.start_z, cs.dim_x, cs.dim_y, cs.dim_z);
        }
    }
    if (pipelined) {
        PostDraw(pipeline, nullptr, false, 0, &CsRegs());
        return;
    }
    DispatchRecord(pipeline);
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    BB_SECTION(DispatchRecord);
    buffer_cache.NewPacket();
    FrameCapture::Poll();
    gbuffer_draw = false;

    scheduler.PopPendingOperations();

    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    // After the resource binding: its transfers (image uploads, downloads) mark themselves.
    const auto mark = [&] {
        GpuProfiler::HudMarkPass(&scheduler, cs.pgm_hash == 0x9a9cf8a9ull
                                                 ? BbStats::GpuSub_Post_BloomGlare
                                                 : BbStats::GpuSub_Comp_ParticlesCs);
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
            });
        }
    };
    if (upscaler->Enabled()) {
        upscaler->OnDispatch(cs.pgm_hash);
    }
    bool hle;
    {
        BB_SECTION(ShaderHle);
        hle = ExecuteShaderHLE(cs, Regs(), cs_program, *this);
    }
    if (hle) {
        return;
    }
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*this);
    }
    // bbport: the game's copy shader running as itself (BB_COPY_SHADER_NATIVE): its destinations stay in
    // the game's memory. They hold data our translator reads on the CPU (next to shader code the
    // game uploads this way); a VRAM copy of them showed it a stale copy (a black scene). The
    // copy list read on the CPU (vk_shader_hle.cpp) kept them in place too.
    buffer_cache.force_writes_in_place =
        Game::BufferCopyShader() != 0 && cs.pgm_hash == Game::BufferCopyShader();
    // What it writes, from its control list: GPU-written marks on those bytes only (the bound
    // destination spans tens of MiB, marked whole it kept the constants there in place).
    thread_local std::vector<std::pair<unsigned long long, unsigned long long>> copy_writes;
    thread_local std::vector<std::pair<unsigned long long, unsigned long long>> copy_reads;
    copy_writes.clear();
    copy_reads.clear();
    if (buffer_cache.force_writes_in_place && CopyDestinationsPrecise() &&
        CopyShaderDestinations(cs, cs_program.dim_x, CopyDestinationGap(), copy_writes,
                               &copy_reads)) {
        buffer_cache.precise_writes = &copy_writes;
        buffer_cache.precise_reads = &copy_reads;
    }
    const bool bound = BindResources(pipeline);
    buffer_cache.force_writes_in_place = false;
    buffer_cache.precise_writes = nullptr;
    buffer_cache.precise_reads = nullptr;
    if (!bound) {
        return;
    }

    scheduler.EndRendering();
    if (needs_barrier || runtime.HasPendingComputeBarriers()) {
        runtime.FlushBarriers(false);
    }
    mark();
    // BB_DISPATCH_LOG=1 (diagnostics): the shape of each kind of real compute dispatch, once: the
    // buffers (size, formatted, written), the groups, and what the image clear detector thinks.
    {
        static const bool log = [] {
            const char* env = std::getenv("BB_DISPATCH_LOG");
            return env && env[0] == '1';
        }();
        if (log) {
            static std::mutex mutex;
            static std::set<std::string> told;
            std::string key = fmt::format("{:016x} x{} y{} z{}", cs.pgm_hash, cs_program.dim_x,
                                          cs_program.dim_y, cs_program.dim_z);
            for (const auto& b : cs.buffers) {
                key += fmt::format(" [{}{}{} {}]", b.is_formatted ? "F" : "R", b.is_written ? "W" : "r",
                                   b.is_formatted ? "" : "", b.GetSharp(cs).GetSize());
            }
            if (cs.buffers.size() >= 2 && cs.buffers[1].is_formatted && cs.buffers[1].is_written) {
                const auto target = cs.buffers[1].GetSharp(cs);
                const auto id = texture_cache.FindImageFromRange(target.base_address, target.GetSize(), false);
                if (id) {
                    const auto& image = texture_cache.GetImage(id);
                    key += fmt::format(" image(size {} bits {} depth {} fmt {})", image.info.guest_size,
                                       image.info.num_bits, image.info.props.is_depth ? 1 : 0,
                                       Gcn::NumBitsPerBlock(target.GetDataFmt()));
                } else {
                    key += fmt::format(" NO-IMAGE at {:#x}", target.base_address);
                }
            }
            std::scoped_lock lock{mutex};
            if (told.size() < 400 && told.insert(key).second) {
                std::printf("Dispatch shape: %s\n", key.c_str());
            }
        }
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()}, true);

    const vk::Pipeline handle = pipeline->Handle();
    const u32 dim_x = cs_program.dim_x, dim_y = cs_program.dim_y, dim_z = cs_program.dim_z;
    if (FrameCapture::Active()) {
        FrameCapture::Dispatch(cs.pgm_hash, dim_x, dim_y, dim_z);
    }
    if (!Breadcrumbs::Enabled()) [[likely]] {
        scheduler.Record([handle, dim_x, dim_y, dim_z](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatch(dim_x, dim_y, dim_z);
        });
    } else {
        const Breadcrumbs::Crumb crumb{
            .kind = Breadcrumbs::Kind::Dispatch,
            .hash = {cs.pgm_hash, 0},
            .program = {cs.ProgramBase(), 0},
            .count = {dim_x, dim_y, dim_z},
        };
        scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatch(dim_x, dim_y, dim_z);
        });
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark right after the traced compute shader's
    // dispatch, so the profile splits its label into the dispatch itself and what follows it
    // (barriers, unlabeled commands) up to the next mark.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = std::getenv("BB_TRACE_SHADER");
        if (traced && std::strtoull(traced, nullptr, 16) == cs.pgm_hash) {
            profiler->Mark(cs.pgm_hash ^ 0xAF7D'0000ull, [&] {
                return fmt::format("after the dispatch of cs {:016x}", cs.pgm_hash);
            });
        }
    }

    ResetBindings(true);
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // bbport: handed to the recording thread like direct dispatches.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        const IndirectDraw indirect = {address + offset, 0, size, 1};
        PostDraw(pipeline, nullptr, false, 0, &CsRegs(), &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address + offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address,
                                        u32 size) {
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*this);
    }
    buffer_cache.NewPacket();
    scheduler.PopPendingOperations();
    if (!BindResources(pipeline)) {
        return;
    }
    if (direct_buffers.NeedsProcessEvents(scheduler.IsRendering())) [[unlikely]] {
        direct_buffers.ProcessEvents(memory);
    }
    const VideoCore::Buffer* buffer = nullptr;
    u64 base = 0;
    bool is_native_indirect_buf = false;
    {
        u64 req_bytes = size;
        std::tie(buffer, base) = direct_buffers.TryResolveReadClamped(address, req_bytes);
        if (buffer && req_bytes >= size) {
            is_native_indirect_buf = true;
        } else {
            std::tie(buffer, base) = buffer_cache.ObtainBuffer(address, size, false);
        }
    }
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);

    scheduler.EndRendering();
    if (needs_barrier || runtime.HasPendingComputeBarriers()) {
        runtime.FlushBarriers(false);
    }
    const auto& cs_crumb = pipeline->GetStage(Shader::SwStage::Compute);
    GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Comp_ParticlesCs);
    if (auto* profiler = GpuProfiler::Get()) {
        profiler->Mark(cs_crumb.pgm_hash ^ 0xD15B0000ull, [&] {
            return fmt::format("dispatch indirect cs {:016x}", cs_crumb.pgm_hash);
        });
    }
    static const bool force_indirect_guard = [] {
        const char* env = std::getenv("BB_FORCE_INDIRECT_GUARD");
        return env && env[0] == '1';
    }();
    // Group counts checked on the GPU first (vk_indirect_guard.h); recorded before the
    // pipeline's bindings, which the check's own pipeline would disturb. Skipped for native
    // zero-initialized DirectBuffers (eliminating 37+ extra compute dispatches and 75+ full-chip
    // barriers per frame).
    vk::Buffer args = buffer->Handle();
    u64 args_offset = base;
    if ((!is_native_indirect_buf || force_indirect_guard) && IndirectGuard::DispatchEnabled()) {
        if (!indirect_guard) {
            indirect_guard = std::make_unique<IndirectGuard>(instance, scheduler);
        }
        std::tie(args, args_offset) = indirect_guard->CheckArgs(*buffer, base);
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()}, true);

    const vk::Pipeline handle = pipeline->Handle();
    if (!Breadcrumbs::Enabled()) [[likely]] {
        scheduler.Record([handle, args, args_offset](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatchIndirect(args, args_offset);
        });
    } else {
        const vk::Buffer raw_args = buffer->Handle();
        const u64 raw_offset = base;
        const u32 args_slot = Breadcrumbs::ArgsSlot();
        const Breadcrumbs::Crumb crumb{
            .kind = Breadcrumbs::Kind::DispatchIndirect,
            .hash = {cs_crumb.pgm_hash, 0},
            .program = {cs_crumb.ProgramBase(), 0},
            .address = address,
            .count = {buffer_cache.IsInPlace(address, size) ? 1u : 0u, 0, 0},
            .args_slot = args_slot,
        };
        scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
            Breadcrumbs::CopyArgs(cmdbuf, raw_args, raw_offset, args_slot);
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatchIndirect(args, args_offset);
        });
    }

    ResetBindings(true);
    runtime.AccessBuffer(buffer, base, size, vk::PipelineStageFlagBits2::eDrawIndirect,
                         vk::AccessFlagBits2::eIndirectCommandRead);
    scheduler.KickRecording();
}

u64 Rasterizer::Flush() {
    BB_SECTION(Flush);
    DrainDrawPipe();
    FlushPendingBufferCopies();
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*this);
    }
    if (buffer_cache.HasPendingInPassShadows()) {
        buffer_cache.FlushInPassShadows();
    }
    direct_buffers.FlushPendingWritebacks();
    if (direct_buffers.NeedsProcessEvents(scheduler.IsRendering())) [[unlikely]] {
        direct_buffers.ProcessEvents(memory);
    }
    PublishSignals(); // the recording thread's (it is idle here, or this is it)
    last_flush_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                        std::memory_order_relaxed);
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    BbTimeline::Note(BbTimeline::VulkanSubmit, current_tick, DrawPipe::OnStageB() ? 1 : 0);
    return current_tick;
}

bool Rasterizer::HasPendingWork() {
    return scheduler.HasPendingWork() || HasPendingHleCopies() || HasUnsubmittedSignals() ||
           direct_buffers.HasPendingWritebacks() || buffer_cache.HasPendingInPassShadows();
}

void Rasterizer::Finish() {
    DrainDrawPipe();
    FlushPendingBufferCopies();
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*this);
    }
    if (buffer_cache.HasPendingInPassShadows()) {
        buffer_cache.FlushInPassShadows();
    }
    direct_buffers.FlushPendingWritebacks();
    PublishSignals();
    scheduler.Finish();
}

void Rasterizer::ProcessDirectTextureEvents() {
    texture_cache.ProcessDirectTextureEvents(memory);
    direct_buffers.ProcessEvents(memory);
}

void Rasterizer::OnSubmit() {
    DrainDrawPipe();
    FlushPendingBufferCopies();
    buffer_cache.NoteTransientSubmitBoundary();
    if (buffer_cache.HasPendingInPassShadows()) {
        buffer_cache.FlushInPassShadows();
    }
    direct_buffers.FlushPendingWritebacks();
    ProcessDirectTextureEvents();
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
    if (const u64 first_draw_ns = g_frame_first_draw_ns.exchange(0, std::memory_order_relaxed);
        first_draw_ns != 0) {
        const u64 now = SteadyNowNs();
        if (now > first_draw_ns) {
            const u64 dec_ns = std::min<u64>(now - first_draw_ns, 50'000'000ULL);
            tls_avg_decode_ns = tls_avg_decode_ns == 0
                                    ? dec_ns
                                    : ((tls_avg_decode_ns * 7ULL + dec_ns) >> 3);
        }
    }
    g_draws_in_frame.store(0, std::memory_order_relaxed);
    g_mid_frame_flushed.store(false, std::memory_order_relaxed);
    const u64 cur_hw_frames = BbStats::hw_gpu_frames.load(std::memory_order_relaxed);
    if (cur_hw_frames > tls_prev_hw_frames) {
        const u64 cur_hw_busy = BbStats::hw_gpu_busy_ns.load(std::memory_order_relaxed);
        const u64 df = cur_hw_frames - tls_prev_hw_frames;
        if (cur_hw_busy >= tls_prev_hw_busy_ns && df <= 256ULL) {
            const u64 sample_busy = (cur_hw_busy - tls_prev_hw_busy_ns) / df;
            tls_avg_gpu_busy_ns = tls_avg_gpu_busy_ns == 0
                                      ? sample_busy
                                      : ((tls_avg_gpu_busy_ns * 7ULL + sample_busy) >> 3);
        }
        tls_prev_hw_frames = cur_hw_frames;
        tls_prev_hw_busy_ns = cur_hw_busy;
    }
}

namespace {

u8 ClassifyHudRenderSubPass(u64 vs_hash, u64 ps_hash, u32 width, u32 height, u32 num_mrts,
                            bool has_depth, bool in_ui_phase) noexcept {
    // 10. UI (Scaleform) & Present (final gamma/LUT1D display blit or UI phase)
    if (ps_hash == 0x38d65b32ull) {
        return BbStats::GpuSub_Ui_RcasPresent;
    }
    if (in_ui_phase || vs_hash == 0x34e8a281ull ||
        ps_hash == 0xbabda34dull || ps_hash == 0xcfa96da9ull || ps_hash == 0x4baeaa00ull ||
        UiComposition::MovieShader(vs_hash)) {
        return BbStats::GpuSub_Ui_ScaleformHud;
    }
    // 9. Anti-Aliasing (in-game DLAA / FXAA / FXAA3 / FXAA3HQ passes)
    if (ps_hash == 0xda1faca9ull || ps_hash == 0x894de4ceull || ps_hash == 0x309a6f67ull ||
        ps_hash == 0x6c5edb96ull) {
        return BbStats::GpuSub_Aa_Upscaler;
    }
    // 2. G-Buffer (5/6/7-MRT deferred geometry & foliage)
    if (num_mrts >= 4) {
        if (ps_hash == 0x1c3953c6ull || ps_hash == 0x998b035eull || ps_hash == 0xca7226ecull ||
            ps_hash == 0xe21e283aull || ps_hash == 0x2140da2bull || ps_hash == 0xa89e01e9ull ||
            ps_hash == 0x70bb42afull || ps_hash == 0xdbeccd9cull || ps_hash == 0x2390c887ull ||
            ps_hash == 0xb9bb200eull) {
            return BbStats::GpuSub_GBuf_FoliageAndCloth;
        }
        return BbStats::GpuSub_GBuf_WorldAndChr;
    }
    // 5. Water / Reflections (SSR raymarching, Hi-Z depth downsample, mip blur & IBL combine)
    if (ps_hash == 0x6c2323c2ull) {
        return BbStats::GpuSub_Ssr_Raymarch;
    }
    if (ps_hash == 0x6320bb24ull || ps_hash == 0xe820ee9dull || ps_hash == 0x01ae5388ull) {
        return BbStats::GpuSub_Ssr_HiZAndMips;
    }
    // 3. SSAO (Ambient Occlusion generation & bilateral depth-aware blur)
    if (ps_hash == 0xcbbca5dcull || ps_hash == 0x83da8467ull || ps_hash == 0x7b1a6a02ull ||
        ps_hash == 0xb6b33378ull || ps_hash == 0x63bf8617ull || ps_hash == 0xca2d8d0aull) {
        return BbStats::GpuSub_Ssao_GenReproj;
    }
    if (ps_hash == 0xc45c51cbull || ps_hash == 0x5196090eull || ps_hash == 0xa707b215ull ||
        ps_hash == 0x114b2257ull || ps_hash == 0x7efecca6ull || ps_hash == 0xb5f8c858ull) {
        return BbStats::GpuSub_Ssao_BilateralBlur;
    }
    // 1. Shadow Maps (4096x4096 CSM/spot shadow atlas, depth-only passes, VSM moments/blur, CSM PCF)
    if ((width == 4096 && height == 4096) ||
        (num_mrts == 0 && has_depth && ps_hash != 0x4fd8db00ull) ||
        ps_hash == 0x006c079bull || ps_hash == 0xa0e88f56ull || ps_hash == 0xa2108f5eull) {
        return BbStats::GpuSub_Shadow_CascadeAtlas;
    }
    if (vs_hash == 0x167daba6ull || ps_hash == 0x1ae03d75ull || ps_hash == 0x0f61fd27ull ||
        ps_hash == 0xf55643deull || ps_hash == 0x11ffdfb5ull || ps_hash == 0x4fd8db00ull ||
        ps_hash == 0x9837d1c8ull || ps_hash == 0x92852ef3ull || ps_hash == 0x7cde8e5full ||
        ps_hash == 0x12483d6dull) {
        return BbStats::GpuSub_Shadow_VsmAndPcf;
    }
    // 4. Deferred PBR Lighting, Light Volumes & Sky/Atmospheric Fog / Godrays
    if (ps_hash == 0x765450dfull || ps_hash == 0xdda49e89ull || ps_hash == 0x32e47788ull ||
        ps_hash == 0xe55e45ebull || vs_hash == 0x637500e0ull || ps_hash == 0xa291c498ull) {
        return BbStats::GpuSub_Def_SkyFogShafts;
    }
    if (num_mrts == 2 || ps_hash == 0xd92266b9ull || ps_hash == 0x3fe9e964ull ||
        vs_hash == 0x8747a367ull || ps_hash == 0x8650418full) {
        return BbStats::GpuSub_Def_DirectLocalPbr;
    }
    // 6. Forward Alpha (Hair, Blood, Cloth, Mist, Half-Res Soft Particles — 1 MRT + depth)
    if (has_depth) {
        if (ps_hash == 0x71c96598ull || ps_hash == 0x5e224cb6ull || ps_hash == 0x9eb30215ull ||
            ps_hash == 0x2eacb983ull) {
            return BbStats::GpuSub_Fwd_SoftParticles;
        }
        return BbStats::GpuSub_Fwd_HairClothAlpha;
    }
    // 8. Yebis 2 Post-FX & GXPostEffect (1 MRT without depth):
    // 8a. Depth of Field (CoC & Bokeh Gather)
    if (ps_hash == 0xa2c4f30aull || ps_hash == 0xd3c8bb21ull || ps_hash == 0x111fce32ull ||
        ps_hash == 0x9e1cb278ull || ps_hash == 0xc902266full || ps_hash == 0x981046d7ull ||
        ps_hash == 0xa9c2becaull || ps_hash == 0xed1e60eeull || ps_hash == 0xf3621495ull ||
        ps_hash == 0xb095e72full || ps_hash == 0x04ef5f2full || ps_hash == 0xf0fa51feull ||
        ps_hash == 0xd61c51ceull || ps_hash == 0xebbf4c04ull || ps_hash == 0x164cdf00ull ||
        ps_hash == 0x5850880cull || ps_hash == 0x7f677cb7ull || ps_hash == 0xcfbd6ad9ull ||
        ps_hash == 0x2ef21350ull || ps_hash == 0x52e4f791ull || ps_hash == 0x8248f5aaull ||
        ps_hash == 0x2c3eebe3ull) {
        return BbStats::GpuSub_Post_DoF;
    }
    // 8b. Motion Blur (Camera/Object Velocity & Bilateral Blur)
    if (ps_hash == 0xe0305cefull || ps_hash == 0x347ef3f4ull || ps_hash == 0x4b766f62ull ||
        ps_hash == 0x86e96401ull || ps_hash == 0x107109bcull || ps_hash == 0xabf92450ull ||
        ps_hash == 0x52b0abfdull || ps_hash == 0xdfff41caull || ps_hash == 0x514b4e45ull ||
        ps_hash == 0x1534534aull || ps_hash == 0x7df9e24aull || ps_hash == 0xc6ea8d61ull ||
        ps_hash == 0xb84726b7ull || ps_hash == 0x23cf4ed2ull || ps_hash == 0x94059df9ull ||
        ps_hash == 0xe4009b2bull || ps_hash == 0xb72bb396ull || ps_hash == 0x8adcefe7ull) {
        return BbStats::GpuSub_Post_MotionBlur;
    }
    // 8c. Bloom, Emissive Glow & Glare Starburst Pyramid
    if (ps_hash == 0x57f3b8c2ull || ps_hash == 0x8c08444cull || ps_hash == 0xf7229091ull ||
        ps_hash == 0x3c2b7974ull || ps_hash == 0x483f5dc1ull || ps_hash == 0x474f0123ull ||
        ps_hash == 0x265a6d2dull || ps_hash == 0xe80526ccull || ps_hash == 0x88414cd6ull ||
        ps_hash == 0xff902740ull || ps_hash == 0x2f82498dull || ps_hash == 0x44a2ed89ull ||
        ps_hash == 0x714ff44eull || ps_hash == 0x825dd04bull || ps_hash == 0x1c3cbd98ull ||
        ps_hash == 0xe93a1097ull || ps_hash == 0xa61e325dull ||
        (width == 1568 && height == 588)) {
        return BbStats::GpuSub_Post_BloomGlare;
    }
    // 8d. HDR -> LDR Tonemap, 3D Color Grading LUT & Pre-Tonemap Clamp
    return BbStats::GpuSub_Post_TonemapLut;
}

} // namespace

void Rasterizer::MarkPass(const GraphicsPipeline* pipeline, const RenderState& state,
                          bool will_begin) {
    auto* profiler = GpuProfiler::Get();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const u64 ps_hash = ps ? ps->pgm_hash : 0;
    const u8 subcat = ClassifyHudRenderSubPass(
        vs.pgm_hash, ps_hash, state.width, state.height, state.num_color_attachments,
        state.depth_stencil_attachment.has_depth,
        ui_pass_started || (upscaler && upscaler->InUiPhase()));
    if (will_begin) {
        GpuProfiler::HudSetNextPassCat(&scheduler, subcat);
    } else if (!state.secondary_contents &&
               scheduler.ActiveReorderBucket() == Scheduler::kNoReorderBucket) {
        GpuProfiler::HudOnSubPassChange(&scheduler, subcat);
    }
    if (IsSfxFireDumpActive()) {
        RecordSfxFireDumpDraw(pipeline, state.width, state.height, state.num_color_attachments,
                              state.depth_stencil_attachment.has_depth, u32(Regs().num_indices),
                              u32(std::max<u32>(1u, Regs().num_instances.NumInstances())), false,
                              ui_pass_started || (upscaler && upscaler->InUiPhase()));
    }
    if (!profiler) {
        return;
    }
    if (profiler->Records(&scheduler)) {
        const u64 idx_inst =
            u64(Regs().num_indices) * u64(std::max<u32>(1u, Regs().num_instances.NumInstances()));
        profiler->NoteDraw(vs.pgm_hash, ps_hash, state.width, state.height,
                           state.num_color_attachments, idx_inst);
    }
    const bool per_draw_inline =
        !will_begin && profiler->PerDrawMode() && profiler->Records(&scheduler) &&
        !state.secondary_contents &&
        scheduler.ActiveReorderBucket() == Scheduler::kNoReorderBucket;
    if (!will_begin && !per_draw_inline) {
        return;
    }
    std::array<u64, Gcn::NUM_COLOR_BUFFERS + 6> parts{};
    parts[0] = state.width | u64(state.height) << 32;
    parts[1] = vs.pgm_hash;
    parts[2] = ps ? ps->pgm_hash : 0;
    parts[3] = db_desc.first ? u64(texture_cache.GetImage(db_desc.first).info.pixel_format) : 0;
    for (u32 cb = 0; cb < state.num_color_attachments && cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        parts[4 + cb] = cb_descs[cb].first
                            ? u64(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format) + 1
                            : 0;
    }
    const u64 key = XXH3_64bits(parts.data(), sizeof(parts));
    auto describe = [&] {
        std::string targets;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            targets += cb_descs[cb].first
                           ? vk::to_string(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format)
                           : std::string{"-"};
            targets += ' ';
        }
        const u64 psh = ps ? ps->pgm_hash : 0;
        const char* nat_ps = Shader::Backend::SPIRV::NativeShaderName(psh);
        const char* nat_vs = Shader::Backend::SPIRV::NativeShaderName(vs.pgm_hash);
        const char* tag = nat_ps ? nat_ps : (nat_vs ? nat_vs : "PS4-SPIRV");
        return fmt::format("{} {}x{} [{}] depth {} vs {:016x} ps {:016x} [{}]",
                           will_begin ? "pass" : "draw", state.width, state.height, targets,
                           db_desc.first ? vk::to_string(
                                               texture_cache.GetImage(db_desc.first).info.pixel_format)
                                         : std::string{"-"},
                           vs.pgm_hash, psh, tag);
    };
    if (will_begin) {
        profiler->Mark(key, describe);
    } else {
        profiler->MarkInline(key, describe);
    }
}

void Rasterizer::NoteFrameStart() {
    g_draws_in_frame.store(0, std::memory_order_relaxed);
    g_mid_frame_flushed.store(false, std::memory_order_relaxed);
    frame_fx_mask_valid = false;
    g_frame_first_draw_ns.store(0, std::memory_order_relaxed);
    if (scene_occlusion_culler) {
        scene_occlusion_culler->OnFrameStart(instance, scheduler, camera_motion.get());
    }
    InvalidateGdsMirror();
    if (HasPendingHleCopies()) {
        FlushPendingHleCopies(*this);
    }
    GpuProfiler::OnFrame();
    const u64 frame = BbStats::gpu_frames.fetch_add(1, std::memory_order_relaxed) + 1;
    // BB_BUFFER_STATS=1: how many earlier frames the GPU has not finished when the GPU thread
    // starts a frame — the lag that decides whether guest memory can be read in place.
    static const bool stats = [] {
        const char* value = std::getenv("BB_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    if (!stats) {
        return;
    }
    static std::array<u64, 16> frame_ticks{};
    static std::array<u64, 17> lag_histogram{};
    u32 lag = 0;
    for (u32 back = 1; back < frame_ticks.size() && back < frame; ++back) {
        const u64 tick = frame_ticks[(frame - back) % frame_ticks.size()];
        if (tick && !scheduler.IsFree(tick)) {
            lag = back;
        }
    }
    ++lag_histogram[lag];
    frame_ticks[frame % frame_ticks.size()] = scheduler.CurrentTick();
    if (frame % 600 == 0) {
        std::printf("GPU lag at frame start (frames not finished: count):");
        for (u32 i = 0; i < lag_histogram.size(); ++i) {
            if (lag_histogram[i]) {
                std::printf(" %u:%llu", i, static_cast<unsigned long long>(lag_histogram[i]));
            }
        }
        std::printf("\n");
        lag_histogram = {};
    }
}

} // namespace Vulkan

namespace Vulkan {

void Rasterizer::WatchDraw(const GraphicsPipeline* pipeline, const RenderState& state) {
    // GPU thread only. Per (vs, ps, first target): the last fields seen and how many draws in a
    // row had them. A change after a stable run is printed with the frame number, and so is the
    // change back, which marks a one-frame anomaly (docs/Flicker_video_2026-10-17.ru.md).
    const auto& regs = Regs();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const VAddr target = cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.guest_address : 0;
    const u64 key = vs.pgm_hash * 0x9E3779B97F4A7C15ull ^ (ps ? ps->pgm_hash : 0) ^ target;
    const auto bits = [](float f) { return u64(std::bit_cast<u32>(f)); };
    std::vector<u64> fields;
    const auto& vp = regs.viewports[0];
    fields.push_back(bits(vp.xscale) << 32 | bits(vp.xoffset));
    fields.push_back(bits(vp.yscale) << 32 | bits(vp.yoffset));
    fields.push_back(u64(std::bit_cast<u32>(regs.viewport_control)) << 32 | u64(regs.IsClipDisabled()));
    const auto& sc = regs.screen_scissor;
    fields.push_back(u64(u16(sc.top_left_x)) << 48 | u64(u16(sc.top_left_y)) << 32 |
                     u64(u16(sc.bottom_right_x)) << 16 | u16(sc.bottom_right_y));
    fields.push_back(bits(target_scale[0]) << 32 | bits(target_scale[1]));
    fields.push_back(bits(push_data.xoffset) << 32 | bits(push_data.yoffset));
    fields.push_back(u64(state.width) << 32 | state.height);
    static const bool textures = std::getenv("BB_DRAW_WATCH_TEXTURES") != nullptr;
    for (const auto id : textures ? bound_images : decltype(bound_images){}) {
        const auto& info = texture_cache.GetImage(id).info;
        fields.push_back(info.guest_address);
        fields.push_back(u64(info.size.width) << 32 | info.size.height);
    }
    struct Seen {
        std::vector<u64> fields;
        std::vector<u64> stable_fields;
        u32 run = 0;
        u64 changed_frame = 0;
    };
    static std::unordered_map<u64, Seen> seen;
    static u32 printed = 0;
    auto& s = seen[key];
    const u64 frame = FrameCapture::Flips();
    if (s.fields == fields) {
        ++s.run;
        return;
    }
    const auto print = [&](const char* what) {
        if (printed++ > 4000) {
            return;
        }
        std::string text = fmt::format("Draw watch: frame {} {} vs {:016x} ps {:016x} target {:#x}:",
                                       frame, what, vs.pgm_hash, ps ? ps->pgm_hash : 0, target);
        const size_t n = std::max(fields.size(), s.fields.size());
        for (size_t i = 0; i < n; ++i) {
            const u64 now = i < fields.size() ? fields[i] : 0;
            const u64 was = i < s.fields.size() ? s.fields[i] : 0;
            if (now != was) {
                text += fmt::format(" [{}] {:#x}->{:#x}", i, was, now);
            }
        }
        std::printf("%s\n", text.c_str());
    };
    if (s.run >= 8) {
        // A stable run broke.
        s.stable_fields = s.fields;
        s.changed_frame = frame;
        print("changed");
    } else if (!s.stable_fields.empty() && fields == s.stable_fields && frame - s.changed_frame <= 2) {
        print("back after a short change");
        s.stable_fields.clear();
    }
    s.fields = std::move(fields);
    s.run = 0;
}

} // namespace Vulkan

namespace Shader::Backend::SPIRV {
[[nodiscard]] bool HasNativeShader(u64 pgm_hash) noexcept;
} // namespace Shader::Backend::SPIRV

namespace Vulkan {

namespace {

struct SfxFireDumpKey {
    u64 ps_hash = 0;
    u64 vs_hash = 0;
    u32 blend_bits = 0;
    u32 width = 0;
    u32 height = 0;
    u16 mrt_fmt = 0;
    u8 flags = 0; // bit0=has_depth, bit1=has_tess, bit2=is_mt, bit3=blend_en

    bool operator==(const SfxFireDumpKey& o) const noexcept = default;
};

struct SfxFireDumpKeyHash {
    size_t operator()(const SfxFireDumpKey& k) const noexcept {
        u64 h = k.ps_hash * 0x9E3779B97F4A7C15ull ^ (k.vs_hash * 0xD6E8FEB86659FD93ull);
        h ^= u64(k.blend_bits) * 0xBF58476D1CE4E5B9ull;
        h ^= (u64(k.width) << 32) | u64(k.height);
        h ^= (u64(k.mrt_fmt) << 8) | u64(k.flags);
        return size_t(h ^ (h >> 33));
    }
};

struct SfxFireDumpStats {
    u64 draws = 0;
    u64 verts = 0;
    u32 num_tex = 0;
    u32 num_buf = 0;
    bool is_native_ps = false;
    bool is_native_vs = false;
};

const char* BlendFactorName(Gcn::BlendControl::BlendFactor f) noexcept {
    using BF = Gcn::BlendControl::BlendFactor;
    switch (f) {
    case BF::Zero: return "0";
    case BF::One: return "1";
    case BF::SrcColor: return "SrcC";
    case BF::OneMinusSrcColor: return "1-SrcC";
    case BF::SrcAlpha: return "SrcA";
    case BF::OneMinusSrcAlpha: return "1-SrcA";
    case BF::DstAlpha: return "DstA";
    case BF::OneMinusDstAlpha: return "1-DstA";
    case BF::DstColor: return "DstC";
    case BF::OneMinusDstColor: return "1-DstC";
    case BF::ConstantColor: return "ConstC";
    case BF::OneMinusConstantColor: return "1-ConstC";
    case BF::ConstantAlpha: return "ConstA";
    case BF::OneMinusConstantAlpha: return "1-ConstA";
    default: return "?";
    }
}

const char* BlendFuncName(Gcn::BlendControl::BlendFunc op) noexcept {
    using Op = Gcn::BlendControl::BlendFunc;
    switch (op) {
    case Op::Add: return "+";
    case Op::Subtract: return "-";
    case Op::Min: return "min";
    case Op::Max: return "max";
    case Op::ReverseSubtract: return "rsub";
    default: return "?";
    }
}

std::atomic<bool> g_sfx_dump_active{false};
std::atomic<s64> g_sfx_dump_deadline_ns{0};
std::atomic<u32> g_sfx_dump_last_count{0};
std::mutex g_sfx_dump_mutex;
std::unordered_map<SfxFireDumpKey, SfxFireDumpStats, SfxFireDumpKeyHash> g_sfx_dump_map;

void FlushSfxFireDumpLocked() noexcept {
    std::vector<std::pair<SfxFireDumpKey, SfxFireDumpStats>> rows;
    rows.reserve(g_sfx_dump_map.size());
    for (const auto& kv : g_sfx_dump_map) {
        rows.push_back(kv);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        const bool a_bl = (a.first.flags & 8u) != 0u;
        const bool b_bl = (b.first.flags & 8u) != 0u;
        if (a_bl != b_bl) return a_bl > b_bl;
        if (a.second.draws != b.second.draws) return a.second.draws > b.second.draws;
        return a.first.ps_hash < b.first.ps_hash;
    });

    auto write_file = [&](const char* path) {
        FILE* f = std::fopen(path, "w");
        if (!f) return false;
        std::fprintf(f, "# Bloodborne SFX/Fire Draw Capture (%zu unique 1-MRT pipelines)\n",
                     rows.size());
        std::fprintf(f, "# Columns: ps_hash vs_hash native(PS/VS) blend(src op dst | aSrc aDst) "
                        "depth tess mt wxh fmt tex buf draws verts\n");
        for (const auto& [k, s] : rows) {
            Gcn::BlendControl bc{};
            std::memcpy(&bc, &k.blend_bits, sizeof(bc));
            const bool bl_en = (k.flags & 8u) != 0u;
            const char* bl_str = bl_en ? "BLEND" : "OPAQUE";
            std::fprintf(
                f,
                "ps=%08llx vs=%08llx nat=%d/%d %s(%s %s %s | a:%s/%s) m=%d d=%d tess=%d mt=%d "
                "%ux%u fmt=%u tex=%u buf=%u draws=%llu verts=%llu\n",
                static_cast<unsigned long long>(k.ps_hash),
                static_cast<unsigned long long>(k.vs_hash), s.is_native_ps ? 1 : 0,
                s.is_native_vs ? 1 : 0, bl_str, BlendFactorName(bc.color_src_factor),
                BlendFuncName(bc.color_func), BlendFactorName(bc.color_dst_factor),
                BlendFactorName(bc.alpha_src_factor), BlendFactorName(bc.alpha_dst_factor),
                (k.flags & 16u) ? 2 : 1, (k.flags & 1u) ? 1 : 0, (k.flags & 2u) ? 1 : 0,
                (k.flags & 4u) ? 1 : 0, k.width,
                k.height, u32(k.mrt_fmt), s.num_tex, s.num_buf,
                static_cast<unsigned long long>(s.draws), static_cast<unsigned long long>(s.verts));
        }
        std::fclose(f);
        return true;
    };

    const bool ok1 = write_file("logs/sfx_fire_dump.txt");
    const bool ok2 = write_file("/tmp/bb_sfx_fire_dump.txt");
    g_sfx_dump_last_count.store(u32(rows.size()), std::memory_order_relaxed);
    std::printf("[SfxFireDump] Captured %zu unique 1-MRT pipelines -> logs/sfx_fire_dump.txt (%d) "
                "/tmp/bb_sfx_fire_dump.txt (%d)\n",
                rows.size(), ok1 ? 1 : 0, ok2 ? 1 : 0);
    g_sfx_dump_map.clear();
}

} // namespace

void StartSfxFireDump(u32 duration_ms) noexcept {
    std::scoped_lock lk{g_sfx_dump_mutex};
    g_sfx_dump_map.clear();
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    g_sfx_dump_deadline_ns.store(now_ns + s64(duration_ms) * 1000000LL, std::memory_order_relaxed);
    g_sfx_dump_active.store(true, std::memory_order_release);
    std::printf("[SfxFireDump] Started %u ms capture window...\n", duration_ms);
}

bool IsSfxFireDumpActive() noexcept {
    return g_sfx_dump_active.load(std::memory_order_acquire);
}

u32 GetSfxFireDumpLastCount() noexcept {
    return g_sfx_dump_last_count.load(std::memory_order_relaxed);
}

void RecordSfxFireDumpDraw(const GraphicsPipeline* pipeline, u32 width, u32 height, u32 num_mrts,
                           bool has_depth, u32 num_indices, u32 num_instances, bool is_mt,
                           bool is_ui_phase) noexcept {
    if (!g_sfx_dump_active.load(std::memory_order_acquire)) {
        return;
    }
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    if (now_ns >= g_sfx_dump_deadline_ns.load(std::memory_order_relaxed)) {
        std::scoped_lock lk{g_sfx_dump_mutex};
        if (g_sfx_dump_active.exchange(false, std::memory_order_acq_rel)) {
            FlushSfxFireDumpLocked();
        }
        return;
    }
    if (!pipeline || (num_mrts != 1u && num_mrts != 2u) || is_ui_phase) {
        return;
    }
    const auto& stages = pipeline->GetStages();
    const auto* vs = stages[u32(Shader::SwStage::Vertex)];
    const auto* ps = stages[u32(Shader::SwStage::Fragment)];
    if (!ps) {
        return;
    }
    const auto& key = pipeline->GetGraphicsKey();
    const Gcn::BlendControl bc = key.blend_controls[0];
    // Capture ALL blended 1-MRT & 2-MRT draws (particles, fire, embers, smoke, lighting, heat) AND
    // depth-tested 1-MRT draws (forward alpha/refraction/ribbons).
    if (!bc.enable && (num_mrts != 1u || !has_depth)) {
        return;
    }

    SfxFireDumpKey dk{};
    dk.ps_hash = ps->pgm_hash;
    dk.vs_hash = vs ? vs->pgm_hash : 0ull;
    std::memcpy(&dk.blend_bits, &bc, sizeof(dk.blend_bits));
    dk.width = width;
    dk.height = height;
    dk.mrt_fmt = static_cast<u16>(key.color_buffers[0].data_format);
    dk.flags = (has_depth ? 1u : 0u) | (key.patch_control_points > 0u ? 2u : 0u) |
               (is_mt ? 4u : 0u) | (bc.enable ? 8u : 0u) | (num_mrts == 2u ? 16u : 0u);

    std::scoped_lock lk{g_sfx_dump_mutex};
    if (!g_sfx_dump_active.load(std::memory_order_relaxed)) {
        return;
    }
    auto& st = g_sfx_dump_map[dk];
    if (st.draws == 0u) {
        st.num_tex = u32(ps->images.size());
        st.num_buf = u32(ps->buffers.size());
        st.is_native_ps = Shader::Backend::SPIRV::HasNativeShader(dk.ps_hash);
        st.is_native_vs = vs ? Shader::Backend::SPIRV::HasNativeShader(dk.vs_hash) : false;
    }
    ++st.draws;
    st.verts += u64(num_indices) * u64(std::max<u32>(1u, num_instances));
}

} // namespace Vulkan
