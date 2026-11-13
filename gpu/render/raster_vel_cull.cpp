// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 10 (Target B): Stage-A Stationary Object-Velocity Culling (`Pass 0x2e`).

#include "render/vk_obj_vel_cull.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#include "buffers/direct_buffers.h"
#include "render/raster_internal.h"
#include "render/vk_vsm_shadow_memo.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end);

namespace Vulkan {

namespace {

// Monotonic steady_clock nanosecond deadline until which `velWrap` may be consumed as a
// silhouette mask (`viewBytes[0x2970]` feedback blur, `viewBytes[0x2971]` radial blur, or
// `viewBytes + 0x2960`). `0` on `99.99%` of frames -> single relaxed load with zero syscalls.
std::atomic<u64> g_silhouette_mask_until_ns{0};

[[nodiscard]] u64 SteadyNowNs() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

[[nodiscard]] bool IsReadableCpuSourceRange(VAddr addr, u64 size) noexcept {
    if (!addr || size == 0u) {
        return false;
    }
    if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, size) ||
        VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, size)) {
        return true;
    }
    uintptr_t mapped_end = 0;
    return BbLookupMappedEnd(addr, mapped_end) && size <= (mapped_end - addr);
}

[[nodiscard]] constexpr bool IsObjectVelPsHash(u64 h) noexcept {
    return h == 0x00000000749e4f9eULL || h == 0x00000000b25e4faeULL ||
           h == 0x0000000034bc187cULL;
}

[[nodiscard]] constexpr bool IsStdObjectVelVsHash(u64 h) noexcept {
    return h == 0x00000000855f61b6ULL || h == 0x000000008a2dba97ULL ||
           h == 0x000000005992ac36ULL || h == 0x00000000f2c65c37ULL;
}

// `0x2b6c1d92` is an instruction-for-instruction SPIR-V twin of `0x8bc6ea32` (`35,672 B`,
// identical `ssbo_3..ssbo_9` rigid-pair / 4-bone skinning layout).
[[nodiscard]] constexpr bool IsHybridObjectVelVsHash(u64 h) noexcept {
    return h == 0x000000008bc6ea32ULL || h == 0x000000002b6c1d92ULL;
}

constexpr u64 kHavokClothObjectVelVsHash = 0x00000000b11756c0ULL;

// Ordered SSE2 comparison (`_mm_cmple_ps`):
// - Early-exits on the first 16-byte lane whose `|cur - prev| > eps` (~2 cycles on moving root bone).
// - Ordered `CMPLEPS` returns `0` if either float is `NaN` or `Inf`, safely rejecting non-finite data.
[[nodiscard]] bool MatricesStationarySse2(const u8* cur_bytes, const u8* prev_bytes,
                                          u32 num_bytes, float eps) noexcept {
    if (cur_bytes == prev_bytes) {
        return true;
    }
    const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    const __m128 eps_v = _mm_set1_ps(eps);
    u32 off = 0u;
    for (; off + 16u <= num_bytes; off += 16u) {
        const __m128 a = _mm_loadu_ps(reinterpret_cast<const float*>(cur_bytes + off));
        const __m128 b = _mm_loadu_ps(reinterpret_cast<const float*>(prev_bytes + off));
        const __m128 diff = _mm_and_ps(_mm_sub_ps(a, b), sign_mask);
        const __m128 le = _mm_cmple_ps(diff, eps_v);
        if (_mm_movemask_ps(le) != 0xF) {
            return false;
        }
    }
    for (; off + 4u <= num_bytes; off += 4u) {
        float a = 0.0f;
        float b = 0.0f;
        std::memcpy(&a, cur_bytes + off, sizeof(float));
        std::memcpy(&b, prev_bytes + off, sizeof(float));
        const float d = std::fabs(a - b);
        if (!(d <= eps)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr size_t HashPairKey(VAddr cur_addr, VAddr prev_addr, u64 cur_gen,
                                           u64 prev_gen, u32 byte_off, u32 cmp_bytes) noexcept {
    u64 h = cur_addr * 0x9E3779B97F4A7C15ULL;
    h ^= prev_addr * 0xBF58476D1CE4E5B9ULL;
    h ^= cur_gen * 0x94D049BB133111EBULL;
    h ^= prev_gen * 0xD6E8FEB86659FD93ULL;
    h ^= (u64(byte_off) << 32) | u64(cmp_bytes);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return static_cast<size_t>(h & (ObjectVelocityCuller::kPairMemoSize - 1u));
}

} // namespace

ObjectVelocityCuller::ObjectVelocityCuller()
    : tiny_meta(kTinySlots),
      tiny_bytes(kTinySlots * size_t(kTinySlotStride), 0u),
      small_meta(kSmallSlots),
      small_bytes(kSmallSlots * size_t(kMaxSmallBytes), 0u),
      large_meta(kLargeSlots),
      large_bytes(kLargeSlots * size_t(kMaxLargeBytes), 0u),
      coarse_epoch_256k(kCoarseBuckets, 0u) {}

bool ObjectVelocityCuller::Enabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_VEL_CULL");
        return !env || (env[0] != '\0' && env[0] != '0');
    }();
    return enabled;
}

float ObjectVelocityCuller::EpsilonThreshold() noexcept {
    static const float eps = [] {
        if (const char* env = std::getenv("BB_VEL_CULL_EPS"); env && env[0] != '\0') {
            const float v = std::strtof(env, nullptr);
            if (std::isfinite(v) && v >= 0.0f && v <= 0.1f) {
                return v;
            }
        }
        // `velWrap` is 8-bit `R8G8B8A8Unorm` (`0x7F` = zero velocity; 1 LSB corresponds to
        // ~0.012f bone matrix delta, see `kObjectVelocityBoneDeltaThreshold`). `0.008f` (8 mm)
        // absorbs subtle idle Morpheme breathing (~0.0005f..0.003f) while preserving 100% of
        // visible limb, weapon, and character motion blur.
        return 0.008f;
    }();
    return eps;
}

void ObjectVelocityCuller::NoteSilhouetteMaskActiveGlobal() noexcept {
    // Hold silhouette protection for 150 ms across any CPU->Stage-A->GPU pipeline lead.
    g_silhouette_mask_until_ns.store(SteadyNowNs() + 150'000'000ULL, std::memory_order_relaxed);
}

bool ObjectVelocityCuller::IsSilhouetteMaskActive() noexcept {
    const u64 until_ns = g_silhouette_mask_until_ns.load(std::memory_order_relaxed);
    if (until_ns == 0ULL) [[likely]] {
        return false;
    }
    if (SteadyNowNs() < until_ns) {
        return true;
    }
    g_silhouette_mask_until_ns.store(0ULL, std::memory_order_relaxed);
    return false;
}

void ObjectVelocityCuller::InvalidateCpuShadow(VAddr addr, u64 size) noexcept {
    if (!addr || size == 0u) {
        return;
    }
    const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
    const u64 span = end - addr;

    // Large (`> 32 KiB`) buffer copies/writes (e.g. `2 MiB` `GXDynamicBufferRingOwner`, compute
    // particle outputs, skinning/index uploads): O(1) `256 KiB` coarse epoch bump instead of
    // scanning all `1024 + 2048 + 512` shadow metadata slots on every large copy!
    if (span > kMaxLargeBytes) {
        const VAddr g_lo = addr >> kCoarseShift;
        const VAddr g_hi = (end - 1u) >> kCoarseShift;
        if ((g_hi - g_lo) < 512u) {
            u32 epoch = ++coarse_clock;
            if (epoch == 0u) [[unlikely]] {
                epoch = ++coarse_clock;
            }
            for (VAddr g = g_lo; g <= g_hi; ++g) {
                coarse_epoch_256k[static_cast<size_t>(g & (kCoarseBuckets - 1u))] = epoch;
            }
        } else {
            ++coarse_wipe_bias;
        }
        return;
    }

    // Exact per-slot invalidation for `<= 32 KiB` writes (never invalidates neighboring CBs):
    {
        const VAddr p_lo = (addr >= kTinySlotStride) ? ((addr - kTinySlotStride) >> 9) : 0ULL;
        const VAddr p_hi = (end - 1u) >> 9;
        for (VAddr p = p_lo; p <= p_hi; ++p) {
            const size_t set_base = TinyPageSet(p) * kTinyWays;
            for (size_t w = 0; w < kTinyWays; ++w) {
                auto& slot = tiny_meta[set_base + w];
                if (slot.byte_size != 0u && slot.dst_va < end &&
                    addr < (slot.dst_va + slot.byte_size)) {
                    slot.byte_size = 0u;
                }
            }
        }
    }
    {
        const VAddr p_lo = (addr >= kMaxSmallBytes) ? ((addr - kMaxSmallBytes) >> 11) : 0ULL;
        const VAddr p_hi = (end - 1u) >> 11;
        for (VAddr p = p_lo; p <= p_hi; ++p) {
            const size_t set_base = SmallPageSet(p) * kSmallWays;
            for (size_t w = 0; w < kSmallWays; ++w) {
                auto& slot = small_meta[set_base + w];
                if (slot.byte_size != 0u && slot.dst_va < end &&
                    addr < (slot.dst_va + slot.byte_size)) {
                    slot.byte_size = 0u;
                }
            }
        }
    }
    {
        const VAddr p_lo = (addr >= kMaxLargeBytes) ? ((addr - kMaxLargeBytes) >> 15) : 0ULL;
        const VAddr p_hi = (end - 1u) >> 15;
        for (VAddr p = p_lo; p <= p_hi; ++p) {
            const size_t set_base = LargePageSet(p) * kLargeWays;
            for (size_t w = 0; w < kLargeWays; ++w) {
                auto& slot = large_meta[set_base + w];
                if (slot.byte_size != 0u && slot.dst_va < end &&
                    addr < (slot.dst_va + slot.byte_size)) {
                    slot.byte_size = 0u;
                }
            }
        }
    }
}

void ObjectVelocityCuller::NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept {
    if (!raw_regions || count == 0u || count > 32u || !Enabled()) {
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw_regions + i * 6u, sizeof(reg));
        if (!reg.dst_va || reg.byte_size == 0u) {
            continue;
        }
        InvalidateCpuShadow(reg.dst_va, reg.byte_size);
        if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource || !reg.src_va ||
            reg.byte_size < 16u || reg.byte_size > kMaxLargeBytes ||
            !IsReadableCpuSourceRange(reg.src_va, reg.byte_size)) {
            continue;
        }

        // Maintained strictly in synchronous Stage-A stream order (never compare against async
        // Stage-B `QueryPage64kWriteDigest`, which increments when `DirectVkCopyBatchTask` runs
        // on `bb:DrawRec` and would self-invalidate Stage-A's own shadow entry).
        const u64 gen = ++shadow_upload_gen;
        const u32 use_stamp = ++shadow_clock;
        const u32 ce0 = CoarseEpochAt(reg.dst_va);
        const u32 ce1 = CoarseEpochAt(reg.dst_va + reg.byte_size - 1u);

        if (reg.byte_size <= kMaxTinyPivot) {
            const size_t set_base = TinyPageSet(reg.dst_va >> 9) * kTinyWays;
            size_t victim = set_base;
            for (size_t w = 0; w < kTinyWays; ++w) {
                const auto& s = tiny_meta[set_base + w];
                if (s.byte_size == 0u || s.dst_va == reg.dst_va) {
                    victim = set_base + w;
                    break;
                }
                if (s.last_use < tiny_meta[victim].last_use) {
                    victim = set_base + w;
                }
            }
            auto& dst = tiny_meta[victim];
            dst.dst_va = reg.dst_va;
            dst.byte_size = reg.byte_size;
            dst.last_use = use_stamp;
            dst.upload_gen = gen;
            dst.coarse_e0 = ce0;
            dst.coarse_e1 = ce1;
            std::memcpy(&tiny_bytes[victim * size_t(kTinySlotStride)],
                        reinterpret_cast<const void*>(reg.src_va), reg.byte_size);
        } else if (reg.byte_size <= kMaxSmallBytes) {
            const size_t set_base = SmallPageSet(reg.dst_va >> 11) * kSmallWays;
            size_t victim = set_base;
            for (size_t w = 0; w < kSmallWays; ++w) {
                const auto& s = small_meta[set_base + w];
                if (s.byte_size == 0u || s.dst_va == reg.dst_va) {
                    victim = set_base + w;
                    break;
                }
                if (s.last_use < small_meta[victim].last_use) {
                    victim = set_base + w;
                }
            }
            auto& dst = small_meta[victim];
            dst.dst_va = reg.dst_va;
            dst.byte_size = reg.byte_size;
            dst.last_use = use_stamp;
            dst.upload_gen = gen;
            dst.coarse_e0 = ce0;
            dst.coarse_e1 = ce1;
            std::memcpy(&small_bytes[victim * size_t(kMaxSmallBytes)],
                        reinterpret_cast<const void*>(reg.src_va), reg.byte_size);
        } else {
            const size_t set_base = LargePageSet(reg.dst_va >> 15) * kLargeWays;
            size_t victim = set_base;
            for (size_t w = 0; w < kLargeWays; ++w) {
                const auto& s = large_meta[set_base + w];
                if (s.byte_size == 0u || s.dst_va == reg.dst_va) {
                    victim = set_base + w;
                    break;
                }
                if (s.last_use < large_meta[victim].last_use) {
                    victim = set_base + w;
                }
            }
            auto& dst = large_meta[victim];
            dst.dst_va = reg.dst_va;
            dst.byte_size = reg.byte_size;
            dst.last_use = use_stamp;
            dst.upload_gen = gen;
            dst.coarse_e0 = ce0;
            dst.coarse_e1 = ce1;
            std::memcpy(&large_bytes[victim * size_t(kMaxLargeBytes)],
                        reinterpret_cast<const void*>(reg.src_va), reg.byte_size);
        }
    }
}

const u8* ObjectVelocityCuller::LookupCpuShadow(VAddr addr, u32 min_bytes, u32 size_hint,
                                                u32* out_rem, u64* out_gen) noexcept {
    if (!addr || min_bytes == 0u || min_bytes > kMaxLargeBytes) {
        return nullptr;
    }

    auto try_tiny = [&]() -> const u8* {
        if (min_bytes > kMaxTinyPivot) {
            return nullptr;
        }
        const VAddr page = addr >> 9;
        for (u32 step = 0; step < 2u; ++step) {
            if (step == 1u && page == 0u) {
                break;
            }
            const size_t set_base = TinyPageSet(page - step) * kTinyWays;
            for (size_t w = 0; w < kTinyWays; ++w) {
                auto& s = tiny_meta[set_base + w];
                if (s.byte_size == 0u || addr < s.dst_va) {
                    continue;
                }
                const u64 sub_off = addr - s.dst_va;
                if (sub_off >= s.byte_size || (s.byte_size - sub_off) < min_bytes) {
                    continue;
                }
                if (s.coarse_e0 != CoarseEpochAt(s.dst_va) ||
                    s.coarse_e1 != CoarseEpochAt(s.dst_va + s.byte_size - 1u)) [[unlikely]] {
                    s.byte_size = 0u;
                    continue;
                }
                s.last_use = ++shadow_clock;
                if (out_rem) {
                    *out_rem = static_cast<u32>(s.byte_size - sub_off);
                }
                if (out_gen) {
                    *out_gen = s.upload_gen;
                }
                return &tiny_bytes[(set_base + w) * size_t(kTinySlotStride) + sub_off];
            }
        }
        return nullptr;
    };

    auto try_small = [&]() -> const u8* {
        if (min_bytes > kMaxSmallBytes) {
            return nullptr;
        }
        const VAddr page = addr >> 11;
        for (u32 step = 0; step < 2u; ++step) {
            if (step == 1u && page == 0u) {
                break;
            }
            const size_t set_base = SmallPageSet(page - step) * kSmallWays;
            for (size_t w = 0; w < kSmallWays; ++w) {
                auto& s = small_meta[set_base + w];
                if (s.byte_size == 0u || addr < s.dst_va) {
                    continue;
                }
                const u64 sub_off = addr - s.dst_va;
                if (sub_off >= s.byte_size || (s.byte_size - sub_off) < min_bytes) {
                    continue;
                }
                if (s.coarse_e0 != CoarseEpochAt(s.dst_va) ||
                    s.coarse_e1 != CoarseEpochAt(s.dst_va + s.byte_size - 1u)) [[unlikely]] {
                    s.byte_size = 0u;
                    continue;
                }
                s.last_use = ++shadow_clock;
                if (out_rem) {
                    *out_rem = static_cast<u32>(s.byte_size - sub_off);
                }
                if (out_gen) {
                    *out_gen = s.upload_gen;
                }
                return &small_bytes[(set_base + w) * size_t(kMaxSmallBytes) + sub_off];
            }
        }
        return nullptr;
    };

    auto try_large = [&]() -> const u8* {
        const VAddr page = addr >> 15;
        for (u32 step = 0; step < 2u; ++step) {
            if (step == 1u && page == 0u) {
                break;
            }
            const size_t set_base = LargePageSet(page - step) * kLargeWays;
            for (size_t w = 0; w < kLargeWays; ++w) {
                auto& s = large_meta[set_base + w];
                if (s.byte_size == 0u || addr < s.dst_va) {
                    continue;
                }
                const u64 sub_off = addr - s.dst_va;
                if (sub_off >= s.byte_size || (s.byte_size - sub_off) < min_bytes) {
                    continue;
                }
                if (s.coarse_e0 != CoarseEpochAt(s.dst_va) ||
                    s.coarse_e1 != CoarseEpochAt(s.dst_va + s.byte_size - 1u)) [[unlikely]] {
                    s.byte_size = 0u;
                    continue;
                }
                s.last_use = ++shadow_clock;
                if (out_rem) {
                    *out_rem = static_cast<u32>(s.byte_size - sub_off);
                }
                if (out_gen) {
                    *out_gen = s.upload_gen;
                }
                return &large_bytes[(set_base + w) * size_t(kMaxLargeBytes) + sub_off];
            }
        }
        return nullptr;
    };

    // Probe the hinted size tier first (`1` cache line hit on `99%` of lookups):
    if (size_hint > kMaxSmallBytes) {
        if (const u8* p = try_large()) return p;
        if (const u8* p = try_small()) return p;
        return try_tiny();
    }
    if (size_hint > kMaxTinyPivot) {
        if (const u8* p = try_small()) return p;
        if (const u8* p = try_large()) return p;
        return try_tiny();
    }
    if (const u8* p = try_tiny()) return p;
    if (const u8* p = try_small()) return p;
    return try_large();
}

bool ObjectVelocityCuller::TryCullDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                       bool is_indexed) {
    if (!pipeline || !is_indexed) {
        return false;
    }
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    if (!ps || !IsObjectVelPsHash(ps->pgm_hash)) [[likely]] {
        return false;
    }
    if (!Enabled() || !rasterizer.OnStageA() || FrameCapture::Active() || BbCeStats::Enabled() ||
        rasterizer.liverpool->pipe_dirty.reset) {
        return false;
    }

    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const u64 vs_hash = vs.pgm_hash;
    if (vs_hash == kHavokClothObjectVelVsHash) {
        // Player's dynamic Havok cloth/coat (`0.9 draws/frame`): intentionally never culled.
        ++stat_cloth_kept;
        return false;
    }

    const bool is_std_vs = IsStdObjectVelVsHash(vs_hash);
    const bool is_hyb_vs = IsHybridObjectVelVsHash(vs_hash);

    static const bool verbose_log = [] {
        const char* env = std::getenv("BB_VEL_CULL_LOG");
        return env && env[0] != '\0' && env[0] != '0';
    }();

    auto maybe_log_summary = [&]() noexcept {
        if ((!first_summary_logged && stat_seen >= 1000u) ||
            (verbose_log && stat_seen != 0u && (stat_seen % 10000u) == 0u)) {
            first_summary_logged = true;
            std::printf(
                "[ObjVelCull] Summary(seen=%llu): culled=%llu (%.1f%%) moving=%llu cloth=%llu "
                "pair_hits=%llu fallbacks=%llu (sil=%llu regs=%llu[0x%x] layout=%llu cur=%llu "
                "prev=%llu inst=%llu unk_vs=0x%llx)\n",
                static_cast<unsigned long long>(stat_seen),
                static_cast<unsigned long long>(stat_culled),
                stat_seen ? (100.0 * double(stat_culled) / double(stat_seen)) : 0.0,
                static_cast<unsigned long long>(stat_moving),
                static_cast<unsigned long long>(stat_cloth_kept),
                static_cast<unsigned long long>(stat_pair_hits),
                static_cast<unsigned long long>(stat_fallbacks),
                static_cast<unsigned long long>(fb_silhouette),
                static_cast<unsigned long long>(fb_regs),
                last_reg_bits,
                static_cast<unsigned long long>(fb_layout),
                static_cast<unsigned long long>(fb_cb_cur),
                static_cast<unsigned long long>(fb_cb_prev),
                static_cast<unsigned long long>(fb_inst),
                static_cast<unsigned long long>(last_unk_vs));
            std::fflush(stdout);
        }
    };

    ++stat_seen;
    auto fail = [&](u64& bucket) noexcept {
        ++bucket;
        ++stat_fallbacks;
        maybe_log_summary();
        return false;
    };

    if (!is_std_vs && !is_hyb_vs) {
        last_unk_vs = vs_hash;
        return fail(fb_layout);
    }
    if (IsSilhouetteMaskActive()) [[unlikely]] {
        return fail(fb_silhouette);
    }

    // Note: in PS4 GNM (`pipeline_cache.cpp:1783`), after a 6-MRT G-Buffer pass, BOTH
    // `regs.color_buffers[1..5]` AND `regs.color_target_mask` (`0xFFFFFF`) stay non-zero in
    // hardware context registers; MRT 1..5 are masked out solely via `cb_shader_mask` / `fs_info`,
    // resulting in `pipeline->GetGraphicsKey().mrt_mask == 0x1u` (`num_color_attachments == 1`).
    const auto& regs = rasterizer.liverpool->regs;
    const u32 mrt_mask = pipeline->GetGraphicsKey().mrt_mask;
    u32 reg_fail_bits = 0u;
    if ((mrt_mask & ~0x81u) != 0u || (mrt_mask & 0x1u) == 0u) {
        reg_fail_bits |= 0x01u;
    }
    if (!regs.color_buffers[0]) {
        reg_fail_bits |= 0x02u;
    }
    if (regs.depth_render_control.depth_clear_enable) {
        reg_fail_bits |= 0x04u;
    }
    if (regs.depth_render_control.stencil_clear_enable) {
        reg_fail_bits |= 0x08u;
    }
    if (vs.uses_dma || ps->uses_dma) {
        reg_fail_bits |= 0x10u;
    }
    for (u32 st = 0u; st < Shader::MaxStageTypes; ++st) {
        const auto* s = pipeline->GetStages()[st];
        if (s && st != u32(Shader::SwStage::Vertex) && st != u32(Shader::SwStage::Fragment)) {
            reg_fail_bits |= 0x20u;
        }
    }
    if (reg_fail_bits != 0u) {
        last_reg_bits |= reg_fail_bits;
        return fail(fb_regs);
    }

    // Collect non-special (`BufferType::Guest`) VS buffer descriptors:
    u32 ns_indices[8]{};
    u32 ns_count = 0u;
    for (u32 bi = 0u; bi < vs.buffers.size(); ++bi) {
        const auto& desc = vs.buffers[bi];
        if (desc.IsSpecial()) {
            continue;
        }
        if (desc.is_written) {
            return fail(fb_layout);
        }
        if (ns_count < 8u) {
            ns_indices[ns_count++] = bi;
        }
    }
    if ((is_std_vs && ns_count < 4u) || (is_hyb_vs && ns_count < 5u)) {
        return fail(fb_layout);
    }

    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
    const bool has_cue = (cue_end > cue_base);

    auto resolve_cb_bytes = [&](const Gcn::Buffer& sharp, u32 min_bytes, u32& out_size,
                                u64& out_gen) -> const u8* {
        out_size = 0u;
        out_gen = 0ULL;
        if (!sharp.Valid() || !sharp.base_address) {
            return nullptr;
        }
        const VAddr addr = sharp.base_address;
        u64 raw_sz = sharp.GetSize();
        const bool in_cue = has_cue && addr >= cue_base && addr < cue_end;
        if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
            raw_sz = VideoCore::DirectBuffers::ClampNativeGpuReadSize(addr, raw_sz);
        } else if (in_cue) {
            raw_sz = (raw_sz > 65536u)
                         ? VideoCore::DirectBuffers::ClampTransientCueReadSize(addr, raw_sz)
                         : std::min<u64>(raw_sz, cue_end - addr);
        }

        // Tier 1: Stage-A CPU shadow (`CopyBufferBatch` with `kCopyFlagKnownCpuSource`).
        u32 shadow_rem = 0u;
        u64 shadow_gen = 0ULL;
        const u32 hint_sz = static_cast<u32>(std::min<u64>(raw_sz, kMaxLargeBytes));
        if (const u8* sh = LookupCpuShadow(addr, min_bytes, hint_sz, &shadow_rem, &shadow_gen)) {
            const u64 bounded = (raw_sz >= min_bytes && raw_sz <= kMaxLargeBytes)
                                    ? std::min<u64>(raw_sz, shadow_rem)
                                    : u64(shadow_rem);
            if (bounded >= min_bytes) {
                out_size = static_cast<u32>(bounded);
                out_gen = shadow_gen;
                return sh;
            }
        }

        // Tier 2: Transient CPU-written `cueHeap` or `auxCpuRing` (`O(1)` range check, zero locks).
        if ((in_cue && (cue_end - addr) >= min_bytes) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, min_bytes)) {
            const u64 avail = in_cue ? (cue_end - addr) : raw_sz;
            const u64 bounded = std::min<u64>(raw_sz, avail);
            if (bounded >= min_bytes) {
                out_size = static_cast<u32>(std::min<u64>(bounded, 131072ULL));
                return reinterpret_cast<const u8*>(addr);
            }
        }

        // Tier 3: Never read un-shadowed native `DirectBuffers` from guest RAM (stale zeroes).
        if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
            return nullptr;
        }

        // Tier 4: Mapped read-only guest RAM not pending GPU modification.
        uintptr_t mapped_end = 0;
        if (raw_sz >= min_bytes && raw_sz <= kMaxLargeBytes &&
            BbLookupMappedEnd(addr, mapped_end) && raw_sz <= (mapped_end - addr) &&
            !rasterizer.PendingWriteOverlaps(addr, raw_sz) &&
            !rasterizer.buffer_cache.IsRegionGpuModified(addr, raw_sz)) {
            out_size = static_cast<u32>(raw_sz);
            return reinterpret_cast<const u8*>(addr);
        }
        return nullptr;
    };

    const float eps = EpsilonThreshold();
    u32 model_cb_slot = 0u;
    u32 cur_pal_slot = 0u;
    u32 prev_pal_slot = 0u;

    if (is_hyb_vs) {
        // `0x8bc6ea32`: `ns[0].dword[32]` (`+128`) selects between rigid `cur_3x4 / prev_3x4`
        // in `ns[1]` (`96 B`) and bone palette skinning in `ns[3]` / `ns[4]`.
        const Gcn::Buffer flag_sharp = vs.buffers[ns_indices[0]].GetSharp(vs);
        u32 flag_sz = 0u;
        u64 flag_gen = 0ULL;
        const u8* flag_ptr = resolve_cb_bytes(flag_sharp, 132u, flag_sz, flag_gen);
        if (!flag_ptr) {
            return fail(fb_cb_cur);
        }
        u32 is_rigid_pair = 0u;
        std::memcpy(&is_rigid_pair, flag_ptr + 128u, sizeof(u32));
        if (is_rigid_pair != 0u) {
            const Gcn::Buffer inst_sharp = vs.buffers[ns_indices[1]].GetSharp(vs);
            u32 inst_sz = 0u;
            u64 inst_gen = 0ULL;
            const u8* inst_ptr = resolve_cb_bytes(inst_sharp, 96u, inst_sz, inst_gen);
            if (!inst_ptr) {
                return fail(fb_inst);
            }
            if (MatricesStationarySse2(inst_ptr, inst_ptr + 48u, 48u, eps)) {
                ++stat_culled;
                BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
                maybe_log_summary();
                return true;
            }
            ++stat_moving;
            maybe_log_summary();
            return false;
        }
        model_cb_slot = ns_indices[2];
        cur_pal_slot = ns_indices[3];
        prev_pal_slot = ns_indices[4];
    } else {
        model_cb_slot = ns_indices[0];
        cur_pal_slot = ns_indices[2];
        prev_pal_slot = ns_indices[3];
    }

    const Gcn::Buffer cur_sharp = vs.buffers[cur_pal_slot].GetSharp(vs);
    const Gcn::Buffer prev_sharp = vs.buffers[prev_pal_slot].GetSharp(vs);
    if (!cur_sharp.Valid() || !prev_sharp.Valid() || !cur_sharp.base_address ||
        !prev_sharp.base_address) {
        return fail(fb_layout);
    }

    // Trivial GPU-binding identity: if both `cur` and `prev` bind the exact same buffer address
    // and size, `cur[i] == prev[i]` bit-for-bit on the GPU regardless of contents.
    if (cur_sharp.base_address == prev_sharp.base_address &&
        cur_sharp.GetSize() == prev_sharp.GetSize()) {
        ++stat_culled;
        BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
        maybe_log_summary();
        return true;
    }

    // Optional `base_bone_idx` at `model_cb + 80` (`dword[20]`):
    u32 base_bone_idx = 0u;
    bool has_base_bone = false;
    {
        const Gcn::Buffer model_sharp = vs.buffers[model_cb_slot].GetSharp(vs);
        u32 model_sz = 0u;
        u64 model_gen = 0ULL;
        if (const u8* model_ptr = resolve_cb_bytes(model_sharp, 84u, model_sz, model_gen)) {
            std::memcpy(&base_bone_idx, model_ptr + 80u, sizeof(u32));
            has_base_bone = true;
        }
    }

    const u64 min_needed_bytes =
        (vs_hash == 0x00000000f2c65c37ULL && has_base_bone && base_bone_idx <= 512u)
            ? (u64(base_bone_idx + 1u) * 48ULL)
            : 48ULL;
    if (min_needed_bytes > kMaxLargeBytes) {
        return fail(fb_layout);
    }

    u32 cur_sz = 0u;
    u64 cur_gen = 0ULL;
    const u8* cur_ptr =
        resolve_cb_bytes(cur_sharp, static_cast<u32>(min_needed_bytes), cur_sz, cur_gen);
    if (!cur_ptr) {
        return fail(fb_cb_cur);
    }

    u32 prev_sz = 0u;
    u64 prev_gen = 0ULL;
    const u8* prev_ptr =
        resolve_cb_bytes(prev_sharp, static_cast<u32>(min_needed_bytes), prev_sz, prev_gen);
    if (!prev_ptr) {
        return fail(fb_cb_prev);
    }

    u32 byte_off = 0u;
    u32 cmp_bytes = 0u;
    if (vs_hash == 0x00000000f2c65c37ULL && has_base_bone) {
        // Rigid 1-bone velocity VS reads solely `[base_bone_idx * 48 .. +48)`.
        const u64 off64 = u64(base_bone_idx) * 48ULL;
        if (off64 + 48ULL > cur_sz || off64 + 48ULL > prev_sz) {
            return fail(fb_layout);
        }
        byte_off = static_cast<u32>(off64);
        cmp_bytes = 48u;
    } else {
        if (cur_sz != prev_sz || cur_sz < 48u || cur_sz > kMaxLargeBytes) {
            return fail(fb_layout);
        }
        const u32 pal_bytes = (cur_sz / 48u) * 48u;
        if (has_base_bone) {
            const u64 off64 = u64(base_bone_idx) * 48ULL;
            if (off64 + 48ULL <= pal_bytes) {
                byte_off = static_cast<u32>(off64);
                // `vs_in_attr2` bone indices are `u8` (`0..255`), so `[base_bone_idx .. +256)`
                // bounds all reachable bones of this submesh.
                cmp_bytes = std::min<u32>(pal_bytes - byte_off, 256u * 48u);
            } else {
                cmp_bytes = pal_bytes;
            }
        } else {
            cmp_bytes = pal_bytes;
        }
    }

    // Check `O(1)` PairMemo when both palettes come from generation-stamped `DirectBuffer` shadows
    // (submeshes 2..N of the same character hit immediately in ~3 ns):
    const bool memo_eligible = (cur_gen != 0ULL && prev_gen != 0ULL);
    size_t memo_idx = 0u;
    if (memo_eligible) {
        memo_idx = HashPairKey(cur_sharp.base_address, prev_sharp.base_address, cur_gen, prev_gen,
                               byte_off, cmp_bytes);
        const auto& m = pair_memo[memo_idx];
        if (m.valid != 0u && m.cur_addr == cur_sharp.base_address &&
            m.prev_addr == prev_sharp.base_address && m.cur_gen == cur_gen &&
            m.prev_gen == prev_gen && m.byte_off == byte_off && m.cmp_bytes == cmp_bytes) {
            ++stat_pair_hits;
            if (m.stationary != 0u) {
                ++stat_culled;
                BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
                maybe_log_summary();
                return true;
            }
            ++stat_moving;
            maybe_log_summary();
            return false;
        }
    }

    const bool stationary =
        MatricesStationarySse2(cur_ptr + byte_off, prev_ptr + byte_off, cmp_bytes, eps);
    if (memo_eligible) {
        auto& m = pair_memo[memo_idx];
        m.cur_addr = cur_sharp.base_address;
        m.prev_addr = prev_sharp.base_address;
        m.cur_gen = cur_gen;
        m.prev_gen = prev_gen;
        m.byte_off = byte_off;
        m.cmp_bytes = cmp_bytes;
        m.stationary = stationary ? 1u : 0u;
        m.valid = 1u;
    }

    if (stationary) {
        ++stat_culled;
        BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
        maybe_log_summary();
        return true;
    }

    ++stat_moving;
    maybe_log_summary();
    return false;
}

} // namespace Vulkan
