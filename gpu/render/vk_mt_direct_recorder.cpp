// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_mt_direct_recorder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>
#include <xxhash.h>

#include "bbport_frame_rate.h"
#include "bbport_settings.h"
#include "buffers/buffer_cache.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "cp/cp_packets.h"
#include "render/direct_shaders.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline.h"
#include "render/rasterizer.h"
#include "render/scheduler.h"
#include "render/vk_frame_capture.h"
#include "render/vk_indirect_guard.h"
#include "render/vk_light_tile_batch.h"
#include "render/vk_obj_vel_cull.h"
#include "render/vk_vsm_shadow_memo.h"
#include "shim/bbport_toggles.h"
#include "textures/direct_textures.h"
#include "textures/texture_cache.h"
#include "upscale/vk_object_motion.h"
#include "upscale/vk_temporal_upscaler.h"
#include "vulkan/instance.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

namespace Vulkan {

void MtCompletedPassMeta::ApplyEndSyncToRegs(Gcn::Regs& dst, Gcn::RegDirty* dirty,
                                             bool track_all) const noexcept {
    if (!has_end_sync) {
        return;
    }
    if (end_reset) {
        BbCp::CommandProcessor::InitGnmDefaultRegs(dst, 3u, dirty, track_all);
    }
    size_t val_cursor = 0;
    for (const u32 hdr : end_span_headers) {
        const u32 word_off = hdr & 0xFFFFu;
        const u32 count = hdr >> 16u;
        if (count != 0u && word_off + count <= Gcn::Regs::NumRegs &&
            val_cursor + count <= end_span_values.size()) {
            std::memcpy(&dst.reg_array[word_off], &end_span_values[val_cursor],
                        count * sizeof(u32));
            if (dirty && dirty->exact_words) [[unlikely]] {
                for (u32 w = word_off; w < word_off + count; ++w) {
                    dirty->exact_words->set(w);
                }
            }
        }
        val_cursor += count;
    }
    if (dirty) {
        dirty->stage_b_mask[0] |= end_stage_b_mask[0];
        dirty->stage_b_mask[1] |= end_stage_b_mask[1];
        if (dirty->track_all_blocks || track_all) {
            dirty->blocks |= end_dirty_blocks;
        }
    }
}

class VkMtDirectRecorder::MtGuestCopyShadowCache {
public:
    static constexpr size_t kNumShards = 16;
    static constexpr u32 kMaxShadowBytes = 32768u;
    static constexpr u32 kMaxHugeLiveBytes = 16u << 20u; // 16 MiB live staging ring window
    static constexpr size_t kHugeLiveSlots = 64u;
    static constexpr u32 kCoarseShift = 18u;        // 256 KiB coarse granule
    static constexpr size_t kCoarseBuckets = 262144u; // covers 64 GiB guest VA

    static constexpr u32 kTinyBytes = 512u;
    static constexpr size_t kTinySets = 128u;
    static constexpr size_t kTinyWays = 4u;

    static constexpr u32 kSmallBytes = 2048u;
    static constexpr size_t kSmallSets = 64u;
    static constexpr size_t kSmallWays = 4u;

    static constexpr u32 kLargeBytes = 32768u;
    static constexpr size_t kLargeSets = 16u;
    static constexpr size_t kLargeWays = 4u;

    struct SlotMeta {
        VAddr dst_va{0};
        VAddr live_src_va{0};
        u32 byte_size{0};
        u32 last_use{0};
        u64 upload_gen{0};
        u32 inv_e0{0};
        u32 inv_e1{0};
    };

    struct HugeLiveSlot {
        VAddr dst_va{0};
        VAddr live_src_va{0};
        u32 byte_size{0};
        u32 inv_bias{0};
        u64 upload_gen{0};
        u64 epoch_sum{0};
    };

    struct alignas(64) Shard {
        mutable std::shared_mutex rw_mutex;
        u32 clock{0};
        std::array<SlotMeta, kTinySets * kTinyWays> tiny_meta{};
        std::array<SlotMeta, kSmallSets * kSmallWays> small_meta{};
        std::array<SlotMeta, kLargeSets * kLargeWays> large_meta{};
        std::vector<u8> tiny_data;
        std::vector<u8> small_data;
        std::vector<u8> large_data;

        Shard()
            : tiny_data(kTinySets * kTinyWays * kTinyBytes, 0u),
              small_data(kSmallSets * kSmallWays * kSmallBytes, 0u),
              large_data(kLargeSets * kLargeWays * kLargeBytes, 0u) {}
    };

    MtGuestCopyShadowCache()
        : coarse_inv_epoch(kCoarseBuckets), coarse_dirty_frame(kCoarseBuckets) {
        for (size_t i = 0; i < kCoarseBuckets; ++i) {
            coarse_inv_epoch[i].store(0u, std::memory_order_relaxed);
            coarse_dirty_frame[i].store(0u, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] static size_t ShardForBlock32k(u64 blk32k) noexcept {
        return static_cast<size_t>((blk32k * 0x9E3779B97F4A7C15ULL) >> 60u);
    }

    [[nodiscard]] u32 InvEpochAt(VAddr va) const noexcept {
        return coarse_inv_epoch[static_cast<size_t>((va >> kCoarseShift) & (kCoarseBuckets - 1u))]
            .load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 EpochSumRange(VAddr dst_va, u32 byte_size) const noexcept {
        if (!dst_va || byte_size == 0u) {
            return 0ULL;
        }
        const u64 p0 = dst_va >> kCoarseShift;
        const u64 p1 = (dst_va + byte_size - 1ULL) >> kCoarseShift;
        u64 sum = 0ULL;
        for (u64 p = p0; p <= p1; ++p) {
            sum += coarse_inv_epoch[static_cast<size_t>(p & (kCoarseBuckets - 1u))].load(
                std::memory_order_acquire);
        }
        return sum;
    }

    void MarkCoarseDirtyAndInvalidate(VAddr dst_va, u64 byte_size, u32 guest_frame) noexcept {
        if (!dst_va || byte_size == 0ULL) {
            return;
        }
        const u64 p0 = dst_va >> kCoarseShift;
        const u64 p1 = (dst_va + byte_size - 1ULL) >> kCoarseShift;
        if (p1 - p0 > 4096ULL) {
            global_inv_bias.fetch_add(1u, std::memory_order_acq_rel);
            global_dirty_frame.store(guest_frame, std::memory_order_release);
            return;
        }
        for (u64 p = p0; p <= p1; ++p) {
            const size_t idx = static_cast<size_t>(p & (kCoarseBuckets - 1u));
            coarse_inv_epoch[idx].fetch_add(1u, std::memory_order_acq_rel);
            coarse_dirty_frame[idx].store(guest_frame, std::memory_order_release);
        }
    }

    [[nodiscard]] bool WasModifiedRecently(VAddr addr, u64 byte_size,
                                           u32 cur_guest_frame) const noexcept {
        if (!addr || byte_size == 0ULL) {
            return false;
        }
        const u32 gf_all = global_dirty_frame.load(std::memory_order_acquire);
        if (gf_all != 0u && gf_all + 2u >= cur_guest_frame) {
            return true;
        }
        const u64 p0 = addr >> kCoarseShift;
        const u64 p1 = (addr + byte_size - 1ULL) >> kCoarseShift;
        const u64 limit = std::min<u64>(p1, p0 + 64ULL);
        for (u64 p = p0; p <= limit; ++p) {
            const u32 df = coarse_dirty_frame[static_cast<size_t>(p & (kCoarseBuckets - 1u))].load(
                std::memory_order_acquire);
            if (df != 0u && df + 2u >= cur_guest_frame) {
                return true;
            }
        }
        return false;
    }

    void RecordBatch(const gx_direct_vk::CopyRegion* regions, u32 count,
                     u32 guest_frame) noexcept {
        if (!regions || count == 0u) {
            return;
        }
        auto* mem = Core::Memory::Instance();
        for (u32 i = 0; i < count; ++i) {
            const auto& r = regions[i];
            if (!r.dst_va || r.byte_size == 0u) {
                continue;
            }
            const bool cpu_src = (r.reserved == gx_direct_vk::kCopyFlagKnownCpuSource);
            const bool live_src = (r.reserved == gx_direct_vk::kCopyFlagLiveStagingPtr);
            if ((!cpu_src && !live_src) || r.byte_size < 4u || r.byte_size > kMaxHugeLiveBytes ||
                !r.src_va || (mem && mem->ClampRangeSize(r.src_va, r.byte_size) != r.byte_size)) {
                MarkCoarseDirtyAndInvalidate(r.dst_va, r.byte_size, guest_frame);
                continue;
            }
            if (r.byte_size > kMaxShadowBytes) {
                const u64 gen = upload_gen.fetch_add(1ULL, std::memory_order_relaxed) + 1ULL;
                const u32 bias = global_inv_bias.load(std::memory_order_acquire);
                const u64 esum = EpochSumRange(r.dst_va, r.byte_size);
                std::unique_lock hlk{huge_rw_mutex};
                size_t victim = huge_cursor & (kHugeLiveSlots - 1u);
                bool reused = false;
                for (size_t i = 0; i < kHugeLiveSlots; ++i) {
                    if (huge_live[i].dst_va == r.dst_va && huge_live[i].byte_size == r.byte_size) {
                        victim = i;
                        reused = true;
                        break;
                    }
                }
                if (!reused) {
                    ++huge_cursor;
                }
                huge_live[victim] = HugeLiveSlot{
                    .dst_va = r.dst_va,
                    .live_src_va = r.src_va,
                    .byte_size = r.byte_size,
                    .inv_bias = bias,
                    .upload_gen = gen,
                    .epoch_sum = esum,
                };
                continue;
            }

            const u64 blk32 = r.dst_va >> 15u;
            Shard& sh = shards[ShardForBlock32k(blk32)];
            const u64 gen = upload_gen.fetch_add(1ULL, std::memory_order_relaxed) + 1ULL;
            const u32 bias = global_inv_bias.load(std::memory_order_acquire);
            const u32 e0 = InvEpochAt(r.dst_va) + bias;
            const u32 e1 = InvEpochAt(r.dst_va + r.byte_size - 1ULL) + bias;

            std::unique_lock lk{sh.rw_mutex};
            const u32 clk = ++sh.clock;
            auto insert_into = [&](SlotMeta* base_meta, size_t ways, u8* base_bytes, u32 stride) {
                size_t victim = 0u;
                u32 oldest = ~0u;
                for (size_t w = 0; w < ways; ++w) {
                    const auto& m = base_meta[w];
                    if (m.dst_va == r.dst_va && m.byte_size == r.byte_size) {
                        victim = w;
                        break;
                    }
                    if (m.dst_va == 0) {
                        victim = w;
                        oldest = 0u;
                        break;
                    }
                    if (m.last_use < oldest) {
                        oldest = m.last_use;
                        victim = w;
                    }
                }
                auto& dst_m = base_meta[victim];
                dst_m.dst_va = r.dst_va;
                dst_m.live_src_va = live_src ? r.src_va : 0ULL;
                dst_m.byte_size = r.byte_size;
                dst_m.last_use = clk;
                dst_m.upload_gen = gen;
                dst_m.inv_e0 = e0;
                dst_m.inv_e1 = e1;
                if (!live_src) {
                    std::memcpy(base_bytes + victim * size_t(stride),
                                reinterpret_cast<const void*>(r.src_va), r.byte_size);
                }
            };

            if (r.byte_size <= kTinyBytes) {
                const size_t set = static_cast<size_t>((r.dst_va >> 9u) & (kTinySets - 1u));
                const size_t base = set * kTinyWays;
                insert_into(&sh.tiny_meta[base], kTinyWays,
                            sh.tiny_data.data() + base * kTinyBytes, kTinyBytes);
            } else if (r.byte_size <= kSmallBytes) {
                const size_t set = static_cast<size_t>((r.dst_va >> 11u) & (kSmallSets - 1u));
                const size_t base = set * kSmallWays;
                insert_into(&sh.small_meta[base], kSmallWays,
                            sh.small_data.data() + base * kSmallBytes, kSmallBytes);
            } else {
                const size_t set = static_cast<size_t>((r.dst_va >> 15u) & (kLargeSets - 1u));
                const size_t base = set * kLargeWays;
                insert_into(&sh.large_meta[base], kLargeWays,
                            sh.large_data.data() + base * kLargeBytes, kLargeBytes);
            }
        }
    }

    [[nodiscard]] bool CopyBytes(VAddr addr, u32 min_bytes, u32 size_hint, void* dst, u32 dst_cap,
                                 u32& out_copied, u64& out_gen) const noexcept {
        out_copied = 0u;
        out_gen = 0ULL;
        if (!addr || min_bytes == 0u || !dst || dst_cap < min_bytes) {
            return false;
        }
        auto* mem = Core::Memory::Instance();
        const u32 bias = global_inv_bias.load(std::memory_order_acquire);
        const u64 blk_cur = addr >> 15u;

        SlotMeta best_meta{};
        bool found = false;

        auto probe_block = [&](u64 blk32) {
            const Shard& sh = shards[ShardForBlock32k(blk32)];
            std::shared_lock lk{sh.rw_mutex};

            auto check_ways = [&](const SlotMeta* base_meta, size_t ways, const u8* base_bytes,
                                  u32 stride) {
                for (size_t w = 0; w < ways; ++w) {
                    const auto& m = base_meta[w];
                    if (m.dst_va == 0 || addr < m.dst_va) {
                        continue;
                    }
                    const u64 off = addr - m.dst_va;
                    if (off >= m.byte_size || (m.byte_size - off) < min_bytes) {
                        continue;
                    }
                    if (m.upload_gen <= best_meta.upload_gen) {
                        continue;
                    }
                    if (InvEpochAt(m.dst_va) + bias != m.inv_e0 ||
                        InvEpochAt(m.dst_va + m.byte_size - 1ULL) + bias != m.inv_e1) {
                        continue;
                    }
                    const u32 avail = static_cast<u32>(m.byte_size - off);
                    const u32 to_copy = std::min<u32>(avail, dst_cap);
                    if (m.live_src_va != 0ULL) {
                        const VAddr src_at = m.live_src_va + off;
                        if (mem && mem->ClampRangeSize(src_at, to_copy) != to_copy) {
                            continue;
                        }
                        std::memcpy(dst, reinterpret_cast<const void*>(src_at), to_copy);
                    } else {
                        std::memcpy(dst, base_bytes + w * size_t(stride) + off, to_copy);
                    }
                    best_meta = m;
                    out_copied = to_copy;
                    out_gen = m.upload_gen;
                    found = true;
                }
            };

            if (size_hint <= kTinyBytes) {
                const u64 g9 = addr >> 9u;
                for (u64 cand : {g9, g9 ? (g9 - u64(1)) : g9}) {
                    if ((cand >> 6u) == blk32) {
                        const size_t base = static_cast<size_t>(cand & (kTinySets - 1u)) * kTinyWays;
                        check_ways(&sh.tiny_meta[base], kTinyWays,
                                   sh.tiny_data.data() + base * kTinyBytes, kTinyBytes);
                    }
                    if (cand == 0ULL) {
                        break;
                    }
                }
            }
            if (size_hint <= kSmallBytes) {
                const u64 g11 = addr >> 11u;
                for (u64 cand : {g11, g11 ? (g11 - u64(1)) : g11}) {
                    if ((cand >> 4u) == blk32) {
                        const size_t base =
                            static_cast<size_t>(cand & (kSmallSets - 1u)) * kSmallWays;
                        check_ways(&sh.small_meta[base], kSmallWays,
                                   sh.small_data.data() + base * kSmallBytes, kSmallBytes);
                    }
                    if (cand == 0ULL) {
                        break;
                    }
                }
            }
            {
                const size_t base = static_cast<size_t>(blk32 & (kLargeSets - 1u)) * kLargeWays;
                check_ways(&sh.large_meta[base], kLargeWays,
                           sh.large_data.data() + base * kLargeBytes, kLargeBytes);
            }
        };

        probe_block(blk_cur);
        if (!found && blk_cur > 0ULL) {
            probe_block(blk_cur - 1ULL);
        }
        if (huge_cursor != 0u) {
            std::shared_lock hlk{huge_rw_mutex};
            for (const auto& hm : huge_live) {
                if (hm.dst_va == 0ULL || addr < hm.dst_va) {
                    continue;
                }
                const u64 off = addr - hm.dst_va;
                if (off >= hm.byte_size || (hm.byte_size - off) < min_bytes) {
                    continue;
                }
                if (hm.upload_gen <= best_meta.upload_gen) {
                    continue;
                }
                if (hm.inv_bias != bias || EpochSumRange(hm.dst_va, hm.byte_size) != hm.epoch_sum) {
                    continue;
                }
                const u32 avail = static_cast<u32>(std::min<u64>(hm.byte_size - off, dst_cap));
                const VAddr src_at = hm.live_src_va + off;
                if (mem && mem->ClampRangeSize(src_at, avail) != avail) {
                    continue;
                }
                std::memcpy(dst, reinterpret_cast<const void*>(src_at), avail);
                best_meta.upload_gen = hm.upload_gen;
                out_copied = avail;
                out_gen = hm.upload_gen;
                found = true;
            }
        }
        return found;
    }

    [[nodiscard]] bool HasValidHugeLiveCover(VAddr addr, u64 min_bytes) const noexcept {
        if (!addr || min_bytes == 0ULL || huge_cursor == 0u) {
            return false;
        }
        const u32 bias = global_inv_bias.load(std::memory_order_acquire);
        std::shared_lock hlk{huge_rw_mutex};
        for (const auto& hm : huge_live) {
            if (hm.dst_va == 0ULL || addr < hm.dst_va) {
                continue;
            }
            const u64 off = addr - hm.dst_va;
            if (off < hm.byte_size && (hm.byte_size - off) >= min_bytes &&
                hm.inv_bias == bias && EpochSumRange(hm.dst_va, hm.byte_size) == hm.epoch_sum) {
                return true;
            }
        }
        return false;
    }

private:
    std::array<Shard, kNumShards> shards{};
    mutable std::shared_mutex huge_rw_mutex;
    std::array<HugeLiveSlot, kHugeLiveSlots> huge_live{};
    u32 huge_cursor{0};
    std::vector<std::atomic<u32>> coarse_inv_epoch;
    std::vector<std::atomic<u32>> coarse_dirty_frame;
    std::atomic<u32> global_inv_bias{0};
    std::atomic<u32> global_dirty_frame{0};
    std::atomic<u64> upload_gen{1};
};

namespace {
constexpr u32 kSecondaryGrowStep = 8;

[[nodiscard]] u64 CurrentOsThreadKey() noexcept {
#ifdef _WIN32
    return static_cast<u64>(GetCurrentThreadId()) | 1ULL;
#else
    return static_cast<u64>(pthread_self()) | 1ULL;
#endif
}
} // namespace

int VkMtDirectRecorder::ConfiguredMode() noexcept {
    static const int cached_mode = []() -> int {
        const char* env = std::getenv("BB_MT_DIRECT_VK");
        if (!env || env[0] == '\0') {
            return 0; // Default off (Stage A + Stage B fast path; opt-in via BB_MT_DIRECT_VK=1/2)
        }
        return std::clamp<int>(static_cast<int>(std::strtol(env, nullptr, 10)), 0, 2);
    }();
    return cached_mode;
}

void VkMtDirectRecorder::WorkerLocalStats::FlushToGlobal(Rasterizer& rasterizer_) noexcept {
    if (native_buf_hits != 0) {
        BbStats::native_buf_hits.fetch_add(native_buf_hits, std::memory_order_relaxed);
    }
    if (native_buf_misses != 0) {
        BbStats::native_buf_misses.fetch_add(native_buf_misses, std::memory_order_relaxed);
    }
    if (native_tex_hits != 0) {
        BbStats::native_tex_hits.fetch_add(native_tex_hits, std::memory_order_relaxed);
    }
    if (native_tex_misses != 0) {
        BbStats::native_tex_misses.fetch_add(native_tex_misses, std::memory_order_relaxed);
    }
    if (desc_push_total != 0) {
        BbStats::desc_push_total.fetch_add(desc_push_total, std::memory_order_relaxed);
    }
    if (desc_push_saved != 0) {
        BbStats::desc_push_saved.fetch_add(desc_push_saved, std::memory_order_relaxed);
    }
    if (occ_candidates != 0 || occ_skipped != 0 || occ_probed != 0 || shadow_skipped != 0 ||
        vel_seen != 0 || vel_culled != 0) {
        rasterizer_.FlushWorkerCullerStats(occ_candidates, occ_skipped, occ_probed, shadow_skipped,
                                           vel_seen, vel_culled);
    }
    Reset();
}

u32 VkMtDirectRecorder::AcquireThreadWorkerId() noexcept {
#ifdef _WIN32
    static const DWORD g_win32_main_tid = GetCurrentThreadId();
#endif
    const u64 key = CurrentOsThreadKey();
    for (u32 i = 0; i < kMaxWorkers; ++i) {
        if (slot_thread_keys[i].load(std::memory_order_relaxed) == key) [[likely]] {
            return i;
        }
    }
    for (u32 i = 0; i < kMaxWorkers; ++i) {
        u64 expected = 0ULL;
        if (slot_thread_keys[i].compare_exchange_strong(expected, key, std::memory_order_acq_rel,
                                                        std::memory_order_relaxed)) {
#ifdef _WIN32
            const bool is_main = (GetCurrentThreadId() == g_win32_main_tid);
#else
            const bool is_main = (::gettid() == ::getpid());
#endif
            slot_is_main_thread[i].store(is_main ? 2u : 1u, std::memory_order_release);
            return i;
        }
        if (expected == key) {
            return i;
        }
    }
    return ~0u;
}

VkMtDirectRecorder::WorkerConstantRing::WorkerConstantRing(const Instance& instance_, u32 wid)
    : buffer{instance_, 0, kWorkerConstantRingBytes, VideoCore::MemoryType::Stream,
             fmt::format("MtDirectConstRing:W{}", wid)} {
    ASSERT(!buffer.mapped_data.empty());
}

MtConstantAllocation VkMtDirectRecorder::WorkerConstantRing::Allocate(u64 size, u64 alignment,
                                                                      const void* src) {
    if (size == 0 || size > kWorkerConstantRingBytes) {
        return {};
    }
    alignment = std::max<u64>(16ULL, alignment);
    u64 at = (position + alignment - 1ULL) & ~(alignment - 1ULL);
    if ((at % kWorkerConstantRingBytes) + size > kWorkerConstantRingBytes) {
        at += kWorkerConstantRingBytes - (at % kWorkerConstantRingBytes);
    }
    if ((at + size > kWorkerConstantRingBytes) &&
        (at + size - kWorkerConstantRingBytes > retired)) {
        return {};
    }
    position = at + size;
    const u64 ring_off = at % kWorkerConstantRingBytes;
    u8* dst = buffer.mapped_data.data() + ring_off;
    if (src) {
        std::memcpy(dst, src, size);
        if (!buffer.is_coherent) {
            buffer.Flush(ring_off, size);
        }
    }
    return MtConstantAllocation{
        .buffer = buffer.Handle(),
        .offset = ring_off,
        .mapped = dst,
    };
}

void VkMtDirectRecorder::WorkerSlot::ReclaimCompletedPools(const Instance& instance_,
                                                           Scheduler& scheduler_) {
    const vk::Device device = instance_.GetDevice();
    const u64 cur_frame = BbStats::guest_frame_number.load(std::memory_order_relaxed);
    for (auto& fp : pools) {
        if (!fp.dirty_since_reset || fp.guest_frame == cur_frame) {
            continue;
        }
        if (fp.last_cmd_seq == 0ULL) {
            // Only aborted buffers were allocated in this slice; Stage B never received them.
            const vk::Result rst = device.resetCommandPool(*fp.pool);
            ASSERT_MSG(rst == vk::Result::eSuccess, "resetCommandPool failed: {}",
                       vk::to_string(rst));
            fp.next_buffer = 0;
            fp.guest_frame = ~0ull;
            fp.end_ring_pos = 0;
            fp.dirty_since_reset = false;
            continue;
        }
        if (!stamp_table) {
            continue;
        }
        const auto& st = stamp_table->slots[fp.last_cmd_seq & (kStampSlots - 1u)];
        const u64 stamped_seq = st.seq.load(std::memory_order_acquire);
        const u64 stamped_tick = st.tick.load(std::memory_order_acquire);
        if (stamped_seq == fp.last_cmd_seq && stamped_tick != 0ULL &&
            scheduler_.IsFree(stamped_tick)) {
            const vk::Result rst = device.resetCommandPool(*fp.pool);
            ASSERT_MSG(rst == vk::Result::eSuccess, "resetCommandPool failed: {}",
                       vk::to_string(rst));
            if (constant_ring && fp.end_ring_pos > constant_ring->retired) {
                constant_ring->retired = fp.end_ring_pos;
            }
            fp.next_buffer = 0;
            fp.guest_frame = ~0ull;
            fp.last_cmd_seq = 0;
            fp.end_ring_pos = 0;
            fp.dirty_since_reset = false;
        }
    }
}

vk::CommandBuffer VkMtDirectRecorder::WorkerSlot::AcquireSecondaryBuffer(
    const Instance& instance_, Scheduler& scheduler_) {
    const vk::Device device = instance_.GetDevice();
    const u64 cur_frame = BbStats::guest_frame_number.load(std::memory_order_relaxed);
    ReclaimCompletedPools(instance_, scheduler_);

    const u32 pool_count = static_cast<u32>(pools.size());
    u32 chosen = ~0u;

    // 1. Prefer continuing the active pool slice already opened for `cur_frame`.
    for (u32 probe = 0; probe < pool_count; ++probe) {
        const u32 idx = (cur_slice + probe) % pool_count;
        const auto& fp = pools[idx];
        if (fp.dirty_since_reset && fp.guest_frame == cur_frame) {
            chosen = idx;
            break;
        }
    }

    // 2. Otherwise grab a clean reset slice for `cur_frame`.
    if (chosen == ~0u) {
        for (u32 probe = 0; probe < pool_count; ++probe) {
            const u32 idx = (cur_slice + probe) % pool_count;
            if (!pools[idx].dirty_since_reset) {
                chosen = idx;
                break;
            }
        }
    }

    if (chosen == ~0u) {
        // All preallocated slices are currently in flight on Stage B / GPU: append an overflow pool
        // so the worker never blocks on `vkWaitSemaphores` mid-frame.
        FrameCmdPool overflow{};
        auto [res, created] = device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = instance_.GetGraphicsQueueFamilyIndex(),
        });
        ASSERT_MSG(res == vk::Result::eSuccess, "MtDirect CommandPool: {}", vk::to_string(res));
        overflow.pool = std::move(created);
        SetObjectName(device, *overflow.pool, "MtDirectCmdPool:W{}:S{}", worker_id, pool_count);
        pools.push_back(std::move(overflow));
        chosen = pool_count;
    }

    cur_slice = chosen;
    active_pool_idx = chosen;
    auto& fp = pools[chosen];
    if (fp.next_buffer >= fp.secondary_buffers.size()) {
        std::array<vk::CommandBuffer, kSecondaryGrowStep> fresh{};
        const vk::CommandBufferAllocateInfo alloc_info{
            .commandPool = *fp.pool,
            .level = vk::CommandBufferLevel::eSecondary,
            .commandBufferCount = kSecondaryGrowStep,
        };
        const vk::Result alloc_res = device.allocateCommandBuffers(&alloc_info, fresh.data());
        ASSERT_MSG(alloc_res == vk::Result::eSuccess, "MtDirect SecondaryCmdBuf: {}",
                   vk::to_string(alloc_res));
        const size_t base_idx = fp.secondary_buffers.size();
        fp.secondary_buffers.insert(fp.secondary_buffers.end(), fresh.begin(), fresh.end());
        for (size_t k = 0; k < kSecondaryGrowStep; ++k) {
            SetObjectName(device, fresh[k], "MtDirectSecCmd:W{}:S{}:B{}", worker_id, chosen,
                          base_idx + k);
        }
    }

    fp.guest_frame = cur_frame;
    fp.dirty_since_reset = true;
    return fp.secondary_buffers[fp.next_buffer++];
}

struct VkMtDirectRecorder::WorkerCompileScratch {
    struct ChunkCopySlice {
        VAddr dst_va{0};
        VAddr src_va{0};
        u32 byte_size{0};
        u64 gen{0};
    };

    struct ChunkGpuWriteSlice {
        VAddr dst_va{0};
        u32 byte_size{0};
        u64 gen{0};
    };

    struct ChunkTargetWrite {
        VAddr addr{0};
        u64 size{0};
    };

    struct UploadedConstMemo {
        VAddr addr{0};
        u32 byte_size{0};
        u64 gen{0};
        vk::Buffer handle{};
        u64 offset{0};
    };

    struct StreamItem {
        enum class Kind : u8 {
            RawPkt = 0,
            SubPass = 1,
        };
        Kind kind{Kind::RawPkt};
        u32 subpass_idx{0};
        const u32* ptr{nullptr};
        u32 words{0};

        static constexpr StreamItem Raw(const u32* p, u32 w) noexcept {
            return StreamItem{Kind::RawPkt, 0u, p, w};
        }
        static constexpr StreamItem Pass(u32 idx) noexcept {
            return StreamItem{Kind::SubPass, idx, nullptr, 7u};
        }
    };

    struct StagedPkt {
        u32 offset{0};
        u32 words{0};
    };

    struct SubPassPlan {
        u32 first_draw_ordinal{0};
        const GraphicsPipeline* first_pso{nullptr};
        Rasterizer::MtPassTargetSummary target_summary{};
        std::array<Gcn::ColorBuffer, Gcn::NUM_COLOR_BUFFERS> cb{};
        Gcn::DepthBuffer db{};
        u32 db_view{0};
        u32 htile{0};
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent{};
        Gcn::CbDbExtent db_extent{};
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> cb_uid{};
        u16 db_uid{0};
        std::array<u32, 0x0180> sh_gfx_regs{};
        std::array<u32, 0x0400> context_regs{};
        std::array<u32, 0x0008> uconfig_so_regs{};
        std::array<u32, 0x0018> uconfig_vgt_regs{};
        bool writes_depth{false};
        bool writes_stencil{false};
        u8 scene_color_submode{0};

        void CaptureRegs(const Gcn::Regs& src) noexcept {
            std::memcpy(sh_gfx_regs.data(), &src.reg_array[0x2C00],
                        sh_gfx_regs.size() * sizeof(u32));
            std::memcpy(context_regs.data(), &src.reg_array[0xA000],
                        context_regs.size() * sizeof(u32));
            std::memcpy(uconfig_so_regs.data(), &src.reg_array[0xC038],
                        uconfig_so_regs.size() * sizeof(u32));
            std::memcpy(uconfig_vgt_regs.data(), &src.reg_array[0xC240],
                        uconfig_vgt_regs.size() * sizeof(u32));
        }
    };

    struct SealedSubPass {
        vk::CommandBuffer sec_cmd{};
        u32 pool_idx{0};
        u64 end_ring_pos{0};
        const GraphicsPipeline* first_pso{nullptr};
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent{};
        Gcn::CbDbExtent db_extent{};
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> cb_uid{};
        u16 db_uid{0};
        std::array<u32, 0x0180> sh_gfx_regs{};
        std::array<u32, 0x0400> context_regs{};
        std::array<u32, 0x0008> uconfig_so_regs{};
        std::array<u32, 0x0018> uconfig_vgt_regs{};
        u32 recorded_draws{0};
        bool is_gbuffer{false};
        bool has_camera_cb{false};
        bool inherits_occlusion_query{false};
        u8 num_sampled_rt_refs{0};
        u8 num_uav_indirect_refs{0};
        u8 motion_slot{0xFFu};
        std::array<MtCompletedPassMeta::SampledRtRef, MtCompletedPassMeta::kMaxSampledRts>
            sampled_rt_refs{};
        std::array<MtCompletedPassMeta::UavIndirectRef, MtCompletedPassMeta::kMaxUavIndirectRefs>
            uav_indirect_refs{};
    };

    alignas(16) std::array<u8, 32768> cb_temp{};
    alignas(16) std::array<u32, 512> vs_flat_ud{};
    alignas(16) std::array<u32, 512> ps_flat_ud{};
    std::bitset<Gcn::Regs::NumRegs> replay_exact_words{};
    std::vector<ChunkCopySlice> chunk_copies;
    std::vector<ChunkGpuWriteSlice> chunk_gpu_writes;
    std::vector<ChunkTargetWrite> chunk_target_writes;
    std::vector<StreamItem> stream_items;
    std::vector<StagedPkt> staged_pkts;
    std::vector<SubPassPlan> subpass_plans;
    std::vector<SealedSubPass> sealed_subpasses;
    std::vector<u32> emit_words;
    std::array<UploadedConstMemo, 32> const_memo{};
    u32 const_memo_cursor{0};

    WorkerCompileScratch() {
        chunk_copies.reserve(128);
        chunk_gpu_writes.reserve(32);
        chunk_target_writes.reserve(32);
        stream_items.reserve(384);
        staged_pkts.reserve(512);
        subpass_plans.reserve(64);
        sealed_subpasses.reserve(64);
        emit_words.reserve(1024);
    }

    void NoteTargetWrite(VAddr addr, u64 size) noexcept {
        if (addr == 0ULL) {
            return;
        }
        const u64 eff_size = std::max<u64>(size, 4096ULL);
        for (const auto& tw : chunk_target_writes) {
            if (tw.addr == addr && tw.size >= eff_size) {
                return;
            }
        }
        chunk_target_writes.push_back({addr, eff_size});
    }

    [[nodiscard]] bool OverlapsTargetWrite(VAddr addr) const noexcept {
        if (addr == 0ULL || chunk_target_writes.empty()) {
            return false;
        }
        for (const auto& tw : chunk_target_writes) {
            const VAddr tw_end = (tw.size > (~0ULL - tw.addr)) ? ~0ULL : (tw.addr + tw.size);
            if (addr >= tw.addr && addr < tw_end) {
                return true;
            }
        }
        return false;
    }

    void ResetConstMemo() noexcept {
        const_memo.fill(UploadedConstMemo{});
        const_memo_cursor = 0;
    }

    [[nodiscard]] u64 MaxGpuWriteGenOverlapping(VAddr addr, u64 size) const noexcept {
        if (!addr || size == 0ULL || chunk_gpu_writes.empty()) {
            return 0ULL;
        }
        const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
        u64 max_gen = 0ULL;
        for (const auto& gw : chunk_gpu_writes) {
            if (!gw.dst_va || gw.byte_size == 0u) {
                continue;
            }
            const VAddr gw_end = gw.dst_va + gw.byte_size;
            if (addr < gw_end && gw.dst_va < end && gw.gen > max_gen) {
                max_gen = gw.gen;
            }
        }
        return max_gen;
    }

    [[nodiscard]] bool FullGpuWriteCovers(VAddr addr, u64 size, u64 min_gen) const noexcept {
        if (!addr || size == 0ULL || chunk_gpu_writes.empty()) {
            return false;
        }
        const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
        for (size_t i = chunk_gpu_writes.size(); i-- > 0;) {
            const auto& gw = chunk_gpu_writes[i];
            if (gw.gen < min_gen) {
                break;
            }
            if (gw.dst_va != 0ULL && addr >= gw.dst_va && end <= gw.dst_va + gw.byte_size) {
                return true;
            }
        }
        return false;
    }
};

VkMtDirectRecorder::WorkerSlot::WorkerSlot() = default;
VkMtDirectRecorder::WorkerSlot::~WorkerSlot() = default;

VkMtDirectRecorder::VkMtDirectRecorder(const Instance& instance_, Scheduler& scheduler_,
                                       Rasterizer& rasterizer_)
    : instance{instance_}, scheduler{scheduler_}, rasterizer{rasterizer_},
      mode{ConfiguredMode()} {
    if (mode > 0) {
        guest_copy_shadow = std::make_unique<MtGuestCopyShadowCache>();
        for (u32 w = 0; w < 6u; ++w) {
            EnsureWorkerInitialized(w);
        }
        std::printf("[bb_vulkan] Phase 4.2b/4.3 VkMtDirectRecorder active (BB_MT_DIRECT_VK=%d, "
                    "prewarmed 6/%u heap worker slots x %u pools, +0 B PT_TLS)\n",
                    mode, kMaxWorkers, kFrameSlices);
    }
    active_instance.store(this, std::memory_order_release);
}

VkMtDirectRecorder::~VkMtDirectRecorder() {
    VkMtDirectRecorder* expected = this;
    active_instance.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

bool VkMtDirectRecorder::EnsureWorkerInitialized(u32 worker_id) {
    if (worker_id >= kMaxWorkers) {
        return false;
    }
    if (worker_ptrs[worker_id].load(std::memory_order_acquire) != nullptr) [[likely]] {
        return true;
    }
    std::scoped_lock lk{worker_init_mutex};
    if (worker_ptrs[worker_id].load(std::memory_order_relaxed) != nullptr) {
        return true;
    }

    const vk::Device device = instance.GetDevice();
    const u32 qf = instance.GetGraphicsQueueFamilyIndex();
    auto slot = std::make_unique<WorkerSlot>();
    slot->worker_id = worker_id;
    slot->pools.resize(kFrameSlices);
    for (u32 s = 0; s < kFrameSlices; ++s) {
        auto [res, created] = device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = qf,
        });
        ASSERT_MSG(res == vk::Result::eSuccess, "MtDirect CommandPool init: {}",
                   vk::to_string(res));
        slot->pools[s].pool = std::move(created);
        SetObjectName(device, *slot->pools[s].pool, "MtDirectCmdPool:W{}:S{}", worker_id, s);
    }
    slot->constant_ring = std::make_unique<WorkerConstantRing>(instance, worker_id);
    slot->stamp_table = std::make_unique<StageBStampTable>();
    slot->completed_meta = std::make_unique<MtCompletedPassMeta[]>(kCompletedMetaSlots);
    slot->scan_regs = std::make_unique<Gcn::Regs>();
    slot->replay_regs = std::make_unique<Gcn::Regs>();
    slot->compile_scratch = std::make_unique<WorkerCompileScratch>();

    WorkerSlot* raw = slot.get();
    workers_owned[worker_id] = std::move(slot);
    worker_ptrs[worker_id].store(raw, std::memory_order_release);
    return true;
}

bool VkMtDirectRecorder::BeginWorkerPass(u32 worker_id, const MtPassInheritanceDesc& pass_desc) {
    if (worker_id >= kMaxWorkers || !EnsureWorkerInitialized(worker_id)) {
        return false;
    }
    auto& w = *worker_ptrs[worker_id].load(std::memory_order_relaxed);
    if (w.is_recording) {
        (void)EndWorkerPass(worker_id);
    }

    const vk::CommandBuffer cmdbuf = w.AcquireSecondaryBuffer(instance, scheduler);
    const u32 color_cnt =
        std::min<u32>(pass_desc.num_color_attachments,
                      static_cast<u32>(pass_desc.color_formats.size()));

    const vk::CommandBufferInheritanceRenderingInfo rendering_inh{
        .viewMask = pass_desc.view_mask,
        .colorAttachmentCount = color_cnt,
        .pColorAttachmentFormats = color_cnt ? pass_desc.color_formats.data() : nullptr,
        .depthAttachmentFormat = pass_desc.depth_format,
        .stencilAttachmentFormat = pass_desc.stencil_format,
        .rasterizationSamples = pass_desc.rasterization_samples,
    };
    const vk::CommandBufferInheritanceInfo inh_info{
        .pNext = &rendering_inh,
        .occlusionQueryEnable = VK_FALSE,
        .queryFlags = {},
    };
    const vk::CommandBufferBeginInfo begin_info{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue |
                 vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        .pInheritanceInfo = &inh_info,
    };
    const vk::Result res = cmdbuf.begin(&begin_info);
    if (res != vk::Result::eSuccess) {
        return false;
    }

    w.active_cmdbuf = cmdbuf;
    w.active_ring_start = w.constant_ring ? w.constant_ring->position : 0ULL;
    w.active_occ_block_id = 0xFFFFu;
    w.is_recording = true;
    w.recorded_draws = 0;
    ++w.desc_epoch_counter;
    w.bound_pipeline = vk::Pipeline{};
    w.push_tracker.InvalidateAll();
    w.pso_state.ResetForPass();
    w.local_stats.Reset();

    if (pass_desc.render_area.extent.width > 0 && pass_desc.render_area.extent.height > 0) {
        const vk::Viewport vp{
            .x = static_cast<float>(pass_desc.render_area.offset.x),
            .y = static_cast<float>(pass_desc.render_area.offset.y),
            .width = static_cast<float>(pass_desc.render_area.extent.width),
            .height = static_cast<float>(pass_desc.render_area.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        cmdbuf.setViewportWithCount(vp);
        cmdbuf.setScissorWithCount(pass_desc.render_area);
    }
    w.dyn_state.Invalidate();
    return true;
}

vk::CommandBuffer VkMtDirectRecorder::EndWorkerPass(u32 worker_id, u64* out_cmd_seq) {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return {};
    }
    const vk::CommandBuffer cmdbuf = w->active_cmdbuf;
    (void)cmdbuf.end();
    const u64 seq = ++w->cmd_seq_gen;
    if (w->active_pool_idx < w->pools.size()) {
        auto& fp = w->pools[w->active_pool_idx];
        fp.last_cmd_seq = seq;
        if (w->constant_ring) {
            fp.end_ring_pos = w->constant_ring->position;
        }
    }
    if (out_cmd_seq) {
        *out_cmd_seq = seq;
    }
    w->local_stats.FlushToGlobal(rasterizer);
    w->active_cmdbuf = vk::CommandBuffer{};
    w->active_occ_block_id = 0xFFFFu;
    w->is_recording = false;
    w->bound_pipeline = vk::Pipeline{};
    w->push_tracker.InvalidateAll();
    return cmdbuf;
}

void VkMtDirectRecorder::AbortWorkerPass(u32 worker_id) noexcept {
    if (worker_id >= kMaxWorkers) {
        return;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording) {
        return;
    }
    if (w->active_cmdbuf) {
        (void)w->active_cmdbuf.end();
    }
    if (w->constant_ring) {
        w->constant_ring->position = w->active_ring_start;
    }
    if (w->active_occ_block_id != 0xFFFFu) {
        rasterizer.AbortMtOcclusionQueryBlock(w->active_occ_block_id);
        w->active_occ_block_id = 0xFFFFu;
    }
    w->local_stats.Reset();
    w->active_cmdbuf = vk::CommandBuffer{};
    w->is_recording = false;
    w->recorded_draws = 0;
    w->bound_pipeline = vk::Pipeline{};
    w->push_tracker.InvalidateAll();
}

vk::CommandBuffer VkMtDirectRecorder::ActiveCommandBuffer(u32 worker_id) const noexcept {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    const auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    return (w && w->is_recording) ? w->active_cmdbuf : vk::CommandBuffer{};
}

const GraphicsPipeline* VkMtDirectRecorder::SelectAndBindPipeline(u32 worker_id,
                                                                  const Gcn::Regs& regs,
                                                                  DrawIndirectParams params,
                                                                  bool indirect,
                                                                  bool allow_compile_fallback) {
    if (worker_id >= kMaxWorkers) {
        return nullptr;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return nullptr;
    }
    const GraphicsPipeline* pso = rasterizer.GetPipelineCache().SelectGraphicsPipelineMt(
        regs, w->pso_state, params, indirect, allow_compile_fallback);
    if (!pso) {
        return nullptr;
    }
    const vk::Pipeline vk_pso = pso->Handle();
    if (w->bound_pipeline != vk_pso) {
        w->active_cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, vk_pso);
        w->bound_pipeline = vk_pso;
    }
    return pso;
}

MtWorkerPsoState* VkMtDirectRecorder::WorkerPsoState(u32 worker_id) noexcept {
    if (worker_id >= kMaxWorkers) {
        return nullptr;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    return w ? &w->pso_state : nullptr;
}

MtConstantAllocation VkMtDirectRecorder::AllocateWorkerConstants(u32 worker_id, u64 byte_size,
                                                                 u64 alignment,
                                                                 const void* initial_src) {
    if (worker_id >= kMaxWorkers || !EnsureWorkerInitialized(worker_id)) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_relaxed);
    if (!w || !w->constant_ring) {
        return {};
    }
    auto alloc = w->constant_ring->Allocate(byte_size, alignment, initial_src);
    if (!alloc) {
        w->ReclaimCompletedPools(instance, scheduler);
        alloc = w->constant_ring->Allocate(byte_size, alignment, initial_src);
    }
    return alloc;
}

VideoCore::DirectBuffers::MtResolvedBuffer VkMtDirectRecorder::ResolveBuffer(
    u32 worker_id, VAddr addr, u64 requested_size) noexcept {
    if (worker_id >= kMaxWorkers) {
        return {};
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w) {
        return {};
    }
    return rasterizer.GetDirectBuffers().TryResolveReadMt(
        addr, requested_size, w->buffer_mru, &w->local_stats.native_buf_hits,
        &w->local_stats.native_buf_misses);
}

vk::ImageView VkMtDirectRecorder::ResolveSrv(u32 worker_id, const Gcn::Image& raw_sharp,
                                             const Shader::ImageResource& res) {
    WorkerSlot* w =
        (worker_id < kMaxWorkers) ? worker_ptrs[worker_id].load(std::memory_order_acquire) : nullptr;
    return rasterizer.ResolveSrvViewMt(
        raw_sharp, res, w ? &w->local_stats.native_tex_hits : nullptr,
        w ? &w->local_stats.native_tex_misses : nullptr);
}

Rasterizer::MtResolvedSrv VkMtDirectRecorder::ResolveSceneSrv(
    u32 worker_id, const Gcn::Image& raw_sharp, const Shader::ImageResource& res,
    const Rasterizer::MtPassTargetSummary& pass_targets) {
    WorkerSlot* w =
        (worker_id < kMaxWorkers) ? worker_ptrs[worker_id].load(std::memory_order_acquire) : nullptr;
    return rasterizer.ResolveSceneSrvViewMt(
        raw_sharp, res, pass_targets, w ? &w->local_stats.native_tex_hits : nullptr,
        w ? &w->local_stats.native_tex_misses : nullptr);
}

vk::Sampler VkMtDirectRecorder::ResolveSampler(u32 worker_id, const Gcn::Sampler& raw_ssharp,
                                               bool is_depth, float lod_bias,
                                               Gcn::BorderColorBuffer bc_base) {
    WorkerSlot* w =
        (worker_id < kMaxWorkers) ? worker_ptrs[worker_id].load(std::memory_order_acquire) : nullptr;
    return rasterizer.ResolveSamplerMt(raw_ssharp, is_depth, lod_bias, bc_base,
                                       w ? &w->local_stats.native_tex_hits : nullptr);
}

void VkMtDirectRecorder::PushGraphicsResources(
    u32 worker_id, vk::PipelineLayout layout, bool uses_push_descriptors,
    const Shader::PushData& push_data, std::span<const vk::WriteDescriptorSet> set_writes) {
    if (worker_id >= kMaxWorkers) {
        return;
    }
    auto* w = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!w || !w->is_recording || !w->active_cmdbuf) {
        return;
    }

    auto& tracker = w->push_tracker;
    tracker.OnDraw(&scheduler, w->desc_epoch_counter, w->desc_epoch_counter, layout);

    const bool need_push_constants =
        !tracker.has_push_data ||
        (std::memcmp(&tracker.push_data, &push_data, sizeof(push_data)) != 0);
    if (need_push_constants) {
        tracker.push_data = push_data;
        tracker.has_push_data = true;
        w->active_cmdbuf.pushConstants(layout, AllGraphicsStageBits, 0u, sizeof(push_data),
                                       &push_data);
    }

    if (!uses_push_descriptors || set_writes.empty()) {
        return;
    }

    std::array<vk::WriteDescriptorSet, kMaxTrackedBindings> delta_writes;
    u32 delta_cnt = 0;
    u64 total_slots = 0;
    u64 saved_slots = 0;
    bool fallback_all = false;

    for (const auto& write : set_writes) {
        total_slots += write.descriptorCount;
        const bool single_img = (write.pImageInfo != nullptr && !write.pBufferInfo);
        const bool single_buf = (write.pBufferInfo != nullptr && !write.pImageInfo);
        if (write.descriptorCount == 1 && write.dstArrayElement == 0 &&
            !write.pTexelBufferView && write.dstBinding < kMaxTrackedBindings &&
            (single_img || single_buf)) [[likely]] {
            const u64 slot_bit = 1ULL << write.dstBinding;
            const bool slot_valid = (tracker.valid_mask & slot_bit) != 0;
            auto& slot = tracker.slots[write.dstBinding];
            if (single_img) {
                const auto& img = *write.pImageInfo;
                const VkSampler s_h = static_cast<VkSampler>(img.sampler);
                const VkImageView v_h = static_cast<VkImageView>(img.imageView);
                if (slot_valid && slot.is_image && slot.type == write.descriptorType &&
                    slot.sampler == s_h && slot.image_view == v_h &&
                    slot.image_layout == img.imageLayout) {
                    ++saved_slots;
                    continue;
                }
                tracker.valid_mask |= slot_bit;
                slot.is_image = true;
                slot.type = write.descriptorType;
                slot.sampler = s_h;
                slot.image_view = v_h;
                slot.image_layout = img.imageLayout;
            } else {
                const auto& buf = *write.pBufferInfo;
                const VkBuffer b_h = static_cast<VkBuffer>(buf.buffer);
                if (slot_valid && !slot.is_image && slot.type == write.descriptorType &&
                    slot.buffer == b_h && slot.offset == buf.offset &&
                    slot.range == buf.range) {
                    ++saved_slots;
                    continue;
                }
                tracker.valid_mask |= slot_bit;
                slot.is_image = false;
                slot.type = write.descriptorType;
                slot.buffer = b_h;
                slot.offset = buf.offset;
                slot.range = buf.range;
            }
            if (delta_cnt < delta_writes.size()) {
                delta_writes[delta_cnt++] = write;
            } else {
                fallback_all = true;
                break;
            }
        } else {
            fallback_all = true;
            break;
        }
    }

    if (fallback_all) {
        tracker.valid_mask = 0;
        w->active_cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0,
                                              set_writes);
        w->local_stats.desc_push_total += total_slots;
        return;
    }

    w->local_stats.desc_push_total += total_slots;
    w->local_stats.desc_push_saved += saved_slots;
    if (delta_cnt != 0) {
        w->active_cmdbuf.pushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, layout, 0,
            std::span<const vk::WriteDescriptorSet>{delta_writes.data(), delta_cnt});
    }
}

void VkMtDirectRecorder::ExecuteSecondaryBuffers(
    std::span<const vk::CommandBuffer> secondaries, u32 worker_id, u64 cmd_seq,
    u16 occ_query_block_id) {
    if (secondaries.empty()) {
        return;
    }

    const auto recorded = scheduler.RecordData(secondaries);
    scheduler.Record([recorded](vk::CommandBuffer primary) {
        primary.executeCommands(static_cast<u32>(recorded.size()), recorded.data());
    });

    const u64 tick = scheduler.CurrentTick();
    if (occ_query_block_id != 0xFFFFu) {
        rasterizer.NoteMtOcclusionQueryBlockExecuted(occ_query_block_id, tick);
    }
    if (worker_id < kMaxWorkers && cmd_seq != 0ULL) [[likely]] {
        auto* slot = worker_ptrs[worker_id].load(std::memory_order_acquire);
        if (slot && slot->stamp_table) [[likely]] {
            auto& st = slot->stamp_table->slots[cmd_seq & (kStampSlots - 1u)];
            st.tick.store(tick, std::memory_order_release);
            st.seq.store(cmd_seq, std::memory_order_release);
        }
    }

    // Secondary command buffers disturb primary command buffer state per Vulkan 1.3 §6.8:
    // bump primary descriptor/push-constant epochs so subsequent Stage B draws rebind cleanly.
    scheduler.BumpDescriptorStateEpoch();
}

void VkMtDirectRecorder::NoteGuestImmediateCopyBatch(const void* raw_regions, u32 count) noexcept {
    if (!guest_copy_shadow || !raw_regions || count == 0u) {
        return;
    }
    const u32 gf = BbStats::guest_frame_number.load(std::memory_order_relaxed);
    guest_copy_shadow->RecordBatch(static_cast<const gx_direct_vk::CopyRegion*>(raw_regions), count,
                                   gf);
}

void VkMtDirectRecorder::InvalidateGuestCopyShadow(VAddr dst_va, u64 byte_size) noexcept {
    if (!guest_copy_shadow || !dst_va || byte_size == 0ULL) {
        return;
    }
    const u32 gf = BbStats::guest_frame_number.load(std::memory_order_relaxed);
    guest_copy_shadow->MarkCoarseDirtyAndInvalidate(dst_va, byte_size, gf);
}

bool VkMtDirectRecorder::CopyFromGuestShadow(VAddr addr, u32 min_bytes, u32 size_hint, void* dst,
                                             u32 dst_cap, u32& out_copied,
                                             u64& out_gen) noexcept {
    if (!guest_copy_shadow) {
        out_copied = 0u;
        out_gen = 0ULL;
        return false;
    }
    return guest_copy_shadow->CopyBytes(addr, min_bytes, size_hint, dst, dst_cap, out_copied,
                                        out_gen);
}

bool VkMtDirectRecorder::WasCoarseRangeModifiedRecently(VAddr addr, u64 byte_size,
                                                        u32 cur_guest_frame) const noexcept {
    return guest_copy_shadow &&
           guest_copy_shadow->WasModifiedRecently(addr, byte_size, cur_guest_frame);
}

void VkMtDirectRecorder::StampEmptyPassOnStageB(u32 worker_id, u64 cmd_seq,
                                                u16 occ_query_block_id) noexcept {
    const u64 tick = scheduler.CurrentTick();
    if (occ_query_block_id != 0xFFFFu) {
        rasterizer.NoteMtOcclusionQueryBlockExecuted(occ_query_block_id, tick);
    }
    if (worker_id < kMaxWorkers && cmd_seq != 0ULL) [[likely]] {
        auto* slot = worker_ptrs[worker_id].load(std::memory_order_acquire);
        if (slot && slot->stamp_table) [[likely]] {
            auto& st = slot->stamp_table->slots[cmd_seq & (kStampSlots - 1u)];
            st.tick.store(tick, std::memory_order_release);
            st.seq.store(cmd_seq, std::memory_order_release);
        }
    }
}

MtCompletedPassMeta* VkMtDirectRecorder::LookupCompletedPass(u32 worker_id,
                                                             u64 cmd_seq) noexcept {
    if (worker_id >= kMaxWorkers || cmd_seq == 0ULL) {
        return nullptr;
    }
    auto* slot = worker_ptrs[worker_id].load(std::memory_order_acquire);
    if (!slot || !slot->completed_meta) {
        return nullptr;
    }
    auto& meta = slot->completed_meta[cmd_seq & (kCompletedMetaSlots - 1u)];
    if (meta.published_seq.load(std::memory_order_acquire) != cmd_seq) {
        return nullptr;
    }
    return &meta;
}

namespace {

Gcn::Scissor MtCombinedScissor(const Gcn::Regs& regs) noexcept {
    const bool offset = !regs.window_scissor.window_offset_disable;
    const s16 offset_x = offset ? regs.window_offset.window_x_offset : 0;
    const s16 offset_y = offset ? regs.window_offset.window_y_offset : 0;
    const auto top_left = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::max({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    const auto bottom_right = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::min({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    Gcn::Scissor scissor{};
    scissor.top_left_x =
        top_left(regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
                 s16(regs.generic_scissor.top_left_x), offset_x);
    scissor.top_left_y =
        top_left(regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
                 s16(regs.generic_scissor.top_left_y), offset_y);
    scissor.bottom_right_x =
        bottom_right(regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
                     regs.generic_scissor.bottom_right_x, offset_x);
    scissor.bottom_right_y =
        bottom_right(regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
                     regs.generic_scissor.bottom_right_y, offset_y);
    return scissor;
}

std::pair<float, float> MtDepthRange(const Gcn::Regs& regs, const Gcn::ViewportBounds& vp,
                                     bool unrestricted) noexcept {
    const auto& ctl = regs.viewport_control;
    const float zoffset = ctl.zoffset_enable ? vp.zoffset : 0.f;
    const float zscale = ctl.zscale_enable ? vp.zscale : 1.f;
    float min_depth = regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW
                          ? zoffset - zscale
                          : zoffset;
    float max_depth = zoffset + zscale;
    if (!unrestricted) {
        min_depth = std::max(min_depth, 0.f);
        max_depth = std::min(max_depth, 1.f);
    }
    return {min_depth, max_depth};
}

bool MtIsListTopology(Gcn::PrimitiveType type) noexcept {
    switch (GcnToVk::PrimitiveType(type)) {
    case vk::PrimitiveTopology::ePointList:
    case vk::PrimitiveTopology::eLineList:
    case vk::PrimitiveTopology::eTriangleList:
    case vk::PrimitiveTopology::eLineListWithAdjacency:
    case vk::PrimitiveTopology::eTriangleListWithAdjacency:
        return true;
    default:
        return false;
    }
}

bool MtIsPatchListTopology(Gcn::PrimitiveType type) noexcept {
    return type == Gcn::PrimitiveType::PatchPrimitive || type == Gcn::PrimitiveType::QuadList ||
           type == Gcn::PrimitiveType::RectList;
}

bool MtUsesOpValue(Gcn::StencilFunc fail, Gcn::StencilFunc zpass, Gcn::StencilFunc zfail) noexcept {
    return fail == Gcn::StencilFunc::ReplaceOp || zpass == Gcn::StencilFunc::ReplaceOp ||
           zfail == Gcn::StencilFunc::ReplaceOp;
}

void UpdateWorkerColorBlendingState(DynamicState& dyn_state, const Gcn::Regs& regs,
                                    const GraphicsPipeline* pipeline) noexcept {
    dyn_state.SetBlendConstants(regs.blend_constants);
    dyn_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dyn_state.SetAttachmentFeedbackLoopEnabled(false);
}

void UpdateWorkerDynamicStateFull(DynamicState& dyn_state, const Instance& instance,
                                  const Gcn::Regs& regs, const GraphicsPipeline* pipeline,
                                  bool is_indexed, const std::array<float, 2>& target_scale,
                                  const std::array<float, 2>& draw_jitter) {
    // 1. Viewports & Scissors (with upscaler target_scale & scene-depth Halton jitter)
    const Gcn::Scissor combined = MtCombinedScissor(regs);
    const auto& ctl = regs.viewport_control;
    const bool scaled = target_scale[0] != 1.0f || target_scale[1] != 1.0f;
    boost::container::static_vector<vk::Viewport, Gcn::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, Gcn::NUM_VIEWPORTS> scissors;
    for (u32 i = 0; i < Gcn::NUM_VIEWPORTS; ++i) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) {
            continue;
        }
        vk::Viewport viewport{};
        std::tie(viewport.minDepth, viewport.maxDepth) =
            MtDepthRange(regs, vp, instance.IsDepthRangeUnrestrictedSupported());
        if (regs.IsClipDisabled()) {
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16384u));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16384u));
        } else {
            const float xoffset = ctl.xoffset_enable ? vp.xoffset : 0.f;
            const float xscale = ctl.xscale_enable ? vp.xscale : 1.f;
            const float yoffset = ctl.yoffset_enable ? vp.yoffset : 0.f;
            const float yscale = ctl.yscale_enable ? vp.yscale : 1.f;
            viewport.x = (xoffset - xscale) * target_scale[0] + draw_jitter[0];
            viewport.y = (yoffset - yscale) * target_scale[1] + draw_jitter[1];
            viewport.width = xscale * 2.0f * target_scale[0];
            viewport.height = yscale * 2.0f * target_scale[1];
        }
        viewports.push_back(viewport);

        Gcn::Scissor scissor = combined;
        if (regs.mode_control.vport_scissor_enable) {
            const auto& own = regs.viewport_scissors[i];
            scissor.top_left_x = std::max(scissor.top_left_x, s16(own.top_left_x));
            scissor.top_left_y = std::max(scissor.top_left_y, s16(own.top_left_y));
            scissor.bottom_right_x =
                std::min(Gcn::Scissor::Clamp(scissor.bottom_right_x), own.bottom_right_x);
            scissor.bottom_right_y =
                std::min(Gcn::Scissor::Clamp(scissor.bottom_right_y), own.bottom_right_y);
        }
        if (scaled) {
            const auto scale = [](s32 v, float f) { return s32(std::lround(float(v) * f)); };
            const s32 x0 = scale(scissor.top_left_x, target_scale[0]);
            const s32 y0 = scale(scissor.top_left_y, target_scale[1]);
            const s32 x1 = scale(scissor.top_left_x + s32(scissor.GetWidth()), target_scale[0]);
            const s32 y1 = scale(scissor.top_left_y + s32(scissor.GetHeight()), target_scale[1]);
            scissors.push_back({
                .offset = {x0, y0},
                .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
            });
        } else {
            scissors.push_back({
                .offset = {scissor.top_left_x, scissor.top_left_y},
                .extent = {scissor.GetWidth(), scissor.GetHeight()},
            });
        }
    }
    if (viewports.empty()) {
        viewports.push_back(
            {.x = -1.0f, .y = -1.0f, .width = 1.0f, .height = 1.0f, .minDepth = 0.0f, .maxDepth = 1.0f});
        scissors.push_back({.offset = {0, 0}, .extent = {1, 1}});
    }
    dyn_state.SetViewports(viewports);
    dyn_state.SetScissors(scissors);

    // 2. Depth, Depth Bias & Stencil
    const auto& depth = regs.depth_control;
    const bool depth_test = depth.depth_enable && regs.depth_buffer.DepthValid();
    dyn_state.SetDepthTestEnabled(depth_test);
    if (depth_test) {
        dyn_state.SetDepthWriteEnabled(depth.depth_write_enable &&
                                       !regs.depth_render_control.depth_clear_enable);
        dyn_state.SetDepthCompareOp(GcnToVk::CompareOp(depth.depth_func));
    }
    dyn_state.SetDepthBoundsTestEnabled(depth.depth_bounds_enable);
    if (depth.depth_bounds_enable) {
        dyn_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }
    const bool depth_bias = regs.polygon_control.NeedsBias();
    dyn_state.SetDepthBiasEnabled(depth_bias);
    if (depth_bias) {
        const auto& offset = regs.poly_offset;
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dyn_state.SetDepthBias(front ? offset.front_offset : offset.back_offset, offset.depth_bias,
                               (front ? offset.front_scale : offset.back_scale) / 16.f);
    }

    const bool stencil_test = depth.stencil_enable && regs.depth_buffer.StencilValid();
    dyn_state.SetStencilTestEnabled(stencil_test);
    if (stencil_test) {
        const auto& sc = regs.stencil_control;
        const bool two_sided = depth.backface_enable;
        const StencilOps front_ops{
            .fail_op = GcnToVk::StencilOp(sc.stencil_fail_front),
            .pass_op = GcnToVk::StencilOp(sc.stencil_zpass_front),
            .depth_fail_op = GcnToVk::StencilOp(sc.stencil_zfail_front),
            .compare_op = GcnToVk::CompareOp(depth.stencil_ref_func),
        };
        const StencilOps back_ops = two_sided
                                        ? StencilOps{
                                              .fail_op = GcnToVk::StencilOp(sc.stencil_fail_back),
                                              .pass_op = GcnToVk::StencilOp(sc.stencil_zpass_back),
                                              .depth_fail_op =
                                                  GcnToVk::StencilOp(sc.stencil_zfail_back),
                                              .compare_op = GcnToVk::CompareOp(depth.stencil_bf_func),
                                          }
                                        : front_ops;
        dyn_state.SetStencilOps(front_ops, back_ops);

        const auto front = regs.stencil_ref_front;
        const auto back = two_sided ? regs.stencil_ref_back : regs.stencil_ref_front;
        const bool front_op =
            MtUsesOpValue(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op = two_sided ? MtUsesOpValue(sc.stencil_fail_back, sc.stencil_zpass_back,
                                                       sc.stencil_zfail_back)
                                       : front_op;
        dyn_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                       back_op ? back.stencil_op_val : back.stencil_test_val);
        const bool front_keeps = sc.stencil_fail_front == Gcn::StencilFunc::Keep &&
                                 sc.stencil_zpass_front == Gcn::StencilFunc::Keep &&
                                 sc.stencil_zfail_front == Gcn::StencilFunc::Keep;
        const bool back_keeps = two_sided ? (sc.stencil_fail_back == Gcn::StencilFunc::Keep &&
                                             sc.stencil_zpass_back == Gcn::StencilFunc::Keep &&
                                             sc.stencil_zfail_back == Gcn::StencilFunc::Keep)
                                          : front_keeps;
        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        dyn_state.SetStencilWriteMasks((stencil_clear || front_keeps) ? 0U
                                                                      : front.stencil_write_mask,
                                       (stencil_clear || back_keeps) ? 0U
                                                                     : back.stencil_write_mask);
        dyn_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }

    // 3. Primitive, Rasterization & Color Blending
    const auto prim_type = regs.primitive_type;
    const bool restart = (regs.enable_primitive_restart & 1) != 0 &&
                         (instance.IsListRestartSupported() || !MtIsListTopology(prim_type)) &&
                         (instance.IsPatchListRestartSupported() || !MtIsPatchListTopology(prim_type));
    dyn_state.SetPrimitiveRestartEnabled(restart);
    dyn_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dyn_state.SetCullMode(GcnToVk::IsPrimitiveCulled(prim_type)
                              ? GcnToVk::CullMode(regs.polygon_control.CullingMode())
                              : vk::CullModeFlagBits::eNone);
    dyn_state.SetFrontFace(GcnToVk::FrontFace(regs.polygon_control.front_face));
    dyn_state.SetLineWidth(regs.line_control.Width());
    UpdateWorkerColorBlendingState(dyn_state, regs, pipeline);
}

enum class MtPktKind : u8 {
    Invalid = 0,
    NoOp,
    StateOnly,
    InitHwState,
    Draw,
    CopyBatch,
    FillBuf,
    DmaMemWrite,
    TargetRawOp,
    SyncRetain,
    PostSyncRetain,
};

enum class MtBailReason : u8 {
    Pkt = 0,
    Pso,
    Target,
    Buf,
    Other,
};

enum class MtOtherKind : u8 {
    Empty = 0,
    Skip = 1,
    Misc = 2,
};

std::atomic<u64> g_mt_lists_seen{0};
std::atomic<u64> g_mt_lists_compiled{0};
std::atomic<u64> g_mt_draws_recorded{0};
std::atomic<u64> g_mt_bail_pkt{0};
std::atomic<u64> g_mt_bail_pso{0};
std::atomic<u64> g_mt_bail_pso_sig{0};
std::atomic<u64> g_mt_bail_pso_spec{0};
std::atomic<u64> g_mt_bail_pso_unsafe{0};
std::atomic<u64> g_mt_bail_target{0};
std::atomic<u64> g_mt_bail_buf{0};
std::atomic<u64> g_mt_bail_other{0};
std::atomic<u64> g_mt_bail_other_empty{0};
std::atomic<u64> g_mt_bail_other_skip{0};
std::atomic<u64> g_mt_bail_other_misc{0};
std::atomic<u32> g_mt_last_report_frame{0};
std::array<std::atomic<u32>, 16> g_mt_seen_bail_ops{};
std::array<std::atomic<u64>, 8> g_mt_seen_unsafe_pso{};
std::array<std::atomic<u64>, 8> g_mt_seen_bail_buf{};
std::array<std::array<std::atomic<u64>, 4>, 6> g_mt_seen_bail_target_by_site{};
std::atomic<u32> g_mt_window_bail_target_count{0};
std::atomic<u64> g_mt_prev_window_compiled{0};

[[nodiscard]] bool MtTryClaimDiagKey(std::span<std::atomic<u64>> table, u64 raw_key) noexcept {
    const u64 key = raw_key | (1ULL << 63u);
    for (auto& slot : table) {
        u64 cur_val = slot.load(std::memory_order_relaxed);
        if (cur_val == key) {
            return false;
        }
        if (cur_val == 0ULL) {
            if (slot.compare_exchange_strong(cur_val, key, std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
                return true;
            }
            if (cur_val == key) {
                return false;
            }
        }
    }
    return false;
}

[[nodiscard]] const char* MtUnsafePsoReasonName(u8 r) noexcept {
    switch (r) {
    case 1u:
        return "no_push_desc";
    case 2u:
        return "non_vs_ps_stage";
    case 3u:
        return "uses_dma";
    case 4u:
        return "paged_buf";
    case 5u:
        return "storage_img";
    case 6u:
        return "has_fmask";
    case 7u:
        return "flat_ud_gt_512";
    case 8u:
        return "written_buf";
    case 9u:
        return "special_buf";
    case 10u:
        return "written_img";
    case 11u:
        return "mip_fallback";
    case 12u:
        return "null_vs_stage";
    default:
        return "unknown";
    }
}

[[nodiscard]] bool MtStatsEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_MT_STATS");
        return env && env[0] == '1';
    }();
    return enabled;
}

void MtMaybeLogUnsafePsoSample(u8 reason, u64 vs_hash, u64 ps_hash) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    const u64 key = (static_cast<u64>(reason) << 56u) |
                    ((vs_hash ^ (ps_hash * 0x9E3779B97F4A7C15ULL)) & 0x00FFFFFFFFFFFFFFULL);
    if (!MtTryClaimDiagKey(g_mt_seen_unsafe_pso, key)) {
        return;
    }
    std::printf("[BB-MT] unsafe_pso sample: reason=%u(%s) vs=0x%016llx ps=0x%016llx\n",
                static_cast<unsigned>(reason), MtUnsafePsoReasonName(reason),
                static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash));
    std::fflush(stdout);
}

[[nodiscard]] const char* MtBailBufSiteName(u8 site) noexcept {
    switch (site) {
    case 1u:
        return "vb";
    case 2u:
        return "ib";
    case 3u:
        return "flatbuf";
    case 4u:
        return "clip_planes";
    case 5u:
        return "special";
    case 6u:
        return "sbuf";
    case 7u:
        return "indirect";
    default:
        return "other";
    }
}

void MtMaybeLogBailBufSample(u8 site, u8 step, bool is_fmt, VAddr addr, u64 sz,
                             u64 vs_hash) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    const u64 key = (static_cast<u64>(site) << 56u) | (static_cast<u64>(step) << 48u) |
                    (static_cast<u64>(is_fmt ? 1u : 0u) << 47u) | ((addr >> 4u) & 0x7FFFFFFFFFFFULL);
    if (!MtTryClaimDiagKey(g_mt_seen_bail_buf, key)) {
        return;
    }
    std::printf("[BB-MT] bail_buf sample: site=%u(%s) step=%u fmt=%u addr=0x%llx sz=0x%llx "
                "vs=0x%016llx\n",
                static_cast<unsigned>(site), MtBailBufSiteName(site), static_cast<unsigned>(step),
                is_fmt ? 1u : 0u, static_cast<unsigned long long>(addr),
                static_cast<unsigned long long>(sz), static_cast<unsigned long long>(vs_hash));
    std::fflush(stdout);
}

[[nodiscard]] const char* MtBailTargetSiteName(u8 site) noexcept {
    switch (site) {
    case 1u:
        return "max_subpasses";
    case 2u:
        return "inspect_targets";
    case 3u:
        return "resolve_srv_null";
    case 4u:
        return "written_no_id";
    case 5u:
        return "sampled_rt_cap";
    default:
        return "other";
    }
}

void MtMaybeLogBailTargetSample(u8 site, VAddr addr, u32 fmt_or_mrt, u64 vs_hash, u64 ps_hash,
                                u8 substep = 0u, u32 flags = 0u, u32 w = 0u, u32 h = 0u,
                                u32 guest_size = 0u) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    if (g_mt_window_bail_target_count.load(std::memory_order_relaxed) >= 4u) {
        return;
    }
    const size_t bucket = std::min<size_t>(site, g_mt_seen_bail_target_by_site.size() - 1u);
    const u64 key = (static_cast<u64>(site) << 56u) | (static_cast<u64>(substep) << 48u) |
                    (static_cast<u64>(fmt_or_mrt & 0xFFu) << 40u) |
                    (((addr >> 8u) ^ vs_hash ^ ps_hash) & 0xFFFFFFFFFFULL);
    if (!MtTryClaimDiagKey(g_mt_seen_bail_target_by_site[bucket], key)) {
        return;
    }
    if (g_mt_window_bail_target_count.fetch_add(1u, std::memory_order_relaxed) >= 4u) {
        return;
    }
    std::printf("[BB-MT] bail_target sample: site=%u(%s) sub=%u addr=0x%llx info=0x%x flags=0x%x "
                "%ux%u gsz=0x%x vs=0x%016llx ps=0x%016llx\n",
                static_cast<unsigned>(site), MtBailTargetSiteName(site),
                static_cast<unsigned>(substep), static_cast<unsigned long long>(addr), fmt_or_mrt,
                flags, w, h, guest_size, static_cast<unsigned long long>(vs_hash),
                static_cast<unsigned long long>(ps_hash));
    std::fflush(stdout);
}

void MtMaybeLogBadPktSample(const u32* cur, u32 rem) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    if (!cur || rem == 0u) {
        return;
    }
    const u32 hdr = cur[0];
    const u32 op = (hdr >> 8u) & 0xFFu;
    const u32 b0 = rem > 1u ? cur[1] : 0u;
    const u32 b1 = rem > 2u ? cur[2] : 0u;
    const u32 vk_op =
        (op == u32(BbCp::Op::Nop) && rem > 2u && b0 == BbCp::NopPayload::DirectVulkan)
            ? (b1 & 0xFFFFu)
            : 0xFFFFu;
    const u32 key = 0x80000000u | (op << 16u) | vk_op;
    for (auto& slot : g_mt_seen_bail_ops) {
        u32 cur_val = slot.load(std::memory_order_relaxed);
        if (cur_val == key) {
            return;
        }
        if (cur_val == 0u) {
            if (slot.compare_exchange_strong(cur_val, key, std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
                std::printf("[BB-MT] bail_pkt sample: hdr=0x%08x op=0x%02x vk_op=0x%04x "
                            "b0=0x%08x b1=0x%08x\n",
                            hdr, op, vk_op, b0, b1);
                std::fflush(stdout);
                return;
            }
            if (cur_val == key) {
                return;
            }
        }
    }
}

void MtReportTelemetryIfDue(u32 cur_guest_frame) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    u32 last = g_mt_last_report_frame.load(std::memory_order_relaxed);
    if (last == 0u) {
        if (g_mt_last_report_frame.compare_exchange_strong(last, cur_guest_frame,
                                                           std::memory_order_relaxed)) {
            return;
        }
    }
    if (static_cast<s32>(cur_guest_frame - last) < 600) {
        return;
    }
    if (!g_mt_last_report_frame.compare_exchange_strong(last, cur_guest_frame,
                                                        std::memory_order_acq_rel)) {
        return;
    }
    const u64 seen = g_mt_lists_seen.exchange(0ULL, std::memory_order_relaxed);
    const u64 comp = g_mt_lists_compiled.exchange(0ULL, std::memory_order_relaxed);
    (void)g_mt_prev_window_compiled.exchange(comp, std::memory_order_relaxed);
    for (auto& bucket : g_mt_seen_bail_target_by_site) {
        for (auto& slot : bucket) {
            slot.store(0ULL, std::memory_order_relaxed);
        }
    }
    g_mt_window_bail_target_count.store(0u, std::memory_order_relaxed);
    const u64 draws = g_mt_draws_recorded.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_pkt = g_mt_bail_pkt.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_pso = g_mt_bail_pso.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_pso_sig = g_mt_bail_pso_sig.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_pso_spec = g_mt_bail_pso_spec.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_pso_uns = g_mt_bail_pso_unsafe.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_tgt = g_mt_bail_target.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_buf = g_mt_bail_buf.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_oth = g_mt_bail_other.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_oth_emp = g_mt_bail_other_empty.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_oth_skp = g_mt_bail_other_skip.exchange(0ULL, std::memory_order_relaxed);
    const u64 b_oth_msc = g_mt_bail_other_misc.exchange(0ULL, std::memory_order_relaxed);
    std::printf("[BB-MT] 600f: compiled %llu/%llu lists (mt_draws=%llu, bail: pkt=%llu, "
                "pso=%llu(sig=%llu,spec=%llu,unsafe=%llu), target=%llu, buf=%llu, "
                "other=%llu(empty=%llu,skip=%llu,misc=%llu))\n",
                static_cast<unsigned long long>(comp), static_cast<unsigned long long>(seen),
                static_cast<unsigned long long>(draws), static_cast<unsigned long long>(b_pkt),
                static_cast<unsigned long long>(b_pso), static_cast<unsigned long long>(b_pso_sig),
                static_cast<unsigned long long>(b_pso_spec),
                static_cast<unsigned long long>(b_pso_uns), static_cast<unsigned long long>(b_tgt),
                static_cast<unsigned long long>(b_buf), static_cast<unsigned long long>(b_oth),
                static_cast<unsigned long long>(b_oth_emp),
                static_cast<unsigned long long>(b_oth_skp),
                static_cast<unsigned long long>(b_oth_msc));
    std::fflush(stdout);
}

void MtNoteBail(MtBailReason reason, u32 cur_guest_frame,
                MtOtherKind other_kind = MtOtherKind::Misc) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    switch (reason) {
    case MtBailReason::Pkt:
        g_mt_bail_pkt.fetch_add(1ULL, std::memory_order_relaxed);
        break;
    case MtBailReason::Pso:
        g_mt_bail_pso.fetch_add(1ULL, std::memory_order_relaxed);
        break;
    case MtBailReason::Target:
        g_mt_bail_target.fetch_add(1ULL, std::memory_order_relaxed);
        break;
    case MtBailReason::Buf:
        g_mt_bail_buf.fetch_add(1ULL, std::memory_order_relaxed);
        break;
    case MtBailReason::Other:
        g_mt_bail_other.fetch_add(1ULL, std::memory_order_relaxed);
        if (other_kind == MtOtherKind::Empty) {
            g_mt_bail_other_empty.fetch_add(1ULL, std::memory_order_relaxed);
        } else if (other_kind == MtOtherKind::Skip) {
            g_mt_bail_other_skip.fetch_add(1ULL, std::memory_order_relaxed);
        } else {
            g_mt_bail_other_misc.fetch_add(1ULL, std::memory_order_relaxed);
        }
        break;
    }
    MtReportTelemetryIfDue(cur_guest_frame);
}

void MtNotePsoBail(PipelineCache::MtPsoFailKind fail_kind, bool unsafe_pso,
                   u32 cur_guest_frame) noexcept {
    if (!MtStatsEnabled()) [[likely]] {
        return;
    }
    g_mt_bail_pso.fetch_add(1ULL, std::memory_order_relaxed);
    if (unsafe_pso) {
        g_mt_bail_pso_unsafe.fetch_add(1ULL, std::memory_order_relaxed);
    } else if (fail_kind == PipelineCache::MtPsoFailKind::SpecMiss) {
        g_mt_bail_pso_spec.fetch_add(1ULL, std::memory_order_relaxed);
    } else {
        g_mt_bail_pso_sig.fetch_add(1ULL, std::memory_order_relaxed);
    }
    MtReportTelemetryIfDue(cur_guest_frame);
}

[[nodiscard]] constexpr bool IsMtVsmDim(u32 dim) noexcept {
    return dim == 32u || dim == 64u || dim == 128u || dim == 256u || dim == 512u;
}

[[nodiscard]] bool IsMtVsmImageSharp(const u64 raw_words[4]) noexcept {
    Gcn::Image img{};
    std::memcpy(&img, raw_words, sizeof(img));
    reinterpret_cast<u32*>(&img)[7] = 0u;
    if (!img.Valid() || img.IsMsaa() || img.base_level != 0u ||
        img.GetDataFmt() != Gcn::DataFormat::Format16_16 ||
        img.GetNumberFmt() != Gcn::NumberFormat::Float) {
        return false;
    }
    const u32 w = static_cast<u32>(img.width) + 1u;
    const u32 h = static_cast<u32>(img.height) + 1u;
    return w == h && IsMtVsmDim(w);
}

[[nodiscard]] bool IsVsmTargetRawPacket(const u32* cur, u32 pkt_words) noexcept {
    if (!VsmShadowMemo::Enabled() || !cur || pkt_words < 3u ||
        cur[1] != BbCp::NopPayload::DirectVulkan) {
        return false;
    }
    const auto vk_op = BbCp::DirectVk::Op(cur[2] & 0xFFFFu);
    switch (vk_op) {
    case BbCp::DirectVk::Op::ClearColorImage: {
        if (pkt_words != 19u) {
            return false;
        }
        BbCp::DirectVk::ClearColorArgs args{};
        std::memcpy(&args, cur + 3u, sizeof(args));
        return args.num_bits == 32u && IsMtVsmImageSharp(args.tsharp);
    }
    case BbCp::DirectVk::Op::ClearDepthStencil: {
        if (pkt_words != 21u) {
            return false;
        }
        BbCp::DirectVk::ClearDepthStencilArgs args{};
        std::memcpy(&args, cur + 3u, sizeof(args));
        const Gcn::CbDbExtent ext{.raw = args.cb_db_extent};
        return args.slice_count == 1u && ext.Valid() && ext.width == ext.height &&
               IsMtVsmDim(ext.width);
    }
    case BbCp::DirectVk::Op::CopyTextureRegion: {
        if (pkt_words != 35u) {
            return false;
        }
        BbCp::DirectVk::CopyTextureRegionArgs args{};
        std::memcpy(&args, cur + 3u, sizeof(args));
        return args.copy_d == 1u && IsMtVsmImageSharp(args.srv_tsharp) &&
               IsMtVsmImageSharp(args.uav_tsharp);
    }
    case BbCp::DirectVk::Op::UploadTextureSubresource: {
        if (pkt_words != 27u) {
            return false;
        }
        BbCp::DirectVk::UploadTextureArgs args{};
        std::memcpy(&args, cur + 3u, sizeof(args));
        return IsMtVsmImageSharp(args.real_tsharp);
    }
    default:
        return false;
    }
}

[[nodiscard]] inline bool IsStageACoalescerShader(u64 vs_hash, u64 ps_hash) noexcept {
    // Skip local point-light (`0x8747a367`) command lists on the very 1st draw when
    // `LightTileBatcher` is enabled (default `1`), avoiding pre-scanning 128 subpasses of
    // per-tile `CopyBatch` packets before bailing to `LightTileBatcher` anyway.
    // (`BB_LIGHT_TILE_BATCH=0` disables Stage-A batching and allows MT compilation).
    if (vs_hash == 0x000000008747a367ULL && LightTileBatcher::Enabled()) {
        return true;
    }
    // Keep `VsmShadowMemo` Sample13Weight blur (`0x0b0acf50` + `0x0f61fd27`) on Stage A where
    // shadow-face elision eliminates whole GPU passes.
    if (vs_hash == 0x000000000b0acf50ULL && ps_hash == 0x000000000f61fd27ULL) { // VSM blur
        return true;
    }
    return false;
}

struct MtDecodedPkt {
    MtPktKind kind{MtPktKind::Invalid};
    u32 pkt_words{0};
    bool is_indexed{false};
    bool is_indirect{false};
    u32 start_index{0};
    VAddr indirect_args_va{0};
};

[[nodiscard]] MtDecodedPkt DecodeAndApplyMtPacket(
    const u32* cur, u32 avail_words, Gcn::Regs& regs,
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS>& cb_extent, Gcn::CbDbExtent& db_extent,
    std::array<u16, Gcn::NUM_COLOR_BUFFERS>& cb_uid, u16& db_uid, u64& pso_epoch, u64& dyn_epoch,
    Core::MemoryManager* mem, bool host_markers_enabled, Gcn::RegDirty* dirty = nullptr,
    u8* cb_written_mask = nullptr, bool* db_written = nullptr) noexcept {
    MtDecodedPkt out{};
    if (avail_words == 0u) {
        return out;
    }
    const u32 header = cur[0];
    const u32 pkt_type = header >> 30u;
    if (pkt_type == 2u) {
        out.kind = MtPktKind::NoOp;
        out.pkt_words = 1u;
        return out;
    }
    if (pkt_type != 3u || (header & 1u) != 0u) {
        return out; // Reject Type-0/1 or predicated packets
    }

    const u32 opcode = (header >> 8u) & 0xFFu;
    const u32 body_words = ((header >> 16u) & 0x3FFFu) + 1u;
    const u32 pkt_words = 1u + body_words;
    if (pkt_words > avail_words) {
        return out;
    }
    out.pkt_words = pkt_words;
    const u32* const body = cur + 1u;
    u64 dummy_csum = 0ULL;

    if (opcode == u32(BbCp::Op::Nop)) {
        if (body_words == 1u) {
            out.kind = MtPktKind::NoOp;
            return out;
        }
        const u32 magic = body[0];
        if (magic != BbCp::NopPayload::DirectVulkan) {
            if (magic == BbCp::NopPayload::PatchedFlip || host_markers_enabled) {
                return out;
            }
            out.kind = MtPktKind::NoOp;
            return out;
        }
        if (body_words < 2u) {
            return out;
        }
        const u32 op_word = body[1];
        const auto vk_op = BbCp::DirectVk::Op(op_word & 0xFFFFu);
        switch (vk_op) {
        case BbCp::DirectVk::Op::SetIndexState:
        case BbCp::DirectVk::Op::SetContextRegsPacked:
        case BbCp::DirectVk::Op::BindVsShader:
        case BbCp::DirectVk::Op::BindPsShader:
        case BbCp::DirectVk::Op::BindEsShader:
        case BbCp::DirectVk::Op::BindGsShader:
        case BbCp::DirectVk::Op::BindHsShader:
        case BbCp::DirectVk::Op::BindLsShader:
        case BbCp::DirectVk::Op::SetVgtControl:
        case BbCp::DirectVk::Op::SetDirectStateIds:
            if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
                return {};
            }
            out.kind = MtPktKind::StateOnly;
            return out;

        case BbCp::DirectVk::Op::SetStageUserData: {
            if (body_words < 3u) {
                return {};
            }
            const u32 sh_offset = body[2] & 0xFFFFu;
            if (sh_offset >= 0x200u) {
                const u32 mask = body[2] >> 16u;
                if (sh_offset <= 0x250u &&
                    static_cast<u32>(std::popcount(mask)) == body_words - 3u) {
                    out.kind = MtPktKind::PostSyncRetain;
                    return out;
                }
                return {};
            }
            if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
                return {};
            }
            out.kind = MtPktKind::StateOnly;
            return out;
        }

        case BbCp::DirectVk::Op::BindCsShader:
            if (body_words != 9u) {
                return {};
            }
            out.kind = MtPktKind::PostSyncRetain;
            return out;

        case BbCp::DirectVk::Op::InitDefaultHardwareState:
            if (body_words != 2u || ((op_word >> 16u) & 0xFFu) > 5u) {
                return {};
            }
            if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                    regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
                return {};
            }
            cb_extent.fill(Gcn::CbDbExtent{});
            db_extent.raw = 0u;
            cb_uid.fill(0u);
            db_uid = 0u;
            if (cb_written_mask) {
                *cb_written_mask = 0xFFu;
            }
            if (db_written) {
                *db_written = true;
            }
            out.kind = MtPktKind::InitHwState;
            return out;

        case BbCp::DirectVk::Op::SetRenderTarget: {
            const u32 slot = (op_word >> 16u) & 7u;
            const bool valid = ((op_word >> 19u) & 1u) != 0u;
            if ((valid && body_words != 14u) || (!valid && body_words != 2u)) {
                return {};
            }
            (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
            if (valid) {
                cb_extent[slot].raw = body[13];
                cb_uid[slot] = static_cast<u16>((op_word >> 20u) & 0xFFFu);
            } else {
                cb_extent[slot].raw = 0u;
                cb_uid[slot] = 0u;
            }
            if (cb_written_mask) {
                *cb_written_mask |= static_cast<u8>(1u << slot);
            }
            out.kind = MtPktKind::StateOnly;
            return out;
        }

        case BbCp::DirectVk::Op::SetDepthRenderTarget: {
            const bool valid = ((op_word >> 16u) & 1u) != 0u;
            if ((valid && body_words != 15u) || (!valid && body_words != 2u)) {
                return {};
            }
            (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
            if (valid) {
                db_extent.raw = body[14];
                db_uid = static_cast<u16>((op_word >> 20u) & 0xFFFu);
            } else {
                db_extent.raw = 0u;
                db_uid = 0u;
            }
            if (db_written) {
                *db_written = true;
            }
            out.kind = MtPktKind::StateOnly;
            return out;
        }

        case BbCp::DirectVk::Op::DrawIndexed:
            if (body_words != 6u) {
                return {};
            }
            (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
            out.kind = MtPktKind::Draw;
            out.is_indexed = true;
            out.start_index = body[4];
            return out;

        case BbCp::DirectVk::Op::DrawIndexAddr:
            if (body_words != 6u) {
                return {};
            }
            (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
            out.kind = MtPktKind::Draw;
            out.is_indexed = true;
            out.start_index = 0u;
            return out;

        case BbCp::DirectVk::Op::DrawAuto:
            if (body_words != 5u) {
                return {};
            }
            (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
            out.kind = MtPktKind::Draw;
            out.is_indexed = false;
            out.start_index = 0u;
            return out;

        case BbCp::DirectVk::Op::DrawBundle: {
            const u32 draw_kind = (op_word >> 16u) & 0xFu;
            if (body_words < 7u || draw_kind > 2u ||
                !BbCp::CommandProcessor::ApplyDrawBundleRegs(
                    regs, dirty, dirty && dirty->track_all_blocks, op_word, body, body_words,
                    &pso_epoch, &dyn_epoch)) {
                return {};
            }
            out.kind = MtPktKind::Draw;
            out.is_indexed = (draw_kind != 1u);
            out.start_index = (draw_kind == 0u) ? body[body_words - 2u] : 0u;
            return out;
        }

        case BbCp::DirectVk::Op::DrawIndirect: {
            if (body_words != 6u) {
                return {};
            }
            BbCp::DirectVk::DrawIndirectArgs args{};
            std::memcpy(&args, body + 2u, sizeof(args));
            const VAddr base_va = static_cast<VAddr>((args.args_va & 0xFFFFFFFFFFULL) & ~7ULL);
            const VAddr eff_va = base_va + args.byte_offset;
            if (base_va == 0ULL || (eff_va & 3ULL) != 0ULL) {
                return {};
            }
            out.kind = MtPktKind::Draw;
            out.is_indirect = true;
            out.is_indexed = (args.is_indexed != 0u);
            out.start_index = 0u;
            out.indirect_args_va = eff_va;
            return out;
        }

        case BbCp::DirectVk::Op::CopyBufferBatch: {
            const u32 count = op_word >> 16u;
            if (count == 0u || count > 32u || body_words != 2u + count * 6u) {
                return {};
            }
            (void)mem;
            for (u32 i = 0; i < count; ++i) {
                BbCp::DirectVk::CopyRegion reg{};
                std::memcpy(&reg, body + 2u + i * 6u, sizeof(reg));
                if (!reg.dst_va || reg.byte_size == 0u) {
                    return {};
                }
            }
            out.kind = MtPktKind::CopyBatch;
            return out;
        }

        case BbCp::DirectVk::Op::FillBuffer: {
            if (body_words != 6u) {
                return {};
            }
            BbCp::DirectVk::FillArgs args{};
            std::memcpy(&args, body + 2u, sizeof(args));
            if (!args.dst_va || args.byte_size == 0u ||
                ((args.dst_va | args.byte_size) & 3ULL) != 0ULL) {
                return {};
            }
            out.kind = MtPktKind::FillBuf;
            return out;
        }

        case BbCp::DirectVk::Op::ClearColorImage:
        case BbCp::DirectVk::Op::EliminateFastClear:
            if (body_words != 18u) {
                return {};
            }
            out.kind = MtPktKind::TargetRawOp;
            return out;

        case BbCp::DirectVk::Op::CopyTextureRegion:
            if (body_words != 34u) {
                return {};
            }
            out.kind = MtPktKind::TargetRawOp;
            return out;

        case BbCp::DirectVk::Op::UploadTextureSubresource:
            if (body_words != 26u) {
                return {};
            }
            out.kind = MtPktKind::TargetRawOp;
            return out;

        case BbCp::DirectVk::Op::DecompressDepth:
        case BbCp::DirectVk::Op::ClearDepthStencil:
            if (body_words != 20u) {
                return {};
            }
            out.kind = MtPktKind::TargetRawOp;
            return out;

        case BbCp::DirectVk::Op::TimestampQuery:
            if (body_words != 4u || (op_word >> 16u) > 2u) {
                return {};
            }
            out.kind = MtPktKind::SyncRetain;
            return out;

        case BbCp::DirectVk::Op::ReleaseLabelsBatch: {
            const u32 count = op_word >> 16u;
            if (count == 0u || count > 32u || body_words != 2u + count * 2u) {
                return {};
            }
            out.kind = MtPktKind::PostSyncRetain;
            return out;
        }

        case BbCp::DirectVk::Op::ReleaseU32LabelsBatch: {
            const u32 count = op_word >> 16u;
            if (count == 0u || count > 32u || body_words != 3u + count * 2u) {
                return {};
            }
            out.kind = MtPktKind::PostSyncRetain;
            return out;
        }

        default:
            return {};
        }
    }

    switch (BbCp::Op(opcode)) {
    case BbCp::Op::ContextControl:
    case BbCp::Op::AcquireMem:
    case BbCp::Op::PfpSyncMe:
    case BbCp::Op::SurfaceSync:
    case BbCp::Op::PreambleCntl:
    case BbCp::Op::IncrementCeCounter:
    case BbCp::Op::IncrementDeCounter:
    case BbCp::Op::WaitOnCeCounter:
    case BbCp::Op::WaitOnDeCounterDiff:
        out.kind = MtPktKind::NoOp;
        return out;

    case BbCp::Op::SetPredication:
        if (body_words >= 2u &&
            BbCp::SetPredication::From(body).Operation() == BbCp::SetPredication::Clear) {
            out.kind = MtPktKind::NoOp;
            return out;
        }
        return {};

    case BbCp::Op::WaitRegMem:
        if (body_words >= 6u && !BbCp::WaitRegMem::From(body).Memory()) {
            out.kind = MtPktKind::NoOp;
            return out;
        }
        return {};

    case BbCp::Op::EventWrite: {
        const auto ev = BbCp::EventWrite::From(body, body_words);
        if (ev.Type() == BbCp::EventWrite::SoVgtStreamoutFlush) {
            return {};
        }
        if (ev.Index() == BbCp::EventWrite::IndexZpassDone &&
            (ev.Type() == BbCp::EventWrite::ZpassDone ||
             ev.Type() == BbCp::EventWrite::PixelPipeStatDump)) {
            return {};
        }
        out.kind = MtPktKind::NoOp;
        return out;
    }

    case BbCp::Op::EventWriteEop:
        if (body_words == 5u) {
            out.kind = MtPktKind::PostSyncRetain;
            return out;
        }
        return {};

    case BbCp::Op::EventWriteEos:
        if (body_words == 4u &&
            BbCp::EventWriteEos::From(body).Command() == BbCp::EventWriteEos::SignalFence) {
            out.kind = MtPktKind::PostSyncRetain;
            return out;
        }
        return {};

    case BbCp::Op::ClearState:
        (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
            regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
        cb_extent.fill(Gcn::CbDbExtent{});
        db_extent.raw = 0u;
        cb_uid.fill(0u);
        db_uid = 0u;
        if (cb_written_mask) {
            *cb_written_mask = 0xFFu;
        }
        if (db_written) {
            *db_written = true;
        }
        out.kind = MtPktKind::InitHwState;
        return out;

    case BbCp::Op::SetContextReg: {
        if (body_words < 2u) {
            return {};
        }
        const u32 ctx_off = body[0] & 0xFFFFu;
        const u32 count_field = body_words - 1u;
        if (0xA000u + ctx_off + count_field > Gcn::Regs::NumRegs) {
            return {};
        }
        (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
            regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
        constexpr u32 kDbZInfo = 0xA010u;
        constexpr u32 kCbColor0Base = 0xA318u;
        constexpr u32 kCbColor0Cmask = 0xA31Fu;
        constexpr u32 kCbColorStride = 0xFu;
        constexpr u32 kNopHeader = 0xC0001000u;
        const u32 reg = 0xA000u + ctx_off;
        const u32* const values = body + 1u;
        if (reg >= kCbColor0Base && reg < kCbColor0Base + 8u * kCbColorStride &&
            (reg - kCbColor0Base) % kCbColorStride == 0u) {
            const u32 target = (reg - kCbColor0Base) / kCbColorStride;
            cb_uid[target] = 0u;
            if (cb_written_mask) {
                *cb_written_mask |= static_cast<u8>(1u << target);
            }
            if ((count_field == 0x0Eu || count_field == 0x0Du || count_field == 0x0Bu) &&
                avail_words >= pkt_words + 2u && values[count_field] == kNopHeader) {
                cb_extent[target].raw = values[count_field + 1u];
            } else {
                cb_extent[target].raw = 0u;
            }
        } else if (reg >= kCbColor0Cmask && reg < kCbColor0Cmask + 8u * kCbColorStride &&
                   (reg - kCbColor0Cmask) % kCbColorStride == 0u) {
            const u32 target = (reg - kCbColor0Cmask) / kCbColorStride;
            cb_uid[target] = 0u;
            if (cb_written_mask) {
                *cb_written_mask |= static_cast<u8>(1u << target);
            }
            if (count_field == 0x04u && avail_words >= pkt_words + 2u &&
                values[count_field] == kNopHeader) {
                cb_extent[target].raw = values[count_field + 1u];
            }
        } else if (reg == kDbZInfo) {
            db_uid = 0u;
            if (db_written) {
                *db_written = true;
            }
            if (count_field == 8u && avail_words >= 24u && values[20] == kNopHeader) {
                db_extent.raw = values[21];
            } else {
                db_extent.raw = 0u;
            }
        }
        out.kind = MtPktKind::StateOnly;
        return out;
    }

    case BbCp::Op::SetShReg:
        if (body_words < 2u || (body[0] & 0xFFFFu) >= 0x200u ||
            0x2C00u + (body[0] & 0xFFFFu) + (body_words - 1u) > Gcn::Regs::NumRegs) {
            return {}; // Reject compute or out-of-range SH registers
        }
        if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
            return {};
        }
        out.kind = MtPktKind::StateOnly;
        return out;

    case BbCp::Op::SetConfigReg:
        if (body_words < 2u ||
            0x2000u + (body[0] & 0xFFFFu) + (body_words - 1u) > Gcn::Regs::NumRegs) {
            return {};
        }
        if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
            return {};
        }
        out.kind = MtPktKind::StateOnly;
        return out;

    case BbCp::Op::SetUconfigReg:
        if (body_words < 2u ||
            0xC000u + (body[0] & 0xFFFFu) + (body_words - 1u) > Gcn::Regs::NumRegs) {
            return {};
        }
        if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
            return {};
        }
        out.kind = MtPktKind::StateOnly;
        return out;

    case BbCp::Op::IndexBase:
        if (body_words < 2u) {
            return {};
        }
        [[fallthrough]];
    case BbCp::Op::IndexType:
    case BbCp::Op::NumInstances:
    case BbCp::Op::IndexBufferSize:
        if (!BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
                regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch)) {
            return {};
        }
        out.kind = MtPktKind::StateOnly;
        return out;

    case BbCp::Op::DrawIndex2:
        if (body_words != 5u || (body[4] & 1u) != 0u) {
            return {};
        }
        (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
            regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
        out.kind = MtPktKind::Draw;
        out.is_indexed = true;
        out.start_index = 0u;
        return out;

    case BbCp::Op::DrawIndexOffset2:
        if (body_words != 4u || (body[3] & 1u) != 0u) {
            return {};
        }
        (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
            regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
        out.kind = MtPktKind::Draw;
        out.is_indexed = true;
        out.start_index = body[1];
        return out;

    case BbCp::Op::DrawIndexAuto:
        if (body_words != 2u || (body[1] & 1u) != 0u) {
            return {};
        }
        (void)BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(
            regs, cur, dummy_csum, dirty, false, &pso_epoch, &dyn_epoch);
        out.kind = MtPktKind::Draw;
        out.is_indexed = false;
        out.start_index = 0u;
        return out;

    case BbCp::Op::DmaData: {
        if (body_words != 6u) {
            return {};
        }
        const auto dma = BbCp::DmaData::From(body);
        if (dma.DstLow() == 0x3022Cu) {
            out.kind = MtPktKind::NoOp;
            return out;
        }
        if (dma.DstRegister()) {
            return {};
        }
        if (dma.Dst() == BbCp::DmaData::Gds &&
            (dma.Src() == BbCp::DmaData::Immediate || dma.SrcIsMemory())) {
            out.kind = MtPktKind::SyncRetain;
            return out;
        }
        if (dma.DstIsMemory() &&
            (dma.Src() == BbCp::DmaData::Immediate || dma.Src() == BbCp::DmaData::Gds ||
             dma.SrcIsMemory()) &&
            dma.DstAddress() != 0ULL && dma.Bytes() != 0u) {
            out.kind = MtPktKind::DmaMemWrite;
            return out;
        }
        return {};
    }

    default:
        return {};
    }
}

[[nodiscard]] bool IsAttachmentlessDummyIndirectDrawPrePso(const Gcn::Regs& regs) noexcept {
    // Hoisted pre-PSO check mirroring `Rasterizer::DrawIndirect` (`rasterizer.cpp:3173`): when
    // `ps_program == 0` (so `fs_info == nullptr` and `key.mrt_mask == 0`), depth/stencil are
    // disabled/unbound, and `vs_program` is the attachmentless dummy indirect VS `0x89cda9e4`
    // (`!uses_dma`, no written buffers/images), skip `SelectGraphicsPipelineMt` altogether so
    // unspecialized dummy user-data never triggers `SpecMiss` (`bail_pso: spec`).
    if (regs.stage_enable.raw != u32(Gcn::ShaderStageEnable::VgtStages::Vs) ||
        regs.vgt_strmout_config.raw != 0u) {
        return false;
    }
    if (regs.stage_enable.IsStageEnabled(static_cast<u32>(Shader::HwStage::Fragment)) &&
        regs.ps_program.Address<VAddr>() != 0ULL) {
        return false;
    }
    const bool has_ds =
        (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
    if (has_ds) {
        return false;
    }
    const VAddr vs_base = regs.vs_program.Address<VAddr>();
    if (vs_base == 0ULL) {
        return false;
    }
    DirectShaders::Snapshot direct_snap;
    if (DirectShaders::TryLookup(vs_base, direct_snap, /*is_worker=*/true) &&
        direct_snap.cpu_immutable != 0) {
        return direct_snap.shader_hash == 0x89cda9e4ull;
    }
    const auto* bin = Gcn::DirectBinaryInfo(reinterpret_cast<const u32*>(vs_base));
    return bin && bin->shader_hash == 0x89cda9e4ull;
}

[[nodiscard]] bool IsAttachmentlessDummyIndirectDraw(const Gcn::Regs& regs,
                                                     const GraphicsPipeline* pso,
                                                     const Shader::Info* vs_st,
                                                     const Shader::Info* ps_st) noexcept {
    if (!pso || !vs_st || ps_st != nullptr || vs_st->pgm_hash != 0x89cda9e4ull || vs_st->uses_dma) {
        return false;
    }
    const auto& key = pso->GetGraphicsKey();
    if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
        for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
            if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb) &&
                (key.mrt_mask & (1u << cb)) != 0u) {
                return false;
            }
        }
    }
    const bool has_ds =
        (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
    return !has_ds;
}

[[nodiscard]] bool IsPipelineMtSafe(const GraphicsPipeline* pso,
                                    u8* out_unsafe_reason = nullptr) noexcept {
    auto reject = [&](u8 r) noexcept -> bool {
        if (out_unsafe_reason) {
            *out_unsafe_reason = r;
        }
        return false;
    };
    if (!pso || !pso->UsesPushDescriptors()) {
        return reject(1u);
    }
    if (!pso->GetStages()[u32(Shader::SwStage::Vertex)]) {
        return reject(12u);
    }
    for (const auto* st : pso->GetStages()) {
        if (!st) {
            continue;
        }
        if (st->sw_stage != Shader::SwStage::Vertex && st->sw_stage != Shader::SwStage::Fragment) {
            return reject(2u);
        }
        if (st->uses_dma) {
            return reject(3u);
        }
        if (st->uses_paged_buffers) {
            return reject(4u);
        }
        if (st->has_storage_images) {
            return reject(5u);
        }
        if (!st->fmasks.empty()) {
            return reject(6u);
        }
        if (st->srt_info.flattened_bufsize_dw > 512u) {
            return reject(7u);
        }
        for (const auto& b : st->buffers) {
            if (b.is_written) {
                return reject(8u);
            }
            if (b.IsSpecial() && b.buffer_type != Shader::BufferType::Flatbuf &&
                b.buffer_type != Shader::BufferType::ClipPlanes &&
                b.buffer_type != Shader::BufferType::BdaPagetable &&
                b.buffer_type != Shader::BufferType::FaultBuffer) {
                return reject(9u);
            }
        }
        for (const auto& img : st->images) {
            if (img.is_written) {
                return reject(10u);
            }
            if (img.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                return reject(11u);
            }
        }
    }
    return true;
}

} // namespace

bool VkMtDirectRecorder::TryCompileDeferredCommandList(const void* /*ctx_base*/,
                                                       void* cmd_list_impl) {
    if (!IsEnabled() || !cmd_list_impl) {
        return false;
    }
    if (FrameCapture::Active() || rasterizer.HostMarkersEnabled() ||
        VideoCore::BufferCache::LayerPagedActive()) {
        return false;
    }

    const u32 worker_id = AcquireThreadWorkerId();
    if (worker_id >= kMaxWorkers) {
        return false;
    }
    static const bool allow_main_thread = [] {
        const char* env = std::getenv("BB_MT_MAIN_THREAD");
        return env && env[0] == '1';
    }();
    if (!allow_main_thread &&
        slot_is_main_thread[worker_id].load(std::memory_order_acquire) == 2u) {
        return false;
    }
    if (!EnsureWorkerInitialized(worker_id)) {
        return false;
    }
    WorkerSlot& w = *worker_ptrs[worker_id].load(std::memory_order_relaxed);
    if (!w.scan_regs || !w.replay_regs || !w.compile_scratch || !w.completed_meta ||
        w.is_recording) {
        return false;
    }
    bool expected_busy = false;
    if (!w.compile_busy.compare_exchange_strong(expected_busy, true, std::memory_order_acq_rel)) {
        return false;
    }
    struct BusyGuard {
        std::atomic<bool>& flag;
        ~BusyGuard() {
            flag.store(false, std::memory_order_release);
        }
    } busy_guard{w.compile_busy};

    // Enumerate DCB slices inside PendingUpdateQueueB (`cmd_list_impl`).
    struct DcbSlice {
        u8* slot_raw{nullptr};
        u32* begin_ptr{nullptr};
        u32 orig_words{0};
    };
    const auto* list_bytes = static_cast<const u8*>(cmd_list_impl);
    u8* block = *reinterpret_cast<u8* const*>(list_bytes + 0x08);
    const u32 entries_per_chunk = *reinterpret_cast<const u32*>(list_bytes + 0x18);
    const u32 tail_used = *reinterpret_cast<const u32*>(list_bytes + 0x1C);
    if (!block || entries_per_chunk == 0u || entries_per_chunk > 4096u) {
        return false;
    }

    std::array<DcbSlice, 64> slices{};
    u32 num_slices = 0;
    u32 total_dwords = 0;
    while (block) {
        u8* const next_block =
            *reinterpret_cast<u8* const*>(block + (static_cast<size_t>(entries_per_chunk) << 6u));
        const u32 count = next_block ? entries_per_chunk : tail_used;
        if (count > entries_per_chunk) {
            return false;
        }
        for (u32 i = 0; i < count; ++i) {
            u8* slot = block + (static_cast<size_t>(i) << 6u);
            u32* begin_ptr = *reinterpret_cast<u32* const*>(slot + 0x00);
            u32* end_ptr = *reinterpret_cast<u32* const*>(slot + 0x08);
            u32* cmd_ptr = *reinterpret_cast<u32* const*>(slot + 0x10);
            if (!begin_ptr || !cmd_ptr || cmd_ptr < begin_ptr || (end_ptr && cmd_ptr > end_ptr) ||
                (reinterpret_cast<uintptr_t>(begin_ptr) & 3u) != 0u ||
                (reinterpret_cast<uintptr_t>(cmd_ptr) & 3u) != 0u) {
                return false;
            }
            const auto words = static_cast<u32>(cmd_ptr - begin_ptr);
            if (words == 0u) {
                continue;
            }
            if (num_slices >= slices.size() || total_dwords + words > 524288u) {
                return false;
            }
            slices[num_slices++] = DcbSlice{slot, begin_ptr, words};
            total_dwords += words;
        }
        block = next_block;
    }
    if (num_slices == 0u || total_dwords < 8u) {
        return false;
    }

    const u32 cur_guest_frame = BbStats::guest_frame_number.load(std::memory_order_relaxed);
    if (MtStatsEnabled()) [[unlikely]] {
        g_mt_lists_seen.fetch_add(1ULL, std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // Phase 1: Pure read-only pre-scan (validates eligibility, warm PSOs, and
    // multi-subpass target layouts with ZERO mutation of `cmd_list_impl`).
    // Deferred worker contexts initialize hardware defaults via
    // `EmitInitDefaultHardwareState(gen = 3u)` (`0xA100 = 0xFFFFFFFF`,
    // `0xA202 = 0xCC0010`, `0xA1B1 = 2`, etc.) and reset dirty caches per list,
    // so `InitGnmDefaultRegs(sregs, 3u)` is the exact initial register state.
    // -------------------------------------------------------------------------
    auto* mem = rasterizer.GetMemoryManager();
    auto& scratch = *w.compile_scratch;
    scratch.stream_items.clear();
    scratch.subpass_plans.clear();
    scratch.sealed_subpasses.clear();

    Gcn::Regs& sregs = *w.scan_regs;
    BbCp::CommandProcessor::InitGnmDefaultRegs(sregs, 3u);
    w.pso_state.ResetForPass();

    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> scan_cb_extent{};
    Gcn::CbDbExtent scan_db_extent{};
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> scan_cb_uid{};
    u16 scan_db_uid = 0u;
    u64 scan_pso_epoch = 0ULL;
    u64 scan_dyn_epoch = 0ULL;

    bool seen_draw = false;
    bool force_new_subpass = false;
    u32 draw_pkt_ordinal = 0;
    u32 candidate_draws = 0;
    bool list_any_gbuffer = false;
    bool list_is_core_geom = false;
    static constexpr u32 kMaxSubPassesPerList = 128u;
    static const bool force_indirect_guard = [] {
        const char* env = std::getenv("BB_FORCE_INDIRECT_GUARD");
        return env && env[0] == '1';
    }();

    const auto calc_scene_color_submode = [](bool is_scene_color,
                                             const Gcn::Regs& regs) noexcept -> u8 {
        if (!is_scene_color) {
            return 0u;
        }
        const u32 num_inst = regs.num_instances.NumInstances();
        if (regs.num_indices > 0u && regs.num_indices <= 6u && num_inst <= 1u) {
            return 2u; // Fullscreen composite quad -> NoteComposite on Stage B
        }
        if (regs.blend_control[0].enable && !regs.color_buffers[0].info.blend_bypass &&
            !regs.depth_control.depth_write_enable) {
            return 1u; // Blended 3D transparent draw -> NoteTransparentPass on Stage B
        }
        return 0u;
    };

    for (u32 s = 0; s < num_slices; ++s) {
        const u32* cur = slices[s].begin_ptr;
        u32 rem = slices[s].orig_words;
        while (rem > 0u) {
            const MtDecodedPkt dp = DecodeAndApplyMtPacket(
                cur, rem, sregs, scan_cb_extent, scan_db_extent, scan_cb_uid, scan_db_uid,
                scan_pso_epoch, scan_dyn_epoch, mem, false);
            if (dp.kind == MtPktKind::Invalid || dp.pkt_words == 0u || dp.pkt_words > rem) {
                MtMaybeLogBadPktSample(cur, rem);
                MtNoteBail(MtBailReason::Pkt, cur_guest_frame);
                return false;
            }

            switch (dp.kind) {
            case MtPktKind::NoOp:
            case MtPktKind::StateOnly:
                break;

            case MtPktKind::InitHwState:
                if (seen_draw) {
                    MtNoteBail(MtBailReason::Pkt, cur_guest_frame);
                    return false;
                }
                scratch.stream_items.push_back(
                    WorkerCompileScratch::StreamItem::Raw(cur, dp.pkt_words));
                break;

            case MtPktKind::TargetRawOp:
                if (IsVsmTargetRawPacket(cur, dp.pkt_words)) {
                    MtNoteBail(MtBailReason::Other, cur_guest_frame, MtOtherKind::Skip);
                    return false;
                }
                [[fallthrough]];
            case MtPktKind::CopyBatch:
            case MtPktKind::FillBuf:
            case MtPktKind::DmaMemWrite:
            case MtPktKind::SyncRetain:
            case MtPktKind::PostSyncRetain:
                scratch.stream_items.push_back(
                    WorkerCompileScratch::StreamItem::Raw(cur, dp.pkt_words));
                if (seen_draw) {
                    force_new_subpass = true;
                }
                break;

            case MtPktKind::Draw: {
                const u32 cur_draw_pkt_ord = draw_pkt_ordinal++;
                if (sregs.stage_enable.raw != u32(Gcn::ShaderStageEnable::VgtStages::Vs) ||
                    sregs.vgt_strmout_config.raw != 0u ||
                    sregs.depth_render_control.depth_clear_enable ||
                    sregs.depth_render_control.stencil_clear_enable ||
                    sregs.depth_render_control.depth_copy ||
                    sregs.depth_render_control.stencil_copy ||
                    sregs.depth_render_control.resummarize_enable ||
                    (dp.is_indirect && force_indirect_guard && IndirectGuard::DrawEnabled())) {
                    MtNoteBail(MtBailReason::Pkt, cur_guest_frame);
                    return false;
                }
                if (dp.is_indirect && IsAttachmentlessDummyIndirectDrawPrePso(sregs)) {
                    break;
                }
                PipelineCache::MtPsoFailKind pso_fail = PipelineCache::MtPsoFailKind::None;
                const GraphicsPipeline* pso =
                    rasterizer.GetPipelineCache().SelectGraphicsPipelineMt(
                        sregs, w.pso_state,
                        dp.is_indirect ? DrawIndirectParams{0u, 0u} : DrawIndirectParams{},
                        dp.is_indirect, /*allow_compile_fallback=*/false, &pso_fail);
                if (!pso) {
                    MtNotePsoBail(pso_fail, /*unsafe_pso=*/false, cur_guest_frame);
                    return false;
                }
                const auto* vs_st = pso->GetStages()[u32(Shader::SwStage::Vertex)];
                const auto* ps_st = pso->GetStages()[u32(Shader::SwStage::Fragment)];
                const u64 vs_hash = vs_st ? vs_st->pgm_hash : 0ULL;
                const u64 ps_hash = ps_st ? ps_st->pgm_hash : 0ULL;
                u8 unsafe_reason = 0u;
                if (!IsPipelineMtSafe(pso, &unsafe_reason)) {
                    MtMaybeLogUnsafePsoSample(unsafe_reason, vs_hash, ps_hash);
                    MtNotePsoBail(pso_fail, /*unsafe_pso=*/true, cur_guest_frame);
                    return false;
                }
                if (dp.is_indirect &&
                    IsAttachmentlessDummyIndirectDraw(sregs, pso, vs_st, ps_st)) {
                    break;
                }
                if (IsStageACoalescerShader(vs_hash, ps_hash)) {
                    MtNoteBail(MtBailReason::Other, cur_guest_frame, MtOtherKind::Skip);
                    return false;
                }
                // Keep 2D full-screen post-FX passes without depth/stencil (`GXFullscreenTriangle`,
                // e.g. SSAO `ps=0xcbbca5dc`, LDR tonemap/copy) on Stage A/B.
                if (vs_hash == 0x000000000b0acf50ULL && !sregs.depth_control.depth_enable &&
                    !(sregs.depth_control.stencil_enable && sregs.depth_buffer.StencilValid())) {
                    MtNoteBail(MtBailReason::Other, cur_guest_frame, MtOtherKind::Skip);
                    return false;
                }

                const bool cur_writes_depth = bool(sregs.depth_control.depth_enable &&
                                                   sregs.depth_control.depth_write_enable);
                const bool cur_writes_stencil = sregs.WritesStencil();
                const u32 cur_db_view = std::bit_cast<u32>(sregs.depth_view);
                const u32 cur_htile = sregs.reg_array[0xA005];

                bool need_new_subpass = !seen_draw || force_new_subpass;
                if (seen_draw && !need_new_subpass) {
                    const auto& prev_sp = scratch.subpass_plans.back();
                    const u8 next_sc_submode =
                        calc_scene_color_submode(prev_sp.target_summary.is_scene_color, sregs);
                    const bool sc_transition =
                        (prev_sp.scene_color_submode == 0u && next_sc_submode == 1u) ||
                        (prev_sp.scene_color_submode != 2u && next_sc_submode == 2u) ||
                        (prev_sp.scene_color_submode == 2u && next_sc_submode != 2u);
                    if (pso->GetGraphicsKey().mrt_mask != prev_sp.first_pso->GetGraphicsKey().mrt_mask ||
                        pso->GetGraphicsKey().motion_vectors !=
                            prev_sp.first_pso->GetGraphicsKey().motion_vectors ||
                        std::memcmp(sregs.color_buffers, prev_sp.cb.data(),
                                    sizeof(prev_sp.cb)) != 0 ||
                        std::memcmp(&sregs.depth_buffer, &prev_sp.db, sizeof(prev_sp.db)) != 0 ||
                        cur_db_view != prev_sp.db_view || cur_htile != prev_sp.htile ||
                        cur_writes_depth != prev_sp.writes_depth ||
                        cur_writes_stencil != prev_sp.writes_stencil || sc_transition) {
                        need_new_subpass = true;
                    }
                }

                if (need_new_subpass) {
                    if (scratch.subpass_plans.size() >= kMaxSubPassesPerList) {
                        MtMaybeLogBailTargetSample(1u, 0ULL, kMaxSubPassesPerList, vs_hash,
                                                   ps_hash);
                        MtNoteBail(MtBailReason::Target, cur_guest_frame);
                        return false;
                    }
                    Rasterizer::MtPassTargetSummary ts{};
                    if (!rasterizer.InspectMtPassTargets(sregs, pso, scan_cb_extent, scan_db_extent,
                                                         ts)) {
                        const VAddr diag_taddr =
                            ts.diag_addr != 0
                                ? ts.diag_addr
                                : (sregs.color_buffers[0] ? sregs.color_buffers[0].Address()
                                                          : sregs.depth_buffer.DepthAddress());
                        MtMaybeLogBailTargetSample(2u, diag_taddr, pso->GetGraphicsKey().mrt_mask,
                                                   vs_hash, ps_hash, ts.bail_substep, ts.diag_flags,
                                                   ts.diag_w, ts.diag_h, ts.diag_guest_size);
                        MtNoteBail(MtBailReason::Target, cur_guest_frame);
                        return false;
                    }
                    const u32 guest_mrt =
                        pso->GetGraphicsKey().motion_vectors
                            ? (pso->GetGraphicsKey().mrt_mask & ~(1u << Shader::MotionVectors::Output))
                            : pso->GetGraphicsKey().mrt_mask;
                    // Keep 3D VSM moment shadow faces (`R16G16Sfloat` + Depth, non-ObjectVelocity)
                    // on Stage A so `VsmShadowMemo` can elide unchanged local shadow faces.
                    if (guest_mrt == 0x1u && ts.color_formats[0] == vk::Format::eR16G16Sfloat &&
                        !ObjectVelocityCuller::IsObjectVelocityPs(ps_hash) &&
                        sregs.depth_control.depth_enable && !sregs.blend_control[0].enable) {
                        MtNoteBail(MtBailReason::Other, cur_guest_frame, MtOtherKind::Skip);
                        return false;
                    }

                    if (!seen_draw) {
                        list_is_core_geom =
                            ts.is_gbuffer || guest_mrt == 0x0u ||
                            vs_hash == 0x000000008747a367ULL ||
                            (guest_mrt == 0x1u &&
                             ObjectVelocityCuller::IsObjectVelocityPs(ps_hash));
                    }
                    list_any_gbuffer |= ts.is_gbuffer;

                    WorkerCompileScratch::SubPassPlan plan{};
                    plan.first_draw_ordinal = cur_draw_pkt_ord;
                    plan.first_pso = pso;
                    plan.target_summary = ts;
                    std::memcpy(plan.cb.data(), sregs.color_buffers, sizeof(plan.cb));
                    plan.db = sregs.depth_buffer;
                    plan.db_view = cur_db_view;
                    plan.htile = cur_htile;
                    plan.cb_extent = scan_cb_extent;
                    plan.db_extent = scan_db_extent;
                    plan.cb_uid = scan_cb_uid;
                    plan.db_uid = scan_db_uid;
                    plan.writes_depth = cur_writes_depth;
                    plan.writes_stencil = cur_writes_stencil;
                    plan.scene_color_submode = calc_scene_color_submode(ts.is_scene_color, sregs);
                    plan.CaptureRegs(sregs);
                    const auto sp_idx = static_cast<u32>(scratch.subpass_plans.size());
                    scratch.subpass_plans.push_back(plan);
                    scratch.stream_items.push_back(WorkerCompileScratch::StreamItem::Pass(sp_idx));
                    seen_draw = true;
                    force_new_subpass = false;
                }
                ++candidate_draws;
                break;
            }

            default:
                MtNoteBail(MtBailReason::Pkt, cur_guest_frame);
                return false;
            }

            cur += dp.pkt_words;
            rem -= dp.pkt_words;
        }
    }

    const u32 min_draws = list_is_core_geom ? 1u : 2u;
    if (!seen_draw || scratch.subpass_plans.empty() || candidate_draws < min_draws) {
        MtNoteBail(MtBailReason::Other, cur_guest_frame,
                   seen_draw ? MtOtherKind::Skip : MtOtherKind::Empty);
        return false;
    }

    const u32 planned_subpasses = static_cast<u32>(scratch.subpass_plans.size());
    if (planned_subpasses >= kCompletedMetaSlots) {
        MtNoteBail(MtBailReason::Other, cur_guest_frame);
        return false;
    }

    // Verify packet-aligned placement into `slices[0 .. num_slices - 1]` up front.
    {
        u32 test_slice = 0;
        u32 used_in_slice = 0;
        auto can_place = [&](u32 pkt_words) -> bool {
            while (test_slice < num_slices) {
                if (used_in_slice + pkt_words <= slices[test_slice].orig_words) {
                    used_in_slice += pkt_words;
                    return true;
                }
                ++test_slice;
                used_in_slice = 0;
            }
            return false;
        };
        for (const auto& item : scratch.stream_items) {
            if (!can_place(item.words)) {
                MtNoteBail(MtBailReason::Other, cur_guest_frame);
                return false;
            }
        }
    }

    for (u32 sp_i = 0; sp_i < planned_subpasses; ++sp_i) {
        const u64 peek_seq = w.cmd_seq_gen + 1ULL + sp_i;
        MtCompletedPassMeta& meta_slot = w.completed_meta[peek_seq & (kCompletedMetaSlots - 1u)];
        if (meta_slot.published_seq.load(std::memory_order_acquire) !=
            meta_slot.consumed_seq.load(std::memory_order_acquire)) {
            MtNoteBail(MtBailReason::Other, cur_guest_frame);
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Parallel Secondary VkCommandBuffer Recording on GXWorker.
    // Transactional rollback: `cmd_seq_gen`, `pools[].last_cmd_seq`, and
    // `BbStats` are NOT mutated until ALL subpasses finish in Phase 3.
    // -------------------------------------------------------------------------
    const u64 orig_ring_pos = w.constant_ring ? w.constant_ring->position : 0ULL;
    w.local_stats.Reset();

    u16 occ_block_id = 0xFFFFu;
    if (list_any_gbuffer) {
        occ_block_id = rasterizer.AcquireMtOcclusionQueryBlock(cur_guest_frame);
        w.active_occ_block_id = occ_block_id;
    }

    u32 cur_subpass_idx = ~0u;
    bool cur_inherits_occ = false;
    u8 cur_motion_slot = 0xFFu;
    u8 cur_num_sampled_rts = 0u;
    u8 cur_num_uav_indirect_refs = 0u;
    std::array<MtCompletedPassMeta::SampledRtRef, MtCompletedPassMeta::kMaxSampledRts>
        cur_sampled_rts{};
    std::array<MtCompletedPassMeta::UavIndirectRef, MtCompletedPassMeta::kMaxUavIndirectRefs>
        cur_uav_indirect_refs{};

    auto open_subpass = [&](u32 sp_idx) -> bool {
        const auto& plan = scratch.subpass_plans[sp_idx];
        const auto& ts = plan.target_summary;
        const vk::CommandBuffer cmdbuf = w.AcquireSecondaryBuffer(instance, scheduler);
        const u32 color_cnt =
            std::min<u32>(ts.num_color_attachments, static_cast<u32>(ts.color_formats.size()));
        const vk::CommandBufferInheritanceRenderingInfo rendering_inh{
            .viewMask = 0u,
            .colorAttachmentCount = color_cnt,
            .pColorAttachmentFormats = color_cnt ? ts.color_formats.data() : nullptr,
            .depthAttachmentFormat = ts.depth_format,
            .stencilAttachmentFormat = ts.stencil_format,
            .rasterizationSamples = ts.rasterization_samples,
        };
        const vk::CommandBufferInheritanceInfo inh_info{
            .pNext = &rendering_inh,
            .occlusionQueryEnable = VK_FALSE,
            .queryFlags = {},
        };
        const vk::CommandBufferBeginInfo begin_info{
            .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue |
                     vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
            .pInheritanceInfo = &inh_info,
        };
        if (cmdbuf.begin(&begin_info) != vk::Result::eSuccess) {
            return false;
        }
        w.active_cmdbuf = cmdbuf;
        w.is_recording = true;
        w.recorded_draws = 0u;
        ++w.desc_epoch_counter;
        w.bound_pipeline = vk::Pipeline{};
        w.push_tracker.InvalidateAll();
        w.dyn_state.Invalidate();
        cur_subpass_idx = sp_idx;
        cur_inherits_occ = false;
        cur_motion_slot = 0xFFu;
        cur_num_sampled_rts = 0u;
        cur_num_uav_indirect_refs = 0u;
        return true;
    };

    auto seal_active_subpass = [&]() noexcept {
        if (!w.is_recording || !w.active_cmdbuf || cur_subpass_idx >= planned_subpasses) {
            return;
        }
        (void)w.active_cmdbuf.end();
        const auto& plan = scratch.subpass_plans[cur_subpass_idx];
        WorkerCompileScratch::SealedSubPass sp{};
        sp.sec_cmd = w.active_cmdbuf;
        sp.pool_idx = w.active_pool_idx;
        sp.end_ring_pos = w.constant_ring ? w.constant_ring->position : 0ULL;
        sp.first_pso = plan.first_pso;
        sp.cb_extent = plan.cb_extent;
        sp.db_extent = plan.db_extent;
        sp.cb_uid = plan.cb_uid;
        sp.db_uid = plan.db_uid;
        sp.sh_gfx_regs = plan.sh_gfx_regs;
        sp.context_regs = plan.context_regs;
        sp.uconfig_so_regs = plan.uconfig_so_regs;
        sp.uconfig_vgt_regs = plan.uconfig_vgt_regs;
        sp.recorded_draws = w.recorded_draws;
        sp.is_gbuffer = plan.target_summary.is_gbuffer;
        sp.inherits_occlusion_query = cur_inherits_occ;
        sp.num_sampled_rt_refs = cur_num_sampled_rts;
        sp.num_uav_indirect_refs = cur_num_uav_indirect_refs;
        sp.motion_slot = cur_motion_slot;
        sp.sampled_rt_refs = cur_sampled_rts;
        sp.uav_indirect_refs = cur_uav_indirect_refs;
        scratch.sealed_subpasses.push_back(sp);
        for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
            if (plan.cb[cb]) {
                scratch.NoteTargetWrite(plan.cb[cb].Address(), 4096ULL);
            }
        }
        if ((plan.writes_depth || plan.writes_stencil) && plan.db.DepthAddress() != 0ULL) {
            scratch.NoteTargetWrite(plan.db.DepthAddress(), 4096ULL);
        }

        w.active_cmdbuf = vk::CommandBuffer{};
        w.is_recording = false;
        w.recorded_draws = 0u;
        w.bound_pipeline = vk::Pipeline{};
        w.push_tracker.InvalidateAll();
    };

    auto abort_pass = [&](MtBailReason reason,
                          PipelineCache::MtPsoFailKind pso_fail =
                              PipelineCache::MtPsoFailKind::None,
                          bool pso_unsafe = false) noexcept -> bool {
        if (w.is_recording && w.active_cmdbuf) {
            (void)w.active_cmdbuf.end();
        }
        if (w.constant_ring) {
            w.constant_ring->position = orig_ring_pos;
        }
        if (occ_block_id != 0xFFFFu) {
            rasterizer.AbortMtOcclusionQueryBlock(occ_block_id);
            w.active_occ_block_id = 0xFFFFu;
            occ_block_id = 0xFFFFu;
        }
        w.local_stats.Reset();
        w.active_cmdbuf = vk::CommandBuffer{};
        w.is_recording = false;
        w.recorded_draws = 0u;
        w.bound_pipeline = vk::Pipeline{};
        w.push_tracker.InvalidateAll();
        scratch.sealed_subpasses.clear();
        if (reason == MtBailReason::Pso) {
            MtNotePsoBail(pso_fail, pso_unsafe, cur_guest_frame);
        } else {
            MtNoteBail(reason, cur_guest_frame);
        }
        return false;
    };

    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);
    const u64 storage_align = instance.StorageMinAlignment();
    const bool dynamic_vertex_input = instance.IsVertexInputDynamicState();
    const u8 fx_mask = []() noexcept -> u8 {
        const auto& settings = BbSettings::Get();
        const int sq = std::clamp(settings.shadow_quality.load(std::memory_order_relaxed), 0, 4);
        u8 m = 0;
        if (settings.ssao_quality.load(std::memory_order_relaxed))    m |= 0x01u;
        if (sq >= 2 && settings.contact_shadows.load(std::memory_order_relaxed)) m |= 0x02u;
        if (settings.ssr_quality.load(std::memory_order_relaxed))     m |= 0x04u;
        if (settings.true_black_hdr.load(std::memory_order_relaxed))  m |= 0x08u;
        if (sq >= 2 && settings.pcss_shadows.load(std::memory_order_relaxed))    m |= 0x10u;
        if (settings.ssgi_enabled.load(std::memory_order_relaxed))    m |= 0x20u;
        if (sq == 0)                                                  m |= 0x40u;
        return m;
    }();
    const u8 mb_mask = []() noexcept -> u8 {
        const auto& settings = BbSettings::Get();
        const u8 cam = static_cast<u8>(
            std::clamp(settings.motion_blur_camera.load(std::memory_order_relaxed), 0, 3));
        const u8 obj = static_cast<u8>(
            std::clamp(settings.motion_blur_object.load(std::memory_order_relaxed), 0, 3));
        const u8 native_bit =
            settings.native_postfx.load(std::memory_order_relaxed) ? 0x10u : 0x00u;
        return static_cast<u8>((cam & 0x3u) | ((obj & 0x3u) << 2u) | native_bit);
    }();
    const u8 mb_fps_q4 = []() noexcept -> u8 {
        const float frames = std::clamp(BbFrameRate::CurrentFramesPer30(), 0.05f, 1.5f);
        const float fps_scale = std::clamp(1.0f / frames, 1.0f, 12.0f);
        return static_cast<u8>(
            std::clamp(static_cast<int>(std::lround(fps_scale * 16.0f)), 16, 255));
    }();
    static const bool all_motion = [] {
        const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
        return value && value[0] == '1';
    }();
    const u32 guest_flip_epoch = TemporalUpscaler::GuestFlipsSubmittedForMt();

    scratch.chunk_copies.clear();
    scratch.chunk_gpu_writes.clear();
    scratch.chunk_target_writes.clear();
    scratch.ResetConstMemo();
    u64 local_copy_gen = 0x8000000000000000ULL;

    auto upload_or_memo_const = [&](VAddr addr, u32 byte_size, u64 gen_tag,
                                    const void* src_bytes, vk::Buffer& out_buf,
                                    u64& out_off) -> bool {
        if (gen_tag != 0ULL) {
            for (const auto& cm : scratch.const_memo) {
                if (cm.addr == addr && cm.byte_size == byte_size && cm.gen == gen_tag &&
                    cm.handle) {
                    out_buf = cm.handle;
                    out_off = cm.offset;
                    return true;
                }
            }
        }
        const auto alloc = AllocateWorkerConstants(worker_id, byte_size, storage_align, src_bytes);
        if (!alloc) {
            return false;
        }
        out_buf = alloc.buffer;
        out_off = alloc.offset;
        if (gen_tag != 0ULL) {
            auto& slot_m = scratch.const_memo[scratch.const_memo_cursor++ &
                                              (scratch.const_memo.size() - 1u)];
            slot_m = WorkerCompileScratch::UploadedConstMemo{
                addr, byte_size, gen_tag, alloc.buffer, alloc.offset};
        }
        return true;
    };

    struct WorkerCbResolveCtx {
        VkMtDirectRecorder* rec;
        Rasterizer* rast;
        Core::MemoryManager* mem;
        WorkerCompileScratch* sc;
        VAddr cue_base;
        VAddr cue_end;
        u32 guest_frame;
    } cb_ctx{this, &rasterizer, mem, &scratch, cue_base, cue_end, cur_guest_frame};

    const auto resolve_cb_for_vel = +[](void* raw_ctx, const Gcn::Buffer& sharp, u32 min_bytes,
                                        const u8*& out_ptr, u32& out_size,
                                        u64& out_gen) noexcept -> bool {
        auto& c = *static_cast<WorkerCbResolveCtx*>(raw_ctx);
        const VAddr addr = sharp.base_address;
        const u32 req_sz = static_cast<u32>(std::min<u64>(sharp.GetSize(), c.sc->cb_temp.size()));
        if (!addr || req_sz < min_bytes || min_bytes == 0u) {
            return false;
        }
        const u64 gw_gen = c.sc->MaxGpuWriteGenOverlapping(addr, req_sz);
        for (size_t i = c.sc->chunk_copies.size(); i-- > 0;) {
            const auto& cp = c.sc->chunk_copies[i];
            if (cp.gen < gw_gen) {
                break;
            }
            if (addr >= cp.dst_va && addr < cp.dst_va + cp.byte_size) {
                const u64 off = addr - cp.dst_va;
                const u64 avail = cp.byte_size - off;
                if (avail >= min_bytes && (gw_gen == 0ULL || avail >= req_sz)) {
                    out_ptr = reinterpret_cast<const u8*>(cp.src_va + off);
                    out_size = static_cast<u32>(std::min<u64>(avail, req_sz));
                    out_gen = cp.gen;
                    return true;
                }
                break;
            }
        }
        if (gw_gen != 0ULL) {
            return false;
        }
        if ((addr >= c.cue_base && addr < c.cue_end) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, req_sz)) {
            if (!c.rast->IsRegionGpuModifiedForMt(addr, req_sz) &&
                (!c.mem || c.mem->ClampRangeSize(addr, req_sz) == req_sz)) {
                out_ptr = reinterpret_cast<const u8*>(addr);
                out_size = req_sz;
                out_gen = 0ULL;
                return true;
            }
            return false;
        }
        u32 copied = 0;
        u64 s_gen = 0;
        if (c.rec->CopyFromGuestShadow(addr, min_bytes, req_sz, c.sc->cb_temp.data(), req_sz,
                                       copied, s_gen)) {
            out_ptr = c.sc->cb_temp.data();
            out_size = copied;
            out_gen = s_gen;
            return true;
        }
        if (c.rec->WasCoarseRangeModifiedRecently(addr, req_sz, c.guest_frame) ||
            c.rast->IsRegionGpuModifiedForMt(addr, req_sz)) {
            return false;
        }
        if (c.rast->GetDirectBuffers().TryCopyCpuMirrorMt(addr, c.sc->cb_temp.data(), req_sz)) {
            out_ptr = c.sc->cb_temp.data();
            out_size = req_sz;
            out_gen = 0x5000000000000000ULL | addr;
            return true;
        }
        if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(addr, req_sz) &&
            (!c.mem || c.mem->ClampRangeSize(addr, req_sz) == req_sz)) {
            out_ptr = reinterpret_cast<const u8*>(addr);
            out_size = req_sz;
            out_gen = 0ULL;
            return true;
        }
        return false;
    };

    const auto resolve_cb_for_occ = +[](void* raw_ctx, const Gcn::Buffer& sharp,
                                        const u8*& out_ptr, u32& out_size) noexcept -> bool {
        u64 dummy_gen = 0ULL;
        const u32 min_b = static_cast<u32>(std::min<u64>(sharp.GetSize(), 48ULL));
        auto& c = *static_cast<WorkerCbResolveCtx*>(raw_ctx);
        const VAddr addr = sharp.base_address;
        const u32 req_sz = static_cast<u32>(std::min<u64>(sharp.GetSize(), c.sc->cb_temp.size()));
        if (!addr || req_sz < min_b || min_b == 0u) {
            return false;
        }
        const u64 gw_gen = c.sc->MaxGpuWriteGenOverlapping(addr, req_sz);
        for (size_t i = c.sc->chunk_copies.size(); i-- > 0;) {
            const auto& cp = c.sc->chunk_copies[i];
            if (cp.gen < gw_gen) {
                break;
            }
            if (addr >= cp.dst_va && addr < cp.dst_va + cp.byte_size) {
                const u64 off = addr - cp.dst_va;
                const u64 avail = cp.byte_size - off;
                if (avail >= min_b && (gw_gen == 0ULL || avail >= req_sz)) {
                    out_ptr = reinterpret_cast<const u8*>(cp.src_va + off);
                    out_size = static_cast<u32>(std::min<u64>(avail, req_sz));
                    return true;
                }
                break;
            }
        }
        if (gw_gen != 0ULL) {
            return false;
        }
        if ((addr >= c.cue_base && addr < c.cue_end) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, req_sz)) {
            if (!c.rast->IsRegionGpuModifiedForMt(addr, req_sz) &&
                (!c.mem || c.mem->ClampRangeSize(addr, req_sz) == req_sz)) {
                out_ptr = reinterpret_cast<const u8*>(addr);
                out_size = req_sz;
                return true;
            }
            return false;
        }
        u32 copied = 0;
        if (c.rec->CopyFromGuestShadow(addr, min_b, req_sz, c.sc->cb_temp.data(), req_sz, copied,
                                       dummy_gen)) {
            out_ptr = c.sc->cb_temp.data();
            out_size = copied;
            return true;
        }
        if (c.rec->WasCoarseRangeModifiedRecently(addr, req_sz, c.guest_frame) ||
            c.rast->IsRegionGpuModifiedForMt(addr, req_sz)) {
            return false;
        }
        if (c.rast->GetDirectBuffers().TryCopyCpuMirrorMt(addr, c.sc->cb_temp.data(), req_sz)) {
            out_ptr = c.sc->cb_temp.data();
            out_size = req_sz;
            return true;
        }
        if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(addr, req_sz) &&
            (!c.mem || c.mem->ClampRangeSize(addr, req_sz) == req_sz)) {
            out_ptr = reinterpret_cast<const u8*>(addr);
            out_size = req_sz;
            return true;
        }
        return false;
    };

    auto resolve_gpu_stream_range = [&](VAddr addr, u64 req_size, u64 min_span,
                                        vk::Buffer& out_buf, u64& out_off, u64& out_total_bytes,
                                        bool is_formatted = false,
                                        u8* out_fail_step = nullptr) -> bool {
        auto fail = [&](u8 step) noexcept -> bool {
            if (out_fail_step) {
                *out_fail_step = step;
            }
            return false;
        };
        if (!addr || req_size == 0ULL) {
            return fail(1u);
        }
        const bool in_transient =
            (addr >= cue_base && addr < cue_end) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, 1ULL);
        const u64 gw_gen = scratch.MaxGpuWriteGenOverlapping(addr, req_size);
        for (size_t i = scratch.chunk_copies.size(); i-- > 0;) {
            const auto& cp = scratch.chunk_copies[i];
            if (cp.gen < gw_gen) {
                break;
            }
            if (addr >= cp.dst_va && addr < cp.dst_va + cp.byte_size) {
                const u64 off = addr - cp.dst_va;
                const u64 avail = cp.byte_size - off;
                if (avail < min_span || (gw_gen != 0ULL && avail < req_size)) {
                    return fail(2u);
                }
                const u64 up_sz64 = std::min<u64>(req_size, avail);
                if (up_sz64 <= 262144ULL) {
                    const u32 up_sz = static_cast<u32>(up_sz64);
                    if (!upload_or_memo_const(addr, up_sz, cp.gen,
                                              reinterpret_cast<const void*>(cp.src_va + off),
                                              out_buf, out_off)) {
                        return fail(2u);
                    }
                    out_total_bytes = out_off + up_sz;
                    return true;
                }
                if (!in_transient) {
                    const auto resolved = ResolveBuffer(worker_id, addr, req_size);
                    if (resolved && resolved.vram_handle != VK_NULL_HANDLE &&
                        resolved.clamped_size >= min_span) {
                        out_buf = vk::Buffer{resolved.vram_handle};
                        out_off = resolved.vram_offset;
                        out_total_bytes = resolved.vram_offset + resolved.clamped_size;
                        return true;
                    }
                }
                return fail(4u);
            }
            if (cp.dst_va > addr && cp.dst_va < addr + req_size) {
                return fail(3u); // Straddles start of an in-list staged copy region
            }
        }

        if (gw_gen != 0ULL) {
            if (!in_transient) {
                const auto resolved = ResolveBuffer(worker_id, addr, req_size);
                if (resolved && resolved.vram_handle != VK_NULL_HANDLE &&
                    resolved.clamped_size >= min_span &&
                    scratch.FullGpuWriteCovers(addr, resolved.clamped_size, gw_gen)) {
                    out_buf = vk::Buffer{resolved.vram_handle};
                    out_off = resolved.vram_offset;
                    out_total_bytes = resolved.vram_offset + resolved.clamped_size;
                    return true;
                }
            }
            return fail(5u);
        }

        if (!in_transient && req_size <= scratch.cb_temp.size()) {
            u32 copied = 0;
            u64 s_gen = 0;
            if (CopyFromGuestShadow(addr, static_cast<u32>(std::min<u64>(min_span, req_size)),
                                    static_cast<u32>(req_size), scratch.cb_temp.data(),
                                    static_cast<u32>(req_size), copied, s_gen)) {
                if (!upload_or_memo_const(addr, copied, s_gen, scratch.cb_temp.data(), out_buf,
                                          out_off)) {
                    return fail(6u);
                }
                out_total_bytes = out_off + copied;
                return true;
            }
        }

        if (!in_transient) {
            const auto resolved = ResolveBuffer(worker_id, addr, req_size);
            if (resolved && resolved.clamped_size >= min_span) {
                const u64 check_sz = std::min<u64>(req_size, resolved.clamped_size);
                if (!resolved.is_gpu_written &&
                    !WasCoarseRangeModifiedRecently(addr, check_sz, cur_guest_frame) &&
                    (resolved.is_static_immutable ||
                     !rasterizer.IsRegionGpuModifiedForMt(addr, check_sz))) {
                    out_buf = vk::Buffer{resolved.handle};
                    out_off = resolved.offset;
                    out_total_bytes = resolved.offset + resolved.clamped_size;
                    return true;
                }
                if (!resolved.is_uav_or_ext_gpu && resolved.vram_handle != VK_NULL_HANDLE) {
                    out_buf = vk::Buffer{resolved.vram_handle};
                    out_off = resolved.vram_offset;
                    out_total_bytes = resolved.vram_offset + resolved.clamped_size;
                    return true;
                }
                if (resolved.vram_handle != VK_NULL_HANDLE && guest_copy_shadow &&
                    guest_copy_shadow->HasValidHugeLiveCover(addr, min_span)) {
                    out_buf = vk::Buffer{resolved.vram_handle};
                    out_off = resolved.vram_offset;
                    out_total_bytes = resolved.vram_offset + resolved.clamped_size;
                    return true;
                }
                return fail(7u);
            }
        }

        if (!in_transient && is_formatted) {
            return fail(8u); // Never touch non-DirectBuffers raw guest RAM for formatted loads
        }

        if (req_size <= 65536ULL && !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(addr, req_size) &&
            (in_transient || !WasCoarseRangeModifiedRecently(addr, req_size, cur_guest_frame)) &&
            !rasterizer.IsRegionGpuModifiedForMt(addr, req_size)) {
            const u64 clamped = mem ? mem->ClampRangeSize(addr, req_size) : req_size;
            if (clamped >= min_span && clamped <= 65536ULL) {
                const u64 memo_tag = in_transient ? (0x4000000000000000ULL | addr) : 0ULL;
                if (!upload_or_memo_const(addr, static_cast<u32>(clamped), memo_tag,
                                          reinterpret_cast<const void*>(addr), out_buf, out_off)) {
                    return fail(9u);
                }
                out_total_bytes = out_off + clamped;
                return true;
            }
        }
        return fail(9u);
    };

    auto resolve_cpu_index_ptr = [&](VAddr ib_addr, u64 ib_bytes) noexcept -> const void* {
        if (!ib_addr || ib_bytes == 0ULL) {
            return nullptr;
        }
        const u64 gw_gen = scratch.MaxGpuWriteGenOverlapping(ib_addr, ib_bytes);
        for (size_t i = scratch.chunk_copies.size(); i-- > 0;) {
            const auto& cp = scratch.chunk_copies[i];
            if (cp.gen < gw_gen) {
                break;
            }
            if (ib_addr >= cp.dst_va && ib_addr < cp.dst_va + cp.byte_size) {
                const u64 off = ib_addr - cp.dst_va;
                return (cp.byte_size - off >= ib_bytes)
                           ? reinterpret_cast<const void*>(cp.src_va + off)
                           : nullptr;
            }
            if (cp.dst_va > ib_addr && cp.dst_va < ib_addr + ib_bytes) {
                return nullptr;
            }
        }
        if (gw_gen != 0ULL) {
            return nullptr;
        }
        if (WasCoarseRangeModifiedRecently(ib_addr, ib_bytes, cur_guest_frame) ||
            rasterizer.IsRegionGpuModifiedForMt(ib_addr, ib_bytes) ||
            (mem && mem->ClampRangeSize(ib_addr, ib_bytes) != ib_bytes)) {
            return nullptr;
        }
        return reinterpret_cast<const void*>(ib_addr);
    };

    Gcn::Regs& rregs = *w.replay_regs;
    BbCp::CommandProcessor::InitGnmDefaultRegs(rregs, 3u);
    w.pso_state.ResetForPass();
    Gcn::RegDirty replay_dirty{};
    replay_dirty.track_all_blocks = true;
    scratch.replay_exact_words.reset();
    replay_dirty.exact_words = &scratch.replay_exact_words;
    u8 replay_cb_written_mask = 0u;
    bool replay_db_written = false;
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> replay_cb_extent{};
    Gcn::CbDbExtent replay_db_extent{};
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> replay_cb_uid{};
    u16 replay_db_uid = 0u;
    u64 replay_pso_epoch = 0ULL;
    u64 replay_dyn_epoch = 0ULL;

    Rasterizer::MtShadowCullState shadow_cull_state{};
    bool has_camera_cb = false;
    std::array<float, 216> camera_cb_floats{};

    bool has_prev_dyn = false;
    u64 prev_dyn_epoch = 0ULL;
    bool prev_dyn_indexed = false;
    std::array<float, 2> prev_dyn_jitter{0.0f, 0.0f};

    VertexInputs<vk::VertexInputBindingDescription2EXT> prev_vb_bindings;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> prev_vb_attrs;
    VertexInputs<vk::Buffer> prev_vb_bufs;
    VertexInputs<vk::DeviceSize> prev_vb_offs;
    bool has_prev_vb_state = false;

    vk::Buffer prev_ib_buf{};
    u64 prev_ib_off = ~0ULL;
    bool prev_ib_16 = false;

    u32 next_plan_idx = 0u;
    u32 draw_ordinal = 0u;

    for (u32 s = 0; s < num_slices; ++s) {
        const u32* cur = slices[s].begin_ptr;
        u32 rem = slices[s].orig_words;
        while (rem > 0u) {
            const MtDecodedPkt dp = DecodeAndApplyMtPacket(
                cur, rem, rregs, replay_cb_extent, replay_db_extent, replay_cb_uid, replay_db_uid,
                replay_pso_epoch, replay_dyn_epoch, mem, false, &replay_dirty,
                &replay_cb_written_mask, &replay_db_written);
            if (dp.kind == MtPktKind::Invalid || dp.pkt_words == 0u) {
                return abort_pass(MtBailReason::Pkt);
            }

            if (dp.kind == MtPktKind::CopyBatch) {
                const u32* body = cur + 1u;
                const u32 count = body[1] >> 16u;
                for (u32 i = 0; i < count; ++i) {
                    BbCp::DirectVk::CopyRegion reg{};
                    std::memcpy(&reg, body + 2u + i * 6u, sizeof(reg));
                    const bool is_cpu_flag =
                        (reg.reserved == gx_direct_vk::kCopyFlagKnownCpuSource ||
                         reg.reserved == gx_direct_vk::kCopyFlagLiveStagingPtr);
                    const bool cpu_readable =
                        is_cpu_flag && reg.src_va != 0ULL && reg.byte_size <= (4u << 20u) &&
                        (!mem || mem->ClampRangeSize(reg.src_va, reg.byte_size) == reg.byte_size);
                    const u64 cur_gen = ++local_copy_gen;
                    if (cpu_readable) {
                        scratch.chunk_copies.push_back(
                            {reg.dst_va, reg.src_va, reg.byte_size, cur_gen});
                    } else {
                        scratch.chunk_gpu_writes.push_back({reg.dst_va, reg.byte_size, cur_gen});
                        scratch.NoteTargetWrite(reg.dst_va, reg.byte_size);
                    }
                }
            } else if (dp.kind == MtPktKind::FillBuf) {
                BbCp::DirectVk::FillArgs fargs{};
                std::memcpy(&fargs, cur + 3u, sizeof(fargs));
                scratch.chunk_gpu_writes.push_back(
                    {fargs.dst_va, fargs.byte_size, ++local_copy_gen});
                scratch.NoteTargetWrite(fargs.dst_va, fargs.byte_size);
            } else if (dp.kind == MtPktKind::DmaMemWrite) {
                const auto dma = BbCp::DmaData::From(cur + 1u);
                scratch.chunk_gpu_writes.push_back(
                    {dma.DstAddress(), dma.Bytes(), ++local_copy_gen});
                scratch.NoteTargetWrite(dma.DstAddress(), dma.Bytes());
            } else if (dp.kind == MtPktKind::TargetRawOp) {
                const auto raw_vk_op = BbCp::DirectVk::Op(cur[2] & 0xFFFFu);
                if (raw_vk_op == BbCp::DirectVk::Op::CopyTextureRegion) {
                    BbCp::DirectVk::CopyTextureRegionArgs cargs{};
                    std::memcpy(&cargs, cur + 3u, sizeof(cargs));
                    scratch.NoteTargetWrite(cargs.dst_va, cargs.dst_byte_size);
                } else if (raw_vk_op == BbCp::DirectVk::Op::UploadTextureSubresource) {
                    BbCp::DirectVk::UploadTextureArgs uargs{};
                    std::memcpy(&uargs, cur + 3u, sizeof(uargs));
                    scratch.NoteTargetWrite(uargs.dst_va, uargs.dst_byte_size);
                } else {
                    VAddr raw_dst_va = 0;
                    u32 raw_dst_sz = 0;
                    std::memcpy(&raw_dst_va, cur + 3u, sizeof(VAddr));
                    std::memcpy(&raw_dst_sz, cur + 5u, sizeof(u32));
                    scratch.NoteTargetWrite(raw_dst_va, raw_dst_sz);
                }
            } else if (dp.kind == MtPktKind::Draw) {
                if (next_plan_idx < planned_subpasses &&
                    draw_ordinal == scratch.subpass_plans[next_plan_idx].first_draw_ordinal) {
                    seal_active_subpass();
                    if (!open_subpass(next_plan_idx++)) {
                        return abort_pass(MtBailReason::Other);
                    }
                    has_prev_dyn = false;
                    has_prev_vb_state = false;
                    prev_vb_bufs.clear();
                    prev_vb_offs.clear();
                    prev_ib_buf = vk::Buffer{};
                    prev_ib_off = ~0ULL;
                }
                ++draw_ordinal;
                if (!w.is_recording || cur_subpass_idx >= planned_subpasses) {
                    cur += dp.pkt_words;
                    rem -= dp.pkt_words;
                    continue;
                }

                const auto& cur_plan = scratch.subpass_plans[cur_subpass_idx];
                const auto& cur_ts = cur_plan.target_summary;
                const u32 num_instances = rregs.num_instances.NumInstances();
                if (dp.is_indirect || (rregs.num_indices != 0u && num_instances != 0u)) {
                    if (dp.is_indirect && IsAttachmentlessDummyIndirectDrawPrePso(rregs)) {
                        cur += dp.pkt_words;
                        rem -= dp.pkt_words;
                        continue;
                    }
                    PipelineCache::MtPsoFailKind p2_fail = PipelineCache::MtPsoFailKind::None;
                    const GraphicsPipeline* pso =
                        rasterizer.GetPipelineCache().SelectGraphicsPipelineMt(
                            rregs, w.pso_state,
                            dp.is_indirect ? DrawIndirectParams{0u, 0u} : DrawIndirectParams{},
                            dp.is_indirect, /*allow_compile_fallback=*/false, &p2_fail);
                    if (!pso) {
                        return abort_pass(MtBailReason::Pso, p2_fail, false);
                    }

                    const auto* vs_stage = pso->GetStages()[u32(Shader::SwStage::Vertex)];
                    const auto* ps_stage = pso->GetStages()[u32(Shader::SwStage::Fragment)];
                    if (!vs_stage) {
                        return abort_pass(MtBailReason::Pso, p2_fail, true);
                    }
                    if (dp.is_indirect &&
                        IsAttachmentlessDummyIndirectDraw(rregs, pso, vs_stage, ps_stage)) {
                        cur += dp.pkt_words;
                        rem -= dp.pkt_words;
                        continue;
                    }

                    // Flatten VS & PS user-data into worker heap scratch (+0 B TLS / stack).
                    std::memcpy(scratch.vs_flat_ud.data(), rregs.vs_program.user_data.data(),
                                16u * sizeof(u32));
                    if (vs_stage->srt_info.walker_func) {
                        vs_stage->srt_info.walker_func(rregs.vs_program.user_data.data(),
                                                       scratch.vs_flat_ud.data());
                    }
                    if (ps_stage) {
                        std::memcpy(scratch.ps_flat_ud.data(), rregs.ps_program.user_data.data(),
                                    16u * sizeof(u32));
                        if (ps_stage->srt_info.walker_func) {
                            ps_stage->srt_info.walker_func(rregs.ps_program.user_data.data(),
                                                           scratch.ps_flat_ud.data());
                        }
                    }
                    const VAddr vs_pgm_base = rregs.vs_program.Address<VAddr>();
                    const VAddr ps_pgm_base = rregs.ps_program.Address<VAddr>();

                    // 1. MicroShadow clutter culler
                    if (!dp.is_indirect &&
                        Rasterizer::TryCullMicroShadowDrawMt(rregs, pso, dp.is_indexed,
                                                             shadow_cull_state,
                                                             &w.local_stats.shadow_skipped)) {
                        cur += dp.pkt_words;
                        rem -= dp.pkt_words;
                        continue;
                    }

                    // 2. Object-Velocity stationary mesh culler (Pass 0x2e)
                    if (!dp.is_indirect && pso->GetGraphicsKey().mrt_mask == 0x1u && ps_stage &&
                        ObjectVelocityCuller::IsObjectVelocityPs(ps_stage->pgm_hash)) {
                        if (ObjectVelocityCuller::TryCullDrawMt(
                                rregs, pso, scratch.vs_flat_ud.data(), vs_pgm_base, dp.is_indexed,
                                resolve_cb_for_vel, &cb_ctx,
                                std::span<ObjectVelocityCuller::PairMemoEntry,
                                          ObjectVelocityCuller::kPairMemoSize>{w.vel_pair_memo},
                                &w.local_stats.vel_seen, &w.local_stats.vel_culled)) {
                            cur += dp.pkt_words;
                            rem -= dp.pkt_words;
                            continue;
                        }
                    }

                    // Extract fetch-shader V# descriptors once for occlusion + VB resolution.
                    VertexInputs<Gcn::Buffer> vsharps;
                    const auto& fetch = pso->GetFetchShader();
                    if (fetch) {
                        const std::span<const u32> vs_ud16{rregs.vs_program.user_data.data(), 16u};
                        for (const auto& attrib : fetch->attributes) {
                            vsharps.push_back(attrib.GetSharpDirect(vs_ud16));
                        }
                    }

                    // 3. Scene Occlusion Culler (G-Buffer 6/7-MRT)
                    Rasterizer::MtOcclusionDecision occ_decision{};
                    if (!dp.is_indirect && cur_ts.is_gbuffer) {
                        occ_decision = rasterizer.EvaluateGBufferOcclusionMt(
                            rregs, pso, scratch.vs_flat_ud.data(), vs_pgm_base, dp.is_indexed,
                            dp.start_index, vsharps, occ_block_id, resolve_cb_for_occ, &cb_ctx,
                            &w.local_stats.occ_candidates, &w.local_stats.occ_skipped,
                            &w.local_stats.occ_probed);
                        if (occ_decision.skip_draw) {
                            cur += dp.pkt_words;
                            rem -= dp.pkt_words;
                            continue;
                        }
                    }

                    // Compute draw vertex/instance offsets up front (shared by ObjectMotion & Draw)
                    u32 vertex_offset = rregs.index_offset;
                    u32 instance_offset = 0u;
                    if (fetch) {
                        if (vertex_offset == 0u && fetch->vertex_offset_sgpr != -1 &&
                            static_cast<u32>(fetch->vertex_offset_sgpr) < 16u) {
                            vertex_offset =
                                rregs.vs_program.user_data[fetch->vertex_offset_sgpr];
                        }
                        if (fetch->instance_offset_sgpr != -1 &&
                            static_cast<u32>(fetch->instance_offset_sgpr) < 16u) {
                            instance_offset =
                                rregs.vs_program.user_data[fetch->instance_offset_sgpr];
                        }
                    }

                    // Resolve Vertex Inputs & Ranges before PushData (needed for ObjectMotion hash)
                    VertexInputs<vk::VertexInputAttributeDescription2EXT> vb_attrs;
                    VertexInputs<vk::VertexInputBindingDescription2EXT> vb_bindings;
                    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> vb_divisors;
                    VertexInputs<Gcn::Buffer> guest_vbs;
                    pso->GetVertexInputs(vb_attrs, vb_bindings, vb_divisors, guest_vbs,
                                         rregs.vgt_instance_step_rate_0,
                                         rregs.vgt_instance_step_rate_1, vsharps);

                    struct VbRange {
                        VAddr begin{0};
                        VAddr end{0};
                        vk::Buffer handle{};
                        u64 offset{0};
                        u64 total_bytes{0};
                    };
                    VertexInputs<VbRange> vb_streams;
                    for (const auto& gb : guest_vbs) {
                        if (gb.base_address != 0 && gb.GetSize() > 0) {
                            u64 raw_span = gb.GetSize();
                            const u64 stride = gb.GetStride();
                            if (stride > 0 && gb.num_records > 0) {
                                const auto dfmt = gb.GetDataFmt();
                                if (dfmt != Gcn::DataFormat::FormatInvalid &&
                                    u32(dfmt) <= u32(Gcn::DataFormat::FormatBc7)) {
                                    const u64 elem_bytes = Gcn::NumBitsPerBlock(dfmt) / 8u;
                                    if (elem_bytes > 0 && elem_bytes <= stride) {
                                        raw_span = (u64(gb.num_records) - 1ULL) * stride + elem_bytes;
                                    }
                                }
                            }
                            const u64 stream_size =
                                VideoCore::DirectBuffers::ClampNativeGpuReadSize(gb.base_address,
                                                                                 raw_span);
                            vb_streams.push_back({gb.base_address, gb.base_address + stream_size});
                        }
                    }
                    std::ranges::sort(vb_streams, {}, &VbRange::begin);
                    VertexInputs<VbRange> vb_ranges;
                    VAddr cur_exact_end = 0;
                    for (const auto& raw_st : vb_streams) {
                        auto st = raw_st;
                        if (cur_exact_end != 0 && st.begin < cur_exact_end &&
                            st.end > cur_exact_end) {
                            st.end = cur_exact_end;
                        }
                        if (vb_ranges.empty() || vb_ranges.back().end <= st.begin) {
                            vb_ranges.push_back(st);
                            cur_exact_end = VideoCore::DirectBuffers::LookupExactBufferEnd(st.begin);
                        } else {
                            vb_ranges.back().end = std::max(vb_ranges.back().end, st.end);
                        }
                    }

                    for (auto& rng : vb_ranges) {
                        u64 min_span = 1ULL;
                        for (const auto& gb : guest_vbs) {
                            if (gb.base_address >= rng.begin && gb.base_address < rng.end) {
                                min_span = std::max<u64>(
                                    min_span, (gb.base_address - rng.begin) +
                                                  std::max<u64>(1ULL, gb.GetStride()));
                            }
                        }
                        u8 vb_step = 0u;
                        if (!resolve_gpu_stream_range(rng.begin, rng.end - rng.begin, min_span,
                                                      rng.handle, rng.offset, rng.total_bytes,
                                                      false, &vb_step)) {
                            MtMaybeLogBailBufSample(1u, vb_step, false, rng.begin,
                                                    rng.end - rng.begin, vs_stage->pgm_hash);
                            return abort_pass(MtBailReason::Buf);
                        }
                    }

                    // Resolve Index Buffer if indexed
                    const bool index16 =
                        rregs.index_buffer_type.index_type == Gcn::IndexType::Index16;
                    const u32 idx_sz = index16 ? 2u : 4u;
                    const VAddr ib_addr =
                        dp.is_indexed
                            ? (rregs.index_base_address.Address<VAddr>() +
                               u64(dp.start_index) * idx_sz)
                            : VAddr{0};
                    const u64 ib_bytes = dp.is_indexed ? (u64(rregs.num_indices) * idx_sz) : 0ULL;
                    vk::Buffer ib_buf{};
                    u64 ib_off = 0;
                    if (dp.is_indexed) {
                        if (ib_bytes == 0ULL) {
                            MtMaybeLogBailBufSample(2u, 1u, false, ib_addr, ib_bytes,
                                                    vs_stage->pgm_hash);
                            return abort_pass(MtBailReason::Buf);
                        }
                        const u64 ib_min_span = dp.is_indirect ? u64(idx_sz) : ib_bytes;
                        u64 ib_tot = 0;
                        u8 ib_step = 0u;
                        if (!resolve_gpu_stream_range(ib_addr, ib_bytes, ib_min_span, ib_buf,
                                                      ib_off, ib_tot, false, &ib_step)) {
                            MtMaybeLogBailBufSample(2u, ib_step, false, ib_addr, ib_bytes,
                                                    vs_stage->pgm_hash);
                            return abort_pass(MtBailReason::Buf);
                        }
                    }

                    // Resolve Indirect Arguments Buffer if DrawIndirect
                    vk::Buffer ind_buf{};
                    u64 ind_off = 0ULL;
                    if (dp.is_indirect) {
                        const VAddr ind_va = dp.indirect_args_va;
                        const u64 ind_bytes = dp.is_indexed ? 20ULL : 16ULL;
                        const bool guard_on = IndirectGuard::DrawEnabled();
                        if (guard_on && force_indirect_guard) {
                            return abort_pass(MtBailReason::Pkt);
                        }
                        bool resolved_ind = false;
                        u8 ind_step = 0u;
                        if (!guard_on) {
                            u64 ind_tot = 0ULL;
                            resolved_ind = resolve_gpu_stream_range(
                                ind_va, ind_bytes, ind_bytes, ind_buf, ind_off, ind_tot, false,
                                &ind_step);
                        }
                        if (!resolved_ind) {
                            const bool ind_in_transient =
                                (ind_va >= cue_base && ind_va < cue_end) ||
                                VideoCore::DirectBuffers::IsTransientAuxCpuRing(ind_va, 1ULL);
                            if (!ind_in_transient) {
                                const auto res_ind = ResolveBuffer(worker_id, ind_va, ind_bytes);
                                if (res_ind && res_ind.clamped_size >= ind_bytes) {
                                    bool shadowed_by_cpu_copy = false;
                                    const u64 gw_gen =
                                        scratch.MaxGpuWriteGenOverlapping(ind_va, ind_bytes);
                                    for (size_t ci = scratch.chunk_copies.size(); ci-- > 0;) {
                                        const auto& cp = scratch.chunk_copies[ci];
                                        if (cp.gen < gw_gen) {
                                            break;
                                        }
                                        if (ind_va < cp.dst_va + cp.byte_size &&
                                            cp.dst_va < ind_va + ind_bytes) {
                                            shadowed_by_cpu_copy = true;
                                            break;
                                        }
                                    }
                                    if (!shadowed_by_cpu_copy) {
                                        if (!res_ind.is_gpu_written &&
                                            !WasCoarseRangeModifiedRecently(ind_va, ind_bytes,
                                                                            cur_guest_frame) &&
                                            !rasterizer.IsRegionGpuModifiedForMt(ind_va,
                                                                                 ind_bytes)) {
                                            ind_buf = vk::Buffer{res_ind.handle};
                                            ind_off = res_ind.offset;
                                            resolved_ind = true;
                                        } else if (res_ind.vram_handle != VK_NULL_HANDLE) {
                                            ind_buf = vk::Buffer{res_ind.vram_handle};
                                            ind_off = res_ind.vram_offset;
                                            resolved_ind = true;
                                            if (res_ind.is_uav_or_ext_gpu ||
                                                rasterizer.IsRegionGpuModifiedForMt(ind_va,
                                                                                    ind_bytes) ||
                                                gw_gen != 0ULL) {
                                                bool dup_ind = false;
                                                for (u8 ui = 0; ui < cur_num_uav_indirect_refs;
                                                     ++ui) {
                                                    const auto& ex = cur_uav_indirect_refs[ui];
                                                    if (ex.addr == ind_va &&
                                                        ex.byte_size >=
                                                            static_cast<u32>(ind_bytes)) {
                                                        dup_ind = true;
                                                        break;
                                                    }
                                                }
                                                if (!dup_ind) {
                                                    if (cur_num_uav_indirect_refs >=
                                                        MtCompletedPassMeta::kMaxUavIndirectRefs) {
                                                        MtMaybeLogBailBufSample(
                                                            7u, 12u, false, ind_va, ind_bytes,
                                                            vs_stage->pgm_hash);
                                                        return abort_pass(MtBailReason::Buf);
                                                    }
                                                    cur_uav_indirect_refs
                                                        [cur_num_uav_indirect_refs++] =
                                                            MtCompletedPassMeta::UavIndirectRef{
                                                                .addr = ind_va,
                                                                .byte_size =
                                                                    static_cast<u32>(ind_bytes),
                                                            };
                                                }
                                            }
                                        } else {
                                            ind_step = 7u;
                                        }
                                    } else {
                                        u64 ind_tot = 0ULL;
                                        resolved_ind = resolve_gpu_stream_range(
                                            ind_va, ind_bytes, ind_bytes, ind_buf, ind_off,
                                            ind_tot, false, &ind_step);
                                    }
                                } else {
                                    ind_step = 9u;
                                }
                            }
                            if (!resolved_ind) {
                                MtMaybeLogBailBufSample(7u, ind_step, false, ind_va, ind_bytes,
                                                        vs_stage->pgm_hash);
                                return abort_pass(MtBailReason::Buf);
                            }
                        }
                    }

                    // Build PushData (target_scale + scene_size + ObjectMotion parameter)
                    Shader::PushData push_data{};
                    {
                        const auto& ctl = rregs.viewport_control;
                        const auto& vp0 = rregs.viewports[0];
                        push_data.xoffset =
                            (ctl.xoffset_enable ? vp0.xoffset : 0.f) * cur_ts.target_scale[0];
                        push_data.xscale =
                            (ctl.xscale_enable ? vp0.xscale : 1.f) * cur_ts.target_scale[0];
                        push_data.yoffset =
                            (ctl.yoffset_enable ? vp0.yoffset : 0.f) * cur_ts.target_scale[1];
                        push_data.yscale =
                            (ctl.yscale_enable ? vp0.yscale : 1.f) * cur_ts.target_scale[1];
                        push_data.scene_size = cur_ts.scene_size;
                        push_data.buf_offsets[37] = mb_fps_q4;
                        push_data.buf_offsets[38] = mb_mask;
                        push_data.buf_offsets[39] = fx_mask;
                    }

                    if (!dp.is_indirect && pso->GetGraphicsKey().motion_vectors != 0u &&
                        !guest_vbs.empty()) {
                        if (auto* obj_motion = rasterizer.GetObjectMotion();
                            obj_motion && obj_motion->Enabled()) {
                            const u64 motion_geometry = XXH3_64bits(
                                guest_vbs.data(), guest_vbs.size() * sizeof(Gcn::Buffer));
                            bool gated = !all_motion;
                            u64 palette = 0ULL;
                            for (const auto& res : vs_stage->buffers) {
                                if (res.IsSpecial()) {
                                    continue;
                                }
                                const Gcn::Buffer bsharp =
                                    res.GetSharpDirect(scratch.vs_flat_ud.data(), vs_pgm_base);
                                const u32 bsz = bsharp.GetSize();
                                if (!bsharp.Valid() || bsharp.GetStride() != 16u) {
                                    continue;
                                }
                                const auto role = Motion::ClassifyBuffer(bsz);
                                if (role == Motion::BufferRole::Skeleton) {
                                    gated = false;
                                    break;
                                }
                                if (role == Motion::BufferRole::SmallSkeleton) {
                                    const u8* pal_ptr = nullptr;
                                    u32 pal_sz = 0u;
                                    if (resolve_cb_for_occ(&cb_ctx, bsharp, pal_ptr, pal_sz) &&
                                        pal_ptr && pal_sz >= bsz) {
                                        palette = XXH3_64bits_withSeed(pal_ptr, bsz, palette);
                                    }
                                }
                            }
                            const void* idx_cpu_ptr =
                                dp.is_indexed ? resolve_cpu_index_ptr(ib_addr, ib_bytes) : nullptr;
                            u32 used_slot = ~0u;
                            push_data.motion_param = obj_motion->PrepareDrawMt(
                                guest_flip_epoch, gated,
                                Motion::History::GateKey{
                                    .shader = vs_stage->pgm_hash,
                                    .geometry = motion_geometry,
                                    .indices = ib_addr,
                                    .index_count = rregs.num_indices,
                                    .instances = num_instances,
                                    .first_instance = instance_offset,
                                },
                                palette, dp.is_indexed, ib_addr, rregs.num_indices, idx_sz,
                                static_cast<s32>(vertex_offset),
                                (rregs.enable_primitive_restart & 1u) != 0u, num_instances,
                                instance_offset, idx_cpu_ptr, used_slot);
                            if (used_slot < 4u) {
                                cur_motion_slot = static_cast<u8>(used_slot);
                            }
                        }
                    }

                    // Bind Graphics Pipeline
                    const vk::Pipeline vk_pso = pso->Handle();
                    if (w.bound_pipeline != vk_pso) {
                        w.active_cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, vk_pso);
                        w.bound_pipeline = vk_pso;
                    }

                    std::array<vk::DescriptorBufferInfo, 64> buf_infos{};
                    std::array<vk::DescriptorImageInfo, 64> img_infos{};
                    std::array<vk::WriteDescriptorSet, 64> desc_writes{};
                    u32 num_buf_infos = 0;
                    u32 num_img_infos = 0;
                    u32 num_desc_writes = 0;
                    Shader::Backend::Bindings binding{};

                    for (const auto* stage : pso->GetStages()) {
                        if (!stage) {
                            continue;
                        }
                        const bool is_vs = (stage->sw_stage == Shader::SwStage::Vertex);
                        const u32* const ud16 = is_vs ? rregs.vs_program.user_data.data()
                                                      : rregs.ps_program.user_data.data();
                        const u32* const flat_ud =
                            is_vs ? scratch.vs_flat_ud.data() : scratch.ps_flat_ud.data();
                        const VAddr pgm_base = is_vs ? vs_pgm_base : ps_pgm_base;

                        u32 ud_mask = stage->ud_mask.mask;
                        while (ud_mask != 0u) {
                            const u32 bit = static_cast<u32>(std::countr_zero(ud_mask));
                            ud_mask &= (ud_mask - 1u);
                            push_data.ud_regs[binding.user_data++] = ud16[bit];
                        }

                        // Buffers (all StorageBuffers per SPIR-V backend)
                        for (const auto& desc : stage->buffers) {
                            if (num_buf_infos >= buf_infos.size() ||
                                num_desc_writes >= desc_writes.size()) {
                                return abort_pass(MtBailReason::Other);
                            }
                            vk::DescriptorBufferInfo& bi = buf_infos[num_buf_infos++];
                            if (desc.IsSpecial()) {
                                switch (desc.buffer_type) {
                                case Shader::BufferType::Flatbuf: {
                                    const u32 fsz =
                                        stage->srt_info.flattened_bufsize_dw * sizeof(u32);
                                    const auto alloc = AllocateWorkerConstants(
                                        worker_id, fsz, storage_align, flat_ud);
                                    if (!alloc) {
                                        MtMaybeLogBailBufSample(3u, 0u, false, 0ULL, fsz,
                                                                stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    bi = vk::DescriptorBufferInfo{alloc.buffer, alloc.offset, fsz};
                                    break;
                                }
                                case Shader::BufferType::ClipPlanes: {
                                    if (rregs.clipper_control.user_clip_plane_enable == 0u) {
                                        bi = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0,
                                                                      VK_WHOLE_SIZE};
                                    } else {
                                        std::array<float, Gcn::NUM_CLIP_PLANES * 4> planes{};
                                        for (u32 p = 0; p < Gcn::NUM_CLIP_PLANES; ++p) {
                                            const auto& pl = rregs.clip_user_data[p];
                                            planes[p * 4 + 0] = std::bit_cast<float>(pl.data_x);
                                            planes[p * 4 + 1] = std::bit_cast<float>(pl.data_y);
                                            planes[p * 4 + 2] = std::bit_cast<float>(pl.data_z);
                                            planes[p * 4 + 3] = std::bit_cast<float>(pl.data_w);
                                        }
                                        const auto alloc = AllocateWorkerConstants(
                                            worker_id, sizeof(planes), storage_align,
                                            planes.data());
                                        if (!alloc) {
                                            MtMaybeLogBailBufSample(4u, 0u, false, 0ULL,
                                                                    sizeof(planes),
                                                                    stage->pgm_hash);
                                            return abort_pass(MtBailReason::Buf);
                                        }
                                        bi = vk::DescriptorBufferInfo{alloc.buffer, alloc.offset,
                                                                      sizeof(planes)};
                                    }
                                    break;
                                }
                                case Shader::BufferType::BdaPagetable: {
                                    const auto* tbl =
                                        rasterizer.GetBufferCache().GetBdaPageTableBuffer();
                                    if (!tbl) {
                                        MtMaybeLogBailBufSample(5u, 1u, false, 0ULL, 0ULL,
                                                                stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    bi = vk::DescriptorBufferInfo{tbl->Handle(), 0,
                                                                  tbl->SizeBytes()};
                                    break;
                                }
                                case Shader::BufferType::FaultBuffer: {
                                    const auto* flt = rasterizer.GetBufferCache().GetFaultBuffer();
                                    if (!flt) {
                                        MtMaybeLogBailBufSample(5u, 2u, false, 0ULL, 0ULL,
                                                                stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    bi = vk::DescriptorBufferInfo{flt->Handle(), 0,
                                                                  flt->SizeBytes()};
                                    break;
                                }
                                default:
                                    MtMaybeLogBailBufSample(5u, 3u, false, 0ULL, 0ULL,
                                                            stage->pgm_hash);
                                    return abort_pass(MtBailReason::Buf);
                                }
                            } else {
                                const Gcn::Buffer vsharp = desc.GetSharpDirect(flat_ud, pgm_base);
                                const VAddr b_addr = vsharp.base_address;
                                const u64 b_size = vsharp.GetSize();
                                if (b_addr == 0 || b_size == 0) {
                                    bi = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                                } else {
                                    if (is_vs && b_size == 864ULL && vsharp.GetStride() == 16u &&
                                        cur_ts.is_gbuffer && !has_camera_cb) {
                                        const u8* cam_ptr = nullptr;
                                        u32 cam_sz = 0;
                                        if (resolve_cb_for_occ(&cb_ctx, vsharp, cam_ptr, cam_sz) &&
                                            cam_ptr && cam_sz >= 864u) {
                                            const float* cf =
                                                reinterpret_cast<const float*>(cam_ptr);
                                            if (cf[0] == 3000.0f && cf[4] >= 64.0f &&
                                                cf[5] >= 64.0f &&
                                                std::abs(cf[1] * cf[0] - 1.0f) <= 1e-3f &&
                                                std::isfinite(cf[183]) && std::isfinite(cf[187]) &&
                                                std::isfinite(cf[191])) {
                                                std::memcpy(camera_cb_floats.data(), cam_ptr, 864u);
                                                has_camera_cb = true;
                                            }
                                        }
                                    }
                                    vk::Buffer r_buf{};
                                    u64 r_off = 0;
                                    u64 r_tot = 0;
                                    u8 sb_step = 0u;
                                    if (!resolve_gpu_stream_range(b_addr, b_size, 1ULL, r_buf,
                                                                  r_off, r_tot, desc.is_formatted,
                                                                  &sb_step)) {
                                        MtMaybeLogBailBufSample(6u, sb_step, desc.is_formatted,
                                                                b_addr, b_size, stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    const u64 aligned_off = Common::AlignDown(r_off, storage_align);
                                    const auto adjust = static_cast<u32>(r_off - aligned_off);
                                    if ((adjust & 3u) != 0u) {
                                        MtMaybeLogBailBufSample(6u, 10u, desc.is_formatted, b_addr,
                                                                b_size, stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    const u64 avail_range =
                                        (r_tot > aligned_off) ? (r_tot - aligned_off) : 0ULL;
                                    const u64 bind_range =
                                        std::min<u64>(b_size + adjust, avail_range);
                                    if (bind_range == 0ULL) {
                                        MtMaybeLogBailBufSample(6u, 11u, desc.is_formatted, b_addr,
                                                                b_size, stage->pgm_hash);
                                        return abort_pass(MtBailReason::Buf);
                                    }
                                    push_data.AddOffset(binding.buffer, adjust);
                                    bi = vk::DescriptorBufferInfo{r_buf, aligned_off, bind_range};
                                }
                            }
                            desc_writes[num_desc_writes++] = vk::WriteDescriptorSet{
                                .dstBinding = binding.unified++,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = vk::DescriptorType::eStorageBuffer,
                                .pBufferInfo = &bi,
                            };
                            ++binding.buffer;
                        }

                        // Sampled Images (native static textures + dynamic scene/shadow RT SRVs)
                        for (const auto& img : stage->images) {
                            if (num_img_infos >= img_infos.size() ||
                                num_desc_writes >= desc_writes.size()) {
                                return abort_pass(MtBailReason::Other);
                            }
                            const Gcn::Image tsharp = img.GetSharpDirect(flat_ud);
                            vk::DescriptorImageInfo& ii = img_infos[num_img_infos++];
                            if (tsharp.Address() == 0 ||
                                tsharp.GetDataFmt() == Gcn::DataFormat::FormatInvalid ||
                                (mem && !mem->IsValidGpuMapping(tsharp.Address(), 0))) {
                                ii = vk::DescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                             vk::ImageLayout::eGeneral};
                            } else {
                                const auto srv = ResolveSceneSrv(worker_id, tsharp, img, cur_ts);
                                if (!srv.view) {
                                    MtMaybeLogBailTargetSample(
                                        3u, tsharp.Address(), u32(tsharp.GetDataFmt()),
                                        vs_stage->pgm_hash, ps_stage ? ps_stage->pgm_hash : 0ULL,
                                        srv.bail_substep, srv.diag_flags, srv.diag_w, srv.diag_h,
                                        srv.diag_guest_size);
                                    return abort_pass(MtBailReason::Target);
                                }
                                const bool written_in_list =
                                    scratch.OverlapsTargetWrite(tsharp.Address());
                                if (written_in_list && !srv.id) {
                                    MtMaybeLogBailTargetSample(
                                        4u, tsharp.Address(), u32(tsharp.GetDataFmt()),
                                        vs_stage->pgm_hash, ps_stage ? ps_stage->pgm_hash : 0ULL);
                                    return abort_pass(MtBailReason::Target);
                                }
                                if (srv.needs_stage_b_transit || (written_in_list && bool(srv.id))) {
                                    bool dup = false;
                                    for (u8 ri = 0; ri < cur_num_sampled_rts; ++ri) {
                                        const auto& ex = cur_sampled_rts[ri];
                                        if (ex.id == srv.id && ex.uid == srv.uid &&
                                            ex.range == srv.range) {
                                            dup = true;
                                            break;
                                        }
                                    }
                                    if (!dup) {
                                        if (cur_num_sampled_rts >=
                                            MtCompletedPassMeta::kMaxSampledRts) {
                                            MtMaybeLogBailTargetSample(
                                                5u, tsharp.Address(), cur_num_sampled_rts,
                                                vs_stage->pgm_hash,
                                                ps_stage ? ps_stage->pgm_hash : 0ULL);
                                            return abort_pass(MtBailReason::Target);
                                        }
                                        cur_sampled_rts[cur_num_sampled_rts++] =
                                            MtCompletedPassMeta::SampledRtRef{
                                                .id = srv.id,
                                                .uid = srv.uid,
                                                .range = srv.range,
                                            };
                                    }
                                }
                                ii = vk::DescriptorImageInfo{VK_NULL_HANDLE, srv.view, srv.layout};
                            }
                            desc_writes[num_desc_writes++] = vk::WriteDescriptorSet{
                                .dstBinding = binding.unified++,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = vk::DescriptorType::eSampledImage,
                                .pImageInfo = &ii,
                            };
                        }

                        // Samplers
                        for (const auto& smp : stage->samplers) {
                            if (num_img_infos >= img_infos.size() ||
                                num_desc_writes >= desc_writes.size()) {
                                return abort_pass(MtBailReason::Other);
                            }
                            const Gcn::Sampler ssharp = smp.GetSharpDirect(flat_ud);
                            const vk::Sampler vk_smp = ResolveSampler(
                                worker_id, ssharp, smp.is_depth, cur_ts.sampler_lod_bias,
                                rregs.ta_bc_base);
                            if (!vk_smp) {
                                return abort_pass(MtBailReason::Target);
                            }
                            vk::DescriptorImageInfo& ii = img_infos[num_img_infos++];
                            ii = vk::DescriptorImageInfo{vk_smp, VK_NULL_HANDLE,
                                                         vk::ImageLayout::eUndefined};
                            desc_writes[num_desc_writes++] = vk::WriteDescriptorSet{
                                .dstBinding = binding.unified++,
                                .dstArrayElement = 0,
                                .descriptorCount = 1,
                                .descriptorType = vk::DescriptorType::eSampler,
                                .pImageInfo = &ii,
                            };
                        }
                    }

                    PushGraphicsResources(
                        worker_id, pso->GetLayout(), pso->UsesPushDescriptors(), push_data,
                        std::span<const vk::WriteDescriptorSet>{desc_writes.data(),
                                                                num_desc_writes});

                    // Bind Vertex Buffers
                    const u32 num_vbs = static_cast<u32>(guest_vbs.size());
                    VertexInputs<vk::Buffer> host_vbs;
                    VertexInputs<vk::DeviceSize> host_offs;
                    VertexInputs<vk::DeviceSize> host_sizes;
                    VertexInputs<vk::DeviceSize> host_strides;
                    for (const auto& gb : guest_vbs) {
                        u64 h_size = 0;
                        if (!dynamic_vertex_input) {
                            h_size = VideoCore::DirectBuffers::ClampNativeGpuReadSize(
                                gb.base_address, gb.GetSize());
                        }
                        if (gb.base_address != 0 && gb.GetSize() > 0) {
                            const auto it = std::ranges::find_if(vb_ranges, [&](const VbRange& r) {
                                return gb.base_address >= r.begin && gb.base_address < r.end;
                            });
                            if (it == vb_ranges.end()) {
                                return abort_pass(MtBailReason::Buf);
                            }
                            const u64 h_off = it->offset + (gb.base_address - it->begin);
                            host_vbs.push_back(it->handle);
                            host_offs.push_back(h_off);
                            if (!dynamic_vertex_input) {
                                h_size = std::min<u64>(
                                    h_size,
                                    it->total_bytes > h_off ? (it->total_bytes - h_off) : 0ULL);
                            }
                        } else {
                            host_vbs.push_back(VK_NULL_HANDLE);
                            host_offs.push_back(0);
                        }
                        if (!dynamic_vertex_input) {
                            host_sizes.push_back(h_size);
                            host_strides.push_back(gb.GetStride());
                        }
                    }

                    if (dynamic_vertex_input) {
                        const bool same_layout =
                            has_prev_vb_state &&
                            vb_bindings.size() == prev_vb_bindings.size() &&
                            vb_attrs.size() == prev_vb_attrs.size() &&
                            std::memcmp(vb_bindings.data(), prev_vb_bindings.data(),
                                        vb_bindings.size() * sizeof(vb_bindings[0])) == 0 &&
                            std::memcmp(vb_attrs.data(), prev_vb_attrs.data(),
                                        vb_attrs.size() * sizeof(vb_attrs[0])) == 0;
                        if (!same_layout) {
                            w.active_cmdbuf.setVertexInputEXT(
                                std::span<const vk::VertexInputBindingDescription2EXT>{
                                    vb_bindings.data(), vb_bindings.size()},
                                std::span<const vk::VertexInputAttributeDescription2EXT>{
                                    vb_attrs.data(), vb_attrs.size()});
                            prev_vb_bindings = vb_bindings;
                            prev_vb_attrs = vb_attrs;
                        }
                        const bool same_bufs =
                            has_prev_vb_state && host_vbs.size() == prev_vb_bufs.size() &&
                            std::memcmp(host_vbs.data(), prev_vb_bufs.data(),
                                        host_vbs.size() * sizeof(vk::Buffer)) == 0 &&
                            std::memcmp(host_offs.data(), prev_vb_offs.data(),
                                        host_offs.size() * sizeof(vk::DeviceSize)) == 0;
                        if (num_vbs != 0u && !same_bufs) {
                            w.active_cmdbuf.bindVertexBuffers(0, num_vbs, host_vbs.data(),
                                                              host_offs.data());
                            prev_vb_bufs = host_vbs;
                            prev_vb_offs = host_offs;
                        } else if (num_vbs == 0u) {
                            prev_vb_bufs.clear();
                            prev_vb_offs.clear();
                        }
                        has_prev_vb_state = true;
                    } else if (num_vbs != 0u) {
                        w.active_cmdbuf.bindVertexBuffers2(0, num_vbs, host_vbs.data(),
                                                           host_offs.data(), host_sizes.data(),
                                                           host_strides.data());
                    }

                    // Bind Index Buffer
                    if (dp.is_indexed) {
                        if (ib_buf != prev_ib_buf || ib_off != prev_ib_off ||
                            index16 != prev_ib_16) {
                            w.active_cmdbuf.bindIndexBuffer(
                                ib_buf, ib_off,
                                index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
                            prev_ib_buf = ib_buf;
                            prev_ib_off = ib_off;
                            prev_ib_16 = index16;
                        }
                    }

                    // Dynamic State Commit (with upscaler target_scale & jitter_eligible check)
                    const bool jitter_eligible =
                        cur_ts.is_scene_depth && vs_stage->pgm_hash != 0x000000008747a367ULL &&
                        (dp.is_indirect || rregs.num_indices > 6u || num_instances > 1u);
                    const std::array<float, 2> cur_draw_jitter =
                        jitter_eligible ? cur_ts.frame_jitter : std::array<float, 2>{0.0f, 0.0f};

                    if (has_prev_dyn && !w.dyn_state.AnyDirty() &&
                        prev_dyn_epoch == replay_dyn_epoch && prev_dyn_indexed == dp.is_indexed &&
                        prev_dyn_jitter == cur_draw_jitter) {
                        UpdateWorkerColorBlendingState(w.dyn_state, rregs, pso);
                    } else {
                        UpdateWorkerDynamicStateFull(w.dyn_state, instance, rregs, pso,
                                                     dp.is_indexed, cur_ts.target_scale,
                                                     cur_draw_jitter);
                        has_prev_dyn = true;
                        prev_dyn_epoch = replay_dyn_epoch;
                        prev_dyn_indexed = dp.is_indexed;
                        prev_dyn_jitter = cur_draw_jitter;
                    }
                    if (w.dyn_state.AnyDirty()) {
                        w.dyn_state.CommitWith(
                            instance.IsDepthBoundsSupported(),
                            instance.IsDynamicColorWriteMaskSupported(),
                            instance.IsAttachmentFeedbackLoopLayoutSupported(),
                            [cmd = w.active_cmdbuf](auto&& fn) { fn(cmd); });
                    }

                    if (dp.is_indirect) {
                        if (dp.is_indexed) {
                            w.active_cmdbuf.drawIndexedIndirect(ind_buf, ind_off, 1u, 20u);
                        } else {
                            w.active_cmdbuf.drawIndirect(ind_buf, ind_off, 1u, 16u);
                        }
                    } else if (occ_decision.query_slot >= 0 && occ_decision.pool) {
                        const auto q_slot = static_cast<u32>(occ_decision.query_slot);
                        w.active_cmdbuf.beginQuery(occ_decision.pool, q_slot, {});
                        if (dp.is_indexed) {
                            w.active_cmdbuf.drawIndexed(rregs.num_indices, num_instances, 0u,
                                                        static_cast<s32>(vertex_offset),
                                                        instance_offset);
                        } else {
                            w.active_cmdbuf.draw(rregs.num_indices, num_instances, vertex_offset,
                                                 instance_offset);
                        }
                        w.active_cmdbuf.endQuery(occ_decision.pool, q_slot);
                    } else {
                        if (dp.is_indexed) {
                            w.active_cmdbuf.drawIndexed(rregs.num_indices, num_instances, 0u,
                                                        static_cast<s32>(vertex_offset),
                                                        instance_offset);
                        } else {
                            w.active_cmdbuf.draw(rregs.num_indices, num_instances, vertex_offset,
                                                 instance_offset);
                        }
                    }
                    ++w.recorded_draws;
                    if (IsSfxFireDumpActive()) {
                        const auto& pk = pso->GetGraphicsKey();
                        RecordSfxFireDumpDraw(
                            pso, static_cast<u32>(rregs.screen_scissor.bottom_right_x),
                            static_cast<u32>(rregs.screen_scissor.bottom_right_y),
                            pk.num_color_attachments,
                            pk.z_format != Gcn::DepthBuffer::ZFormat::Invalid,
                            static_cast<u32>(rregs.num_indices), num_instances, true, false);
                    }
                }
            }

            cur += dp.pkt_words;
            rem -= dp.pkt_words;
        }
    }

    seal_active_subpass();
    if (scratch.sealed_subpasses.size() != planned_subpasses) {
        return abort_pass(MtBailReason::Other);
    }

    // -------------------------------------------------------------------------
    // Phase 3: Atomic Commit & Rewrite DCB Slots In-Place (`m_beginptr` intact).
    // -------------------------------------------------------------------------
    for (const auto& gw : scratch.chunk_gpu_writes) {
        InvalidateGuestCopyShadow(gw.dst_va, gw.byte_size);
    }
    if (occ_block_id != 0xFFFFu) {
        rasterizer.SealMtOcclusionQueryBlock(occ_block_id);
        w.active_occ_block_id = 0xFFFFu;
    }
    w.local_stats.FlushToGlobal(rasterizer);

    scratch.staged_pkts.clear();
    scratch.emit_words.clear();
    auto stage_pkt = [&](const u32* words_ptr, u32 cnt) {
        const auto off = static_cast<u32>(scratch.emit_words.size());
        scratch.emit_words.insert(scratch.emit_words.end(), words_ptr, words_ptr + cnt);
        scratch.staged_pkts.push_back({off, cnt});
    };

    u64 total_recorded_draws = 0ULL;
    for (const auto& item : scratch.stream_items) {
        if (item.kind == WorkerCompileScratch::StreamItem::Kind::RawPkt) {
            stage_pkt(item.ptr, item.words);
            continue;
        }
        const u32 sp_i = item.subpass_idx;
        const bool is_last_sp = (sp_i + 1u == planned_subpasses);
        const auto& sp = scratch.sealed_subpasses[sp_i];
        const u64 cmd_seq = ++w.cmd_seq_gen;
        if (sp.pool_idx < w.pools.size()) {
            auto& fp = w.pools[sp.pool_idx];
            fp.last_cmd_seq = cmd_seq;
            fp.end_ring_pos = sp.end_ring_pos;
        }
        const u16 sp_occ_id = is_last_sp ? occ_block_id : 0xFFFFu;

        MtCompletedPassMeta& pub_meta = w.completed_meta[cmd_seq & (kCompletedMetaSlots - 1u)];
        pub_meta.sh_gfx_regs = sp.sh_gfx_regs;
        pub_meta.context_regs = sp.context_regs;
        pub_meta.uconfig_so_regs = sp.uconfig_so_regs;
        pub_meta.uconfig_vgt_regs = sp.uconfig_vgt_regs;
        pub_meta.cb_extent = sp.cb_extent;
        pub_meta.db_extent = sp.db_extent;
        pub_meta.cb_uid = sp.cb_uid;
        pub_meta.db_uid = sp.db_uid;
        pub_meta.first_pso = sp.first_pso;
        pub_meta.sec_cmd = sp.sec_cmd;
        pub_meta.recorded_draws = sp.recorded_draws;
        pub_meta.guest_frame = cur_guest_frame;
        pub_meta.occ_block_id = sp_occ_id;
        pub_meta.is_gbuffer = sp.is_gbuffer;
        pub_meta.has_camera_cb = sp.is_gbuffer && has_camera_cb;
        pub_meta.inherits_occlusion_query = sp.inherits_occlusion_query;
        pub_meta.num_sampled_rt_refs = sp.num_sampled_rt_refs;
        pub_meta.num_uav_indirect_refs = sp.num_uav_indirect_refs;
        pub_meta.motion_slot = sp.motion_slot;
        if (sp.num_sampled_rt_refs != 0u) {
            pub_meta.sampled_rt_refs = sp.sampled_rt_refs;
        }
        if (sp.num_uav_indirect_refs != 0u) {
            pub_meta.uav_indirect_refs = sp.uav_indirect_refs;
        }
        if (pub_meta.has_camera_cb) {
            pub_meta.camera_cb_floats = camera_cb_floats;
        }
        if (is_last_sp) {
            pub_meta.has_end_sync = true;
            pub_meta.end_reset = replay_dirty.reset;
            pub_meta.end_db_written = replay_db_written;
            pub_meta.end_cb_written_mask = replay_cb_written_mask;
            pub_meta.end_stage_b_mask[0] = replay_dirty.stage_b_mask[0];
            pub_meta.end_stage_b_mask[1] = replay_dirty.stage_b_mask[1];
            pub_meta.end_dirty_blocks = replay_dirty.blocks;
            pub_meta.end_cb_extent = replay_cb_extent;
            pub_meta.end_db_extent = replay_db_extent;
            pub_meta.end_cb_uid = replay_cb_uid;
            pub_meta.end_db_uid = replay_db_uid;
            pub_meta.end_span_headers.clear();
            pub_meta.end_span_values.clear();

            u32 span_start = ~0u;
            u32 span_len = 0u;
            const auto flush_span = [&]() {
                if (span_len == 0u) {
                    return;
                }
                pub_meta.end_span_headers.push_back((span_len << 16u) | span_start);
                const u32* const src_w = &rregs.reg_array[span_start];
                pub_meta.end_span_values.insert(pub_meta.end_span_values.end(), src_w,
                                                src_w + span_len);
                span_start = ~0u;
                span_len = 0u;
            };
            replay_dirty.ForEachBlock([&](size_t blk) {
                const u32 base = static_cast<u32>(blk * Gcn::RegDirty::BlockWords);
                const u32 end =
                    std::min<u32>(base + Gcn::RegDirty::BlockWords, Gcn::Regs::NumRegs);
                for (u32 idx = base; idx < end; ++idx) {
                    if (scratch.replay_exact_words.test(idx)) {
                        if (span_len != 0u && idx == span_start + span_len && span_len < 0xFFFFu) {
                            ++span_len;
                        } else {
                            flush_span();
                            span_start = idx;
                            span_len = 1u;
                        }
                    } else {
                        flush_span();
                    }
                }
            });
            flush_span();
        } else {
            pub_meta.has_end_sync = false;
        }
        pub_meta.published_seq.store(cmd_seq, std::memory_order_release);

        std::array<u32, 7> exec_pkt{};
        exec_pkt[0] = (3u << 30u) | ((6u - 1u) << 16u) | (u32(BbCp::Op::Nop) << 8u);
        exec_pkt[1] = BbCp::NopPayload::DirectVulkan;
        exec_pkt[2] = u32(BbCp::DirectVk::Op::ExecuteSecondaryCmds);
        const BbCp::DirectVk::ExecuteSecondaryArgs exec_args{
            .worker_and_draws = (worker_id & 0xFFu) | (sp.recorded_draws << 8u),
            .occ_and_flags = static_cast<u32>(sp_occ_id) |
                             (is_last_sp ? BbCp::DirectVk::kExecSecondaryFlagSyncRegs : 0u),
            .cmd_seq = cmd_seq,
        };
        std::memcpy(&exec_pkt[3], &exec_args, sizeof(exec_args));
        stage_pkt(exec_pkt.data(), static_cast<u32>(exec_pkt.size()));
        total_recorded_draws += sp.recorded_draws;
    }

    const auto num_staged = static_cast<u32>(scratch.staged_pkts.size());
    u32 pkt_cursor = 0;
    for (u32 s = 0; s < num_slices; ++s) {
        u32 written = 0;
        while (pkt_cursor < num_staged &&
               written + scratch.staged_pkts[pkt_cursor].words <= slices[s].orig_words) {
            const auto& sp = scratch.staged_pkts[pkt_cursor++];
            std::memcpy(slices[s].begin_ptr + written, scratch.emit_words.data() + sp.offset,
                        sp.words * sizeof(u32));
            written += sp.words;
        }
        if (written == 0u) {
            // Preserve valid non-zero IB length while keeping m_beginptr (+0x00) untouched.
            slices[s].begin_ptr[0] = 0x80000000u; // PM4 Type-2 1-dword NOP
            written = 1u;
        }
        *reinterpret_cast<u32**>(slices[s].slot_raw + 0x10) = slices[s].begin_ptr + written;
    }

    if (MtStatsEnabled()) [[unlikely]] {
        g_mt_lists_compiled.fetch_add(1ULL, std::memory_order_relaxed);
        g_mt_draws_recorded.fetch_add(total_recorded_draws, std::memory_order_relaxed);
        MtReportTelemetryIfDue(cur_guest_frame);
    }
    return true;
}

} // namespace Vulkan

extern "C" {

int PS4_SYSV_ABI bb_direct_vk_mt_mode() noexcept {
    return Vulkan::VkMtDirectRecorder::ConfiguredMode();
}

int PS4_SYSV_ABI bb_direct_vk_mt_worker_begin_pass(uint32_t worker_id,
                                                   const void* pass_desc_raw) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled() || !pass_desc_raw) {
        return 0;
    }
    const auto* desc = static_cast<const Vulkan::MtPassInheritanceDesc*>(pass_desc_raw);
    return rec->BeginWorkerPass(worker_id, *desc) ? 1 : 0;
}

uint64_t PS4_SYSV_ABI bb_direct_vk_mt_worker_end_pass(uint32_t worker_id) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled()) {
        return 0ULL;
    }
    const vk::CommandBuffer cmd = rec->EndWorkerPass(worker_id);
    return reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(cmd));
}

void PS4_SYSV_ABI bb_direct_vk_mt_execute_secondaries(const uint64_t* cmd_handles,
                                                      uint32_t count) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled() || !cmd_handles || count == 0u) {
        return;
    }
    std::array<vk::CommandBuffer, 16> stack_cmds;
    const u32 n = std::min<u32>(count, static_cast<u32>(stack_cmds.size()));
    u32 valid_n = 0;
    for (u32 i = 0; i < n; ++i) {
        if (cmd_handles[i] != 0ULL) {
            stack_cmds[valid_n++] =
                vk::CommandBuffer{reinterpret_cast<VkCommandBuffer>(cmd_handles[i])};
        }
    }
    if (valid_n != 0u) {
        rec->ExecuteSecondaryBuffers(std::span<const vk::CommandBuffer>{stack_cmds.data(), valid_n});
    }
}

int PS4_SYSV_ABI bb_direct_vk_mt_try_compile_cmd_list(const void* ctx_base,
                                                      void* cmd_list_impl) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (!rec || !rec->IsEnabled() || !cmd_list_impl) {
        return 0;
    }
    return rec->TryCompileDeferredCommandList(ctx_base, cmd_list_impl) ? 1 : 0;
}

void PS4_SYSV_ABI bb_direct_vk_mt_note_copy_batch(const void* raw_regions,
                                                  uint32_t count) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (rec && rec->IsEnabled() && raw_regions && count != 0u) {
        rec->NoteGuestImmediateCopyBatch(raw_regions, count);
    }
}

void PS4_SYSV_ABI bb_direct_vk_mt_invalidate_copy_shadow(uint64_t dst_va,
                                                         uint64_t byte_size) noexcept {
    auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance();
    if (rec && rec->IsEnabled() && dst_va != 0ULL && byte_size != 0ULL) {
        rec->InvalidateGuestCopyShadow(static_cast<VAddr>(dst_va), byte_size);
    }
}

} // extern "C"
