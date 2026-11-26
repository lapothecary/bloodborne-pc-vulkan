// SPDX-License-Identifier: GPL-2.0-or-later
// The texture cache: write-backs of GPU-written images and the collector that evicts images
// and samplers when memory runs short.
#include "textures/texture_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bbport_free_check.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "buffers/buffer_cache.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"

namespace VideoCore {

namespace {

constexpr u64 MiB = 1ull << 20;
constexpr u64 GiB = 1ull << 30;
/// Without a budget from the driver.
constexpr u64 DefaultPressureMemory = GiB + 512 * MiB;
constexpr u64 DefaultCriticalMemory = 3 * GiB;
/// The usage the marks are placed under (cards with more memory keep the rest for it).
constexpr s64 TargetThreshold = 8 * GiB;

u64 SteadySeconds() {
    return u64(std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

u64 EnvU64(const char* name, u64 fallback) {
    const char* env = std::getenv(name);
    return env ? std::strtoull(env, nullptr, 10) : fallback;
}

void ComputeLowVramMarks(u64 budget, u64& trigger_out, u64& pressure_out, u64& critical_out) {
    u64 press_pct = 80;
    u64 crit_pct = 90;
    if (const char* marks = std::getenv("BB_GC_SMALL_MARKS")) {
        unsigned p = 0, c = 0;
        if (std::sscanf(marks, "%u,%u", &p, &c) == 2 && p >= 30 && c > p && c <= 98) {
            press_pct = p;
            crit_pct = c;
        }
    }
    const u64 trig_pct = press_pct > 12 ? (press_pct - 12) : (press_pct * 3 / 4);
    trigger_out = budget / 100 * trig_pct;
    pressure_out = budget / 100 * press_pct;
    critical_out = budget / 100 * crit_pct;
}

bool IsOrphanableTransientTarget(const Image& image) noexcept {
    // Strictly transient depth/stencil targets only: NEVER drop GpuModified color render targets
    // without writeback, because Bloodborne keeps persistent tiled character/weapon/enemy
    // blood-splatter accumulation maps (>= 1 MiB, single-mip, cleared once on spawn) untouched
    // for minutes until a hit occurs.
    return !image.usage.vo_surface && !image.usage.storage && !image.binding.is_bound &&
           !image.binding.is_target && !image.info.props.is_block &&
           image.info.resources.levels == 1u && image.info.resources.layers == 1u &&
           image.backing && image.info.props.is_depth;
}

} // namespace

void TextureCache::UpdateMemoryBudget() {
    if (!instance.CanReportMemoryUsage()) {
        if (instance.IsLowVramDevice()) {
            ComputeLowVramMarks(instance.GetPhysicalVramBytes(), trigger_gc_memory,
                                pressure_gc_memory, critical_gc_memory);
            return;
        }
        trigger_gc_memory = 0;
        pressure_gc_memory = DefaultPressureMemory;
        critical_gc_memory = DefaultCriticalMemory;
        return;
    }
    if (instance.IsLowVramDevice()) {
        const u64 raw_budget = instance.GetDeviceMemoryBudgetNow();
        const u64 eff_budget =
            raw_budget != 0 ? std::min(raw_budget, instance.GetPhysicalVramBytes())
                            : instance.GetPhysicalVramBytes();
        ComputeLowVramMarks(eff_budget, trigger_gc_memory, pressure_gc_memory, critical_gc_memory);
        std::printf("Texture cache: low-VRAM collector marks trigger=%llu MiB, pressure=%llu MiB, "
                    "critical=%llu MiB (VRAM=%llu MiB)\n",
                    (unsigned long long)(trigger_gc_memory >> 20),
                    (unsigned long long)(pressure_gc_memory >> 20),
                    (unsigned long long)(critical_gc_memory >> 20),
                    (unsigned long long)(instance.GetPhysicalVramBytes() >> 20));
        return;
    }
    const s64 local = s64(instance.GetTotalMemoryBudget());
    const s64 threshold = std::min<s64>(local, TargetThreshold);
    // Pressure: 60 % of the threshold free, but on cards under 16 GB 30 % of the budget (at least
    // 2 GB): 60 % of an 8 GB card's budget had the collector evicting hundreds of images every
    // 5 s at 3 GB used (#204). Critical: 20 % of the threshold (at least 512 MB).
    const s64 pressure_free = std::min(threshold * 6 / 10, std::max(local * 3 / 10, s64(2 * GiB)));
    pressure_gc_memory = std::max<u64>(
        u64(std::max<s64>(0, std::min(local - pressure_free, local - s64(GiB)))),
        DefaultPressureMemory);
    critical_gc_memory = std::max<u64>(
        u64(std::max<s64>(0, std::min(local - threshold * 2 / 10, local - s64(512 * MiB)))),
        DefaultCriticalMemory);
    trigger_gc_memory = u64(std::max<s64>(0, (local - threshold) / 2));
}

void TextureCache::ProcessDownloadImages() {
    // Every fence comes here (~450 a frame): the lock only when something is queued.
    if (!downloads_queued.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk{download_mutex};
    downloads_queued.store(false, std::memory_order_relaxed);
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const ImageInfo& info = image.info;
    const u32 bytes =
        info.pitch * info.size.height * info.size.depth * info.resources.layers * (info.num_bits / 8);
    ASSERT(bytes <= info.guest_size);
    const auto copy_to = [&](vk::DeviceSize offset) {
        return vk::BufferImageCopy{
            .bufferOffset = offset,
            .bufferRowLength = info.pitch,
            .bufferImageHeight = info.size.height,
            .imageSubresource{
                .aspectMask = info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                  : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {info.size.width, info.size.height, info.size.depth},
        };
    };

    // The game's memory in place: the GPU copies the image there itself, in stream order (before
    // the fences that follow). Where the memory has a VRAM copy it copies into that too, so
    // buffers bound over it read the image's data in order (issue #122: motion blur read a render
    // target as a buffer from a copy refreshed only once the CPU write-back landed, frames later).
    if (GuestInPlace() && info.guest_address != 0) {
        const auto [arena, offset] = buffer_cache.ObtainBuffer(info.guest_address, bytes, true);
        auto copy = copy_to(offset);
        runtime.DownloadImage(&image, arena, std::span{&copy, 1});
        if (buffer_cache.IsInPlace(info.guest_address, bytes)) {
            return;
        }
    }
    const auto staging = runtime.GetStagingPool().Request(bytes, MemoryType::HostCached, 16, !sync);
    auto copy = copy_to(staging.offset);
    runtime.DownloadImage(&image, staging.buffer, std::span{&copy, 1});
    const auto write_back = [this, address = info.guest_address, bytes](const auto& download) {
        download.Invalidate();
        BbFreeCheck::Check(address, bytes, download.mapped, BbFreeCheck::ImageDownload);
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(address), download.mapped,
                                                  bytes);
        if (!WriteTracking()) {
            buffer_cache.InvalidateMemory(address, bytes);
        }
    };
    if (sync) {
        scheduler.Finish();
        write_back(staging);
    } else {
        scheduler.DeferPriorityOperation([this, staging, write_back] {
            write_back(staging);
            runtime.GetStagingPool().FreeDeferred(staging);
        });
    }
}

void TextureCache::NoteVramShort() {
    vram_short_until.store(SteadySeconds() + 30, std::memory_order_relaxed);
}

bool TextureCache::VramShort() {
    return SteadySeconds() < vram_short_until.load(std::memory_order_relaxed);
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        atomic_gc_tick.store(++gc_tick, std::memory_order_relaxed);
    };
    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
        // Integrated GPUs (Steam Deck): the usage covers system memory heaps holding much more
        // than images, over the startup budget: the marks follow the driver's current budget.
        // BB_GC_BUDGET_MB=N: so on any GPU with a fixed budget (tests).
        static const u64 forced_budget = EnvU64("BB_GC_BUDGET_MB", 0) << 20;
        if (instance.IsIntegrated() || forced_budget) {
            const u64 budget = forced_budget ? forced_budget : instance.GetDeviceMemoryBudgetNow();
            if (budget != 0) {
                trigger_gc_memory = budget / 10 * 7;
                pressure_gc_memory = budget / 100 * 85;
                critical_gc_memory = budget / 100 * 95;
            }
        }
    }
    BbStats::gc_used_bytes.store(total_used_memory, std::memory_order_relaxed);
    BbStats::gc_trigger_bytes.store(trigger_gc_memory, std::memory_order_relaxed);
    BbStats::gc_critical_bytes.store(critical_gc_memory, std::memory_order_relaxed);

    // Ages in seconds: gc_tick (a guest submission each, hundreds a second) at each of the last
    // 64 seconds. Once a second: near the driver's budget the collector evicts by submissions.
    const u64 second = SteadySeconds();
    bool new_second = false;
    if (second != gc_second) {
        gc_second = second;
        new_second = true;
        if (instance.CanReportMemoryUsage() && !instance.IsIntegrated()) {
            const u64 budget = instance.GetDeviceMemoryBudgetNow();
            static const u64 forced_budget_chk = EnvU64("BB_GC_BUDGET_MB", 0) << 20;
            if (instance.IsLowVramDevice() && !forced_budget_chk && budget != 0) {
                const u64 eff_budget = std::min(budget, instance.GetPhysicalVramBytes());
                ComputeLowVramMarks(eff_budget, trigger_gc_memory, pressure_gc_memory,
                                    critical_gc_memory);
            }
            if (budget != 0 && total_used_memory >= budget / 100 * 92 &&
                vram_short_until.load(std::memory_order_relaxed) < second + 2) {
                vram_short_until.store(second + 2, std::memory_order_relaxed);
            }
        }
        BbStats::coarse_second.store(u32(second), std::memory_order_relaxed);
        gc_tick_at_second[second % gc_tick_at_second.size()] = gc_tick;
    }

    // Below the pressure mark an image goes once unused for BB_GC_IDLE_SECONDS (20): the
    // textures of an area left behind. Under pressure after BB_GC_PRESSURE_IDLE_SECONDS (5), over
    // the critical mark after 1 s, and only then by submissions (80/160 ticks): textures off
    // screen for a moment were evicted and uploaded again when the camera turned back, and the
    // frame rate fell the longer one played.
    static const u64 idle_seconds = std::clamp<u64>(EnvU64("BB_GC_IDLE_SECONDS", 20), 1, 63);
    static const u64 pressure_idle_seconds =
        std::clamp<u64>(EnvU64("BB_GC_PRESSURE_IDLE_SECONDS", 5), 0, 63);
    const auto tick_seconds_ago = [&](u64 seconds) {
        return gc_tick_at_second[(second - seconds) % gc_tick_at_second.size()];
    };
    const u64 idle_tick = tick_seconds_ago(idle_seconds);

    if (!BbToggle::Disabled(BbToggle::TextureCollector) && new_second &&
        total_used_memory < trigger_gc_memory && idle_tick != 0 && gc_tick > idle_tick) {
        std::scoped_lock lock{mutex};
        const u64 cur_sched_tick = CurrentTick();
        size_t idle_deletions = 16;
        u32 idle_visited = 0;
        lru_cache.ForEachItemBelow(idle_tick, [&](ImageId image_id) {
            if (idle_deletions == 0 || ++idle_visited > 64) {
                return true;
            }
            Image& image = slot_images[image_id];
            if (True(image.flags & (ImageFlagBits::DirectImmutable | ImageFlagBits::GpuModified)) ||
                image.SafeToDownload() || image.MtLastGcTick() >= idle_tick ||
                cur_sched_tick - image.LastAccessTick() <= 64u) {
                TouchImage(image);
                return false;
            }
            --idle_deletions;
            const u64 freed_bytes = image.backing ? u64(image.GetHostImageSize()) : 0ull;
            total_used_memory =
                total_used_memory > freed_bytes ? (total_used_memory - freed_bytes) : 0ull;
            BbStats::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
            FreeImage(image_id);
            return false;
        });
    }
    if (BbToggle::Disabled(BbToggle::TextureCollector) || total_used_memory < trigger_gc_memory) {
        return;
    }

    std::scoped_lock lock{mutex};
    const u64 cur_sched_tick = CurrentTick();
    bool pressured = false;
    bool aggressive = false;
    bool is_emergency = false;
    u64 below_tick = 0;
    size_t deletions = 0;
    u32 visited = 0;
    // `emergency`: over the critical mark after what was unused for a second went.
    const auto configure = [&](bool allow_aggressive, bool emergency = false) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggressive = allow_aggressive && total_used_memory >= critical_gc_memory;
        is_emergency = emergency;
        below_tick = gc_tick - std::min<u64>(aggressive ? 160 : pressured ? 80 : 16, gc_tick);
        if (!pressured && !aggressive) {
            below_tick = std::min(below_tick, idle_tick);
        } else if (!emergency && second >= vram_short_until.load(std::memory_order_relaxed)) {
            below_tick =
                std::min(below_tick, tick_seconds_ago(aggressive ? 1 : pressure_idle_seconds));
        }
        deletions = aggressive ? 40 : pressured ? 20 : 10;
        visited = 0;
    };
    const u64 orphan_pressured_tick = tick_seconds_ago(15);
    const auto evict = [&](ImageId image_id) {
        if (deletions == 0 || ++visited > 256) {
            return true;
        }
        Image& image = slot_images[image_id];
        if (image.binding.is_bound || image.binding.is_target) {
            TouchImage(image);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (is_emergency) {
            // Even in emergency mode, protect textures bound in the current in-flight frame so
            // low-VRAM scenes never thrash active G-Buffer / character / world textures every tick.
            if (cur_sched_tick - image.LastAccessTick() <= 8u) {
                TouchImage(image);
                BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        } else if (image.MtLastGcTick() >= below_tick ||
                   cur_sched_tick - image.LastAccessTick() <= 32u) {
            TouchImage(image);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (True(image.flags & ImageFlagBits::DirectImmutable) &&
            True(image.flags & ImageFlagBits::GpuModified)) {
            TouchImage(image);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const bool write_back = image.SafeToDownload();
        const bool orphan_rt =
            instance.IsLowVramDevice() && pressured && IsOrphanableTransientTarget(image) &&
            orphan_pressured_tick != 0 && gc_tick > orphan_pressured_tick &&
            image.MtLastGcTick() < orphan_pressured_tick &&
            cur_sched_tick - image.LastAccessTick() > 256u;
        // Images that cannot go now (tiled GPU-written ones cannot be written back; GPU-written
        // ones only under pressure) neither use up the deletions nor stay first in line: the
        // oldest ten being such targets stopped the collector for good. Orphaned transient
        // single-mip 2D render/depth targets left behind by NowLoading/deaths can be dropped cleanly.
        if (!orphan_rt && ((write_back && image.info.IsTiled()) || (write_back && !pressured))) {
            TouchImage(image);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        --deletions;
        if (write_back && !orphan_rt) {
            // Synchronously, while the image still watches its pages: a deferred write-back
            // landed after they were given up, over whatever the game stored there meanwhile.
            DownloadImageMemory(image_id, true);
            ++gc_downloads;
        }
        const u64 freed_bytes = image.backing ? u64(image.GetHostImageSize()) : 0ull;
        total_used_memory =
            total_used_memory > freed_bytes ? (total_used_memory - freed_bytes) : 0ull;
        ++gc_evictions;
        BbStats::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggressive) {
                deletions >>= 2;
                aggressive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    configure(false);
    lru_cache.ForEachItemBelow(below_tick, evict);
    if (total_used_memory >= critical_gc_memory) {
        configure(true);
        lru_cache.ForEachItemBelow(below_tick, evict);
    }
    if (total_used_memory >= critical_gc_memory) {
        // Still over it: what was used a moment ago goes too (out of memory otherwise).
        configure(true, true);
        lru_cache.ForEachItemBelow(below_tick, evict);
    }
    // Evictions under pressure, at most every 5 s.
    if (pressured || gc_downloads != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - gc_report_time >= std::chrono::seconds(5)) {
            std::printf("Texture cache: memory pressure, %llu of %llu MiB (critical %llu): "
                        "%llu images evicted, %llu written back since the last report\n",
                        (unsigned long long)(total_used_memory >> 20),
                        (unsigned long long)(pressure_gc_memory >> 20),
                        (unsigned long long)(critical_gc_memory >> 20),
                        (unsigned long long)gc_evictions, (unsigned long long)gc_downloads);
            gc_report_time = now;
            gc_evictions = gc_downloads = 0;
        }
    }
}

void TextureCache::GarbageCollectSamplers() {
    std::vector<Sampler> retired;
    {
        std::scoped_lock lock{samplers_mutex};
        if (gc_tick == 0 || samplers.size() < trigger_gc_samplers) {
            return;
        }
        size_t deletions = 0;
        u64 below_tick = 0;
        const auto run_pass = [&](u64 min_age, size_t max_deletions) {
            // Never evict samplers used on the current `gc_tick` (strictly `below_tick < gc_tick`).
            if (gc_tick <= min_age) {
                return;
            }
            below_tick = gc_tick - min_age;
            deletions = max_deletions;
            const auto evict = [&](u64 key) {
                if (deletions == 0) {
                    return true;
                }
                auto it = samplers.find(key);
                if (it == samplers.end()) {
                    return false;
                }
                --deletions;
                sampler_lru.Free(it->second.lru_id);
                retired.push_back(std::move(it.value()));
                samplers.erase(it);
                sampler_evict_gen.fetch_add(1, std::memory_order_release);
                return false;
            };
            sampler_lru.ForEachItemBelow(below_tick, evict);
        };

        const u64 used = samplers.size();
        const bool pressured = used >= pressure_gc_samplers;
        run_pass(pressured ? 64u : 160u, pressured ? 20u : 10u);
        if (samplers.size() >= critical_gc_samplers) {
            run_pass(16u, 40u);
        }
        if (samplers.size() >= critical_gc_samplers) {
            // Emergency cap protection: evict any sampler older than the current tick.
            run_pass(1u, 64u);
        }
    }
    if (!retired.empty()) {
        // Defer vkDestroySampler until all in-flight GPU work referencing these samplers finishes.
        scheduler.DeferOperation([retired = std::move(retired)]() mutable { retired.clear(); });
    }
}

u64 TextureCache::PurgeIdleForAllocation(u64 desired_bytes) {
    u64 reclaimed = 0;
    u32 count = 0;
    {
        std::scoped_lock lock{mutex};
        const u64 cur_sched_tick = CurrentTick();
        const u64 second = SteadySeconds();
        const u64 orphan_thresh = gc_tick_at_second[(second - 5) % gc_tick_at_second.size()];
        const auto run_purge_pass = [&](u64 min_gc_age, u64 min_sched_age) {
            const u64 min_below = gc_tick > min_gc_age ? (gc_tick - min_gc_age) : 0ull;
            lru_cache.ForEachItemBelow(min_below, [&](ImageId image_id) {
                if (reclaimed >= desired_bytes) {
                    return true;
                }
                Image& image = slot_images[image_id];
                if (image.binding.is_bound || image.binding.is_target ||
                    cur_sched_tick - image.LastAccessTick() <= min_sched_age) {
                    return false;
                }
                const bool orphan_rt =
                    IsOrphanableTransientTarget(image) && orphan_thresh != 0 &&
                    gc_tick > orphan_thresh && image.MtLastGcTick() < orphan_thresh &&
                    cur_sched_tick - image.LastAccessTick() > 128u;
                if (!orphan_rt &&
                    (True(image.flags & ImageFlagBits::GpuModified) || image.SafeToDownload())) {
                    return false;
                }
                const u64 bytes = image.backing ? u64(image.GetHostImageSize()) : 0ull;
                reclaimed += bytes;
                ++count;
                BbStats::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
                FreeImage(image_id);
                return false;
            });
        };
        run_purge_pass(8u, 16u);
        if (reclaimed < desired_bytes) {
            // Second pass for 30 FPS low-VRAM sessions where active BC world textures can be
            // re-uploaded cleanly from guest RAM to give FSR 3 / upscaler contexts headroom.
            run_purge_pass(2u, 4u);
        }
    }
    if (count > 0) {
        // Execute deferred vmaDestroyImage callbacks immediately so VMA returns empty blocks to WDDM.
        scheduler.Finish();
        std::printf("Texture cache: purged %u idle images (%llu MiB) to make room for upscaler\n",
                    count, (unsigned long long)(reclaimed >> 20));
    }
    return reclaimed;
}

} // namespace VideoCore
