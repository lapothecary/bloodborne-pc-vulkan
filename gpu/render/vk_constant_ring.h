// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: small read-only guest buffers (constants) copied by the GPU command thread (stage A of
// the draw pipeline, vk_draw_pipe.h) into this ring, so the draw recording thread only binds
// them. Positions are monotonic byte counts; a region is reused once the submission that
// recorded its last draw has completed on the GPU. Stage B stamps each packet it records with
// the submission tick (Stamp), stage A retires stamps whose tick the GPU has passed.

#pragma once

#include <array>
#include <atomic>

#include "common/assert.h"
#include "common/types.h"
#include "buffers/buffer.h"
#include "render/scheduler.h"

namespace Vulkan {

class ConstantRing {
public:
    /// 32 MiB; BB_CONSTANT_RING_MB (diagnostics) another size.
    static inline const u64 Capacity = [] {
        const char* env = std::getenv("BB_CONSTANT_RING_MB");
        const u64 mb = env ? std::strtoull(env, nullptr, 10) : 0;
        return (mb ? mb : 32ull) << 20;
    }();

    ConstantRing(const Instance& instance, Scheduler& scheduler_)
        : scheduler{scheduler_},
          buffer{instance, 0, Capacity, VideoCore::MemoryType::Stream, "bbport constant ring"} {
        ASSERT(!buffer.mapped_data.empty());
    }

    [[nodiscard]] vk::Buffer Handle() const noexcept {
        return buffer.Handle();
    }

    /// Stage A: `size` bytes at `alignment`; returns the ring offset, or nullopt when the ring is
    /// still in use by the GPU (the caller leaves the binding to stage B).
    std::optional<u64> Allocate(u64 size, u64 alignment) {
        // Proactively drain completed stamps before the 1024-entry SPSC queue can fill during
        // long bursts of tiny menu/loading submissions prior to the first 32 MiB wrap.
        if (stamps_head.load(std::memory_order_relaxed) -
                stamps_tail.load(std::memory_order_relaxed) >=
            (stamps.size() / 2)) [[unlikely]] {
            Retire();
        }
        u64 at = (position + alignment - 1) & ~(alignment - 1);
        if (at % Capacity + size > Capacity) {
            at += Capacity - at % Capacity; // wrap to the ring start
        }
        const auto in_use = [&] { return at + size > Capacity && at + size - Capacity > retired; };
        if (in_use()) {
            Retire();
            if (in_use()) {
                return std::nullopt;
            }
        }
        position = at + size;
        return at % Capacity;
    }

    /// Stage A: checks whether `size` bytes at `alignment` fit without advancing `position`.
    [[nodiscard]] bool HasFreeSpace(u64 size, u64 alignment = 256) noexcept {
        u64 at = (position + alignment - 1) & ~(alignment - 1);
        if (at % Capacity + size > Capacity) {
            at += Capacity - at % Capacity;
        }
        const auto in_use = [&] { return at + size > Capacity && at + size - Capacity > retired; };
        if (in_use()) {
            Retire();
            if (in_use()) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] u8* Data(u64 offset) noexcept {
        return buffer.mapped_data.data() + offset;
    }

    /// Stage A, after writing: makes the bytes visible to the GPU.
    void Flush(u64 offset, u64 size) {
        if (!buffer.is_coherent) {
            buffer.Flush(offset, size);
        }
    }

    /// Stage A: the position the next packet's allocations end at.
    [[nodiscard]] u64 Position() const noexcept {
        return position;
    }

    /// Stage B, after recording a packet whose allocations end at `end`. Stamps of one
    /// submission tick are merged: only the last end per tick is published. Always flushes any
    /// older tick's pending end even when `end == pending_end` so a full ring can never starve.
    void Stamp(u64 end) {
        const u64 tick = scheduler.CurrentTick();
        if (tick != pending_tick && pending_end != last_published) {
            Publish(pending_end, pending_tick);
        }
        if (end > pending_end) {
            pending_end = end;
            pending_tick = tick;
        }
    }

    /// Stage B: publish any pending end once `scheduler.CurrentTick()` has advanced past
    /// `pending_tick`, even when subsequent packets did not allocate from this ring.
    void FlushIfTickAdvanced() noexcept {
        if (pending_end == last_published) [[likely]] {
            return;
        }
        const u64 tick = scheduler.CurrentTick();
        if (tick != pending_tick) {
            Publish(pending_end, pending_tick);
        }
    }

private:
    void UpdateLatest(u64 end, u64 tick) noexcept {
        const u64 seq = latest_seq.load(std::memory_order_relaxed);
        latest_seq.store(seq + 1, std::memory_order_release);
        latest_end.store(end, std::memory_order_relaxed);
        latest_tick.store(tick, std::memory_order_relaxed);
        latest_seq.store(seq + 2, std::memory_order_release);
    }

    bool ReadLatest(u64& out_end, u64& out_tick) const noexcept {
        for (int attempt = 0; attempt < 4; ++attempt) {
            const u64 s1 = latest_seq.load(std::memory_order_acquire);
            if ((s1 & 1u) != 0u || s1 == 0u) {
                return false;
            }
            const u64 e = latest_end.load(std::memory_order_relaxed);
            const u64 t = latest_tick.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            const u64 s2 = latest_seq.load(std::memory_order_relaxed);
            if (s1 == s2) {
                out_end = e;
                out_tick = t;
                return true;
            }
        }
        return false;
    }

    void Publish(u64 end, u64 tick) noexcept {
        last_published = end;
        UpdateLatest(end, tick);
        const u64 at = stamps_head.load(std::memory_order_relaxed);
        if (at - stamps_tail.load(std::memory_order_acquire) >= stamps.size()) {
            return; // full: latest_{end,tick} guarantees full retirement once tick finishes
        }
        stamps[at % stamps.size()] = {end, tick};
        stamps_head.store(at + 1, std::memory_order_release);
    }

    void Retire() {
        auto* semaphore = scheduler.GetWorkSemaphore();
        semaphore->Refresh();
        u64 at = stamps_tail.load(std::memory_order_relaxed);
        const u64 head = stamps_head.load(std::memory_order_acquire);
        while (at < head && semaphore->IsFree(stamps[at % stamps.size()].tick)) {
            const u64 stamp_end = stamps[at % stamps.size()].end;
            if (stamp_end > retired) {
                retired = stamp_end;
            }
            ++at;
        }
        stamps_tail.store(at, std::memory_order_release);
        if (at == head) {
            u64 l_end = 0;
            u64 l_tick = 0;
            if (ReadLatest(l_end, l_tick) && l_end > retired && semaphore->IsFree(l_tick)) {
                retired = l_end;
            }
        }
    }

    struct StampEntry {
        u64 end;
        u64 tick;
    };
    Scheduler& scheduler;
    VideoCore::Buffer buffer;
    u64 position = 0;     ///< stage A
    u64 retired = 0;      ///< stage A: the GPU is done with everything before this
    u64 pending_end = 0, pending_tick = 0, last_published = 0; ///< stage B
    std::array<StampEntry, 1024> stamps{};
    alignas(64) std::atomic<u64> stamps_head{0}; ///< stage B writes
    alignas(64) std::atomic<u64> stamps_tail{0}; ///< stage A reads
    alignas(64) std::atomic<u64> latest_seq{0};  ///< stage B writes (seqlock for overflow recovery)
    std::atomic<u64> latest_end{0};
    std::atomic<u64> latest_tick{0};
};

} // namespace Vulkan
