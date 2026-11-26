// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_runtime.h
//
// Zero-TLS runtime bridge for 1-to-1 decompiled PS4 Havok Cloth (`hcl*`) and
// FromSoftware Wind (`FD4Wind` / `SprjWindAction`) functions.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <x86intrin.h>

#include "common/types.h"

namespace BbClothExp {

/// Guest `eboot.bin` runtime base address (set once in `PatchImage`).
extern std::uintptr_t g_cloth_image_base;
#if defined(_WIN32)
extern std::uint32_t g_cloth_win32_tls_teb_offset;
#endif

/// Resolves an unslid PS4 `eboot.bin` VMA to a live host pointer.
[[gnu::always_inline]] inline std::uintptr_t ResolveVma(std::uintptr_t vma) noexcept {
    return g_cloth_image_base + vma;
}

template <typename FnPtr>
[[gnu::always_inline]] inline FnPtr ResolveFn(std::uintptr_t vma) noexcept {
    return reinterpret_cast<FnPtr>(g_cloth_image_base + vma);
}

/// Reads the guest Orbis TLS base (`fs:0x0` on PS4 -> `gs:0x0` on Linux `bbport`,
/// `gs:[teb_off]` on Windows `bbport`). Uses zero host `thread_local` storage (`+0 B` `PT_TLS`).
[[gnu::always_inline]] inline std::uintptr_t GuestTlsBase() noexcept {
#if defined(_WIN32)
    std::uintptr_t tcb = 0;
    const std::uintptr_t off = static_cast<std::uintptr_t>(g_cloth_win32_tls_teb_offset);
    __asm__ volatile("movq %%gs:(%1), %0" : "=r"(tcb) : "r"(off));
    return tcb;
#elif defined(__x86_64__)
    std::uintptr_t tcb = 0;
    __asm__ volatile("movq %%gs:0x0, %0" : "=r"(tcb));
    return tcb;
#else
    return 0;
#endif
}

/// Resolves a PS4 GOT `TPOFF64` slot (e.g. `0x53e40f8` = `hkMemoryRouter*`,
/// `0x53e4100` = `hkMonitorStream*`) to the current thread's guest TLS variable address.
template <typename T>
[[gnu::always_inline]] inline T* GuestTlsSlot(std::uintptr_t got_vma) noexcept {
    const std::int64_t tpoff = *reinterpret_cast<const std::int64_t*>(ResolveVma(got_vma));
    return reinterpret_cast<T*>(static_cast<std::int64_t>(GuestTlsBase()) + tpoff);
}

/// Exact match for Havok's inline `hkMonitorStream` timer record push (`0x53e4100`).
struct HkMonitorStreamLayout {
    u8* start;     // +0x00
    u8* current;   // +0x08
    u8* capacity;  // +0x10
    u8* memoryEnd; // +0x18
};

[[gnu::always_inline]] inline void MonitorPushTimer(std::uintptr_t tag_vma) noexcept {
    HkMonitorStreamLayout* mon = *GuestTlsSlot<HkMonitorStreamLayout*>(0x53e4100ULL);
    if (!mon) return;
    u8* cur = mon->current;
    if (cur < mon->memoryEnd) {
        *reinterpret_cast<std::uintptr_t*>(cur) = ResolveVma(tag_vma);
        *reinterpret_cast<std::uint32_t*>(cur + 0x08) = static_cast<std::uint32_t>(__rdtsc());
        mon->current = cur + 0x10;
    }
}

[[gnu::always_inline]] inline void MonitorPushObjectTimer(std::uintptr_t tag_vma,
                                                          std::uint64_t obj_payload) noexcept {
    HkMonitorStreamLayout* mon = *GuestTlsSlot<HkMonitorStreamLayout*>(0x53e4100ULL);
    if (!mon) return;
    u8* cur = mon->current;
    if (cur < mon->memoryEnd) {
        *reinterpret_cast<std::uintptr_t*>(cur) = ResolveVma(tag_vma);
        *reinterpret_cast<std::uint64_t*>(cur + 0x10) = obj_payload;
        *reinterpret_cast<std::uint32_t*>(cur + 0x08) = static_cast<std::uint32_t>(__rdtsc());
        mon->current = cur + 0x18;
    }
}

/// Returns `hkMemoryRouter::getInstance().temp()` (`[tls(0x53e40f8)] + 0x58` `hkAllocator*`).
[[gnu::always_inline]] inline u8* GetHkTempAllocator() noexcept {
    u8* router = *GuestTlsSlot<u8*>(0x53e40f8ULL);
    return router ? *reinterpret_cast<u8**>(router + 0x58) : nullptr;
}

/// Inline `hkLifoAllocator` bump allocation on `hkMemoryRouter::getInstance().stack()` (`+0x00`),
/// matching `0x76e851..0x76e89d`, `0x773e96..0x773ed5`, and `0x76ddbc..0x76ddfa` bit-for-bit.
[[gnu::always_inline]] inline u8* HkLifoAlloc(std::int32_t rounded_bytes) noexcept {
    u8* lifo = *GuestTlsSlot<u8*>(0x53e40f8ULL);
    const std::int32_t slab_sz = *reinterpret_cast<const std::int32_t*>(lifo + 0x10);
    if (rounded_bytes <= slab_sz) {
        u8* cur = *reinterpret_cast<u8* const*>(lifo + 0x18);
        u8* next = cur + static_cast<std::ptrdiff_t>(rounded_bytes);
        if (next <= *reinterpret_cast<u8* const*>(lifo + 0x20)) {
            *reinterpret_cast<u8**>(lifo + 0x18) = next;
            return cur;
        }
    }
    using SlowAllocFn = u8*(PS4_SYSV_ABI*)(u8*, int);
    return ResolveFn<SlowAllocFn>(0x0437420ULL)(lifo, rounded_bytes);
}

/// Inline `hkLifoAllocator` LIFO pop on `hkMemoryRouter::getInstance().stack()` (`+0x00`),
/// matching `0x76ef86..0x76f012`, `0x7741f6..0x774255`, and `0x76e0f0..0x76e139` bit-for-bit.
[[gnu::always_inline]] inline void HkLifoFree(void* ptr, std::int32_t rounded_bytes) noexcept {
    u8* lifo = *GuestTlsSlot<u8*>(0x53e40f8ULL);
    u8* byte_ptr = static_cast<u8*>(ptr);
    const std::int32_t slab_sz = *reinterpret_cast<const std::int32_t*>(lifo + 0x10);
    if (slab_sz >= rounded_bytes &&
        byte_ptr + static_cast<std::ptrdiff_t>(rounded_bytes) ==
            *reinterpret_cast<u8* const*>(lifo + 0x18) &&
        *reinterpret_cast<u8* const*>(lifo + 0x28) != byte_ptr) {
        *reinterpret_cast<u8**>(lifo + 0x18) = byte_ptr;
    } else {
        using SlowFreeFn = void(PS4_SYSV_ABI*)(u8*, void*, int);
        ResolveFn<SlowFreeFn>(0x04377f0ULL)(lifo, ptr, rounded_bytes);
    }
}

} // namespace BbClothExp
