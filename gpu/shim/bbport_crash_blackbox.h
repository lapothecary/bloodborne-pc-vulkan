// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// bb_vulkan: Universal Linux & Windows Engine Crash Black-Box Reporter
// File: gpu/shim/bbport_crash_blackbox.h
//
// Sealed C++20 diagnostics & Dantelion2 heap layout definitions (eboot 1.09):
//   1. Dual-sink crash report writer (`stderr` + `logs/crash-YYYYMMDD_HHMMSS.log`)
//   2. Hybrid Guest (`eboot.bin`) + Host (`dantelion_gx` / `libbbgpu` / `DbgHelp` / `dladdr`)
//      stack walker (`RBP` chain + bounded `RSP` return-address scan)
//   3. Sealed FromSoftware `DLHeapManager` / `DLRegularHeap` arena & chunk-header
//      corruption analyzer (`0x1fc58e0..0x1fc6010`, `0x207b720..0x207bc50`, `0x2085350..0x2085a4e`)
//   4. Lock-free 128-event Flight Recorder (`0 B` PT_TLS, `0.000 ms` render-loop cost)
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace BbCrashBlackBox {

// ============================================================================
// Sealed FromSoftware Dantelion2 Heap Binary Layouts (eboot.bin v1.09)
// Extracted 1:1 from:
//   - DLHeapManager::GetAllocatorForAddr @ 0x207b720 / 0x207bbf0
//   - DLHeapObject / DLDefaultHeapStrategy @ 0x1fc5ac0..0x1fc6010
//   - DLRegularHeap::getBlockSize @ 0x20856c0, DLRegularHeap::free @ 0x20858a0..0x2085a4e
// ============================================================================
namespace DantelionHeap {

inline constexpr std::uint64_t kDLHeapManagerGlobalVma   = 0x056a1c10ULL; // DLHeapManager* singleton
inline constexpr std::uint64_t kDLHeapManagerVecOffset   = 0x00000468ULL; // DLVector<DLHeapArenaEntry>*
inline constexpr std::uint64_t kDLTlsTempHeapSlotVma     = 0x053e4568ULL; // TLS offset for thread temp heap
inline constexpr std::uint64_t kGXMainAllocatorGlobalVma = 0x05540680ULL; // g_pGXMainAllocator ("GFX")

// DLRegularHeap methods in eboot.bin (for automatic crash-RIP classification)
inline constexpr std::uint64_t kDLRegularHeapAllocBegin  = 0x02085350ULL;
inline constexpr std::uint64_t kDLRegularHeapAllocEnd    = 0x020856bfULL;
inline constexpr std::uint64_t kDLRegularHeapSizeBegin   = 0x020856c0ULL;
inline constexpr std::uint64_t kDLRegularHeapSizeEnd     = 0x0208589fULL;
inline constexpr std::uint64_t kDLRegularHeapFreeBegin   = 0x020858a0ULL;
inline constexpr std::uint64_t kDLRegularHeapFreeEnd     = 0x02085a4fULL;
inline constexpr std::uint64_t kDLDefaultStratFreeBegin  = 0x01fc5fc0ULL;
inline constexpr std::uint64_t kDLDefaultStratFreeEnd    = 0x01fc6010ULL;

// Entry in *(DLHeapManager + 0x468) vector (`begin` at +0x08, `end` at +0x10, stride 0x18)
struct DLHeapArenaEntry {
    std::uint64_t allocator;   // +0x00: DLAllocator* (e.g. DLDefaultHeapStrategy* = DLHeapObject + 0x20)
    std::uint64_t arena_start; // +0x08: inclusive lower bound
    std::uint64_t arena_end;   // +0x10: exclusive upper bound
};
static_assert(sizeof(DLHeapArenaEntry) == 0x18, "DLHeapArenaEntry must be 0x18 bytes");

struct DLHeapArenaVector {
    std::uint64_t allocator_or_vtable; // +0x00
    std::uint64_t begin_ptr;           // +0x08: DLHeapArenaEntry*
    std::uint64_t end_ptr;             // +0x10: DLHeapArenaEntry*
    std::uint64_t cap_ptr;             // +0x18: DLHeapArenaEntry*
};
static_assert(sizeof(DLHeapArenaVector) == 0x20, "DLHeapArenaVector layout");

// Embedded at `heap_impl + 0x28` (`DLHeapObject + 0x30`, where `heap_impl = *(allocator + 0x08)`):
struct DLRegularHeapBin {
    std::uint64_t next;
    std::uint64_t prev;
};
static_assert(sizeof(DLRegularHeapBin) == 0x10, "DLRegularHeapBin must be 16 bytes");

struct DLRegularHeapState {
    std::uint64_t    free_bin_bitmap;    // +0x00 (impl + 0x28)
    std::uint64_t    arena_first_chunk;  // +0x08 (impl + 0x30)
    std::uint64_t    arena_end_sentinel; // +0x10 (impl + 0x38)
    std::uint64_t    used_bytes;         // +0x18 (impl + 0x40)
    std::uint64_t    alloc_count;        // +0x20 (impl + 0x48)
    DLRegularHeapBin bins[64];           // +0x28..+0x428 (impl + 0x50..+0x450)
};
static_assert(sizeof(DLRegularHeapState) == 0x428, "DLRegularHeapState must be 0x428 bytes");

// Offsets relative to `heap_impl = *(allocator + 0x08)` (`DLHeapObject + 0x08`):
inline constexpr std::size_t kImplHeapIdOffset      = 0x008; // uint32_t heap_id (0x1fc5d04)
inline constexpr std::size_t kImplRegularHeapOffset = 0x028; // DLRegularHeapState (0x1fc5fc0: rbx+0x28)
inline constexpr std::size_t kImplHeapNamePtrOffset = 0x460; // const char* / const char16_t* (0x1fc5d24)

// Physical chunk header at `user_ptr - 0x10` (user_ptr is 16-byte aligned):
struct DLRegularChunkHeader {
    std::uint64_t prev_phys_hdr;  // +0x00 ([user - 0x10]): valid when (next_and_flags & 2) == 0
    std::uint64_t next_and_flags; // +0x08 ([user - 0x08]): (next_hdr & ~3ULL) | prev_in_use(2) | curr_in_use(1)
};
static_assert(sizeof(DLRegularChunkHeader) == 0x10, "DLRegularChunkHeader must be 16 bytes");

inline constexpr std::uint64_t kChunkFlagCurrInUse = 0x1ULL;
inline constexpr std::uint64_t kChunkFlagPrevInUse = 0x2ULL;
inline constexpr std::uint64_t kChunkFlagsMask     = 0x3ULL;

} // namespace DantelionHeap

void Initialize() noexcept;
void SetGuestImage(const unsigned char* image, std::uint64_t size) noexcept;
void RecordEvent(const char* category, const char* detail) noexcept;
void RecordEventFmt(const char* category, const char* fmt, ...) noexcept
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
void UpdatePlayerPosition(float x, float y, float z) noexcept;
void NoteFrameTick() noexcept;

void WriteCrashReport(int sig, const void* fault_addr, void* ucontext_ptr,
                      const void* guest_image, std::uint64_t guest_size) noexcept;

} // namespace BbCrashBlackBox

extern "C" {
void bbgpu_flight_record(const char* category, const char* detail);
int bbgpu_resolve_code_address(std::uintptr_t addr, char* out_buf, std::size_t out_size);
void bbgpu_write_crash_blackbox(int sig, const void* fault_addr, void* ucontext_ptr,
                                const void* guest_image, std::uint64_t guest_size);
}
