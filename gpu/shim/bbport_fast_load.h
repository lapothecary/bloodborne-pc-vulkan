// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// bb_vulkan: Fast Location & Respawn Loading Subsystem (BB_FAST_LOAD)
// File: gpu/shim/bbport_fast_load.h
//
// Seals the reverse-engineered PS4 System V LP64 zlib stream layout (0x70 B),
// compiler-inlined inflateEnd bridge contract (+0x40 window slot), and exact
// boot.bin VMAs for both statically linked zlib copies:
//   - Copy #1 (zlib 1.2.7, Scaleform .gfx / PNG / DLZlibReader): 0x03a77c0..0x03a9a95
//   - Copy #2 (zlib 1.2.3, all .dcx DCP\0DFLT & DCP\0EDGE archives): 0x2067850..0x2069815
//   - DcxDfltDecompressor streaming read chunk size (16 KiB -> 64 KiB): 0x262603d
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <zlib.h>

namespace BbFastLoad {

/// Exact PS4 System V LP64 z_stream layout (112 bytes = 0x70).
/// Note: Windows x64 is LLP64 where host z_stream::total_in / total_out / adler
/// are 32-bit unsigned longs (sizeof(z_stream) == 0x58), whereas PS4 guest code
/// always addresses 64-bit fields at +0x10, +0x28, +0x60 (sizeof == 0x70).
struct GuestZStream {
    const std::uint8_t* next_in;   // +0x00
    std::uint32_t avail_in;        // +0x08
    std::uint32_t pad0_;           // +0x0c
    std::uint64_t total_in;        // +0x10
    std::uint8_t* next_out;        // +0x18
    std::uint32_t avail_out;       // +0x20
    std::uint32_t pad1_;           // +0x24
    std::uint64_t total_out;       // +0x28
    const char* msg;               // +0x30
    void* state;                   // +0x38
    void* zalloc;                  // +0x40
    void* zfree;                   // +0x48
    void* opaque;                  // +0x50
    std::int32_t data_type;        // +0x58
    std::uint32_t pad2_;           // +0x5c
    std::uint64_t adler;           // +0x60
    std::uint64_t reserved;        // +0x68
};

static_assert(offsetof(GuestZStream, next_in) == 0x00);
static_assert(offsetof(GuestZStream, avail_in) == 0x08);
static_assert(offsetof(GuestZStream, total_in) == 0x10);
static_assert(offsetof(GuestZStream, next_out) == 0x18);
static_assert(offsetof(GuestZStream, avail_out) == 0x20);
static_assert(offsetof(GuestZStream, total_out) == 0x28);
static_assert(offsetof(GuestZStream, msg) == 0x30);
static_assert(offsetof(GuestZStream, state) == 0x38);
static_assert(offsetof(GuestZStream, zalloc) == 0x40);
static_assert(offsetof(GuestZStream, zfree) == 0x48);
static_assert(offsetof(GuestZStream, opaque) == 0x50);
static_assert(offsetof(GuestZStream, data_type) == 0x58);
static_assert(offsetof(GuestZStream, adler) == 0x60);
static_assert(offsetof(GuestZStream, reserved) == 0x68);
static_assert(sizeof(GuestZStream) == 0x70);

/// Host bridge state stored at GuestZStream::state (+0x38).
/// Eleven Dcx destructor/seek sites in boot.bin (0x2065b62..0x206760f) inline
/// inflateEnd directly: they read `*(uint64_t*)(state + 0x40)` (`state->window`),
/// free it via `strm->zfree(opaque, window)` if non-zero, and then call
/// `strm->zfree(opaque, state)`. Keeping `fake_window_must_be_zero == 0` at
/// exact offset +0x40 and pointing `strm->zfree` to `HostZlibBridgeFree` makes
/// both explicit and compiler-inlined inflateEnd 100% leak-free and guest-heap-safe.
struct alignas(16) HostZlibBridgeState {
    std::uint64_t magic;                     // +0x00
    void* saved_guest_zfree;                 // +0x08
    std::int32_t window_bits;                // +0x10
    std::int32_t pad0_;                      // +0x14
    std::uint64_t reserved_[5];              // +0x18..+0x3f
    std::uint64_t fake_window_must_be_zero;  // +0x40 (MUST stay 0 for inlined inflateEnd!)
    z_stream host_strm;                      // +0x48
};

static_assert(offsetof(HostZlibBridgeState, magic) == 0x00);
static_assert(offsetof(HostZlibBridgeState, saved_guest_zfree) == 0x08);
static_assert(offsetof(HostZlibBridgeState, window_bits) == 0x10);
static_assert(offsetof(HostZlibBridgeState, fake_window_must_be_zero) == 0x40);
static_assert(offsetof(HostZlibBridgeState, host_strm) == 0x48);

/// Sealed boot.bin VMA offsets (ImageBase = 0x800000000).
namespace Vma {
// Embedded zlib Copy #1: zlib 1.2.7 (Scaleform .gfx, PNG, DLZlibReader)
inline constexpr std::uint64_t Z127_InflateReset  = 0x03a77c0ull;
inline constexpr std::uint64_t Z127_InflateReset2 = 0x03a7880ull;
inline constexpr std::uint64_t Z127_InflateInit2_ = 0x03a79d0ull;
inline constexpr std::uint64_t Z127_InflateInit_  = 0x03a7ab0ull;
inline constexpr std::uint64_t Z127_Inflate       = 0x03a7bf0ull;
inline constexpr std::uint64_t Z127_InflateEnd    = 0x03a9a40ull;

// Embedded zlib Copy #2: zlib 1.2.3 (all .dcx DCP\0DFLT & DCP\0EDGE archives)
inline constexpr std::uint64_t Z123_InflateReset  = 0x2067850ull;
inline constexpr std::uint64_t Z123_InflateInit2_ = 0x2067900ull;
inline constexpr std::uint64_t Z123_InflateInit_  = 0x2067aa0ull;
inline constexpr std::uint64_t Z123_Inflate       = 0x2067ac0ull;
inline constexpr std::uint64_t Z123_InflateEnd    = 0x20697c0ull;

// DcxDfltDecompressor::ctor (0x2626010): `mov eax, 0x4000` -> `0x10000` (64 KiB read chunk)
inline constexpr std::uint64_t DcxDflt_ChunkImm   = 0x262603dull;
} // namespace Vma

/// Whether BB_FAST_LOAD is enabled (default 1; BB_FAST_LOAD=0 disables all fast-load paths).
bool Enabled() noexcept;

/// Installs host SIMD zlib detours and DcxDflt 64 KiB read-chunk upgrade into the guest image.
void PatchImage(unsigned char* image, std::uint64_t size);

/// Called once per guest frame on MOMainThread (inside PacerWait) with current monotonic time and
/// last frame's draw count. Returns true when an active non-3D loading screen burst is underway.
bool UpdateFrameAndCheckBurst(std::uint64_t now_ns, std::uint64_t frame_draws) noexcept;

/// Lock-free query for Presenter / Display threads: true while a non-3D loading burst is active.
bool IsLoadingBurstActive() noexcept;

/// Total bytes decompressed through host SIMD zlib since startup.
std::uint64_t DecompressedBytes() noexcept;

} // namespace BbFastLoad
