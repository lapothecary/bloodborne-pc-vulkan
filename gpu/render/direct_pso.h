// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 2 (Stage A elimination): Direct NativePsoId table, exact sampled-image
// format-class SpecWordCheck masks, and L1-resident (VS, PS, non_shader_epoch) PSO cache.

#pragma once

#include <array>
#include <cstddef>

#include "common/types.h"
#include "gcn/formats.h"

namespace Vulkan::DirectPso {

/// Dense 12-bit runtime PSO handle (`0` = uncached/fallback, `1..4095` = assigned PSOs).
using NativePsoId = u16;
inline constexpr NativePsoId kInvalidPsoId = 0u;
inline constexpr u32 kMaxNativePsos = 4096u;

/// Exact per-`NumberFormat` (0..15) bitmask of `DataFormat` (0..63) values that change either
/// `Gcn::RemapNumberFormat(nfmt, dfmt)` or `Gcn::MapNumberConversion(nfmt, dfmt)` relative to
/// the common default for that `NumberFormat`.
///
/// Used by `FastSelect` `SpecWordCheck` (`kind = 4`) on non-storage (`!desc.is_written`) sampled
/// textures so switching between e.g. `BC1`, `BC2`, `BC3`, `BC5`, `BC7`, `RGBA8`, `R8`, `RG16`
/// with the same `NumberFormat` never triggers a false `SpecWordCheck` mismatch and `MatchesInPlace`
/// fallback, while remaining 100% bit-exact for every `(NumberFormat, DataFormat)` pair.
inline constexpr std::array<u64, 16> kExcludedDfmtMaskByNfmt = []() constexpr {
    std::array<u64, 16> tbl{};
    auto bit = [](Gcn::DataFormat df) constexpr -> u64 { return 1ULL << static_cast<u32>(df); };
    // Unorm (0): 32-bit-per-component formats remap to Uint + Uint32ToUnorm.
    tbl[static_cast<u32>(Gcn::NumberFormat::Unorm)] =
        bit(Gcn::DataFormat::Format32) | bit(Gcn::DataFormat::Format32_32) |
        bit(Gcn::DataFormat::Format32_32_32) | bit(Gcn::DataFormat::Format32_32_32_32);
    // Snorm (1), Uscaled (2), Sscaled (3), Uint (4), Sint (5), Ubnorm (10):
    // RemapNumberFormat and MapNumberConversion are independent of DataFormat -> mask = 0.
    // SnormNz (6): depends on 8-bit vs 16-bit vs other -> exclude all (fall back to exact mask).
    tbl[static_cast<u32>(Gcn::NumberFormat::SnormNz)] = ~0ULL;
    // Float (7): 8-bit float (stencil read) remaps to Uint.
    tbl[static_cast<u32>(Gcn::NumberFormat::Float)] = bit(Gcn::DataFormat::Format8);
    // Srgb (9): BC4/BC5/BC6 remap to Unorm (and BC6 adds SrgbToNorm).
    tbl[static_cast<u32>(Gcn::NumberFormat::Srgb)] =
        bit(Gcn::DataFormat::FormatBc4) | bit(Gcn::DataFormat::FormatBc5) |
        bit(Gcn::DataFormat::FormatBc6);
    // Reserved/unused number formats (8, 11..15): conservative exact check.
    tbl[8] = ~0ULL;
    tbl[11] = ~0ULL;
    tbl[12] = ~0ULL;
    tbl[13] = ~0ULL;
    tbl[14] = ~0ULL;
    tbl[15] = ~0ULL;
    return tbl;
}();

/// Returns true if a non-storage sampled texture with `(nfmt, dfmt)` belongs to the common
/// `DataFormat`-agnostic equivalence class for `nfmt` and can use the 2-instruction `kind = 4`
/// check instead of an exact `dfmt` comparison.
[[nodiscard]] inline constexpr bool CanUseFastSampledNfmtCheck(Gcn::NumberFormat nfmt,
                                                               Gcn::DataFormat dfmt) noexcept {
    const u32 nf = static_cast<u32>(nfmt) & 0xFu;
    const u32 df = static_cast<u32>(dfmt) & 0x3Fu;
    return ((kExcludedDfmtMaskByNfmt[nf] >> df) & 1ULL) == 0ULL;
}

/// Evaluates the 2-instruction `kind = 4` check against `T#` dword 1 (`flat_ptr[dw1_off]`).
/// `expected_nfmt_shifted` is `(u32(nfmt) << 26)`, `nfmt_idx` is `u32(nfmt) & 0xF`.
[[nodiscard]] [[gnu::always_inline]] inline constexpr bool
MatchesFastSampledNfmtDw1(u32 dw1, u32 expected_nfmt_shifted, u32 nfmt_idx) noexcept {
    if ((dw1 & 0x3C000000u) != expected_nfmt_shifted) {
        return false;
    }
    const u64 bad_mask = kExcludedDfmtMaskByNfmt[nfmt_idx & 0xFu];
    if (bad_mask == 0ULL) {
        return true;
    }
    const u32 df = (dw1 >> 20) & 0x3Fu;
    return ((bad_mask >> df) & 1ULL) == 0ULL;
}

/// Branchless equivalent of `!MatchesFastSampledNfmtDw1`: returns 0 iff `dw1` matches.
[[nodiscard]] [[gnu::always_inline]] inline constexpr u32
FastSampledNfmtMismatchBits(u32 dw1, u32 expected_nfmt_shifted, u32 nfmt_idx) noexcept {
    const u32 nfmt_diff = (dw1 & 0x3C000000u) ^ expected_nfmt_shifted;
    const u32 df = (dw1 >> 20) & 0x3Fu;
    const u32 dfmt_bad = static_cast<u32>((kExcludedDfmtMaskByNfmt[nfmt_idx & 0xFu] >> df) & 1ULL);
    return nfmt_diff | dfmt_bad;
}

/// Compact L1D-resident cache mapping `(vs_addr, ps_addr, state_key, state_ids_packed, env_key)`
/// directly to a previously matched `FastSelect` bucket (`set_base`, `first_match`,
/// `matching_ways_mask`, `NativePsoId`) without rebuilding `BuildFastSelectSig` or running
/// `XXH3_64bits` + `180 B memcmp` across repeated shader, state, and render-target switches.
struct ShaderPairSlot {
    u64 vs_addr = 0;
    u64 ps_addr = 0;
    u64 state_key = 0;
    u64 env_key = 0;
    u32 set_base = 0;
    u16 first_match_idx = 0;
    u8 matching_ways_mask = 0;
    u8 valid = 0;
    NativePsoId pso_id = kInvalidPsoId;
    u16 pad = 0;
    u32 vs_rsrc1 = 0;
    u32 ps_rsrc1 = 0;
    u32 state_ids_packed = 0;
};
static_assert(sizeof(ShaderPairSlot) == 56u, "Compact ShaderPairSlot");

struct ShaderPairCache {
    static constexpr u32 NumSets = 256u;
    static constexpr u32 NumWays = 2u;
    static constexpr u32 TotalSlots = NumSets * NumWays; // 512 slots * 56 B = 28 KiB (L1D/L2)

    std::array<ShaderPairSlot, TotalSlots> slots{};
    std::array<u8, NumSets> repl_way{};
    u64 active_state_key = 0;

    void Clear() noexcept {
        for (auto& s : slots) {
            s.valid = 0;
        }
        active_state_key = 0;
    }

    [[nodiscard]] static inline u32 HashSet(u64 vs_addr, u64 ps_addr, u64 state_key,
                                            u32 state_ids_packed) noexcept {
        const u64 h = (vs_addr >> 8) ^ ((ps_addr >> 8) * 0x9E3779B97F4A7C15ULL) ^
                      (state_key * 0xBF58476D1CE4E5B9ULL) ^
                      (u64(state_ids_packed) * 0x94D049BB133111EBULL);
        return static_cast<u32>((h ^ (h >> 16) ^ (h >> 32)) & (NumSets - 1u));
    }

    [[nodiscard]] inline const ShaderPairSlot* Lookup(u64 vs_addr, u64 ps_addr, u32 vs_rsrc1,
                                                      u32 ps_rsrc1, u64 state_key, u64 env_key,
                                                      u32 state_ids_packed = 0u) const noexcept {
        const u32 base = HashSet(vs_addr, ps_addr, state_key, state_ids_packed) * NumWays;
        for (u32 w = 0; w < NumWays; ++w) {
            const auto& s = slots[base + w];
            if (s.valid && s.state_key == state_key && s.state_ids_packed == state_ids_packed &&
                s.vs_addr == vs_addr && s.ps_addr == ps_addr && s.vs_rsrc1 == vs_rsrc1 &&
                s.ps_rsrc1 == ps_rsrc1 && s.env_key == env_key) {
                return &s;
            }
        }
        return nullptr;
    }

    inline void Insert(u64 vs_addr, u64 ps_addr, u32 vs_rsrc1, u32 ps_rsrc1, u64 state_key,
                       u64 env_key, u32 set_base, u16 first_match_idx, u8 matching_ways_mask,
                       NativePsoId pso_id, u32 state_ids_packed = 0u) noexcept {
        const u32 set = HashSet(vs_addr, ps_addr, state_key, state_ids_packed);
        const u32 base = set * NumWays;
        u32 target_w = repl_way[set] & (NumWays - 1u);
        for (u32 w = 0; w < NumWays; ++w) {
            const auto& s = slots[base + w];
            const bool stale_epoch_fallback =
                ((s.state_key >> 63) == 0u) && (s.state_key != state_key);
            if (!s.valid || stale_epoch_fallback ||
                (s.vs_addr == vs_addr && s.ps_addr == ps_addr && s.vs_rsrc1 == vs_rsrc1 &&
                 s.ps_rsrc1 == ps_rsrc1 && s.state_key == state_key &&
                 s.state_ids_packed == state_ids_packed)) {
                target_w = w;
                break;
            }
        }
        repl_way[set] = static_cast<u8>((target_w + 1u) & (NumWays - 1u));
        auto& dst = slots[base + target_w];
        dst.vs_addr = vs_addr;
        dst.ps_addr = ps_addr;
        dst.state_key = state_key;
        dst.env_key = env_key;
        dst.set_base = set_base;
        dst.first_match_idx = first_match_idx;
        dst.matching_ways_mask = matching_ways_mask;
        dst.valid = 1u;
        dst.pso_id = pso_id;
        dst.vs_rsrc1 = vs_rsrc1;
        dst.ps_rsrc1 = ps_rsrc1;
        dst.state_ids_packed = state_ids_packed;
        active_state_key = state_key;
    }
};

} // namespace Vulkan::DirectPso
