// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// Module 04: FromSoftware SprjClothWind Action, World Wind & Turbulence (Batch 4)
//
// 1-to-1 C++20 decompilation of the 4 FromSoftware cloth wind runtime functions
// (10,196 B PS4 machine code):
//   [36] 0x1051fa0 ( 844 B) ClothExp_SprjWindAction_apply
//   [37] 0x10522f0 (2037 B) ClothExp_SprjWindEvalWorldDir
//   [38] 0x1052af0 (4244 B) ClothExp_SprjPointWind_apply
//   [39] 0x1053b90 (3071 B) ClothExp_SprjDirWind_apply
// ============================================================================
#pragma once

#include <cstdint>
#include <immintrin.h>

#include "common/types.h"

namespace BbClothExp {

extern "C" {

bool PS4_SYSV_ABI ClothExp_SprjWindAction_apply(
    u8* action, u8* sim, __m128* forces, float substep_dt) noexcept;

__m128 PS4_SYSV_ABI ClothExp_SprjWindEvalWorldDir(
    const u8* action, u8* wind_entry, std::uint64_t opt_seed_tag) noexcept;

bool PS4_SYSV_ABI ClothExp_SprjPointWind_apply(
    const u8* action, u8* sim, __m128* forces, u8* wind_entry, float substep_dt) noexcept;

bool PS4_SYSV_ABI ClothExp_SprjDirWind_apply(
    const u8* action,
    u8* sim,
    __m128* forces,
    const __m128* sum_dir_ptr,
    const float* avg_param_ptr,
    float substep_dt) noexcept;

} // extern "C"

} // namespace BbClothExp
